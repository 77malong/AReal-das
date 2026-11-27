"""Run native GDN operators and modules on the requested device."""

from functools import wraps
import hashlib
import importlib
import importlib.metadata
from itertools import accumulate, combinations
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import gdn_precision_reference as ref


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_fixture(directory, tp):
    manifest = json.loads((directory / "fixture.json").read_text())
    loaded = []
    for name in ("fixture.pt", f"reference-tp{tp}.pt"):
        path = directory / name
        assert file_hash(path) == manifest["fixture_files"][name], path
        loaded.append(torch.load(path, map_location="cpu", weights_only=True))
    return *loaded, manifest


def setup_device(expected, tp):
    actual = "hcu" if torch.version.hip else "nvidia" if torch.version.cuda else "cpu"
    assert actual == expected and torch.cuda.is_available(), (expected, actual)
    assert int(os.environ["WORLD_SIZE"]) == tp
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if expected == "hcu":
        from hcu_areal.areal_adaptor import apply_training_patches

        apply_training_patches("megatron")
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    dist.init_process_group("nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(2026)
    return mpu


def float_stages():
    return {
        "l2norm": "fla.modules.l2norm",
        "recurrence": "fla.ops.gated_delta_rule.chunk",
        "gdn": "megatron.core.ssm.gated_delta_net.gdn",
        "common": "megatron.core.ssm.gated_delta_net.common",
        "te": "transformer_engine",
        "triton": "triton",
    }


def runtime(expected, mpu, manifest):
    modules = {}
    for name, path in float_stages().items():
        mod = importlib.import_module(path)
        modules[name] = {
            "import": path,
            "file": mod.__file__,
            "sha256": file_hash(mod.__file__),
        }
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "triton",
            "transformer-engine",
            "fla-core",
            "flash-linear-attention",
        )
    }
    return {
        "backend": expected,
        "rank": dist.get_rank(),
        "tp_rank": mpu.get_tensor_model_parallel_rank(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "modules": modules,
        "packages": packages,
        "fixture_files": manifest["fixture_files"],
    }


def from_cpu(value, layout=None):
    if layout is None:
        return value.cuda()
    stride = layout["stride"]
    offset = layout["storage_offset"]
    size = (
        offset + 1 + sum((d - 1) * s for d, s in zip(value.shape, stride, strict=True))
    )
    result = torch.empty(size, dtype=value.dtype, device="cuda").as_strided(
        value.shape, stride, offset
    )
    result.copy_(value)
    return result


def build_operator(case):
    from megatron.core.ssm.gated_delta_net import common

    inputs = case["inputs"]
    if "layouts" in case:
        args = tuple(
            from_cpu(x, layout)
            for x, layout in zip(inputs, case["layouts"], strict=True)
        )
    else:
        args = tuple(x.cuda() for x in inputs)
    kind = case["op"]

    def forward():
        if kind == "l2norm":
            return {"l2norm": common.l2norm(*args)}
        if kind == "gate":
            outputs = common._GDNBase._compute_g_and_beta(None, *args)
            return dict(zip(("g", "beta"), outputs, strict=True))
        q, k, v, g, beta = args
        output, _ = common.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=False,
            output_final_state=False,
        )
        return {"recurrence": output}

    return forward


def build_module(weights, tp, mpu):
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_gated_delta_net_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.ssm.gated_delta_net.common import chunk_gated_delta_rule
    from megatron.core.transformer import TransformerConfig

    config = TransformerConfig(
        hidden_size=ref.HIDDEN,
        num_layers=1,
        num_attention_heads=ref.HEADS,
        linear_num_key_heads=ref.HEADS,
        linear_num_value_heads=ref.HEADS,
        linear_key_head_dim=ref.DIM,
        linear_value_head_dim=ref.DIM,
        linear_conv_kernel_dim=ref.CONV,
        activation_func=F.silu,
        normalization="RMSNorm",
        layernorm_epsilon=ref.NORM_EPS,
        layernorm_zero_centered_gamma=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=False,
        transformer_impl="transformer_engine",
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        deterministic_mode=False,
    )
    spec = get_gated_delta_net_module_spec(config)
    module = (
        GatedDeltaNet(
            config,
            submodules=spec.submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=ProcessGroupCollection(
                tp=mpu.get_tensor_model_parallel_group(),
                cp=mpu.get_context_parallel_group(),
            ),
        )
        .cuda()
        .eval()
    )
    assert module.gated_delta_rule is chunk_gated_delta_rule
    expected = ref.weight_shards(weights, tp)[mpu.get_tensor_model_parallel_rank()]
    actual = dict(module.named_parameters())
    assert actual.keys() == expected.keys(), (list(actual), list(expected))
    with torch.no_grad():
        for name, param in actual.items():
            assert (
                param.shape == expected[name].shape
                and param.dtype == expected[name].dtype
            ), name
            param.copy_(expected[name])
    return module


class ObservedModule:
    """Record the operands around real module calls and copy only after forward."""

    def __init__(self, module, case, tp_rank, tp):
        from megatron.core.packed_seq_params import PackedSeqParams

        self.module = module
        self.hidden = case["hidden"].chunk(tp, dim=0)[tp_rank].contiguous().cuda()
        self.params = None
        if case["packed"]:
            offsets = torch.tensor(
                [0, *accumulate(case["lengths"])], dtype=torch.int32, device="cuda"
            )
            self.params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=offsets,
                cu_seqlens_kv=offsets,
                max_seqlen_q=max(case["lengths"]),
                max_seqlen_kv=max(case["lengths"]),
            )
        self.values = {}
        self.module.in_proj.register_forward_hook(self.projection)
        for name, input_name, outputs in (
            ("_prepare_input_for_gated_delta_rule", "short_conv", ("q", "k", "v")),
            ("_compute_g_and_beta", None, ("g", "beta")),
            ("gated_delta_rule", None, ("recurrence",)),
        ):
            setattr(
                module, name, self.observe(getattr(module, name), input_name, outputs)
            )

    def projection(self, _module, _args, result):
        self.values["input_projection"] = result[0]

    def observe(self, fn, input_name, outputs):
        @wraps(fn)
        def record(*args, **kwargs):
            if input_name:
                self.values[input_name] = args[0]
            result = fn(*args, **kwargs)
            self.values.update(zip(outputs, result[: len(outputs)], strict=True))
            return result

        return record

    def __call__(self):
        # A single forward updates all observed entries.
        self.values.clear()
        output, bias = self.module(self.hidden, None, packed_seq_params=self.params)
        assert bias is None
        self.values["output"] = output
        return self.values


def raw_bytes(x):
    flat = x.contiguous().reshape(-1)
    return flat.view(torch.uint8).reshape(flat.numel(), x.element_size())


def bitwise_differences(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    return (raw_bytes(actual) != raw_bytes(expected)).any(dim=1).count_nonzero().item()


def bf16_ulp(actual, expected):
    def ordered(tensor):
        bits = tensor.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(actual) - ordered(expected)).abs()


def error_metrics(actual, expected, tolerance, ulp_limit=None):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = torch.isfinite(actual).all() and torch.isfinite(expected).all()
    atol, rtol = tolerance
    if not finite:
        return {"finite": False, "outside": actual.numel(), "atol": atol, "rtol": rtol}
    expected32 = expected.float()
    diff = (actual.float() - expected32).abs().reshape(-1)
    index = diff.argmax().item()
    report = {
        "finite": True,
        "count": diff.numel(),
        "mean_abs": diff.mean().item(),
        "p99": torch.kthvalue(diff, max(1, int(diff.numel() * 0.99))).values.item(),
        "max_abs": diff[index].item(),
        "max_coordinate": torch.unravel_index(torch.tensor(index), actual.shape),
        "atol": atol,
        "rtol": rtol,
        "outside": (diff > (atol + rtol * expected32.abs()).reshape(-1))
        .count_nonzero()
        .item(),
    }
    report["max_coordinate"] = [x.item() for x in report["max_coordinate"]]
    if actual.dtype == torch.bfloat16:
        ulps = bf16_ulp(actual, expected)
        report["max_ulp"] = ulps.max().item()
        if ulp_limit is not None:
            report["ulp_limit"] = ulp_limit
            report["outside"] = (ulps > ulp_limit).count_nonzero().item()
    return report


def measure(call, expected, tolerances, ulp_limits, directory, repeats):
    with torch.no_grad():
        outputs = [
            {
                name: value.detach().cpu().contiguous().clone()
                for name, value in call().items()
            }
            for _ in range(repeats)
        ]
    for output in outputs:
        assert output.keys() == expected.keys(), (list(output), list(expected))
    precision = [
        {
            name: error_metrics(
                output[name], value, tolerances[name], ulp_limits.get(name)
            )
            for name, value in expected.items()
        }
        for output in outputs
    ]
    differences = {
        f"{a + 1}:{b + 1}": {
            name: bitwise_differences(outputs[a][name], outputs[b][name])
            for name in expected
        }
        for a, b in combinations(range(repeats), 2)
    }
    flagged = [
        next((name for name, value in record.items() if value["outside"]), None)
        for record in precision
    ]
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, directory / "outputs.pt")
    report = {
        "precision": precision,
        "first_precision_failure": flagged,
        "within_start": differences,
    }
    write_json(directory / "metrics.json", report)
    return report
