"""DeepSeek-V3/R1 BF16/FP8 dispatch routed through the existing AWEX hooks."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import sys

from ..feature import AbstractFeature
from .deepseek_awex_layout import (
    is_deepseek, deepseek_convert_gathered, assert_deepseek_transfer_metadata,
)

TRAIN_MODULE = "areal.v2.weight_update.awex.megatron_adapter"
INFER_MODULE = "areal.v2.weight_update.awex.sglang_adapter"


def validate_inference_model(adapter):
    model = adapter._get_model()
    if getattr(model, "quant_config", None) is not None:
        from .deepseek_fp8 import runtime_plan
        runtime_plan(adapter)
    architectures=getattr(model.config,'architectures',()) or ()
    if any('NextN' in name for name in architectures) or 'NextN' in type(model).__name__:
        raise NotImplementedError("DeepSeek AWEX patch does not yet support active MTP/NextN inference")
    return model


def _has_active_mtp(engine):
    """Check the built model, not the stale pre-provider HF-derived config."""
    mcore = getattr(engine, 'mcore_config', None)
    enabled = getattr(mcore, 'enable_mtp', None)
    if enabled is True:
        return True
    for model in (getattr(engine, 'model', None) or ()):
        seen = set()
        while model is not None and id(model) not in seen:
            seen.add(id(model))
            if getattr(model, 'mtp_process', False) or getattr(model, 'mtp', None) is not None:
                return True
            model = getattr(model, 'module', None)
    # When the engine does not expose the effective setting, fail closed.
    return enabled is not False and bool(getattr(engine.tf_config, 'mtp_num_layers', 0))


def deepseek_train_iter(adapter, theta_by_id=None, consume_overrides=False):
    import torch
    from areal.engine.megatron_utils.megatron import all_gather_param, get_named_parameters

    engine = adapter._engine
    if getattr(engine.config, "use_lora", False):
        raise NotImplementedError("DeepSeek AWEX patch requires full-model training")
    from .deepseek_fp8 import (target_config, validate_quant_config, source_scale_map,
                               compute_source, export_weight)
    quantization = target_config(engine)
    source_quantization = validate_quant_config(getattr(engine, 'quantization_config', None))
    source_scales = source_scale_map(engine.model)
    scale_ids = {id(scale) for scale in source_scales.values()}
    if _has_active_mtp(engine):
        raise NotImplementedError("DeepSeek AWEX patch requires active MTP training to be disabled")
    overrides = theta_by_id if theta_by_id is not None else {}
    experts = getattr(engine.tf_config, "num_moe_experts", None)
    for name, param in get_named_parameters(engine.model, experts):
        if id(param) in scale_ids:
            continue  # Export scales from the gathered logical weights, not as model weights.
        src = overrides.get(id(param), param)
        if src is not param:
            for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
                if hasattr(param, attr):
                    setattr(src, attr, getattr(param, attr))
        src = compute_source(src, source_quantization, source_scales)
        gathered = all_gather_param(
            name, src, fp8_direct_convert=False, quantization_config=source_quantization,
            duplicated_param_names=engine._duplicated_param_names,
            gated_linear_unit=bool(engine.tf_config.gated_linear_unit),
        )
        for hf_name, tensor in deepseek_convert_gathered(name, gathered):
            if getattr(engine.hf_config, "tie_word_embeddings", False) and hf_name == "lm_head.weight":
                continue
            if tensor.dtype not in (torch.bfloat16, torch.float32):
                raise NotImplementedError(f"DeepSeek BF16 AWEX unsupported dtype: {hf_name} {tensor.dtype}")
            vocab = getattr(engine.hf_config, "vocab_size", None)
            if vocab and hf_name in ("model.embed_tokens.weight", "lm_head.weight"):
                if tensor.shape[0] < vocab:
                    raise ValueError(f"DeepSeek vocabulary tensor is too short: {hf_name}")
                tensor = tensor[:vocab]
            # R1 has no DSA indexer: retain each parameter's original precision,
            # including the FP32 expert correction bias.
            yield from export_weight(hf_name, tensor.detach(), quantization)
        if consume_overrides:
            overrides.pop(id(param), None)


def deepseek_metadata_report(role, adapter, metadata):
    if not is_deepseek(adapter, role):
        return
    trace = os.getenv("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    if not trace:
        return
    path = Path(trace)
    path.mkdir(parents=True, exist_ok=True)
    rows = [dict(name=m.name, global_shape=list(m.global_shape), dtype=str(m.dtype),
                 shards=[dict(shape=list(s.shape), global_offset=list(s.global_offset),
                              sharding_type=str(s.sharding_type), sharding_dim=s.sharding_dim,
                              num_shards=s.num_shards, global_rank=s.global_rank)
                         for s in m.shards]) for m in metadata]
    (path / f"deepseek-{role}-pid{os.getpid()}.json").write_text(
        json.dumps(dict(role=role, metadata=rows), indent=2), encoding="utf-8"
    )


def deepseek_group_preflight(adapter, args, kwargs):
    pair_name = args[0] if args else kwargs.get("pair_name")
    kv_store_url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
    if not pair_name or not kv_store_url:
        raise TypeError("DeepSeek AWEX preflight requires pair_name and kv_store_url")
    from areal.v2.weight_update.awex import fetch_kv_metadata
    inf, trn = fetch_kv_metadata(kv_store_url, pair_name)
    count = assert_deepseek_transfer_metadata(inf, trn)
    print(f"[hcu_areal pid={os.getpid()}] DeepSeek AWEX metadata PASS: train/infer={count}",
          file=sys.stderr, flush=True)


def _mla_refresh(adapter):
    model = validate_inference_model(adapter)
    refresh = getattr(model, "post_load_weights", None)
    if not callable(refresh):
        raise RuntimeError("DeepSeek AWEX requires SGLang post_load_weights for MLA cache refresh")

    def run():
        import torch
        from areal.infra.platforms import current_platform
        from .deepseek_fp8 import refresh_context, commit_staging
        commit_staging(adapter)
        with torch.no_grad(), refresh_context(adapter):
            refresh()
        current_platform.synchronize()
    return run


def _refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_deepseek(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _mla_refresh(self)
        result = original(self, *args, **kwargs)
        refresh()
        return result
    return wrapper


def _colocate_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if not is_deepseek(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _mla_refresh(self)
        client = self._colocate_http_client
        completed = False

        class CompletionClient:
            def put(proxy, url, *put_args, **put_kwargs):
                nonlocal completed
                if "/colocate_done_rank" in str(url):
                    refresh()
                    completed = True
                return client.put(url, *put_args, **put_kwargs)

            def __getattr__(proxy, name):
                return getattr(client, name)

        self._colocate_http_client = CompletionClient()
        try:
            result = original(self, *args, **kwargs)
            if not completed:
                raise RuntimeError("DeepSeek colocate update did not publish its completion event")
            return result
        finally:
            self._colocate_http_client = client
    return wrapper


class DeepseekAwexFeature(AbstractFeature):
    feature_name = "deepseek_awex"
    default_patches = True
    enable_env = ""

    def enabled(self, env):
        return str(env.get("HCU_AREAL_DISABLE_DEEPSEEK_AWEX", "")).strip().lower() not in ("1", "true", "on", "yes")

    def register_patches(self, manager, env):
        # The manager forbids registering the same method twice. Existing
        # model-guarded hooks dispatch to this module; install fallbacks only
        # if the features that normally own those hooks were disabled.
        from . import glm5_awex_feature as glm
        from .qwen35_awex_feature import (
            _infer_parts_factory, _infer_strategy_factory, _infer_metadata_factory,
        )
        from .gemma3_awex_feature import _group_preflight_factory
        entries = [
            (TRAIN_MODULE, "AwexMegatronAdapter._iter_hf_params", glm._train_iter_factory,
             ("glm5.awex.train.params",), "deepseek.awex.train.params"),
            (TRAIN_MODULE, "AwexMegatronAdapter.get_weight_metadata", glm._train_metadata_factory,
             ("glm5.awex.train.meta",), "deepseek.awex.train.meta"),
            (TRAIN_MODULE, "AwexMegatronAdapter.init_weight_update_group", glm._train_preflight_factory,
             ("glm5.awex.train.preflight",), "deepseek.awex.train.preflight"),
            (INFER_MODULE, "AwexSGLangAdapter.execute_weight_update", glm._refresh_factory,
             ("glm5.awex.infer.refresh",), "deepseek.awex.infer.refresh"),
            (INFER_MODULE, "AwexSGLangAdapter.execute_colocate_weight_update", glm._colocate_refresh_factory,
             ("glm5.awex.infer.colocate_refresh",), "deepseek.awex.infer.colocate_refresh"),
            (INFER_MODULE, "AwexSGLangAdapter._unfuse_params", _infer_parts_factory,
             ("qwen35.awex.infer.unfuse", "glm5.awex.infer.unfuse"), "deepseek.awex.infer.unfuse"),
            (INFER_MODULE, "AwexSGLangAdapter._build_sharding_strategy", _infer_strategy_factory,
             ("qwen35.awex.infer.sharding", "glm5.awex.infer.sharding"), "deepseek.awex.infer.sharding"),
            (INFER_MODULE, "AwexSGLangAdapter.get_weight_metadata", _infer_metadata_factory,
             ("qwen35.awex.infer.meta", "glm5.awex.infer.meta"), "deepseek.awex.infer.meta"),
            (INFER_MODULE, "AwexSGLangAdapter.init_weight_update_group",
             lambda original: _group_preflight_factory(original, "infer"),
             ("gemma3.awex.infer.preflight", "glm5.awex.infer.preflight"), "deepseek.awex.infer.preflight"),
        ]
        for module, method, factory, aliases, patch_id in entries:
            if not any(marker in manager.patch_ids for marker in aliases):
                manager.register_patch(f"{module}.{method}", factory, module_name=module,
                                       patch_id=patch_id, apply_wrapper=True)
