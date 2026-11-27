"""Forward accuracy and repeatability of native GDN on NVIDIA and HCU."""

from importlib import import_module
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ref = import_module("gdn_precision_reference")


def test_cpu_l2norm_reference():
    x = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        ref.l2norm(x), x / math.sqrt(25 + ref.NORM_EPS), atol=1e-15, rtol=0
    )
    assert torch.count_nonzero(ref.l2norm(torch.zeros(2, 3))) == 0


def test_cpu_decay_reference():
    a_log, dt = torch.zeros(1), torch.zeros(1)
    alpha, beta = torch.zeros(1, 2, 1), torch.zeros(1, 2, 1)
    g, b = ref.decay(a_log, dt, alpha, beta)
    torch.testing.assert_close(g, torch.full_like(g, -math.log(2)), atol=0, rtol=0)
    torch.testing.assert_close(b, torch.full_like(b, 0.5), atol=0, rtol=0)


def test_cpu_delta_rule_reference():
    # The second token decays the first state from 1 to 0.5 and adds half of 4 - 0.5.
    q = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64).repeat(1, 2, 1, 1)
    v = torch.tensor([2.0, 4.0], dtype=torch.float64).reshape(1, 2, 1, 1)
    g = torch.full((1, 2, 1), math.log(0.5), dtype=torch.float64)
    beta = torch.full_like(g, 0.5)
    assert ref.recurrence(q, q, v, g, beta).flatten().tolist() == [0.5, 1.125]
    assert ref.recurrence(q, q, v, g, beta, (1, 1)).flatten().tolist() == [0.5, 1.0]


@pytest.fixture(scope="module")
def device():
    backend = os.environ.get("GDN_PRECISION_BACKEND")
    if not backend:
        pytest.skip("Set GDN_PRECISION_BACKEND for an explicit hardware run")

    import torch.distributed as dist

    import gdn_precision_device as gpu

    tp = int(os.environ["WORLD_SIZE"])
    input_dir = Path(os.environ["GDN_PRECISION_FIXTURE"])
    output_dir = Path(os.environ["GDN_PRECISION_OUTPUT"]) / f"rank-{os.environ['RANK']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data, references, manifest = gpu.load_fixture(input_dir, tp)
    mpu = gpu.setup_device(backend, tp)
    try:
        gpu.write_json(output_dir / "runtime.json", gpu.runtime(backend, mpu, manifest))
        yield gpu, mpu, tp, data, references, output_dir
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def assert_results(report):
    precision = report["first_precision_failure"]
    repeats = {
        pair: next((name for name, count in differences.items() if count), None)
        for pair, differences in report["within_start"].items()
    }
    assert not any(precision) and not any(repeats.values()), {
        "first_precision_failure": precision,
        "first_repeatability_failure": repeats,
    }


@pytest.mark.parametrize(
    "name",
    (
        "l2norm-synthetic",
        "l2norm-captured",
        "gate-synthetic",
        "gate-captured",
        "recurrence-tail",
    ),
)
def test_device_operator(device, name):
    gpu, _, tp, data, _, output_dir = device
    if tp != 1:
        pytest.skip("Operators are exercised by the single-card job")
    report = gpu.measure(
        gpu.build_operator(data["operators"][name]),
        data["operator_refs"][name],
        ref.OPERATOR_TOLERANCES,
        ref.BF16_ULP_LIMIT,
        output_dir / name,
        repeats=3,
    )
    assert_results(report)


@pytest.mark.parametrize("name", ("short64", "short65", "mb0", "mb1", "packed"))
def test_device_module(device, name):
    gpu, mpu, tp, data, references, output_dir = device
    if name not in references:
        pytest.skip(f"{name} belongs to the other TP configuration")
    module = gpu.build_module(data["weights"], tp, mpu)
    rank = mpu.get_tensor_model_parallel_rank()
    observed = gpu.ObservedModule(module, data["cases"][name], rank, tp)
    report = gpu.measure(
        observed,
        references[name][rank],
        ref.MODULE_TOLERANCES,
        {},
        output_dir / name,
        repeats=3,
    )
    assert_results(report)
