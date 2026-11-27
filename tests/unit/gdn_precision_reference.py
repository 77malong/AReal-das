"""Independent CPU references for the Qwen3.5 Gated DeltaNet tests."""

import math

import torch
import torch.nn.functional as F

HIDDEN = 2048
HEADS = 16
DIM = 128
CONV = 4
NORM_EPS = 1e-6
PROJECTION_SECTIONS = (HEADS * DIM,) * 4 + (HEADS,) * 2

# Absolute floors and relative budgets for the BF16 module boundaries.
# Isolated operators have tighter budgets. Repeatability is checked separately.
MODULE_TOLERANCES = {
    "input_projection": (0.003, 0.015),
    "short_conv": (0.004, 0.020),
    "q": (0.0015, 0.025),
    "k": (0.0015, 0.025),
    "v": (0.004, 0.020),
    "g": (0.008, 0.025),
    "beta": (0.003, 0.020),
    "recurrence": (0.003, 0.040),
    "output": (0.004, 0.040),
}
OPERATOR_TOLERANCES = {
    "l2norm": (0.0, 0.0),  # One BF16 ULP is checked directly on the rounded reference.
    "g": (2e-6, 0.004),  # One BF16 rounding of exp(A_log) can precede promotion.
    "beta": (0.0, 0.0),  # One BF16 ULP.
    "recurrence": (2e-4, 0.010),
}
BF16_ULP_LIMIT = {"l2norm": 1, "beta": 1}


def l2norm(x):
    x64 = x.double()
    return (x64 * (x64.square().sum(dim=-1, keepdim=True) + NORM_EPS).rsqrt()).to(
        x.dtype
    )


def decay(a_log, dt_bias, alpha, beta):
    """Evaluate the gate formulas in FP64, with the production output dtypes."""
    g = -a_log.double().exp() * F.softplus(alpha.double() + dt_bias.double())
    return g.float(), beta.double().sigmoid().to(beta.dtype)


def recurrence(q, k, v, g, beta, lengths=None):
    """Evaluate the gated delta rule one token at a time using FP64 state."""
    output = []
    lengths = lengths or (q.shape[1],)
    tensors = [x.double().transpose(0, 1) for x in (q, k, v, g, beta)]
    start = 0
    for length in lengths:
        state = q.new_zeros(
            (q.size(0), q.size(2), q.size(-1), v.size(-1)), dtype=torch.float64
        )
        stop = start + length
        for qt, kt, vt, gt, bt in zip(*(x[start:stop] for x in tensors), strict=True):
            state.mul_(gt.exp()[..., None, None])
            key = kt.unsqueeze(-2)
            residual = vt - (key @ state).squeeze(-2)
            state.add_(kt.unsqueeze(-1) * (bt.unsqueeze(-1) * residual).unsqueeze(-2))
            output.append(
                ((qt.unsqueeze(-2) @ state) / math.sqrt(q.size(-1))).squeeze(-2)
            )
        start = stop
    return torch.stack(output, dim=1).to(q.dtype)


def split_sections(tensor, sections, tp):
    """Partition each Q/K/V/z/beta/alpha section before combining local shards."""
    return [
        torch.cat(slices)
        for slices in zip(
            *(section.chunk(tp, dim=0) for section in tensor.split(sections, dim=0)),
            strict=True,
        )
    ]


def weight_shards(weights, tp):
    in_proj = split_sections(weights["in_proj.weight"], PROJECTION_SECTIONS, tp)
    conv = split_sections(weights["conv1d.weight"], PROJECTION_SECTIONS[:3], tp)
    return [
        {
            "in_proj.weight": in_proj[rank],
            "conv1d.weight": conv[rank],
            "in_proj.layer_norm_weight": weights["in_proj.layer_norm_weight"],
            "out_norm.weight": weights["out_norm.weight"],
            "out_proj.weight": weights["out_proj.weight"].chunk(tp, dim=1)[rank],
            "A_log": weights["A_log"].chunk(tp)[rank],
            "dt_bias": weights["dt_bias"].chunk(tp)[rank],
        }
        for rank in range(tp)
    ]


def rmsnorm(x, weight):
    x64 = x.double()
    return (
        x64
        * (x64.square().mean(-1, keepdim=True) + NORM_EPS).rsqrt()
        * (1 + weight.double())
    ).to(x.dtype)


def causal_conv_silu(x, weight, lengths):
    width = weight.size(-1)
    weight = weight.double().transpose(0, 1)
    projected = []
    for segment in x.double().split(lengths, dim=1):
        padded = F.pad(segment, (0, 0, width - 1, 0))
        terms = [
            padded[:, off : off + segment.size(1)] * weight[off] for off in range(width)
        ]
        projected.append(F.silu(sum(terms)))
    return torch.cat(projected, dim=1).to(x.dtype)


def module_reference(hidden, weights, tp, lengths):
    """Evaluate all native module boundaries on CPU and return per-rank expected values."""
    results, partitions = [], []
    heads = HEADS // tp
    sections = [
        width // tp
        for width in (sum(PROJECTION_SECTIONS[:3]), *PROJECTION_SECTIONS[3:])
    ]
    for local in weight_shards(weights, tp):
        norm = rmsnorm(hidden, local["in_proj.layer_norm_weight"])
        in_proj = F.linear(norm.double(), local["in_proj.weight"].double()).to(
            hidden.dtype
        )
        qkv, z, beta, alpha = in_proj.transpose(0, 1).split(sections, dim=-1)
        conv = causal_conv_silu(qkv, local["conv1d.weight"].squeeze(1), lengths)
        q, k, v = (
            item.reshape(1, hidden.size(0), heads, DIM)
            for item in conv.chunk(3, dim=-1)
        )
        q, k = l2norm(q), l2norm(k)
        g, beta = decay(local["A_log"], local["dt_bias"], alpha, beta)
        core = recurrence(q, k, v, g, beta, lengths)
        norm = rmsnorm(core, local["out_norm.weight"])
        gated = (norm.double() * F.silu(z.double().reshape_as(norm))).to(hidden.dtype)
        gated = gated.reshape(1, hidden.size(0), heads * DIM).transpose(0, 1)
        partitions.append(
            F.linear(gated.double(), local["out_proj.weight"].double()).to(hidden.dtype)
        )
        results.append(
            {
                "input_projection": in_proj,
                "short_conv": conv,
                "q": q,
                "k": k,
                "v": v,
                "g": g,
                "beta": beta,
                "recurrence": core,
            }
        )
    output = sum(part.double() for part in partitions).to(hidden.dtype)
    for result, shard in zip(results, output.chunk(tp, dim=0), strict=True):
        result["output"] = shard
    return results
