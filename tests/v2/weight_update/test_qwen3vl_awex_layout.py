# SPDX-License-Identifier: Apache-2.0
"""Shape and canonical-layout smoke tests for Qwen3-VL AWEX DP/TP export."""

import pytest
import torch

from areal.v2.weight_update.awex.vlm_layout import (
    is_qwen3_vl_vision_qkv,
    normalize_qwen3_vl_train_name,
    normalize_qwen3_vl_visual_infer_name,
    pack_vision_qkv_rank_major,
    qwen3_vl_vision_sharding_dim,
)


@pytest.mark.parametrize(
    "old,expected",
    [
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.attn.qkv.weight",
        ),
        ("visual.pos_embed.weight", "model.visual.pos_embed.weight"),
    ],
)
def test_train_name_normalization(old, expected):
    assert normalize_qwen3_vl_train_name(old) == expected


def test_visual_infer_normalization():
    assert normalize_qwen3_vl_visual_infer_name(
        "visual.blocks.0.attn.qkv_proj.weight"
    ) == "model.visual.blocks.0.attn.qkv.weight"
    assert is_qwen3_vl_vision_qkv("model.visual.blocks.0.attn.qkv.weight")


@pytest.mark.parametrize("shape", [(96,), (96, 12)])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_repack_matches_per_rank_qkv_blocks(shape, tp):
    src = torch.arange(int(torch.tensor(shape).prod().item())).reshape(shape)
    dst = pack_vision_qkv_rank_major(src, tp)
    q, k, v = src.chunk(3, dim=0)
    for rank, dst_shard in enumerate(dst.chunk(tp, dim=0)):
        expected = torch.cat(
            [part.chunk(tp, dim=0)[rank] for part in (q, k, v)], dim=0
        )
        assert torch.equal(dst_shard, expected)


def test_invalid_padding_fails_closed():
    with pytest.raises(ValueError, match="dummy heads"):
        pack_vision_qkv_rank_major(torch.zeros(27, 8), tp_size=4)


@pytest.mark.parametrize(
    "name,dim",
    [
        ("model.visual.patch_embed.proj.weight", None),
        ("model.visual.pos_embed.weight", 0),
        ("model.visual.blocks.0.norm1.weight", None),
        ("model.visual.blocks.0.attn.qkv.weight", 0),
        ("model.visual.blocks.0.attn.qkv.bias", 0),
        ("model.visual.blocks.0.attn.proj.weight", 1),
        ("model.visual.blocks.0.attn.proj.bias", None),
        ("model.visual.blocks.0.mlp.linear_fc1.weight", 0),
        ("model.visual.blocks.0.mlp.linear_fc1.bias", 0),
        ("model.visual.blocks.0.mlp.linear_fc2.weight", 1),
        ("model.visual.blocks.0.mlp.linear_fc2.bias", None),
        ("model.visual.merger.linear_fc1.weight", 0),
        ("model.visual.deepstack_merger_list.0.linear_fc2.weight", 1),
    ],
)
def test_visual_sharding_contract(name, dim):
    assert qwen3_vl_vision_sharding_dim(name) == dim
