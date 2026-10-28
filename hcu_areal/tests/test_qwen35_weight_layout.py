"""CPU numerical tests for Qwen3.5 AWEX view layouts (no AWEX installation)."""
import unittest
from types import SimpleNamespace

import torch

from hcu_areal.features_manager.weight_update.qwen35_awex_layout import (
    canonical_name, segments_for_training, slice_segment, inference_parts,
    infer_strategy_override,
)

CONFIG = SimpleNamespace(
    model_type="qwen3_5", linear_num_key_heads=16, linear_key_head_dim=128,
    linear_num_value_heads=32, linear_value_head_dim=128,
    num_attention_heads=16, num_key_value_heads=4,
    head_dim=256, hidden_size=2560, attn_output_gate=True,
)


def apply_rectangles(train_views, infer_views):
    """Copy each overlapping source rectangle into the target view, like AWEX planner."""
    for name, (target, target_offset) in infer_views.items():
        owned = 0
        for src, src_off in train_views.get(name, []):
            b = max(src_off, target_offset)
            e = min(src_off + len(src), target_offset + len(target))
            if b < e:
                target[b-target_offset:e-target_offset].copy_(src[b-src_off:e-src_off])
                owned += e-b
        if owned != len(target):
            raise AssertionError(f"AWEX would leave holes in {name}: {owned}/{len(target)}")


def train_views(names_and_full):
    sources = {}
    for name, full in names_and_full.items():
        for rank in range(4):
            assert len(full)%4 == 0
            n = len(full)//4
            local = full[rank*n:(rank+1)*n]
            for new_name, start, length in segments_for_training(name, full.shape, CONFIG):
                part = slice_segment(local, (rank*n,), full.shape, start, length)
                if part is not None:
                    tensor, offs, gshape = part
                    sources.setdefault(new_name, []).append((tensor, offs[0]))
    return sources


class TestCanonicalNames(unittest.TestCase):
    def test_conversions_and_non_qwen3_collisions(self):
        self.assertEqual(canonical_name('model.language_model.layers.11.self_attn.q_proj.weight'),
                         'model.layers.11.q_proj.weight')
        self.assertEqual(canonical_name('model.visual.blocks.0.attn.qkv.weight'),
                         'visual.blocks.0.attn.qkv.weight')
        self.assertEqual(infer_strategy_override('visual.patch_embed.proj.bias', 4),('none',0,1))
        self.assertEqual(infer_strategy_override('visual.blocks.0.attn.proj.weight', 4),('tp',1,4))
        self.assertEqual(infer_strategy_override('visual.blocks.0.mlp.linear_fc2.weight', 4),('tp',1,4))

    def test_training_fragment_across_segment_boundary(self):
        full = torch.arange(3072, dtype=torch.float32).reshape(-1, 1)
        pieces = train_views({'model.visual.blocks.0.attn.qkv.weight':full})
        self.assertEqual(len(pieces['visual.blocks.0.attn.q_proj.weight']),2)
        self.assertEqual([off for _,off in pieces['visual.blocks.0.attn.q_proj.weight']],[0,768])
        self.assertEqual(len(pieces['visual.blocks.0.attn.k_proj.weight']),2)
        self.assertEqual(len(pieces['visual.blocks.0.attn.v_proj.weight']),2)


class TestAwexNumericalLayout(unittest.TestCase):
    def _compare(self, train, fused_kind, infer_parts):
        src = train_views(train)
        reference = {}
        for raw_name, full in train.items():
            for cname,start,size in segments_for_training(raw_name, full.shape, CONFIG):
                reference[cname] = full.narrow(0,start,size)
        for rank in range(4):
            dims = (sum(infer_parts), *next(iter(train.values())).shape[1:])
            target = torch.full(dims, -1, dtype=torch.float32)
            mapped = inference_parts(fused_kind, target, CONFIG, 4)
            self.assertIsNotNone(mapped)
            assert len(mapped) == len(infer_parts)
            dest = {}
            for (name,view), n in zip(mapped, infer_parts):
                self.assertEqual(len(view), n)
                dest[name] = (view, rank*n)
            apply_rectangles(src, dest)
            self.assertFalse(bool((target == -1).any()),f"Some dst rows were not filled for rank={rank}")
            # AWEX operations populate the actual fused target tensor.
            expected = torch.cat([
                reference[name].narrow(0, rank*n, n) for (name, _), n in zip(mapped, infer_parts)
            ], dim=0)
            self.assertTrue(torch.equal(target, expected))

    def test_gated_delta_net_qkvz(self):
        qkv = torch.arange(8192, dtype=torch.float32).reshape(-1,1)+100000
        z = torch.arange(4096,dtype=torch.float32).reshape(-1,1)+200000
        self._compare({'model.language_model.layers.0.linear_attn.in_proj_qkv.weight':qkv,
                       'model.language_model.layers.0.linear_attn.in_proj_z.weight':z},
                      'model.layers.0.linear_attn.in_proj_qkvz.weight', (512,512,1024,1024))

    def test_gated_delta_net_conv1d(self):
        conv = torch.arange(8192,dtype=torch.float32).reshape(-1,1)+300000
        self._compare({'model.language_model.layers.0.linear_attn.conv1d.weight':conv},
                      'model.layers.0.linear_attn.conv1d.weight',(512,512,1024))

    def test_ba(self):
        b=torch.arange(32,dtype=torch.float32).reshape(-1,1)+400000
        a=torch.arange(32,dtype=torch.float32).reshape(-1,1)+500000
        self._compare({'model.language_model.layers.0.linear_attn.in_proj_b.weight':b,
                       'model.language_model.layers.0.linear_attn.in_proj_a.weight':a},
                      'model.layers.0.linear_attn.in_proj_ba.weight',(8,8))

    def test_vision_qkv(self):
        qkv=torch.arange(3072,dtype=torch.float32).reshape(-1,1)+600000
        self._compare({'model.visual.blocks.0.attn.qkv.weight':qkv},
                      'visual.blocks.0.attn.qkv_proj.weight',(256,256,256))

    def test_vision_bias(self):
        qkv=torch.arange(3072,dtype=torch.float32).reshape(-1)+700000
        self._compare({'model.visual.blocks.0.attn.qkv.bias':qkv},
                      'visual.blocks.0.attn.qkv_proj.bias',(256,256,256))

    def test_full_attention_gated_qkv(self):
        q=torch.arange(8192,dtype=torch.float32).reshape(-1,1)+800000
        k=torch.arange(1024,dtype=torch.float32).reshape(-1,1)+900000
        v=torch.arange(1024,dtype=torch.float32).reshape(-1,1)+1000000
        self._compare({'model.language_model.layers.11.self_attn.q_proj.weight':q,
                       'model.language_model.layers.11.self_attn.k_proj.weight':k,
                       'model.language_model.layers.11.self_attn.v_proj.weight':v},
                      'model.layers.11.qkv_proj.weight',(2048,256,256))

if __name__=='__main__':
    unittest.main()
