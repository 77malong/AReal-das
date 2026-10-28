"""Gemma3 AWEX regressions, no HCU or imports of installed AWEX required."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import types
import unittest
from unittest.mock import patch

from hcu_areal.features_manager.weight_update.gemma3_awex_layout import (
    is_gemma3, gemma3_train_to_infer_name, gemma3_inference_parts,
    Gemma3ShardingProxy, assert_gemma3_transfer_metadata,
)
from hcu_areal.features_manager.weight_update.gemma3_awex_feature import _train_name_factory
from hcu_areal.features_manager.weight_update.qwen35_awex_feature import (
    _infer_parts_factory, _infer_strategy_factory,
)


class TestGemma3Names(unittest.TestCase):
    def test_exact_883_names_from_bw67_log(self):
        f = Path(__file__).parent / 'data' / 'gemma3_883_names.json'
        fixture = json.loads(f.read_text(encoding='utf-8'))
        trn = fixture['train']
        inf = fixture['infer']
        self.assertEqual(len(trn), 883)
        self.assertEqual(len(inf), 883)
        names = [gemma3_train_to_infer_name(n) for n in trn]
        self.assertEqual(set(names), set(inf))
        self.assertEqual(len(set(names)), 883)
        self.assertEqual(sum('.self_attn.out_proj.' in n for n in trn), 54)

    def test_non_gemma_names_unchanged(self):
        for name in ('model.layers.0.q_proj.weight', 'visual.blocks.0.attn.qkv.weight', 'lm_head.weight'):
            self.assertEqual(gemma3_train_to_infer_name(name), name)

    def test_guard_only_gemma3_composite_model(self):
        train = SimpleNamespace(_engine=SimpleNamespace(model_config=SimpleNamespace(model_type='gemma3')))
        self.assertTrue(is_gemma3(train, 'train'))
        train._engine.model_config.model_type = 'qwen3_5'
        self.assertFalse(is_gemma3(train, 'train'))
        train._engine.model_config.model_type = 'gemma3_text'
        self.assertFalse(is_gemma3(train, 'train'))
        infer = SimpleNamespace(_get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type='gemma3')))
        self.assertTrue(is_gemma3(infer, 'infer'))
        with patch.dict(os.environ, {'HCU_AREAL_DISABLE_GEMMA3_AWEX':'1'}):
            self.assertFalse(is_gemma3(infer, 'infer'))

    def test_adapter_name_wrapper(self):
        adapter = SimpleNamespace(_engine=SimpleNamespace(model_config=SimpleNamespace(model_type='gemma3')))
        wrapper = _train_name_factory(lambda self, n: n)
        self.assertEqual(wrapper(adapter, 'model.language_model.layers.1.mlp.up_proj.weight'),
                         'language_model.model.layers.1.mlp.up_proj.weight')
        adapter._engine.model_config.model_type='qwen3_5'
        self.assertEqual(wrapper(adapter, 'model.language_model.layers.1.mlp.up_proj.weight'),
                         'model.language_model.layers.1.mlp.up_proj.weight')


class TestGemma3Views(unittest.TestCase):
    def test_vision_qkv_weight_bias_are_zero_copy_views(self):
        import torch
        for suffix,shape in [('weight',(48,64)), ('bias',(48,))]:
            x=torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(shape)
            name='vision_tower.vision_model.encoder.layers.11.self_attn.qkv_proj.'+suffix
            y=gemma3_inference_parts(name,x)
            self.assertEqual([n.split('.')[-2] for n,_ in y],['q_proj','k_proj','v_proj'])
            self.assertEqual([t.shape[0] for _,t in y],[16,16,16])
            self.assertTrue(torch.equal(torch.cat([t for _,t in y],dim=0),x))
            y[1][1].zero_()
            self.assertTrue(torch.all(x[16:32]==0))

    def test_vision_view_not_applied_to_text(self):
        import torch
        x=torch.arange(48).float()
        self.assertIsNone(gemma3_inference_parts('language_model.model.layers.1.self_attn.qkv_proj.weight', x))

    def test_shared_qwen_infer_wrapper_dispatches_gemma3(self):
        import torch
        adapter = SimpleNamespace(_get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type='gemma3')))
        f=_infer_parts_factory(lambda self,n,x: [(n,x)])
        x=torch.arange(48).float()
        self.assertEqual(len(f(adapter, 'vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.bias',x)),3)
        self.assertEqual(len(f(adapter, 'language_model.model.layers.0.self_attn.o_proj.weight',x)),1)


class TestGemma3Sharding(unittest.TestCase):
    def test_gemma3_inference_sharding_overrides(self):
        class FakeEnum:
            NO_SHARDING = 'no'
            TP_SHARDING = 'tp'
        stubs = {ns:sys.modules.get(ns, types.ModuleType(ns)) for ns in ('awex', 'awex.sharding')}
        stubs['awex.sharding.param_sharding'] = SimpleNamespace(ShardingType=FakeEnum)
        with patch.dict(sys.modules, stubs):
            base=SimpleNamespace(get_sharding_strategy=lambda n,**kw: ('legacy',0,4))
            strategy=Gemma3ShardingProxy(base,4)
            self.assertEqual(strategy.get_sharding_strategy('language_model.model.embed_tokens.weight'),('no',0,1))
            self.assertEqual(strategy.get_sharding_strategy('multi_modal_projector.mm_input_projection_weight'),('no',0,1))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.embeddings.patch_embedding.weight'),('no',0,1))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.encoder.layers.3.self_attn.proj.weight'),('tp',1,4))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.encoder.layers.3.mlp.fc2.weight'),('tp',1,4))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.encoder.layers.3.self_attn.proj.bias'),('no',0,1))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.encoder.layers.3.mlp.fc2.bias'),('no',0,1))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight'),('legacy',0,4))
            self.assertEqual(strategy.get_sharding_strategy('vision_tower.vision_model.embeddings.position_embedding.weight'),('legacy',0,4))

    def test_shared_qwen_strategy_wrapper_does_not_change_other_model(self):
        class FakeEnum:
            NO_SHARDING = 'no'
            TP_SHARDING = 'tp'
        stubs = {ns:sys.modules.get(ns, types.ModuleType(ns)) for ns in ('awex', 'awex.sharding')}
        stubs['awex.sharding.param_sharding'] = SimpleNamespace(ShardingType=FakeEnum)
        with patch.dict(sys.modules, stubs):
            adapter=SimpleNamespace(_get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type='gemma3')))
            proxy=_infer_strategy_factory(lambda self,r:'base')(adapter,SimpleNamespace(tp_size=4))
            self.assertIsInstance(proxy,Gemma3ShardingProxy)
            adapter._get_model=lambda: SimpleNamespace(config=SimpleNamespace(model_type='qwen2'))
            self.assertEqual(_infer_strategy_factory(lambda self,r:'base')(adapter,SimpleNamespace(tp_size=4)), 'base')


class TestPreflight(unittest.TestCase):
    @staticmethod
    def meta(n,shape,dt='torch.bfloat16'):
        return SimpleNamespace(name=n,global_shape=shape,dtype=dt)

    def test_preflight_pass_and_fail_closed(self):
        trn=[self.meta('a',(8,16)),self.meta('b',(3,5))]
        inf=[self.meta('b',(3,5)),self.meta('a',(8,16))]
        self.assertEqual(assert_gemma3_transfer_metadata(inf,trn),2)
        inf[0].global_shape=(3,6)
        with self.assertRaisesRegex(ValueError,'shape_mismatch=1'):
            assert_gemma3_transfer_metadata(inf,trn)
        inf[0].global_shape=(3,5)
        inf[0].dtype='torch.float32'
        with self.assertRaisesRegex(ValueError,'dtype_mismatch=1'):
            assert_gemma3_transfer_metadata(inf,trn)
        with self.assertRaisesRegex(ValueError,'missing=1'):
            assert_gemma3_transfer_metadata(inf[1:],trn)

if __name__=='__main__':
    unittest.main()
