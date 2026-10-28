"""Real checkpoint geometry, model isolation and DeepSeek AWEX failure paths."""
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from hcu_areal.features_manager.weight_update.deepseek_awex_layout import (
    is_deepseek, deepseek_inference_parts, DeepseekShardingProxy,
    deepseek_mapping_registry, deepseek_convert_gathered, assert_deepseek_transfer_metadata,
)
from hcu_areal.features_manager.weight_update.deepseek_awex_feature import (
    deepseek_train_iter, validate_inference_model,
)
from hcu_areal.features_manager.weight_update.glm5_awex_feature import (
    _train_iter_factory, _train_preflight_factory, _refresh_factory, _colocate_refresh_factory,
)
from hcu_areal.features_manager.weight_update.qwen35_awex_feature import (
    _infer_parts_factory, _infer_strategy_factory,
)
from hcu_areal.features_manager.weight_update.gemma3_awex_feature import _group_preflight_factory


def fixture():
    return json.loads((Path(__file__).parent/'data/deepseek_r1_4layers_shapes.json').read_text())


class TestDeepseekLayout(unittest.TestCase):
    def test_runtime_kv_scales_are_not_training_weights(self):
        import torch
        model=NS(config=NS(model_type='deepseek_v3'),quant_config=None)
        adapter=NS(_get_model=lambda:model)
        wrap=_infer_parts_factory(lambda self,name,tensor:[(name,tensor)])
        value=torch.tensor(1.0)
        for attention in ('mha','mqa'):
            for kind in ('k','v'):
                name=f'model.layers.3.self_attn.attn_{attention}.{kind}_scale'
                self.assertEqual(wrap(adapter,name,value),[])
        for name in ('model.layers.3.self_attn.q_b_proj.weight_scale_inv',
                     'model.layers.3.self_attn.attn_other.k_scale'):
            self.assertEqual(wrap(adapter,name,value),[(name,value)])

    def test_effective_mtp_state_not_stale_checkpoint_config(self):
        from hcu_areal.features_manager.weight_update.deepseek_awex_feature import _has_active_mtp
        base=NS(mtp_process=False)
        engine=NS(model=[NS(module=base)],tf_config=NS(mtp_num_layers=1),
                  mcore_config=NS(enable_mtp=False))
        self.assertFalse(_has_active_mtp(engine))
        engine.mcore_config.enable_mtp=True
        self.assertTrue(_has_active_mtp(engine))
        engine.mcore_config.enable_mtp=False
        base.mtp_process=True
        self.assertTrue(_has_active_mtp(engine))
        base.mtp_process=False
        del engine.mcore_config
        self.assertTrue(_has_active_mtp(engine))

    def test_guard_views_and_disable(self):
        import torch
        cfg = NS(model_type='deepseek_v3',q_lora_rank=4,kv_lora_rank=2,qk_rope_head_dim=2,hidden_size=3)
        adapter = NS(_get_model=lambda:NS(config=cfg))
        name = 'model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight'
        x = torch.arange(24).reshape(8,3).clone()
        fn = _infer_parts_factory(lambda self,n,t:[(n,t)])
        parts = fn(adapter,name,x)
        self.assertTrue(is_deepseek(adapter,'infer'))
        self.assertEqual(len(parts),2)
        parts[0][1].fill_(-1)
        self.assertTrue(torch.all(x[:4]==-1))
        self.assertTrue(torch.equal(x[4:],torch.arange(12,24).reshape(4,3)))
        parts[1][1].fill_(-2)
        self.assertTrue(torch.all(x[4:]==-2))
        with self.assertRaisesRegex(ValueError,'Invalid DeepSeek fused'):
            deepseek_inference_parts(name,x[:7],cfg)
        with patch.dict(os.environ,{'HCU_AREAL_DISABLE_DEEPSEEK_AWEX':'1'}):
            self.assertFalse(is_deepseek(adapter,'infer'))
            self.assertEqual(len(fn(adapter,name,x)),1)
        for kind in ('deepseek_v2','glm4_moe','qwen2'):
            cfg.model_type=kind
            self.assertFalse(is_deepseek(adapter,'infer'))
            self.assertEqual(len(fn(adapter,name,x)),1)

    def test_replica_semantics_and_base_fallback(self):
        from awex.sharding.param_sharding import ShardingType
        base=NS(get_sharding_strategy=lambda *a,**k:('base',1,8))
        adapter=NS(_get_model=lambda:NS(config=NS(model_type='deepseek_v3')))
        strategy=_infer_strategy_factory(lambda self,rank:base)(adapter,NS(tp_size=8))
        self.assertIsInstance(strategy,DeepseekShardingProxy)
        for name in ('model.layers.0.self_attn.q_a_proj.weight',
                     'model.layers.0.self_attn.kv_a_proj_with_mqa.weight',
                     'model.layers.3.mlp.gate.e_score_correction_bias'):
            self.assertEqual(strategy.get_sharding_strategy(name),(ShardingType.NO_SHARDING,0,1))
        self.assertEqual(strategy.get_sharding_strategy('model.layers.0.self_attn.q_b_proj.weight'),('base',1,8))

    def test_metadata_rejects_corruption(self):
        meta=lambda name,shape=(3,4),dtype='bf16':NS(name=name,global_shape=shape,dtype=dtype)
        good=[meta('a'),meta('b')]
        self.assertEqual(assert_deepseek_transfer_metadata(good,good),2)
        for infer,train,pattern in (
            (good[:1],good,'missing=1'),
            (good+[meta('extra')],good,'extra=1'),
            ([meta('a',(4,3)),meta('b')],good,'shape_mismatch=1'),
            ([meta('a',dtype='fp32'),meta('b')],good,'dtype_mismatch=1'),
            ([good[0],good[0]],good,'duplicate infer'),
            ([],[],'must not be empty'),
        ):
            with self.assertRaisesRegex(ValueError,pattern):
                assert_deepseek_transfer_metadata(infer,train)

    def test_both_group_preflights_reject_before_creating_groups(self):
        import areal.v2.weight_update.awex as awex
        good=[NS(name='a',global_shape=(2,),dtype='bf16')]
        def should_not_run(*a,**k):
            self.fail('Communication groups must not be created with mismatching metadata')
        train=NS(_engine=NS(hf_config=NS(model_type='deepseek_v3')))
        infer=NS(_get_model=lambda:NS(config=NS(model_type='deepseek_v3')))
        with patch.object(awex,'fetch_kv_metadata',return_value=([],good)):
            for fn,adapter in ((_train_preflight_factory(should_not_run),train),
                               (_group_preflight_factory(should_not_run,'infer'),infer)):
                with self.assertRaisesRegex(ValueError,'missing=1'):
                    fn(adapter,pair_name='test',kv_store_url='http://local')

    def test_unsupported_quantization_and_mtp_fail_closed(self):
        model=NS(config=NS(model_type='deepseek_v3'),quant_config=object())
        adapter=NS(_get_model=lambda:model)
        with self.assertRaisesRegex(NotImplementedError,'unquantized'):
            validate_inference_model(adapter)
        model.quant_config=None
        model.config.num_nextn_predict_layers=1
        # The HF checkpoint may declare optional MTP while its base model does
        # not execute it. Only an actual NextN runtime must be rejected.
        validate_inference_model(adapter)
        model.config.architectures=['DeepseekV3ForCausalLMNextN']
        with self.assertRaisesRegex(NotImplementedError,'MTP'):
            validate_inference_model(adapter)

    def test_feature_registration_with_every_legacy_feature_toggle(self):
        from hcu_areal.areal_adaptor import patch_features
        keys=('HCU_AREAL_DISABLE_QWEN35_AWEX','HCU_AREAL_DISABLE_GEMMA3_AWEX','HCU_AREAL_DISABLE_GLM5_AWEX')
        for bits in itertools.product(('0','1'),repeat=3):
            env=dict(zip(keys,bits))
            env['HCU_AREAL_DISABLE_DEEPSEEK_AWEX']='0'
            manager=patch_features(env)
            targets={entry.target_path for entry in manager._registered.values()}
            self.assertEqual(len(targets),len(manager.patch_ids))
            for name in ('_iter_hf_params','get_weight_metadata','init_weight_update_group'):
                self.assertIn('areal.v2.weight_update.awex.megatron_adapter.AwexMegatronAdapter.'+name,targets)
            for name in ('_unfuse_params','_build_sharding_strategy','get_weight_metadata',
                         'init_weight_update_group','execute_weight_update','execute_colocate_weight_update'):
                self.assertIn('areal.v2.weight_update.awex.sglang_adapter.AwexSGLangAdapter.'+name,targets)


class TestDeepseekRefresh(unittest.TestCase):
    def test_receive_refresh_and_failed_update(self):
        events=[]
        model=NS(config=NS(model_type='deepseek_v3'),quant_config=None,
                 post_load_weights=lambda:events.append('refresh'))
        adapter=NS(_get_model=lambda:model)
        def receive(self,version):
            events.append('receive')
            return 'done'
        with patch('areal.infra.platforms.current_platform',NS(synchronize=lambda:events.append('sync'))):
            self.assertEqual(_refresh_factory(receive)(adapter,1),'done')
            self.assertEqual(events,['receive','refresh','sync'])
            events.clear()
            def failed(*a):raise RuntimeError('failed')
            with self.assertRaisesRegex(RuntimeError,'failed'):
                _refresh_factory(failed)(adapter,2)
            self.assertEqual(events,[])
            del model.post_load_weights
            with self.assertRaisesRegex(RuntimeError,'post_load_weights'):
                _refresh_factory(receive)(adapter,3)
            self.assertEqual(events,[])

    def test_colocate_ack_order_and_client_restoration(self):
        events=[]
        model=NS(config=NS(model_type='deepseek_v3'),quant_config=None,
                 post_load_weights=lambda:events.append('refresh'))
        client=NS(put=lambda *a,**k:events.append('ack'))
        adapter=NS(_get_model=lambda:model,_colocate_http_client=client)
        def receive(self,version):
            events.append('receive')
            self._colocate_http_client.put('http://local/colocate_done_rank0_1')
        with patch('areal.infra.platforms.current_platform',NS(synchronize=lambda:events.append('sync'))):
            _colocate_refresh_factory(receive)(adapter,1)
            self.assertEqual(events,['receive','refresh','sync','ack'])
            self.assertIs(adapter._colocate_http_client,client)
            with self.assertRaisesRegex(RuntimeError,'completion event'):
                _colocate_refresh_factory(lambda *a:None)(adapter,2)
            self.assertIs(adapter._colocate_http_client,client)


class TestDeepseekCheckpoint(unittest.TestCase):
    def test_all_821_native_bridge_mappings(self):
        import torch
        headers=fixture()['tensors']
        registry=deepseek_mapping_registry()
        mappings={}
        for name in headers:
            mapping=registry.hf_to_megatron_lookup(name)
            self.assertIsNotNone(mapping,name)
            mappings[mapping.megatron_param]=mapping
        converted={}
        for name,mapping in mappings.items():
            dst=mapping.hf_param
            info=headers[dst['gate']] if isinstance(dst,dict) else headers[dst]
            shape=list(info['shape'])
            if isinstance(dst,dict):shape[0]*=2
            tensor=torch.empty(shape,device='meta',dtype={'BF16':torch.bfloat16,'F32':torch.float32}[info['dtype']])
            for key,value in deepseek_convert_gathered('module.module.'+name,tensor):
                self.assertNotIn(key,converted)
                self.assertEqual(list(value.shape),headers[key]['shape'],key)
                self.assertEqual(value.dtype,{'BF16':torch.bfloat16,'F32':torch.float32}[headers[key]['dtype']])
                converted[key]=value
        self.assertEqual(set(converted),set(headers))
        with self.assertRaisesRegex(ValueError,'Unsupported DeepSeek AWEX mapping'):
            deepseek_convert_gathered('unknown.weight',torch.empty(1))

    def test_tp8_real_adapter_geometry(self):
        import torch
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
        data=fixture()
        cfg=NS(**data['config'])
        runtime=[]
        for name,info in data['tensors'].items():
            shape=list(info['shape'])
            if name.endswith('.q_a_proj.weight'):
                shape[0]+=cfg.kv_lora_rank+cfg.qk_rope_head_dim
                name=name.replace('q_a_proj','fused_qkv_a_proj_with_mqa')
            elif name.endswith('.kv_a_proj_with_mqa.weight'):
                continue
            elif not ('norm' in name or '.mlp.gate.' in name):
                axis=1 if name.endswith(('.o_proj.weight','.down_proj.weight')) else 0
                self.assertEqual(shape[axis]%8,0,name)
                shape[axis]//=8
            dtype={'BF16':torch.bfloat16,'F32':torch.float32}[info['dtype']]
            runtime.append((name,torch.nn.Parameter(torch.empty(shape,dtype=dtype,device='meta'),requires_grad=False)))
        self.assertEqual(len(runtime),817)
        model=NS(config=cfg,named_parameters=lambda:iter(runtime))
        expected=[NS(name=n,global_shape=i['shape'],dtype={'BF16':torch.bfloat16,'F32':torch.float32}[i['dtype']]) for n,i in data['tensors'].items()]
        all_meta=[]
        for rank in range(8):
            server=NS(tp_size=8,pp_size=1,dp_size=1,ep_size=1,enable_dp_attention=False,enable_dp_lm_head=False,moe_dense_tp_size=None)
            scheduler=NS(server_args=server,tp_rank=rank,pp_rank=0,tp_worker=NS(model_runner=NS(model=model)))
            adapter=AwexSGLangAdapter(scheduler)
            metadata=adapter.get_weight_metadata()
            self.assertEqual(assert_deepseek_transfer_metadata(metadata,expected),821)
            self.assertEqual(set(adapter.get_local_shard_parameters()),set(data['tensors']))
            all_meta.append({m.name:m for m in metadata})
        for name,info in data['tensors'].items():
            shards=[row[name].shards[0] for row in all_meta]
            first=shards[0]
            if first.num_shards==1:
                self.assertTrue(all(not any(s.global_offset) for s in shards))
            else:
                axis=first.sharding_dim
                ranges=sorted((s.global_offset[axis],s.global_offset[axis]+s.shape[axis]) for s in shards)
                self.assertEqual(ranges[0][0],0)
                self.assertEqual(ranges[-1][1],info['shape'][axis])
                self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))

    def test_train_override_glu_dtype_and_vocab_padding(self):
        import torch
        p=torch.nn.Parameter(torch.arange(24).reshape(8,3).bfloat16())
        p.tensor_model_parallel,p.partition_dim,p.partition_stride=True,0,1
        bias=torch.nn.Parameter(torch.arange(256).float()*0.000001,requires_grad=False)
        embedding=torch.nn.Parameter(torch.arange(24).reshape(8,3).bfloat16())
        engine=NS(hf_config=NS(model_type='deepseek_v3',vocab_size=6),config=NS(use_lora=False),model=None,
                  tf_config=NS(num_moe_experts=256,gated_linear_unit=True),_duplicated_param_names=set())
        adapter=NS(_engine=engine)
        theta=p.detach().clone()+100
        overrides={id(p):theta}
        named=[('module.module.decoder.layers.0.mlp.linear_fc1.weight',p),
               ('module.module.decoder.layers.3.mlp.router.expert_bias',bias),
               ('module.module.embedding.word_embeddings.weight',embedding)]
        with patch('areal.engine.megatron_utils.megatron.get_named_parameters',return_value=named), patch(
                'areal.engine.megatron_utils.megatron.all_gather_param',side_effect=lambda name,src,**k:src) as gather:
            parts=dict(_train_iter_factory(lambda *a:iter(()))(adapter,overrides,True))
        self.assertTrue(all(call.kwargs['gated_linear_unit'] for call in gather.call_args_list))
        self.assertEqual(overrides,{})
        self.assertTrue(torch.equal(parts['model.layers.0.mlp.gate_proj.weight'],theta[:4]))
        self.assertTrue(torch.equal(parts['model.layers.0.mlp.up_proj.weight'],theta[4:]))
        self.assertTrue(torch.equal(parts['model.layers.3.mlp.gate.e_score_correction_bias'],bias))
        self.assertEqual(parts['model.layers.3.mlp.gate.e_score_correction_bias'].dtype,torch.float32)
        self.assertEqual(tuple(parts['model.embed_tokens.weight'].shape),(6,3))


if __name__=='__main__':
    unittest.main()
