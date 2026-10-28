"""FP8 mode dispatch, real byte/scale views, native checkpoint dequantization."""
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hcu_areal.features_manager.weight_update import deepseek_fp8 as fp8
from hcu_areal.features_manager.weight_update.deepseek_awex_layout import DeepseekShardingProxy
from hcu_areal.features_manager.weight_update.deepseek_fp8_source_feature import _load_factory
from hcu_areal.features_manager.weight_update.qwen35_awex_feature import _infer_parts_factory


def config(block=2):
    return dict(quant_method='fp8',fmt='e4m3',activation_scheme='dynamic',weight_block_size=[block,block])


def hipblaslt_w8a8_block_fp8_linear():pass


def model(layers=1,fnuz=False):
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    root=nn.Module()
    root.config=NS(model_type='deepseek_v3',num_hidden_layers=layers,q_lora_rank=4,
                   kv_lora_rank=2,qk_rope_head_dim=2,hidden_size=8,n_routed_experts=2,
                   architectures=['DeepseekV3ForCausalLM'],num_nextn_predict_layers=1)
    root.quant_config=Fp8Config.from_config(config())
    root.model=nn.Module()
    root.model.layers=nn.ModuleList([nn.Module() for _ in range(layers)])
    dtype=torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    for layer in root.model.layers:
        layer.self_attn=nn.Module()
        for name,shape in [('fused_qkv_a_proj_with_mqa',(8,8)),('kv_b_proj',(8,2))]:
            module=nn.Module()
            module.weight=nn.Parameter(torch.ones(shape,dtype=dtype).T.contiguous(),requires_grad=False)
            scale=torch.ones(tuple(math.ceil(x/2) for x in shape),dtype=torch.float32).T.contiguous()
            module.weight_scale_inv=nn.Parameter(scale,requires_grad=False)
            module.weight_scale_inv.format_ue8m0=False
            module.quant_method=NS(block_quant=True,quant_config=root.quant_config,use_mxfp8=False,
                                   use_marlin=False,w8a8_block_fp8_linear=hipblaslt_w8a8_block_fp8_linear)
            layer.self_attn.add_module(name,module)
    root.post_load_weights=lambda:None
    return root


def parts(adapter,name,tensor):
    return _infer_parts_factory(lambda self,n,t:[(n,t)])(adapter,name,tensor)


class TestDeepseekFp8(unittest.TestCase):
    def test_config_dispatch_prefers_quantization_over_bf16_default_dtype(self):
        engine=NS(quantization_config=config(),dtype=torch.bfloat16)
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop(fp8.TARGET_ENV,None)
            self.assertEqual(fp8.target_config(engine)['quant_method'],'fp8')
        with patch.dict(os.environ,{fp8.TARGET_ENV:'null'}):
            self.assertIsNone(fp8.target_config(engine))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'target.json';p.write_text(json.dumps(config()))
            with patch.dict(os.environ,{fp8.TARGET_ENV:str(p)}):
                self.assertEqual(fp8.target_config(NS(quantization_config=None)),config())

    def test_unsupported_formats_fail_before_transport(self):
        for bad in (dict(config(),quant_method='nvfp4'),dict(config(),fmt='e5m2'),
                    dict(config(),activation_scheme='static'),dict(config(),weight_block_size=None)):
            with self.assertRaises(NotImplementedError):fp8.validate_quant_config(bad)

    def test_export_bytes_and_scales_use_independent_block_math(self):
        name='model.layers.0.self_attn.q_b_proj.weight'
        x=torch.tensor([[0.,1.,-2.,3.,4.],[5.,6.,7.,-8.,9.],[1.,-1.,2.,-2.,0.]],dtype=torch.bfloat16)
        actual=dict(fp8.export_weight(name,x,config()))
        expected=torch.empty_like(x,dtype=torch.float8_e4m3fn)
        scales=torch.empty(2,3)
        for row in range(0,3,2):
            for col in range(0,5,2):
                block=x[row:row+2,col:col+2].float()
                scale=block.abs().max().clamp(min=1e-10)/448
                scales[row//2,col//2]=scale
                expected[row:row+2,col:col+2]=(block*scale.reciprocal()).clamp(-448,448).to(torch.float8_e4m3fn)
        self.assertEqual(actual[name].dtype,torch.uint8)
        self.assertTrue(torch.equal(actual[name],expected.view(torch.uint8)))
        self.assertTrue(torch.equal(actual[name[:-len('.weight')]+'.weight_scale_inv'],scales))

    def test_zero_blocks_have_finite_scale_and_zero_payload(self):
        p=dict(fp8.export_weight('model.layers.0.mlp.down_proj.weight',torch.zeros(3,5).bfloat16(),config()))
        self.assertTrue(torch.all(next(iter(p.values()))==0))
        self.assertTrue(torch.isfinite(list(p.values())[1]).all())
        self.assertTrue((list(p.values())[1]>0).all())

    def test_mixed_precision_exclusions_and_nonquantized_parameters(self):
        c=dict(config(),ignored_layers=['model.layers.0.self_attn.fused_qkv_a_proj_with_mqa'])
        for suffix in ('q_a_proj','kv_a_proj_with_mqa'):
            self.assertFalse(fp8.quantized_name('model.layers.0.self_attn.'+suffix+'.weight',c))
        for n in ('model.embed_tokens.weight','lm_head.weight','model.layers.0.mlp.gate.weight',
                  'model.layers.0.input_layernorm.weight'):
            self.assertFalse(fp8.quantized_name(n,config()))
        bias=torch.tensor([1.000001],dtype=torch.float32)
        name='model.layers.0.mlp.gate.e_score_correction_bias'
        self.assertIs(fp8.export_weight(name,bias,config())[0][1],bias)

    def test_native_bridge_load_dequantizes_fp8_and_preserves_fp32(self):
        name='model.layers.0.self_attn.q_b_proj.weight'
        value=torch.ones(3,5,dtype=torch.float8_e4m3fn)
        scale=torch.arange(1,7,dtype=torch.float32).reshape(2,3)*.125
        state={name:value,name[:-len('.weight')]+'.weight_scale_inv':scale,
               'bias':torch.tensor([1.000001],dtype=torch.float32)}
        fn=_load_factory(lambda self,n,s:s[n])
        owner=NS(_hf_config=NS(quantization_config=config()))
        actual=fn(owner,{'w':name,'b':'bias'},state)
        expected=scale.repeat_interleave(2,0).repeat_interleave(2,1)[:3,:5].bfloat16()
        self.assertTrue(torch.equal(actual['w'],expected))
        self.assertIs(actual['b'],state['bias'])
        with self.assertRaisesRegex(ValueError,'missing scale'):
            fn(owner,name,{name:value})
        self.assertIs(fn(NS(_hf_config=NS()),'bias',state),state['bias'])

    def test_training_plain_fp8_requires_scale_and_dequantizes_before_gather(self):
        source=nn.Module();source.weight=nn.Parameter(torch.ones(2,4,dtype=torch.float8_e4m3fn),requires_grad=False)
        source.register_buffer('weight_scale_inv',torch.tensor([[.25,.5]]))
        scales=fp8.source_scale_map([source])
        out=fp8.compute_source(source.weight,config(),scales)
        self.assertTrue(torch.equal(out,torch.tensor([[.25,.25,.5,.5]]*2).bfloat16()))
        with self.assertRaisesRegex(ValueError,'no paired scale'):
            fp8.compute_source(source.weight,config(),{})

    def test_transposed_fused_qkv_weight_and_scale_views_write_through(self):
        m=model();a=NS(_get_model=lambda:m);module=m.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa
        prefix='model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.'
        w=parts(a,prefix+'weight',module.weight)
        s=parts(a,prefix+'weight_scale_inv',module.weight_scale_inv)
        self.assertEqual([tuple(x.shape) for _,x in w],[(4,8),(4,8)])
        self.assertEqual([tuple(x.shape) for _,x in s],[(2,4),(2,4)])
        w[0][1].fill_(0);s[1][1].fill_(7)
        self.assertTrue(torch.all(module.weight.T[:4].view(torch.uint8)==0))
        self.assertTrue(torch.all(module.weight_scale_inv.T[2:]==7))
        self.assertTrue(torch.all(module.weight_scale_inv.T[:2]==1))

    def test_scale_sharding_follows_corresponding_weight(self):
        seen=[];strategy=DeepseekShardingProxy(NS(get_sharding_strategy=lambda n,**k:seen.append(n) or ('tp',1,8)))
        n='model.layers.3.mlp.experts.7.down_proj.weight_scale_inv'
        self.assertEqual(strategy.get_sharding_strategy(n),('tp',1,8))
        self.assertEqual(seen,[n[:-len('_scale_inv')]])

    def test_fnuz_commit_updates_scale_once_per_received_version(self):
        m=model(fnuz=True);a=NS(_get_model=lambda:m)
        for name,p in m.named_parameters():parts(a,name,p)
        plan=fp8.runtime_plan(a)
        for version in (1,2):
            for name,value in plan['staged'].items():
                value.fill_(version if name.endswith('_scale_inv') else 56) # FN 1.0
            with torch.no_grad():fp8.commit_staging(a)
            for module,attr,_ in plan['modules'].values():
                self.assertTrue(torch.all(getattr(module,attr).float()==.5))
                self.assertTrue(torch.all(getattr(module,attr+'_scale_inv')==2*version))

    def test_refresh_restores_runtime_storage_even_on_failure(self):
        m=model();a=NS(_get_model=lambda:m);module=m.model.layers[0].self_attn.kv_b_proj
        weight,scale=module.weight,module.weight_scale_inv
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with fp8.refresh_context(a):
                self.assertEqual(tuple(module.weight.shape),(8,2))
                self.assertEqual(tuple(module.weight_scale_inv.shape),(4,1))
                raise RuntimeError('test failure')
        self.assertIs(module.weight,weight);self.assertIs(module.weight_scale_inv,scale)

    def test_refresh_accepts_sglang_parameter_subclass_and_restores_it(self):
        from sglang.srt.layers.parameter import ModelWeightParameter
        module=nn.Module()
        module.weight=ModelWeightParameter(data=torch.ones(8,2,dtype=torch.float8_e4m3fn),
                                           input_dim=1,output_dim=0,weight_loader=lambda *a:None)
        module.weight_scale_inv=ModelWeightParameter(data=torch.ones(4,1),
                                                     input_dim=1,output_dim=0,weight_loader=lambda *a:None)
        original_weight,original_scale=module.weight,module.weight_scale_inv
        plan={'model':NS(post_load_weights=lambda:None),
              'modules':{'model.layers.0.self_attn.kv_b_proj.weight':(module,'weight',False)}}
        with patch.object(fp8,'runtime_plan',return_value=plan):
            with fp8.refresh_context(NS()):
                self.assertEqual(module.weight.data_ptr(),original_weight.data_ptr())
                self.assertEqual(module.weight_scale_inv.data_ptr(),original_scale.data_ptr())
            self.assertIs(module.weight,original_weight)
            self.assertIs(module.weight_scale_inv,original_scale)

    def test_layout_is_not_limited_to_four_layers(self):
        for count in (2,8,61):
            m=model(count);a=NS(_get_model=lambda:m)
            self.assertEqual(len(fp8.runtime_plan(a)['modules']),2*count)
            for name,p in m.named_parameters():
                for _,view in parts(a,name,p):self.assertIn(view.dtype,(torch.uint8,torch.float32))

    def test_entrypoint_auto_dispatch_and_arbitrary_layer_count(self):
        from hcu_areal.validation.deepseek_dual_train import validate_config
        with tempfile.TemporaryDirectory() as tmp:
            source,target=Path(tmp)/'train',Path(tmp)/'infer'
            source.mkdir();target.mkdir()
            for count in (2,8,61):
                for quant in (None,config()):
                    base=dict(model_type='deepseek_v3',num_hidden_layers=count,hidden_size=8,
                              vocab_size=32,torch_dtype='bfloat16')
                    (source/'config.json').write_text(json.dumps(base))
                    (target/'config.json').write_text(json.dumps(dict(base,quantization_config=quant)))
                    actor=NS(_version='v2',use_lora=False,weight_update_mode='awex',path=str(source),
                             scheduling_spec=[NS(env_vars={})])
                    rollout=NS(_version='v2',path=str(target),scheduling_spec=[NS(env_vars={})])
                    c=NS(actor=actor,rollout=rollout,cluster=NS(n_nodes=2,n_gpus_per_node=8),
                         saver=NS(freq_epochs=None,freq_steps=None,freq_secs=None),
                         recover=NS(freq_epochs=None,freq_steps=None,freq_secs=None),
                         sglang=NS(model_path=str(target),dtype='bfloat16',kv_cache_dtype='bfloat16'))
                    validate_config(c)
                    for role in (actor,rollout):
                        self.assertEqual(json.loads(role.scheduling_spec[0].env_vars[fp8.TARGET_ENV]),quant)

    def test_fnuz_refresh_passes_checkpoint_format_to_native_normalizer(self):
        m=model(fnuz=True);a=NS(_get_model=lambda:m)
        module=m.model.layers[0].self_attn.kv_b_proj
        weight,scale=module.weight,module.weight_scale_inv
        def refresh():pass
        m.post_load_weights=refresh
        with patch.dict(refresh.__globals__,{'_is_fp8_fnuz':True}):
            with fp8.refresh_context(a):
                self.assertEqual(module.weight.dtype,torch.float8_e4m3fn)
                self.assertTrue(torch.all(module.weight_scale_inv==.5))
        self.assertIs(module.weight,weight);self.assertIs(module.weight_scale_inv,scale)


if __name__=='__main__':unittest.main()
