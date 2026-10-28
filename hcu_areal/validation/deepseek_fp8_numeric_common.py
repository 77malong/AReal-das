"""Scaled DeepSeek FP8 fixtures, loaded by the unchanged native SGLang loader.

These fixtures allocate real float8 storage and FP32 block scales. The v1
oracle invokes native loading before the runtime GEMM transpose. The AWEX
receiver writes the runtime views directly. No weight checkpoint is saved.
"""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import torch
from torch import nn

from .deepseek_numeric_common import InferenceFixture as BaseInferenceFixture, values, v1_reference
from .glm5_numeric_common import hf_axis
from .test_deepseek_awex_update_parity import TrainingFixture


def quant_config(mixed=False, block=2):
    return dict(quant_method='fp8', fmt='e4m3', activation_scheme='dynamic',
                weight_block_size=[block, block], ignored_layers=(
                    ['model.layers.0.self_attn.q_b_proj', 'model.layers.2.mlp.down_proj']
                    if mixed else []))


def native_config(config):
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    return Fp8Config.from_config(config)


def hipblaslt_w8a8_block_fp8_linear(*args, **kwargs):
    raise AssertionError('Fixture GEMM is tested separately with the actual native kernel')


def effective_values(version, plain_fp8_source=False):
    result = values(version)
    if plain_fp8_source:
        name='model.layers.0.self_attn.q_a_proj.weight'
        scale=.125 if version==1 else .0625
        result[name]=((result[name].float()/scale).to(torch.float8_e4m3fn).float()*scale).bfloat16()
    return result


def reference_hf(version, config, device, plain_fp8_source=False):
    """Native v1 Bridge, then the native FP8 quantizer, outside the new patch."""
    from areal.engine.megatron_utils.fp8.quantize import _quantize_param
    # Native SGLang exclusion semantics are checked separately with real modules.
    ignored=set(config.get('ignored_layers', []))
    result={}
    for name,tensor in v1_reference(effective_values(version,plain_fp8_source)).items():
        tensor=tensor.to(device)
        eligible=(name.endswith('.weight') and tensor.ndim==2
                  and name.startswith('model.layers.') and '.mlp.gate.' not in name)
        if eligible and name[:-len('.weight')] not in ignored:
            if tensor.device.type=='cpu':
                from hcu_areal.features_manager.weight_update.deepseek_fp8 import export_weight
                pair=dict(export_weight(name,tensor,config))
                pair[name]=pair[name].view(torch.float8_e4m3fn)
                result.update(pair)
            else:
                result.update(_quantize_param(name,tensor,config['weight_block_size'],scale_dtype=torch.float32))
        else:
            result[name]=tensor
    return result


class Fp8TrainingFixture(TrainingFixture):
    def __init__(self,rank,device,plain_fp8_source=False):
        self.plain_fp8_source=plain_fp8_source
        super().__init__(rank,device)

    def update(self,version,initialize=False):
        name='module.decoder.layers.0.self_attention.linear_q_down_proj.weight'
        if self.plain_fp8_source and not initialize:
            for key,param in self._params:
                if key==name:
                    param.data=torch.empty(param.shape,device=self.device,dtype=torch.bfloat16)
        super().update(version,initialize)
        if self.plain_fp8_source:
            scale=torch.tensor(.125 if version==1 else .0625,device=self.device,dtype=torch.float32)
            param=dict(self._params)[name]
            param.data=(param.data.float()/scale).to(torch.float8_e4m3fn)
            key=name+'_scale_inv'
            if initialize:self._buffers.append((key,scale))
            else:dict(self._buffers)[key].copy_(scale)


class Fp8InferenceFixture(BaseInferenceFixture):
    def __init__(self,rank,device='cpu',fused_shared=False,mixed=False,v1=False):
        super().__init__(rank,device,fused_shared,v1)
        config=quant_config(mixed)
        self.quant_config=native_config(config)
        # Execute actual native FP8 MLA cache reconstruction, with the DCU flags.
        from sglang.srt.layers.quantization import fp8_utils
        env=self._native['post_load_weights'].__globals__
        env.update(_is_cuda=False,_is_dcu=True,_is_xpu=False,_is_fp8_fnuz=False,
                   block_quant_to_tensor_quant=fp8_utils.block_quant_to_tensor_quant,
                   normalize_e4m3fn_to_e4m3fnuz=fp8_utils.normalize_e4m3fn_to_e4m3fnuz,
                   should_deepgemm_weight_requant_ue8m0=lambda **kwargs:False)
        self._fp8_modules=[]
        for prefix,module in list(self.named_modules()):
            weight=getattr(module,'weight',None)
            eligible=(weight is not None and weight.ndim==2 and prefix.startswith('model.layers.')
                      and '.mlp.gate' not in prefix and 'norm' not in prefix)
            # gate_up_proj is a quantized fused MLP, not the BF16 routing gate.
            if weight is not None and prefix.endswith('gate_up_proj'):eligible=True
            if eligible and prefix not in config['ignored_layers']:
                weight.data=torch.zeros_like(weight,dtype=torch.float8_e4m3fn)
                scale=nn.Parameter(torch.ones(tuple(math.ceil(x/2) for x in weight.shape),
                                             dtype=torch.float32,device=device),requires_grad=False)
                scale.format_ue8m0=False
                if prefix.endswith('gate_up_proj'):
                    size=scale.shape[0]//2
                    def load(p,w,shard_id,size=size):
                        p.data[shard_id*size:(shard_id+1)*size].copy_(w.chunk(8,dim=0)[rank])
                else:
                    axis=None if prefix.endswith('fused_qkv_a_proj_with_mqa') else hf_axis(prefix+'.weight')
                    def load(p,w,axis=axis):
                        p.data.copy_(w if axis is None else w.chunk(8,dim=axis)[rank])
                scale.weight_loader=load
                module.register_parameter('weight_scale_inv',scale)
                module.quant_method=NS(block_quant=True,quant_config=self.quant_config,
                    w8a8_block_fp8_linear=hipblaslt_w8a8_block_fp8_linear,
                    use_mxfp8=False,use_marlin=False)
                self._fp8_modules.append(module)
        experts=self.model.layers[3].mlp.experts
        experts.quant_method=NS(block_quant=True,quant_config=self.quant_config,use_mxfp8=False,use_marlin=False)
        for attr in ('w13_weight','w2_weight'):
            p=getattr(experts,attr)
            p.data=torch.zeros_like(p,dtype=torch.float8_e4m3fn)
            shape=(*p.shape[:-2],math.ceil(p.shape[-2]/2),math.ceil(p.shape[-1]/2))
            scale=nn.Parameter(torch.ones(shape,dtype=torch.float32,device=device),requires_grad=False)
            scale.format_ue8m0=False
            if attr=='w13_weight':
                def load(p,w,name,shard_id,expert_id):
                    offset=0 if shard_id=='w1' else 1
                    p.data[expert_id,offset:offset+1].copy_(w.chunk(8,dim=0)[rank])
            else:
                def load(p,w,name,shard_id,expert_id):p.data[expert_id].copy_(w.chunk(8,dim=1)[rank])
            scale.weight_loader=load
            experts.register_parameter(attr+'_scale_inv',scale)
        self._runtime=False
        self.to_runtime()

    def to_runtime(self):
        if not self._runtime:
            for module in self._fp8_modules:
                module.weight.data=module.weight.data.T.contiguous()
                module.weight_scale_inv.data=module.weight_scale_inv.data.T.contiguous()
            self._runtime=True

    def load_v1(self,weights):
        # Native checkpoint loading operates before GEMM transposition.
        if self._runtime:
            for module in self._fp8_modules:
                module.weight.data=module.weight.data.T
                module.weight_scale_inv.data=module.weight_scale_inv.data.T
            self._runtime=False
        try:
            super().load_v1(weights)
        finally:
            self.to_runtime()


def compare_fp8_models(actual,reference):
    a,b=dict(actual.named_parameters()),dict(reference.named_parameters())
    assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
    for name,x in a.items():
        y=b[name]
        assert x.shape==y.shape and x.dtype==y.dtype,(name,x.shape,y.shape,x.dtype,y.dtype)
        if x.dtype in (torch.float8_e4m3fn,torch.float8_e4m3fnuz):
            x,y=x.view(torch.uint8),y.view(torch.uint8)
        assert torch.equal(x,y),(name,'raw storage mismatch')
    for x,y in zip(actual.model.layers,reference.model.layers):
        for key in ('w_kc','w_vc'):
            assert torch.equal(getattr(x.self_attn,key),getattr(y.self_attn,key)),('stale MLA cache',key)
        q=torch.arange(4,device=x.self_attn.w_kc.device).float().view(1,4)/4
        def probe(a):
            z=torch.einsum('hd,hdk->hk',q,a.w_kc.float())
            return torch.einsum('hk,hkv->hv',z,a.w_vc.float())
        assert torch.equal(probe(x.self_attn),probe(y.self_attn)),'MLA probe mismatch'
