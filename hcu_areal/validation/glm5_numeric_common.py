"""Small numeric GLM fixtures with the real four-layer/256-expert key schema.

Reference conversion executes the native v1 Bridge stream, not the new patch.
Reference inference loading executes the native SGLang loader source. Only the
model allocations, rank context and parameter loader callbacks are fixtures.
No checkpoint tensors are read or written.
"""
from __future__ import annotations

import ast
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace as NS
from typing import Iterable, Optional

import torch
from torch import nn

HERE = Path(__file__).resolve()
SCHEMA = HERE.parents[1] / 'tests/data/glm5_4layers_shapes.json'


def config():
    return NS(model_type='glm_moe_dsa', architectures=['GlmMoeDsaForCausalLM'],
              num_hidden_layers=4, hidden_size=16, vocab_size=64,
              num_attention_heads=8, q_lora_rank=8, kv_lora_rank=8,
              qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
              index_n_heads=8, index_head_dim=8, n_routed_experts=256,
              n_shared_experts=1, first_k_dense_replace=3,
              intermediate_size=32, moe_intermediate_size=16,
              num_nextn_predict_layers=0, tie_word_embeddings=False)


def shapes():
    c = config()
    names = json.loads(SCHEMA.read_text())['tensors']
    result = {}
    for name in names:
        if name in ('model.embed_tokens.weight', 'lm_head.weight'):
            shape = (c.vocab_size, c.hidden_size)
        elif name == 'model.norm.weight' or name.endswith(('.input_layernorm.weight', '.post_attention_layernorm.weight')):
            shape = (c.hidden_size,)
        else:
            suffix = name.split('.self_attn.')[-1] if '.self_attn.' in name else ''
            attn = {
                'q_a_proj.weight': (8,16), 'q_a_layernorm.weight':(8,),
                'q_b_proj.weight':(64,8), 'kv_a_proj_with_mqa.weight':(12,16),
                'kv_a_layernorm.weight':(8,), 'kv_b_proj.weight':(96,8),
                'o_proj.weight':(16,64), 'indexer.wq_b.weight':(64,8),
                'indexer.wk.weight':(8,16), 'indexer.k_norm.weight':(8,),
                'indexer.k_norm.bias':(8,), 'indexer.weights_proj.weight':(8,16),
            }
            if suffix in attn:
                shape = attn[suffix]
            elif name.endswith('.gate.e_score_correction_bias'):
                shape = (256,)
            elif name.endswith('.mlp.gate.weight'):
                shape = (256,16)
            else:
                layer = int(name.split('.')[2])
                width = 32 if layer < 3 else 16
                shape = (16,width) if name.endswith('.down_proj.weight') else (width,16)
        result[name] = (shape, torch.float32 if name.endswith('e_score_correction_bias') else torch.bfloat16)
    assert len(result) == 841
    return result


def values(version, device='cpu'):
    result = {}
    for name,(shape,dtype) in shapes().items():
        seed = int.from_bytes(hashlib.sha256((name+':'+str(version)).encode()).digest()[:8], 'little')
        generator = torch.Generator().manual_seed(seed)
        # Exactly representable in BF16; signs and positions differ per name.
        result[name] = (torch.randint(-128,128,shape,generator=generator).float()/16).to(device=device,dtype=dtype)
    return result


def hf_axis(name):
    """Independent fixture layout, based on v1 module storage semantics."""
    if ('norm' in name or '.indexer.' in name or '.mlp.gate.' in name
            or name.endswith(('.q_a_proj.weight','.kv_a_proj_with_mqa.weight'))):
        return None
    return 1 if name.endswith(('.o_proj.weight','.down_proj.weight')) else 0


def registry():
    from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
    return GLM5Bridge.__new__(GLM5Bridge).mapping_registry()


def mcore_full(hf):
    reg = registry()
    mappings = {}
    for name in hf:
        mapping = reg.hf_to_megatron_lookup(name)
        assert mapping is not None, name
        mappings[mapping.megatron_param] = mapping
    result = {}
    for name,mapping in mappings.items():
        dst = mapping.hf_param
        result[name] = (torch.cat([hf[dst['gate']],hf[dst['up']]],dim=0)
                        if isinstance(dst,dict) else hf[dst].clone())
    return result,mappings


def v1_reference(hf):
    """Actual stream called underneath v1 AutoBridge.export_hf_weights.

    Pre-built conversion tasks bypass the need to allocate a full GPTModel;
    all tensor transformations and expert handling are the original Bridge.
    """
    from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    full,mappings = mcore_full(hf)
    cfg = NS(share_embeddings_and_output_weights=False,num_moe_experts=256,mtp_num_layers=None)
    model = nn.Module()
    model.config = cfg
    tasks = []
    for name,tensor in full.items():
        owner = nn.Module()
        owner.config = cfg
        owner.tensor_model_parallel = False
        tasks.append(WeightConversionTask(param_name=name,global_param_name=name,
                     mapping=mappings[name],megatron_module=owner,param_weight=tensor))
    bridge = GLM5Bridge.__new__(GLM5Bridge)
    output = dict(bridge.stream_weights_megatron_to_hf(
        [model],NS(state={}),cpu=False,show_progress=False,
        conversion_tasks=tasks,merge_adapter_weights=False))
    assert_equal(output,hf)
    return {k:v.detach().clone() for k,v in output.items()}


def runtime_expected(hf):
    # Independent of glm5_wire_tensor: native non-AITER indexer stores FP32.
    return {k:v.float() if '.indexer.k_norm.' in k else v for k,v in hf.items()}


def assert_equal(actual, expected):
    assert set(actual) == set(expected), (sorted(set(expected)-set(actual))[:5], sorted(set(actual)-set(expected))[:5])
    for name,ref in expected.items():
        got = actual[name]
        assert got.shape == ref.shape, (name,got.shape,ref.shape)
        assert got.dtype == ref.dtype, (name,got.dtype,ref.dtype)
        assert torch.equal(got,ref), (name,'value mismatch',float((got.float()-ref.float()).abs().max()))


def source_function(path,name,env):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return env[name]


def sglang_root(v1=False):
    if v1:
        root = Path(os.getenv('GLM_V1_SGLANG_ROOT','/workspace/sglang'))
    else:
        root = Path(os.environ['SGLANG_ROOT'])
    return root/'python/sglang/srt'


def native_sglang_methods(v1=False):
    """Execute unchanged loader/post-load source without importing GPU kernels."""
    root = sglang_root(v1)
    loader = root/'models/deepseek_common/deepseek_weight_loader.py'
    env = dict(globals(),_is_npu=False,_is_dcu=False,_is_hip=False,
               _is_musa=False,_use_aiter_gfx95=False,_is_fp8_fnuz=False,
               logger=logging.getLogger('glm5_parity'),RUNAI_STREAMER_TENSOR_ATTR='_runai_streamer',
               should_async_load=lambda t:False,
               maybe_executor_submit=lambda **k:k['func'](*k.get('func_args',()),**k.get('func_kwargs',{})),
               get_layer_id=lambda n:int(n.split('.')[2]) if n.startswith('model.layers.') else None,
               log_info_on_rank0=lambda *a,**k:None)
    tree=ast.parse(loader.read_text())
    classes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('NextNEnabledConfig','NextNDisabledConfig')]
    env['dataclass']=dataclasses.dataclass
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+classes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(loader),'exec'),env)
    source_function(root/'utils/common.py','bind_or_assign',env)
    source_function(root/'model_loader/weight_utils.py','default_weight_loader',env)
    mapping=source_function(root/'layers/moe/fused_moe_triton/layer.py','make_expert_params_mapping',env)
    env['FusedMoE']=NS(make_expert_params_mapping=lambda **k:mapping(None,**k))
    source_function(loader,'_clone_if_runai_streamed_tensor',env)
    methods={n:source_function(loader,n,env) for n in ('do_load_weights','_initialize_nextn_conf','post_load_weights')}
    return methods


def set_parameter(module,name,tensor):
    parts=name.split('.')
    for part in parts[:-1]:
        if part.isdigit():
            module=module[int(part)]
        else:
            if not hasattr(module,part):module.add_module(part,nn.Module())
            module=getattr(module,part)
    p=nn.Parameter(tensor,requires_grad=False)
    module.register_parameter(parts[-1],p)
    return p


class InferenceFixture(nn.Module):
    """Fused SGLang storage with native v1 load and native post-load methods."""
    fixture_config = staticmethod(config)
    fixture_shapes = staticmethod(shapes)

    def __init__(self,tp_rank,device='cpu',fused_shared=False,v1=False):
        super().__init__()
        self.config=self.fixture_config()
        self.quant_config=None
        self.tp_rank=tp_rank
        self.num_fused_shared_experts=int(fused_shared)
        self.pp_group=NS(is_first_rank=True,is_last_rank=True)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Module() for _ in range(4)])
        self.model.start_layer,self.model.end_layer=0,4
        self._native=native_sglang_methods(v1)
        self.refreshes=0
        schema=self.fixture_shapes()
        # Most parameters keep canonical names. Build native fused blocks below.
        for name,(shape,dtype) in schema.items():
            if name.endswith(('.q_a_proj.weight','.kv_a_proj_with_mqa.weight')):continue
            if '.mlp.experts.' in name:continue
            if fused_shared and '.mlp.shared_experts.' in name:continue
            if name.endswith(('.gate_proj.weight','.up_proj.weight')):continue
            if '.indexer.k_norm.' in name:dtype=torch.float32
            axis=hf_axis(name)
            local=list(shape)
            if axis is not None:local[axis]//=8
            p=set_parameter(self,name,torch.full(local,float('nan'),dtype=dtype,device=device))
            def load(param,weight,axis=axis):
                if axis is not None:weight=weight.chunk(8,dim=axis)[self.tp_rank]
                param.data.copy_(weight)
            p.weight_loader=load
        for layer in range(4):
            base=f'model.layers.{layer}.self_attn'
            p=set_parameter(self,base+'.fused_qkv_a_proj_with_mqa.weight',torch.full((20,16),float('nan'),dtype=torch.bfloat16,device=device))
            p.weight_loader=lambda param,w:param.data.copy_(w)
            attention=self.model.layers[layer].self_attn
            attention.qk_nope_head_dim,attention.v_head_dim=4,8
            attention.w_kc=attention.w_vc=attention.w_scale=None
            if layer<3:self._add_gated(f'model.layers.{layer}.mlp',32,device)
        if not fused_shared:self._add_gated('model.layers.3.mlp.shared_experts',16,device)
        count=256+int(fused_shared)
        for kind,shape in [('w13_weight',(count,4,16)),('w2_weight',(count,16,2))]:
            p=set_parameter(self,'model.layers.3.mlp.experts.'+kind,torch.full(shape,float('nan'),dtype=torch.bfloat16,device=device))
            def load_expert(param,w,name,shard_id,expert_id):
                if shard_id=='w2':param.data[expert_id].copy_(w.chunk(8,dim=1)[self.tp_rank])
                else:
                    offset=0 if shard_id=='w1' else 2
                    param.data[expert_id,offset:offset+2].copy_(w.chunk(8,dim=0)[self.tp_rank])
            p.weight_loader=load_expert

    def _add_gated(self,prefix,width,device):
        p=set_parameter(self,prefix+'.gate_up_proj.weight',torch.full((2*width//8,16),float('nan'),dtype=torch.bfloat16,device=device))
        def load(param,w,shard_id):
            length=width//8
            param.data[shard_id*length:(shard_id+1)*length].copy_(w.chunk(8,dim=0)[self.tp_rank])
        p.weight_loader=load

    def _initialize_nextn_conf(self,is_nextn):
        return self._native['_initialize_nextn_conf'](self,is_nextn)

    def load_v1(self,weights):
        with torch.no_grad():self._native['do_load_weights'](self,weights.items())

    def post_load_weights(self,*args,**kwargs):
        self._native['post_load_weights'](self,*args,**kwargs)
        self.refreshes+=1


def compare_models(actual,reference):
    assert_equal(dict(actual.named_parameters()),dict(reference.named_parameters()))
    for a,b in zip(actual.model.layers,reference.model.layers):
        for key in ('w_kc','w_vc'):
            x,y=getattr(a.self_attn,key),getattr(b.self_attn,key)
            assert torch.equal(x,y),('MLA stale/incorrect',key)
        # Independent small MLA matmul probe, not a whole-model logits claim.
        q=torch.arange(4,device=a.self_attn.w_kc.device).float().view(1,4)/4
        def probe(attn):
            latent=torch.einsum('hd,hdk->hk',q,attn.w_kc.float())
            return torch.einsum('hk,hkv->hv',latent,attn.w_vc.float())
        assert torch.equal(probe(a.self_attn),probe(b.self_attn)), 'MLA matmul probe mismatch'


def provenance():
    from megatron.bridge.models.glm_moe_dsa import glm5_bridge
    from megatron.bridge.models.conversion import model_bridge, param_mapping
    sources=[Path(glm5_bridge.__file__),Path(model_bridge.__file__),Path(param_mapping.__file__),
             sglang_root(True)/'models/deepseek_common/deepseek_weight_loader.py',
             sglang_root(False)/'models/deepseek_common/deepseek_weight_loader.py']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
