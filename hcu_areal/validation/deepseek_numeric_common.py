"""DeepSeek fixtures with native v1 conversion and the shared native MLA loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch import nn

from .glm5_numeric_common import (
    config as glm_config, shapes as glm_shapes, values as glm_values,
    hf_axis, assert_equal, InferenceFixture as BaseInferenceFixture,
    compare_models, sglang_root,
)


def config():
    cfg = vars(glm_config()).copy()
    cfg.update(model_type='deepseek_v3', architectures=['DeepseekV3ForCausalLM'])
    cfg.pop('index_n_heads')
    cfg.pop('index_head_dim')
    return NS(**cfg)


def shapes():
    schema = json.loads((Path(__file__).parents[1]/'tests/data/deepseek_r1_4layers_shapes.json').read_text())['tensors']
    result = {n:s for n,s in glm_shapes().items() if n in schema}
    assert len(result) == 821 and set(result) == set(schema)
    return result


def values(version, device='cpu'):
    allowed = set(shapes())
    result = {n:t.to(device) for n,t in glm_values(version).items() if n in allowed}
    # The FP32 correction bias contains values that BF16 cannot represent,
    # so accidental downcasting is detectable as a numerical mismatch.
    key = 'model.layers.3.mlp.gate.e_score_correction_bias'
    result[key] += torch.arange(256, device=device).float() * 0.000001
    return result


def registry():
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    return DeepSeekV3Bridge.__new__(DeepSeekV3Bridge).mapping_registry()


def mcore_full(hf):
    mappings = {}
    reg = registry()
    for name in hf:
        mapping = reg.hf_to_megatron_lookup(name)
        assert mapping is not None, name
        mappings[mapping.megatron_param] = mapping
    full = {}
    for name,mapping in mappings.items():
        dst = mapping.hf_param
        full[name] = (torch.cat([hf[dst['gate']],hf[dst['up']]],dim=0)
                      if isinstance(dst,dict) else hf[dst].clone())
    return full,mappings


def v1_reference(hf):
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
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
    bridge = DeepSeekV3Bridge.__new__(DeepSeekV3Bridge)
    output = dict(bridge.stream_weights_megatron_to_hf(
        [model],NS(state={}),cpu=False,show_progress=False,
        conversion_tasks=tasks,merge_adapter_weights=False))
    assert_equal(output,hf)
    return {n:t.detach().clone() for n,t in output.items()}


class InferenceFixture(BaseInferenceFixture):
    fixture_config = staticmethod(config)
    fixture_shapes = staticmethod(shapes)


def provenance():
    from megatron.bridge.models.deepseek import deepseek_v3_bridge, common
    from megatron.bridge.models.conversion import model_bridge, param_mapping
    sources = [Path(m.__file__) for m in (deepseek_v3_bridge,common,model_bridge,param_mapping)]
    sources += [sglang_root(v1)/'models/deepseek_common/deepseek_weight_loader.py' for v1 in (True,False)]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
