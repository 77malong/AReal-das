"""Load checkpoint FP8 + scales as effective compute values for native Bridge."""
from __future__ import annotations

import functools

from ..feature import AbstractFeature
from .deepseek_fp8 import validate_quant_config, dequantize_blocks, float8_dtypes

MODULE='megatron.bridge.models.deepseek.deepseek_v3_bridge'


def _load_factory(original):
    @functools.wraps(original)
    def wrapper(self,hf_param,hf_state_dict):
        cfg=getattr(self,'_hf_config',None)
        raw=getattr(cfg,'quantization_config',None)
        if raw is None:
            result=original(self,hf_param,hf_state_dict)
            values=result.values() if isinstance(result,dict) else (result,)
            if any(value.dtype in float8_dtypes() for value in values):
                raise ValueError('FP8 checkpoint weights require quantization_config with scale block geometry')
            return result
        config=validate_quant_config(raw)
        def load(name):
            value=hf_state_dict[name]
            if value.dtype not in float8_dtypes():
                return value
            if not name.endswith('.weight'):
                raise ValueError(f'Unexpected checkpoint FP8 parameter: {name}')
            scale_name=name[:-len('.weight')]+'.weight_scale_inv'
            if scale_name not in hf_state_dict:
                raise ValueError(f'FP8 checkpoint is missing scale: {scale_name}')
            return dequantize_blocks(value,hf_state_dict[scale_name],config['weight_block_size'])
        return load(hf_param) if isinstance(hf_param,str) else {key:load(name) for key,name in hf_param.items()}
    return wrapper


class DeepseekFp8SourceFeature(AbstractFeature):
    feature_name='deepseek_fp8_source'
    default_patches=True
    enable_env=''

    def enabled(self,env):
        return str(env.get('HCU_AREAL_DISABLE_DEEPSEEK_AWEX','')).strip().lower() not in ('1','true','yes','on')

    def register_patches(self,manager,env):
        manager.register_patch(MODULE+'.DeepSeekV3Bridge.maybe_modify_loaded_hf_weight',
            _load_factory,module_name=MODULE,patch_id='deepseek.fp8.train.load',apply_wrapper=True)
