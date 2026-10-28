"""Keep DeepSeek BF16 decode compatible with modern FlashMLA scheduler objects."""
from __future__ import annotations

import functools
from types import SimpleNamespace

from ..feature import AbstractFeature

MODULE = 'sglang.srt.layers.attention.dcu_mla_backend'


@functools.lru_cache(maxsize=1)
def native_bf16_api():
    try:
        from flash_mla.flash_mla_interface import (
            FlashMLASchedMeta, get_mla_metadata, flash_mla_with_kvcache,
        )
    except ImportError:
        return None  # Older FlashMLA keeps its original tensor metadata path.
    return SimpleNamespace(metadata_type=FlashMLASchedMeta,get_metadata=get_mla_metadata,
                           decode=flash_mla_with_kvcache)


def _init_factory(original):
    @functools.wraps(original)
    def wrapper(self, model_runner, *args, **kwargs):
        original(self,model_runner,*args,**kwargs)
        import torch
        cfg=getattr(model_runner.model_config,'hf_config',None)
        self._hcu_deepseek_bf16_flashmla=(
            getattr(cfg,'model_type',None)=='deepseek_v3'
            and self.data_type==torch.bfloat16 and native_bf16_api() is not None
        )
    return wrapper


def _decode_factory(original):
    @functools.wraps(original)
    def wrapper(self, q, k_cache, cache_seqlens, bs, layer, q_rope=None):
        if not getattr(self,'_hcu_deepseek_bf16_flashmla',False):
            return original(self,q,k_cache,cache_seqlens,bs,layer,q_rope=q_rope)
        import torch
        api=native_bf16_api()
        if q_rope is None:
            query=q.view(bs,-1,layer.tp_q_head_num,layer.head_dim)
        else:
            query=torch.cat([
                q.view(bs,-1,layer.tp_q_head_num,self.kv_lora_rank),
                q_rope.view(bs,-1,layer.tp_q_head_num,self.qk_rope_head_dim),
            ],dim=-1)
        # DCU's generic builder aliases the FP8 tensor-metadata builder even
        # for BF16 KV cache. The modern BF16 entrypoint instead needs its own
        # scheduler object. Recreate it per invocation: cache lengths change
        # between decode steps and FlashMLA forbids reusing stale metadata.
        metadata,_=api.get_metadata()
        if not isinstance(metadata,api.metadata_type):
            raise TypeError('Modern FlashMLA did not return FlashMLASchedMeta')
        output,_=api.decode(
            q=query,k_cache=k_cache.view(-1,64,1,self.kv_cache_dim),
            block_table=self.forward_metadata.block_kv_indices[:bs],
            cache_seqlens=cache_seqlens,head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=metadata,num_splits=None,
            softmax_scale=layer.scaling,causal=True,is_fp8_kvcache=False,
        )
        return output
    return wrapper


class DeepseekMlaApiFeature(AbstractFeature):
    feature_name='deepseek_mla_api'
    default_patches=True
    enable_env=''

    def enabled(self, env):
        truthy=('1','true','yes','on')
        return all(str(env.get(key,'')).strip().lower() not in truthy for key in (
            'HCU_AREAL_DISABLE_DEEPSEEK_AWEX','HCU_AREAL_DISABLE_DEEPSEEK_MLA_API'))

    def register_patches(self, manager, env):
        for method,factory,marker in (
            ('__init__',_init_factory,'deepseek.mla.bf16.init'),
            ('_call_decode',_decode_factory,'deepseek.mla.bf16.decode'),
        ):
            manager.register_patch(f'{MODULE}.DCUMLABackend.{method}',factory,
                module_name=MODULE,patch_id=marker,apply_wrapper=True)
