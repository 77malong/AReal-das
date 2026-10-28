"""Qwen3.5 dense FSDP DP4 -> SGLang TP4 AWEX layout adaptation.

This file is HCU-specific and does not edit AReaL/AWEX/SGLang sources.
Built from measured 4x train + 4x infer metadata. No GPU-run verified yet.
"""
from __future__ import annotations
import functools
import json
import os
import sys
from dataclasses import replace
from math import prod
from pathlib import Path
from ..feature import AbstractFeature
from .qwen35_awex_layout import (
    is_qwen35, canonical_name, segments_for_training, slice_segment,
    inference_parts, infer_strategy_override, enabled,
)

TRAIN_MODULE="areal.v2.weight_update.awex.fsdp_adapter"
INFER_MODULE="areal.v2.weight_update.awex.sglang_adapter"


def _qwen35_report(role,adapter,meta):
    if not is_qwen35(adapter,role):
        return
    trace=os.environ.get("HCU_AREAL_PATCH_TRACE_DIR","").strip()
    print(f"[hcu_areal pid={os.getpid()}] qwen35 {role}: canonical={len(meta)}",file=sys.stderr,flush=True)
    if not trace:
        return
    base=Path(trace)
    base.mkdir(parents=True,exist_ok=True)
    rank=os.environ.get("RANK",os.environ.get("LOCAL_RANK","unknown"))
    def summarized(m):
        return {"name":m.name,"global_shape":list(m.global_shape),"dtype":str(m.dtype),
                "shards":[{"name":s.name,"shape":list(s.shape),"global_offset":list(s.global_offset),
                           "sharding_type":str(s.sharding_type),"sharding_dim":s.sharding_dim,
                           "num_shards":s.num_shards,"tp_rank":s.tp_rank,"global_rank":s.global_rank}
                          for s in m.shards]}
    file=base/f"qwen35-v3-{role}-pid{os.getpid()}-rank{rank}.json"
    file.write_text(json.dumps({"role":role,"pid":os.getpid(),"rank":rank,
                                "metadata":[summarized(m) for m in meta]},indent=2),encoding="utf-8")
    print(f"[hcu_areal pid={os.getpid()}] metadata written: {file}",file=sys.stderr,flush=True)


def _train_iter(adapter):
    """Yield (name, view, global_offset, logical_global_shape, base_shard) per rank.

    Special-case slicing is performed on the *local* FSDP shard, not the
    globally gathered parameter. No hidden synchronizations or materialization.
    """
    from torch.distributed.tensor import DTensor
    engine=adapter._engine
    rank_info=adapter._build_rank_info()
    cfg=engine.model_config
    for raw_name,parameter in engine.model.named_parameters():
        name=canonical_name(raw_name)
        if adapter._tie_word_embeddings and name=="lm_head.weight":
            continue
        data=parameter.data
        if isinstance(data, DTensor):
            view=data._local_tensor
            meta=adapter._extract_dtensor_shard_meta(name,data,rank_info)
            global_shape=tuple(data.shape)
        else:
            view=data
            meta=adapter._extract_plain_shard_meta(name,data,rank_info)
            global_shape=tuple(data.shape)
        source_offset=tuple(meta.global_offset)
        if len(global_shape)==0:
            raise ValueError(f"Scalar Qwen3.5 weights are not yet supported: {name}")
        for cname,start,length in segments_for_training(name,global_shape,cfg):
            part=slice_segment(view,source_offset,global_shape,start,length)
            if part is None:
                continue
            part_view,part_offset,part_shape=part
            yield cname,part_view,part_offset,part_shape,meta


def _train_metadata_factory(original):
    @functools.wraps(original)
    def wrapper(self,*args,**kwargs):
        if not is_qwen35(self,"train"):
            return original(self,*args,**kwargs)
        if args or kwargs:
            raise TypeError("Qwen3.5 get_weight_metadata unexpected arguments")
        import torch
        from awex.meta.weight_meta import ParameterMeta,ParameterReplicaMeta
        result=[]
        for cname,view,offset,gshape,source_meta in _train_iter(self):
            dtype=torch.float32 if cname.endswith(".linear_attn.A_log") else self._wire_dtype(view)
            shard=replace(source_meta,name=cname,shape=tuple(view.shape),numel=int(view.numel()),
                          global_offset=tuple(offset),dtype=dtype)
            result.append(ParameterMeta(name=cname,global_numel=prod(gshape),
                                        global_shape=tuple(gshape),dtype=dtype,
                                        shards=[shard],replicas=[ParameterReplicaMeta(shards=[shard])]))
        _qwen35_report("train",self,result)
        return result
    return wrapper


def _train_params_factory(original):
    @functools.wraps(original)
    def wrapper(self,required_names=None):
        if not is_qwen35(self,"train"):
            return original(self,required_names)
        import torch
        required=set(required_names) if required_names else None
        result={}
        for name,tensor,offset,gshape,meta in _train_iter(self):
            if required is not None and name not in required:
                continue
            # A_log is FP32 in SGLang, unlike default FSDP BF16 export.
            result[name]=(tensor.to(torch.float32) if name.endswith(".linear_attn.A_log")
                          else self._engine._cast_to_compute_dtype(tensor))
        return result
    return wrapper


def _infer_parts_factory(original):
    @functools.wraps(original)
    def wrapper(self,name,tensor):
        if not is_qwen35(self,"infer"):
            return original(self,name,tensor)
        ctx=self._get_model_context()
        cfg=self._get_model().config
        converted=inference_parts(name,tensor,cfg,int(ctx["tp_size"]))
        if converted is not None:
            return converted
        return [(canonical_name(key),value) for key,value in original(self,name,tensor)]
    return wrapper


class _StrategyProxy:
    def __init__(self,base,tp_size):
        self._base=base
        self._tp_size=tp_size
    def get_sharding_strategy(self,name,**kwargs):
        chosen=infer_strategy_override(name,self._tp_size)
        if chosen is None:
            return self._base.get_sharding_strategy(name,**kwargs)
        from awex.sharding.param_sharding import ShardingType
        kind,axis,parts=chosen
        return (ShardingType.TP_SHARDING if kind=="tp" else ShardingType.NO_SHARDING,axis,parts)
    def __getattr__(self,k):
        return getattr(self._base,k)


def _infer_strategy_factory(original):
    @functools.wraps(original)
    def wrapper(self,rank_info):
        base=original(self,rank_info)
        if not is_qwen35(self,"infer"):
            return base
        return _StrategyProxy(base,int(rank_info.tp_size))
    return wrapper


def _infer_metadata_factory(original):
    @functools.wraps(original)
    def wrapper(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        _qwen35_report("infer",self,value)
        return value
    return wrapper


class Qwen35AwexFeature(AbstractFeature):
    feature_name="qwen35_awex"
    # Always register; wrappers fall through to originals for other models.
    default_patches=True
    enable_env=""

    def enabled(self,env):
        return not enabled(env.get("HCU_AREAL_DISABLE_QWEN35_AWEX",""))

    def register_patches(self,manager,env):
        entries=[
            (TRAIN_MODULE,"AwexFSDPAdapter.get_weight_metadata",_train_metadata_factory,"qwen35.awex.train.meta"),
            (TRAIN_MODULE,"AwexFSDPAdapter.get_local_shard_parameters",_train_params_factory,"qwen35.awex.train.params"),
            (INFER_MODULE,"AwexSGLangAdapter._unfuse_params",_infer_parts_factory,"qwen35.awex.infer.unfuse"),
            (INFER_MODULE,"AwexSGLangAdapter._build_sharding_strategy",_infer_strategy_factory,"qwen35.awex.infer.sharding"),
            (INFER_MODULE,"AwexSGLangAdapter.get_weight_metadata",_infer_metadata_factory,"qwen35.awex.infer.meta"),
        ]
        for module,suffix,replacement,patch_id in entries:
            manager.register_patch(f"{module}.{suffix}",replacement,module_name=module,
                                   patch_id=patch_id,apply_wrapper=True)
