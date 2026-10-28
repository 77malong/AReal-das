"""Qwen3.5 AWEX *metadata trace only*; no tensor conversion is implemented.

This opt-in feature measures two independent AWEX adapters. It intentionally
preserves the original exception on unmatched train/infer keys. Not a fix.
"""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path
from ..feature import AbstractFeature

TRAIN_MODULE = "areal.v2.weight_update.awex.fsdp_adapter"
INFER_MODULE = "areal.v2.weight_update.awex.sglang_adapter"

def _qwen35_model(adapter, role):
    engine = getattr(adapter, "_engine", None)
    model = (getattr(engine, "model", None) if role == "train" else None)
    config = getattr(engine, "model_config", None) if role == "train" else None
    if role == "infer":
        get_model = getattr(adapter, "_get_model", None)
        model = get_model() if callable(get_model) else None
        config = getattr(model, "config", None)
    if config is None and model is not None:
        config = getattr(model, "config", None)
    values = [getattr(config, "model_type", ""), type(model).__name__]
    values.extend(getattr(config, "architectures", None) or ())
    return any("qwen3_5" in str(s).lower() or "qwen3.5" in str(s).lower()
               for s in values)

def _safe_list(value):
    if value is None:
        return None
    try:
        return list(value)
    except TypeError:
        return str(value)

def _summary(item):
    shards = getattr(item, "shards", None) or []
    return {
        "name": str(getattr(item, "name", "")),
        "global_shape": _safe_list(getattr(item, "global_shape", None)),
        "dtype": str(getattr(item, "dtype", "")),
        "shards": [
            {
                "name": str(getattr(x, "name", "")),
                "shape": _safe_list(getattr(x, "shape", None)),
                "global_offset": _safe_list(getattr(x, "global_offset", None)),
                "sharding_type": str(getattr(x, "sharding_type", "")),
                "sharding_dim": getattr(x, "sharding_dim", None),
                "num_shards": getattr(x, "num_shards", None),
                "tp_rank": getattr(x, "tp_rank", None),
                "global_rank": getattr(x, "global_rank", None),
            } for x in shards
        ],
    }

def _emit_trace(role, adapter, metadata):
    if not _qwen35_model(adapter, role):
        return
    trace_dir = os.environ.get("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    print(f"[hcu_areal pid={os.getpid()}] qwen35 {role} metadata: {len(metadata)}", flush=True)
    if not trace_dir:
        return
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
    out = path / f"qwen35-{role}-pid{os.getpid()}-rank{rank}.json"
    payload = {"role": role, "pid": os.getpid(), "rank": rank,
               "metadata": [_summary(meta) for meta in metadata]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hcu_areal] metadata written: {out}", flush=True)

def _trace_metadata(role):
    def factory(original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            _emit_trace(role, self, value)
            return value
        return wrapper
    return factory

class Qwen35AwexFeature(AbstractFeature):
    feature_name = "qwen35_awex"
    enable_env = "HCU_AREAL_PATCH_QWEN35_AWEX"

    def register_patches(self, patch_manager, env):
        patch_manager.register_patch(
            f"{TRAIN_MODULE}.AwexFSDPAdapter.get_weight_metadata",
            _trace_metadata("train"), module_name=TRAIN_MODULE,
            patch_id="qwen35.awex.trace.train", apply_wrapper=True,
        )
        patch_manager.register_patch(
            f"{INFER_MODULE}.AwexSGLangAdapter.get_weight_metadata",
            _trace_metadata("infer"), module_name=INFER_MODULE,
            patch_id="qwen35.awex.trace.infer", apply_wrapper=True,
        )
