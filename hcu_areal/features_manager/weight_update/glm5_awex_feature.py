"""GLM-5 Megatron -> SGLang AWEX runtime feature, using the existing hooks."""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

from ..feature import AbstractFeature
from .glm5_awex_layout import is_glm5, glm5_convert_gathered, glm5_wire_tensor, assert_glm5_transfer_metadata

TRAIN_MODULE = "areal.v2.weight_update.awex.megatron_adapter"
INFER_MODULE = "areal.v2.weight_update.awex.sglang_adapter"


def _train_iter_factory(original):
    @functools.wraps(original)
    def wrapper(self, theta_by_id=None, consume_overrides=False):
        from .deepseek_awex_feature import is_deepseek, deepseek_train_iter
        if is_deepseek(self, "train"):
            yield from deepseek_train_iter(self, theta_by_id, consume_overrides)
            return
        if not is_glm5(self, "train"):
            yield from original(self, theta_by_id, consume_overrides)
            return
        from areal.engine.megatron_utils.megatron import all_gather_param, get_named_parameters

        engine = self._engine
        if getattr(engine, "quantization_config", None) or getattr(engine.config, "use_lora", False):
            raise NotImplementedError("GLM AWEX patch currently supports unquantized full-model training")
        overrides = theta_by_id if theta_by_id is not None else {}
        experts = getattr(engine.tf_config, "num_moe_experts", None)
        for name, param in get_named_parameters(engine.model, experts):
            src = overrides.get(id(param), param)
            if src is not param:
                for attr in ("tensor_model_parallel", "partition_dim", "partition_stride"):
                    if hasattr(param, attr):
                        setattr(src, attr, getattr(param, attr))
            gathered = all_gather_param(
                name, src, fp8_direct_convert=False, quantization_config=None,
                duplicated_param_names=engine._duplicated_param_names,
                gated_linear_unit=bool(engine.tf_config.gated_linear_unit),
            )
            for hf_name, tensor in glm5_convert_gathered(name, gathered):
                if getattr(engine.hf_config, "tie_word_embeddings", False) and hf_name == "lm_head.weight":
                    continue
                yield hf_name, glm5_wire_tensor(hf_name, tensor.detach())
            if consume_overrides:
                overrides.pop(id(param), None)
    return wrapper


def glm5_metadata_report(role, adapter, metadata):
    if not is_glm5(adapter, role):
        return
    trace = os.getenv("HCU_AREAL_PATCH_TRACE_DIR", "").strip()
    if not trace:
        return
    path = Path(trace)
    path.mkdir(parents=True, exist_ok=True)
    rows = [dict(name=m.name, global_shape=list(m.global_shape), dtype=str(m.dtype),
                 shards=[dict(shape=list(s.shape), global_offset=list(s.global_offset),
                              sharding_type=str(s.sharding_type), sharding_dim=s.sharding_dim,
                              num_shards=s.num_shards, global_rank=s.global_rank)
                         for s in m.shards]) for m in metadata]
    (path / f"glm5-{role}-pid{os.getpid()}.json").write_text(
        json.dumps(dict(role=role, metadata=rows), indent=2), encoding="utf-8"
    )


def _train_metadata_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        glm5_metadata_report("train", self, result)
        from .deepseek_awex_feature import deepseek_metadata_report
        deepseek_metadata_report("train", self, result)
        return result
    return wrapper


def glm5_group_preflight(adapter, args, kwargs):
    pair_name = args[0] if args else kwargs.get("pair_name")
    kv_store_url = args[5] if len(args) > 5 else kwargs.get("kv_store_url")
    if not pair_name or not kv_store_url:
        raise TypeError("GLM AWEX preflight requires pair_name and kv_store_url")
    from areal.v2.weight_update.awex import fetch_kv_metadata

    inf, trn = fetch_kv_metadata(kv_store_url, pair_name)
    count = assert_glm5_transfer_metadata(inf, trn)
    print(f"[hcu_areal pid={os.getpid()}] GLM AWEX metadata PASS: train/infer={count}",
          file=sys.stderr, flush=True)


def _train_preflight_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        from .deepseek_awex_feature import is_deepseek, deepseek_group_preflight
        if is_deepseek(self, "train"):
            deepseek_group_preflight(self, args, kwargs)
        if is_glm5(self, "train"):
            glm5_group_preflight(self, args, kwargs)
        return original(self, *args, **kwargs)
    return wrapper


def _mla_refresh(adapter):
    model = adapter._get_model()
    refresh = getattr(model, "post_load_weights", None)
    if not callable(refresh):
        raise RuntimeError("GLM AWEX requires SGLang post_load_weights for MLA cache refresh")
    if getattr(model, "quant_config", None) is not None:
        raise NotImplementedError("GLM AWEX patch currently requires unquantized inference weights")

    def run():
        import torch
        from areal.infra.platforms import current_platform
        with torch.no_grad():
            refresh()
        current_platform.synchronize()
    return run


def _refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        from .deepseek_awex_feature import is_deepseek, _refresh_factory as deepseek_factory
        if is_deepseek(self, "infer"):
            return deepseek_factory(original)(self, *args, **kwargs)
        if not is_glm5(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _mla_refresh(self)
        result = original(self, *args, **kwargs)
        # AWEX writes parameters directly, bypassing SGLang's load_weights.
        # Rebuild absorbed MLA w_kc/w_vc from the updated kv_b_proj before ACK.
        refresh()
        return result
    return wrapper


def _colocate_refresh_factory(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        from .deepseek_awex_feature import is_deepseek, _colocate_refresh_factory as deepseek_factory
        if is_deepseek(self, "infer"):
            return deepseek_factory(original)(self, *args, **kwargs)
        if not is_glm5(self, "infer"):
            return original(self, *args, **kwargs)
        refresh = _mla_refresh(self)
        client = self._colocate_http_client
        completed = False
        # The original colocate method publishes completion internally. Refresh
        # immediately before that publication, while IPC tensors remain alive,
        # rather than after the trainer has already observed the ACK.
        class CompletionClient:
            def put(proxy, url, *put_args, **put_kwargs):
                nonlocal completed
                if "/colocate_done_rank" in str(url):
                    refresh()
                    completed = True
                return client.put(url, *put_args, **put_kwargs)

            def __getattr__(proxy, name):
                return getattr(client, name)

        self._colocate_http_client = CompletionClient()
        try:
            result = original(self, *args, **kwargs)
            if not completed:
                raise RuntimeError("GLM colocate update did not publish its completion event")
            return result
        finally:
            self._colocate_http_client = client
    return wrapper


class Glm5AwexFeature(AbstractFeature):
    feature_name = "glm5_awex"
    default_patches = True
    enable_env = ""

    def enabled(self, env):
        return str(env.get("HCU_AREAL_DISABLE_GLM5_AWEX", "")).strip().lower() not in ("1", "true", "on", "yes")

    def register_patches(self, manager, env):
        entries = [
            (TRAIN_MODULE, "AwexMegatronAdapter._iter_hf_params", _train_iter_factory, "glm5.awex.train.params"),
            (TRAIN_MODULE, "AwexMegatronAdapter.get_weight_metadata", _train_metadata_factory, "glm5.awex.train.meta"),
            (TRAIN_MODULE, "AwexMegatronAdapter.init_weight_update_group", _train_preflight_factory, "glm5.awex.train.preflight"),
            (INFER_MODULE, "AwexSGLangAdapter.execute_weight_update", _refresh_factory, "glm5.awex.infer.refresh"),
            (INFER_MODULE, "AwexSGLangAdapter.execute_colocate_weight_update", _colocate_refresh_factory, "glm5.awex.infer.colocate_refresh"),
        ]
        for module, suffix, factory, patch_id in entries:
            manager.register_patch(f"{module}.{suffix}", factory, module_name=module,
                                   patch_id=patch_id, apply_wrapper=True)
        # Shared hooks are normally owned by the earlier Qwen/Gemma features.
        # Register them here too when those features are explicitly disabled,
        # so GLM can be used independently without weakening conflict checks.
        from .qwen35_awex_feature import (
            _infer_parts_factory, _infer_strategy_factory, _infer_metadata_factory,
        )
        from .gemma3_awex_feature import _group_preflight_factory
        fallbacks = [
            ("qwen35.awex.infer.unfuse", "_unfuse_params", _infer_parts_factory, "glm5.awex.infer.unfuse"),
            ("qwen35.awex.infer.sharding", "_build_sharding_strategy", _infer_strategy_factory, "glm5.awex.infer.sharding"),
            ("qwen35.awex.infer.meta", "get_weight_metadata", _infer_metadata_factory, "glm5.awex.infer.meta"),
            ("gemma3.awex.infer.preflight", "init_weight_update_group",
             lambda original: _group_preflight_factory(original, "infer"), "glm5.awex.infer.preflight"),
        ]
        for shared_id, method, factory, patch_id in fallbacks:
            if shared_id not in manager.patch_ids:
                manager.register_patch(f"{INFER_MODULE}.AwexSGLangAdapter.{method}", factory,
                                       module_name=INFER_MODULE, patch_id=patch_id, apply_wrapper=True)
