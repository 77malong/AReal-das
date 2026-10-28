"""Gemma3 AWEX Feature (names + runtime preflight), isolated from Qwen3.5.

SGLang inference's two already-registered Qwen3.5 wrappers dispatch to this
module's *guarded* Vision QKV/TP geometry handlers. This avoids conflicting
registrations in the current AReaLPatchesManager while preserving the
previously validated Qwen3.5 code path.
"""
from __future__ import annotations

import functools
import os
import sys

from ..feature import AbstractFeature
from .gemma3_awex_layout import (
    is_gemma3,
    gemma3_train_to_infer_name,
    assert_gemma3_transfer_metadata,
)

_TRAIN_MODULE = "areal.v2.weight_update.awex.fsdp_adapter"
_INFER_MODULE = "areal.v2.weight_update.awex.sglang_adapter"


def _train_name_factory(original):
    @functools.wraps(original)
    def wrapper(self, name):
        converted = original(self, name)
        if not is_gemma3(self, "train"):
            return converted
        return gemma3_train_to_infer_name(converted)
    return wrapper


def _group_preflight_factory(original, role="train"):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        from .deepseek_awex_layout import is_deepseek
        if is_deepseek(self, role):
            from .deepseek_awex_feature import deepseek_group_preflight
            deepseek_group_preflight(self, args, kwargs)
            return original(self, *args, **kwargs)
        from .glm5_awex_layout import is_glm5
        if is_glm5(self, role):
            from .glm5_awex_feature import glm5_group_preflight
            glm5_group_preflight(self, args, kwargs)
            return original(self, *args, **kwargs)
        if not is_gemma3(self, role):
            return original(self, *args, **kwargs)
        # Both names are explicit, first two positional parameters in the
        # original AwexFSDPAdapter.init_weight_update_group signature.
        if args:
            pair_name = args[0]
        else:
            pair_name = kwargs.get("pair_name")
        if len(args) > 5:
            kv_store_url = args[5]
        else:
            kv_store_url = kwargs.get("kv_store_url")
        if not pair_name or not kv_store_url:
            raise TypeError("Gemma3 AWEX preflight requires pair_name, kv_store_url")

        from areal.v2.weight_update.awex import fetch_kv_metadata
        inf, trn = fetch_kv_metadata(kv_store_url, pair_name)
        count = assert_gemma3_transfer_metadata(inf, trn)
        print(
            f"[hcu_areal pid={os.getpid()}] gemma3 AWEX {role} metadata PASS: "
            f"train/infer={count}, global shapes and dtypes match",
            file=sys.stderr, flush=True,
        )
        return original(self, *args, **kwargs)
    return wrapper


class Gemma3AwexFeature(AbstractFeature):
    feature_name = "gemma3_awex"
    default_patches = True
    enable_env = ""

    def enabled(self, env):
        return str(env.get("HCU_AREAL_DISABLE_GEMMA3_AWEX", "")).lower() not in (
            "1", "true", "on", "yes"
        )

    def register_patches(self, manager, env):
        manager.register_patch(
            f"{_TRAIN_MODULE}.AwexFSDPAdapter._to_hf_name",
            _train_name_factory,
            module_name=_TRAIN_MODULE,
            patch_id="gemma3.awex.train.names",
            apply_wrapper=True,
        )
        manager.register_patch(
            f"{_TRAIN_MODULE}.AwexFSDPAdapter.init_weight_update_group",
            lambda original: _group_preflight_factory(original, "train"),
            module_name=_TRAIN_MODULE,
            patch_id="gemma3.awex.train.preflight",
            apply_wrapper=True,
        )
        manager.register_patch(
            f"{_INFER_MODULE}.AwexSGLangAdapter.init_weight_update_group",
            lambda original: _group_preflight_factory(original, "infer"),
            module_name=_INFER_MODULE,
            patch_id="gemma3.awex.infer.preflight",
            apply_wrapper=True,
        )
