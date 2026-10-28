"""Check model-guarded AWEX hooks; optionally verify Megatron GLM hooks."""
from __future__ import annotations
import argparse
import os
from . import bootstrap
from .areal_adaptor import patch_features

EXPECTED = {
    "DeepSeekV3Bridge":{"maybe_modify_loaded_hf_weight":"deepseek.fp8.train.load"},
    "DCUMLABackend": {"__init__":"deepseek.mla.bf16.init",
                      "_call_decode":"deepseek.mla.bf16.decode"},
    "AwexFSDPAdapter": {"get_weight_metadata":"qwen35.awex.train.meta",
                        "get_local_shard_parameters":"qwen35.awex.train.params",
                        "_to_hf_name":"gemma3.awex.train.names",
                        "init_weight_update_group":"gemma3.awex.train.preflight"},
    "AwexSGLangAdapter": {"_unfuse_params":"qwen35.awex.infer.unfuse",
                           "_build_sharding_strategy":"qwen35.awex.infer.sharding",
                           "get_weight_metadata":"qwen35.awex.infer.meta",
                           "init_weight_update_group":"gemma3.awex.infer.preflight",
                           "execute_weight_update":"glm5.awex.infer.refresh",
                           "execute_colocate_weight_update":"glm5.awex.infer.colocate_refresh"},
    "AwexMegatronAdapter": {"_iter_hf_params":"glm5.awex.train.params",
                            "get_weight_metadata":"glm5.awex.train.meta",
                            "init_weight_update_group":"glm5.awex.train.preflight"},
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full",action="store_true")
    ap.add_argument("--megatron", action="store_true", help="Also import and check Megatron GLM hooks")
    ap.add_argument("--flashmla", action="store_true", help="Also check DeepSeek BF16 FlashMLA compatibility hooks")
    ns=ap.parse_args()
    print("bootstrap_active:",bootstrap.is_installed())
    print("registered:",patch_features().patch_ids)
    if not bootstrap.is_installed():
        raise RuntimeError("hcu_areal startup hook inactive; run install_hook.py install")
    count = 0
    if ns.full or ns.megatron or ns.flashmla:
        from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
        classes = [AwexFSDPAdapter, AwexSGLangAdapter]
        if ns.megatron:
            from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
            from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
            classes.extend([AwexMegatronAdapter,DeepSeekV3Bridge])
        if ns.flashmla:
            from sglang.srt.layers.attention.dcu_mla_backend import DCUMLABackend
            classes.append(DCUMLABackend)
        for cls in classes:
            for method,marker in EXPECTED[cls.__name__].items():
                markers=getattr(getattr(cls,method),"__hcu_areal_patch_ids__",frozenset())
                if marker not in markers:
                    raise RuntimeError(f"Missing {marker} at {cls.__name__}.{method}")
                print("verified",cls.__name__,method,marker)
                count += 1
    print("PASS:",f"{count} model-guarded AWEX hooks verified" if count else "startup hook active")

if __name__=="__main__":
    main()
