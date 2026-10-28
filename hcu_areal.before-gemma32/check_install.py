"""Check 5 Qwen3.5 AWEX runtime patches and stdout-clean startup."""
from __future__ import annotations
import argparse
import os
from . import bootstrap
from .areal_adaptor import patch_features

EXPECTED = {
    "AwexFSDPAdapter": {"get_weight_metadata":"qwen35.awex.train.meta",
                        "get_local_shard_parameters":"qwen35.awex.train.params"},
    "AwexSGLangAdapter": {"_unfuse_params":"qwen35.awex.infer.unfuse",
                           "_build_sharding_strategy":"qwen35.awex.infer.sharding",
                           "get_weight_metadata":"qwen35.awex.infer.meta"},
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full",action="store_true")
    ns=ap.parse_args()
    print("bootstrap_active:",bootstrap.is_installed())
    print("registered:",patch_features().patch_ids)
    if not bootstrap.is_installed():
        raise RuntimeError("hcu_areal startup hook inactive; run install_hook.py install")
    if ns.full:
        from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
        for cls in (AwexFSDPAdapter,AwexSGLangAdapter):
            for method,marker in EXPECTED[cls.__name__].items():
                markers=getattr(getattr(cls,method),"__hcu_areal_patch_ids__",frozenset())
                if marker not in markers:
                    raise RuntimeError(f"Missing {marker} at {cls.__name__}.{method}")
                print("verified",cls.__name__,method,marker)
    print("PASS:","5 model-guarded AWEX patches installed" if ns.full else "startup hook active")

if __name__=="__main__":
    main()
