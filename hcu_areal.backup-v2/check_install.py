"""Verify live startup installation; --full checks both AReaL adapter methods."""
from __future__ import annotations
import argparse
import os
from . import bootstrap
from .areal_adaptor import patch_features

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="import AWEX adapters and verify wrappers")
    ns = ap.parse_args()
    print("feature_enabled:", os.getenv("HCU_AREAL_PATCH_QWEN35_AWEX"))
    print("bootstrap_active:", bootstrap.is_installed())
    registered = patch_features()
    print("registered:", registered.patch_ids)
    if not bootstrap.is_installed():
        raise RuntimeError("Patch bootstrap not active in this Python process. "
                           "Install the .pth hook and start a fresh interpreter with the feature flag.")
    if ns.full:
        from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
        for cls in (AwexFSDPAdapter, AwexSGLangAdapter):
            markers = getattr(cls.get_weight_metadata,
                              "__hcu_areal_patch_ids__", frozenset())
            print(cls.__name__, sorted(markers))
            if not any(m.startswith("qwen35.awex.trace.") for m in markers):
                raise RuntimeError("Missing runtime patch in adapter: " + cls.__name__)
    print("PASS:", "both adapters patched" if ns.full else "startup hook active")

if __name__ == "__main__":
    main()
