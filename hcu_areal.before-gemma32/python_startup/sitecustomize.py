"""Alternative per-interpreter entry point; .pth is recommended by default."""
import os
if os.getenv("HCU_AREAL_PATCH_QWEN35_AWEX", "0").lower() in ("1", "true", "yes", "on"):
    try:
        from hcu_areal.bootstrap import install_if_enabled
        install_if_enabled()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(86)
