"""Register enabled HCU features, like hcu_megatron.megatron_adaptor.

Different from Megatron entry-point imports, the actual replacement is
deferred until the AReaL AWEX adapter module has finished importing.
"""
from __future__ import annotations
import os
from .patch_utils import AReaLPatchesManager
from .features_manager import ADAPTOR_FEATURES

def patch_features(env=None) -> AReaLPatchesManager:
    env = os.environ if env is None else env
    manager = AReaLPatchesManager()
    for feature in ADAPTOR_FEATURES:
        if feature.enabled(env):
            feature.register_patches(manager, env)
    return manager

def repatch():
    """Remove and reinstall for tests; restart Ray in real training."""
    from . import bootstrap
    bootstrap.uninstall()
    return bootstrap.install_if_enabled()
