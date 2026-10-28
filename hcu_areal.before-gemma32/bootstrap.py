"""Lazy per-interpreter installation for AReaL V2 Ray/SGLang workers.

The .pth import never imports torch, areal, awex or sglang before worker
environment setup. Patched modules are handled just after their import.
"""
from __future__ import annotations
import importlib.abc
import importlib.machinery
import os
import sys
from .areal_adaptor import patch_features

_MANAGER = None
_FINDER = None

class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, original, callback):
        self.original = original
        self.callback = callback
    def create_module(self, spec):
        creator = getattr(self.original, "create_module", None)
        return creator(spec) if creator else None
    def exec_module(self, module):
        self.original.exec_module(module)
        self.callback(module)

class _HcuAReaLFinder(importlib.abc.MetaPathFinder):
    def __init__(self, manager):
        self.manager = manager
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.manager.module_names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _AfterImportLoader(spec.loader, self.manager.apply_module)
        return spec

def install(manager=None):
    global _MANAGER, _FINDER
    if _MANAGER is not None:
        return _MANAGER
    manager = patch_features() if manager is None else manager
    if not manager.module_names:
        return manager
    finder = _HcuAReaLFinder(manager)
    sys.meta_path.insert(0, finder)
    try:
        manager.apply_patches()
    except Exception:
        sys.meta_path.remove(finder)
        manager.remove_patches()
        raise
    _MANAGER, _FINDER = manager, finder
    print(f"[hcu_areal pid={os.getpid()}] lazy hook installed: "
          + ", ".join(manager.patch_ids), file=sys.stderr, flush=True)
    return manager

def install_if_enabled():
    # Megatron-style default feature registration; wrappers guard model identity.
    # HCU_AREAL_DISABLE_QWEN35_AWEX=1 disables it when debugging other models.
    return install()

def is_installed() -> bool:
    return _MANAGER is not None and _FINDER in sys.meta_path

def uninstall():
    global _MANAGER, _FINDER
    if _MANAGER is not None:
        _MANAGER.remove_patches()
    if _FINDER is not None and _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    _MANAGER, _FINDER = None, None
