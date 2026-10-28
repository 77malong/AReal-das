"""Common feature interface (Megatron-LM-das AbstractFeature pattern)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping

class AbstractFeature(ABC):
    feature_name: str = "unnamed"
    enable_env: str = ""
    optimization_level: int = 2
    default_patches: bool = False

    def enabled(self, env: Mapping[str, str]) -> bool:
        if not self.enable_env:
            return self.default_patches
        raw = str(env.get(self.enable_env, "")).lower().strip()
        return raw in ("1", "true", "yes", "on")

    @abstractmethod
    def register_patches(self, patch_manager, env: Mapping[str, str]):
        """Register feature-specific methods without importing heavy dependencies."""
