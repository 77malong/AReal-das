"""Opt-in compatibility features, inspired by hcu_megatron.features_manager."""
from .weight_update.qwen35_awex_feature import Qwen35AwexFeature

ADAPTOR_FEATURES = [Qwen35AwexFeature()]
