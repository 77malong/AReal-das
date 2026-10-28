"""Opt-in compatibility features, inspired by hcu_megatron.features_manager."""
from .weight_update.qwen35_awex_feature import Qwen35AwexFeature

from .weight_update.gemma3_awex_feature import Gemma3AwexFeature
from .weight_update.glm5_awex_feature import Glm5AwexFeature
from .weight_update.deepseek_awex_feature import DeepseekAwexFeature
from .weight_update.deepseek_mla_api_feature import DeepseekMlaApiFeature
from .weight_update.deepseek_fp8_source_feature import DeepseekFp8SourceFeature

ADAPTOR_FEATURES = [Qwen35AwexFeature(), Gemma3AwexFeature(), Glm5AwexFeature(), DeepseekAwexFeature(), DeepseekMlaApiFeature(), DeepseekFp8SourceFeature()]
