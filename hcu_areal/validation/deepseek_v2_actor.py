"""Isolated two-node v2 actor; no training-weight export."""
from areal.engine.megatron_engine import MegatronPPOActor
from areal.trainer.rl_trainer import PPOTrainer


class DeepseekV2Actor(MegatronPPOActor):
    def __init__(self, config):
        from megatron.core.transformer.moe import moe_utils
        from hcu_areal.features_manager.weight_update.deepseek_awex_feature import DeepseekAwexFeature
        from hcu_areal import bootstrap
        from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
        import os
        markers = getattr(AwexMegatronAdapter._iter_hf_params,'__hcu_areal_patch_ids__',())
        if (not bootstrap.is_installed() or not DeepseekAwexFeature().enabled(os.environ)
                or not set(markers).intersection(('glm5.awex.train.params','deepseek.awex.train.params'))):
            raise RuntimeError('DeepSeek AWEX runtime patch is disabled')
        # Same FP32 router workaround as the successful GLM run. This belongs
        # to the validation harness, not the AWEX conversion feature.
        moe_utils.te_general_gemm = None
        super().__init__(config)

    def _build_hf_mcore_bridge(self):
        bridge = super()._build_hf_mcore_bridge()
        if self.bridge_cls != 'megatron-bridge' or bridge is None:
            raise ValueError('The DeepSeek run requires native Megatron-Bridge')
        cfg = getattr(bridge.hf_pretrained,'config',bridge.hf_pretrained)
        if cfg.model_type != 'deepseek_v3' or cfg.num_hidden_layers <= 0:
            raise ValueError('Expected a DeepSeek-V3/R1 model with a positive layer count')
        return bridge

    def _save_model_to_hf(self, *args, **kwargs):
        raise RuntimeError('Training-weight export is disabled for this test')


class DeepseekV2Trainer(PPOTrainer):
    def _create_train_engine(self, actor_config, alloc):
        if alloc.backend != 'megatron' or actor_config._version != 'v2':
            raise ValueError('This test requires Megatron with controller v2')
        actor = DeepseekV2Actor.as_controller(actor_config,self.scheduler)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def close(self):
        actor = getattr(self,'actor',None)
        update = getattr(actor,'_weight_update_ctrl',None)
        try:
            if update is not None:
                update.destroy()
                actor._weight_update_ctrl = None
        finally:
            super().close()
