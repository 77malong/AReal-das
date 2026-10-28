"""DeepSeek two-node v2 entrypoint. --check-only validates without starting Ray."""
import json
import os
from pathlib import Path
import sys


def validate_config(config):
    if config.actor._version != 'v2' or config.rollout._version != 'v2':
        raise ValueError('Only v2 controllers are permitted')
    if config.actor.use_lora or config.actor.weight_update_mode != 'awex':
        raise ValueError('Only full-model AWEX updates are permitted')
    if config.cluster.n_nodes != 2 or config.cluster.n_gpus_per_node != 8:
        raise ValueError('This profile requires exactly two nodes with eight GPUs each')
    for section in (config.saver,config.recover):
        for frequency in ('freq_epochs','freq_steps','freq_secs'):
            if getattr(section,frequency) is not None:
                raise ValueError('All checkpoint frequencies must be null')
    cfg = json.loads((Path(config.actor.path)/'config.json').read_text())
    if cfg['model_type'] != 'deepseek_v3' or cfg.get('num_hidden_layers',0) <= 0:
        raise ValueError('Expected a DeepSeek-V3/R1 model with a positive layer count')
    from hcu_areal.features_manager.weight_update.deepseek_fp8 import TARGET_ENV,validate_quant_config
    validate_quant_config(cfg.get('quantization_config'))
    target_path=Path(config.sglang.model_path or config.rollout.path)
    target_cfg=json.loads((target_path/'config.json').read_text())
    if target_cfg['model_type']!='deepseek_v3' or any(
            target_cfg.get(key)!=cfg.get(key) for key in ('num_hidden_layers','hidden_size','vocab_size')):
        raise ValueError('Training and inference DeepSeek geometry must match')
    target_quantization=validate_quant_config(target_cfg.get('quantization_config'))
    if config.sglang.kv_cache_dtype != 'bfloat16' or config.sglang.dtype != 'bfloat16':
        raise ValueError('This smoke profile requires BF16 inference weights and KV cache')
    # Ray service workers need these values explicitly; environment inherited
    # by the outer driver alone is insufficient.
    keys = ('SGLANG_USE_AITER','SGLANG_USE_AITER_AR','SGLANG_ROCM_USE_AITER_MOE',
            'SGLANG_USE_FP8_W8A8_MOE','SGLANG_USE_RMS_QUANT_PATH',
            'USE_FUSED_RMS_QUANT_PATH','SGLANG_USE_FUSED_RMS_QUANT',
            'SGLANG_USE_FUSED_SILU_MUL_QUANT','SGLANG_USE_FUSED_RMSNORM_ROPE',
            'SGLANG_USE_FUSED_MLA_CAT','SGLANG_USE_MARLIN_W16A16_MOE')
    for engine in (config.actor,config.rollout):
        for spec in engine.scheduling_spec:
            spec.env_vars[TARGET_ENV]=json.dumps(target_quantization)
            for key in keys:
                spec.env_vars[key] = '0'
            for key in ('AWEX_WU_USE_GROUP','LD_LIBRARY_PATH','NCCL_NET','NCCL_IB_DISABLE','NCCL_IB_GID_INDEX',
                        'NCCL_SOCKET_FAMILY','NCCL_DEBUG','NCCL_DEBUG_SUBSYS',
                        'HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HOME',
                        'HF_DATASETS_CACHE','HF_HUB_CACHE','HCU_AREAL_PATCH_TRACE_DIR'):
                if key in os.environ:
                    spec.env_vars[key] = os.environ[key]
    # Interface/HCA names and NCCL_DEBUG_FILE remain node-local: their spelling
    # can differ between machines and must be set before Ray starts there.


def main(argv):
    from areal.api.cli_args import GRPOConfig, load_expr_config
    check_only = '--check-only' in argv
    argv = [arg for arg in argv if arg != '--check-only']
    config,_ = load_expr_config(argv,GRPOConfig)
    validate_config(config)
    if check_only:
        print(json.dumps(dict(status='passed',nodes=2,controller='v2',weight_update='awex',
                              actor_backend=config.actor.backend,rollout_backend=config.rollout.backend,
                              inference_max_running_requests=config.sglang.max_running_requests,
                              rollout_max_concurrent=config.rollout.max_concurrent_rollouts,
                              weights_exported=False)))
        return
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer
    from hcu_areal.validation.deepseek_v2_actor import DeepseekV2Trainer
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    datasets = {split:get_custom_dataset(split=split,dataset_config=section,tokenizer=tokenizer)
                for split,section in (('train',config.train_dataset),('test',config.valid_dataset))}
    workflow = 'areal.workflow.openai.math_agent.MathAgent'
    kwargs = dict(temperature=config.gconfig.temperature,top_p=config.gconfig.top_p,
                  max_completion_tokens=config.gconfig.max_new_tokens)
    with DeepseekV2Trainer(config,train_dataset=datasets['train'],valid_dataset=datasets['test']) as trainer:
        trainer.train(workflow=workflow,workflow_kwargs=kwargs,
                      eval_workflow=workflow,eval_workflow_kwargs=kwargs)


if __name__ == '__main__':
    main(sys.argv[1:])
