#!/usr/bin/env bash
# Source on BOTH selected nodes before starting their dedicated Ray instances.
# This file only sets environment variables; it launches no services.
_deepseek_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AREAL_HOME="${AREAL_HOME:-$_deepseek_repo}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
export VENV="${VENV:-}" VENV_PATH="${VENV_PATH:-}"
export MEGATRON_HOME="${MEGATRON_HOME:-/workspace/dcu_megatron}"
export MEGATRON_3RDPARTY_HOME="${MEGATRON_3RDPARTY_HOME:-$MEGATRON_HOME}"
export SGLANG_ROOT="${SGLANG_ROOT:-/workspace/test_v1.0.4/areal_runs/codex-qwen3-moe-bw61-bw67-20261023/deps/sglang}"
export MODEL_PATH="${MODEL_PATH:-/workspace/deepseek-r1-4layers-bf16}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export CONTROLLER_VERSION=v2 WEIGHT_UPDATE_MODE=awex
export N_NODES=2 N_GPUS_PER_NODE=8
export ACTOR_BACKEND='megatron:(attn:d1p1t8|ffn:d1p1t1e8)'
export ROLLOUT_BACKEND='sglang:d1p1t8'
export BRIDGE_TYPE=megatron-bridge USE_MBRIDGE_SAVE=false
export GRPO_ENTRYPOINT=hcu_areal/validation/deepseek_dual_train.py
export GRPO_CONFIG=hcu_areal/validation/deepseek_dual_config.yaml
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}" VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-1}"
export N_SAMPLES="${N_SAMPLES:-4}" ACTOR_N_MBS="${ACTOR_N_MBS:-4}"
export ACTOR_MB_TOKENS="${ACTOR_MB_TOKENS:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}" TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-2}"
export TOTAL_TRAIN_EPOCHS=1
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-4}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.45}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-1024}"
export SGLANG_PAGE_SIZE=64
source "$_deepseek_repo/hcu_example/common/deepseek_bf16_env.sh"
# Keep the proven HSA device remapping for v2 guard processes, using this
# profile's own import-light helper rather than the GLM-only actor module.
export AREAL_EXTRA_PYTHONPATH="$AREAL_HOME/hcu_areal/validation/worker_env"
export DTE_DELTA_TRANSFER=0 DTE_SEPARATION_WEIGHT_UPDATE=0
unset _deepseek_repo
