#!/usr/bin/env bash
# Source BEFORE starting Ray, on every node, and before the training launcher.
# Uses the existing DeepSeek launcher; no default launchers are changed.

export AREAL_ENV_PROFILE=deepseek
export SGLANG_KV_CACHE_DTYPE=bfloat16
# The local sglang/sglang-new sources register this backend as dcu_mla.
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-dcu_mla}"

# Force off quantizing paths, including values inherited from an earlier job.
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_RMS_QUANT_PATH=0
export USE_FUSED_RMS_QUANT_PATH=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_FUSED_SILU_MUL_QUANT=0
export SGLANG_USE_FUSED_RMSNORM_ROPE=0
export SGLANG_USE_FUSED_MLA_CAT=0
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AR=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_MARLIN_W16A16_MOE=0

# Conservative smoke-run defaults; override before sourcing when needed.
# Unquantized MoE falls back to the existing Triton BF16 implementation.
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
export ACTOR_MB_TOKENS="${ACTOR_MB_TOKENS:-4096}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export N_SAMPLES="${N_SAMPLES:-2}"
export TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-2}"
