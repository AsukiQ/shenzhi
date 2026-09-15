#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
export APPWORLD_CACHE=${APPWORLD_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld}
export IPYTHONDIR=${IPYTHONDIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
METHOD=${METHOD:-qwen_only}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_multistep_executor_smoke/${METHOD}}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-}
CLSTR_MODEL_CONFIG=${CLSTR_MODEL_CONFIG:-}
CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-}
SKILLROUTER_MODEL_NAME_OR_PATH=${SKILLROUTER_MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
SKILLROUTER_CHECKPOINT_PATH=${SKILLROUTER_CHECKPOINT_PATH:-outputs/appworld_skillrouter_base_train_v1/model.pt}
SKILLROUTER_BATCH_SIZE=${SKILLROUTER_BATCH_SIZE:-8}
SKILLROUTER_MAX_LENGTH=${SKILLROUTER_MAX_LENGTH:-1024}
SKILLROUTER_DEVICE=${SKILLROUTER_DEVICE:-}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
TOP_K=${TOP_K:-5}
MAX_STEPS=${MAX_STEPS:-3}
RANKING_MODE=${RANKING_MODE:-auto}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-}
CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing}
POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.25}
ALLOW_LEGACY_POLICY_SKILL_ROUTER=${ALLOW_LEGACY_POLICY_SKILL_ROUTER:-0}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
CLSTR_ALPHA=${CLSTR_ALPHA:-0.25}
RECURRENT_BELIEF=${RECURRENT_BELIEF:-1}
DEDUPE_CANONICAL_SKILLS=${DEDUPE_CANONICAL_SKILLS:-0}
MAX_AUTH_LIKE_SKILLS=${MAX_AUTH_LIKE_SKILLS:-}
APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-0}
SKILL_CONTEXT_MODE=${SKILL_CONTEXT_MODE:-safe_metadata}
MAX_TASKS=${MAX_TASKS:-5}
MAX_APIS_PER_APP=${MAX_APIS_PER_APP:-12}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-10}
USE_STOP_HEAD=${USE_STOP_HEAD:-0}

ARGS=(
  --method "${METHOD}"
  --tasks_path "${TASKS_PATH}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --appworld_root "${APPWORLD_ROOT}"
  --appworld_cache "${APPWORLD_CACHE}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --top_k "${TOP_K}"
  --max_steps "${MAX_STEPS}"
  --ranking_mode "${RANKING_MODE}"
  --candidate_source "${CANDIDATE_SOURCE}"
  --policy_blend_alpha "${POLICY_BLEND_ALPHA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --clstr_alpha "${CLSTR_ALPHA}"
  --skillrouter_model_name_or_path "${SKILLROUTER_MODEL_NAME_OR_PATH}"
  --skillrouter_checkpoint_path "${SKILLROUTER_CHECKPOINT_PATH}"
  --skillrouter_batch_size "${SKILLROUTER_BATCH_SIZE}"
  --skillrouter_max_length "${SKILLROUTER_MAX_LENGTH}"
  --skill_context_mode "${SKILL_CONTEXT_MODE}"
  --max_tasks "${MAX_TASKS}"
  --max_apis_per_app "${MAX_APIS_PER_APP}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --torch_dtype "${TORCH_DTYPE}"
  --timeout_seconds "${TIMEOUT_SECONDS}"
  --max_interactions "${MAX_INTERACTIONS}"
)

if [ -n "${PREDICTIONS_PATH}" ]; then
  ARGS+=(--predictions_path "${PREDICTIONS_PATH}")
fi
if [ -n "${BASE_SKILL_POOL_PATH}" ]; then
  ARGS+=(--base_skill_pool_path "${BASE_SKILL_POOL_PATH}")
fi
if [ -n "${CLSTR_MODEL_CONFIG}" ]; then
  ARGS+=(--clstr_model_config "${CLSTR_MODEL_CONFIG}")
fi
if [ -n "${CLSTR_CHECKPOINT_PATH}" ]; then
  ARGS+=(--clstr_checkpoint_path "${CLSTR_CHECKPOINT_PATH}")
fi
if [ -n "${CANDIDATE_TOP_K}" ]; then
  ARGS+=(--candidate_top_k "${CANDIDATE_TOP_K}")
fi
if [ "${ALLOW_LEGACY_POLICY_SKILL_ROUTER}" = "1" ]; then
  ARGS+=(--allow_legacy_policy_skill_router)
fi
if [ -n "${SKILLROUTER_DEVICE}" ]; then
  ARGS+=(--skillrouter_device "${SKILLROUTER_DEVICE}")
fi
if [ "${RECURRENT_BELIEF}" = "1" ]; then
  ARGS+=(--recurrent_belief)
else
  ARGS+=(--no-recurrent_belief)
fi
if [ "${DEDUPE_CANONICAL_SKILLS}" = "1" ]; then
  ARGS+=(--dedupe_canonical_skills)
fi
if [ -n "${MAX_AUTH_LIKE_SKILLS}" ]; then
  ARGS+=(--max_auth_like_skills "${MAX_AUTH_LIKE_SKILLS}")
fi
if [ "${APPWORLD_EXECUTOR_COMPATIBLE_ONLY}" = "1" ]; then
  ARGS+=(--appworld_executor_compatible_only)
fi
if [ "${USE_STOP_HEAD}" = "1" ]; then
  ARGS+=(--use_stop_head)
else
  ARGS+=(--no-use_stop_head)
fi

"${PYTHON_BIN}" scripts/run_appworld_multistep_executor_eval.py "${ARGS[@]}"
