#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export APPWORLD_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root
export APPWORLD_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
METHOD=${METHOD:-qwen_only}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_executor_smoke/${METHOD}}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
TOP_K=${TOP_K:-5}
DEDUPE_CANONICAL_SKILLS=${DEDUPE_CANONICAL_SKILLS:-0}
MAX_AUTH_LIKE_SKILLS=${MAX_AUTH_LIKE_SKILLS:-}
SKILL_CONTEXT_MODE=${SKILL_CONTEXT_MODE:-raw}
MAX_TASKS=${MAX_TASKS:-5}
MAX_APIS_PER_APP=${MAX_APIS_PER_APP:-12}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-3}

ARGS=(
  --method "${METHOD}"
  --tasks_path "${TASKS_PATH}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --appworld_root "${APPWORLD_ROOT}"
  --appworld_cache "${APPWORLD_CACHE}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --top_k "${TOP_K}"
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
if [ "${DEDUPE_CANONICAL_SKILLS}" = "1" ]; then
  ARGS+=(--dedupe_canonical_skills)
fi
if [ -n "${MAX_AUTH_LIKE_SKILLS}" ]; then
  ARGS+=(--max_auth_like_skills "${MAX_AUTH_LIKE_SKILLS}")
fi

"${PYTHON_BIN}" scripts/run_appworld_qwen_executor_eval.py "${ARGS[@]}"
