#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
export APPWORLD_CACHE=${APPWORLD_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=${PROJECT_ROOT}/.deps/appworld_py310:${PROJECT_ROOT}:${PYTHONPATH:-}

METHOD=${METHOD:-qwen_only}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_smoke}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
TOP_K=${TOP_K:-5}
MAX_STEPS=${MAX_STEPS:-3}
MAX_TASKS=${MAX_TASKS:-2}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-10}
MAX_APIS_PER_APP=${MAX_APIS_PER_APP:-12}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
SKILL_CONTEXT_MODE=${SKILL_CONTEXT_MODE:-safe_metadata}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/run_appworld_multistep_executor_eval.py \
  --method "${METHOD}" \
  --tasks_path "${TASKS_PATH}" \
  --skill_pool_path "${SKILL_POOL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --appworld_root "${APPWORLD_ROOT}" \
  --appworld_cache "${APPWORLD_CACHE}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --top_k "${TOP_K}" \
  --max_steps "${MAX_STEPS}" \
  --max_tasks "${MAX_TASKS}" \
  --max_interactions "${MAX_INTERACTIONS}" \
  --max_apis_per_app "${MAX_APIS_PER_APP}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --timeout_seconds "${TIMEOUT_SECONDS}" \
  --skill_context_mode "${SKILL_CONTEXT_MODE}" \
  --local_files_only \
  --no-enable_thinking \
  | tee "${OUTPUT_DIR}/stdout.json"
