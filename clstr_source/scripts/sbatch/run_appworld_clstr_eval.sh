#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model/appworld_mini.yaml}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
QRELS_PATH=${QRELS_PATH:-data/appworld_routing/dev_qrels.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_clstr_eval/dev}
TOP_K=${TOP_K:-20}
RANKING_MODE=${RANKING_MODE:-skill_table}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-}
POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.25}
ALLOW_LEGACY_POLICY_SKILL_ROUTER=${ALLOW_LEGACY_POLICY_SKILL_ROUTER:-0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/appworld_clstr_train/checkpoints}
if [ -z "${CHECKPOINT_PATH}" ]; then
  CHECKPOINT_PATH=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')
fi
if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: no checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

ARGS=(
  --model_config "${MODEL_CONFIG}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --tasks_path "${TASKS_PATH}" \
  --qrels_path "${QRELS_PATH}" \
  --skill_pool_path "${SKILL_POOL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --top_k "${TOP_K}" \
  --ranking_mode "${RANKING_MODE}" \
  --policy_blend_alpha "${POLICY_BLEND_ALPHA}"
)
if [ -n "${CANDIDATE_TOP_K}" ]; then
  ARGS+=(--candidate_top_k "${CANDIDATE_TOP_K}")
fi
if [ "${ALLOW_LEGACY_POLICY_SKILL_ROUTER}" = "1" ]; then
  ARGS+=(--allow_legacy_policy_skill_router)
fi

"${PYTHON_BIN}" scripts/run_appworld_clstr_eval.py "${ARGS[@]}"
