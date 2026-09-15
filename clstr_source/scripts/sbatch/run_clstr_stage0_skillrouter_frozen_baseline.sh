#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
TOP_K=${TOP_K:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_QUERIES=${MAX_QUERIES:-}
MAX_SKILLS=${MAX_SKILLS:-}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/run_clstr_stage0_skillrouter_frozen_baseline.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --top_k "${TOP_K}"
  --batch_size "${BATCH_SIZE}"
  --score_batch_size "${SCORE_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)

if [ -n "${MAX_QUERIES}" ]; then
  ARGS+=(--max_queries "${MAX_QUERIES}")
fi

if [ -n "${MAX_SKILLS}" ]; then
  ARGS+=(--max_skills "${MAX_SKILLS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/baseline_stdout.json"
