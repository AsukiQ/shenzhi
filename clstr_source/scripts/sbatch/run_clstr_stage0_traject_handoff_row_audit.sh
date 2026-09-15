#!/bin/bash
#SBATCH --time=00:30:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_stage0_handoff_row_diagnostics/v4_traject_bench_960}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-${OUTPUT_DIR}/model_cache}
BENCHMARK=${BENCHMARK:-traject_bench}
QUERY_MODE=${QUERY_MODE:-skillrouter_state}
MAX_ROWS=${MAX_ROWS:-960}
BATCH_SIZE=${BATCH_SIZE:-8}
TOP_K=${TOP_K:-10}
K_VALUES=${K_VALUES:-20,50,100,200,500}

K_VALUES="${K_VALUES//:/,}"
K_VALUES="${K_VALUES//;/,}"

if [[ -s "${CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 row audit requires completed checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -s "${TRAIN_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 row audit requires trajectories: ${TRAIN_PATH}" >&2
  exit 2
fi

if [[ -s "${SKILLS_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 row audit requires skill pool: ${SKILLS_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${MODEL_CACHE_DIR}"
rm -f "${OUTPUT_DIR}/report.json"
rm -f "${OUTPUT_DIR}/summary.md"
rm -f "${OUTPUT_DIR}/row_diagnostics.jsonl"
rm -f "${OUTPUT_DIR}/missed_gt500.jsonl"
rm -f "${OUTPUT_DIR}/audit_stdout.json"

"${PYTHON_BIN}" scripts/audit_clstr_stage0_handoff_rows.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --benchmark "${BENCHMARK}" \
  --query_mode "${QUERY_MODE}" \
  --max_rows "${MAX_ROWS}" \
  --batch_size "${BATCH_SIZE}" \
  --top_k "${TOP_K}" \
  --k_values "${K_VALUES}" \
  --model_cache_dir "${MODEL_CACHE_DIR}" | tee "${OUTPUT_DIR}/audit_stdout.json"
