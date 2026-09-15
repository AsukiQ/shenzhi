#!/bin/bash
#SBATCH --time=04:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final}
QUERIES_PATH=${QUERIES_PATH:-${DATA_ROOT}/retrieval.jsonl}
SKILLS_PATH=${SKILLS_PATH:-${DATA_ROOT}/skill_pool.jsonl}
QRELS_PATH=${QRELS_PATH:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/qrels.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-.cache/hf_models/SkillRouter-Embedding-0.6B}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TOP_K=${TOP_K:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
MODEL_DIM=${MODEL_DIM:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}
RUN_NAME=${RUN_NAME:-clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Stage0 full retrieval eval requires completed checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ ! -s "${QRELS_PATH}" ]]; then
  echo "ERROR: Stage0 full retrieval eval requires same-pool qrels from baseline: ${QRELS_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/metrics.json"
rm -f "${OUTPUT_DIR}/eval_stdout.json"
rm -f "${OUTPUT_DIR}/run.tsv"
rm -f "${OUTPUT_DIR}/predictions.jsonl"
rm -f "${OUTPUT_DIR}/run_report.json"
rm -f "${OUTPUT_DIR}/progress.json"
rm -f "${OUTPUT_DIR}/export_stdout.json"

"${PYTHON_BIN}" scripts/export_clstr_retrieval_run.py \
  --queries_path "${QUERIES_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model_name "${BASE_MODEL_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --model_dim "${MODEL_DIM}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --freeze_backbone \
  --normalize_embeddings \
  --skill_text_format skillret_official \
  --skill_table_batch_size "${SKILL_TABLE_BATCH_SIZE}" \
  --disable_cross_encoder \
  --query_text_format skillrouter \
  --local_files_only \
  --run_name "${RUN_NAME}" | tee "${OUTPUT_DIR}/export_stdout.json"

"${PYTHON_BIN}" scripts/evaluate_retrieval_run.py \
  --qrels_path "${QRELS_PATH}" \
  --run_path "${OUTPUT_DIR}/run.tsv" \
  --output_dir "${OUTPUT_DIR}" \
  --run_format trec \
  --k_values 20 50 100 \
  --benchmark clstr_unified_stage0 \
  --method "${RUN_NAME}" | tee "${OUTPUT_DIR}/eval_stdout.json"
