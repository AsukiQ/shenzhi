#!/bin/bash
#SBATCH --job-name=clstr_vnext_s0_smoke
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT must come from the freshly audited causal manifest}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be a fresh Stage0 smoke directory}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${ASSET_ROOT}/models/Qwen3-Embedding-0.6B}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR must identify this smoke chain cache}
MAX_STEPS=${MAX_STEPS:-2}
RESUME_STEP=${RESUME_STEP:-1}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-1}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-${VALIDATION_INTERVAL}}
MAX_DEV_ROWS_PER_KIND=${MAX_DEV_ROWS_PER_KIND:-32}
MAX_LENGTH=${MAX_LENGTH:-2048}
BELIEF_TOP_K=${BELIEF_TOP_K:-64}

if (( RESUME_STEP <= 0 || MAX_STEPS <= RESUME_STEP )); then
  printf 'ERROR: require 0 < RESUME_STEP < MAX_STEPS\n' >&2
  exit 2
fi

for path in \
  "${BUNDLE_ROOT}/training_skills.jsonl" \
  "${BUNDLE_ROOT}/retrieval_rows.jsonl" \
  "${BUNDLE_ROOT}/retrieval_dev_rows.jsonl" \
  "${BUNDLE_ROOT}/static_route_rows.jsonl" \
  "${BUNDLE_ROOT}/static_route_dev_rows.jsonl" \
  "${BUNDLE_ROOT}/inventory_catalogs.jsonl" \
  "${BUNDLE_ROOT}/manifest.json"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: missing smoke bundle artifact: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  printf 'ERROR: missing local Qwen-0.6B model: %s\n' "${MODEL_NAME_OR_PATH}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: Stage0 smoke output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

common_args=(
  --skills_path "${BUNDLE_ROOT}/training_skills.jsonl"
  --retrieval_rows_path "${BUNDLE_ROOT}/retrieval_rows.jsonl"
  --retrieval_dev_rows_path "${BUNDLE_ROOT}/retrieval_dev_rows.jsonl"
  --static_route_rows_path "${BUNDLE_ROOT}/static_route_rows.jsonl"
  --static_route_dev_rows_path "${BUNDLE_ROOT}/static_route_dev_rows.jsonl"
  --inventory_catalogs_path "${BUNDLE_ROOT}/inventory_catalogs.jsonl"
  --data_contract_path "${BUNDLE_ROOT}/manifest.json"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --batch_size 16
  --gradient_accumulation_steps 1
  --learning_rate 2.0e-5
  --model_dim 1024
  --max_length "${MAX_LENGTH}"
  --torch_dtype bfloat16
  --skill_table_batch_size 64
  --belief_top_k "${BELIEF_TOP_K}"
  --cache_batch_size 64
  --cache_shard_size 64
  --frozen_cache_dir "${FROZEN_CACHE_DIR}"
  --skill_cache_shard_size 128
  --seed 17
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --validation_interval "${VALIDATION_INTERVAL}"
  --validation_batch_size 32
  --max_dev_rows_per_kind "${MAX_DEV_ROWS_PER_KIND}"
  --minimum_dev_score_gain -1.0
)

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage0_train.py \
  "${common_args[@]}" \
  --max_steps "${RESUME_STEP}" \
  | tee "${OUTPUT_DIR}.step${RESUME_STEP}.stdout.log"

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage0_train.py \
  "${common_args[@]}" \
  --max_steps "${MAX_STEPS}" \
  --resume_checkpoint_path "${OUTPUT_DIR}/checkpoints/clstr_vnext_stage0-step${RESUME_STEP}.pt" \
  | tee "${OUTPUT_DIR}.resume_step${MAX_STEPS}.stdout.log"
