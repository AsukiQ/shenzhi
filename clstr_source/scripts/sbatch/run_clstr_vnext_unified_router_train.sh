#!/bin/bash
#SBATCH --job-name=clstr_unified_router
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:?STAGE2_CHECKPOINT_PATH is required}
STAGE2_RELEASE_SELECTION_PATH=${STAGE2_RELEASE_SELECTION_PATH:?STAGE2_RELEASE_SELECTION_PATH is required}
FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:?FOUNDATION_CHECKPOINT_PATH is required}
TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:?TRAINING_SKILLS_PATH is required}
TRAJECTORY_DEV_ROWS_PATH=${TRAJECTORY_DEV_ROWS_PATH:?TRAJECTORY_DEV_ROWS_PATH is required}
STATIC_ROUTE_DEV_ROWS_PATH=${STATIC_ROUTE_DEV_ROWS_PATH:?STATIC_ROUTE_DEV_ROWS_PATH is required}
INVENTORY_CATALOGS_PATH=${INVENTORY_CATALOGS_PATH:?INVENTORY_CATALOGS_PATH is required}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}

for path in \
  "${STAGE2_CHECKPOINT_PATH}" \
  "${STAGE2_RELEASE_SELECTION_PATH}" \
  "${FOUNDATION_CHECKPOINT_PATH}" \
  "${TRAINING_SKILLS_PATH}" \
  "${TRAJECTORY_DEV_ROWS_PATH}" \
  "${STATIC_ROUTE_DEV_ROWS_PATH}" \
  "${INVENTORY_CATALOGS_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: missing unified-router input: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: unified-router output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/train_clstr_vnext_unified_router.py \
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}" \
  --stage2_release_selection_path "${STAGE2_RELEASE_SELECTION_PATH}" \
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}" \
  --training_skills_path "${TRAINING_SKILLS_PATH}" \
  --trajectory_dev_rows_path "${TRAJECTORY_DEV_ROWS_PATH}" \
  --static_route_dev_rows_path "${STATIC_ROUTE_DEV_ROWS_PATH}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS_PATH}" \
  --frozen_cache_dir "${FROZEN_CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_dynamic_rows "${MAX_DYNAMIC_ROWS:-2048}" \
  --max_static_rows "${MAX_STATIC_ROWS:-4096}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --cache_batch_size "${CACHE_BATCH_SIZE:-128}" \
  --belief_top_k "${BELIEF_TOP_K:-64}" \
  --coarse_k 500 \
  --compressed_m 64 \
  --max_horizon 16 \
  --epochs "${EPOCHS:-60}" \
  --learning_rate "${LEARNING_RATE:-0.003}" \
  --expert_utility_loss_weight "${EXPERT_UTILITY_LOSS_WEIGHT:-1.0}" \
  --routing_mode "${ROUTING_MODE:-sparse_top1_expert}" \
  --seed "${SEED:-29}" \
  | tee "${OUTPUT_DIR}.stdout.log"
