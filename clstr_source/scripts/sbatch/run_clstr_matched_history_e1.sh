#!/bin/bash
#SBATCH --job-name=clstr_e1_history
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

ENCODER_KIND=${ENCODER_KIND:?ENCODER_KIND is required}
FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:?FOUNDATION_CHECKPOINT_PATH is required}
SKILLS_PATH=${SKILLS_PATH:?SKILLS_PATH is required}
TRAIN_ROWS_PATH=${TRAIN_ROWS_PATH:?TRAIN_ROWS_PATH is required}
DEV_ROWS_PATH=${DEV_ROWS_PATH:?DEV_ROWS_PATH is required}
INVENTORY_CATALOGS_PATH=${INVENTORY_CATALOGS_PATH:?INVENTORY_CATALOGS_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR is required}

args=(
  --encoder_kind "${ENCODER_KIND}"
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}"
  --skills_path "${SKILLS_PATH}"
  --train_rows_path "${TRAIN_ROWS_PATH}"
  --dev_rows_path "${DEV_ROWS_PATH}"
  --inventory_catalogs_path "${INVENTORY_CATALOGS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --frozen_cache_dir "${FROZEN_CACHE_DIR}"
  --max_steps "${MAX_STEPS:-500}"
  --batch_size "${BATCH_SIZE:-8}"
  --learning_rate "${LEARNING_RATE:-1.0e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --seed "${SEED:-23}"
  --validation_interval "${VALIDATION_INTERVAL:-100}"
  --validation_batch_size "${VALIDATION_BATCH_SIZE:-32}"
  --cache_batch_size "${CACHE_BATCH_SIZE:-128}"
  --cache_shard_size "${CACHE_SHARD_SIZE:-4096}"
)
if [[ -n "${MAX_TRAIN_ROWS_PER_BENCHMARK:-}" ]]; then
  args+=(--max_train_rows_per_benchmark "${MAX_TRAIN_ROWS_PER_BENCHMARK}")
fi
if [[ -n "${MAX_DEV_ROWS_PER_BENCHMARK:-}" ]]; then
  args+=(--max_dev_rows_per_benchmark "${MAX_DEV_ROWS_PER_BENCHMARK}")
fi

"${PYTHON_BIN}" scripts/run_clstr_matched_history_e1.py "${args[@]}"
