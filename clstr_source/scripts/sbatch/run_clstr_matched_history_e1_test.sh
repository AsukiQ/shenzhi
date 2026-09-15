#!/bin/bash
#SBATCH --job-name=clstr_e1_test
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

ENCODER_KIND=${ENCODER_KIND:?ENCODER_KIND is required}
SEED=${SEED:?SEED is required}
FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:?FOUNDATION_CHECKPOINT_PATH is required}
SKILLS_PATH=${SKILLS_PATH:?SKILLS_PATH is required}
TEST_ROWS_PATH=${TEST_ROWS_PATH:?TEST_ROWS_PATH is required}
TEST_DATA_REPORT_PATH=${TEST_DATA_REPORT_PATH:?TEST_DATA_REPORT_PATH is required}
INVENTORY_CATALOGS_PATH=${INVENTORY_CATALOGS_PATH:?INVENTORY_CATALOGS_PATH is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
TRAIN_REPORT_PATH=${TRAIN_REPORT_PATH:?TRAIN_REPORT_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR is required}

"${PYTHON_BIN}" scripts/run_clstr_matched_history_e1_test.py \
  --encoder_kind "${ENCODER_KIND}" \
  --seed "${SEED}" \
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --test_rows_path "${TEST_ROWS_PATH}" \
  --test_data_report_path "${TEST_DATA_REPORT_PATH}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --train_report_path "${TRAIN_REPORT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --frozen_cache_dir "${FROZEN_CACHE_DIR}" \
  --validation_batch_size "${VALIDATION_BATCH_SIZE:-32}" \
  --cache_batch_size "${CACHE_BATCH_SIZE:-128}" \
  --cache_shard_size "${CACHE_SHARD_SIZE:-4096}"
