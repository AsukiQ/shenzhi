#!/bin/bash
#SBATCH --job-name=clstr_e2_probe
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:?FOUNDATION_CHECKPOINT_PATH is required}
SKILLS_PATH=${SKILLS_PATH:?SKILLS_PATH is required}
ALIGNED_ROWS_PATH=${ALIGNED_ROWS_PATH:?ALIGNED_ROWS_PATH is required}
SERIALIZED_CHECKPOINT_PATH=${SERIALIZED_CHECKPOINT_PATH:?SERIALIZED_CHECKPOINT_PATH is required}
GRU_CHECKPOINT_PATH=${GRU_CHECKPOINT_PATH:?GRU_CHECKPOINT_PATH is required}
TRANSFORMER_CHECKPOINT_PATH=${TRANSFORMER_CHECKPOINT_PATH:?TRANSFORMER_CHECKPOINT_PATH is required}
LSTR_CHECKPOINT_PATH=${LSTR_CHECKPOINT_PATH:?LSTR_CHECKPOINT_PATH is required}
EXPECTED_E1_SEED=${EXPECTED_E1_SEED:?EXPECTED_E1_SEED is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR is required}

"${PYTHON_BIN}" scripts/run_clstr_state_probe_e2.py \
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --aligned_rows_path "${ALIGNED_ROWS_PATH}" \
  --serialized_checkpoint_path "${SERIALIZED_CHECKPOINT_PATH}" \
  --gru_checkpoint_path "${GRU_CHECKPOINT_PATH}" \
  --transformer_checkpoint_path "${TRANSFORMER_CHECKPOINT_PATH}" \
  --lstr_checkpoint_path "${LSTR_CHECKPOINT_PATH}" \
  --expected_e1_seed "${EXPECTED_E1_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --frozen_cache_dir "${FROZEN_CACHE_DIR}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --cache_batch_size "${CACHE_BATCH_SIZE:-128}" \
  --cache_shard_size "${CACHE_SHARD_SIZE:-4096}" \
  --probe_max_iter "${PROBE_MAX_ITER:-100}" \
  --bootstrap_draws "${BOOTSTRAP_DRAWS:-2000}" \
  --bootstrap_seed "${BOOTSTRAP_SEED:-29}"
