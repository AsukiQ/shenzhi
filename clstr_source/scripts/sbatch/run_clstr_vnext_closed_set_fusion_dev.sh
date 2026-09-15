#!/bin/bash
#SBATCH --job-name=clstr-fusion-dev
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:?TRAINING_SKILLS_PATH is required}
STATIC_ROUTE_DEV_ROWS=${STATIC_ROUTE_DEV_ROWS:?STATIC_ROUTE_DEV_ROWS is required}
INVENTORY_CATALOGS=${INVENTORY_CATALOGS:?INVENTORY_CATALOGS is required}
EVAL_CACHE_ROOT=${EVAL_CACHE_ROOT:?EVAL_CACHE_ROOT is required}
OUTPUT_PATH=${OUTPUT_PATH:?OUTPUT_PATH is required}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_clstr_vnext_closed_set_fusion_dev.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --training_skills_path "${TRAINING_SKILLS_PATH}" \
  --static_route_dev_rows_path "${STATIC_ROUTE_DEV_ROWS}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS}" \
  --frozen_cache_dir "${EVAL_CACHE_ROOT}" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size 128 \
  --belief_top_k 64
