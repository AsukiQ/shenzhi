#!/bin/bash
#SBATCH --job-name=clstr_s0_legacy_compat
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH is required}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
BACKBONE_SNAPSHOT_PATH=${BACKBONE_SNAPSHOT_PATH:?BACKBONE_SNAPSHOT_PATH is required}
LEGACY_INIT_CHECKPOINT_PATH=${LEGACY_INIT_CHECKPOINT_PATH:?LEGACY_INIT_CHECKPOINT_PATH is required}
LEGACY_INIT_SKILLS_PATH=${LEGACY_INIT_SKILLS_PATH:?LEGACY_INIT_SKILLS_PATH is required}

eval "$("${PYTHON_BIN}" scripts/resolve_clstr_vnext_full_inputs.py "${DATA_MANIFEST_PATH}" --shell)"

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage0_train.py \
  --skills_path "${TRAINING_SKILLS}" \
  --retrieval_rows_path "${RETRIEVAL_ROWS}" \
  --retrieval_dev_rows_path "${RETRIEVAL_DEV_ROWS}" \
  --static_route_rows_path "${STATIC_ROUTE_ROWS}" \
  --static_route_dev_rows_path "${STATIC_ROUTE_DEV_ROWS}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS}" \
  --data_contract_path "${DATA_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_steps 1 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0 \
  --model_dim 1024 \
  --max_length 2048 \
  --torch_dtype bfloat16 \
  --skill_table_batch_size 64 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 2048 \
  --skill_cache_shard_size 2048 \
  --frozen_cache_dir "${OUTPUT_DIR}/frozen_qwen_cache" \
  --backbone_snapshot_path "${BACKBONE_SNAPSHOT_PATH}" \
  --hard_negative_loss_weight 0.2 \
  --hard_negative_margin 0.1 \
  --hard_negative_top_k 32 \
  --max_retrieval_rows 128 \
  --max_static_rows 128 \
  --seed 17 \
  --checkpoint_interval 1 \
  --validation_interval 1 \
  --validation_batch_size 64 \
  --max_dev_rows_per_kind 128 \
  --minimum_dev_score_gain 0.0 \
  --legacy_init_checkpoint_path "${LEGACY_INIT_CHECKPOINT_PATH}" \
  --legacy_init_skills_path "${LEGACY_INIT_SKILLS_PATH}" \
  --require_clean_source
