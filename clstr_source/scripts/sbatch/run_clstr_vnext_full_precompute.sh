#!/bin/bash
#SBATCH --job-name=clstr_vnext_precompute
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
CACHE_ROOT=$(realpath -m "${CACHE_ROOT:-${RUN_ROOT}/frozen_qwen_cache}")
case "${CACHE_ROOT}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: writable CACHE_ROOT must remain inside RUN_ROOT: cache=%s run=%s\n' \
      "${CACHE_ROOT}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
OUTPUT_DIR=${RUN_ROOT}/precompute

eval "$("${PYTHON_BIN}" scripts/resolve_clstr_vnext_full_inputs.py "${DATA_MANIFEST_PATH}" --shell)"
if [[ -e "${OUTPUT_DIR}/precompute_report.json" ]]; then
  printf 'ERROR: precompute report already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/run_clstr_vnext_precompute.py \
  --skills_path "${TRAINING_SKILLS}" \
  --retrieval_rows_path "${RETRIEVAL_ROWS}" \
  --retrieval_dev_rows_path "${RETRIEVAL_DEV_ROWS}" \
  --static_route_rows_path "${STATIC_ROUTE_ROWS}" \
  --static_route_dev_rows_path "${STATIC_ROUTE_DEV_ROWS}" \
  --trajectory_rows_path "${TRAJECTORY_ROWS}" \
  --trajectory_dev_rows_path "${TRAJECTORY_DEV_ROWS}" \
  --causal_pair_support_rows_path "${PAIR_SUPPORT_ROWS}" \
  --causal_pair_support_dev_rows_path "${PAIR_SUPPORT_DEV_ROWS}" \
  --data_contract_path "${DATA_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_root "${CACHE_ROOT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --model_dim 1024 \
  --max_length 2048 \
  --torch_dtype bfloat16 \
  --skill_table_batch_size 64 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --skill_cache_shard_size 2048
