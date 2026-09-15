#!/bin/bash
#SBATCH --job-name=clstr_vnext_data_filter
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

required=(
  INPUT_ROOT PREFLIGHT_REPORT_PATH OUTPUT_ROOT
  TOOLBENCH_EVAL_TRAJECTORIES_PATH TRAJECT_EVAL_QUERIES_PATH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is empty: %s\n' "${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_ROOT}" ]]; then
  printf 'ERROR: OUTPUT_ROOT already exists: %s\n' "${OUTPUT_ROOT}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/filter_clstr_protected_near_duplicates.py \
  --input_root "${INPUT_ROOT}" \
  --preflight_report_path "${PREFLIGHT_REPORT_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  | tee "${PREFLIGHT_REPORT_PATH%/*}/filter_stdout.json"

"${PYTHON_BIN}" scripts/audit_clean_training_preflight.py \
  --data_root "${OUTPUT_ROOT}" \
  --toolbench_eval_trajectories_path "${TOOLBENCH_EVAL_TRAJECTORIES_PATH}" \
  --traject_eval_queries_path "${TRAJECT_EVAL_QUERIES_PATH}" \
  --output_path "${OUTPUT_ROOT}/clean_preflight.json" \
  --near_duplicate_threshold "${NEAR_DUPLICATE_THRESHOLD:-0.55}" \
  --near_duplicate_max_chars "${NEAR_DUPLICATE_MAX_CHARS:-1000}" \
  --near_duplicate_max_shingle_postings "${NEAR_DUPLICATE_MAX_SHINGLE_POSTINGS:-256}" \
  --near_duplicate_max_query_features "${NEAR_DUPLICATE_MAX_QUERY_FEATURES:-64}" \
  --near_duplicate_max_candidates_per_row "${NEAR_DUPLICATE_MAX_CANDIDATES_PER_ROW:-128}" \
  --fail_on_near_duplicate \
  --fail_on_leakage \
  --require_structured_current_state \
  | tee "${OUTPUT_ROOT}/clean_preflight_stdout.json"
