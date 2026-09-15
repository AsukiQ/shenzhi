#!/bin/bash
#SBATCH --job-name=clstr_vnext_views
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

required=(DATA_ROOT CLEAN_PREFLIGHT_REPORT OUTPUT_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is empty: %s\n' "${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: OUTPUT_DIR already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

args=(
  "${PYTHON_BIN}" scripts/build_clstr_vnext_views.py
  --skills_path "${DATA_ROOT}/skill_pool.jsonl"
  --retrieval_path "${DATA_ROOT}/retrieval.jsonl"
  --trajectories_path "${DATA_ROOT}/trajectories.jsonl"
  --clean_preflight_report "${CLEAN_PREFLIGHT_REPORT}"
  --output_dir "${OUTPUT_DIR}"
  --dev_fraction "${DEV_FRACTION:-0.1}"
  --dynamic_holdout_fraction "${DYNAMIC_HOLDOUT_FRACTION:-0.01}"
  --maximum_source_step_shortcut_accuracy "${MAXIMUM_SOURCE_STEP_SHORTCUT_ACCURACY:-0.8}"
  --minimum_shortcut_feature_coverage "${MINIMUM_SHORTCUT_FEATURE_COVERAGE:-0.5}"
)

if [[ -n "${MAX_TRAJECTORY_ROWS:-}" ]]; then
  args+=(--max_trajectory_rows "${MAX_TRAJECTORY_ROWS}")
fi
if [[ -n "${MAX_RETRIEVAL_ROWS:-}" ]]; then
  args+=(--max_retrieval_rows "${MAX_RETRIEVAL_ROWS}")
fi
if [[ "${REQUIRE_ROBUST_PREFIX_VIEWS:-0}" == "1" ]]; then
  args+=(--require_robust_prefix_views)
fi
if [[ "${REQUIRE_CAUSAL_OUTCOME_PAIRS:-0}" == "1" ]]; then
  args+=(--require_causal_outcome_pairs)
fi
if [[ "${REQUIRE_CAUSAL_BRANCH_PAIRS:-1}" != "1" ]]; then
  args+=(--allow_empty_causal_branch_pairs)
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
"${args[@]}" | tee "${OUTPUT_DIR}.stdout.json"
