#!/bin/bash
#SBATCH --job-name=clstr_clean_preflight
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:30:00
set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}
TOOLBENCH_EVAL_TRAJECTORIES_PATH=${TOOLBENCH_EVAL_TRAJECTORIES_PATH:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
TRAJECT_EVAL_QUERIES_PATH=${TRAJECT_EVAL_QUERIES_PATH:-data/traject_eval_traject_split_test/queries.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_clean_training_preflight/belief_fix_preflight.json}
FAIL_ON_LEAKAGE=${FAIL_ON_LEAKAGE:-1}
FAIL_ON_NEAR_DUPLICATE=${FAIL_ON_NEAR_DUPLICATE:-0}
REQUIRE_STRUCTURED_CURRENT_STATE=${REQUIRE_STRUCTURED_CURRENT_STATE:-0}
NEAR_DUPLICATE_THRESHOLD=${NEAR_DUPLICATE_THRESHOLD:-0.55}
NEAR_DUPLICATE_MAX_CHARS=${NEAR_DUPLICATE_MAX_CHARS:-1000}
NEAR_DUPLICATE_MAX_SHINGLE_POSTINGS=${NEAR_DUPLICATE_MAX_SHINGLE_POSTINGS:-256}
NEAR_DUPLICATE_MAX_QUERY_FEATURES=${NEAR_DUPLICATE_MAX_QUERY_FEATURES:-64}
NEAR_DUPLICATE_MAX_CANDIDATES_PER_ROW=${NEAR_DUPLICATE_MAX_CANDIDATES_PER_ROW:-128}
TAU2_TEST_ROWS_PATH=${TAU2_TEST_ROWS_PATH:-}
TOOLSANDBOX_TEST_ROWS_PATH=${TOOLSANDBOX_TEST_ROWS_PATH:-}
BASE_DATA_ROOT=${BASE_DATA_ROOT:-}
BASE_CLEAN_PREFLIGHT_REPORT=${BASE_CLEAN_PREFLIGHT_REPORT:-}
MATCHED_UNION_MANIFEST_PATH=${MATCHED_UNION_MANIFEST_PATH:-}

mkdir -p "$(dirname "${OUTPUT_PATH}")"

ARGS=(
  scripts/audit_clean_training_preflight.py
  --data_root "${DATA_ROOT}"
  --toolbench_eval_trajectories_path "${TOOLBENCH_EVAL_TRAJECTORIES_PATH}"
  --traject_eval_queries_path "${TRAJECT_EVAL_QUERIES_PATH}"
  --output_path "${OUTPUT_PATH}"
  --near_duplicate_threshold "${NEAR_DUPLICATE_THRESHOLD}"
  --near_duplicate_max_chars "${NEAR_DUPLICATE_MAX_CHARS}"
  --near_duplicate_max_shingle_postings "${NEAR_DUPLICATE_MAX_SHINGLE_POSTINGS}"
  --near_duplicate_max_query_features "${NEAR_DUPLICATE_MAX_QUERY_FEATURES}"
  --near_duplicate_max_candidates_per_row "${NEAR_DUPLICATE_MAX_CANDIDATES_PER_ROW}"
)

if [[ "${FAIL_ON_LEAKAGE}" == "1" || "${FAIL_ON_LEAKAGE}" == "true" ]]; then
  ARGS+=(--fail_on_leakage)
fi
if [[ "${FAIL_ON_NEAR_DUPLICATE}" == "1" || "${FAIL_ON_NEAR_DUPLICATE}" == "true" ]]; then
  ARGS+=(--fail_on_near_duplicate)
fi
if [[ "${REQUIRE_STRUCTURED_CURRENT_STATE}" == "1" || "${REQUIRE_STRUCTURED_CURRENT_STATE}" == "true" ]]; then
  ARGS+=(--require_structured_current_state)
fi
if [[ -n "${TAU2_TEST_ROWS_PATH}" ]]; then
  ARGS+=(--tau2_test_rows_path "${TAU2_TEST_ROWS_PATH}")
fi
if [[ -n "${TOOLSANDBOX_TEST_ROWS_PATH}" ]]; then
  ARGS+=(--toolsandbox_test_rows_path "${TOOLSANDBOX_TEST_ROWS_PATH}")
fi
if [[ -n "${BASE_DATA_ROOT}" || -n "${BASE_CLEAN_PREFLIGHT_REPORT}" || -n "${MATCHED_UNION_MANIFEST_PATH}" ]]; then
  if [[ -z "${BASE_DATA_ROOT}" || -z "${BASE_CLEAN_PREFLIGHT_REPORT}" || -z "${MATCHED_UNION_MANIFEST_PATH}" ]]; then
    printf 'ERROR: incremental preflight requires base root/report and union manifest\n' >&2
    exit 2
  fi
  ARGS+=(
    --base_data_root "${BASE_DATA_ROOT}"
    --base_clean_preflight_report "${BASE_CLEAN_PREFLIGHT_REPORT}"
    --matched_union_manifest_path "${MATCHED_UNION_MANIFEST_PATH}"
  )
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_PATH}.stdout.json"
