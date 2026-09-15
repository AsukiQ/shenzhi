#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SOURCE_PROJECT_ROOT}}
cd "${PROJECT_ROOT}"
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
CONVERTED_ANSWER_PATH=${CONVERTED_ANSWER_PATH:-outputs/toolbench_g3/stabletoolbench_converted}
SAVE_PATH=${SAVE_PATH:-outputs/toolbench_g3/stabletoolbench_pass_rate}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}
TEST_SET=${TEST_SET:-G3_instruction}
TEST_IDS_DIR=${TEST_IDS_DIR:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_query_ids}
API_POOL_FILE=${API_POOL_FILE:-${STABLETOOLBENCH_ROOT}/openai_key.json}
EVALUATOR=${EVALUATOR:-tooleval_gpt-3.5-turbo_default}
MAX_EVAL_THREADS=${MAX_EVAL_THREADS:-1}
EVALUATE_TIMES=${EVALUATE_TIMES:-3}
RUN_PASS_RATE=${RUN_PASS_RATE:-1}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-${SAVE_PATH}/readiness.json}

project_abs() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${PROJECT_ROOT}" "${path}"
  fi
}

mkdir -p "${SAVE_PATH}/${CANDIDATE_MODEL}"
CONVERTED_ANSWER_PATH_ABS=$(project_abs "${CONVERTED_ANSWER_PATH}")
SAVE_PATH_ABS=$(project_abs "${SAVE_PATH}")
AUDIT_OUTPUT_ABS=$(project_abs "${AUDIT_OUTPUT}")

"${PYTHON_BIN}" scripts/audit_stabletoolbench_pass_rate.py \
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}" \
  --converted_answer_path "${CONVERTED_ANSWER_PATH_ABS}" \
  --api_pool_file "${API_POOL_FILE}" \
  --candidate_model "${CANDIDATE_MODEL}" \
  --test_set "${TEST_SET}" \
  --save_path "${SAVE_PATH_ABS}" \
  --test_ids_dir "${TEST_IDS_DIR}" \
  --evaluator "${EVALUATOR}" \
  --max_eval_threads "${MAX_EVAL_THREADS}" \
  --evaluate_times "${EVALUATE_TIMES}" \
  --command_root "${PROJECT_ROOT}" \
  --output_path "${AUDIT_OUTPUT_ABS}" \
  --fail_on_action_required | tee "${SAVE_PATH_ABS}/readiness_stdout.json"

if [[ "${RUN_PASS_RATE}" != "1" ]]; then
  echo "RUN_PASS_RATE=${RUN_PASS_RATE}; audited readiness only."
  exit 0
fi

export API_POOL_FILE
pushd "${STABLETOOLBENCH_ROOT}/toolbench/tooleval" >/dev/null
"${PYTHON_BIN}" eval_pass_rate.py \
  --converted_answer_path "${CONVERTED_ANSWER_PATH_ABS}" \
  --save_path "${SAVE_PATH_ABS}/${CANDIDATE_MODEL}" \
  --reference_model "${CANDIDATE_MODEL}" \
  --test_ids "${TEST_IDS_DIR}" \
  --evaluator "${EVALUATOR}" \
  --max_eval_threads "${MAX_EVAL_THREADS}" \
  --evaluate_times "${EVALUATE_TIMES}" \
  --test_set "${TEST_SET}" | tee "${SAVE_PATH_ABS}/${CANDIDATE_MODEL}/eval_pass_rate_stdout.txt"
popd >/dev/null
