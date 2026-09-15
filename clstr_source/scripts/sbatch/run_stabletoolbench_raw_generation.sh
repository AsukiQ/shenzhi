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

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
TOOL_ROOT_DIR=${TOOL_ROOT_DIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools}
RAW_ANSWER_PATH=${RAW_ANSWER_PATH:-outputs/toolbench_g3/stabletoolbench_raw_answers}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}
TEST_SET=${TEST_SET:-G3_instruction}
INPUT_QUERY_FILE=${INPUT_QUERY_FILE:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_instruction/${TEST_SET}.json}
METHOD=${METHOD:-CLSTR@1}
BACKBONE_MODEL=${BACKBONE_MODEL:-chatgpt_function}
CHATGPT_MODEL=${CHATGPT_MODEL:-gpt-4-turbo-2024-04-09}
BASE_URL=${BASE_URL:-https://api.openai.com/v1}
MODEL_PATH=${MODEL_PATH:-}
SERVICE_URL=${SERVICE_URL:-}
TOOLBENCH_KEY=${TOOLBENCH_KEY:-}
MAX_OBSERVATION_LENGTH=${MAX_OBSERVATION_LENGTH:-1024}
SINGLE_CHAIN_MAX_STEP=${SINGLE_CHAIN_MAX_STEP:-50}
MAX_QUERY_COUNT=${MAX_QUERY_COUNT:-200}
NUM_THREAD=${NUM_THREAD:-1}
RUN_GENERATION=${RUN_GENERATION:-0}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-outputs/toolbench_g3/stabletoolbench_raw_generation_readiness.json}

"${PYTHON_BIN}" scripts/audit_stabletoolbench_raw_generation.py \
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}" \
  --tool_root_dir "${TOOL_ROOT_DIR}" \
  --raw_answer_path "${RAW_ANSWER_PATH}" \
  --candidate_model "${CANDIDATE_MODEL}" \
  --test_set "${TEST_SET}" \
  --input_query_file "${INPUT_QUERY_FILE}" \
  --method "${METHOD}" \
  --backbone_model "${BACKBONE_MODEL}" \
  --model_path "${MODEL_PATH}" \
  --service_url "${SERVICE_URL}" \
  --toolbench_key "${TOOLBENCH_KEY}" \
  --base_url "${BASE_URL}" \
  --chatgpt_model "${CHATGPT_MODEL}" \
  --max_observation_length "${MAX_OBSERVATION_LENGTH}" \
  --single_chain_max_step "${SINGLE_CHAIN_MAX_STEP}" \
  --max_query_count "${MAX_QUERY_COUNT}" \
  --num_thread "${NUM_THREAD}" \
  --command_root "${PROJECT_ROOT}" \
  --output_path "${AUDIT_OUTPUT}" \
  --fail_on_action_required | tee "${AUDIT_OUTPUT}.stdout.json"

if [[ "${RUN_GENERATION}" != "1" ]]; then
  echo "RUN_GENERATION=${RUN_GENERATION}; audited readiness only."
  exit 0
fi

mkdir -p "${RAW_ANSWER_PATH}/${CANDIDATE_MODEL}/${TEST_SET}"
export SERVICE_URL
pushd "${STABLETOOLBENCH_ROOT}" >/dev/null
"${PYTHON_BIN}" toolbench/inference/qa_pipeline_multithread.py \
  --tool_root_dir "${TOOL_ROOT_DIR}" \
  --backbone_model "${BACKBONE_MODEL}" \
  --chatgpt_model "${CHATGPT_MODEL}" \
  --base_url "${BASE_URL}" \
  --openai_key "${OPENAI_KEY:-}" \
  --model_path "${MODEL_PATH}" \
  --max_observation_length "${MAX_OBSERVATION_LENGTH}" \
  --single_chain_max_step "${SINGLE_CHAIN_MAX_STEP}" \
  --max_query_count "${MAX_QUERY_COUNT}" \
  --method "${METHOD}" \
  --input_query_file "${INPUT_QUERY_FILE}" \
  --output_answer_file "${PROJECT_ROOT}/${RAW_ANSWER_PATH}/${CANDIDATE_MODEL}/${TEST_SET}" \
  --toolbench_key "${TOOLBENCH_KEY}" \
  --num_thread "${NUM_THREAD}" | tee "${PROJECT_ROOT}/${RAW_ANSWER_PATH}/${CANDIDATE_MODEL}/${TEST_SET}/generation_stdout.txt"
popd >/dev/null
