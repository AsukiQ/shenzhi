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
RAW_ANSWER_PATH=${RAW_ANSWER_PATH:-outputs/toolbench_g3/stabletoolbench_raw_answers}
CONVERTED_ANSWER_PATH=${CONVERTED_ANSWER_PATH:-outputs/toolbench_g3/stabletoolbench_converted}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}
TEST_SET=${TEST_SET:-G3_instruction}
METHOD=${METHOD:-CLSTR@1}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-outputs/toolbench_g3/stabletoolbench_answer_conversion_readiness.json}
RUN_CONVERT=${RUN_CONVERT:-1}

ARGS=(
  scripts/convert_stabletoolbench_answers.py
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}"
  --raw_answer_path "${RAW_ANSWER_PATH}"
  --converted_answer_path "${CONVERTED_ANSWER_PATH}"
  --candidate_model "${CANDIDATE_MODEL}"
  --test_set "${TEST_SET}"
  --method "${METHOD}"
  --command_root "${PROJECT_ROOT}"
  --output_path "${AUDIT_OUTPUT}"
  --fail_on_action_required
)

if [[ "${RUN_CONVERT}" = "1" ]]; then
  ARGS+=(--run_convert)
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${AUDIT_OUTPUT}.stdout.json"
