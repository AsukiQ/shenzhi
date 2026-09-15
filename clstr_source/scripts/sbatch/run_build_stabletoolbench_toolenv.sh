#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
TEST_SET=${TEST_SET:-G3_instruction}
QUERY_FILE=${QUERY_FILE:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_instruction/${TEST_SET}.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${STABLETOOLBENCH_ROOT}/toolenv/tools}
MANIFEST_PATH=${MANIFEST_PATH:-${STABLETOOLBENCH_ROOT}/toolenv/toolenv_${TEST_SET}_manifest.json}

"${PYTHON_BIN}" scripts/build_stabletoolbench_toolenv.py \
  --query_file "${QUERY_FILE}" \
  --output_root "${OUTPUT_ROOT}" \
  --manifest_path "${MANIFEST_PATH}" \
  --fail_on_action_required | tee "${MANIFEST_PATH}.stdout.json"
