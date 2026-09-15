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
TOP_K=${TOP_K:-50}
ORIGINAL_QUERY_FILE=${ORIGINAL_QUERY_FILE:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_instruction/${TEST_SET}.json}
SKILLS_PATH=${SKILLS_PATH:-data/toolbench_g3/skills.jsonl}
RUN_PATH=${RUN_PATH:-outputs/toolbench_g3/stabletoolbench_solvable_clstr_retrieval/run.tsv}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3/stabletoolbench_clstr_topk}
OUTPUT_QUERY_FILE=${OUTPUT_QUERY_FILE:-${OUTPUT_DIR}/${TEST_SET}.json}
REPORT_PATH=${REPORT_PATH:-${OUTPUT_DIR}/${TEST_SET}_build_report.json}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/build_stabletoolbench_clstr_queries.py \
  --original_query_file "${ORIGINAL_QUERY_FILE}" \
  --skills_path "${SKILLS_PATH}" \
  --run_path "${RUN_PATH}" \
  --output_query_file "${OUTPUT_QUERY_FILE}" \
  --report_path "${REPORT_PATH}" \
  --top_k "${TOP_K}" \
  --fail_on_action_required | tee "${REPORT_PATH}.stdout.json"
