#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python-only data import"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-data/toolret_eval}
SPLIT=${SPLIT:-test}
MAX_QUERIES_ARG=()
MAX_TOOLS_ARG=()
QUERY_ARG=()
TOOL_ARG=()

if [[ -n "${MAX_QUERIES:-}" ]]; then
  MAX_QUERIES_ARG=(--max_queries "${MAX_QUERIES}")
fi

if [[ -n "${MAX_TOOLS:-}" ]]; then
  MAX_TOOLS_ARG=(--max_tools "${MAX_TOOLS}")
fi

if [[ -n "${QUERY_SOURCE:-}" ]]; then
  QUERY_ARG=(--query_source "${QUERY_SOURCE}")
elif [[ "${ALLOW_HF_STREAMING:-0}" != "1" ]]; then
  echo "ERROR: QUERY_SOURCE is required on compute nodes because they cannot access the internet."
  echo "Download ToolRet-Queries on the login node first, then submit with:"
  echo "  QUERY_SOURCE=/path/to/toolret_queries.jsonl TOOL_SOURCE=/path/to/toolret_tools.jsonl sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolret_eval.sh"
  echo "For login-node smoke only, run scripts/import_toolret_eval.py directly or set ALLOW_HF_STREAMING=1."
  exit 2
fi

if [[ -n "${TOOL_SOURCE:-}" ]]; then
  TOOL_ARG=(--tool_source "${TOOL_SOURCE}")
elif [[ "${ALLOW_HF_STREAMING:-0}" != "1" ]]; then
  echo "ERROR: TOOL_SOURCE is required on compute nodes because they cannot access the internet."
  echo "Download ToolRet-Tools on the login node first, then submit with:"
  echo "  QUERY_SOURCE=/path/to/toolret_queries.jsonl TOOL_SOURCE=/path/to/toolret_tools.jsonl sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolret_eval.sh"
  echo "For login-node smoke only, run scripts/import_toolret_eval.py directly or set ALLOW_HF_STREAMING=1."
  exit 2
fi

"${PYTHON_BIN}" scripts/import_toolret_eval.py \
  "${QUERY_ARG[@]}" \
  "${TOOL_ARG[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  "${MAX_QUERIES_ARG[@]}" \
  "${MAX_TOOLS_ARG[@]}"
