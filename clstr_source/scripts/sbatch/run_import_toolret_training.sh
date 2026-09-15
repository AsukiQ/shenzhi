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
OUTPUT_DIR=${OUTPUT_DIR:-data/toolret_training}
MAX_ROWS_ARG=()
SOURCE_ARG=()

if [[ -n "${MAX_ROWS:-}" ]]; then
  MAX_ROWS_ARG=(--max_rows "${MAX_ROWS}")
fi

if [[ -n "${SOURCE_PATH:-}" ]]; then
  SOURCE_ARG=(--source_path "${SOURCE_PATH}")
elif [[ "${ALLOW_HF_STREAMING:-0}" != "1" ]]; then
  echo "ERROR: SOURCE_PATH is required on compute nodes because they cannot access the internet."
  echo "Download ToolRet-Training-20w on the login node first, then submit with:"
  echo "  SOURCE_PATH=/path/to/toolret_train.jsonl sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolret_training.sh"
  echo "For login-node smoke only, run scripts/import_toolret_training.py directly or set ALLOW_HF_STREAMING=1."
  exit 2
fi

"${PYTHON_BIN}" scripts/import_toolret_training.py \
  "${SOURCE_ARG[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  "${MAX_ROWS_ARG[@]}"
