#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_executor_comparison}

if [ "$#" -eq 0 ]; then
  echo "ERROR: pass one or more report.json paths as arguments" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/build_appworld_executor_comparison.py \
  --reports "$@" \
  --output_dir "${OUTPUT_DIR}"
