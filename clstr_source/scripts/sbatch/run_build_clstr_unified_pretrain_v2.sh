#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python-only data build"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-data/clstr_unified_pretrain_v4_2_progressive_final}
SCHEMA_VERSION=${SCHEMA_VERSION:-v3_trajectory_retrieval_caps}
TRAJECTORY_RETRIEVAL_CAPS=${TRAJECTORY_RETRIEVAL_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000}

"${PYTHON_BIN}" scripts/build_clstr_unified_pretrain.py \
  --repo_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr \
  --output_dir "${OUTPUT_DIR}" \
  --schema_version "${SCHEMA_VERSION}" \
  --trajectory_retrieval_caps "${TRAJECTORY_RETRIEVAL_CAPS}"
