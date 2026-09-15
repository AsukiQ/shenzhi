#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1

export PYTHONUNBUFFERED=1
export APPWORLD_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root
export APPWORLD_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.venvs/appworld/bin/python scripts/run_appworld_adapter_smoke.py \
  --appworld_root "${APPWORLD_ROOT}" \
  --appworld_cache "${APPWORLD_CACHE}" \
  --output_dir outputs/appworld_adapter_smoke
