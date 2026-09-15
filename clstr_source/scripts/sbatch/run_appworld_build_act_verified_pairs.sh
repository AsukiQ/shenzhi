#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1

export PYTHONUNBUFFERED=1
export APPWORLD_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root
export APPWORLD_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/train_tasks.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
OUTPUT_JSONL=${OUTPUT_JSONL:-data/appworld_act/verified_pairs_train.jsonl}
MANIFEST_PATH=${MANIFEST_PATH:-data/appworld_act/manifest.json}
TOP_K=${TOP_K:-20}
MAX_TASKS=${MAX_TASKS:-}

ARGS=(
  --appworld_root "${APPWORLD_ROOT}"
  --tasks_path "${TASKS_PATH}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --output_jsonl "${OUTPUT_JSONL}"
  --manifest_path "${MANIFEST_PATH}"
  --top_k "${TOP_K}"
)

if [ -n "${MAX_TASKS}" ]; then
  ARGS+=(--max_tasks "${MAX_TASKS}")
fi

"${PYTHON_BIN}" scripts/build_appworld_act_verified_pairs.py "${ARGS[@]}"
