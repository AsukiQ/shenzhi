#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export APPWORLD_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.venvs/appworld/bin/python}

"${PYTHON_BIN}" scripts/build_appworld_routing_data.py \
  --appworld_root "${APPWORLD_ROOT}" \
  --skill_pool_path data/appworld_skill_pool/skill_pool.jsonl \
  --output_dir data/appworld_routing \
  --splits train dev test_normal test_challenge \
  --positives_per_task 5

"${PYTHON_BIN}" scripts/run_appworld_skillrouter_baseline.py \
  --tasks_path data/appworld_routing/dev_tasks.jsonl \
  --qrels_path data/appworld_routing/dev_qrels.jsonl \
  --skill_pool_path data/appworld_skill_pool/skill_pool.jsonl \
  --output_dir outputs/appworld_skillrouter_baseline/dev \
  --top_k 20
