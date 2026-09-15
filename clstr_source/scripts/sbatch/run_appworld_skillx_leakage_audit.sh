#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment"

export PYTHONUNBUFFERED=1

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
SKILLX_ROOT=${SKILLX_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/SkillX}
ROUTING_DIR=${ROUTING_DIR:-data/appworld_routing}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/skillx_appworld_leakage_audit}
NEAR_DUPLICATE_THRESHOLD=${NEAR_DUPLICATE_THRESHOLD:-0.8}

"${PYTHON_BIN}" scripts/audit_skillx_appworld.py \
  --leakage_audit \
  --skillx_root "${SKILLX_ROOT}" \
  --skill_pool_path "${SKILL_POOL_PATH}" \
  --routing_dir "${ROUTING_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --near_duplicate_threshold "${NEAR_DUPLICATE_THRESHOLD}"
