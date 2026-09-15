#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.jsonl}
REPORT_PATH=${REPORT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_report.json}
REVIEW_OUTPUT_PATH=${REVIEW_OUTPUT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.reviewed.jsonl}
REVIEW_REPORT_PATH=${REVIEW_REPORT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json}
MAX_CANDIDATES=${MAX_CANDIDATES:-20000}
MAX_BUCKET_SIZE=${MAX_BUCKET_SIZE:-500}

"${PYTHON_BIN}" scripts/audit_skill_dedup_borderline.py \
  --skill_pool_path "${SKILL_POOL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --report_path "${REPORT_PATH}" \
  --review_output_path "${REVIEW_OUTPUT_PATH}" \
  --review_report_path "${REVIEW_REPORT_PATH}" \
  --max_candidates "${MAX_CANDIDATES}" \
  --max_bucket_size "${MAX_BUCKET_SIZE}"
