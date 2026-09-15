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
PUBLIC_DATA=${PUBLIC_DATA:-../TRAJECT-Bench/public_data}
OUTPUT_DIR=${OUTPUT_DIR:-data/traject_eval_traject_split_test}
SPLIT=${SPLIT:-test}
SPLIT_PARTITION=${SPLIT_PARTITION:-test}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-outputs/traject_eval_traject_split_test/traject_eval_audit.json}
MAX_QUERIES_ARG=()
TRAJECTORY_TYPES_ARG=()
DOMAINS_ARG=()
SPLIT_PARTITION_ARG=()

if [[ -n "${MAX_QUERIES:-}" ]]; then
  MAX_QUERIES_ARG=(--max_queries "${MAX_QUERIES}")
fi

if [[ -n "${TRAJECTORY_TYPES:-}" ]]; then
  read -r -a TRAJECTORY_TYPES_VALUES <<< "${TRAJECTORY_TYPES}"
  TRAJECTORY_TYPES_ARG=(--trajectory_types "${TRAJECTORY_TYPES_VALUES[@]}")
fi

if [[ -n "${DOMAINS:-}" ]]; then
  read -r -a DOMAINS_VALUES <<< "${DOMAINS}"
  DOMAINS_ARG=(--domains "${DOMAINS_VALUES[@]}")
fi

if [[ -n "${SPLIT_PARTITION:-}" ]]; then
  SPLIT_PARTITION_ARG=(--split_partition "${SPLIT_PARTITION}")
fi

if [[ ! -d "${PUBLIC_DATA}" ]]; then
  echo "ERROR: TRAJECT public_data directory not found: ${PUBLIC_DATA}"
  echo "Expected local data under /data/home/scyb713/run/xzf/AAAI/autodl-tmp, e.g. ../TRAJECT-Bench/public_data"
  exit 2
fi

"${PYTHON_BIN}" scripts/import_traject_eval.py \
  --public_data "${PUBLIC_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  "${SPLIT_PARTITION_ARG[@]}" \
  "${MAX_QUERIES_ARG[@]}" \
  "${TRAJECTORY_TYPES_ARG[@]}" \
  "${DOMAINS_ARG[@]}"

"${PYTHON_BIN}" scripts/audit_traject_eval_data.py \
  --data_dir "${OUTPUT_DIR}" \
  --output_path "${AUDIT_OUTPUT}" \
  --fail_on_action_required
