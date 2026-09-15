#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python-only data import"

export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-data/toolbench_g3}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-outputs/toolbench_g3/toolbench_g3_audit.json}
RAW_VERIFY_OUTPUT=${RAW_VERIFY_OUTPUT:-outputs/toolbench_g3/toolbench_g3_raw_verify.json}
EXPECTED_G3_ANSWER_FILES=${EXPECTED_G3_ANSWER_FILES:-5000}
MAX_QUERIES_ARG=()

if [[ -z "${SOURCE_ROOT:-}" ]]; then
  echo "ERROR: SOURCE_ROOT is required. Point it to the downloaded ToolBench/data directory."
  echo "Example:"
  echo "  SOURCE_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolbench_g3.sh"
  exit 2
fi

if [[ -n "${MAX_QUERIES:-}" ]]; then
  MAX_QUERIES_ARG=(--max_queries "${MAX_QUERIES}")
fi

RAW_VERIFY_EXTRACT_ROOT="${SOURCE_ROOT}"
if [[ -f "${SOURCE_ROOT}/instruction/G3_query.json" ]]; then
  RAW_VERIFY_EXTRACT_ROOT="${SOURCE_ROOT}/.."
fi

"${PYTHON_BIN}" scripts/download_toolbench_data.py \
  --verify_only \
  --extract_root "${RAW_VERIFY_EXTRACT_ROOT}" \
  --manifest_path "${RAW_VERIFY_OUTPUT}" \
  --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}"

"${PYTHON_BIN}" scripts/import_toolbench_g3.py \
  --source_root "${SOURCE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  "${MAX_QUERIES_ARG[@]}"

"${PYTHON_BIN}" scripts/audit_toolbench_g3_data.py \
  --data_dir "${OUTPUT_DIR}" \
  --source_root "${SOURCE_ROOT}" \
  --output_path "${AUDIT_OUTPUT}" \
  --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}" \
  --fail_on_action_required
