#!/bin/bash
#SBATCH --time=01:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}

cd "${PROJECT_ROOT}"

SOURCE_ROOTS=${SOURCE_ROOTS:-data/toolret_training,data/toolbench_g3}
OUTPUT_DIR=${OUTPUT_DIR:-data/bge_toolrex_unified}
MAX_ROWS=${MAX_ROWS:-}
SEED=${SEED:-13}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/build_bge_toolrex_training_data.py
  --source_roots "${SOURCE_ROOTS}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)

if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/stdout.log"

