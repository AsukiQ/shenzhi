#!/bin/bash
#SBATCH --job-name=clstr_vnext_bundle
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH must name the freshly audited causal-view manifest}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must name a fresh smoke-bundle directory}

if [[ ! -f "${DATA_MANIFEST_PATH}" ]]; then
  printf 'ERROR: missing vNext data manifest: %s\n' "${DATA_MANIFEST_PATH}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: smoke bundle output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

required_benchmark_args=()
if [[ -n "${REQUIRED_BENCHMARKS:-}" ]]; then
  read -r -a required_benchmarks <<< "${REQUIRED_BENCHMARKS}"
  required_benchmark_args=(--required_benchmarks "${required_benchmarks[@]}")
fi

"${PYTHON_BIN}" scripts/build_clstr_vnext_smoke_bundle.py \
  --data_manifest_path "${DATA_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --maximum_skills "${MAXIMUM_SKILLS:-768}" \
  --stage0_rows_per_kind "${STAGE0_ROWS_PER_KIND:-32}" \
  --stage0_dev_rows_per_kind "${STAGE0_DEV_ROWS_PER_KIND:-32}" \
  --trajectory_limit "${TRAJECTORY_LIMIT:-8}" \
  --minimum_trajectory_length "${MINIMUM_TRAJECTORY_LENGTH:-3}" \
  --pair_limit "${PAIR_LIMIT:-4}" \
  --negatives_per_catalog "${NEGATIVES_PER_CATALOG:-8}" \
  "${required_benchmark_args[@]}"
