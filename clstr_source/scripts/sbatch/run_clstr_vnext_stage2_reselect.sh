#!/bin/bash
#SBATCH --job-name=clstr_vnext_s2_select
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must identify a completed Stage2 output}
SELECTION_MODE=${SELECTION_MODE:-training}
if [[ "${SELECTION_MODE}" == "open_pool_full_pool" ]]; then
  RELEASE_OUTPUT_DIR=${RELEASE_OUTPUT_DIR:?RELEASE_OUTPUT_DIR is required for open-pool selection}
  "${PYTHON_BIN}" scripts/reselect_clstr_vnext_stage2.py \
    "${OUTPUT_DIR}" \
    --mode open_pool_full_pool \
    --release_output_dir "${RELEASE_OUTPUT_DIR}"
elif [[ "${SELECTION_MODE}" == "training" ]]; then
  "${PYTHON_BIN}" scripts/reselect_clstr_vnext_stage2.py "${OUTPUT_DIR}"
else
  printf 'ERROR: unsupported SELECTION_MODE: %s\n' "${SELECTION_MODE}" >&2
  exit 2
fi
