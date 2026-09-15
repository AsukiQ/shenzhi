#!/bin/bash
#SBATCH --job-name=q06_cmc_direct_utility
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${REQUESTED_PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=
cd "${PROJECT_ROOT}"

DYNAMIC_SELECTION_PATH=${DYNAMIC_SELECTION_PATH:-outputs/qwen06_clstr_postfix/stage4_cmc_full/stage4_dynamic_selection.json}
ROUTE_MANIFEST_PATH=${ROUTE_MANIFEST_PATH:-outputs/qwen06_clstr_postfix/stage4_cmc_full/validation/step3000.route_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen06_clstr_postfix/cmc_direct_utility/recalibration}

for path in "${DYNAMIC_SELECTION_PATH}" "${ROUTE_MANIFEST_PATH}"; do
  if [[ ! -s "${path}" ]]; then
    printf 'ERROR: missing direct-utility input: %s\n' "${path}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" scripts/run_qwen06_cmc_direct_utility_recalibration.py \
  --dynamic_selection_path "${DYNAMIC_SELECTION_PATH}" \
  --route_manifest_path "${ROUTE_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --bootstrap_samples 2000 \
  --temperature 0.5 \
  --fused_rank_weight 1.0 \
  --static_no_regret_weight 1.0 \
  --direct_gate_bce_weight 1.0 \
  --max_steps 300 \
  --learning_rate 0.01 \
  --seed 17
