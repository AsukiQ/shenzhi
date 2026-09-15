#!/bin/bash
#SBATCH --job-name=clstr_e3_selector
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=00:45:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
ARTIFACT_RUN_ROOT=${ARTIFACT_RUN_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1}
E1_ROOT=${E1_ROOT:-${ARTIFACT_RUN_ROOT}/matched_history_e1_v1}
FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/checkpoints/clstr_vnext_stage2-step0.pt}
SKILLS_PATH=${SKILLS_PATH:-${ARTIFACT_RUN_ROOT}/stage0/selected_skills.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-${E1_ROOT}/controlled_tau2_e3_v1/selector_smoke_v1/report.json}

source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
"${PYTHON_BIN}" scripts/smoke_clstr_matched_history_online_e3.py \
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --e1_root "${E1_ROOT}" \
  --output_path "${OUTPUT_PATH}" \
  --device cuda
