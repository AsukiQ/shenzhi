#!/bin/bash
#SBATCH --job-name=clstr_belief_audit
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_belief_state_audit/clean_stage0_pre_fix/report.json}
SAMPLE_ROWS=${SAMPLE_ROWS:-512}
DEVICE=${DEVICE:-cuda}
FAIL_ON_ACTION_REQUIRED=${FAIL_ON_ACTION_REQUIRED:-0}

mkdir -p "$(dirname "${OUTPUT_PATH}")"

ARGS=(
  scripts/audit_clstr_belief_state.py
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --skills_path "${SKILLS_PATH}"
  --trajectories_path "${TRAJECTORIES_PATH}"
  --sample_rows "${SAMPLE_ROWS}"
  --device "${DEVICE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ -n "${HEAD_CHECKPOINT_PATH}" ]]; then
  ARGS+=(--head_checkpoint_path "${HEAD_CHECKPOINT_PATH}")
fi

if [[ "${FAIL_ON_ACTION_REQUIRED}" == "1" || "${FAIL_ON_ACTION_REQUIRED}" == "true" ]]; then
  ARGS+=(--fail_on_action_required)
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "$(dirname "${OUTPUT_PATH}")/audit_stdout.json"
