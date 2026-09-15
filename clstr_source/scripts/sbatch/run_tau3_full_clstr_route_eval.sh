#!/bin/bash
#SBATCH --job-name=tau3_clstr_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench/data/tau2}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/checkpoints/clstr_unified_retrieval_v2-step300.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_trainl025_calib_full/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-outputs/clstr_unified_stage4_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_l050_joint_act/checkpoints/clstr_stage4_act-step2000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/tau3_full_clstr_route_eval/smoke_${SLURM_JOB_ID:-manual}}

DOMAINS=${DOMAINS:-airline,retail,telecom,banking_knowledge}
MAX_TASKS_PER_DOMAIN=${MAX_TASKS_PER_DOMAIN:-20}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
STAGE0_CANDIDATE_BATCH_SIZE=${STAGE0_CANDIDATE_BATCH_SIZE:-16}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.5}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_tau3_full_clstr_route_eval.py
  --data_root "${DATA_ROOT}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --domains "${DOMAINS}"
  --batch_size "${BATCH_SIZE}"
  --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --route_scorer "${ROUTE_SCORER}"
)

if [[ -n "${MAX_TASKS_PER_DOMAIN}" && "${MAX_TASKS_PER_DOMAIN}" != "ALL" ]]; then
  args+=(--max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ -n "${CANDIDATE_COUNT:-}" && "${CANDIDATE_COUNT}" != "ALL" ]]; then
  args+=(--candidate_count "${CANDIDATE_COUNT}")
fi

printf '[tau3-clstr-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[tau3-clstr-eval] route_scorer=%s\n' "${ROUTE_SCORER}"
printf '[tau3-clstr-eval] domains=%s max_tasks_per_domain=%s max_eval_rows=%s\n' "${DOMAINS}" "${MAX_TASKS_PER_DOMAIN:-ALL}" "${MAX_EVAL_ROWS:-ALL}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
