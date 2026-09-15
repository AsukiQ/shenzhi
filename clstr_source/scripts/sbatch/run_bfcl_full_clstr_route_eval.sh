#!/bin/bash
#SBATCH --job-name=bfcl_clstr_eval
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

DATA_ROOT=${DATA_ROOT:-.tmp/benchmark_probe_direct/gorilla-llm__Berkeley-Function-Calling-Leaderboard}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bfcl_full_clstr_route_eval/smoke_${SLURM_JOB_ID:-manual}}
CATEGORIES=${CATEGORIES:-BFCL_v3_multiple,BFCL_v3_parallel_multiple,BFCL_v3_multi_turn_base}
MAX_ROWS_PER_CATEGORY=${MAX_ROWS_PER_CATEGORY:-5}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
STAGE0_CANDIDATE_BATCH_SIZE=${STAGE0_CANDIDATE_BATCH_SIZE:-16}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_bfcl_full_clstr_route_eval.py
  --data_root "${DATA_ROOT}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --categories "${CATEGORIES}"
  --batch_size "${BATCH_SIZE}"
  --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
)
if [[ -n "${STAGE4_CHECKPOINT}" ]]; then
  args+=(--stage4_checkpoint_path "${STAGE4_CHECKPOINT}")
fi
if [[ -n "${MAX_ROWS_PER_CATEGORY}" && "${MAX_ROWS_PER_CATEGORY}" != "ALL" ]]; then
  args+=(--max_rows_per_category "${MAX_ROWS_PER_CATEGORY}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[bfcl-clstr-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[bfcl-clstr-eval] categories=%s max_rows_per_category=%s max_eval_rows=%s\n' "${CATEGORIES}" "${MAX_ROWS_PER_CATEGORY:-ALL}" "${MAX_EVAL_ROWS:-ALL}"
printf '[bfcl-clstr-eval] stage4_checkpoint=%s\n' "${STAGE4_CHECKPOINT:-none}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
