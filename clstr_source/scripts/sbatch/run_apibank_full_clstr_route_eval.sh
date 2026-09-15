#!/bin/bash
#SBATCH --job-name=apibank_clstr_eval
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

DATA_ROOT=${DATA_ROOT:-.tmp/benchmark_probe_direct/liminghao1630__API-Bank}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apibank_full_clstr_route_eval/smoke_${SLURM_JOB_ID:-manual}}
FILES=${FILES:-}
PREBUILT_SOURCE_ROWS_PATH=${PREBUILT_SOURCE_ROWS_PATH:-}
PREBUILT_SKILLS_PATH=${PREBUILT_SKILLS_PATH:-}
INCLUDE_TRIVIAL=${INCLUDE_TRIVIAL:-0}
MAX_ROWS_PER_FILE=${MAX_ROWS_PER_FILE:-ALL}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
STAGE0_CANDIDATE_BATCH_SIZE=${STAGE0_CANDIDATE_BATCH_SIZE:-16}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_apibank_full_clstr_route_eval.py
  --data_root "${DATA_ROOT}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --route_scorer "${ROUTE_SCORER}"
)
if [[ -n "${STAGE4_CHECKPOINT}" ]]; then
  args+=(--stage4_checkpoint_path "${STAGE4_CHECKPOINT}")
fi
if [[ -n "${FILES}" ]]; then
  args+=(--files "${FILES}")
fi
if [[ -n "${PREBUILT_SOURCE_ROWS_PATH}" ]]; then
  args+=(--prebuilt_source_rows_path "${PREBUILT_SOURCE_ROWS_PATH}")
fi
if [[ -n "${PREBUILT_SKILLS_PATH}" ]]; then
  args+=(--prebuilt_skills_path "${PREBUILT_SKILLS_PATH}")
fi
if [[ "${INCLUDE_TRIVIAL}" == "1" || "${INCLUDE_TRIVIAL}" == "true" ]]; then
  args+=(--include_trivial)
fi
if [[ -n "${MAX_ROWS_PER_FILE}" && "${MAX_ROWS_PER_FILE}" != "ALL" ]]; then
  args+=(--max_rows_per_file "${MAX_ROWS_PER_FILE}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[apibank-clstr-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[apibank-clstr-eval] stage4_checkpoint=%s\n' "${STAGE4_CHECKPOINT:-none}"
printf '[apibank-clstr-eval] max_rows_per_file=%s max_eval_rows=%s include_trivial=%s\n' "${MAX_ROWS_PER_FILE:-ALL}" "${MAX_EVAL_ROWS:-ALL}" "${INCLUDE_TRIVIAL}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
