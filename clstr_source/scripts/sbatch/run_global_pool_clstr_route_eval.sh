#!/bin/bash
#SBATCH --job-name=clstr_global_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
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

BENCHMARK=${BENCHMARK:-tau2}
RUN_TAG=${RUN_TAG:-belief_fix_global_pool_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${BENCHMARK}_global_pool_clstr_route_eval/${RUN_TAG}}

BASE_SKILLS_PATH=${BASE_SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_stage12_belief_seed_fullsupport_rankprior_full_a800_20260706_004234/stage2_full_base/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-outputs/clstr_stage4_belief_seed_fullsupport_rankprior_full_a800_20260706_from_s12/checkpoints/clstr_stage4_act-step2000.pt}

MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-ALL}
STATIC_K=${STATIC_K:-${STAGE0_TOP_M:-500}}
DYNAMIC_EXTRA_K=${DYNAMIC_EXTRA_K:-64}
FINAL_K=${FINAL_K:-${CANDIDATE_COUNT:-64}}
BATCH_SIZE=${BATCH_SIZE:-8}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-8}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
RELIABILITY_MODE=${RELIABILITY_MODE:-dynamic}
FIXED_ALPHA=${FIXED_ALPHA:-1.0}
PREBUILT_SOURCE_ROWS_PATH=${PREBUILT_SOURCE_ROWS_PATH:-}
PREBUILT_SKILLS_PATH=${PREBUILT_SKILLS_PATH:-}
ROUTE_RECORDS_PATH=${ROUTE_RECORDS_PATH:-}
ROUTE_RECORD_MANIFEST_PATH=${ROUTE_RECORD_MANIFEST_PATH:-}
ROUTE_RECORD_MODEL_DIGEST=${ROUTE_RECORD_MODEL_DIGEST:-}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_global_pool_clstr_route_eval.py
  --benchmark "${BENCHMARK}"
  --base_skills_path "${BASE_SKILLS_PATH}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --static_k "${STATIC_K}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K}"
  --final_k "${FINAL_K}"
  --batch_size "${BATCH_SIZE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --route_scorer "${ROUTE_SCORER}"
  --reliability_mode "${RELIABILITY_MODE}"
  --fixed_alpha "${FIXED_ALPHA}"
)

if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ "${BENCHMARK}" == "tau2" ]]; then
  DOMAINS=${DOMAINS:-airline,retail,telecom}
  args+=(--domains "${DOMAINS}")
  if [[ -n "${MAX_TASKS_PER_DOMAIN:-}" && "${MAX_TASKS_PER_DOMAIN}" != "ALL" ]]; then
    args+=(--max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}")
  fi
elif [[ "${BENCHMARK}" == "toolsandbox" ]]; then
  if [[ -n "${MAX_SCENARIOS:-}" && "${MAX_SCENARIOS}" != "ALL" ]]; then
    args+=(--max_scenarios "${MAX_SCENARIOS}")
  fi
elif [[ "${BENCHMARK}" == "bfcl" ]]; then
  if [[ -n "${CATEGORIES:-}" ]]; then
    args+=(--categories "${CATEGORIES}")
  fi
  if [[ -n "${MAX_ROWS_PER_CATEGORY:-}" && "${MAX_ROWS_PER_CATEGORY}" != "ALL" ]]; then
    args+=(--max_rows_per_category "${MAX_ROWS_PER_CATEGORY}")
  fi
elif [[ "${BENCHMARK}" == "apibank" ]]; then
  if [[ -n "${FILES:-}" ]]; then
    args+=(--files "${FILES}")
  fi
  if [[ -n "${MAX_ROWS_PER_FILE:-}" && "${MAX_ROWS_PER_FILE}" != "ALL" ]]; then
    args+=(--max_rows_per_file "${MAX_ROWS_PER_FILE}")
  fi
fi

if [[ "${INCLUDE_TRIVIAL:-0}" == "1" || "${INCLUDE_TRIVIAL:-0}" == "true" ]]; then
  args+=(--include_trivial)
fi
if [[ -n "${PREBUILT_SOURCE_ROWS_PATH}" ]]; then
  args+=(--prebuilt_source_rows_path "${PREBUILT_SOURCE_ROWS_PATH}")
fi
if [[ -n "${PREBUILT_SKILLS_PATH}" ]]; then
  args+=(--prebuilt_skills_path "${PREBUILT_SKILLS_PATH}")
fi
if [[ -n "${ROUTE_RECORDS_PATH}" ]]; then
  args+=(--route_records_path "${ROUTE_RECORDS_PATH}")
fi
if [[ -n "${ROUTE_RECORD_MANIFEST_PATH}" ]]; then
  args+=(--route_record_manifest_path "${ROUTE_RECORD_MANIFEST_PATH}")
fi
if [[ -n "${ROUTE_RECORD_MODEL_DIGEST}" ]]; then
  args+=(--route_record_model_digest "${ROUTE_RECORD_MODEL_DIGEST}")
fi

printf '[global-clstr-eval] benchmark=%s output_dir=%s\n' "${BENCHMARK}" "${OUTPUT_DIR}"
printf '[global-clstr-eval] route_scorer=%s\n' "${ROUTE_SCORER}"
printf '[global-clstr-eval] reliability_mode=%s fixed_alpha=%s query_mode=%s\n' "${RELIABILITY_MODE}" "${FIXED_ALPHA}" "${STAGE0_HANDOFF_QUERY_MODE}"
printf '[global-clstr-eval] prebuilt_source_rows_path=%s prebuilt_skills_path=%s\n' "${PREBUILT_SOURCE_ROWS_PATH}" "${PREBUILT_SKILLS_PATH}"
printf '[global-clstr-eval] static_k=%s dynamic_extra_k=%s final_k=%s max_eval_rows=%s\n' "${STATIC_K}" "${DYNAMIC_EXTRA_K}" "${FINAL_K}" "${MAX_EVAL_ROWS}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
