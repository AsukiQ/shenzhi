#!/bin/bash
#SBATCH --job-name=tb_g3_clstr_full_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-}
TRAIN_TRAJECTORIES=${TRAIN_TRAJECTORIES:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/train_trajectories.jsonl}
EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3_official_skillrouter_comparison/clstr_full_route_eval_smoke}

STATIC_K=${STATIC_K:-${TOP_M:-350}}
DYNAMIC_EXTRA_K=${DYNAMIC_EXTRA_K:-64}
FINAL_K=${FINAL_K:-${CANDIDATE_COUNT:-64}}
BATCH_SIZE=${BATCH_SIZE:-8}
FEEDBACK_MODE=${FEEDBACK_MODE:-eval_prefix}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
RELIABILITY_MODE=${RELIABILITY_MODE:-dynamic}
FIXED_ALPHA=${FIXED_ALPHA:-1.0}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-256}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-128}
ROUTE_RECORDS_PATH=${ROUTE_RECORDS_PATH:-}
ROUTE_RECORD_MANIFEST_PATH=${ROUTE_RECORD_MANIFEST_PATH:-}
ROUTE_RECORD_MODEL_DIGEST=${ROUTE_RECORD_MODEL_DIGEST:-}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_toolbench_g3_full_clstr_route_eval.py
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --train_trajectories_path "${TRAIN_TRAJECTORIES}"
  --eval_trajectories_path "${EVAL_TRAJECTORIES}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --static_k "${STATIC_K}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K}"
  --final_k "${FINAL_K}"
  --batch_size "${BATCH_SIZE}"
  --feedback_mode "${FEEDBACK_MODE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --route_scorer "${ROUTE_SCORER}"
  --reliability_mode "${RELIABILITY_MODE}"
  --fixed_alpha "${FIXED_ALPHA}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
)
if [[ -n "${STAGE4_CHECKPOINT}" ]]; then
  args+=(--stage4_checkpoint_path "${STAGE4_CHECKPOINT}")
fi

if [[ -n "${MAX_TRAIN_ROWS}" && "${MAX_TRAIN_ROWS}" != "ALL" ]]; then
  args+=(--max_train_rows "${MAX_TRAIN_ROWS}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
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

printf '[clstr-full-route-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[clstr-full-route-eval] route_scorer=%s\n' "${ROUTE_SCORER}"
printf '[clstr-full-route-eval] reliability_mode=%s fixed_alpha=%s\n' "${RELIABILITY_MODE}" "${FIXED_ALPHA}"
printf '[clstr-full-route-eval] feedback_mode=%s memory=%s weight=%s\n' "${FEEDBACK_MODE}" "${ONLINE_MEMORY_MODE}" "${ONLINE_MEMORY_WEIGHT}"
printf '[clstr-full-route-eval] transition_inventory=%s min=%s\n' "${TRANSITION_INVENTORY_MASK_MODE}" "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
printf '[clstr-full-route-eval] static_k=%s dynamic_extra_k=%s final_k=%s\n' "${STATIC_K}" "${DYNAMIC_EXTRA_K}" "${FINAL_K}"
printf '[clstr-full-route-eval] stage4_checkpoint=%s\n' "${STAGE4_CHECKPOINT:-none}"
printf '[clstr-full-route-eval] max_train_rows=%s max_eval_rows=%s\n' "${MAX_TRAIN_ROWS:-ALL}" "${MAX_EVAL_ROWS:-ALL}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
