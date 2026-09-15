#!/bin/bash
#SBATCH --time=06:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}
STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-auto}
STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}
STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}
STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-skill_prior_plus_action_observation_residual}
ROUTE_SCORER=${ROUTE_SCORER:-legacy_prior_residual}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
MAX_STEPS=${MAX_STEPS:-3000}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.7}
TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.2}
TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.5}
BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}
EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
MAX_ROWS=${MAX_ROWS:-}
INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1}
# Use colon-separated BENCHMARK_CAPS when exporting through sbatch, because
# Slurm splits --export assignments on commas before the script receives them.
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}

if [[ -s "${ROUTING_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage1 requires completed Stage0 checkpoint: ${ROUTING_CHECKPOINT_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/run_clstr_stage1_heads_init.py
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --policy_loss_weight "${POLICY_LOSS_WEIGHT}"
  --transition_loss_weight "${TRANSITION_LOSS_WEIGHT}"
  --transition_skill_ce_loss_weight "${TRANSITION_SKILL_CE_LOSS_WEIGHT}"
  --belief_loss_weight "${BELIEF_LOSS_WEIGHT}"
  --embedding_cache_mode "${EMBEDDING_CACHE_MODE}"
  --embedding_cache_max_rows "${EMBEDDING_CACHE_MAX_ROWS}"
  --stage0_top_m "${STAGE0_TOP_M}"
  --stage0_positive_missing_policy "${STAGE0_POSITIVE_MISSING_POLICY}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  --stage0_handoff_cache_mode "${STAGE0_HANDOFF_CACHE_MODE}"
  --stage0_handoff_cache_dir "${STAGE0_HANDOFF_CACHE_DIR}"
  --stage0_handoff_cache_format "${STAGE0_HANDOFF_CACHE_FORMAT}"
  --stage0_handoff_cache_shard_size "${STAGE0_HANDOFF_CACHE_SHARD_SIZE}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
  --transition_loss_type "${TRANSITION_LOSS_TYPE}"
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --route_scorer "${ROUTE_SCORER}"
  --sampling_strategy "${SAMPLING_STRATEGY}"
)

if [[ -n "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}" ]]; then
  ARGS+=(--stage0_handoff_sample_multiplier "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}")
fi

if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

if [[ -n "${ALLOWED_BENCHMARKS}" ]]; then
  ARGS+=(--allowed_benchmarks "${ALLOWED_BENCHMARKS}")
fi

if [[ -n "${BENCHMARK_CAPS}" ]]; then
  ARGS+=(--benchmark_caps "${BENCHMARK_CAPS}")
fi

if [ "${INCLUDE_AVAILABLE_ACTIONS_IN_STATE}" = "1" ]; then
  ARGS+=(--include_available_actions_in_state)
fi

if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  ARGS+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/train_stdout.json"
