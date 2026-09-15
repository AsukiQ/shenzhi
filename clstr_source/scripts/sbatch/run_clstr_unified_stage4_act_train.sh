#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_memory_stage4_full_pool_counterfactual}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_memory_stage0_full_b128_20260707_052137/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-outputs/clstr_unified_memory_stage12_full_mmargin_retry_20260707_080738/stage2_full_base/checkpoints/clstr_full_base-step10000.pt}
BASE_CMC_CHECKPOINT_PATH=${BASE_CMC_CHECKPOINT_PATH:-}
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-3.0e-5}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-3.0e-6}
LEARNING_RATE_WARMUP_FRACTION=${LEARNING_RATE_WARMUP_FRACTION:-0.05}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.10}
VALIDATION_ROWS_PER_BENCHMARK=${VALIDATION_ROWS_PER_BENCHMARK:-256}
MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${MINIMUM_VALIDATION_ROWS_PER_BENCHMARK:-128}
GATE_ROWS_PER_BENCHMARK=${GATE_ROWS_PER_BENCHMARK:-512}
VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-400}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
SEED=${SEED:-17}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}
STAGE0_TOP_M=${STAGE0_TOP_M:-500}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-8}
STAGE0_INVENTORY_MIN_CANDIDATES=${STAGE0_INVENTORY_MIN_CANDIDATES:-0}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-off}
STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}
STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}
STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}
MAX_ROWS=${MAX_ROWS:-}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
NEXT_SKILL_POOL_MODE=${NEXT_SKILL_POOL_MODE:-full_pool}
STAGE4_METHOD=${STAGE4_METHOD:-stage4_safe_memory_v1}
COUNTERFACTUAL_UTILITY_WEIGHT=${COUNTERFACTUAL_UTILITY_WEIGHT:-0.05}
COUNTERFACTUAL_GAIN_MARGIN=${COUNTERFACTUAL_GAIN_MARGIN:-0.1}
COUNTERFACTUAL_SAFETY_TOLERANCE=${COUNTERFACTUAL_SAFETY_TOLERANCE:-0.01}
COUNTERFACTUAL_GAIN_WEIGHT=${COUNTERFACTUAL_GAIN_WEIGHT:-1.0}
COUNTERFACTUAL_SAFETY_WEIGHT=${COUNTERFACTUAL_SAFETY_WEIGHT:-1.0}
COUNTERFACTUAL_WARMUP_FRACTION=${COUNTERFACTUAL_WARMUP_FRACTION:-0.05}
SAFE_MEMORY_RESIDUAL_BOUND=${SAFE_MEMORY_RESIDUAL_BOUND:-2.0}
SAFE_LOCAL_CANDIDATE_SIZES=${SAFE_LOCAL_CANDIDATE_SIZES:-2,3,4,5,8,10}
TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}
AUTO_REPLAY_PREFIX_MAX_STEPS=${AUTO_REPLAY_PREFIX_MAX_STEPS:-3}
TRAINABLE_REPLAY_PREFIX=${TRAINABLE_REPLAY_PREFIX:-1}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-}
# Use colon-separated BENCHMARK_CAPS when exporting through sbatch, because
# Slurm splits --export assignments on commas before the script receives them.
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"

if [[ -s "${ROUTING_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage4 requires completed Stage0 routing checkpoint: ${ROUTING_CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -s "${HEAD_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage4 requires completed ACT init checkpoint: ${HEAD_CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ "${STAGE4_METHOD}" == "candidate_admission_residual_v1" && ! -s "${BASE_CMC_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: candidate-admission Stage4 requires BASE_CMC_CHECKPOINT_PATH" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/run_clstr_stage4_act_train.py
  --trajectories_path "${TRAJECTORIES_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}"
  --head_checkpoint_path "${HEAD_CHECKPOINT_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --minimum_learning_rate "${MINIMUM_LEARNING_RATE}"
  --learning_rate_warmup_fraction "${LEARNING_RATE_WARMUP_FRACTION}"
  --validation_fraction "${VALIDATION_FRACTION}"
  --validation_rows_per_benchmark "${VALIDATION_ROWS_PER_BENCHMARK}"
  --minimum_validation_rows_per_benchmark "${MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}"
  --gate_rows_per_benchmark "${GATE_ROWS_PER_BENCHMARK}"
  --validation_interval_steps "${VALIDATION_INTERVAL_STEPS}"
  --seed "${SEED}"
  --candidate_count "${CANDIDATE_COUNT}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --route_scorer "${ROUTE_SCORER}"
  --next_skill_pool_mode "${NEXT_SKILL_POOL_MODE}"
  --stage4_method "${STAGE4_METHOD}"
  --counterfactual_utility_weight "${COUNTERFACTUAL_UTILITY_WEIGHT}"
  --counterfactual_gain_margin "${COUNTERFACTUAL_GAIN_MARGIN}"
  --counterfactual_safety_tolerance "${COUNTERFACTUAL_SAFETY_TOLERANCE}"
  --counterfactual_gain_weight "${COUNTERFACTUAL_GAIN_WEIGHT}"
  --counterfactual_safety_weight "${COUNTERFACTUAL_SAFETY_WEIGHT}"
  --counterfactual_warmup_fraction "${COUNTERFACTUAL_WARMUP_FRACTION}"
  --safe_memory_residual_bound "${SAFE_MEMORY_RESIDUAL_BOUND}"
  --safe_local_candidate_sizes "${SAFE_LOCAL_CANDIDATE_SIZES}"
  --auto_replay_prefix_max_steps "${AUTO_REPLAY_PREFIX_MAX_STEPS}"
)

if [[ -n "${BASE_CMC_CHECKPOINT_PATH}" ]]; then
  ARGS+=(--base_cmc_checkpoint_path "${BASE_CMC_CHECKPOINT_PATH}")
fi

if [[ -n "${STAGE0_TOP_M}" ]]; then
  ARGS+=(
    --stage0_top_m "${STAGE0_TOP_M}"
    --stage0_positive_missing_policy "${STAGE0_POSITIVE_MISSING_POLICY}"
    --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
    --stage0_handoff_sample_multiplier "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}"
    --stage0_inventory_min_candidates "${STAGE0_INVENTORY_MIN_CANDIDATES}"
    --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
    --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
    --stage0_handoff_cache_mode "${STAGE0_HANDOFF_CACHE_MODE}"
    --stage0_handoff_cache_dir "${STAGE0_HANDOFF_CACHE_DIR}"
    --stage0_handoff_cache_format "${STAGE0_HANDOFF_CACHE_FORMAT}"
    --stage0_handoff_cache_shard_size "${STAGE0_HANDOFF_CACHE_SHARD_SIZE}"
  )
fi

if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  ARGS+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

if [[ -n "${ALLOWED_BENCHMARKS}" ]]; then
  ARGS+=(--allowed_benchmarks "${ALLOWED_BENCHMARKS}")
fi

if [[ -n "${BENCHMARK_CAPS}" ]]; then
  ARGS+=(--benchmark_caps "${BENCHMARK_CAPS}")
fi

if [[ "${TRAIN_TRANSITION}" = "1" ]]; then
  ARGS+=(--train_transition)
fi

if [[ "${TRAINABLE_REPLAY_PREFIX}" = "1" || "${TRAINABLE_REPLAY_PREFIX}" == "true" ]]; then
  ARGS+=(--trainable_replay_prefix)
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/stdout.log" "${OUTPUT_DIR}/train_stdout.json"
