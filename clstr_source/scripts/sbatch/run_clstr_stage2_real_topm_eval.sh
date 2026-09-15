#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_real_topm_eval}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-outputs/clstr_unified_stage2_v2_loss_sampler_full_top350_inject/checkpoints/clstr_full_base-step10000.pt}
TOP_M=${TOP_M:-350}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_ROWS=${MAX_ROWS:-}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-}
SEED=${SEED:-17}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
MIN_CURRENT_COVERAGE=${MIN_CURRENT_COVERAGE:-0.85}
MIN_NEXT_COVERAGE=${MIN_NEXT_COVERAGE:-0.85}
MIN_TRANSITION_RECALL_AT_5=${MIN_TRANSITION_RECALL_AT_5:-0.5}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-cross_entropy}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-single}
FAIL_ON_ACTION_REQUIRED=${FAIL_ON_ACTION_REQUIRED:-0}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/evaluate_clstr_stage2_real_topm.py
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}"
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --top_m "${TOP_M}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --sampling_strategy "${SAMPLING_STRATEGY}"
  --embedding_cache_mode "${EMBEDDING_CACHE_MODE}"
  --embedding_cache_max_rows "${EMBEDDING_CACHE_MAX_ROWS}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  --min_current_coverage "${MIN_CURRENT_COVERAGE}"
  --min_next_coverage "${MIN_NEXT_COVERAGE}"
  --min_transition_recall_at_5 "${MIN_TRANSITION_RECALL_AT_5}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_loss_type "${TRANSITION_LOSS_TYPE}"
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}"
)

if [[ -n "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}" ]]; then
  ARGS+=(--stage0_handoff_sample_multiplier "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}")
fi

if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
  ARGS+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
fi

if [[ -n "${ALLOWED_BENCHMARKS}" ]]; then
  ARGS+=(--allowed_benchmarks "${ALLOWED_BENCHMARKS}")
fi

if [[ -n "${BENCHMARK_CAPS}" ]]; then
  ARGS+=(--benchmark_caps "${BENCHMARK_CAPS}")
fi

if [ "${FAIL_ON_ACTION_REQUIRED}" = "1" ]; then
  ARGS+=(--fail_on_action_required)
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/real_topm_eval_stdout.json"
