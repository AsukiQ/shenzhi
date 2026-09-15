#!/bin/bash
#SBATCH --time=12:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_belief_fix_full_20260704/stage1_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_function_aug_v2_toolbench_clean_belief_fix_full_20260705}

MAX_STEPS=${MAX_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1}
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"

STAGE0_TOP_M=${STAGE0_TOP_M:-500}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}

TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-stage0_topk_trajectory_prior}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-50}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-benchmark_transition_quota_random}
AUTO_REPLAY_PREFIX_MAX_STEPS=${AUTO_REPLAY_PREFIX_MAX_STEPS:-3}
TRAINABLE_REPLAY_PREFIX=${TRAINABLE_REPLAY_PREFIX:-true}

mkdir -p "${OUTPUT_DIR}"

if [[ ! -s "${ROUTING_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: missing Stage0 checkpoint: ${ROUTING_CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -s "${STAGE1_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: missing Stage1 checkpoint: ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 2
fi

TRAINABLE_REPLAY_PREFIX_ARGS=()
if [[ "${TRAINABLE_REPLAY_PREFIX}" == "true" || "${TRAINABLE_REPLAY_PREFIX}" == "1" ]]; then
  TRAINABLE_REPLAY_PREFIX_ARGS+=(--trainable_replay_prefix)
fi

"${PYTHON_BIN}" scripts/run_clstr_stage2_preflight.py \
  --checkpoint_path "${ROUTING_CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --model_dim 1024 \
  --output_path "${OUTPUT_DIR}/stage2_preflight.json"

"${PYTHON_BIN}" scripts/run_clstr_stage2_full_base_train.py \
  --train_path "${TRAIN_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}" \
  --stage1_checkpoint_path "${STAGE1_CHECKPOINT_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --allowed_benchmarks "${ALLOWED_BENCHMARKS}" \
  --benchmark_caps "${BENCHMARK_CAPS}" \
  --stage0_top_m "${STAGE0_TOP_M}" \
  --stage0_positive_missing_policy "${STAGE0_POSITIVE_MISSING_POLICY}" \
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}" \
  --stage0_handoff_sample_multiplier "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}" \
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}" \
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}" \
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}" \
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}" \
  --transition_loss_type "${TRANSITION_LOSS_TYPE}" \
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}" \
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}" \
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}" \
  --sampling_strategy "${SAMPLING_STRATEGY}" \
  --auto_replay_prefix_max_steps "${AUTO_REPLAY_PREFIX_MAX_STEPS}" \
  "${TRAINABLE_REPLAY_PREFIX_ARGS[@]}" \
  | tee "${OUTPUT_DIR}/train_stdout.json"
