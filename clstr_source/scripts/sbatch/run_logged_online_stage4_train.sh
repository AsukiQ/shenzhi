#!/bin/bash
#SBATCH --time=01:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/logged_online_stage4_v4_2_progressive_smoke}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}

USE_STAGE0_HANDOFF=${USE_STAGE0_HANDOFF:-1}
MAX_ROWS=${MAX_ROWS:-256}
MAX_UPDATES=${MAX_UPDATES:-30}
BATCH_SIZE=${BATCH_SIZE:-2}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_ROWS=${EVAL_ROWS:-32}
EVAL_SPLIT_MODE=${EVAL_SPLIT_MODE:-sequential_tail}
TRAJECTORY_EVAL_STEPS=${TRAJECTORY_EVAL_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}
POSITIVE_MISSING_POLICY=${POSITIVE_MISSING_POLICY:-skip}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-8}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-5}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-""}
EVAL_BENCHMARK_CAPS=${EVAL_BENCHMARK_CAPS:-""}
SAVE_STAGE4_ROWS_PATH=${SAVE_STAGE4_ROWS_PATH:-}
PREBUILT_STAGE4_ROWS_PATH=${PREBUILT_STAGE4_ROWS_PATH:-}
REPLAY_PREFIX_SOURCE_ROWS_PATH=${REPLAY_PREFIX_SOURCE_ROWS_PATH:-}
REPLAY_PREFIX_MAX_STEPS=${REPLAY_PREFIX_MAX_STEPS:-3}
TRAIN_TRANSITION=${TRAIN_TRANSITION:-0}
TRAIN_SCORE_CALIBRATOR=${TRAIN_SCORE_CALIBRATOR:-0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-0.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_STATE_SIMILARITY_THRESHOLD=${ONLINE_MEMORY_STATE_SIMILARITY_THRESHOLD:-0.2}
ONLINE_MEMORY_STATE_SIMILARITY_TEMPERATURE=${ONLINE_MEMORY_STATE_SIMILARITY_TEMPERATURE:-1.0}
ONLINE_MEMORY_AUTO_GATE=${ONLINE_MEMORY_AUTO_GATE:-0}
ONLINE_MEMORY_GATE_SCOPE=${ONLINE_MEMORY_GATE_SCOPE:-global}
ONLINE_MEMORY_GATE_EVAL_ROWS=${ONLINE_MEMORY_GATE_EVAL_ROWS:-64}
ONLINE_MEMORY_GATE_MIN_DELTA_MRR=${ONLINE_MEMORY_GATE_MIN_DELTA_MRR:-0.0}
ONLINE_MEMORY_GATE_MIN_DELTA_RECALL5=${ONLINE_MEMORY_GATE_MIN_DELTA_RECALL5:-0.0}

if [[ -s "${ROUTING_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: logged-online Stage4 requires completed frozen retriever checkpoint: ${ROUTING_CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -s "${HEAD_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: logged-online Stage4 requires completed Stage2/head checkpoint: ${HEAD_CHECKPOINT_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

BENCHMARK_CAPS_NORMALIZED=${BENCHMARK_CAPS//:/,}
EVAL_BENCHMARK_CAPS_NORMALIZED=${EVAL_BENCHMARK_CAPS//:/,}

ARGS=(
  scripts/run_logged_online_stage4_train.py
  --trajectories_path "${TRAJECTORIES_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}"
  --head_checkpoint_path "${HEAD_CHECKPOINT_PATH}"
  --max_rows "${MAX_ROWS}"
  --max_updates "${MAX_UPDATES}"
  --batch_size "${BATCH_SIZE}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_rows "${EVAL_ROWS}"
  --eval_split_mode "${EVAL_SPLIT_MODE}"
  --trajectory_eval_steps "${TRAJECTORY_EVAL_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --candidate_count "${CANDIDATE_COUNT}"
  --positive_missing_policy "${POSITIVE_MISSING_POLICY}"
  --allowed_benchmarks "${ALLOWED_BENCHMARKS}"
  --benchmark_caps "${BENCHMARK_CAPS_NORMALIZED}"
  --eval_benchmark_caps "${EVAL_BENCHMARK_CAPS_NORMALIZED}"
  --replay_prefix_max_steps "${REPLAY_PREFIX_MAX_STEPS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --route_scorer "${ROUTE_SCORER}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_state_similarity_threshold "${ONLINE_MEMORY_STATE_SIMILARITY_THRESHOLD}"
  --online_memory_state_similarity_temperature "${ONLINE_MEMORY_STATE_SIMILARITY_TEMPERATURE}"
  --online_memory_gate_scope "${ONLINE_MEMORY_GATE_SCOPE}"
  --online_memory_gate_eval_rows "${ONLINE_MEMORY_GATE_EVAL_ROWS}"
  --online_memory_gate_min_delta_mrr "${ONLINE_MEMORY_GATE_MIN_DELTA_MRR}"
  --online_memory_gate_min_delta_recall5 "${ONLINE_MEMORY_GATE_MIN_DELTA_RECALL5}"
)

if [[ "${USE_STAGE0_HANDOFF}" = "1" ]]; then
  ARGS+=(
    --use_stage0_handoff
    --stage0_top_m "${STAGE0_TOP_M}"
    --stage0_positive_missing_policy "${STAGE0_POSITIVE_MISSING_POLICY}"
    --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
    --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
    --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  )
fi

if [[ "${TRAIN_TRANSITION}" = "1" ]]; then
  ARGS+=(--train_transition)
fi

if [[ "${TRAIN_SCORE_CALIBRATOR}" = "1" ]]; then
  ARGS+=(--train_score_calibrator)
fi

if [[ "${ONLINE_MEMORY_AUTO_GATE}" = "1" ]]; then
  ARGS+=(--online_memory_auto_gate)
fi

if [[ -n "${SAVE_STAGE4_ROWS_PATH}" ]]; then
  ARGS+=(--save_stage4_rows_path "${SAVE_STAGE4_ROWS_PATH}")
fi

if [[ -n "${PREBUILT_STAGE4_ROWS_PATH}" ]]; then
  ARGS+=(--prebuilt_stage4_rows_path "${PREBUILT_STAGE4_ROWS_PATH}")
fi

if [[ -n "${REPLAY_PREFIX_SOURCE_ROWS_PATH}" ]]; then
  ARGS+=(--replay_prefix_source_rows_path "${REPLAY_PREFIX_SOURCE_ROWS_PATH}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/train_stdout.json"
