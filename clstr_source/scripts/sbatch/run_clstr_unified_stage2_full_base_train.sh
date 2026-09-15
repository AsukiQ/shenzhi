#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json}
STAGE0_BASELINE_METRICS_PATH=${STAGE0_BASELINE_METRICS_PATH:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json}
STAGE0_QUALITY_GATE_MODE=${STAGE0_QUALITY_GATE_MODE:-strict}
STAGE0_HANDOFF_GATE_PATH=${STAGE0_HANDOFF_GATE_PATH:-}
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
ALLOW_FULL_POOL_STAGE2_DEBUG=${ALLOW_FULL_POOL_STAGE2_DEBUG:-0}
WARM_START_CHECKPOINT_PATH=${WARM_START_CHECKPOINT_PATH:-}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
TARGET_TOTAL_STEPS=${TARGET_TOTAL_STEPS:-}
MAX_STEPS=${MAX_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-}
CHECKPOINT_INTERVAL_STEPS=${CHECKPOINT_INTERVAL_STEPS:-400}
MODEL_DIM=${MODEL_DIM:-1024}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.2}
TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}
TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-1.0}
COUNTERFACTUAL_HISTORY_LOSS_WEIGHT=${COUNTERFACTUAL_HISTORY_LOSS_WEIGHT:-0.0}
COUNTERFACTUAL_HISTORY_MARGIN=${COUNTERFACTUAL_HISTORY_MARGIN:-0.1}
TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER=${TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER:-1.0}
TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER=${TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER:-1.0}
NEXT_SKILL_POOL_MODE=${NEXT_SKILL_POOL_MODE:-full_pool}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-explicit_only}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
ANCHORED_ROUTING_FOUNDATION=${ANCHORED_ROUTING_FOUNDATION:-1}
STATIC_ROUTE_ANCHOR_WEIGHT=${STATIC_ROUTE_ANCHOR_WEIGHT:-0.1}
STATIC_ROUTE_ANCHOR_MAX_REGRESSION=${STATIC_ROUTE_ANCHOR_MAX_REGRESSION:-0.005}
AUTO_REPLAY_PREFIX_MAX_STEPS=${AUTO_REPLAY_PREFIX_MAX_STEPS:-auto}
TRAINABLE_REPLAY_PREFIX=${TRAINABLE_REPLAY_PREFIX:-auto}
STAGE0_SCORE_PRIOR_CALIBRATION=${STAGE0_SCORE_PRIOR_CALIBRATION:-off}
BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}
EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
MAX_ROWS=${MAX_ROWS:-}
INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
# Stage 2 defaults to the current action-aware progressive-final mainline.
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1}
# Use colon-separated BENCHMARK_CAPS when exporting through sbatch, because
# Slurm splits --export assignments on commas before the script receives them.
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"

if [[ -n "${WARM_START_CHECKPOINT_PATH}" && -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: warm-start and exact resume are mutually exclusive" >&2
  exit 2
fi

if [[ "${AUTO_REPLAY_PREFIX_MAX_STEPS}" == "auto" ]]; then
  if [[ "${ROUTE_SCORER}" == "unified_memory" ]]; then
    AUTO_REPLAY_PREFIX_MAX_STEPS=3
  else
    AUTO_REPLAY_PREFIX_MAX_STEPS=0
  fi
fi

if [[ "${TRAINABLE_REPLAY_PREFIX}" == "auto" ]]; then
  if [[ "${ROUTE_SCORER}" == "unified_memory" ]]; then
    TRAINABLE_REPLAY_PREFIX=1
  else
    TRAINABLE_REPLAY_PREFIX=0
  fi
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -s "${STAGE1_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage2 requires completed Stage1 heads checkpoint: ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 2
fi

case "${STAGE0_QUALITY_GATE_MODE}" in
  strict)
    "${PYTHON_BIN}" scripts/audit_clstr_stage0_quality.py \
      --output_dir "${STAGE0_OUTPUT_DIR}" \
      --checkpoint_path "${ROUTING_CHECKPOINT_PATH}" \
      --stage0_eval_metrics_path "${STAGE0_EVAL_METRICS_PATH}" \
      --baseline_metrics_path "${STAGE0_BASELINE_METRICS_PATH}" \
      --output_path "${OUTPUT_DIR}/stage0_quality_gate.json" \
      --fail_on_action_required
    ;;
  handoff_gate)
    if [[ -s "${STAGE0_HANDOFF_GATE_PATH}" ]]; then
      :
    else
      echo "ERROR: Stage2 handoff_gate mode requires completed Stage0 handoff gate: ${STAGE0_HANDOFF_GATE_PATH}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" - "${STAGE0_HANDOFF_GATE_PATH}" "${OUTPUT_DIR}/stage0_handoff_gate.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = json.loads(source.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    print(f"ERROR: Stage0 handoff gate is not ok: {source}", file=sys.stderr)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(3)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "mode": "handoff_gate", "source": str(source), "target": str(target)}, ensure_ascii=False))
PY
    ;;
  *)
    echo "ERROR: unsupported STAGE0_QUALITY_GATE_MODE=${STAGE0_QUALITY_GATE_MODE}; expected strict or handoff_gate" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" scripts/run_clstr_stage2_preflight.py \
  --checkpoint_path "${ROUTING_CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --model_dim "${MODEL_DIM}" \
  --output_path "${OUTPUT_DIR}/stage2_preflight.json"

ARGS=(
  scripts/run_clstr_stage2_full_base_train.py
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}"
  --stage1_checkpoint_path "${STAGE1_CHECKPOINT_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}"
  --policy_loss_weight "${POLICY_LOSS_WEIGHT}"
  --transition_loss_weight "${TRANSITION_LOSS_WEIGHT}"
  --transition_skill_ce_loss_weight "${TRANSITION_SKILL_CE_LOSS_WEIGHT}"
  --counterfactual_history_loss_weight "${COUNTERFACTUAL_HISTORY_LOSS_WEIGHT}"
  --counterfactual_history_margin "${COUNTERFACTUAL_HISTORY_MARGIN}"
  --transition_real_candidate_ce_multiplier "${TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER}"
  --transition_injected_candidate_ce_multiplier "${TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER}"
  --next_skill_pool_mode "${NEXT_SKILL_POOL_MODE}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
  --transition_loss_type "${TRANSITION_LOSS_TYPE}"
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --route_scorer "${ROUTE_SCORER}"
  --static_route_anchor_weight "${STATIC_ROUTE_ANCHOR_WEIGHT}"
  --static_route_anchor_max_regression "${STATIC_ROUTE_ANCHOR_MAX_REGRESSION}"
  --auto_replay_prefix_max_steps "${AUTO_REPLAY_PREFIX_MAX_STEPS}"
  --stage0_score_prior_calibration "${STAGE0_SCORE_PRIOR_CALIBRATION}"
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
  --sampling_strategy "${SAMPLING_STRATEGY}"
)

if [[ -n "${MINIMUM_LEARNING_RATE}" ]]; then
  ARGS+=(--minimum_learning_rate "${MINIMUM_LEARNING_RATE}")
fi

case "${ANCHORED_ROUTING_FOUNDATION}" in
  1|true|TRUE|yes|YES|on|ON)
    ARGS+=(--anchored_routing_foundation)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "ERROR: unsupported ANCHORED_ROUTING_FOUNDATION=${ANCHORED_ROUTING_FOUNDATION}; expected 1/0/true/false" >&2
    exit 2
    ;;
esac

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

if [ "${ALLOW_FULL_POOL_STAGE2_DEBUG}" = "1" ]; then
  ARGS+=(--allow_full_pool_stage2_debug)
fi

case "${TRAINABLE_REPLAY_PREFIX}" in
  1|true|TRUE|yes|YES|on|ON)
    ARGS+=(--trainable_replay_prefix)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "ERROR: unsupported TRAINABLE_REPLAY_PREFIX=${TRAINABLE_REPLAY_PREFIX}; expected auto/1/0/true/false" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  if [[ -s "${RESUME_CHECKPOINT_PATH}" ]]; then
    :
  else
    echo "ERROR: RESUME_CHECKPOINT_PATH does not exist or is empty: ${RESUME_CHECKPOINT_PATH}" >&2
    exit 2
  fi
  ARGS+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

if [[ -n "${WARM_START_CHECKPOINT_PATH}" ]]; then
  if [[ -s "${WARM_START_CHECKPOINT_PATH}" ]]; then
    :
  else
    echo "ERROR: WARM_START_CHECKPOINT_PATH does not exist or is empty: ${WARM_START_CHECKPOINT_PATH}" >&2
    exit 2
  fi
  ARGS+=(--warm_start_checkpoint_path "${WARM_START_CHECKPOINT_PATH}")
fi

if [[ -n "${TARGET_TOTAL_STEPS}" ]]; then
  ARGS+=(--target_total_steps "${TARGET_TOTAL_STEPS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/stdout.log" "${OUTPUT_DIR}/train_stdout.json"
