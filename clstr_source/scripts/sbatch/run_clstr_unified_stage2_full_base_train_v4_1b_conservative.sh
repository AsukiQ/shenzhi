#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_1b/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_unified_stage1_v4_1b_top350_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
V4_1B_HANDOFF_MANIFEST=${V4_1B_HANDOFF_MANIFEST:-outputs/clstr_unified_stage1_v4_1b_top350_heads_init/stage0_candidate_handoff.json}
STAGE0_HANDOFF_GATE_PATH=${STAGE0_HANDOFF_GATE_PATH:-${OUTPUT_DIR}/stage0_handoff_gate_v4_1b.json}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${V4_1B_HANDOFF_MANIFEST}" "${STAGE0_HANDOFF_GATE_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if not source.is_file():
    print(f"ERROR: missing v4.1b Stage0 handoff manifest: {source}", file=sys.stderr)
    raise SystemExit(2)
payload = json.loads(source.read_text(encoding="utf-8"))
current_cov = float(payload.get("current_positive_coverage@M") or 0.0)
next_cov = float(payload.get("next_positive_coverage@M") or 0.0)
top_m = int(payload.get("top_m") or 0)
missing_policy = str(payload.get("positive_missing_policy") or "")
injected_rows = int(payload.get("injected_positive_rows") or 0)
blockers = []
if top_m != 350:
    blockers.append("unexpected_top_m")
if missing_policy != "skip":
    blockers.append("unexpected_positive_missing_policy")
if injected_rows != 0:
    blockers.append("unexpected_injected_positive_rows")
if current_cov < 0.98:
    blockers.append("current_positive_coverage_below_0.98")
if next_cov < 0.98:
    blockers.append("next_positive_coverage_below_0.98")
gate = {
    "status": "ok" if not blockers else "action_required",
    "blockers": blockers,
    "source_manifest": str(source),
    "top_m": top_m,
    "positive_missing_policy": missing_policy,
    "injected_positive_rows": injected_rows,
    "current_positive_coverage@M": current_cov,
    "next_positive_coverage@M": next_cov,
    "note": "v4.1b conservative Stage2 uses the existing Stage1 handoff manifest as its Stage0 handoff gate.",
}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if gate["status"] != "ok":
    print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(3)
print(json.dumps(gate, ensure_ascii=False))
PY

export TRAIN_PATH SKILLS_PATH OUTPUT_DIR ROUTING_CHECKPOINT_PATH STAGE1_CHECKPOINT_PATH
export STAGE0_OUTPUT_DIR=outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350
export STAGE0_QUALITY_GATE_MODE=handoff_gate
export STAGE0_HANDOFF_GATE_PATH
export STAGE0_TOP_M=${STAGE0_TOP_M:-350}
export STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
export STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}
export MAX_STEPS=${MAX_STEPS:-10000}
export BATCH_SIZE=${BATCH_SIZE:-4}
export LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
export POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
export POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}
export Q_SUCCESS_LOSS_WEIGHT=${Q_SUCCESS_LOSS_WEIGHT:-0.0}
export TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.05}
export TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.2}
export TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}
export TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER=${TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER:-1.0}
export TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER=${TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER:-1.0}
export TRANSITION_HARD_NEGATIVE_MARGIN=${TRANSITION_HARD_NEGATIVE_MARGIN:-1.0}
export TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}
export TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}
export TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-cross_entropy}
export TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-single}
export TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
export TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
export BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.2}
export RETRIEVAL_LOSS_WEIGHT=${RETRIEVAL_LOSS_WEIGHT:-0.0}
export RETRIEVAL_NUM_NEGATIVES=${RETRIEVAL_NUM_NEGATIVES:-32}
export RETRIEVAL_HARD_RATIO=${RETRIEVAL_HARD_RATIO:-0.5}
export EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
export EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
export INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}
export SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
export ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
export BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}

exec bash scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
