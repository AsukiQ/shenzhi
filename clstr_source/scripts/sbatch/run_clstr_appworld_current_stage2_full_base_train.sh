#!/bin/bash
#SBATCH --time=06:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_appworld_current_route_v1/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_appworld_current_route_v1/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_appworld_current_stage2_full_base}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step2000.pt}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage1_heads_init/checkpoints/clstr_stage1_heads-step2000.pt}
APPWORLD_STAGE1_HANDOFF_MANIFEST=${APPWORLD_STAGE1_HANDOFF_MANIFEST:-outputs/clstr_appworld_current_stage1_heads_init/stage0_candidate_handoff.json}
STAGE0_HANDOFF_GATE_PATH=${STAGE0_HANDOFF_GATE_PATH:-${OUTPUT_DIR}/stage0_handoff_gate_appworld.json}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${APPWORLD_STAGE1_HANDOFF_MANIFEST}" "${STAGE0_HANDOFF_GATE_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if not source.is_file():
    print(f"ERROR: missing AppWorld Stage1 handoff manifest: {source}", file=sys.stderr)
    raise SystemExit(2)
payload = json.loads(source.read_text(encoding="utf-8"))
missing_policy = str(payload.get("positive_missing_policy") or "")
injected_rows = int(payload.get("injected_positive_rows") or 0)
blockers = []
if missing_policy != "skip":
    blockers.append("unexpected_positive_missing_policy")
if injected_rows != 0:
    blockers.append("unexpected_injected_positive_rows")
gate = {
    "status": "ok" if not blockers else "action_required",
    "blockers": blockers,
    "source_manifest": str(source),
    "top_m": payload.get("top_m"),
    "positive_missing_policy": missing_policy,
    "injected_positive_rows": injected_rows,
    "current_positive_coverage@M": payload.get("current_positive_coverage@M"),
    "next_positive_coverage@M": payload.get("next_positive_coverage@M"),
    "note": "AppWorld current-route Stage2 uses the matching Stage1 Stage0 handoff manifest as its Stage0 gate.",
}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if gate["status"] != "ok":
    print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(3)
print(json.dumps(gate, ensure_ascii=False))
PY

export TRAIN_PATH SKILLS_PATH OUTPUT_DIR ROUTING_CHECKPOINT_PATH STAGE1_CHECKPOINT_PATH
export STAGE0_QUALITY_GATE_MODE=handoff_gate
export STAGE0_HANDOFF_GATE_PATH
export STAGE0_TOP_M=${STAGE0_TOP_M:-100}
export STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
export STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}
export MAX_STEPS=${MAX_STEPS:-5000}
export BATCH_SIZE=${BATCH_SIZE:-4}
export LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
export POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.2}
export TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}
export TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-1.0}
export TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.1}
export TRANSITION_HARD_NEGATIVE_MARGIN=${TRANSITION_HARD_NEGATIVE_MARGIN:-1.0}
export TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}
export TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}
export TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
export TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
export TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
export TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-skill_prior_plus_action_observation_residual}
export BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.1}
export RETRIEVAL_LOSS_WEIGHT=${RETRIEVAL_LOSS_WEIGHT:-0.0}
export EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
export EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
export SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
export ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-appworld}
export BENCHMARK_CAPS=${BENCHMARK_CAPS:-appworld=-1}

exec bash scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
