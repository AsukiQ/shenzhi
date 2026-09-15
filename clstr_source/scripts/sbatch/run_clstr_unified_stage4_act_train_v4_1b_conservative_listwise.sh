#!/bin/bash
#SBATCH --time=06:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_unified_pretrain_v4_1b/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt}
STAGE2_QUALITY_GATE_PATH=${STAGE2_QUALITY_GATE_PATH:-outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/stage2_quality_gate.json}
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
SEED=${SEED:-17}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-8}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-8}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
MAX_ROWS=${MAX_ROWS:-}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}
EXPECTED_TRANSITION_SCORING_MODE=${EXPECTED_TRANSITION_SCORING_MODE:-v4_1b_action_observation}
EXPECTED_TRANSITION_RESIDUAL_LAMBDA=${EXPECTED_TRANSITION_RESIDUAL_LAMBDA:-0.0}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=4000:traject_bench=4000:alfworld=-1:webshop=4000}
# Use colon-separated BENCHMARK_CAPS when exporting through sbatch, because
# Slurm splits --export assignments on commas before the script receives them.
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"

if [[ -s "${STAGE2_QUALITY_GATE_PATH}" ]]; then
  "${PYTHON_BIN}" - "${STAGE2_QUALITY_GATE_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    print(f"ERROR: Stage4 requires Stage2 quality gate ok: {path}", file=sys.stderr)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(3)
print(json.dumps({"status": "ok", "stage2_quality_gate": str(path)}, ensure_ascii=False))
PY
else
  echo "ERROR: Stage4 requires completed Stage2 quality gate: ${STAGE2_QUALITY_GATE_PATH}" >&2
  exit 2
fi

export TRAJECTORIES_PATH SKILLS_PATH OUTPUT_DIR ROUTING_CHECKPOINT_PATH HEAD_CHECKPOINT_PATH
export MAX_STEPS BATCH_SIZE LEARNING_RATE SEED CANDIDATE_COUNT MAX_ROWS
export STAGE0_TOP_M STAGE0_POSITIVE_MISSING_POLICY STAGE0_HANDOFF_QUERY_MODE
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER STAGE0_CANDIDATE_ENCODE_BATCH_SIZE
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES
export TRANSITION_RESIDUAL_LAMBDA TRAIN_TRANSITION
export TRANSITION_SCORING_MODE
export EXPECTED_TRANSITION_SCORING_MODE EXPECTED_TRANSITION_RESIDUAL_LAMBDA
export ALLOWED_BENCHMARKS BENCHMARK_CAPS

exec bash scripts/sbatch/run_clstr_unified_stage4_act_train.sh
