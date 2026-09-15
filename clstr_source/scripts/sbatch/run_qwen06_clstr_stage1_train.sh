#!/bin/bash
#SBATCH --job-name=qwen06_clstr_s1
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

FULL_RUN=${FULL_RUN:-0}
SMOKE_MAX_STEPS=2
SMOKE_MAX_ROWS=128
FULL_MAX_STEPS=3000

case "${FULL_RUN}" in
  0)
    MAX_STEPS=${MAX_STEPS:-${SMOKE_MAX_STEPS}}
    MAX_ROWS=${MAX_ROWS:-${SMOKE_MAX_ROWS}}
    ;;
  1)
    MAX_STEPS=${FULL_MAX_STEPS}
    MAX_ROWS=
    ;;
  *)
    printf 'ERROR: FULL_RUN must be 0 or 1; got %s\n' "${FULL_RUN}" >&2
    exit 2
    ;;
esac

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
DATA_ROOT=${DATA_ROOT:-${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${ASSET_ROOT}/models/Qwen3-Embedding-0.6B}
STAGE0_SELECTION_PATH=${STAGE0_SELECTION_PATH:-${RUN_ROOT}/stage0_full/stage0_selection.json}
if [[ "${FULL_RUN}" == "1" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage1_full}
else
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage1_smoke}
fi

require_file() {
  local path=$1
  local label=$2
  if [[ ! -s "${path}" ]]; then
    printf 'ERROR: missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
}

selection=$(
  "${PYTHON_BIN}" - "${STAGE0_SELECTION_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing Stage0 selection: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok" or payload.get("release_status") != "ok":
    raise SystemExit(f"Stage0 selection is not release-safe: {path}")
values = [
    payload.get("selected_checkpoint_path"),
    payload.get("selected_lineage_path"),
    payload.get("selected_report_path"),
]
if not all(values):
    raise SystemExit(f"Stage0 selection is incomplete: {path}")
print("\t".join(str(value) for value in values))
PY
)
IFS=$'\t' read -r ROUTING_CHECKPOINT_PATH STAGE0_LINEAGE_PATH STAGE0_REPORT_PATH <<<"${selection}"
require_file "${ROUTING_CHECKPOINT_PATH}" "selected Stage0 checkpoint"
require_file "${STAGE0_LINEAGE_PATH}" "selected Stage0 lineage"
require_file "${STAGE0_REPORT_PATH}" "selected Stage0 audit report"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
  --manifest_path "${STAGE0_LINEAGE_PATH}" \
  --expected_stage stage0 \
  --expected_model_path "${MODEL_NAME_OR_PATH}" \
  --expected_skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --expected_data_manifest_path "${DATA_ROOT}/manifest.json"

export TRAIN_PATH="${DATA_ROOT}/trajectories.jsonl"
export SKILLS_PATH="${DATA_ROOT}/skill_pool.jsonl"
export OUTPUT_DIR ROUTING_CHECKPOINT_PATH MAX_STEPS MAX_ROWS
export BATCH_SIZE=16
export LEARNING_RATE=1.0e-4
export POLICY_LOSS_WEIGHT=0.6
export TRANSITION_LOSS_WEIGHT=0.2
export TRANSITION_SKILL_CE_LOSS_WEIGHT=0.6
export ROUTE_SCORER=unified_memory
export STAGE0_TOP_M=500
export STAGE0_POSITIVE_MISSING_POLICY=skip
export STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=4
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=50
export STAGE0_HANDOFF_CACHE_MODE=auto
export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"
export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1
export STAGE0_HANDOFF_CACHE_SHARD_SIZE=2048
export TRANSITION_INVENTORY_MASK_MODE=auto
export TRANSITION_INVENTORY_MIN_CANDIDATES=64
export TRANSITION_LOSS_TYPE=listwise_nll
export TRANSITION_POSITIVE_MODE=gold_plus_equivalent
export TRANSITION_RESIDUAL_LAMBDA=0.25
export TRANSITION_SCORING_MODE=stage0_rank_prior_plus_transition_residual
export SAMPLING_STRATEGY=balanced_random
export EMBEDDING_CACHE_MODE=auto
export EMBEDDING_CACHE_MAX_ROWS=20000
export ALLOWED_BENCHMARKS=toolbench_g3,traject_bench,alfworld,webshop
export BENCHMARK_CAPS=toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1

mkdir -p "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage1] mode=%s steps=%s rows=%s output=%s\n' \
  "${FULL_RUN}" "${MAX_STEPS}" "${MAX_ROWS:-all}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage1] stage0=%s route=%s query_mode=%s\n' \
  "${ROUTING_CHECKPOINT_PATH}" "${ROUTE_SCORER}" "${STAGE0_HANDOFF_QUERY_MODE}"

bash scripts/sbatch/run_clstr_unified_stage1_heads_init.sh

if [[ "${FULL_RUN}" == "1" ]]; then
  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt"
else
  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step${MAX_STEPS}.pt"
fi
QUALITY_GATE_PATH="${OUTPUT_DIR}/stage1_quality_gate.json"
LINEAGE_PATH="${OUTPUT_DIR}/lineage.json"
require_file "${CHECKPOINT_PATH}" "Stage1 checkpoint"

AUDIT_ARGS=(
  scripts/audit_clstr_stage1_heads_quality.py
  --output_dir "${OUTPUT_DIR}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --output_path "${QUALITY_GATE_PATH}"
  --min_steps "${MAX_STEPS}"
  --expected_route_scorer unified_memory
  --fail_on_action_required
)
if [[ "${FULL_RUN}" == "0" ]]; then
  AUDIT_ARGS+=(
    --first_window 1
    --last_window 1
    --min_loss_drop -1000000000
    --max_transition_recall_at_5_drop 1.0
    --max_transition_skill_ce_increase 1000000000
    --min_transition_recall_at_1_tail 0.0
    --min_transition_recall_at_5_tail 0.0
  )
fi
"${PYTHON_BIN}" "${AUDIT_ARGS[@]}"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py create \
  --stage stage1 \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --expected_checkpoint_stage clstr_stage1_heads_init \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --data_manifest_path "${DATA_ROOT}/manifest.json" \
  --parent "stage0=${STAGE0_LINEAGE_PATH}" \
  --output_path "${LINEAGE_PATH}"

require_file "${QUALITY_GATE_PATH}" "Stage1 quality gate"
require_file "${LINEAGE_PATH}" "Stage1 lineage"
