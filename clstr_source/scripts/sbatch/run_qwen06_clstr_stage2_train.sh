#!/bin/bash
#SBATCH --job-name=qwen06_clstr_s2
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
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
FULL_MAX_STEPS=10000

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
if [[ "${FULL_RUN}" == "1" ]]; then
  TARGET_TOTAL_STEPS=${TARGET_TOTAL_STEPS:-${FULL_MAX_STEPS}}
else
  TARGET_TOTAL_STEPS=${TARGET_TOTAL_STEPS:-${MAX_STEPS}}
fi

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
DATA_ROOT=${DATA_ROOT:-${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${ASSET_ROOT}/models/Qwen3-Embedding-0.6B}
STAGE0_SELECTION_PATH=${STAGE0_SELECTION_PATH:-${RUN_ROOT}/stage0_full/stage0_selection.json}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-${RUN_ROOT}/stage1_full}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt}
STAGE1_QUALITY_GATE_PATH=${STAGE1_QUALITY_GATE_PATH:-${STAGE1_OUTPUT_DIR}/stage1_quality_gate.json}
STAGE1_LINEAGE_PATH=${STAGE1_LINEAGE_PATH:-${STAGE1_OUTPUT_DIR}/lineage.json}
if [[ "${FULL_RUN}" == "1" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage2_full}
else
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage2_smoke}
fi

require_file() {
  local path=$1
  local label=$2
  if [[ ! -s "${path}" ]]; then
    printf 'ERROR: missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
}

require_ok_gate() {
  local path=$1
  local label=$2
  require_file "${path}" "${label}"
  "${PYTHON_BIN}" - "${path}" "${label}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"{label} is not ok: {path}")
PY
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
values = [payload.get("selected_checkpoint_path"), payload.get("selected_lineage_path")]
if not all(values):
    raise SystemExit(f"Stage0 selection is incomplete: {path}")
print("\t".join(str(value) for value in values))
PY
)
IFS=$'\t' read -r ROUTING_CHECKPOINT_PATH STAGE0_LINEAGE_PATH <<<"${selection}"
require_file "${ROUTING_CHECKPOINT_PATH}" "selected Stage0 checkpoint"
require_file "${STAGE0_LINEAGE_PATH}" "selected Stage0 lineage"
require_file "${STAGE1_CHECKPOINT_PATH}" "Stage1 checkpoint"
require_ok_gate "${STAGE1_QUALITY_GATE_PATH}" "Stage1 quality gate"
require_file "${STAGE1_LINEAGE_PATH}" "Stage1 lineage"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
  --manifest_path "${STAGE0_LINEAGE_PATH}" \
  --expected_stage stage0 \
  --expected_model_path "${MODEL_NAME_OR_PATH}" \
  --expected_skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --expected_data_manifest_path "${DATA_ROOT}/manifest.json"
"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
  --manifest_path "${STAGE1_LINEAGE_PATH}" \
  --expected_stage stage1 \
  --expected_parent "stage0=${STAGE0_LINEAGE_PATH}" \
  --expected_model_path "${MODEL_NAME_OR_PATH}" \
  --expected_skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --expected_data_manifest_path "${DATA_ROOT}/manifest.json"

export TRAIN_PATH="${DATA_ROOT}/trajectories.jsonl"
export SKILLS_PATH="${DATA_ROOT}/skill_pool.jsonl"
export OUTPUT_DIR ROUTING_CHECKPOINT_PATH STAGE1_CHECKPOINT_PATH MAX_STEPS MAX_ROWS
export TARGET_TOTAL_STEPS
export MODEL_DIM=1024
export BATCH_SIZE=16
export LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
export ROUTE_SCORER=unified_memory
# Standard Stage2 defaults: ANCHORED_ROUTING_FOUNDATION=1, STATIC_ROUTE_ANCHOR_WEIGHT=0.1.
# Repair launchers may override both before entering this shared protocol.
export ANCHORED_ROUTING_FOUNDATION=${ANCHORED_ROUTING_FOUNDATION:-1}
export STATIC_ROUTE_ANCHOR_WEIGHT=${STATIC_ROUTE_ANCHOR_WEIGHT:-0.1}
export STATIC_ROUTE_ANCHOR_MAX_REGRESSION=0.005
export STAGE0_TOP_M=500
export STAGE0_POSITIVE_MISSING_POLICY=skip
export STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=50
export STAGE0_HANDOFF_CACHE_MODE=auto
export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"
export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1
export STAGE0_HANDOFF_CACHE_SHARD_SIZE=2048
export STAGE0_QUALITY_GATE_MODE=handoff_gate
export STAGE0_HANDOFF_GATE_PATH="${STAGE0_SELECTION_PATH}"
export NEXT_SKILL_POOL_MODE=full_pool
export TRANSITION_INVENTORY_MASK_MODE=explicit_only
export TRANSITION_INVENTORY_MIN_CANDIDATES=64
export TRANSITION_LOSS_TYPE=listwise_nll
export TRANSITION_POSITIVE_MODE=gold_plus_equivalent
export AUTO_REPLAY_PREFIX_MAX_STEPS=3
export TRAINABLE_REPLAY_PREFIX=1
export SAMPLING_STRATEGY=balanced_random
if [[ "${FULL_RUN}" == "1" ]]; then
  EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-always}
else
  EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
fi
EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
if [[ "${FULL_RUN}" == "1" && "${EMBEDDING_CACHE_MODE}" != "always" ]]; then
  printf 'ERROR: full Stage2 requires EMBEDDING_CACHE_MODE=always; got %s\n' \
    "${EMBEDDING_CACHE_MODE}" >&2
  exit 2
fi
export EMBEDDING_CACHE_MODE EMBEDDING_CACHE_MAX_ROWS
export ALLOWED_BENCHMARKS=toolbench_g3,traject_bench,alfworld,webshop
export BENCHMARK_CAPS=toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1

mkdir -p "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage2] mode=%s steps=%s rows=%s output=%s\n' \
  "${FULL_RUN}" "${MAX_STEPS}" "${MAX_ROWS:-all}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage2] stage0=%s stage1=%s route=%s pool=%s\n' \
  "${ROUTING_CHECKPOINT_PATH}" "${STAGE1_CHECKPOINT_PATH}" "${ROUTE_SCORER}" "${NEXT_SKILL_POOL_MODE}"

bash scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh

CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/clstr_full_base-step${TARGET_TOTAL_STEPS}.pt"
QUALITY_GATE_PATH="${OUTPUT_DIR}/stage2_quality_gate.json"
LINEAGE_PATH="${OUTPUT_DIR}/lineage.json"
require_file "${CHECKPOINT_PATH}" "Stage2 checkpoint"

AUDIT_ARGS=(
  scripts/audit_clstr_stage2_quality.py
  --output_dir "${OUTPUT_DIR}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --output_path "${QUALITY_GATE_PATH}"
  --min_steps "${TARGET_TOTAL_STEPS}"
  --expected_route_scorer unified_memory
  --expected_next_skill_pool_mode full_pool
  --static_route_anchor_max_regression "${STATIC_ROUTE_ANCHOR_MAX_REGRESSION}"
  --fail_on_action_required
)
if [[ "${FULL_RUN}" == "0" ]]; then
  AUDIT_ARGS+=(
    --first_window 1
    --last_window 1
    --min_loss_drop -1000000000
    --max_transition_recall_at_5_drop 1.0
    --max_transition_skill_ce_increase 1000000000
    --min_transition_recall_at_5_tail 0.0
    --max_stage2_recall_at_5_drop_vs_stage0_prior 1.0
    --max_stage2_mrr_drop_vs_stage0_prior 1.0
    --max_stage2_worse_than_stage0_prior_fraction 1.0
  )
fi
"${PYTHON_BIN}" "${AUDIT_ARGS[@]}"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py create \
  --stage stage2 \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --expected_checkpoint_stage clstr_full_base_component_complete \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --data_manifest_path "${DATA_ROOT}/manifest.json" \
  --parent "stage0=${STAGE0_LINEAGE_PATH}" \
  --parent "stage1=${STAGE1_LINEAGE_PATH}" \
  --output_path "${LINEAGE_PATH}"

require_file "${QUALITY_GATE_PATH}" "Stage2 quality gate"
require_file "${LINEAGE_PATH}" "Stage2 lineage"
