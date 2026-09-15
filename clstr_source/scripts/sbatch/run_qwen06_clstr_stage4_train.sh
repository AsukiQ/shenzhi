#!/bin/bash
#SBATCH --job-name=qwen06_clstr_s4
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

FULL_RUN=${FULL_RUN:-0}
STAGE4_METHOD_OVERRIDE=${STAGE4_METHOD:-}
STAGE4_METHOD=counterfactual_memory_calibration_v1
STAGE4_METHOD=${STAGE4_METHOD_OVERRIDE:-${STAGE4_METHOD}}
SMOKE_MAX_STEPS_OVERRIDE=${SMOKE_MAX_STEPS:-}
FULL_MAX_STEPS_OVERRIDE=${FULL_MAX_STEPS:-}
SMOKE_MAX_ROWS_OVERRIDE=${SMOKE_MAX_ROWS:-}
SMOKE_VALIDATION_INTERVAL_STEPS_OVERRIDE=${SMOKE_VALIDATION_INTERVAL_STEPS:-}
FULL_VALIDATION_INTERVAL_STEPS_OVERRIDE=${FULL_VALIDATION_INTERVAL_STEPS:-}
SMOKE_VALIDATION_ROWS_PER_BENCHMARK_OVERRIDE=${SMOKE_VALIDATION_ROWS_PER_BENCHMARK:-}
SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK_OVERRIDE=${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK:-}
SMOKE_GATE_ROWS_PER_BENCHMARK_OVERRIDE=${SMOKE_GATE_ROWS_PER_BENCHMARK:-}
SMOKE_MAX_STEPS=2
SMOKE_MAX_ROWS=128
FULL_MAX_STEPS=3000
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS_OVERRIDE:-${SMOKE_MAX_STEPS}}
FULL_MAX_STEPS=${FULL_MAX_STEPS_OVERRIDE:-${FULL_MAX_STEPS}}
SMOKE_MAX_ROWS=${SMOKE_MAX_ROWS_OVERRIDE:-${SMOKE_MAX_ROWS}}
SMOKE_VALIDATION_INTERVAL_STEPS=1
FULL_VALIDATION_INTERVAL_STEPS=400
SMOKE_VALIDATION_ROWS_PER_BENCHMARK=4
SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=1
SMOKE_GATE_ROWS_PER_BENCHMARK=1
SMOKE_VALIDATION_INTERVAL_STEPS=${SMOKE_VALIDATION_INTERVAL_STEPS_OVERRIDE:-${SMOKE_VALIDATION_INTERVAL_STEPS}}
FULL_VALIDATION_INTERVAL_STEPS=${FULL_VALIDATION_INTERVAL_STEPS_OVERRIDE:-${FULL_VALIDATION_INTERVAL_STEPS}}
SMOKE_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_VALIDATION_ROWS_PER_BENCHMARK_OVERRIDE:-${SMOKE_VALIDATION_ROWS_PER_BENCHMARK}}
SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK_OVERRIDE:-${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}}
SMOKE_GATE_ROWS_PER_BENCHMARK=${SMOKE_GATE_ROWS_PER_BENCHMARK_OVERRIDE:-${SMOKE_GATE_ROWS_PER_BENCHMARK}}
FULL_VALIDATION_ROWS_PER_BENCHMARK=256
FULL_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=128
FULL_GATE_ROWS_PER_BENCHMARK=1
SMOKE_BENCHMARK_CAPS=toolbench_g3=32:traject_bench=32:alfworld=32:webshop=32
FULL_BENCHMARK_CAPS=toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1

case "${FULL_RUN}" in
  0)
    MAX_STEPS=${MAX_STEPS:-${SMOKE_MAX_STEPS}}
    MAX_ROWS=${MAX_ROWS:-${SMOKE_MAX_ROWS}}
    VALIDATION_INTERVAL_STEPS=${SMOKE_VALIDATION_INTERVAL_STEPS}
    VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_VALIDATION_ROWS_PER_BENCHMARK}
    MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}
    GATE_ROWS_PER_BENCHMARK=${SMOKE_GATE_ROWS_PER_BENCHMARK}
    ;;
  1)
    MAX_STEPS=${FULL_MAX_STEPS}
    MAX_ROWS=
    VALIDATION_INTERVAL_STEPS=${FULL_VALIDATION_INTERVAL_STEPS}
    VALIDATION_ROWS_PER_BENCHMARK=${FULL_VALIDATION_ROWS_PER_BENCHMARK}
    MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${FULL_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}
    GATE_ROWS_PER_BENCHMARK=${FULL_GATE_ROWS_PER_BENCHMARK}
    ;;
  *)
    printf 'ERROR: FULL_RUN must be 0 or 1; got %s\n' "${FULL_RUN}" >&2
    exit 2
    ;;
esac

if [[ "${FULL_RUN}" == "0" ]]; then
  BENCHMARK_CAPS=${BENCHMARK_CAPS:-${SMOKE_BENCHMARK_CAPS}}
else
  BENCHMARK_CAPS=${BENCHMARK_CAPS:-${FULL_BENCHMARK_CAPS}}
fi

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
DATA_ROOT=${DATA_ROOT:-${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${ASSET_ROOT}/models/Qwen3-Embedding-0.6B}
STAGE0_SELECTION_PATH=${STAGE0_SELECTION_PATH:-${RUN_ROOT}/stage0_full/stage0_selection.json}
STAGE1_LINEAGE_PATH=${STAGE1_LINEAGE_PATH:-${RUN_ROOT}/stage1_full/lineage.json}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/stage2_anchored_full10000_v1}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-${STAGE2_OUTPUT_DIR}/checkpoints/clstr_full_base-step10000.pt}
STAGE2_QUALITY_GATE_PATH=${STAGE2_QUALITY_GATE_PATH:-${STAGE2_OUTPUT_DIR}/stage2_quality_gate.json}
STAGE2_LINEAGE_PATH=${STAGE2_LINEAGE_PATH:-${STAGE2_OUTPUT_DIR}/lineage.json}
if [[ "${FULL_RUN}" == "1" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_cmc_full}
else
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_cmc_smoke}
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
require_file "${STAGE1_LINEAGE_PATH}" "Stage1 lineage"
require_file "${STAGE2_CHECKPOINT_PATH}" "Stage2 checkpoint"
require_ok_gate "${STAGE2_QUALITY_GATE_PATH}" "Stage2 quality gate"
require_file "${STAGE2_LINEAGE_PATH}" "Stage2 lineage"

if [[ "${STAGE4_METHOD}" == "candidate_admission_residual_v1" ]]; then
  BASE_CMC_SELECTION_PATH=${BASE_CMC_SELECTION_PATH:-${RUN_ROOT}/stage4_cmc_full/stage4_selection.json}
  BASE_CMC_CHECKPOINT_PATH=$(
    "${PYTHON_BIN}" - "${BASE_CMC_SELECTION_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from clstr.qwen_clstr_lineage import sha256_path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"base CMC selection is not complete: {path}")
if payload.get("stage4_method") != "counterfactual_memory_calibration_v1":
    raise SystemExit(f"base selection is not CMC Stage4: {path}")
checkpoint = Path(str(payload.get("selected_checkpoint_path") or "")).resolve()
if sha256_path(checkpoint)["sha256"] != str(
    payload.get("selected_checkpoint_sha256") or ""
):
    raise SystemExit("base CMC checkpoint identity mismatch")
print(checkpoint)
PY
  )
  require_file "${BASE_CMC_CHECKPOINT_PATH}" "selected base CMC checkpoint"
  export BASE_CMC_CHECKPOINT_PATH
fi

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
  --manifest_path "${STAGE0_LINEAGE_PATH}" \
  --expected_stage stage0 \
  --expected_model_path "${MODEL_NAME_OR_PATH}" \
  --expected_skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --expected_data_manifest_path "${DATA_ROOT}/manifest.json"
"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
  --manifest_path "${STAGE2_LINEAGE_PATH}" \
  --expected_stage stage2 \
  --expected_parent "stage0=${STAGE0_LINEAGE_PATH}" \
  --expected_parent "stage1=${STAGE1_LINEAGE_PATH}" \
  --expected_model_path "${MODEL_NAME_OR_PATH}" \
  --expected_skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --expected_data_manifest_path "${DATA_ROOT}/manifest.json"

export TRAJECTORIES_PATH="${DATA_ROOT}/trajectories.jsonl"
export SKILLS_PATH="${DATA_ROOT}/skill_pool.jsonl"
export OUTPUT_DIR ROUTING_CHECKPOINT_PATH MAX_STEPS MAX_ROWS
export HEAD_CHECKPOINT_PATH="${STAGE2_CHECKPOINT_PATH}"
export BATCH_SIZE=16
export LEARNING_RATE=3.0e-5
export MINIMUM_LEARNING_RATE=3.0e-6
export LEARNING_RATE_WARMUP_FRACTION=0.05
export VALIDATION_INTERVAL_STEPS
export VALIDATION_ROWS_PER_BENCHMARK MINIMUM_VALIDATION_ROWS_PER_BENCHMARK
export GATE_ROWS_PER_BENCHMARK
export ROUTE_SCORER=unified_memory
export STAGE4_METHOD
export STAGE0_TOP_M=500
export STAGE0_INVENTORY_MIN_CANDIDATES=64
export STAGE0_POSITIVE_MISSING_POLICY=skip
export STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=8
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=100
export STAGE0_HANDOFF_CACHE_MODE=auto
export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"
export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1
export STAGE0_HANDOFF_CACHE_SHARD_SIZE=2048
export CANDIDATE_COUNT=64
export NEXT_SKILL_POOL_MODE=full_pool
export AUTO_REPLAY_PREFIX_MAX_STEPS=3
export TRAINABLE_REPLAY_PREFIX=0
export TRAIN_TRANSITION=0
export ALLOWED_BENCHMARKS=toolbench_g3,traject_bench,alfworld,webshop
export BENCHMARK_CAPS

mkdir -p "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage4] mode=%s steps=%s rows=%s output=%s\n' \
  "${FULL_RUN}" "${MAX_STEPS}" "${MAX_ROWS:-all}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage4] stage0=%s stage2=%s route=%s pool=%s\n' \
  "${ROUTING_CHECKPOINT_PATH}" "${HEAD_CHECKPOINT_PATH}" "${ROUTE_SCORER}" "${NEXT_SKILL_POOL_MODE}"

bash scripts/sbatch/run_clstr_unified_stage4_act_train.sh

DYNAMIC_SELECTION_PATH="${OUTPUT_DIR}/stage4_dynamic_selection.json"
FINAL_SELECTION_PATH="${OUTPUT_DIR}/stage4_selection.json"
STAGE2_BASELINE_REPORT_PATH="${OUTPUT_DIR}/validation/stage2_baseline_report.json"
QUALITY_GATE_PATH="${OUTPUT_DIR}/stage4_quality_gate.json"
LINEAGE_PATH="${OUTPUT_DIR}/lineage.json"
require_file "${DYNAMIC_SELECTION_PATH}" "Stage4 dynamic selection"
require_file "${STAGE2_BASELINE_REPORT_PATH}" "Stage2 validation baseline"

selection=$(
  "${PYTHON_BIN}" - "${DYNAMIC_SELECTION_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
digest_payload = dict(payload)
recorded = str(digest_payload.pop("manifest_sha256", ""))
if not recorded or canonical_digest(digest_payload) != recorded:
    raise SystemExit("Stage4 dynamic selection self-hash mismatch")
if payload.get("status") != "ok" or payload.get("release_status") not in {"ok", "action_required"}:
    raise SystemExit("Stage4 dynamic selection is invalid")
for key, label in (
    ("selected_checkpoint_path", "selected checkpoint"),
    ("validation_route_records", "validation route records"),
    ("gate_route_records", "gate route records"),
    ("gate_route_manifest", "gate route manifest"),
):
    value = payload.get(key)
    identity = value if isinstance(value, dict) else {"path": value, "sha256": payload.get("selected_checkpoint_sha256")}
    if not identity.get("path") or sha256_path(identity["path"])["sha256"] != str(identity.get("sha256") or ""):
        raise SystemExit(f"Stage4 {label} identity mismatch")
print("\t".join([
    str(payload["release_status"]),
    str(Path(payload["selected_checkpoint_path"]).resolve()),
    str(Path(payload["validation_route_records"]["path"]).resolve()),
    str(Path(payload["gate_route_records"]["path"]).resolve()),
    str(Path(payload["gate_route_manifest"]["path"]).resolve()),
]))
PY
)
IFS=$'\t' read -r DYNAMIC_RELEASE_STATUS CHECKPOINT_PATH VALIDATION_ROUTE_RECORDS_PATH GATE_ROUTE_RECORDS_PATH GATE_ROUTE_MANIFEST_PATH <<<"${selection}"
require_file "${CHECKPOINT_PATH}" "selected Stage4 checkpoint"

FINALIZE_ARGS=(
  scripts/finalize_clstr_stage4_selection.py
  --dynamic_selection_path "${DYNAMIC_SELECTION_PATH}"
  --output_path "${FINAL_SELECTION_PATH}"
)
"${PYTHON_BIN}" "${FINALIZE_ARGS[@]}"
require_file "${FINAL_SELECTION_PATH}" "final Stage4 selection"

AUDIT_ARGS=(
  scripts/audit_clstr_stage4_quality.py
  --output_dir "${OUTPUT_DIR}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_report_path "${OUTPUT_DIR}/train_report.json"
  --output_path "${QUALITY_GATE_PATH}"
  --min_steps "${MAX_STEPS}"
  --fail_on_action_required
)
if [[ "${FULL_RUN}" == "0" ]]; then
  AUDIT_ARGS+=(
    --first_window 1
    --last_window 1
    --min_stage4_recall_at_5_tail 0.0
    --max_stage4_act_loss_increase 1000000000
  )
else
  AUDIT_ARGS+=(
    --selection_path "${FINAL_SELECTION_PATH}"
    --stage2_baseline_report_path "${STAGE2_BASELINE_REPORT_PATH}"
  )
fi
"${PYTHON_BIN}" "${AUDIT_ARGS[@]}"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py create-derived \
  --stage stage4 \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --expected_checkpoint_stage clstr_stage4_transition_conditioned_next_skill \
  --model_path "${MODEL_NAME_OR_PATH}" \
  --skill_pool_path "${DATA_ROOT}/skill_pool.jsonl" \
  --data_manifest_path "${DATA_ROOT}/manifest.json" \
  --parent "stage0=${STAGE0_LINEAGE_PATH}" \
  --parent "stage2=${STAGE2_LINEAGE_PATH}" \
  --identity_parent_role stage2 \
  --stage4_selection_path "${FINAL_SELECTION_PATH}" \
  --output_path "${LINEAGE_PATH}"

require_file "${QUALITY_GATE_PATH}" "Stage4 quality gate"
require_file "${LINEAGE_PATH}" "Stage4 lineage"
