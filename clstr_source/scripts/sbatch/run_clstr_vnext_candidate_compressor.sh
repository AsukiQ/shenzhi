#!/bin/bash
#SBATCH --job-name=clstr_vnext_compressor
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
MAX_STEPS=${MAX_STEPS:-1200}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-${MAX_STEPS}}
COVERAGE_MARGIN=${COVERAGE_MARGIN:-0.05}
RECOVERABLE_ROW_WEIGHT=${RECOVERABLE_ROW_WEIGHT:-4.0}
LISTWISE_LOSS_WEIGHT=${LISTWISE_LOSS_WEIGHT:-0.05}
STATIC_RERANKER_SCALE_INITIAL=${STATIC_RERANKER_SCALE_INITIAL:-1.0}
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
CACHE_ROOT=$(realpath -m "${CACHE_ROOT:-${RUN_ROOT}/frozen_qwen_cache}")
case "${CACHE_ROOT}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: writable CACHE_ROOT must remain inside RUN_ROOT: cache=%s run=%s\n' \
      "${CACHE_ROOT}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-${RUN_ROOT}/stage0}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/candidate_compressor}
STAGE0_OUTPUT_DIR=$(realpath -m "${STAGE0_OUTPUT_DIR}")
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR}")
case "${OUTPUT_DIR}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: compressor OUTPUT_DIR must remain inside RUN_ROOT: output=%s run=%s\n' \
      "${OUTPUT_DIR}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
eval "$("${PYTHON_BIN}" scripts/resolve_clstr_vnext_full_inputs.py "${DATA_MANIFEST_PATH}" --shell)"

readarray -t STAGE0_SELECTED < <("${PYTHON_BIN}" - "${STAGE0_OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
report = json.loads((root / "train_report.json").read_text())
selection = json.loads((root / "stage0_selection.json").read_text())
if report.get("status") != "ok" or report.get("stage") != "clstr_vnext_stage0":
    raise SystemExit("candidate compressor requires a completed canonical Stage0 report")
if selection.get("status") != "ok" or report.get("selection") != selection:
    raise SystemExit("candidate compressor requires a quality-gated Stage0 checkpoint")
step = int(report.get("step") or 0)
checkpoint = Path(str(report.get("checkpoint_path") or ""))
skills = root / "selected_skills.jsonl"
if step <= 0 or checkpoint.name != f"clstr_vnext_stage0-step{step}.pt":
    raise SystemExit("candidate compressor Stage0 report does not name its final checkpoint")
if not checkpoint.is_file() or not skills.is_file():
    raise SystemExit("candidate compressor Stage0 handoff is incomplete")
print(checkpoint.resolve())
print(skills.resolve())
PY
)
STAGE0_CHECKPOINT_PATH=${STAGE0_SELECTED[0]}
STAGE0_SKILLS_PATH=${STAGE0_SELECTED[1]}

if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: candidate compressor output already exists; use a fresh run root: %s\n' \
    "${OUTPUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/run_clstr_vnext_candidate_compressor_train.py \
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}" \
  --skills_path "${STAGE0_SKILLS_PATH}" \
  --trajectory_rows_path "${TRAJECTORY_ROWS}" \
  --trajectory_dev_rows_path "${TRAJECTORY_DEV_ROWS}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS}" \
  --data_contract_path "${DATA_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE}" \
  --belief_top_k 64 \
  --coarse_k 500 \
  --compressed_m 500 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --frozen_cache_dir "${CACHE_ROOT}" \
  --seed 37 \
  --checkpoint_interval "${VALIDATION_INTERVAL}" \
  --validation_interval "${VALIDATION_INTERVAL}" \
  --validation_batch_size 16 \
  --max_dev_rows 4096 \
  --minimum_dev_score_gain 0.0 \
  --coverage_margin "${COVERAGE_MARGIN}" \
  --recoverable_row_weight "${RECOVERABLE_ROW_WEIGHT}" \
  --listwise_loss_weight "${LISTWISE_LOSS_WEIGHT}" \
  --objective_mode static_route_query_residual \
  --static_reranker_scale_initial "${STATIC_RERANKER_SCALE_INITIAL}" \
  --require_clean_source
