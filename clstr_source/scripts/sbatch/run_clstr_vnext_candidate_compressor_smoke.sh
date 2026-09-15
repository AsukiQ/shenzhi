#!/bin/bash
#SBATCH --job-name=clstr_vnext_static_smoke
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT must come from the freshly audited causal manifest}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:?STAGE0_OUTPUT_DIR must be the matching Stage0 smoke}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be a fresh static-reranker smoke directory}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR must identify the matching smoke cache}
MAX_STEPS=${MAX_STEPS:-50}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-25}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-${VALIDATION_INTERVAL}}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-512}
MAX_DEV_ROWS=${MAX_DEV_ROWS:-128}

for path in \
  "${BUNDLE_ROOT}/trajectory_rows.jsonl" \
  "${BUNDLE_ROOT}/trajectory_dev_rows.jsonl" \
  "${BUNDLE_ROOT}/inventory_catalogs.jsonl" \
  "${BUNDLE_ROOT}/manifest.json" \
  "${STAGE0_OUTPUT_DIR}/train_report.json" \
  "${STAGE0_OUTPUT_DIR}/stage0_selection.json" \
  "${STAGE0_OUTPUT_DIR}/selected_skills.jsonl"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: missing static-reranker smoke input: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: static-reranker smoke output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

STAGE0_CHECKPOINT_PATH=$("${PYTHON_BIN}" - "${STAGE0_OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root / "train_report.json").read_text())
selection = json.loads((root / "stage0_selection.json").read_text())
if report.get("status") != "ok" or report.get("stage") != "clstr_vnext_stage0":
    raise SystemExit("static-reranker smoke requires a completed Stage0 smoke")
if selection.get("status") != "ok" or report.get("selection") != selection:
    raise SystemExit("static-reranker smoke requires a quality-gated Stage0 smoke")
step = int(report.get("step") or 0)
checkpoint = Path(str(report.get("checkpoint_path") or ""))
if step <= 0 or checkpoint.name != f"clstr_vnext_stage0-step{step}.pt":
    raise SystemExit("static-reranker smoke Stage0 report does not name its final checkpoint")
if not checkpoint.is_file():
    raise SystemExit("static-reranker smoke Stage0 checkpoint is missing")
print(checkpoint.resolve())
PY
)

"${PYTHON_BIN}" scripts/run_clstr_vnext_candidate_compressor_train.py \
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}" \
  --skills_path "${STAGE0_OUTPUT_DIR}/selected_skills.jsonl" \
  --trajectory_rows_path "${BUNDLE_ROOT}/trajectory_rows.jsonl" \
  --trajectory_dev_rows_path "${BUNDLE_ROOT}/trajectory_dev_rows.jsonl" \
  --inventory_catalogs_path "${BUNDLE_ROOT}/inventory_catalogs.jsonl" \
  --data_contract_path "${BUNDLE_ROOT}/manifest.json" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-4 \
  --belief_top_k 64 \
  --coarse_k 500 \
  --compressed_m 500 \
  --cache_batch_size 64 \
  --cache_shard_size 64 \
  --frozen_cache_dir "${FROZEN_CACHE_DIR}" \
  --seed 37 \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --validation_interval "${VALIDATION_INTERVAL}" \
  --validation_batch_size 16 \
  --max_train_rows "${MAX_TRAIN_ROWS}" \
  --max_dev_rows "${MAX_DEV_ROWS}" \
  --minimum_dev_score_gain -1.0 \
  --coverage_margin 0.05 \
  --recoverable_row_weight 4.0 \
  --listwise_loss_weight 0.05 \
  --objective_mode static_route_query_residual \
  --static_reranker_scale_initial 1.0 \
  --require_clean_source \
  | tee "${OUTPUT_DIR}.stdout.log"
