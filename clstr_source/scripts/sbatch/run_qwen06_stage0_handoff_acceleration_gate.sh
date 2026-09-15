#!/bin/bash
#SBATCH --job-name=qwen06_handoff_gate
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

TRAIN_PROJECT_ROOT=${TRAIN_PROJECT_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix}
TRAIN_RUN_ROOT=${TRAIN_RUN_ROOT:-${TRAIN_PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
DATA_ROOT=${DATA_ROOT:-${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
STAGE0_SELECTION_PATH=${STAGE0_SELECTION_PATH:-${TRAIN_RUN_ROOT}/stage0_full/stage0_selection.json}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix/handoff_acceleration_gate}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/handoff_acceleration_gate.json}
ROW_COUNT=${ROW_COUNT:-2048}
TOP_M=${TOP_M:-500}
INVENTORY_MIN_CANDIDATES=${INVENTORY_MIN_CANDIDATES:-64}
QUERY_MODE=${QUERY_MODE:-checkpoint_state_query}
LEGACY_BATCH_SIZE=${LEGACY_BATCH_SIZE:-16}
CANDIDATE_BATCH_SIZES=${CANDIDATE_BATCH_SIZES:-16,64,128}
CACHE_SHARD_SIZE=${CACHE_SHARD_SIZE:-2048}
SCORE_ATOL=${SCORE_ATOL:-1.0e-5}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}

if [[ ! -s "${STAGE0_SELECTION_PATH}" ]]; then
  printf 'ERROR: missing Stage0 selection: %s\n' "${STAGE0_SELECTION_PATH}" >&2
  exit 2
fi

CHECKPOINT_PATH=$(
  "${PYTHON_BIN}" - "${STAGE0_SELECTION_PATH}" "${TRAIN_PROJECT_ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
payload = json.loads(selection_path.read_text(encoding="utf-8"))
if payload.get("status") != "ok" or payload.get("release_status") != "ok":
    raise SystemExit(f"Stage0 selection is not release-safe: {selection_path}")
checkpoint = Path(str(payload.get("selected_checkpoint_path") or ""))
if not checkpoint.is_absolute():
    checkpoint = project_root / checkpoint
if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
    raise SystemExit(f"selected Stage0 checkpoint is missing: {checkpoint}")
print(checkpoint)
PY
)

TRAIN_PATH=${TRAIN_PATH:-${DATA_ROOT}/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-${DATA_ROOT}/skill_pool.jsonl}
for required_path in "${CHECKPOINT_PATH}" "${TRAIN_PATH}" "${SKILLS_PATH}"; do
  if [[ ! -s "${required_path}" ]]; then
    printf 'ERROR: missing handoff gate input: %s\n' "${required_path}" >&2
    exit 2
  fi
done

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export TRAIN_ENCODER_BACKBONE=0

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/audit_qwen06_stage0_handoff_acceleration.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --row_count "${ROW_COUNT}" \
  --top_m "${TOP_M}" \
  --inventory_min_candidates "${INVENTORY_MIN_CANDIDATES}" \
  --query_mode "${QUERY_MODE}" \
  --legacy_batch_size "${LEGACY_BATCH_SIZE}" \
  --candidate_batch_sizes "${CANDIDATE_BATCH_SIZES}" \
  --cache_shard_size "${CACHE_SHARD_SIZE}" \
  --score_atol "${SCORE_ATOL}" \
  --allowed_benchmarks "${ALLOWED_BENCHMARKS}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
