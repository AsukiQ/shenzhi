#!/bin/bash
#SBATCH --job-name=clstr_vnext_s0_full
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
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}
SEGMENT_END=${SEGMENT_END:?SEGMENT_END is required}
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
BACKBONE_SNAPSHOT_PATH=$(realpath -m "${BACKBONE_SNAPSHOT_PATH:-${RUN_ROOT}/backbone_snapshot.json}")
CACHE_ROOT=$(realpath -m "${CACHE_ROOT:-${RUN_ROOT}/frozen_qwen_cache}")
case "${CACHE_ROOT}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: writable CACHE_ROOT must remain inside RUN_ROOT: cache=%s run=%s\n' \
      "${CACHE_ROOT}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-${RUN_ROOT}/stage0}")
case "${OUTPUT_DIR}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: writable OUTPUT_DIR must remain inside RUN_ROOT: output=%s run=%s\n' \
      "${OUTPUT_DIR}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
eval "$("${PYTHON_BIN}" scripts/resolve_clstr_vnext_full_inputs.py "${DATA_MANIFEST_PATH}" --shell)"
resume_args=()
legacy_init_args=()
if [[ -f "${OUTPUT_DIR}/stage0_selection.json" ]]; then
  RESUME_CHECKPOINT=$("${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
selection = json.loads((root / "stage0_selection.json").read_text())
records = selection.get("validation_records") or []
if not records:
    raise SystemExit("Stage0 selection has no complete validation record")
step = int(records[-1]["step"])
path = root / "checkpoints" / f"clstr_vnext_stage0-step{step}.pt"
if not path.is_file():
    raise SystemExit(f"Stage0 completed-boundary checkpoint is missing: {path}")
print(path.resolve())
PY
  )
  resume_args=(--resume_checkpoint_path "${RESUME_CHECKPOINT}")
elif [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: Stage0 output exists without a completed selection boundary: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
elif [[ -n "${LEGACY_INIT_CHECKPOINT_PATH:-}" || -n "${LEGACY_INIT_SKILLS_PATH:-}" ]]; then
  if [[ -z "${LEGACY_INIT_CHECKPOINT_PATH:-}" || -z "${LEGACY_INIT_SKILLS_PATH:-}" ]]; then
    printf 'ERROR: legacy initialization requires both checkpoint and skills paths\n' >&2
    exit 2
  fi
  legacy_init_args=(
    --legacy_init_checkpoint_path "${LEGACY_INIT_CHECKPOINT_PATH}"
    --legacy_init_skills_path "${LEGACY_INIT_SKILLS_PATH}"
  )
fi

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage0_train.py \
  --skills_path "${TRAINING_SKILLS}" \
  --retrieval_rows_path "${RETRIEVAL_ROWS}" \
  --retrieval_dev_rows_path "${RETRIEVAL_DEV_ROWS}" \
  --static_route_rows_path "${STATIC_ROUTE_ROWS}" \
  --static_route_dev_rows_path "${STATIC_ROUTE_DEV_ROWS}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS}" \
  --data_contract_path "${DATA_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_steps "${SEGMENT_END}" \
  --batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2.0e-5 \
  --model_dim 1024 \
  --max_length 2048 \
  --torch_dtype bfloat16 \
  --skill_table_batch_size 64 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --skill_cache_shard_size 2048 \
  --frozen_cache_dir "${CACHE_ROOT}" \
  --backbone_snapshot_path "${BACKBONE_SNAPSHOT_PATH}" \
  --hard_negative_loss_weight 0.2 \
  --hard_negative_margin 0.1 \
  --hard_negative_top_k 32 \
  --seed 17 \
  --checkpoint_interval 500 \
  --validation_interval 500 \
  --validation_batch_size 128 \
  --max_dev_rows_per_kind 4096 \
  --minimum_dev_score_gain 0.0 \
  --require_clean_source \
  "${resume_args[@]}" \
  "${legacy_init_args[@]}"
