#!/bin/bash
#SBATCH --job-name=qwen06_clstr_s0
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

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
FULL_RUN=${FULL_RUN:-0}
TARGET_STEP=${TARGET_STEP:-}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
PREVIOUS_GATE_PATH=${PREVIOUS_GATE_PATH:-}
SMOKE_GATE_PATH=${SMOKE_GATE_PATH:-${RUN_ROOT}/stage0_smoke/smoke_gate.json}

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
  local expected_step=${2:-}
  require_file "${path}" "quality gate"
  "${PYTHON_BIN}" - "${path}" "${expected_step}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_step = sys.argv[2].strip()
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"quality gate is not ok: {path}")
if expected_step:
    observed = payload.get("target_step")
    if observed is not None and int(observed) != int(expected_step):
        raise SystemExit(
            f"quality gate target_step mismatch: expected {expected_step}, got {observed}"
        )
PY
}

case "${FULL_RUN}" in
  0)
    TARGET_STEP=${TARGET_STEP:-80}
    if [[ "${TARGET_STEP}" != "80" ]]; then
      printf 'ERROR: smoke mode fixes TARGET_STEP=80; got %s\n' "${TARGET_STEP}" >&2
      exit 2
    fi
    if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
      printf 'ERROR: smoke mode cannot resume from a checkpoint\n' >&2
      exit 2
    fi
    OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage0_smoke}
    MAX_SKILLS=8192
    MAX_QUERIES=4096
    ;;
  1)
    TARGET_STEP=${TARGET_STEP:-1200}
    case "${TARGET_STEP}" in
      1200|2400|3600|5000)
        ;;
      *)
        printf 'ERROR: FULL_RUN=1 requires TARGET_STEP in {1200,2400,3600,5000}; got %s\n' "${TARGET_STEP}" >&2
        exit 2
        ;;
    esac
    OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage0_full}
    MAX_SKILLS=
    MAX_QUERIES=
    if [[ "${TARGET_STEP}" == "1200" ]]; then
      if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
        printf 'ERROR: TARGET_STEP=1200 must start from fresh Qwen/CLSTR initialization\n' >&2
        exit 2
      fi
      require_ok_gate "${SMOKE_GATE_PATH}"
    else
      case "${TARGET_STEP}" in
        2400) PREVIOUS_STEP=1200 ;;
        3600) PREVIOUS_STEP=2400 ;;
        5000) PREVIOUS_STEP=3600 ;;
      esac
      EXPECTED_RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step${PREVIOUS_STEP}.pt"
      if [[ -z "${RESUME_CHECKPOINT_PATH}" ]]; then
        printf 'ERROR: TARGET_STEP=%s requires explicit RESUME_CHECKPOINT_PATH\n' "${TARGET_STEP}" >&2
        exit 2
      fi
      if [[ "$(readlink -f "${RESUME_CHECKPOINT_PATH}")" != "$(readlink -f "${EXPECTED_RESUME_CHECKPOINT}")" ]]; then
        printf 'ERROR: resume checkpoint must be the canonical step-%s checkpoint: %s\n' "${PREVIOUS_STEP}" "${EXPECTED_RESUME_CHECKPOINT}" >&2
        exit 2
      fi
      require_file "${RESUME_CHECKPOINT_PATH}" "resume checkpoint"
      if [[ -z "${PREVIOUS_GATE_PATH}" ]]; then
        printf 'ERROR: TARGET_STEP=%s requires explicit PREVIOUS_GATE_PATH\n' "${TARGET_STEP}" >&2
        exit 2
      fi
      require_ok_gate "${PREVIOUS_GATE_PATH}" "${PREVIOUS_STEP}"
      PREVIOUS_LINEAGE_PATH="${OUTPUT_DIR}/lineage-step${PREVIOUS_STEP}.json"
      require_file "${PREVIOUS_LINEAGE_PATH}" "previous Stage0 lineage"
      "${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
        --manifest_path "${PREVIOUS_LINEAGE_PATH}" \
        --expected_stage stage0
    fi
    ;;
  *)
    printf 'ERROR: FULL_RUN must be 0 or 1; got %s\n' "${FULL_RUN}" >&2
    exit 2
    ;;
esac

export DATA_ROOT="${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2"
export MODEL_NAME_OR_PATH="${ASSET_ROOT}/models/Qwen3-Embedding-0.6B"
export OUTPUT_DIR
export MAX_STEPS="${TARGET_STEP}"
export BATCH_SIZE=128
export GRADIENT_ACCUMULATION_STEPS=2
export MODEL_DIM=1024
export TOP_K=100
export MAX_SKILLS
export MAX_QUERIES
export LEARNING_RATE=2.0e-5
export TORCH_DTYPE=bfloat16
export MAX_LENGTH=2048
export SKILL_TABLE_BATCH_SIZE=32
export ENCODER_POOLING=last_token
export CROSS_ENCODER_POOLING=last_token
export TOKENIZER_PADDING_SIDE=left
export QUERY_TEXT_FORMAT=skillrouter
export STATE_QUERY_PROMPT_VERSION=clstr_causal_state_v1
export STATE_QUERY_MAX_CHARS=2000
export STATE_QUERY_TRUNCATION=head_tail_v1
export ROUTE_SCORER=unified_memory
export BELIEF_TOP_K=64
export RETRIEVAL_LOSS_MODE=multi_positive_nll
export SAMPLING_STRATEGY=handoff_balanced
export EXPAND_ALIAS_POSITIVES=1
export TRAIN_SKILL_EMBEDDINGS=0
export TRAIN_SKILL_BIAS=1
export TRAIN_ENCODER_BACKBONE=0
export FROZEN_BACKBONE_CACHE_MODE=${FROZEN_BACKBONE_CACHE_MODE:-schedule}
export FROZEN_BACKBONE_CACHE_BATCH_SIZE=${FROZEN_BACKBONE_CACHE_BATCH_SIZE:-128}
if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  export RESUME_SKILL_TABLE_MODE=verified_checkpoint
else
  export RESUME_SKILL_TABLE_MODE=rebuild
fi
export TRAIN_ENCODER_PROJECTION=1
export TRAIN_SKILL_ADAPTER=1
export TRAIN_RETRIEVAL_SCALE=1
export EXPLICIT_NEGATIVE_LOSS_WEIGHT=0.0
export MINED_HARD_NEGATIVE_LOSS_WEIGHT=0.2
export MINED_HARD_NEGATIVE_MARGIN=0.1
export MINED_HARD_NEGATIVE_TOP_K=32
export INIT_CHECKPOINT_PATH=
export INIT_CHECKPOINT_SKILLS_PATH=
export RESUME_CHECKPOINT_PATH
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

mkdir -p "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage0] project_root=%s\n' "${PROJECT_ROOT}"
printf '[qwen06-clstr-stage0] mode=%s target_step=%s output_dir=%s\n' "${FULL_RUN}" "${TARGET_STEP}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-stage0] model=%s data=%s backbone_train=%s route=%s\n' \
  "${MODEL_NAME_OR_PATH}" "${DATA_ROOT}" "${TRAIN_ENCODER_BACKBONE}" "${ROUTE_SCORER}"

bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"

TRAIN_REPORT_PATH="${OUTPUT_DIR}/train_report.json"
STEP0_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step0.pt"
TARGET_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step${TARGET_STEP}.pt"
STEP0_LINEAGE_PATH="${OUTPUT_DIR}/lineage-step0.json"
TARGET_LINEAGE_PATH="${OUTPUT_DIR}/lineage-step${TARGET_STEP}.json"
SKILL_POOL_PATH="${DATA_ROOT}/skill_pool.jsonl"
DATA_MANIFEST_PATH="${DATA_ROOT}/manifest.json"

require_file "${TRAIN_REPORT_PATH}" "Stage0 train report"
require_file "${STEP0_CHECKPOINT_PATH}" "Stage0 step-0 checkpoint"
require_file "${TARGET_CHECKPOINT_PATH}" "Stage0 target checkpoint"

create_lineage() {
  local checkpoint_path=$1
  local output_path=$2
  "${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py create \
    --stage stage0 \
    --checkpoint_path "${checkpoint_path}" \
    --expected_checkpoint_stage clstr_unified_retrieval_v2 \
    --model_path "${MODEL_NAME_OR_PATH}" \
    --skill_pool_path "${SKILL_POOL_PATH}" \
    --data_manifest_path "${DATA_MANIFEST_PATH}" \
    --output_path "${output_path}"
}

create_lineage "${STEP0_CHECKPOINT_PATH}" "${STEP0_LINEAGE_PATH}"
create_lineage "${TARGET_CHECKPOINT_PATH}" "${TARGET_LINEAGE_PATH}"

if [[ "${FULL_RUN}" == "0" ]]; then
  "${PYTHON_BIN}" scripts/audit_qwen06_clstr_stage0_gate.py smoke \
    --train_report_path "${TRAIN_REPORT_PATH}" \
    --lineage_manifest_path "${TARGET_LINEAGE_PATH}" \
    --output_path "${OUTPUT_DIR}/smoke_gate.json"
fi
