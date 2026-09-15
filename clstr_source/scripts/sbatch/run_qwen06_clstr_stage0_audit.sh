#!/bin/bash
#SBATCH --job-name=qwen06_s0_audit
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
TARGET_STEP=${TARGET_STEP:-1200}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-${RUN_ROOT}/stage0_full}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage0_audits/step${TARGET_STEP}}
BASELINE_REPORT=${BASELINE_REPORT:-}
PREVIOUS_REPORT=${PREVIOUS_REPORT:-}

case "${TARGET_STEP}" in
  1200|2400|3600|5000)
    ;;
  *)
    printf 'ERROR: TARGET_STEP must be one of 1200, 2400, 3600, 5000; got %s\n' "${TARGET_STEP}" >&2
    exit 2
    ;;
esac

require_file() {
  local path=$1
  local label=$2
  if [[ ! -s "${path}" ]]; then
    printf 'ERROR: missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
}

validate_lineage() {
  local path=$1
  require_file "${path}" "Stage0 lineage"
  "${PYTHON_BIN}" scripts/audit_qwen06_clstr_lineage.py validate \
    --manifest_path "${path}" \
    --expected_stage stage0
}

run_handoff_audit() {
  local checkpoint_path=$1
  local audit_output_dir=$2
  mkdir -p "${audit_output_dir}/model_cache"
  "${PYTHON_BIN}" scripts/audit_clstr_stage0_handoff_coverage.py \
    --checkpoint_path "${checkpoint_path}" \
    --train_path "${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/trajectories.jsonl" \
    --skills_path "${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl" \
    --output_path "${audit_output_dir}/report.json" \
    --top_k_values "20,50,100,200,500" \
    --query_modes checkpoint_state_query \
    --max_rows 2048 \
    --batch_size 8 \
    --model_cache_dir "${audit_output_dir}/model_cache" \
    2>&1 | tee "${audit_output_dir}/stdout.log"
}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step${TARGET_STEP}.pt}
LINEAGE_PATH=${LINEAGE_PATH:-${STAGE0_OUTPUT_DIR}/lineage-step${TARGET_STEP}.json}
require_file "${CHECKPOINT_PATH}" "target Stage0 checkpoint"
validate_lineage "${LINEAGE_PATH}"

if [[ "${TARGET_STEP}" == "1200" ]]; then
  STEP0_CHECKPOINT_PATH="${STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step0.pt"
  STEP0_LINEAGE_PATH="${STAGE0_OUTPUT_DIR}/lineage-step0.json"
  BASELINE_OUTPUT_DIR="${RUN_ROOT}/stage0_audits/step0"
  BASELINE_REPORT="${BASELINE_OUTPUT_DIR}/report.json"
  PREVIOUS_REPORT="${BASELINE_REPORT}"
  require_file "${STEP0_CHECKPOINT_PATH}" "Stage0 step-0 checkpoint"
  validate_lineage "${STEP0_LINEAGE_PATH}"
  run_handoff_audit "${STEP0_CHECKPOINT_PATH}" "${BASELINE_OUTPUT_DIR}"
else
  if [[ -z "${BASELINE_REPORT}" || -z "${PREVIOUS_REPORT}" ]]; then
    printf 'ERROR: later audits require explicit BASELINE_REPORT and PREVIOUS_REPORT\n' >&2
    exit 2
  fi
  require_file "${BASELINE_REPORT}" "step-0 baseline report"
  require_file "${PREVIOUS_REPORT}" "previous-segment report"
fi

run_handoff_audit "${CHECKPOINT_PATH}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_qwen06_clstr_stage0_gate.py promote \
  --baseline_report "${BASELINE_REPORT}" \
  --current_report "${OUTPUT_DIR}/report.json" \
  --previous_report "${PREVIOUS_REPORT}" \
  --target_step "${TARGET_STEP}" \
  --output_path "${OUTPUT_DIR}/promotion_gate.json"
