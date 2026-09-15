#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -p gpu_a800
#SBATCH --gpus=1
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl}
DYNAMIC_SKILL_POOL_PATH=${DYNAMIC_SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_2_nowweak_append/skill_pool.jsonl}
RETRIEVAL_PATH=${RETRIEVAL_PATH:-outputs/appworld_current_route_online_stage4_smoke/stage0_online_correction_6171bbc_3_20260617_222301/stage0_retrieval_corrections.jsonl}
TASKS_PATH=${TASKS_PATH:-data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl}
APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_dynamic_controller_candidate_audit/v4_2_nowweak_anchor_w1e3_appworld6171_candidate20}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/report.json}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-${OUTPUT_DIR}/model_cache}
MAX_PAIRS=${MAX_PAIRS:-6}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-head}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-20}
TOP_K=${TOP_K:-5}
DEVICE=${DEVICE:-auto}
QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-auto}
STATE_TEXT_SOURCE=${STATE_TEXT_SOURCE:-retrieval_query}
RANKING_MODE=${RANKING_MODE:-skill_table}
CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing}
APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: controller candidate audit requires Stage0 checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -s "${BASE_SKILL_POOL_PATH}" ]]; then
  echo "ERROR: missing base skill pool: ${BASE_SKILL_POOL_PATH}" >&2
  exit 2
fi
if [[ ! -s "${DYNAMIC_SKILL_POOL_PATH}" ]]; then
  echo "ERROR: missing dynamic skill pool: ${DYNAMIC_SKILL_POOL_PATH}" >&2
  exit 2
fi
if [[ ! -s "${RETRIEVAL_PATH}" ]]; then
  echo "ERROR: missing retrieval/correction rows: ${RETRIEVAL_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${MODEL_CACHE_DIR}"
rm -f "${OUTPUT_PATH}" "${OUTPUT_PATH}.progress.json" "${OUTPUT_DIR}/audit_stdout.json"

EXTRA_ARGS=()
if [[ "${APPWORLD_EXECUTOR_COMPATIBLE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--appworld_executor_compatible_only)
fi

"${PYTHON_BIN}" scripts/audit_appworld_dynamic_controller_candidates.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --base_skill_pool_path "${BASE_SKILL_POOL_PATH}" \
  --dynamic_skill_pool_path "${DYNAMIC_SKILL_POOL_PATH}" \
  --retrieval_path "${RETRIEVAL_PATH}" \
  --tasks_path "${TASKS_PATH}" \
  --appworld_root "${APPWORLD_ROOT}" \
  --output_path "${OUTPUT_PATH}" \
  --model_cache_dir "${MODEL_CACHE_DIR}" \
  --max_pairs "${MAX_PAIRS}" \
  --sampling_strategy "${SAMPLING_STRATEGY}" \
  --candidate_top_k "${CANDIDATE_TOP_K}" \
  --top_k "${TOP_K}" \
  --device "${DEVICE}" \
  --query_text_format "${QUERY_TEXT_FORMAT}" \
  --state_text_source "${STATE_TEXT_SOURCE}" \
  --ranking_mode "${RANKING_MODE}" \
  --candidate_source "${CANDIDATE_SOURCE}" \
  "${EXTRA_ARGS[@]}" | tee "${OUTPUT_DIR}/audit_stdout.json"
