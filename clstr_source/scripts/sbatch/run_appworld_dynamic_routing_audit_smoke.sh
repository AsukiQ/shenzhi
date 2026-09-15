#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -p gpu_a800
#SBATCH --gpus=1
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
DYNAMIC_SKILL_POOL_PATH=${DYNAMIC_SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}
RETRIEVAL_PATH=${RETRIEVAL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/retrieval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_dynamic_routing_audit/smoke_max32}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/report.json}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-${OUTPUT_DIR}/model_cache}
MAX_PAIRS=${MAX_PAIRS:-32}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-stride}
BATCH_SIZE=${BATCH_SIZE:-4}
K_VALUES=${K_VALUES:-"1 5 10 20 50 100 350"}
DEVICE=${DEVICE:-auto}
QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-auto}

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: AppWorld dynamic routing audit requires Stage0 checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -s "${BASE_SKILL_POOL_PATH}" ]]; then
  echo "ERROR: missing base skill pool: ${BASE_SKILL_POOL_PATH}" >&2
  exit 2
fi
if [[ ! -s "${DYNAMIC_SKILL_POOL_PATH}" ]]; then
  echo "ERROR: missing dynamic skill pool; run scripts/build_clstr_appworld_current_route.py first: ${DYNAMIC_SKILL_POOL_PATH}" >&2
  exit 2
fi
if [[ ! -s "${RETRIEVAL_PATH}" ]]; then
  echo "ERROR: missing AppWorld retrieval rows: ${RETRIEVAL_PATH}" >&2
  exit 2
fi

K_VALUES="${K_VALUES//:/ }"
K_VALUES="${K_VALUES//,/ }"
read -r -a K_VALUE_ARGS <<< "${K_VALUES}"

mkdir -p "${OUTPUT_DIR}" "${MODEL_CACHE_DIR}"
rm -f "${OUTPUT_PATH}" "${OUTPUT_PATH}.progress.json" "${OUTPUT_DIR}/audit_stdout.json"

"${PYTHON_BIN}" scripts/audit_appworld_dynamic_routing.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --base_skill_pool_path "${BASE_SKILL_POOL_PATH}" \
  --dynamic_skill_pool_path "${DYNAMIC_SKILL_POOL_PATH}" \
  --retrieval_path "${RETRIEVAL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model_cache_dir "${MODEL_CACHE_DIR}" \
  --max_pairs "${MAX_PAIRS}" \
  --sampling_strategy "${SAMPLING_STRATEGY}" \
  --batch_size "${BATCH_SIZE}" \
  --k_values "${K_VALUE_ARGS[@]}" \
  --device "${DEVICE}" \
  --query_text_format "${QUERY_TEXT_FORMAT}" | tee "${OUTPUT_DIR}/audit_stdout.json"
