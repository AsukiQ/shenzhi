#!/bin/bash
#SBATCH --job-name=bge_sr_rank
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SOURCE_PROJECT_ROOT}}
if [[ ! -f "${PROJECT_ROOT}/scripts/run_bge_sr_reranker_train.py" ]]; then
  printf 'ERROR: PROJECT_ROOT is not a CLSTR checkout: %s\n' "${PROJECT_ROOT}" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SR_EMBEDDING_OUTPUT_DIR=${SR_EMBEDDING_OUTPUT_DIR:-outputs/bge_sr_embedding_full_train/full_b64_trainfp32_miningbf16_latest}
SR_EMBEDDING_CHECKPOINT_PATH=${SR_EMBEDDING_CHECKPOINT_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_sr_reranker_train/smoke_${STAMP}}
RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH:-${PROJECT_ROOT}/models/BAAI/bge-reranker-v2-m3}

MAX_STEPS=${MAX_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
TOP_K=${TOP_K:-20}
ENCODER_MAX_LENGTH=${ENCODER_MAX_LENGTH:-512}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-64}
RANK_BATCH_SIZE=${RANK_BATCH_SIZE:-256}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-512}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SEED=${SEED:-13}
EVAL_EVERY=${EVAL_EVERY:-10}
LOG_EVERY=${LOG_EVERY:-5}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10}
MAX_CHECKPOINTS_TO_KEEP=${MAX_CHECKPOINTS_TO_KEEP:-2}
MAX_TRAIN_QUERIES=${MAX_TRAIN_QUERIES:-1024}
MAX_EVAL_QUERIES=${MAX_EVAL_QUERIES:-128}
MAX_TRAIN_GROUPS=${MAX_TRAIN_GROUPS:-256}
MAX_EVAL_GROUPS=${MAX_EVAL_GROUPS:-64}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-0}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
CANDIDATE_CACHE_DIR=${CANDIDATE_CACHE_DIR:-${PROJECT_ROOT}/outputs/shared_exact_candidate_cache}

mkdir -p "${OUTPUT_DIR}"
if [[ -z "${SR_EMBEDDING_CHECKPOINT_PATH}" ]]; then
  echo "[bge-sr-rank] ERROR: SR_EMBEDDING_CHECKPOINT_PATH must point to a full-finetuned BGE-SR-Emb HuggingFace checkpoint directory" | tee "${OUTPUT_DIR}/stdout.log"
  exit 2
fi

args=(
  "${PYTHON_BIN}" scripts/run_bge_sr_reranker_train.py
  --sr_embedding_output_dir "${SR_EMBEDDING_OUTPUT_DIR}"
  --sr_embedding_checkpoint_path "${SR_EMBEDDING_CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --reranker_model_path "${RERANKER_MODEL_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --top_k "${TOP_K}"
  --encoder_max_length "${ENCODER_MAX_LENGTH}"
  --encoder_batch_size "${ENCODER_BATCH_SIZE}"
  --rank_batch_size "${RANK_BATCH_SIZE}"
  --reranker_max_length "${RERANKER_MAX_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
  --eval_every "${EVAL_EVERY}"
  --log_every "${LOG_EVERY}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
  --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP}"
  --candidate_cache_dir "${CANDIDATE_CACHE_DIR}"
)

if [[ -n "${MAX_TRAIN_QUERIES}" && "${MAX_TRAIN_QUERIES}" != "ALL" ]]; then
  args+=(--max_train_queries "${MAX_TRAIN_QUERIES}")
fi
if [[ -n "${MAX_EVAL_QUERIES}" && "${MAX_EVAL_QUERIES}" != "ALL" ]]; then
  args+=(--max_eval_queries "${MAX_EVAL_QUERIES}")
fi
if [[ -n "${MAX_TRAIN_GROUPS}" && "${MAX_TRAIN_GROUPS}" != "ALL" ]]; then
  args+=(--max_train_groups "${MAX_TRAIN_GROUPS}")
fi
if [[ -n "${MAX_EVAL_GROUPS}" && "${MAX_EVAL_GROUPS}" != "ALL" ]]; then
  args+=(--max_eval_groups "${MAX_EVAL_GROUPS}")
fi
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  args+=(--freeze_backbone)
fi
if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  args+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

{
  printf '[bge-sr-rank] output_dir=%s\n' "${OUTPUT_DIR}"
  printf '[bge-sr-rank] sr_embedding_output_dir=%s\n' "${SR_EMBEDDING_OUTPUT_DIR}"
  printf '[bge-sr-rank] sr_embedding_checkpoint=%s reranker=%s\n' "${SR_EMBEDDING_CHECKPOINT_PATH}" "${RERANKER_MODEL_PATH}"
  printf '[bge-sr-rank] top_k=%s max_steps=%s batch_size=%s lr=%s\n' "${TOP_K}" "${MAX_STEPS}" "${BATCH_SIZE}" "${LEARNING_RATE}"
  printf '[bge-sr-rank] max_train_queries=%s max_train_groups=%s max_eval_groups=%s freeze_backbone=%s\n' \
    "${MAX_TRAIN_QUERIES:-ALL}" "${MAX_TRAIN_GROUPS:-ALL}" "${MAX_EVAL_GROUPS:-ALL}" "${FREEZE_BACKBONE}"
  printf '[bge-sr-rank] resume_checkpoint_path=%s\n' "${RESUME_CHECKPOINT_PATH:-none}"
  printf '[bge-sr-rank] checkpoint_every=%s max_checkpoints_to_keep=%s\n' "${CHECKPOINT_EVERY}" "${MAX_CHECKPOINTS_TO_KEEP}"
  printf '[bge-sr-rank] candidate_cache_dir=%s\n' "${CANDIDATE_CACHE_DIR}"
} | tee "${OUTPUT_DIR}/stdout.log"

"${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log"
