#!/bin/bash
#SBATCH --job-name=bge_sr_emb
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_sr_embedding_train/smoke_${STAMP}}
ENCODER_MODEL=${ENCODER_MODEL:-${PROJECT_ROOT}/models/BAAI/bge-m3}

MAX_ROWS=${MAX_ROWS:-4096}
MAX_SKILLS=${MAX_SKILLS:-8192}
EVAL_ROWS=${EVAL_ROWS:-256}
MAX_STEPS=${MAX_STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TEMPERATURE=${TEMPERATURE:-0.05}
SEED=${SEED:-13}
ENCODER_MAX_LENGTH=${ENCODER_MAX_LENGTH:-2048}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-32}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TOKENIZER_PADDING_SIDE=${TOKENIZER_PADDING_SIDE:-auto}
PROJECTION_INIT=${PROJECTION_INIT:-identity}
TRAIN_DOC_PROJECTION=${TRAIN_DOC_PROJECTION:-0}
EVAL_EVERY=${EVAL_EVERY:-50}
LOG_EVERY=${LOG_EVERY:-10}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-400}
POOLING=${POOLING:-auto}
QUERY_TEXT_MODE=${QUERY_TEXT_MODE:-auto}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_unified_skillrouter_finetune.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --encoder_model_path "${ENCODER_MODEL}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --encoder_max_length "${ENCODER_MAX_LENGTH}"
  --encoder_batch_size "${ENCODER_BATCH_SIZE}"
  --torch_dtype "${TORCH_DTYPE}"
  --tokenizer_padding_side "${TOKENIZER_PADDING_SIDE}"
  --projection_init "${PROJECTION_INIT}"
  --eval_rows "${EVAL_ROWS}"
  --eval_every "${EVAL_EVERY}"
  --log_every "${LOG_EVERY}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
  --pooling "${POOLING}"
  --query_text_mode "${QUERY_TEXT_MODE}"
)

if [[ -n "${MAX_ROWS}" && "${MAX_ROWS}" != "ALL" ]]; then
  args+=(--max_rows "${MAX_ROWS}")
fi
if [[ -n "${MAX_SKILLS}" && "${MAX_SKILLS}" != "ALL" ]]; then
  args+=(--max_skills "${MAX_SKILLS}")
fi
if [[ "${TRAIN_DOC_PROJECTION}" == "1" ]]; then
  args+=(--train_doc_projection)
fi

{
  printf '[bge-sr-emb] output_dir=%s\n' "${OUTPUT_DIR}"
  printf '[bge-sr-emb] data_root=%s encoder_model=%s\n' "${DATA_ROOT}" "${ENCODER_MODEL}"
  printf '[bge-sr-emb] max_rows=%s max_skills=%s eval_rows=%s max_steps=%s\n' \
    "${MAX_ROWS:-ALL}" "${MAX_SKILLS:-ALL}" "${EVAL_ROWS}" "${MAX_STEPS}"
  printf '[bge-sr-emb] batch=%s encoder_batch=%s lr=%s checkpoint_every=%s\n' \
    "${BATCH_SIZE}" "${ENCODER_BATCH_SIZE}" "${LEARNING_RATE}" "${CHECKPOINT_EVERY}"
  printf '[bge-sr-emb] pooling=%s query_text_mode=%s padding_side=%s\n' \
    "${POOLING}" "${QUERY_TEXT_MODE}" "${TOKENIZER_PADDING_SIDE}"
} | tee "${OUTPUT_DIR}/stdout.log"

"${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log"
