#!/bin/bash
#SBATCH --job-name=bge_sr_emb_full
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SOURCE_PROJECT_ROOT}}
if [[ ! -f "${PROJECT_ROOT}/scripts/run_bge_sr_embedding_full_train.py" ]]; then
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
DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}
PREPARED_CORPUS_DIR=${PREPARED_CORPUS_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_sr_embedding_full_train/smoke_${STAMP}}
ENCODER_MODEL=${ENCODER_MODEL:-${PROJECT_ROOT}/models/BAAI/bge-m3}

MAX_ROWS=${MAX_ROWS:-1024}
MAX_SKILLS=${MAX_SKILLS:-2048}
EVAL_ROWS=${EVAL_ROWS:-64}
MAX_STEPS=${MAX_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NEGATIVES_PER_QUERY=${NEGATIVES_PER_QUERY:-3}
HARD_NEGATIVE_TOP_K=${HARD_NEGATIVE_TOP_K:-8}
MAX_HARD_NEGATIVE_QUERIES=${MAX_HARD_NEGATIVE_QUERIES:-256}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TEMPERATURE=${TEMPERATURE:-0.05}
SEED=${SEED:-13}
MAX_LENGTH=${MAX_LENGTH:-256}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-32}
MINE_SCORE_BATCH_SIZE=${MINE_SCORE_BATCH_SIZE:-64}
TORCH_DTYPE=${TORCH_DTYPE:-float32}
USE_BF16_AUTOCAST=${USE_BF16_AUTOCAST:-0}
MINING_USE_BF16_AUTOCAST=${MINING_USE_BF16_AUTOCAST:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
EVAL_EVERY=${EVAL_EVERY:-5}
LOG_EVERY=${LOG_EVERY:-2}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
MAX_CHECKPOINTS_TO_KEEP=${MAX_CHECKPOINTS_TO_KEEP:-2}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
NONFINITE_GRADIENT_ACTION=${NONFINITE_GRADIENT_ACTION:-error}
MAX_NONFINITE_GRADIENT_FRACTION=${MAX_NONFINITE_GRADIENT_FRACTION:-0.0}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_bge_sr_embedding_full_train.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --encoder_model_path "${ENCODER_MODEL}"
  --eval_rows "${EVAL_ROWS}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --negatives_per_query "${NEGATIVES_PER_QUERY}"
  --hard_negative_top_k "${HARD_NEGATIVE_TOP_K}"
  --learning_rate "${LEARNING_RATE}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --max_length "${MAX_LENGTH}"
  --encode_batch_size "${ENCODE_BATCH_SIZE}"
  --mine_score_batch_size "${MINE_SCORE_BATCH_SIZE}"
  --torch_dtype "${TORCH_DTYPE}"
  --eval_every "${EVAL_EVERY}"
  --log_every "${LOG_EVERY}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
  --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP}"
  --nonfinite_gradient_action "${NONFINITE_GRADIENT_ACTION}"
  --max_nonfinite_gradient_fraction "${MAX_NONFINITE_GRADIENT_FRACTION}"
)

if [[ "${USE_BF16_AUTOCAST}" == "1" ]]; then
  args+=(--use_bf16_autocast)
else
  args+=(--no-use_bf16_autocast)
fi
if [[ "${MINING_USE_BF16_AUTOCAST}" == "1" ]]; then
  args+=(--mining_use_bf16_autocast)
else
  args+=(--no-mining_use_bf16_autocast)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  args+=(--gradient_checkpointing)
else
  args+=(--no-gradient_checkpointing)
fi

if [[ -n "${MAX_ROWS}" && "${MAX_ROWS}" != "ALL" ]]; then
  args+=(--max_rows "${MAX_ROWS}")
fi
if [[ -z "${PREPARED_CORPUS_DIR}" && -n "${MAX_SKILLS}" && "${MAX_SKILLS}" != "ALL" ]]; then
  args+=(--max_skills "${MAX_SKILLS}")
fi
if [[ -n "${PREPARED_CORPUS_DIR}" ]]; then
  args+=(--prepared_corpus_dir "${PREPARED_CORPUS_DIR}")
fi
if [[ -n "${MAX_HARD_NEGATIVE_QUERIES}" && "${MAX_HARD_NEGATIVE_QUERIES}" != "ALL" ]]; then
  args+=(--max_hard_negative_queries "${MAX_HARD_NEGATIVE_QUERIES}")
fi
if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  args+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

{
  printf '[bge-sr-emb-full] output_dir=%s\n' "${OUTPUT_DIR}"
  printf '[bge-sr-emb-full] data_root=%s encoder_model=%s\n' "${DATA_ROOT}" "${ENCODER_MODEL}"
  printf '[bge-sr-emb-full] prepared_corpus_dir=%s\n' "${PREPARED_CORPUS_DIR:-none}"
  printf '[bge-sr-emb-full] max_rows=%s max_skills=%s eval_rows=%s max_steps=%s\n' \
    "${MAX_ROWS:-ALL}" "${MAX_SKILLS:-ALL}" "${EVAL_ROWS}" "${MAX_STEPS}"
  printf '[bge-sr-emb-full] micro_batch=%s grad_accum=%s effective_batch=%s negatives=%s hard_top_k=%s lr=%s max_length=%s\n' \
    "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))" \
    "${NEGATIVES_PER_QUERY}" "${HARD_NEGATIVE_TOP_K}" "${LEARNING_RATE}" "${MAX_LENGTH}"
  printf '[bge-sr-emb-full] dtype=%s autocast=%s mining_autocast=%s gradient_checkpointing=%s\n' \
    "${TORCH_DTYPE}" "${USE_BF16_AUTOCAST}" "${MINING_USE_BF16_AUTOCAST}" "${GRADIENT_CHECKPOINTING}"
  printf '[bge-sr-emb-full] resume_checkpoint_path=%s\n' "${RESUME_CHECKPOINT_PATH:-none}"
  printf '[bge-sr-emb-full] checkpoint_every=%s max_checkpoints_to_keep=%s\n' "${CHECKPOINT_EVERY}" "${MAX_CHECKPOINTS_TO_KEEP}"
  printf '[bge-sr-emb-full] nonfinite_gradient_action=%s max_nonfinite_gradient_fraction=%s\n' \
    "${NONFINITE_GRADIENT_ACTION}" "${MAX_NONFINITE_GRADIENT_FRACTION}"
} | tee "${OUTPUT_DIR}/stdout.log"

"${args[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log"
