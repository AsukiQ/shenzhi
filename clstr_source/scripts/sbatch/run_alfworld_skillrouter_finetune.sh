#!/bin/bash
#SBATCH --job-name=alf_sr_ft
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

REPLAY_PATH=${REPLAY_PATH:-data/alfworld_policy_replay/train_replay.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/alfworld_skillrouter_finetune/smoke}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
MAX_STEPS=${MAX_STEPS:-128}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TEMPERATURE=${TEMPERATURE:-0.05}
SEED=${SEED:-13}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
PROJECTION_INIT=${PROJECTION_INIT:-identity}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}

if [[ ! -s "${REPLAY_PATH}" ]]; then
  echo "ERROR: missing ALFWorld replay path: ${REPLAY_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "ERROR: missing SkillRouter embedding model: ${MODEL_NAME_OR_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
echo "[alf-sr-ft] output_dir=${OUTPUT_DIR}"
echo "[alf-sr-ft] replay_path=${REPLAY_PATH} max_steps=${MAX_STEPS} batch_size=${BATCH_SIZE}"

"${PYTHON_BIN}" scripts/run_alfworld_skillrouter_finetune.py \
  --replay_path "${REPLAY_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --eval_fraction "${EVAL_FRACTION}" \
  --projection_init "${PROJECTION_INIT}" \
  --encode_batch_size "${ENCODE_BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
