#!/bin/bash
#SBATCH --job-name=alf_sr_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
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

OFFICIAL_REPO=${OFFICIAL_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo}
DATA_DIR=${DATA_DIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_data}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-}
SPLIT=${SPLIT:-valid_seen}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/alfworld_eval/skillrouter_frozen_admissible_smoke_${SLURM_JOB_ID:-manual}}
RUN_NAME=${RUN_NAME:-skillrouter_frozen_admissible}
MAX_EPISODES=${MAX_EPISODES:-5}
MAX_STEPS=${MAX_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_alfworld_skillrouter_eval.py
  --official_repo "${OFFICIAL_REPO}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --split "${SPLIT}"
  --run_name "${RUN_NAME}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --encode_batch_size "${ENCODE_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)
if [[ -n "${ADAPTER_CHECKPOINT_PATH}" ]]; then
  args+=(--adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}")
fi
if [[ -n "${MAX_EPISODES}" && "${MAX_EPISODES}" != "ALL" ]]; then
  args+=(--max_episodes "${MAX_EPISODES}")
fi

printf '[alfworld-skillrouter-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[alfworld-skillrouter-eval] split=%s max_episodes=%s max_steps=%s\n' "${SPLIT}" "${MAX_EPISODES:-ALL}" "${MAX_STEPS}"
printf '[alfworld-skillrouter-eval] adapter_checkpoint_path=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
