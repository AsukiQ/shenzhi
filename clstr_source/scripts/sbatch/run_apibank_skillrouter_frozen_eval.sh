#!/bin/bash
#SBATCH --job-name=apibank_skillrouter
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
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

DATA_ROOT=${DATA_ROOT:-.tmp/benchmark_probe_direct/liminghao1630__API-Bank}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/apibank_skillrouter_frozen_eval/smoke_${SLURM_JOB_ID:-manual}}
FILES=${FILES:-}
INCLUDE_TRIVIAL=${INCLUDE_TRIVIAL:-0}
MAX_ROWS_PER_FILE=${MAX_ROWS_PER_FILE:-ALL}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-64}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_apibank_skillrouter_frozen_eval.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)
if [[ -n "${ADAPTER_CHECKPOINT_PATH}" ]]; then
  args+=(--adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}")
fi
if [[ -n "${FILES}" ]]; then
  args+=(--files "${FILES}")
fi
if [[ "${INCLUDE_TRIVIAL}" == "1" || "${INCLUDE_TRIVIAL}" == "true" ]]; then
  args+=(--include_trivial)
fi
if [[ -n "${MAX_ROWS_PER_FILE}" && "${MAX_ROWS_PER_FILE}" != "ALL" ]]; then
  args+=(--max_rows_per_file "${MAX_ROWS_PER_FILE}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[apibank-skillrouter-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[apibank-skillrouter-eval] adapter_checkpoint=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"
printf '[apibank-skillrouter-eval] max_rows_per_file=%s max_eval_rows=%s include_trivial=%s\n' "${MAX_ROWS_PER_FILE:-ALL}" "${MAX_EVAL_ROWS:-ALL}" "${INCLUDE_TRIVIAL}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
