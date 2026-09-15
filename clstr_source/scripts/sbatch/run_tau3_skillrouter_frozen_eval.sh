#!/bin/bash
#SBATCH --job-name=tau3_skillrouter
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:45:00
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

DATA_ROOT=${DATA_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench/data/tau2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-outputs/unified_skillrouter_finetune/function_aug_v2_full_20260628_212251/checkpoints/unified_skillrouter_finetune-step2000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/tau3_skillrouter_finetuned_eval/smoke_${SLURM_JOB_ID:-manual}}

DOMAINS=${DOMAINS:-airline,retail,telecom,banking_knowledge}
MAX_TASKS_PER_DOMAIN=${MAX_TASKS_PER_DOMAIN:-20}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-128}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_tau3_skillrouter_frozen_eval.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --domains "${DOMAINS}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)

if [[ -n "${MAX_TASKS_PER_DOMAIN}" && "${MAX_TASKS_PER_DOMAIN}" != "ALL" ]]; then
  args+=(--max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ -n "${CANDIDATE_COUNT:-}" && "${CANDIDATE_COUNT}" != "ALL" ]]; then
  args+=(--candidate_count "${CANDIDATE_COUNT}")
fi
if [[ -n "${ADAPTER_CHECKPOINT_PATH}" ]]; then
  args+=(--adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}")
fi

printf '[tau3-skillrouter-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[tau3-skillrouter-eval] domains=%s max_tasks_per_domain=%s max_eval_rows=%s\n' "${DOMAINS}" "${MAX_TASKS_PER_DOMAIN:-ALL}" "${MAX_EVAL_ROWS:-ALL}"
printf '[tau3-skillrouter-eval] adapter_checkpoint_path=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
