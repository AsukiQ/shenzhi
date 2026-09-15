#!/bin/bash
#SBATCH --job-name=toolsandbox_skillrouter
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

SCENARIOS_ROOT=${SCENARIOS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios}
TOOLS_ROOT=${TOOLS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-outputs/unified_skillrouter_finetune/function_aug_v2_full_20260628_212251/checkpoints/unified_skillrouter_finetune-step2000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolsandbox_skillrouter_finetuned_eval/smoke_${SLURM_JOB_ID:-manual}}

MAX_SCENARIOS=${MAX_SCENARIOS:-20}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-128}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_toolsandbox_skillrouter_frozen_eval.py
  --scenarios_root "${SCENARIOS_ROOT}"
  --tools_root "${TOOLS_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)

if [[ -n "${MAX_SCENARIOS}" && "${MAX_SCENARIOS}" != "ALL" ]]; then
  args+=(--max_scenarios "${MAX_SCENARIOS}")
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

printf '[toolsandbox-skillrouter-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[toolsandbox-skillrouter-eval] max_scenarios=%s max_eval_rows=%s\n' "${MAX_SCENARIOS:-ALL}" "${MAX_EVAL_ROWS:-ALL}"
printf '[toolsandbox-skillrouter-eval] adapter_checkpoint_path=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
