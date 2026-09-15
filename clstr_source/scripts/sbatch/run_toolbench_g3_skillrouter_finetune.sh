#!/bin/bash
#SBATCH --job-name=tb_g3_sr_ft
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3_official_skillrouter_comparison/skillrouter_finetune_smoke}
ENCODER_MODEL=${ENCODER_MODEL:-${PROJECT_ROOT}/.cache/hf_models/SkillRouter-Embedding-0.6B}
RERANKER_MODEL=${RERANKER_MODEL:-${PROJECT_ROOT}/.cache/hf_models/SkillRouter-Reranker-0.6B}
SKILLROUTER_REPO=${SKILLROUTER_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter}

MAX_TRAIN_TASKS=${MAX_TRAIN_TASKS:-128}
MAX_EVAL_TASKS=${MAX_EVAL_TASKS:-128}
MAX_SKILLS=${MAX_SKILLS:-2048}
MAX_STEPS=${MAX_STEPS:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TEMPERATURE=${TEMPERATURE:-0.05}
SEED=${SEED:-13}
ENCODER_MAX_LENGTH=${ENCODER_MAX_LENGTH:-4096}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-4096}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-16}
RERANKER_BATCH_SIZE=${RERANKER_BATCH_SIZE:-8}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K:-20}
PROMPT_FORMAT=${PROMPT_FORMAT:-flat-full}
PROJECTION_INIT=${PROJECTION_INIT:-identity}
TRAIN_DOC_PROJECTION=${TRAIN_DOC_PROJECTION:-0}
QUERY_VARIANT=${QUERY_VARIANT:-history}
LOG_EVERY=${LOG_EVERY:-10}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_toolbench_g3_skillrouter_finetune.py
  --data_root "${DATA_ROOT}"
  --encoder_model_path "${ENCODER_MODEL}"
  --reranker_model_path "${RERANKER_MODEL}"
  --official_repo_path "${SKILLROUTER_REPO}"
  --output_dir "${OUTPUT_DIR}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --encoder_max_length "${ENCODER_MAX_LENGTH}"
  --reranker_max_length "${RERANKER_MAX_LENGTH}"
  --encoder_batch_size "${ENCODER_BATCH_SIZE}"
  --reranker_batch_size "${RERANKER_BATCH_SIZE}"
  --torch_dtype "${TORCH_DTYPE}"
  --retrieval_top_k "${RETRIEVAL_TOP_K}"
  --prompt_format "${PROMPT_FORMAT}"
  --projection_init "${PROJECTION_INIT}"
  --query_variant "${QUERY_VARIANT}"
  --log_every "${LOG_EVERY}"
)

if [[ -n "${MAX_TRAIN_TASKS}" && "${MAX_TRAIN_TASKS}" != "ALL" ]]; then
  args+=(--max_train_tasks "${MAX_TRAIN_TASKS}")
fi
if [[ -n "${MAX_EVAL_TASKS}" && "${MAX_EVAL_TASKS}" != "ALL" ]]; then
  args+=(--max_eval_tasks "${MAX_EVAL_TASKS}")
fi
if [[ -n "${MAX_SKILLS}" && "${MAX_SKILLS}" != "ALL" ]]; then
  args+=(--max_skills "${MAX_SKILLS}")
fi
if [[ "${TRAIN_DOC_PROJECTION}" == "1" ]]; then
  args+=(--train_doc_projection)
fi

printf '[skillrouter-ft-sbatch] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[skillrouter-ft-sbatch] max_train_tasks=%s max_eval_tasks=%s max_skills=%s max_steps=%s\n' \
  "${MAX_TRAIN_TASKS:-ALL}" "${MAX_EVAL_TASKS:-ALL}" "${MAX_SKILLS:-ALL}" "${MAX_STEPS}"
printf '[skillrouter-ft-sbatch] train_doc_projection=%s\n' "${TRAIN_DOC_PROJECTION}"
printf '[skillrouter-ft-sbatch] query_variant=%s\n' "${QUERY_VARIANT}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
