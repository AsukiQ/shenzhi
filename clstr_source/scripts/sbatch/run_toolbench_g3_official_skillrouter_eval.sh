#!/bin/bash
#SBATCH --job-name=tb_g3_skillrouter_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

SKILLROUTER_REPO=${SKILLROUTER_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter}
DATA_ROOT=${DATA_ROOT:-outputs/toolbench_g3_official_skillrouter_comparison/smoke_data/eval}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3_official_skillrouter_comparison/official_skillrouter_frozen_smoke}
ENCODER_MODEL=${ENCODER_MODEL:-${PROJECT_ROOT}/.cache/hf_models/SkillRouter-Embedding-0.6B}
RERANKER_MODEL=${RERANKER_MODEL:-${PROJECT_ROOT}/.cache/hf_models/SkillRouter-Reranker-0.6B}
RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K:-20}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-16}
RERANKER_BATCH_SIZE=${RERANKER_BATCH_SIZE:-4}
ENCODER_MAX_LENGTH=${ENCODER_MAX_LENGTH:-4096}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-4096}
PROMPT_FORMAT=${PROMPT_FORMAT:-flat-full}
TIERS=${TIERS:-easy}

mkdir -p "${OUTPUT_DIR}"
cd "${SKILLROUTER_REPO}"

"${PYTHON_BIN}" -m src.run_open_model_eval \
  --data_root "${PROJECT_ROOT}/${DATA_ROOT}" \
  --encoder_model_or_path "${ENCODER_MODEL}" \
  --reranker_model_or_path "${RERANKER_MODEL}" \
  --tiers ${TIERS} \
  --retrieval_top_k "${RETRIEVAL_TOP_K}" \
  --encoder_batch_size "${ENCODER_BATCH_SIZE}" \
  --reranker_batch_size "${RERANKER_BATCH_SIZE}" \
  --encoder_max_length "${ENCODER_MAX_LENGTH}" \
  --reranker_max_length "${RERANKER_MAX_LENGTH}" \
  --prompt_format "${PROMPT_FORMAT}" \
  --output_dir "${PROJECT_ROOT}/${OUTPUT_DIR}" \
  | tee "${PROJECT_ROOT}/${OUTPUT_DIR}/stdout.json"

