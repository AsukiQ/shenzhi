#!/bin/bash
#SBATCH --job-name=bge_smoke
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_backend_smoke/full_${TIMESTAMP}}
EMBEDDING_MODEL_NAME_OR_PATH=${EMBEDDING_MODEL_NAME_OR_PATH:-models/BAAI/bge-m3}
RERANKER_MODEL_NAME_OR_PATH=${RERANKER_MODEL_NAME_OR_PATH:-models/BAAI/bge-reranker-v2-m3}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-2}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-512}
RERANKER_BATCH_SIZE=${RERANKER_BATCH_SIZE:-2}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-512}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKIP_RERANKER=${SKIP_RERANKER:-0}

mkdir -p "${OUTPUT_DIR}"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

args=(
  "${PYTHON_BIN}" scripts/run_bge_backend_smoke.py
  --embedding_model_name_or_path "${EMBEDDING_MODEL_NAME_OR_PATH}"
  --reranker_model_name_or_path "${RERANKER_MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}"
  --embedding_max_length "${EMBEDDING_MAX_LENGTH}"
  --reranker_batch_size "${RERANKER_BATCH_SIZE}"
  --reranker_max_length "${RERANKER_MAX_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
)

if [[ "${SKIP_RERANKER}" == "1" ]]; then
  args+=(--skip_reranker)
fi

printf '[bge-smoke] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[bge-smoke] embedding=%s reranker=%s skip_reranker=%s\n' "${EMBEDDING_MODEL_NAME_OR_PATH}" "${RERANKER_MODEL_NAME_OR_PATH}" "${SKIP_RERANKER}"
printf '[bge-smoke] batch embedding=%s reranker=%s dtype=%s\n' "${EMBEDDING_BATCH_SIZE}" "${RERANKER_BATCH_SIZE}" "${TORCH_DTYPE}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
