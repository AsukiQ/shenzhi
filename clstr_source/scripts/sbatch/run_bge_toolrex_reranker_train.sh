#!/bin/bash
#SBATCH --job-name=bge_tool_rank
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SOURCE_PROJECT_ROOT}}
if [[ ! -f "${PROJECT_ROOT}/scripts/run_bge_toolrex_reranker_train.py" ]]; then
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
TOOL_EMBED_OUTPUT_DIR=${TOOL_EMBED_OUTPUT_DIR:-}
TOOL_EMBED_CHECKPOINT_PATH=${TOOL_EMBED_CHECKPOINT_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_toolrex_reranker_train/full_b64_pairwise_${STAMP}}
RERANKER_MODEL=${RERANKER_MODEL:-${PROJECT_ROOT}/models/BAAI/bge-reranker-v2-m3}

MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-1.0e-5}
TOP_K=${TOP_K:-20}
NEGATIVES_PER_QUERY=${NEGATIVES_PER_QUERY:-5}
ENCODER_MAX_LENGTH=${ENCODER_MAX_LENGTH:-512}
ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE:-64}
RANK_BATCH_SIZE=${RANK_BATCH_SIZE:-256}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-512}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SEED=${SEED:-13}
EVAL_EVERY=${EVAL_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-10}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-400}
MAX_CHECKPOINTS_TO_KEEP=${MAX_CHECKPOINTS_TO_KEEP:-2}
MAX_TRAIN_QUERIES=${MAX_TRAIN_QUERIES:-}
MAX_EVAL_QUERIES=${MAX_EVAL_QUERIES:-}
MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS:-}
MAX_EVAL_PAIRS=${MAX_EVAL_PAIRS:-}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-0}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
CANDIDATE_CACHE_DIR=${CANDIDATE_CACHE_DIR:-${PROJECT_ROOT}/outputs/shared_exact_candidate_cache}

if [[ -z "${TOOL_EMBED_OUTPUT_DIR}" ]]; then
  echo "ERROR: TOOL_EMBED_OUTPUT_DIR is required" >&2
  exit 2
fi
if [[ -z "${TOOL_EMBED_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: TOOL_EMBED_CHECKPOINT_PATH is required" >&2
  exit 2
fi
if [[ ! -d "${TOOL_EMBED_OUTPUT_DIR}" ]]; then
  echo "ERROR: TOOL_EMBED_OUTPUT_DIR does not exist: ${TOOL_EMBED_OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${TOOL_EMBED_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: TOOL_EMBED_CHECKPOINT_PATH does not exist: ${TOOL_EMBED_CHECKPOINT_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_bge_toolrex_reranker_train.py
  --tool_embed_output_dir "${TOOL_EMBED_OUTPUT_DIR}"
  --tool_embed_checkpoint_path "${TOOL_EMBED_CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --reranker_model_path "${RERANKER_MODEL}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --top_k "${TOP_K}"
  --negatives_per_query "${NEGATIVES_PER_QUERY}"
  --encoder_max_length "${ENCODER_MAX_LENGTH}"
  --encoder_batch_size "${ENCODER_BATCH_SIZE}"
  --rank_batch_size "${RANK_BATCH_SIZE}"
  --reranker_max_length "${RERANKER_MAX_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
  --eval_every "${EVAL_EVERY}"
  --log_every "${LOG_EVERY}"
  --checkpoint_every "${CHECKPOINT_EVERY}"
  --max_checkpoints_to_keep "${MAX_CHECKPOINTS_TO_KEEP}"
  --candidate_cache_dir "${CANDIDATE_CACHE_DIR}"
)

if [[ -n "${MAX_TRAIN_QUERIES}" ]]; then
  args+=(--max_train_queries "${MAX_TRAIN_QUERIES}")
fi
if [[ -n "${MAX_EVAL_QUERIES}" ]]; then
  args+=(--max_eval_queries "${MAX_EVAL_QUERIES}")
fi
if [[ -n "${MAX_TRAIN_PAIRS}" ]]; then
  args+=(--max_train_pairs "${MAX_TRAIN_PAIRS}")
fi
if [[ -n "${MAX_EVAL_PAIRS}" ]]; then
  args+=(--max_eval_pairs "${MAX_EVAL_PAIRS}")
fi
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  args+=(--freeze_backbone)
fi
if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  args+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

printf '[bge-toolrex-tool-rank] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[bge-toolrex-tool-rank] tool_embed_checkpoint=%s\n' "${TOOL_EMBED_CHECKPOINT_PATH}"
printf '[bge-toolrex-tool-rank] batch=%s max_steps=%s objective=true_false_pair_bce\n' "${BATCH_SIZE}" "${MAX_STEPS}"
printf '[bge-toolrex-tool-rank] checkpoint_every=%s max_checkpoints_to_keep=%s\n' "${CHECKPOINT_EVERY}" "${MAX_CHECKPOINTS_TO_KEEP}"
printf '[bge-toolrex-tool-rank] candidate_cache_dir=%s\n' "${CANDIDATE_CACHE_DIR}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
