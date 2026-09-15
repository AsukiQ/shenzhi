#!/bin/bash
#SBATCH --job-name=qwen06_s0_cache
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

ASSET_ROOT=${ASSET_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage0_cache_benchmark}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${RUN_ROOT}/stage0_full/checkpoints/clstr_unified_retrieval_v2-step1200.pt}
SKILLS_PATH=${SKILLS_PATH:-${RUN_ROOT}/stage0_full/selected_skills.jsonl}
DATA_ROOT=${DATA_ROOT:-${ASSET_ROOT}/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/cache_benchmark.json}
TARGET_STEP=${TARGET_STEP:-2400}
BATCH_SIZE=${BATCH_SIZE:-128}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-3}
TIMED_ITERATIONS=${TIMED_ITERATIONS:-10}
LEARNING_RATE=${LEARNING_RATE:-2.0e-5}
SEED=${SEED:-13}

export TRAIN_ENCODER_BACKBONE=0
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

for required_path in "${CHECKPOINT_PATH}" "${SKILLS_PATH}" "${DATA_ROOT}/skill_pool.jsonl" "${DATA_ROOT}/retrieval.jsonl"; do
  if [[ ! -s "${required_path}" ]]; then
    printf 'ERROR: missing cache benchmark input: %s\n' "${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_stage0_frozen_backbone_cache.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_path "${OUTPUT_PATH}" \
  --target_step "${TARGET_STEP}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --cache_batch_size "${CACHE_BATCH_SIZE}" \
  --warmup_iterations "${WARMUP_ITERATIONS}" \
  --timed_iterations "${TIMED_ITERATIONS}" \
  --learning_rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
