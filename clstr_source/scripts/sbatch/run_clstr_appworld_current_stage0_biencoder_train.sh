#!/bin/bash
#SBATCH --time=04:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export DATA_ROOT=${DATA_ROOT:-data/clstr_appworld_current_route_v1}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_appworld_current_stage0_biencoder}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
export MAX_STEPS=${MAX_STEPS:-2000}
export BATCH_SIZE=${BATCH_SIZE:-64}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export MODEL_DIM=${MODEL_DIM:-1024}
export TOP_K=${TOP_K:-100}
export LEARNING_RATE=${LEARNING_RATE:-2.0e-5}
export RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}
export SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}
export EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}
export TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}
export TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}
export TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}
export TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-0}
export TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-0}

exec bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
