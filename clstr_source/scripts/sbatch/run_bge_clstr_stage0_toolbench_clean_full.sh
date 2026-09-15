#!/bin/bash
#SBATCH --job-name=bge_clstr_s0
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
export DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_clstr_stage0_toolbench_clean_full_${STAMP}}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/models/BAAI/bge-m3}

export ENCODER_POOLING=${ENCODER_POOLING:-cls}
export CROSS_ENCODER_POOLING=${CROSS_ENCODER_POOLING:-cls}
export TOKENIZER_PADDING_SIDE=${TOKENIZER_PADDING_SIDE:-right}
export QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-raw}

export TOP_K=${TOP_K:-350}
export MAX_STEPS=${MAX_STEPS:-5000}
export BATCH_SIZE=${BATCH_SIZE:-64}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export MODEL_DIM=${MODEL_DIM:-1024}
export LEARNING_RATE=${LEARNING_RATE:-1.0e-5}
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export MAX_LENGTH=${MAX_LENGTH:-512}
export SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-64}

export ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
export BELIEF_TOP_K=${BELIEF_TOP_K:-64}
export RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}
export SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}
export TEMPERED_CORRECTION_FRACTION=${TEMPERED_CORRECTION_FRACTION:-0.2}
export EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}

export TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}
export TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}
export TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-1}
export TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}
export TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}
export PRESERVATION_ANCHOR_WEIGHT=${PRESERVATION_ANCHOR_WEIGHT:-0.0}

echo "[bge-clstr-stage0] output_dir=${OUTPUT_DIR}"
echo "[bge-clstr-stage0] model=${MODEL_NAME_OR_PATH}"
echo "[bge-clstr-stage0] pooling=${ENCODER_POOLING}/${CROSS_ENCODER_POOLING} padding=${TOKENIZER_PADDING_SIDE} query=${QUERY_TEXT_FORMAT}"
echo "[bge-clstr-stage0] batch=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} max_steps=${MAX_STEPS} route_scorer=${ROUTE_SCORER}"

exec bash scripts/sbatch/run_clstr_unified_stage0_function_aug_v2_toolbench_clean_full.sh
