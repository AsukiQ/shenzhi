#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-128}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
MODEL_DIM=${MODEL_DIM:-1024}
TOP_K=${TOP_K:-100}
MAX_SKILLS=${MAX_SKILLS:-}
MAX_QUERIES=${MAX_QUERIES:-}
LEARNING_RATE=${LEARNING_RATE:-2.0e-5}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
MAX_LENGTH=${MAX_LENGTH:-2048}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}
ENCODER_POOLING=${ENCODER_POOLING:-last_token}
CROSS_ENCODER_POOLING=${CROSS_ENCODER_POOLING:-last_token}
TOKENIZER_PADDING_SIDE=${TOKENIZER_PADDING_SIDE:-left}
QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-skillrouter}
STATE_QUERY_PROMPT_VERSION=${STATE_QUERY_PROMPT_VERSION:-}
STATE_QUERY_MAX_CHARS=${STATE_QUERY_MAX_CHARS:-}
STATE_QUERY_TRUNCATION=${STATE_QUERY_TRUNCATION:-}
RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}
TEMPERED_CORRECTION_FRACTION=${TEMPERED_CORRECTION_FRACTION:-0.2}
EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}
TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}
TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-0}
TRAIN_ENCODER_BACKBONE=${TRAIN_ENCODER_BACKBONE:-0}
ENCODER_BACKBONE_LEARNING_RATE=${ENCODER_BACKBONE_LEARNING_RATE:-}
TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-0}
TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}
TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}
INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-}
INIT_CHECKPOINT_SKILLS_PATH=${INIT_CHECKPOINT_SKILLS_PATH:-}
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}
FROZEN_BACKBONE_CACHE_MODE=${FROZEN_BACKBONE_CACHE_MODE:-off}
FROZEN_BACKBONE_CACHE_BATCH_SIZE=${FROZEN_BACKBONE_CACHE_BATCH_SIZE:-128}
RESUME_SKILL_TABLE_MODE=${RESUME_SKILL_TABLE_MODE:-rebuild}
PRESERVATION_ANCHOR_WEIGHT=${PRESERVATION_ANCHOR_WEIGHT:-0.0}
EXPLICIT_NEGATIVE_LOSS_WEIGHT=${EXPLICIT_NEGATIVE_LOSS_WEIGHT:-0.0}
EXPLICIT_NEGATIVE_MARGIN=${EXPLICIT_NEGATIVE_MARGIN:-0.1}
MINED_HARD_NEGATIVE_LOSS_WEIGHT=${MINED_HARD_NEGATIVE_LOSS_WEIGHT:-0.0}
MINED_HARD_NEGATIVE_MARGIN=${MINED_HARD_NEGATIVE_MARGIN:-0.1}
MINED_HARD_NEGATIVE_TOP_K=${MINED_HARD_NEGATIVE_TOP_K:-32}
ROUTE_SCORER=${ROUTE_SCORER:-legacy_biencoder}
BELIEF_TOP_K=${BELIEF_TOP_K:-64}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/run_clstr_stage0_biencoder_train.py
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --model_dim "${MODEL_DIM}"
  --top_k "${TOP_K}"
  --learning_rate "${LEARNING_RATE}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_length "${MAX_LENGTH}"
  --skill_table_batch_size "${SKILL_TABLE_BATCH_SIZE}"
  --encoder_pooling "${ENCODER_POOLING}"
  --cross_encoder_pooling "${CROSS_ENCODER_POOLING}"
  --tokenizer_padding_side "${TOKENIZER_PADDING_SIDE}"
  --query_text_format "${QUERY_TEXT_FORMAT}"
  --retrieval_loss_mode "${RETRIEVAL_LOSS_MODE}"
  --sampling_strategy "${SAMPLING_STRATEGY}"
  --tempered_correction_fraction "${TEMPERED_CORRECTION_FRACTION}"
  --preservation_anchor_weight "${PRESERVATION_ANCHOR_WEIGHT}"
  --explicit_negative_loss_weight "${EXPLICIT_NEGATIVE_LOSS_WEIGHT}"
  --explicit_negative_margin "${EXPLICIT_NEGATIVE_MARGIN}"
  --mined_hard_negative_loss_weight "${MINED_HARD_NEGATIVE_LOSS_WEIGHT}"
  --mined_hard_negative_margin "${MINED_HARD_NEGATIVE_MARGIN}"
  --mined_hard_negative_top_k "${MINED_HARD_NEGATIVE_TOP_K}"
  --route_scorer "${ROUTE_SCORER}"
  --belief_top_k "${BELIEF_TOP_K}"
  --frozen_backbone_cache_mode "${FROZEN_BACKBONE_CACHE_MODE}"
  --frozen_backbone_cache_batch_size "${FROZEN_BACKBONE_CACHE_BATCH_SIZE}"
  --resume_skill_table_mode "${RESUME_SKILL_TABLE_MODE}"
)

if [ -n "${STATE_QUERY_PROMPT_VERSION}" ]; then
  ARGS+=(--state_query_prompt_version "${STATE_QUERY_PROMPT_VERSION}")
fi

if [ -n "${STATE_QUERY_MAX_CHARS}" ]; then
  ARGS+=(--state_query_max_chars "${STATE_QUERY_MAX_CHARS}")
fi

if [ -n "${STATE_QUERY_TRUNCATION}" ]; then
  ARGS+=(--state_query_truncation "${STATE_QUERY_TRUNCATION}")
fi

if [ "${EXPAND_ALIAS_POSITIVES}" = "0" ]; then
  ARGS+=(--no-expand_alias_positives)
fi

if [ -n "${MAX_SKILLS}" ]; then
  ARGS+=(--max_skills "${MAX_SKILLS}")
fi

if [ -n "${MAX_QUERIES}" ]; then
  ARGS+=(--max_queries "${MAX_QUERIES}")
fi

if [ "${TRAIN_SKILL_EMBEDDINGS}" = "1" ]; then
  ARGS+=(--train_skill_embeddings)
fi

if [ "${TRAIN_SKILL_BIAS}" = "1" ]; then
  ARGS+=(--train_skill_bias)
fi

if [ "${TRAIN_ENCODER_BACKBONE}" = "1" ]; then
  ARGS+=(--train_encoder_backbone)
fi

if [ -n "${ENCODER_BACKBONE_LEARNING_RATE}" ]; then
  ARGS+=(--encoder_backbone_learning_rate "${ENCODER_BACKBONE_LEARNING_RATE}")
fi

if [ "${TRAIN_ENCODER_PROJECTION}" = "1" ]; then
  ARGS+=(--train_encoder_projection)
fi

if [ "${TRAIN_SKILL_ADAPTER}" = "0" ]; then
  ARGS+=(--freeze_skill_adapter)
fi

if [ "${TRAIN_RETRIEVAL_SCALE}" = "0" ]; then
  ARGS+=(--freeze_retrieval_scale)
fi

if [ -n "${INIT_CHECKPOINT_PATH}" ]; then
  ARGS+=(--init_checkpoint_path "${INIT_CHECKPOINT_PATH}")
fi

if [ -n "${INIT_CHECKPOINT_SKILLS_PATH}" ]; then
  ARGS+=(--init_checkpoint_skills_path "${INIT_CHECKPOINT_SKILLS_PATH}")
fi

if [ -n "${RESUME_CHECKPOINT_PATH}" ]; then
  ARGS+=(--resume_checkpoint_path "${RESUME_CHECKPOINT_PATH}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/train_stdout.json"
