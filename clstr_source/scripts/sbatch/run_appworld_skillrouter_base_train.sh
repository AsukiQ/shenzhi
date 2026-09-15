#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_skillrouter_base_train_v1}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/train_tasks.jsonl}
QRELS_PATH=${QRELS_PATH:-data/appworld_routing/train_qrels.jsonl}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
TOP_K=${TOP_K:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-1024}
EPOCHS=${EPOCHS:-120}
LEARNING_RATE=${LEARNING_RATE:-0.05}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
SEED=${SEED:-13}
BUILD_ROUTING_DATA=${BUILD_ROUTING_DATA:-1}

if [ "${BUILD_ROUTING_DATA}" = "1" ]; then
  "${PYTHON_BIN}" scripts/build_appworld_routing_data.py \
    --appworld_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root \
    --skill_pool_path "${SKILL_POOL_PATH}" \
    --output_dir data/appworld_routing \
    --splits train dev test_normal test_challenge \
    --positives_per_task 5
fi

"${PYTHON_BIN}" scripts/run_appworld_skillrouter_base_train.py \
  --tasks_path "${TASKS_PATH}" \
  --qrels_path "${QRELS_PATH}" \
  --skill_pool_path "${SKILL_POOL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --seed "${SEED}"
