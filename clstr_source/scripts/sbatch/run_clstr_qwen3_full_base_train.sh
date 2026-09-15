#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
TRAIN_PATH=${TRAIN_PATH:-data/clstr_qwen3_8b_full_base_gate/train_alfworld_official.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_full_base_train/skills.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_qwen3_8b_full_base_after_appworld_base_v1}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-models/Qwen3-8B}
CACHE_DIR=${CACHE_DIR:-.cache/huggingface}
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
MODEL_DIM=${MODEL_DIM:-}
INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/run_clstr_qwen3_full_base_train.py
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --qwen_model_path "${QWEN_MODEL_PATH}"
  --cache_dir "${CACHE_DIR}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --max_length "${MAX_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
  --local_files_only
)

if [ -n "${MODEL_DIM}" ]; then
  ARGS+=(--model_dim "${MODEL_DIM}")
fi

if [ "${INCLUDE_AVAILABLE_ACTIONS_IN_STATE}" = "1" ]; then
  ARGS+=(--include_available_actions_in_state)
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/train_stdout.json"
