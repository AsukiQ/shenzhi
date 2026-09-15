#!/bin/bash
#SBATCH --job-name=aw_train_s0_top350
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

QUERIES_PATH=${QUERIES_PATH:-data/appworld_routing/train_tasks.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_train_stage0_top350_v1}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-.cache/hf_models/SkillRouter-Embedding-0.6B}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}

TOP_K=${TOP_K:-350}
BATCH_SIZE=${BATCH_SIZE:-8}
MODEL_DIM=${MODEL_DIM:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/export_clstr_retrieval_run.py \
  --queries_path "${QUERIES_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model_name "${BASE_MODEL_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --model_dim "${MODEL_DIM}" \
  --encoder_pooling last_token \
  --cross_encoder_pooling last_token \
  --tokenizer_padding_side left \
  --torch_dtype "${TORCH_DTYPE}" \
  --freeze_backbone \
  --max_length "${MAX_LENGTH}" \
  --projection_init identity \
  --normalize_embeddings \
  --skill_text_format skillret_official \
  --skill_table_batch_size "${SKILL_TABLE_BATCH_SIZE}" \
  --skill_table_adapter_init identity \
  --disable_cross_encoder \
  --local_files_only \
  --query_text_format skillrouter \
  --run_name appworld_train_stage0_top350 \
  | tee "${OUTPUT_DIR}/export_stdout.json"
