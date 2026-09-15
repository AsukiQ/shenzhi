#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model/appworld_skillrouter_init_mt_fusion.yaml}
BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-outputs/appworld_clstr_train_skillrouter_init_routing_full_v1/checkpoints/stage2_base-step500.pt}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/appworld_multistep/oracle_train_trajectories.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_mt_fusion_train/oracle_v1}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-16}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-}
TRAJECTORY_BATCH_SIZE=${TRAJECTORY_BATCH_SIZE:-8}
DETACH_BELIEF_BETWEEN_STEPS=${DETACH_BELIEF_BETWEEN_STEPS:-1}

ARGS=(
  --model_config "${MODEL_CONFIG}"
  --checkpoint_path "${BASE_CHECKPOINT_PATH}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --trajectories_path "${TRAJECTORIES_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --candidate_top_k "${CANDIDATE_TOP_K}"
  --trajectory_batch_size "${TRAJECTORY_BATCH_SIZE}"
)

if [ -n "${MAX_TRAJECTORIES}" ]; then
  ARGS+=(--max_trajectories "${MAX_TRAJECTORIES}")
fi
if [ "${DETACH_BELIEF_BETWEEN_STEPS}" = "1" ]; then
  ARGS+=(--detach_belief_between_steps)
else
  ARGS+=(--no-detach_belief_between_steps)
fi

"${PYTHON_BIN}" scripts/run_appworld_mt_fusion_train.py "${ARGS[@]}"
