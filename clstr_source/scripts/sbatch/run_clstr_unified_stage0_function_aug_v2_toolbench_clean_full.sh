#!/bin/bash
#SBATCH --time=12:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000}
TOOLBENCH_EVAL_TRAJECTORIES_PATH=${TOOLBENCH_EVAL_TRAJECTORIES_PATH:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
TRAJECT_EVAL_QUERIES_PATH=${TRAJECT_EVAL_QUERIES_PATH:-data/traject_eval_traject_split_test/queries.jsonl}
PREFLIGHT_OUTPUT_PATH=${PREFLIGHT_OUTPUT_PATH:-${OUTPUT_DIR}/clean_training_preflight.json}

TOP_K=${TOP_K:-350}
MAX_STEPS=${MAX_STEPS:-5000}
LEARNING_RATE=${LEARNING_RATE:-1.0e-5}
RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}
TEMPERED_CORRECTION_FRACTION=${TEMPERED_CORRECTION_FRACTION:-0.2}
EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}

# Safe throughput-only acceleration: larger skill-table encode batches reduce
# encoder calls and progress JSONL writes without changing rows, labels, loss,
# topK, optimizer, or sampling.
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-64}

# True Stage0 adaptation from SkillRouter initialization to the clean CLSTR v2
# skill pool. This is not a warm-start from prior CLSTR checkpoints.
INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-}
INIT_CHECKPOINT_SKILLS_PATH=${INIT_CHECKPOINT_SKILLS_PATH:-}
TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}
TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}
TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-1}
TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}
TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}
PRESERVATION_ANCHOR_WEIGHT=${PRESERVATION_ANCHOR_WEIGHT:-0.0}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_clean_training_preflight.py \
  --data_root "${DATA_ROOT}" \
  --toolbench_eval_trajectories_path "${TOOLBENCH_EVAL_TRAJECTORIES_PATH}" \
  --traject_eval_queries_path "${TRAJECT_EVAL_QUERIES_PATH}" \
  --output_path "${PREFLIGHT_OUTPUT_PATH}" \
  --fail_on_leakage | tee "${OUTPUT_DIR}/clean_training_preflight_stdout.json"

export DATA_ROOT OUTPUT_DIR TOP_K MAX_STEPS LEARNING_RATE
export RETRIEVAL_LOSS_MODE SAMPLING_STRATEGY TEMPERED_CORRECTION_FRACTION
export EXPAND_ALIAS_POSITIVES INIT_CHECKPOINT_PATH INIT_CHECKPOINT_SKILLS_PATH
export TRAIN_SKILL_EMBEDDINGS TRAIN_SKILL_BIAS TRAIN_ENCODER_PROJECTION
export TRAIN_SKILL_ADAPTER TRAIN_RETRIEVAL_SCALE PRESERVATION_ANCHOR_WEIGHT
export SKILL_TABLE_BATCH_SIZE

exec bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
