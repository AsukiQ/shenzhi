#!/bin/bash
#SBATCH --job-name=qwen06_s2cf
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
export FULL_RUN=0
export MAX_STEPS=1200
export TARGET_TOTAL_STEPS=1200
export MAX_ROWS=97932
export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage2_counterfactual_repair_smoke}
export WARM_START_CHECKPOINT_PATH=${WARM_START_CHECKPOINT_PATH:-${RUN_ROOT}/stage2_anchored_full10000_v1/checkpoints/clstr_full_base-step10000.pt}

export BATCH_SIZE=16
export LEARNING_RATE=3.0e-5
export MINIMUM_LEARNING_RATE=3.0e-6
export CHECKPOINT_INTERVAL_STEPS=300
export EMBEDDING_CACHE_MODE=always
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=1

export POLICY_LOSS_WEIGHT=0.0
export TRANSITION_LOSS_WEIGHT=0.3
export TRANSITION_SKILL_CE_LOSS_WEIGHT=1.0
export BELIEF_LOSS_WEIGHT=0.1
export COUNTERFACTUAL_HISTORY_LOSS_WEIGHT=1.0
export COUNTERFACTUAL_HISTORY_MARGIN=0.1
export ANCHORED_ROUTING_FOUNDATION=0
export STATIC_ROUTE_ANCHOR_WEIGHT=0.0
export AUTO_REPLAY_PREFIX_MAX_STEPS=3
export TRAINABLE_REPLAY_PREFIX=1

printf '[qwen06-stage2-cf] warm_start=%s output=%s steps=%s lr=%s->%s\n' \
  "${WARM_START_CHECKPOINT_PATH}" "${OUTPUT_DIR}" "${MAX_STEPS}" \
  "${LEARNING_RATE}" "${MINIMUM_LEARNING_RATE}"

bash scripts/sbatch/run_qwen06_clstr_stage2_train.sh
