#!/bin/bash
#SBATCH --time=04:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_appworld_current_route_v1/trajectories.jsonl}
export SKILLS_PATH=${SKILLS_PATH:-data/clstr_appworld_current_route_v1/skill_pool.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_appworld_current_stage4_act}
export ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step2000.pt}
export HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage2_full_base/checkpoints/clstr_full_base-step5000.pt}
export MAX_STEPS=${MAX_STEPS:-1000}
export BATCH_SIZE=${BATCH_SIZE:-8}
export LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
export SEED=${SEED:-17}
export CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}
export STAGE0_TOP_M=${STAGE0_TOP_M:-100}
export STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
export STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-8}
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-8}
export STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
export TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
export TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-skill_prior_plus_action_observation_residual}
export TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}
export ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-appworld}
export BENCHMARK_CAPS=${BENCHMARK_CAPS:-appworld=-1}

exec bash scripts/sbatch/run_clstr_unified_stage4_act_train.sh
