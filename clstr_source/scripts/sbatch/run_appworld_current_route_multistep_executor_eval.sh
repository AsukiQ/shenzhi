#!/bin/bash
#SBATCH --time=02:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export METHOD=${METHOD:-clstr_multistep}
export TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
export SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_current_route_v1/skill_pool.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_current_route_multistep_executor_dev10}
export CLSTR_MODEL_CONFIG=${CLSTR_MODEL_CONFIG:-configs/model/appworld_skillrouter_init.yaml}
export CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage4_act/checkpoints/clstr_stage4_act-step1000.pt}
export RANKING_MODE=${RANKING_MODE:-policy_blend}
export CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-100}
export CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing_belief_union_after_update}
export POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.5}
export TOP_K=${TOP_K:-5}
export MAX_TASKS=${MAX_TASKS:-10}
export MAX_STEPS=${MAX_STEPS:-3}
export SKILL_CONTEXT_MODE=${SKILL_CONTEXT_MODE:-safe_metadata}
export APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
export USE_STOP_HEAD=${USE_STOP_HEAD:-0}

exec bash scripts/sbatch/run_appworld_multistep_executor_eval.sh
