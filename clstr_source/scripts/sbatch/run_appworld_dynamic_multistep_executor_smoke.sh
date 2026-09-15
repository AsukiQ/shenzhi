#!/bin/bash
#SBATCH --time=00:45:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export METHOD=${METHOD:-clstr_multistep}
export TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
export SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}
export BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_dynamic_multistep_executor_smoke/dev3}
export CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
export RANKING_MODE=${RANKING_MODE:-skill_table}
export CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-350}
export CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing}
export TOP_K=${TOP_K:-5}
export MAX_TASKS=${MAX_TASKS:-3}
export MAX_STEPS=${MAX_STEPS:-3}
export SKILL_CONTEXT_MODE=${SKILL_CONTEXT_MODE:-safe_metadata}
export APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
export TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
export MAX_INTERACTIONS=${MAX_INTERACTIONS:-10}
export USE_STOP_HEAD=${USE_STOP_HEAD:-0}

exec bash scripts/sbatch/run_appworld_multistep_executor_eval.sh
