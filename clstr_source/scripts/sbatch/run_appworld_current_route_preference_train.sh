#!/bin/bash
#SBATCH --job-name=appworld_pref_train
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

PREFERENCE_SAMPLES_PATH=${PREFERENCE_SAMPLES_PATH:-outputs/appworld_official_executor_trace_smoke/current_route_execok_state_text_50e1ac9_1_a800/current_route_preference_samples.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_current_route_preference_train_smoke/one_task_step1}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt}
MAX_STEPS=${MAX_STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
MAX_SAMPLES=${MAX_SAMPLES:-2}
TRAIN_TRANSITION=${TRAIN_TRANSITION:-true}
LOSS_SCORE_MODE=${LOSS_SCORE_MODE:-policy_transition_blend}
ENABLE_SUPPRESS_LOSS=${ENABLE_SUPPRESS_LOSS:-false}
ENABLE_PAIRWISE_CORRECTION_LOSS=${ENABLE_PAIRWISE_CORRECTION_LOSS:-false}
PAIRWISE_CORRECTION_MARGIN=${PAIRWISE_CORRECTION_MARGIN:-0.5}
PAIRWISE_CORRECTION_WEIGHT=${PAIRWISE_CORRECTION_WEIGHT:-1.0}
POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.5}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
LEARNED_COMPONENT_MIN_RANGE=${LEARNED_COMPONENT_MIN_RANGE:-0.1}
LEARNED_COMPONENT_TRUST_TOP_K=${LEARNED_COMPONENT_TRUST_TOP_K:-80}

args=(
  --preference_samples_path "${PREFERENCE_SAMPLES_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --base_skill_pool_path "${BASE_SKILL_POOL_PATH}"
  --clstr_checkpoint_path "${CLSTR_CHECKPOINT_PATH}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --loss_score_mode "${LOSS_SCORE_MODE}"
  --pairwise_correction_margin "${PAIRWISE_CORRECTION_MARGIN}"
  --pairwise_correction_weight "${PAIRWISE_CORRECTION_WEIGHT}"
  --policy_blend_alpha "${POLICY_BLEND_ALPHA}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --learned_component_min_range "${LEARNED_COMPONENT_MIN_RANGE}"
  --learned_component_trust_top_k "${LEARNED_COMPONENT_TRUST_TOP_K}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max_samples "${MAX_SAMPLES}")
fi
case "${TRAIN_TRANSITION}" in
  1|true|TRUE|yes|YES)
    args+=(--train_transition)
    ;;
esac
case "${ENABLE_SUPPRESS_LOSS}" in
  1|true|TRUE|yes|YES)
    args+=(--enable_suppress_loss)
    ;;
esac
case "${ENABLE_PAIRWISE_CORRECTION_LOSS}" in
  1|true|TRUE|yes|YES)
    args+=(--enable_pairwise_correction_loss)
    ;;
esac

"${PYTHON_BIN}" scripts/run_appworld_current_route_preference_train.py "${args[@]}"
