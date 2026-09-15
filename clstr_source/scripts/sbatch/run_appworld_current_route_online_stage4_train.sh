#!/bin/bash
#SBATCH --job-name=appworld_online_s4
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

export APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
export APPWORLD_CACHE=${APPWORLD_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld}
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

METHOD=${METHOD:-clstr_multistep}
TASKS_PATH=${TASKS_PATH:-data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_current_route_online_stage4_smoke/one_update}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-14B}
MAX_TASKS=${MAX_TASKS:-3}
TASK_IDS=${TASK_IDS:-}
TASK_IDS="${TASK_IDS//:/,}"
MAX_INTERACTIONS=${MAX_INTERACTIONS:-40}
MAX_WRONG_COMPLETION_RETRIES=${MAX_WRONG_COMPLETION_RETRIES:-0}
PREFLIGHT_MAX_REPAIRS=${PREFLIGHT_MAX_REPAIRS:-0}
COMPLETION_PRECHECK_MODE=${COMPLETION_PRECHECK_MODE:-constraint_tokens}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-120}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
DEVICE=${DEVICE:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
ENABLE_THINKING=${ENABLE_THINKING:-false}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-clstr_appworld_current_route_online_stage4}

SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}
CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt}
RANKING_MODE=${RANKING_MODE:-policy_transition_blend}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-350}
CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing}
POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.5}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
LEARNED_COMPONENT_MIN_RANGE=${LEARNED_COMPONENT_MIN_RANGE:-0.1}
LEARNED_COMPONENT_TRUST_TOP_K=${LEARNED_COMPONENT_TRUST_TOP_K:-80}
RECURRENT_BELIEF=${RECURRENT_BELIEF:-true}
DEDUPE_CANONICAL_SKILLS=${DEDUPE_CANONICAL_SKILLS:-true}
MAX_AUTH_LIKE_SKILLS=${MAX_AUTH_LIKE_SKILLS:-1}
APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-true}
TOP_K=${TOP_K:-20}
MAX_EVIDENCE_CHARS=${MAX_EVIDENCE_CHARS:-1600}
HANDOFF_VISIBLE_SKILL_LIMIT=${HANDOFF_VISIBLE_SKILL_LIMIT:-5}
HANDOFF_PROMPT_STYLE=${HANDOFF_PROMPT_STYLE:-legacy_hints}

MAX_ONLINE_UPDATES=${MAX_ONLINE_UPDATES:-1}
ONLINE_UPDATE_EPOCHS=${ONLINE_UPDATE_EPOCHS:-1}
REPLAY_SUCCESS_ROLLOUTS_PATH=${REPLAY_SUCCESS_ROLLOUTS_PATH:-}
MAX_REPLAY_UPDATES=${MAX_REPLAY_UPDATES:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
TRAIN_TRANSITION=${TRAIN_TRANSITION:-true}
STABILITY_KL_WEIGHT=${STABILITY_KL_WEIGHT:-0.1}
LOSS_SCORE_MODE=${LOSS_SCORE_MODE:-policy_transition_blend}
ENABLE_FAILURE_STAGE0_CORRECTION=${ENABLE_FAILURE_STAGE0_CORRECTION:-false}
ENABLE_FAILURE_STAGE4_CORRECTION=${ENABLE_FAILURE_STAGE4_CORRECTION:-false}
ENABLE_ONLINE_STAGE0_CORRECTION_UPDATE=${ENABLE_ONLINE_STAGE0_CORRECTION_UPDATE:-false}
ONLINE_STAGE0_LEARNING_RATE=${ONLINE_STAGE0_LEARNING_RATE:-}
TRAIN_ONLINE_STAGE0_SKILL_ADAPTER=${TRAIN_ONLINE_STAGE0_SKILL_ADAPTER:-true}
TRAIN_ONLINE_STAGE0_RETRIEVAL_SCALE=${TRAIN_ONLINE_STAGE0_RETRIEVAL_SCALE:-true}
TRAIN_ONLINE_STAGE0_SKILL_BIAS=${TRAIN_ONLINE_STAGE0_SKILL_BIAS:-true}
TRAIN_ONLINE_STAGE0_ENCODER_PROJECTION=${TRAIN_ONLINE_STAGE0_ENCODER_PROJECTION:-false}
APPWORLD_TASKS_ROOT=${APPWORLD_TASKS_ROOT:-}
FAILURE_CORRECTION_MAX_TARGETS=${FAILURE_CORRECTION_MAX_TARGETS:-3}

args=(
  --method "${METHOD}"
  --tasks_path "${TASKS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --appworld_root "${APPWORLD_ROOT}"
  --appworld_cache "${APPWORLD_CACHE}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --max_tasks "${MAX_TASKS}"
  --task_ids "${TASK_IDS}"
  --max_interactions "${MAX_INTERACTIONS}"
  --max_wrong_completion_retries "${MAX_WRONG_COMPLETION_RETRIES}"
  --preflight_max_repairs "${PREFLIGHT_MAX_REPAIRS}"
  --completion_precheck_mode "${COMPLETION_PRECHECK_MODE}"
  --timeout_seconds "${TIMEOUT_SECONDS}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --experiment_name "${EXPERIMENT_NAME}"
  --skill_pool_path "${SKILL_POOL_PATH}"
  --base_skill_pool_path "${BASE_SKILL_POOL_PATH}"
  --clstr_checkpoint_path "${CLSTR_CHECKPOINT_PATH}"
  --ranking_mode "${RANKING_MODE}"
  --candidate_top_k "${CANDIDATE_TOP_K}"
  --candidate_source "${CANDIDATE_SOURCE}"
  --policy_blend_alpha "${POLICY_BLEND_ALPHA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --learned_component_min_range "${LEARNED_COMPONENT_MIN_RANGE}"
  --learned_component_trust_top_k "${LEARNED_COMPONENT_TRUST_TOP_K}"
  --top_k "${TOP_K}"
  --max_evidence_chars "${MAX_EVIDENCE_CHARS}"
  --handoff_visible_skill_limit "${HANDOFF_VISIBLE_SKILL_LIMIT}"
  --handoff_prompt_style "${HANDOFF_PROMPT_STYLE}"
  --max_online_updates "${MAX_ONLINE_UPDATES}"
  --online_update_epochs "${ONLINE_UPDATE_EPOCHS}"
  --replay_success_rollouts_path "${REPLAY_SUCCESS_ROLLOUTS_PATH}"
  --max_replay_updates "${MAX_REPLAY_UPDATES}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --stability_kl_weight "${STABILITY_KL_WEIGHT}"
  --loss_score_mode "${LOSS_SCORE_MODE}"
  --failure_correction_max_targets "${FAILURE_CORRECTION_MAX_TARGETS}"
)

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "true" ]]; then
  args+=(--local_files_only)
else
  args+=(--no-local_files_only)
fi
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  args+=(--enable_thinking)
else
  args+=(--no-enable_thinking)
fi
if [[ "${RECURRENT_BELIEF}" == "true" ]]; then
  args+=(--recurrent_belief)
else
  args+=(--no-recurrent_belief)
fi
if [[ "${DEDUPE_CANONICAL_SKILLS}" == "true" ]]; then
  args+=(--dedupe_canonical_skills)
else
  args+=(--no-dedupe_canonical_skills)
fi
if [[ -n "${MAX_AUTH_LIKE_SKILLS}" ]]; then
  args+=(--max_auth_like_skills "${MAX_AUTH_LIKE_SKILLS}")
fi
if [[ "${APPWORLD_EXECUTOR_COMPATIBLE_ONLY}" == "true" ]]; then
  args+=(--appworld_executor_compatible_only)
else
  args+=(--no-appworld_executor_compatible_only)
fi
case "${TRAIN_TRANSITION}" in
  1|true|TRUE|yes|YES)
    args+=(--train_transition)
    ;;
esac
case "${ENABLE_FAILURE_STAGE0_CORRECTION}" in
  1|true|TRUE|yes|YES)
    args+=(--enable_failure_stage0_correction)
    ;;
esac
case "${ENABLE_FAILURE_STAGE4_CORRECTION}" in
  1|true|TRUE|yes|YES)
    args+=(--enable_failure_stage4_correction)
    ;;
esac
case "${ENABLE_ONLINE_STAGE0_CORRECTION_UPDATE}" in
  1|true|TRUE|yes|YES)
    args+=(--enable_online_stage0_correction_update)
    ;;
esac
if [[ -n "${ONLINE_STAGE0_LEARNING_RATE}" ]]; then
  args+=(--online_stage0_learning_rate "${ONLINE_STAGE0_LEARNING_RATE}")
fi
if [[ "${TRAIN_ONLINE_STAGE0_SKILL_ADAPTER}" == "true" ]]; then
  args+=(--train_online_stage0_skill_adapter)
else
  args+=(--no-train_online_stage0_skill_adapter)
fi
if [[ "${TRAIN_ONLINE_STAGE0_RETRIEVAL_SCALE}" == "true" ]]; then
  args+=(--train_online_stage0_retrieval_scale)
else
  args+=(--no-train_online_stage0_retrieval_scale)
fi
if [[ "${TRAIN_ONLINE_STAGE0_SKILL_BIAS}" == "true" ]]; then
  args+=(--train_online_stage0_skill_bias)
else
  args+=(--no-train_online_stage0_skill_bias)
fi
if [[ "${TRAIN_ONLINE_STAGE0_ENCODER_PROJECTION}" == "true" ]]; then
  args+=(--train_online_stage0_encoder_projection)
else
  args+=(--no-train_online_stage0_encoder_projection)
fi
if [[ -n "${APPWORLD_TASKS_ROOT}" ]]; then
  args+=(--appworld_tasks_root "${APPWORLD_TASKS_ROOT}")
fi

"${PYTHON_BIN}" scripts/run_appworld_current_route_online_stage4_train.py "${args[@]}"
