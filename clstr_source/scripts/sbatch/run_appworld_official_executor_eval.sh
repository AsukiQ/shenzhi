#!/bin/bash
#SBATCH --job-name=appworld_official_exec
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=01:00:00

set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
export APPWORLD_CACHE=${APPWORLD_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld}
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
METHOD=${METHOD:-qwen_only}
TASKS_PATH=${TASKS_PATH:-data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_official_executor_smoke/${METHOD}}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-14B}
MAX_TASKS=${MAX_TASKS:-1}
TASK_IDS=${TASK_IDS:-}
TASK_IDS="${TASK_IDS//:/,}"
MAX_INTERACTIONS=${MAX_INTERACTIONS:-40}
MAX_WRONG_COMPLETION_RETRIES=${MAX_WRONG_COMPLETION_RETRIES:-0}
PREFLIGHT_MAX_REPAIRS=${PREFLIGHT_MAX_REPAIRS:-0}
COMPLETION_PRECHECK_MODE=${COMPLETION_PRECHECK_MODE:-off}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-120}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
DEVICE=${DEVICE:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-true}
ENABLE_THINKING=${ENABLE_THINKING:-false}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-clstr_appworld_official_executor}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-}
BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-}
CLSTR_MODEL_CONFIG=${CLSTR_MODEL_CONFIG:-}
CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-}
RANKING_MODE=${RANKING_MODE:-auto}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-}
CANDIDATE_SOURCE=${CANDIDATE_SOURCE:-routing}
POLICY_BLEND_ALPHA=${POLICY_BLEND_ALPHA:-0.5}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
CLSTR_ALPHA=${CLSTR_ALPHA:-0.25}
RECURRENT_BELIEF=${RECURRENT_BELIEF:-true}
TOP_K=${TOP_K:-5}
MAX_EVIDENCE_CHARS=${MAX_EVIDENCE_CHARS:-1600}
HANDOFF_PROMPT_STYLE=${HANDOFF_PROMPT_STYLE:-structured_evidence}
HANDOFF_VISIBLE_SKILL_LIMIT=${HANDOFF_VISIBLE_SKILL_LIMIT:-}
CLSTR_STATE_CONTEXT=${CLSTR_STATE_CONTEXT:-api_schema}
CURRENT_ROUTE_ROLLOUTS_PATH=${CURRENT_ROUTE_ROLLOUTS_PATH:-}
DEDUPE_CANONICAL_SKILLS=${DEDUPE_CANONICAL_SKILLS:-false}
MAX_AUTH_LIKE_SKILLS=${MAX_AUTH_LIKE_SKILLS:-}
APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-false}

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
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --torch_dtype "${TORCH_DTYPE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --experiment_name "${EXPERIMENT_NAME}"
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
if [[ -n "${SKILL_POOL_PATH}" ]]; then
  args+=(--skill_pool_path "${SKILL_POOL_PATH}")
fi
if [[ -n "${BASE_SKILL_POOL_PATH}" ]]; then
  args+=(--base_skill_pool_path "${BASE_SKILL_POOL_PATH}")
fi
if [[ -n "${PREDICTIONS_PATH}" ]]; then
  args+=(--predictions_path "${PREDICTIONS_PATH}")
fi
if [[ -n "${CLSTR_MODEL_CONFIG}" ]]; then
  args+=(--clstr_model_config "${CLSTR_MODEL_CONFIG}")
fi
if [[ -n "${CLSTR_CHECKPOINT_PATH}" ]]; then
  args+=(--clstr_checkpoint_path "${CLSTR_CHECKPOINT_PATH}")
fi
args+=(
  --ranking_mode "${RANKING_MODE}"
  --candidate_source "${CANDIDATE_SOURCE}"
  --policy_blend_alpha "${POLICY_BLEND_ALPHA}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --clstr_alpha "${CLSTR_ALPHA}"
  --top_k "${TOP_K}"
  --max_evidence_chars "${MAX_EVIDENCE_CHARS}"
  --handoff_prompt_style "${HANDOFF_PROMPT_STYLE}"
  --clstr_state_context "${CLSTR_STATE_CONTEXT}"
)
if [[ -n "${HANDOFF_VISIBLE_SKILL_LIMIT}" ]]; then
  args+=(--handoff_visible_skill_limit "${HANDOFF_VISIBLE_SKILL_LIMIT}")
fi
if [[ -n "${CANDIDATE_TOP_K}" ]]; then
  args+=(--candidate_top_k "${CANDIDATE_TOP_K}")
fi
if [[ -n "${CURRENT_ROUTE_ROLLOUTS_PATH}" ]]; then
  args+=(--current_route_rollouts_path "${CURRENT_ROUTE_ROLLOUTS_PATH}")
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

"${PYTHON_BIN}" scripts/run_appworld_official_executor_eval.py "${args[@]}"
