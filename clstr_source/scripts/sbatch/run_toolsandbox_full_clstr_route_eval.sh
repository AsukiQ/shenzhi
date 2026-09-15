#!/bin/bash
#SBATCH --job-name=toolsandbox_clstr_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

SCENARIOS_ROOT=${SCENARIOS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios}
TOOLS_ROOT=${TOOLS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/checkpoints/clstr_unified_retrieval_v2-step300.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_trainl025_calib_full/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-outputs/clstr_unified_stage4_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_l050_joint_act/checkpoints/clstr_stage4_act-step2000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolsandbox_full_clstr_route_eval/smoke_${SLURM_JOB_ID:-manual}}

MAX_SCENARIOS=${MAX_SCENARIOS:-20}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
STAGE0_CANDIDATE_BATCH_SIZE=${STAGE0_CANDIDATE_BATCH_SIZE:-16}
ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}
ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}
ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}
ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.5}
ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
WRITE_MT_ABLATION_REPORT=${WRITE_MT_ABLATION_REPORT:-0}
MT_ABLATION_AUTO_REPLAY_PREFIX_MAX_STEPS=${MT_ABLATION_AUTO_REPLAY_PREFIX_MAX_STEPS:-3}
INCLUDE_MT_EFFECT_DIAGNOSTICS=${INCLUDE_MT_EFFECT_DIAGNOSTICS:-0}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_toolsandbox_full_clstr_route_eval.py
  --scenarios_root "${SCENARIOS_ROOT}"
  --tools_root "${TOOLS_ROOT}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}"
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE}"
  --online_memory_mode "${ONLINE_MEMORY_MODE}"
  --online_memory_weight "${ONLINE_MEMORY_WEIGHT}"
  --online_memory_next_skill_bonus "${ONLINE_MEMORY_NEXT_SKILL_BONUS}"
  --online_memory_exact_transition_bonus "${ONLINE_MEMORY_EXACT_TRANSITION_BONUS}"
  --transition_residual_lambda "${TRANSITION_RESIDUAL_LAMBDA}"
  --route_scorer "${ROUTE_SCORER}"
)

if [[ -n "${MAX_SCENARIOS}" && "${MAX_SCENARIOS}" != "ALL" ]]; then
  args+=(--max_scenarios "${MAX_SCENARIOS}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ -n "${CANDIDATE_COUNT:-}" && "${CANDIDATE_COUNT}" != "ALL" ]]; then
  args+=(--candidate_count "${CANDIDATE_COUNT}")
fi
if [[ "${WRITE_MT_ABLATION_REPORT}" == "1" || "${WRITE_MT_ABLATION_REPORT}" == "true" || "${WRITE_MT_ABLATION_REPORT}" == "TRUE" ]]; then
  args+=(
    --write_mt_ablation_report
    --mt_ablation_auto_replay_prefix_max_steps "${MT_ABLATION_AUTO_REPLAY_PREFIX_MAX_STEPS}"
  )
  if [[ "${INCLUDE_MT_EFFECT_DIAGNOSTICS}" == "1" || "${INCLUDE_MT_EFFECT_DIAGNOSTICS}" == "true" || "${INCLUDE_MT_EFFECT_DIAGNOSTICS}" == "TRUE" ]]; then
    args+=(--include_mt_effect_diagnostics)
  fi
fi

printf '[toolsandbox-clstr-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[toolsandbox-clstr-eval] route_scorer=%s\n' "${ROUTE_SCORER}"
printf '[toolsandbox-clstr-eval] max_scenarios=%s max_eval_rows=%s\n' "${MAX_SCENARIOS:-ALL}" "${MAX_EVAL_ROWS:-ALL}"
printf '[toolsandbox-clstr-eval] write_mt_ablation_report=%s mt_ablation_auto_replay_prefix_max_steps=%s\n' "${WRITE_MT_ABLATION_REPORT}" "${MT_ABLATION_AUTO_REPLAY_PREFIX_MAX_STEPS}"
printf '[toolsandbox-clstr-eval] include_mt_effect_diagnostics=%s\n' "${INCLUDE_MT_EFFECT_DIAGNOSTICS}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
