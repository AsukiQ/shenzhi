#!/bin/bash
#SBATCH --job-name=traj_sr_ft_eval
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-.tmp/stage4_sanitized_trajectbench/trajectories_visible_global_full.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/trajectbench_skillrouter_finetuned_eval/function_aug_v2_adapter_$(date +%Y%m%d_%H%M%S)}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/checkpoints/clstr_unified_retrieval_v2-step300.pt}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-outputs/unified_skillrouter_finetune/function_aug_v2_full_20260628_212251/checkpoints/unified_skillrouter_finetune-step2000.pt}
PREBUILT_STAGE4_ROWS_PATH=${PREBUILT_STAGE4_ROWS_PATH:-}

MAX_ROWS=${MAX_ROWS:-1024}
EVAL_ROWS=${EVAL_ROWS:-256}
EVAL_SPLIT_MODE=${EVAL_SPLIT_MODE:-trajectory_prefix}
TRAJECTORY_EVAL_STEPS=${TRAJECTORY_EVAL_STEPS:-1}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_trajectbench_skillrouter_finetuned_eval.py
  --trajectories_path "${TRAJECTORIES_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}"
  --eval_rows "${EVAL_ROWS}"
  --eval_split_mode "${EVAL_SPLIT_MODE}"
  --trajectory_eval_steps "${TRAJECTORY_EVAL_STEPS}"
  --stage0_top_m "${STAGE0_TOP_M}"
  --stage0_positive_missing_policy "${STAGE0_POSITIVE_MISSING_POLICY}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)

if [[ -n "${MAX_ROWS}" && "${MAX_ROWS}" != "ALL" ]]; then
  args+=(--max_rows "${MAX_ROWS}")
fi
if [[ -n "${CANDIDATE_COUNT}" ]]; then
  args+=(--candidate_count "${CANDIDATE_COUNT}")
fi
if [[ -n "${PREBUILT_STAGE4_ROWS_PATH}" ]]; then
  args+=(--prebuilt_stage4_rows_path "${PREBUILT_STAGE4_ROWS_PATH}")
fi

printf '[trajectbench-skillrouter-ft] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[trajectbench-skillrouter-ft] max_rows=%s eval_rows=%s stage0_top_m=%s\n' "${MAX_ROWS:-ALL}" "${EVAL_ROWS}" "${STAGE0_TOP_M}"
printf '[trajectbench-skillrouter-ft] adapter=%s\n' "${ADAPTER_CHECKPOINT_PATH}"
printf '[trajectbench-skillrouter-ft] prebuilt_stage4_rows=%s\n' "${PREBUILT_STAGE4_ROWS_PATH:-<none>}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
