#!/bin/bash
#SBATCH --job-name=sr_global_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
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

BENCHMARK=${BENCHMARK:-tau2}
RUN_TAG=${RUN_TAG:-global_pool_${BENCHMARK}_skillrouter_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${BENCHMARK}_global_pool_skillrouter_eval/${RUN_TAG}}

BASE_SKILLS_PATH=${BASE_SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-ALL}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-16}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}

mkdir -p "${OUTPUT_DIR}"
args=(
  "${PYTHON_BIN}" scripts/run_global_pool_skillrouter_eval.py
  --benchmark "${BENCHMARK}"
  --base_skills_path "${BASE_SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --encode_batch_size "${ENCODE_BATCH_SIZE}"
  --score_batch_size "${SCORE_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
)

if [[ -n "${ADAPTER_CHECKPOINT_PATH}" ]]; then
  args+=(--adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}")
fi
if [[ -n "${PREBUILT_SOURCE_ROWS_PATH:-}" ]]; then
  args+=(--prebuilt_source_rows_path "${PREBUILT_SOURCE_ROWS_PATH}")
fi
if [[ -n "${PREBUILT_SKILLS_PATH:-}" ]]; then
  args+=(--prebuilt_skills_path "${PREBUILT_SKILLS_PATH}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ "${BENCHMARK}" == "tau2" ]]; then
  DOMAINS=${DOMAINS:-airline,retail,telecom}
  args+=(--domains "${DOMAINS}")
  if [[ -n "${MAX_TASKS_PER_DOMAIN:-}" && "${MAX_TASKS_PER_DOMAIN}" != "ALL" ]]; then
    args+=(--max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}")
  fi
elif [[ "${BENCHMARK}" == "toolsandbox" ]]; then
  if [[ -n "${MAX_SCENARIOS:-}" && "${MAX_SCENARIOS}" != "ALL" ]]; then
    args+=(--max_scenarios "${MAX_SCENARIOS}")
  fi
elif [[ "${BENCHMARK}" == "bfcl" ]]; then
  if [[ -n "${CATEGORIES:-}" ]]; then
    args+=(--categories "${CATEGORIES}")
  fi
  if [[ -n "${MAX_ROWS_PER_CATEGORY:-}" && "${MAX_ROWS_PER_CATEGORY}" != "ALL" ]]; then
    args+=(--max_rows_per_category "${MAX_ROWS_PER_CATEGORY}")
  fi
elif [[ "${BENCHMARK}" == "apibank" ]]; then
  if [[ -n "${FILES:-}" ]]; then
    args+=(--files "${FILES}")
  fi
  if [[ -n "${MAX_ROWS_PER_FILE:-}" && "${MAX_ROWS_PER_FILE}" != "ALL" ]]; then
    args+=(--max_rows_per_file "${MAX_ROWS_PER_FILE}")
  fi
elif [[ "${BENCHMARK}" == "toolbench_g3" ]]; then
  TOOLBENCH_EVAL_TRAJECTORIES=${TOOLBENCH_EVAL_TRAJECTORIES:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
  TOOLBENCH_SKILLS_PATH=${TOOLBENCH_SKILLS_PATH:-data/toolbench_g3/skills.jsonl}
  args+=(--toolbench_eval_trajectories_path "${TOOLBENCH_EVAL_TRAJECTORIES}")
  args+=(--toolbench_skills_path "${TOOLBENCH_SKILLS_PATH}")
fi

if [[ "${INCLUDE_TRIVIAL:-0}" == "1" || "${INCLUDE_TRIVIAL:-0}" == "true" ]]; then
  args+=(--include_trivial)
fi

printf '[global-skillrouter-eval] benchmark=%s output_dir=%s\n' "${BENCHMARK}" "${OUTPUT_DIR}"
printf '[global-skillrouter-eval] adapter_checkpoint=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
