#!/bin/bash
#SBATCH --job-name=tb_g3_qwen_ranker
#SBATCH -p gpu_h200
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-Reranker-8B}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3_official_skillrouter_comparison/qwen3_reranker_8b_top100_smoke}

TOP_K=${TOP_K:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-32}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_SKILL_CHARS=${MAX_SKILL_CHARS:-900}
SCORE_MODE=${SCORE_MODE:-logit_diff}
INSTRUCTION=${INSTRUCTION:-}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-50}
ROW_PROGRESS_INTERVAL=${ROW_PROGRESS_INTERVAL:-25}

mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" scripts/run_toolbench_g3_qwen_reranker_eval.py
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}"
  --eval_trajectories_path "${EVAL_TRAJECTORIES}"
  --skills_path "${SKILLS_PATH}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --top_k "${TOP_K}"
  --batch_size "${BATCH_SIZE}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_length "${MAX_LENGTH}"
  --max_skill_chars "${MAX_SKILL_CHARS}"
  --score_mode "${SCORE_MODE}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  --row_progress_interval "${ROW_PROGRESS_INTERVAL}"
)

if [[ -n "${INSTRUCTION}" ]]; then
  args+=(--instruction "${INSTRUCTION}")
fi

if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[qwen3-reranker] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[qwen3-reranker] model=%s top_k=%s max_eval_rows=%s batch_size=%s\n' "${MODEL_NAME_OR_PATH}" "${TOP_K}" "${MAX_EVAL_ROWS:-ALL}" "${BATCH_SIZE}"
printf '[qwen3-reranker] max_length=%s max_skill_chars=%s score_mode=%s\n' "${MAX_LENGTH}" "${MAX_SKILL_CHARS}" "${SCORE_MODE}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
