#!/bin/bash
#SBATCH --job-name=bge_route
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

BENCHMARK=${BENCHMARK:-toolbench_g3}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
EMBEDDING_MODEL_NAME_OR_PATH=${EMBEDDING_MODEL_NAME_OR_PATH:-models/BAAI/bge-m3}
RERANKER_MODEL_NAME_OR_PATH=${RERANKER_MODEL_NAME_OR_PATH:-none}
ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-}
MODE=${MODE:-retrieval_only}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_route_eval/${BENCHMARK}_${MODE}_${TIMESTAMP}}

TOP_K=${TOP_K:-100}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-32}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-64}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-512}
RERANKER_BATCH_SIZE=${RERANKER_BATCH_SIZE:-16}
RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-512}
MAX_SKILL_CHARS=${MAX_SKILL_CHARS:-900}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SCORE_MODE=${SCORE_MODE:-logit_diff}
QUERY_TEXT_MODE=${QUERY_TEXT_MODE:-auto}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-256}
ROW_PROGRESS_INTERVAL=${ROW_PROGRESS_INTERVAL:-25}
EMBEDDING_CACHE_DIR=${EMBEDDING_CACHE_DIR:-outputs/bge_route_eval/cache}

TOOLBENCH_EVAL_TRAJECTORIES_PATH=${TOOLBENCH_EVAL_TRAJECTORIES_PATH:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl}
TOOLBENCH_SKILLS_PATH=${TOOLBENCH_SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl}
TAU_DATA_ROOT=${TAU_DATA_ROOT:-.tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data}
TAU_DOMAINS=${TAU_DOMAINS:-airline,retail,telecom}
TAU_MAX_TASKS_PER_DOMAIN=${TAU_MAX_TASKS_PER_DOMAIN:-}
TOOLSANDBOX_SCENARIOS_ROOT=${TOOLSANDBOX_SCENARIOS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios}
TOOLSANDBOX_TOOLS_ROOT=${TOOLSANDBOX_TOOLS_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools}
TOOLSANDBOX_MAX_SCENARIOS=${TOOLSANDBOX_MAX_SCENARIOS:-}

mkdir -p "${OUTPUT_DIR}"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

args=(
  "${PYTHON_BIN}" scripts/run_qwen3_embedding_reranker_route_eval.py
  --benchmark "${BENCHMARK}"
  --output_dir "${OUTPUT_DIR}"
  --embedding_model_name_or_path "${EMBEDDING_MODEL_NAME_OR_PATH}"
  --reranker_model_name_or_path "${RERANKER_MODEL_NAME_OR_PATH}"
  --top_k "${TOP_K}"
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}"
  --embedding_max_length "${EMBEDDING_MAX_LENGTH}"
  --reranker_batch_size "${RERANKER_BATCH_SIZE}"
  --reranker_max_length "${RERANKER_MAX_LENGTH}"
  --max_skill_chars "${MAX_SKILL_CHARS}"
  --torch_dtype "${TORCH_DTYPE}"
  --score_mode "${SCORE_MODE}"
  --query_text_mode "${QUERY_TEXT_MODE}"
  --score_batch_size "${SCORE_BATCH_SIZE}"
  --row_progress_interval "${ROW_PROGRESS_INTERVAL}"
  --embedding_cache_dir "${EMBEDDING_CACHE_DIR}"
  --toolbench_eval_trajectories_path "${TOOLBENCH_EVAL_TRAJECTORIES_PATH}"
  --toolbench_skills_path "${TOOLBENCH_SKILLS_PATH}"
  --tau_data_root "${TAU_DATA_ROOT}"
  --tau_domains "${TAU_DOMAINS}"
  --toolsandbox_scenarios_root "${TOOLSANDBOX_SCENARIOS_ROOT}"
  --toolsandbox_tools_root "${TOOLSANDBOX_TOOLS_ROOT}"
)

if [[ -n "${ADAPTER_CHECKPOINT_PATH}" ]]; then
  args+=(--adapter_checkpoint_path "${ADAPTER_CHECKPOINT_PATH}")
fi
if [[ -n "${MAX_EVAL_ROWS}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ -n "${TAU_MAX_TASKS_PER_DOMAIN}" ]]; then
  args+=(--tau_max_tasks_per_domain "${TAU_MAX_TASKS_PER_DOMAIN}")
fi
if [[ -n "${TOOLSANDBOX_MAX_SCENARIOS}" ]]; then
  args+=(--toolsandbox_max_scenarios "${TOOLSANDBOX_MAX_SCENARIOS}")
fi

printf '[bge-route] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[bge-route] benchmark=%s mode=%s top_k=%s max_eval_rows=%s\n' "${BENCHMARK}" "${MODE}" "${TOP_K}" "${MAX_EVAL_ROWS:-ALL}"
printf '[bge-route] embedding=%s reranker=%s\n' "${EMBEDDING_MODEL_NAME_OR_PATH}" "${RERANKER_MODEL_NAME_OR_PATH}"
printf '[bge-route] adapter=%s\n' "${ADAPTER_CHECKPOINT_PATH:-none}"
printf '[bge-route] query_text_mode=%s\n' "${QUERY_TEXT_MODE}"
printf '[bge-route] batch embedding=%s score=%s reranker=%s dtype=%s\n' "${EMBEDDING_BATCH_SIZE}" "${SCORE_BATCH_SIZE}" "${RERANKER_BATCH_SIZE}" "${TORCH_DTYPE}"

"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
