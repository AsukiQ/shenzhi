#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
TOOLBENCH_DATA_DIR=${TOOLBENCH_DATA_DIR:-data/toolbench_g3}
SOURCE_ROOT=${SOURCE_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data}
EXPECTED_G3_ANSWER_FILES=${EXPECTED_G3_ANSWER_FILES:-5000}
ROUTING_EVAL_DIR=${ROUTING_EVAL_DIR:-data/toolbench_g3_routing_eval}
QUERIES_PATH=${QUERIES_PATH:-data/toolbench_g3_routing_eval/queries.jsonl}
QRELS_PATH=${QRELS_PATH:-data/toolbench_g3_routing_eval/qrels.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/toolbench_g3/skills.jsonl}
RETRIEVAL_PATH=${RETRIEVAL_PATH:-data/toolbench_g3/retrieval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3/clstr_routing_eval}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-models/Qwen3-8B}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TOP_K=${TOP_K:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
MODEL_DIM=${MODEL_DIM:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKILL_TEXT_FORMAT=${SKILL_TEXT_FORMAT:-clstr}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}
RUN_NAME=${RUN_NAME:-clstr_toolbench_g3_routing}
RUN_EVAL=${RUN_EVAL:-1}
BENCHMARK=${BENCHMARK:-toolbench_g3_routing}
RUN_PREFLIGHT_AUDIT=${RUN_PREFLIGHT_AUDIT:-1}

mkdir -p "${OUTPUT_DIR}" "${ROUTING_EVAL_DIR}"

if [[ "${RUN_PREFLIGHT_AUDIT}" = "1" ]]; then
  "${PYTHON_BIN}" scripts/audit_toolbench_g3_data.py \
    --data_dir "${TOOLBENCH_DATA_DIR}" \
    --source_root "${SOURCE_ROOT}" \
    --output_path "${OUTPUT_DIR}/toolbench_g3_audit.json" \
    --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}" \
    --fail_on_action_required | tee "${OUTPUT_DIR}/toolbench_g3_audit_stdout.json"
fi

"${PYTHON_BIN}" scripts/prepare_toolbench_g3_routing_eval.py \
  --retrieval_path "${RETRIEVAL_PATH}" \
  --output_dir "${ROUTING_EVAL_DIR}" | tee "${OUTPUT_DIR}/prepare_stdout.json"

"${PYTHON_BIN}" scripts/export_clstr_retrieval_run.py \
  --queries_path "${QUERIES_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model_name "${BASE_MODEL_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --model_dim "${MODEL_DIM}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --freeze_backbone \
  --normalize_embeddings \
  --skill_text_format "${SKILL_TEXT_FORMAT}" \
  --skill_table_batch_size "${SKILL_TABLE_BATCH_SIZE}" \
  --disable_cross_encoder \
  --local_files_only \
  --run_name "${RUN_NAME}" | tee "${OUTPUT_DIR}/export_stdout.json"

if [[ "${RUN_EVAL}" = "1" ]]; then
  "${PYTHON_BIN}" scripts/evaluate_retrieval_run.py \
    --qrels_path "${QRELS_PATH}" \
    --run_path "${OUTPUT_DIR}/run.tsv" \
    --output_dir "${OUTPUT_DIR}" \
    --run_format trec \
    --k_values 5 10 \
    --benchmark "${BENCHMARK}" \
    --method "${RUN_NAME}" | tee "${OUTPUT_DIR}/eval_stdout.json"
fi
