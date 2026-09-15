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
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
TEST_SET=${TEST_SET:-G3_instruction}
ORIGINAL_QUERY_FILE=${ORIGINAL_QUERY_FILE:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_instruction/${TEST_SET}.json}
QUERIES_PATH=${QUERIES_PATH:-data/stabletoolbench_g3_solvable_queries.jsonl}
QUERY_EXPORT_REPORT=${QUERY_EXPORT_REPORT:-outputs/toolbench_g3/stabletoolbench_solvable_clstr_retrieval/query_export_report.json}
SKILLS_PATH=${SKILLS_PATH:-data/toolbench_g3/skills.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3/stabletoolbench_solvable_clstr_retrieval}
CLSTR_QUERY_OUTPUT_DIR=${CLSTR_QUERY_OUTPUT_DIR:-outputs/toolbench_g3/stabletoolbench_clstr_topk}
OUTPUT_QUERY_FILE=${OUTPUT_QUERY_FILE:-${CLSTR_QUERY_OUTPUT_DIR}/${TEST_SET}.json}
CLSTR_QUERY_REPORT=${CLSTR_QUERY_REPORT:-${CLSTR_QUERY_OUTPUT_DIR}/${TEST_SET}_build_report.json}
TOOLENV_OUTPUT_ROOT=${TOOLENV_OUTPUT_ROOT:-outputs/toolbench_g3/stabletoolbench_clstr_topk_toolenv/tools}
TOOLENV_MANIFEST_PATH=${TOOLENV_MANIFEST_PATH:-outputs/toolbench_g3/stabletoolbench_clstr_topk_toolenv/toolenv_${TEST_SET}_manifest.json}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-models/Qwen3-8B}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TOP_K=${TOP_K:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
MODEL_DIM=${MODEL_DIM:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKILL_TEXT_FORMAT=${SKILL_TEXT_FORMAT:-clstr}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}
RUN_NAME=${RUN_NAME:-clstr_stabletoolbench_solvable}

mkdir -p \
  "$(dirname "${QUERIES_PATH}")" \
  "${OUTPUT_DIR}" \
  "$(dirname "${QUERY_EXPORT_REPORT}")" \
  "${CLSTR_QUERY_OUTPUT_DIR}" \
  "${TOOLENV_OUTPUT_ROOT}" \
  "$(dirname "${TOOLENV_MANIFEST_PATH}")"

"${PYTHON_BIN}" scripts/build_stabletoolbench_clstr_queries.py \
  --export_retrieval_queries \
  --query_file "${ORIGINAL_QUERY_FILE}" \
  --output_path "${QUERIES_PATH}" \
  --report_path "${QUERY_EXPORT_REPORT}" | tee "${QUERY_EXPORT_REPORT}.stdout.json"

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

"${PYTHON_BIN}" scripts/build_stabletoolbench_clstr_queries.py \
  --original_query_file "${ORIGINAL_QUERY_FILE}" \
  --skills_path "${SKILLS_PATH}" \
  --run_path "${OUTPUT_DIR}/run.tsv" \
  --output_query_file "${OUTPUT_QUERY_FILE}" \
  --report_path "${CLSTR_QUERY_REPORT}" \
  --top_k "${TOP_K}" \
  --fail_on_action_required | tee "${CLSTR_QUERY_REPORT}.stdout.json"

"${PYTHON_BIN}" scripts/build_stabletoolbench_toolenv.py \
  --query_file "${OUTPUT_QUERY_FILE}" \
  --output_root "${TOOLENV_OUTPUT_ROOT}" \
  --manifest_path "${TOOLENV_MANIFEST_PATH}" \
  --fail_on_action_required | tee "${TOOLENV_MANIFEST_PATH}.stdout.json"
