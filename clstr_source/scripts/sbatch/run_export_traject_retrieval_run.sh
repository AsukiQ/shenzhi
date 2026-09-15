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
TRAJECT_DATA_DIR=${TRAJECT_DATA_DIR:-data/traject_eval_traject_split_test}
QUERIES_PATH=${QUERIES_PATH:-data/traject_eval_traject_split_test/queries.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/traject_eval_traject_split_test/skills.jsonl}
QRELS_PATH=${QRELS_PATH:-data/traject_eval_traject_split_test/qrels.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/traject_eval_traject_split_test/clstr_retrieval}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-models/Qwen3-8B}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TOP_K=${TOP_K:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
MODEL_DIM=${MODEL_DIM:-1024}
MAX_LENGTH=${MAX_LENGTH:-2048}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SKILL_TEXT_FORMAT=${SKILL_TEXT_FORMAT:-clstr}
SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-32}
RUN_NAME=${RUN_NAME:-clstr_unified_retrieval}
RUN_EVAL=${RUN_EVAL:-1}
BENCHMARK=${BENCHMARK:-traject}
RUN_PREFLIGHT_AUDIT=${RUN_PREFLIGHT_AUDIT:-1}
RUN_SEQUENCE_PROXY_EVAL=${RUN_SEQUENCE_PROXY_EVAL:-1}

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_PREFLIGHT_AUDIT}" = "1" ]]; then
  "${PYTHON_BIN}" scripts/audit_traject_eval_data.py \
    --data_dir "${TRAJECT_DATA_DIR}" \
    --output_path "${OUTPUT_DIR}/traject_eval_audit.json" \
    --fail_on_action_required | tee "${OUTPUT_DIR}/traject_eval_audit_stdout.json"
fi

ARGS=(
  scripts/export_clstr_retrieval_run.py
  --queries_path "${QUERIES_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --base_model_name "${BASE_MODEL_NAME}"
  --top_k "${TOP_K}"
  --batch_size "${BATCH_SIZE}"
  --model_dim "${MODEL_DIM}"
  --max_length "${MAX_LENGTH}"
  --torch_dtype "${TORCH_DTYPE}"
  --freeze_backbone
  --normalize_embeddings
  --skill_text_format "${SKILL_TEXT_FORMAT}"
  --skill_table_batch_size "${SKILL_TABLE_BATCH_SIZE}"
  --disable_cross_encoder
  --local_files_only
  --run_name "${RUN_NAME}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/export_stdout.json"

if [[ "${RUN_EVAL}" = "1" ]]; then
  if [[ ! -f "${QRELS_PATH}" ]]; then
    echo "WARN: RUN_EVAL=1 but QRELS_PATH does not exist: ${QRELS_PATH}; skipping evaluation."
  else
    "${PYTHON_BIN}" scripts/evaluate_retrieval_run.py \
      --qrels_path "${QRELS_PATH}" \
      --run_path "${OUTPUT_DIR}/run.tsv" \
      --output_dir "${OUTPUT_DIR}" \
      --run_format trec \
      --k_values 5 10 \
      --benchmark "${BENCHMARK}" \
      --method "${RUN_NAME}" | tee "${OUTPUT_DIR}/eval_stdout.json"
  fi
fi

if [[ "${RUN_SEQUENCE_PROXY_EVAL}" = "1" ]]; then
  if [[ ! -f "${QRELS_PATH}" || ! -f "${QUERIES_PATH}" || ! -f "${OUTPUT_DIR}/run.tsv" ]]; then
    echo "WARN: RUN_SEQUENCE_PROXY_EVAL=1 but required inputs are missing; skipping TRAJECT sequence proxy evaluation."
  else
    "${PYTHON_BIN}" scripts/evaluate_traject_sequence_proxy.py \
      --queries_path "${QUERIES_PATH}" \
      --qrels_path "${QRELS_PATH}" \
      --run_path "${OUTPUT_DIR}/run.tsv" \
      --output_dir "${OUTPUT_DIR}" \
      --run_format trec \
      --method "${RUN_NAME}" | tee "${OUTPUT_DIR}/traject_sequence_proxy_stdout.json"
  fi
fi
