#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
COMMAND=${COMMAND:-duplicate-aware}
SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/appworld_skill_pool/skill_pool.jsonl}
QRELS_PATH=${QRELS_PATH:-data/appworld_routing/dev_qrels.jsonl}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_routing_diagnostics/dev}
METHOD=${METHOD:-appworld_diagnostic}
TRAIN_TASKS_PATH=${TRAIN_TASKS_PATH:-data/appworld_routing/train_tasks.jsonl}
DEV_TASKS_PATH=${DEV_TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
TASKS_PATH=${TASKS_PATH:-data/appworld_routing/dev_tasks.jsonl}
RUN_ARGS=${RUN_ARGS:-}
PREDICTION_ARGS=${PREDICTION_ARGS:-}
FOCUS_METHOD=${FOCUS_METHOD:-clstr_base}
REFERENCE_METHODS=${REFERENCE_METHODS:-skillrouter_base qwen_only}
TOP_K=${TOP_K:-5}

if [ "${COMMAND}" = "duplicate-aware" ]; then
  if [ -z "${PREDICTIONS_PATH}" ]; then
    echo "ERROR: PREDICTIONS_PATH is required for duplicate-aware diagnostics" >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/run_appworld_routing_diagnostics.py duplicate-aware \
    --skill_pool_path "${SKILL_POOL_PATH}" \
    --qrels_path "${QRELS_PATH}" \
    --predictions_path "${PREDICTIONS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --method "${METHOD}"
elif [ "${COMMAND}" = "train-dev-overlap" ]; then
  "${PYTHON_BIN}" scripts/run_appworld_routing_diagnostics.py train-dev-overlap \
    --train_tasks_path "${TRAIN_TASKS_PATH}" \
    --dev_tasks_path "${DEV_TASKS_PATH}" \
    --output_dir "${OUTPUT_DIR}"
elif [ "${COMMAND}" = "executor-failure-topk" ]; then
  if [ -z "${RUN_ARGS}" ]; then
    echo "ERROR: RUN_ARGS is required for executor-failure-topk diagnostics" >&2
    exit 1
  fi
  if [ -z "${PREDICTION_ARGS}" ]; then
    echo "ERROR: PREDICTION_ARGS is required for executor-failure-topk diagnostics" >&2
    exit 1
  fi
  ARGS=(
    executor-failure-topk
    --skill_pool_path "${SKILL_POOL_PATH}"
    --tasks_path "${TASKS_PATH}"
    --qrels_path "${QRELS_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --focus_method "${FOCUS_METHOD}"
    --top_k "${TOP_K}"
  )
  for run_arg in ${RUN_ARGS}; do
    ARGS+=(--run "${run_arg}")
  done
  for prediction_arg in ${PREDICTION_ARGS}; do
    ARGS+=(--predictions "${prediction_arg}")
  done
  for reference_method in ${REFERENCE_METHODS}; do
    ARGS+=(--reference_method "${reference_method}")
  done
  "${PYTHON_BIN}" scripts/run_appworld_routing_diagnostics.py "${ARGS[@]}"
else
  echo "ERROR: unsupported COMMAND=${COMMAND}" >&2
  exit 1
fi
