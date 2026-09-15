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
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act/checkpoints/clstr_stage4_act-step2000.pt}
STAGE4_OUTPUT_DIR=${STAGE4_OUTPUT_DIR:-outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act}
MIN_STAGE4_STEPS=${MIN_STAGE4_STEPS:-2000}
TOOLRET_EVAL_DIR=${TOOLRET_EVAL_DIR:-data/toolret_eval}
TOOLRET_RUN_PATH=${TOOLRET_RUN_PATH:-outputs/toolret_eval/clstr_retrieval/run.tsv}
TRAJECT_EVAL_DIR=${TRAJECT_EVAL_DIR:-data/traject_eval_traject_split_test}
TRAJECT_SEQUENCE_PROXY_PATH=${TRAJECT_SEQUENCE_PROXY_PATH:-outputs/traject_eval_traject_split_test/traject_sequence_proxy_metrics.json}
TRAJECT_OFFICIAL_METRICS_PATH=${TRAJECT_OFFICIAL_METRICS_PATH:-outputs/traject_eval_traject_split_test/official_metrics.json}
TOOLBENCH_G3_DATA_DIR=${TOOLBENCH_G3_DATA_DIR:-data/toolbench_g3}
TOOLBENCH_G3_SOURCE_ROOT=${TOOLBENCH_G3_SOURCE_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data}
TOOLBENCH_G3_ROUTING_REPORT=${TOOLBENCH_G3_ROUTING_REPORT:-outputs/toolbench_g3/clstr_routing_eval/metrics.json}
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
CONVERTED_ANSWER_PATH=${CONVERTED_ANSWER_PATH:-outputs/toolbench_g3/stabletoolbench_converted}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}
TEST_SET=${TEST_SET:-G3_instruction}
API_POOL_FILE=${API_POOL_FILE:-${STABLETOOLBENCH_ROOT}/openai_key.json}
APPWORLD_COMBINATION_REPORT=${APPWORLD_COMBINATION_REPORT:-outputs/appworld_combination_plot/report.json}
EXPECTED_TOOLBENCH_G3_ANSWER_FILES=${EXPECTED_TOOLBENCH_G3_ANSWER_FILES:-}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_eval_matrix_readiness/eval_matrix_readiness.json}

ARGS=(
  scripts/audit_clstr_eval_matrix_readiness.py
  --stage4_checkpoint "${STAGE4_CHECKPOINT}"
  --stage4_output_dir "${STAGE4_OUTPUT_DIR}"
  --min_stage4_steps "${MIN_STAGE4_STEPS}"
  --toolret_eval_dir "${TOOLRET_EVAL_DIR}"
  --toolret_run_path "${TOOLRET_RUN_PATH}"
  --traject_eval_dir "${TRAJECT_EVAL_DIR}"
  --traject_sequence_proxy_path "${TRAJECT_SEQUENCE_PROXY_PATH}"
  --traject_official_metrics_path "${TRAJECT_OFFICIAL_METRICS_PATH}"
  --toolbench_g3_data_dir "${TOOLBENCH_G3_DATA_DIR}"
  --toolbench_g3_source_root "${TOOLBENCH_G3_SOURCE_ROOT}"
  --toolbench_g3_routing_report "${TOOLBENCH_G3_ROUTING_REPORT}"
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}"
  --converted_answer_path "${CONVERTED_ANSWER_PATH}"
  --candidate_model "${CANDIDATE_MODEL}"
  --test_set "${TEST_SET}"
  --api_pool_file "${API_POOL_FILE}"
  --appworld_combination_report "${APPWORLD_COMBINATION_REPORT}"
  --output_path "${OUTPUT_PATH}"
  --fail_on_action_required
)

if [[ -n "${EXPECTED_TOOLBENCH_G3_ANSWER_FILES}" ]]; then
  ARGS+=(--expected_toolbench_g3_answer_files "${EXPECTED_TOOLBENCH_G3_ANSWER_FILES}")
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_PATH}.stdout.json"
