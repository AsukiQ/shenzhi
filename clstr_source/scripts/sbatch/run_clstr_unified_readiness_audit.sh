#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
TOOLRET_EVAL_DIR=${TOOLRET_EVAL_DIR:-data/toolret_eval}
TRAJECT_EVAL_DIR=${TRAJECT_EVAL_DIR:-data/traject_eval_traject_split_test}
TOOLBENCH_G3_DATA_DIR=${TOOLBENCH_G3_DATA_DIR:-data/toolbench_g3}
TOOLBENCH_G3_SOURCE_ROOT=${TOOLBENCH_G3_SOURCE_ROOT:-../ToolBench/data}
EXPECTED_TOOLBENCH_G3_ANSWER_FILES=${EXPECTED_TOOLBENCH_G3_ANSWER_FILES:-5000}
SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH=${SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json}
STAGE0_BASELINE_METRICS_PATH=${STAGE0_BASELINE_METRICS_PATH:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
MEMORY_UTILITY_GATE_MODE=${MEMORY_UTILITY_GATE_MODE:-dynamic}
MEMORY_UTILITY_AUDIT_REPORT_PATH=${MEMORY_UTILITY_AUDIT_REPORT_PATH:-}
MEMORY_UTILITY_GATE_CHECKPOINT_PATH=${MEMORY_UTILITY_GATE_CHECKPOINT_PATH:-}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_unified_readiness_audit/readiness_report_v4_2_progressive_final_prior_residual_l025.json}

mkdir -p "$(dirname "${OUTPUT_PATH}")"

extra_args=()
if [[ -n "${MEMORY_UTILITY_AUDIT_REPORT_PATH}" ]]; then
  extra_args+=(--memory_utility_audit_report_path "${MEMORY_UTILITY_AUDIT_REPORT_PATH}")
fi
if [[ -n "${MEMORY_UTILITY_GATE_CHECKPOINT_PATH}" ]]; then
  extra_args+=(--memory_utility_gate_checkpoint_path "${MEMORY_UTILITY_GATE_CHECKPOINT_PATH}")
fi

"${PYTHON_BIN}" scripts/audit_clstr_unified_training_readiness.py \
  --data_root "${DATA_ROOT}" \
  --allowed_benchmarks "${ALLOWED_BENCHMARKS}" \
  --skill_dedup_borderline_review_report_path "${SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH}" \
  --stage0_checkpoint "${STAGE0_CHECKPOINT}" \
  --stage0_output_dir "${STAGE0_OUTPUT_DIR}" \
  --stage0_eval_metrics_path "${STAGE0_EVAL_METRICS_PATH}" \
  --stage0_baseline_metrics_path "${STAGE0_BASELINE_METRICS_PATH}" \
  --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
  --stage1_output_dir "${STAGE1_OUTPUT_DIR}" \
  --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
  --stage2_output_dir "${STAGE2_OUTPUT_DIR}" \
  --toolret_eval_dir "${TOOLRET_EVAL_DIR}" \
  --traject_eval_dir "${TRAJECT_EVAL_DIR}" \
  --toolbench_g3_data_dir "${TOOLBENCH_G3_DATA_DIR}" \
  --toolbench_g3_source_root "${TOOLBENCH_G3_SOURCE_ROOT}" \
  --expected_toolbench_g3_answer_files "${EXPECTED_TOOLBENCH_G3_ANSWER_FILES}" \
  --memory_utility_gate_mode "${MEMORY_UTILITY_GATE_MODE}" \
  "${extra_args[@]}" \
  --output_path "${OUTPUT_PATH}" | tee "$(dirname "${OUTPUT_PATH}")/readiness_stdout.json"
