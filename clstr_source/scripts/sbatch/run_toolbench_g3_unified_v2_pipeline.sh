#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python-only data pipeline"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data}
EXPECTED_G3_ANSWER_FILES=${EXPECTED_G3_ANSWER_FILES:-5000}
TOOLBENCH_OUTPUT_DIR=${TOOLBENCH_OUTPUT_DIR:-data/toolbench_g3}
UNIFIED_OUTPUT_DIR=${UNIFIED_OUTPUT_DIR:-data/clstr_unified_pretrain_v4_2_toolbench_g3_refresh}
SCHEMA_VERSION=${SCHEMA_VERSION:-v4_2_toolbench_g3_refresh}
RAW_VERIFY_OUTPUT=${RAW_VERIFY_OUTPUT:-outputs/toolbench_g3/toolbench_g3_raw_verify.json}
TOOLBENCH_AUDIT_OUTPUT=${TOOLBENCH_AUDIT_OUTPUT:-outputs/toolbench_g3/toolbench_g3_audit.json}
READINESS_OUTPUT=${READINESS_OUTPUT:-outputs/clstr_unified_readiness_audit/v4_2_toolbench_g3_refresh_readiness_report.json}
TOOLRET_EVAL_DIR=${TOOLRET_EVAL_DIR:-data/toolret_eval}
TRAJECT_EVAL_DIR=${TRAJECT_EVAL_DIR:-data/traject_eval_traject_split_test}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}
SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH=${SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH:-outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json}
STAGE0_BASELINE_METRICS_PATH=${STAGE0_BASELINE_METRICS_PATH:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025}

RAW_VERIFY_EXTRACT_ROOT="${SOURCE_ROOT}"
if [[ -f "${SOURCE_ROOT}/instruction/G3_query.json" ]]; then
  RAW_VERIFY_EXTRACT_ROOT="${SOURCE_ROOT}/.."
fi

mkdir -p "$(dirname "${RAW_VERIFY_OUTPUT}")" "$(dirname "${TOOLBENCH_AUDIT_OUTPUT}")" "$(dirname "${READINESS_OUTPUT}")"

"${PYTHON_BIN}" scripts/download_toolbench_data.py \
  --verify_only \
  --extract_root "${RAW_VERIFY_EXTRACT_ROOT}" \
  --manifest_path "${RAW_VERIFY_OUTPUT}" \
  --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}"

"${PYTHON_BIN}" scripts/import_toolbench_g3.py \
  --source_root "${SOURCE_ROOT}" \
  --output_dir "${TOOLBENCH_OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_toolbench_g3_data.py \
  --data_dir "${TOOLBENCH_OUTPUT_DIR}" \
  --source_root "${SOURCE_ROOT}" \
  --output_path "${TOOLBENCH_AUDIT_OUTPUT}" \
  --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}" \
  --fail_on_action_required

"${PYTHON_BIN}" scripts/build_clstr_unified_pretrain.py \
  --repo_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr \
  --output_dir "${UNIFIED_OUTPUT_DIR}" \
  --schema_version "${SCHEMA_VERSION}"

"${PYTHON_BIN}" scripts/audit_clstr_unified_training_readiness.py \
  --data_root "${UNIFIED_OUTPUT_DIR}" \
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
  --toolbench_g3_data_dir "${TOOLBENCH_OUTPUT_DIR}" \
  --toolbench_g3_source_root "${SOURCE_ROOT}" \
  --expected_toolbench_g3_answer_files "${EXPECTED_G3_ANSWER_FILES}" \
  --output_path "${READINESS_OUTPUT}" | tee "$(dirname "${READINESS_OUTPUT}")/toolbench_g3_readiness_stdout.json"
