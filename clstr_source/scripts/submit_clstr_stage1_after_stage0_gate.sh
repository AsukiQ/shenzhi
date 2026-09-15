#!/bin/bash
set -euo pipefail

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
PARTITION=${PARTITION:-gpu_a800}
GPUS=${GPUS:-1}
MAX_ACTIVE_JOBS=${MAX_ACTIVE_JOBS:-4}

STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-${STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-${STAGE0_OUTPUT_DIR}/full_retrieval_eval/metrics.json}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
MODEL_DIM=${MODEL_DIM:-1024}

if [[ -s "${STAGE0_CHECKPOINT}" ]]; then
  :
else
  echo "ERROR: Stage1 requires completed Stage0 checkpoint: ${STAGE0_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${STAGE1_OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_clstr_stage0_quality.py \
  --output_dir "${STAGE0_OUTPUT_DIR}" \
  --checkpoint_path "${STAGE0_CHECKPOINT}" \
  --stage0_eval_metrics_path "${STAGE0_EVAL_METRICS_PATH}" \
  --output_path "${STAGE1_OUTPUT_DIR}/stage0_quality_gate.json" \
  --fail_on_action_required

"${PYTHON_BIN}" scripts/run_clstr_stage2_preflight.py \
  --checkpoint_path "${STAGE0_CHECKPOINT}" \
  --skills_path "${SKILLS_PATH}" \
  --model_dim "${MODEL_DIM}" \
  --output_path "${STAGE1_OUTPUT_DIR}/stage1_preflight.json"

bash scripts/guard_clstr_job_budget.sh

sbatch --gpus="${GPUS}" -p "${PARTITION}" \
  --export=ALL,ROUTING_CHECKPOINT_PATH="${STAGE0_CHECKPOINT}",OUTPUT_DIR="${STAGE1_OUTPUT_DIR}",SKILLS_PATH="${SKILLS_PATH}",MODEL_DIM="${MODEL_DIM}" \
  scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh
