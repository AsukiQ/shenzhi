#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak}
QRELS_PATH=${QRELS_PATH:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/qrels.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TOP_K=${TOP_K:-350}
RUN_NAME=${RUN_NAME:-clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}

export DATA_ROOT QRELS_PATH OUTPUT_DIR CHECKPOINT_PATH TOP_K RUN_NAME

exec bash scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh
