#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

echo "DEPRECATED: Stage1 retrieval warmup has been replaced by Stage0 SkillRouter-compatible bi-encoder training." >&2
echo "Delegating to scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh." >&2

export OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}

exec bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
