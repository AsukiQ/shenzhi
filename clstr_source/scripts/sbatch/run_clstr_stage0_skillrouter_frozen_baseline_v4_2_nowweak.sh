#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak}
TOP_K=${TOP_K:-350}

export DATA_ROOT OUTPUT_DIR TOP_K

exec bash scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
