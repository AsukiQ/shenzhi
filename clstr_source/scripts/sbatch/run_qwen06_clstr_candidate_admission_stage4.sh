#!/bin/bash
#SBATCH --job-name=qwen06_admit_s4
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/outputs/qwen06_clstr_postfix}
FULL_RUN=${FULL_RUN:-0}

export PROJECT_ROOT RUN_ROOT FULL_RUN
export STAGE4_METHOD=candidate_admission_residual_v1
export SMOKE_MAX_STEPS=300
export FULL_MAX_STEPS=1200
export SMOKE_VALIDATION_INTERVAL_STEPS=100
export FULL_VALIDATION_INTERVAL_STEPS=300
export SMOKE_MAX_ROWS=${SMOKE_MAX_ROWS:-4096}
export SMOKE_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_VALIDATION_ROWS_PER_BENCHMARK:-128}
export SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK:-64}
export SMOKE_GATE_ROWS_PER_BENCHMARK=${SMOKE_GATE_ROWS_PER_BENCHMARK:-128}
SMOKE_BENCHMARK_CAPS=toolbench_g3=1024:traject_bench=1024:alfworld=1024:webshop=1024
export BATCH_SIZE=16
export LEARNING_RATE=3.0e-5
export MINIMUM_LEARNING_RATE=3.0e-6
export BASE_CMC_SELECTION_PATH=${BASE_CMC_SELECTION_PATH:-${RUN_ROOT}/stage4_cmc_full/stage4_selection.json}
export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"
export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1

if [[ "${FULL_RUN}" == "1" ]]; then
  export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_candidate_admission_full}
else
  export BENCHMARK_CAPS="${BENCHMARK_CAPS:-${SMOKE_BENCHMARK_CAPS}}"
  export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_candidate_admission_smoke}
fi

exec bash "${PROJECT_ROOT}/scripts/sbatch/run_qwen06_clstr_stage4_train.sh"
