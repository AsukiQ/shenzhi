#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH must name a freshly audited causal-view manifest}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name a fresh canonical full-run directory}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/models/Qwen3-Embedding-0.6B}
STAGE0_SEGMENTS=${STAGE0_SEGMENTS:-"1000 2000 3000 4000 5000"}
STATIC_RERANK_STEPS=${STATIC_RERANK_STEPS:-1200}
STATIC_RERANK_VALIDATION_INTERVAL=${STATIC_RERANK_VALIDATION_INTERVAL:-300}
STAGE2_SEGMENTS=${STAGE2_SEGMENTS:-"2000 4000 6000 8000 10000"}
ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION:-0}
SYNCHRONIZATION_PAIR_DIM=${SYNCHRONIZATION_PAIR_DIM:-96}
SYNCHRONIZATION_TRACE_LENGTH=${SYNCHRONIZATION_TRACE_LENGTH:-8}
SYNCHRONIZATION_SCALE_INITIAL=${SYNCHRONIZATION_SCALE_INITIAL:-0.05}

if [[ -e "${RUN_ROOT}" ]]; then
  printf 'ERROR: canonical full-run root already exists: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/slurm"

common_export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH},RUN_ROOT=${RUN_ROOT},MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH},ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION},SYNCHRONIZATION_PAIR_DIM=${SYNCHRONIZATION_PAIR_DIM},SYNCHRONIZATION_TRACE_LENGTH=${SYNCHRONIZATION_TRACE_LENGTH},SYNCHRONIZATION_SCALE_INITIAL=${SYNCHRONIZATION_SCALE_INITIAL}"
dependency=$(sbatch --parsable --export="${common_export}" \
  --output="${RUN_ROOT}/slurm/precompute-%j.out" \
  "${PROJECT_ROOT}/scripts/sbatch/run_clstr_vnext_full_precompute.sh")

for segment_end in ${STAGE0_SEGMENTS}; do
  dependency=$(sbatch --parsable --dependency="afterok:${dependency}" \
    --export="${common_export},SEGMENT_END=${segment_end}" \
    --output="${RUN_ROOT}/slurm/stage0-${segment_end}-%j.out" \
    "${PROJECT_ROOT}/scripts/sbatch/run_clstr_vnext_full_stage0_segment.sh")
done

dependency=$(sbatch --parsable --dependency="afterok:${dependency}" \
  --export="${common_export},MAX_STEPS=${STATIC_RERANK_STEPS},VALIDATION_INTERVAL=${STATIC_RERANK_VALIDATION_INTERVAL}" \
  --output="${RUN_ROOT}/slurm/static-reranker-${STATIC_RERANK_STEPS}-%j.out" \
  "${PROJECT_ROOT}/scripts/sbatch/run_clstr_vnext_candidate_compressor.sh")

for segment_end in ${STAGE2_SEGMENTS}; do
  dependency=$(sbatch --parsable --dependency="afterok:${dependency}" \
    --export="${common_export},SEGMENT_END=${segment_end},CURRICULUM_TOTAL_STEPS=10000" \
    --output="${RUN_ROOT}/slurm/stage2-${segment_end}-%j.out" \
    "${PROJECT_ROOT}/scripts/sbatch/run_clstr_vnext_full_stage2_segment.sh")
done

printf '%s\n' "${dependency}"
