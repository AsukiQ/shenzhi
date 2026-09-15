#!/bin/bash
#SBATCH --job-name=q06_tb_global_corpus
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
BASELINE_PROJECT_ROOT=${BASELINE_PROJECT_ROOT:?BASELINE_PROJECT_ROOT is required}
NATIVE_PROFILED_CORPUS_DIR=${NATIVE_PROFILED_CORPUS_DIR:?NATIVE_PROFILED_CORPUS_DIR is required}
GLOBAL_SKILLS_PATH=${GLOBAL_SKILLS_PATH:?GLOBAL_SKILLS_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

args=(
  "${PYTHON_BIN}" scripts/build_qwen06_toolbench_global_baseline_corpus.py
  --baseline_project_root "${BASELINE_PROJECT_ROOT}"
  --native_profiled_corpus_dir "${NATIVE_PROFILED_CORPUS_DIR}"
  --global_skills_path "${GLOBAL_SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

"${args[@]}"
