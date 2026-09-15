#!/bin/bash
#SBATCH --job-name=clstr-vnext-tb-export
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT must contain the audited clean skill/trajectory files}
TOOLBENCH_SOURCE_ROOT=${TOOLBENCH_SOURCE_ROOT:?TOOLBENCH_SOURCE_ROOT must contain raw ToolBench data}
SOURCE_EVAL_TRAJECTORIES=${SOURCE_EVAL_TRAJECTORIES:?SOURCE_EVAL_TRAJECTORIES must name the SR/ToolREx-aligned official split}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be a fresh verified ToolBench export directory}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: verified ToolBench output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi
RAW_IMPORT_DIR=${RAW_IMPORT_DIR:-${OUTPUT_DIR}.raw_import}
if [[ -e "${RAW_IMPORT_DIR}" ]]; then
  printf 'ERROR: raw ToolBench import output already exists: %s\n' "${RAW_IMPORT_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
"${PYTHON_BIN}" scripts/import_toolbench_g3.py \
  --source_root "${TOOLBENCH_SOURCE_ROOT}" \
  --output_dir "${RAW_IMPORT_DIR}" \
  | tee "${RAW_IMPORT_DIR}.stdout.json"
"${PYTHON_BIN}" scripts/verify_toolbench_g3_official_eval_results.py \
  --source_eval_trajectories_path "${SOURCE_EVAL_TRAJECTORIES}" \
  --skills_path "${DATA_ROOT}/skill_pool.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --verification_skills_path "${RAW_IMPORT_DIR}/skills.jsonl" \
  | tee "${OUTPUT_DIR}.stdout.json"
