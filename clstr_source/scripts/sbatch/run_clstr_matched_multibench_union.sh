#!/bin/bash
#SBATCH --job-name=clstr_matched_union
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

BASE_DATA_ROOT=${BASE_DATA_ROOT:?BASE_DATA_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
TAU2_DATA_ROOT=${TAU2_DATA_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench/data/tau2}
TOOLSANDBOX_SCENARIOS_ROOT=${TOOLSANDBOX_SCENARIOS_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios}
TOOLSANDBOX_TOOLS_ROOT=${TOOLSANDBOX_TOOLS_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools}

"${PYTHON_BIN}" scripts/build_clstr_matched_multibench_union.py \
  --base_data_root "${BASE_DATA_ROOT}" \
  --tau2_data_root "${TAU2_DATA_ROOT}" \
  --toolsandbox_scenarios_root "${TOOLSANDBOX_SCENARIOS_ROOT}" \
  --toolsandbox_tools_root "${TOOLSANDBOX_TOOLS_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --split_seed clstr-matched-v1 \
  --tau2_dev_fraction 0.2 \
  --toolsandbox_train_fraction 0.6 \
  --toolsandbox_dev_fraction 0.2 \
  | tee "${OUTPUT_ROOT}.stdout.json"
