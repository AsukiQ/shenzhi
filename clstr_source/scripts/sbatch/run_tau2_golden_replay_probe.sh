#!/bin/bash
#SBATCH --job-name=tau2_replay_probe
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAU2_REPO_ROOT=${TAU2_REPO_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench}
MAX_TASKS_PER_DOMAIN=${MAX_TASKS_PER_DOMAIN:-2}
TAU2_PYTHON_BIN=${TAU2_PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}

"${TAU2_PYTHON_BIN}" scripts/probe_tau2_golden_replay.py \
  --tau2_repo_root "${TAU2_REPO_ROOT}" \
  --max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}"
