#!/bin/bash
#SBATCH --job-name=shenzhi_action_s0
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/action-stage0-smoke-%j.out
set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
source activate /data/home/scyb713/run/miniconda3/envs/xzf
export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
bash retrieval_backend/run_action_stage0_smoke.sh
