#!/bin/bash
#SBATCH --job-name=shenzhi_s0_full
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/stage0-full-%j.out

set -euo pipefail

SHENZHI_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
mkdir -p "$SHENZHI_ROOT/logs"
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
source activate /data/home/scyb713/run/miniconda3/envs/xzf

export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache

cd "$SHENZHI_ROOT"
bash retrieval_backend/run_stage0_full.sh
