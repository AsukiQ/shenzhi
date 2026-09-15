#!/bin/bash
#SBATCH --job-name=shenzhi_s2_online
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/stage2-online-smoke-%j.out
set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
source activate /data/home/scyb713/run/miniconda3/envs/xzf
export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
CLSTR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
cd "$ROOT/retrieval_backend"
"$CONDA_PREFIX/bin/python" stage2_online_smoke.py \
  --db "$ROOT/retrieval_backend/papers_fts.db" \
  --checkpoint "$ROOT/outputs/action_vnext_stage2_skillrouter_full/checkpoints/clstr_vnext_stage2-step800.pt" \
  --skills "$ROOT/outputs/action_vnext_stage0_skillrouter_full/selected_skills.jsonl" \
  --clstr-source "$CLSTR" \
  --device cuda \
  --output "$ROOT/outputs/stage2_online_smoke.json"
