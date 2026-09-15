#!/bin/bash
#SBATCH --job-name=shenzhi_infer
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/inference-smoke-%j.out

set -euo pipefail

ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
CLSTR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
CHECKPOINT="$ROOT/outputs/paper_vnext_stage0_skillrouter_smoke/checkpoints/clstr_vnext_stage0-step20.pt"
SKILLS="$ROOT/outputs/paper_vnext_stage0_skillrouter_smoke/selected_skills.jsonl"
RERANKER=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B

mkdir -p "$ROOT/logs" "$ROOT/outputs/inference_smoke_cache"
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
source activate /data/home/scyb713/run/miniconda3/envs/xzf
export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache

cd "$ROOT/retrieval_backend"
python gpu_inference_smoke.py \
  --db papers_fts.db \
  --checkpoint "$CHECKPOINT" \
  --skills "$SKILLS" \
  --clstr-source "$CLSTR" \
  --model-cache-dir "$ROOT/outputs/inference_smoke_cache" \
  --reranker-model "$RERANKER" \
  --output "$ROOT/outputs/gpu_inference_smoke.json"
