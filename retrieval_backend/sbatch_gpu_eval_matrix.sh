#!/bin/bash
#SBATCH --job-name=shenzhi_eval
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/gpu-eval-%j.out

set -euo pipefail

ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
CLSTR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs/paper_vnext_stage0_skillrouter_full}
CHECKPOINT=${CHECKPOINT:-}
SKILLS=${SKILLS:-$OUTPUT_ROOT/selected_skills.jsonl}
RERANKER=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT=$(/data/home/scyb713/run/miniconda3/envs/xzf/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/outputs/paper_vnext_stage0_skillrouter_full/stage0_selection.json')
if not p.is_file():
    raise SystemExit('full Stage0 selection is not available')
print(json.loads(p.read_text())['selected_checkpoint_path'])
PY
)
fi
for required in "$CHECKPOINT" "$SKILLS" "$RERANKER/config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "required GPU evaluation input not found: $required" >&2
    exit 2
  fi
done

mkdir -p "$ROOT/logs" "$ROOT/outputs/gpu_eval_cache"
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
source activate /data/home/scyb713/run/miniconda3/envs/xzf
export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache

LOCAL_DB=${SLURM_TMPDIR:-/tmp}/shenzhi-papers-${SLURM_JOB_ID}.db
cp "$ROOT/retrieval_backend/papers_fts.db" "$LOCAL_DB"
trap 'rm -f "$LOCAL_DB"' EXIT

cd "$ROOT/retrieval_backend"
python gpu_eval_matrix.py \
  --db "$LOCAL_DB" \
  --queries ../evaluation/weak_test_sample_200_v1.jsonl \
  --checkpoint "$CHECKPOINT" \
  --skills "$SKILLS" \
  --clstr-source "$CLSTR" \
  --model-cache-dir "$ROOT/outputs/gpu_eval_cache" \
  --reranker-model "$RERANKER" \
  --top-k "${TOP_K:-100}" \
  --recall-k "${RECALL_K:-1000}" \
  --rerank-k "${RERANK_K:-20}" \
  --rerank-max-queries "${RERANK_MAX_QUERIES:-200}" \
  --output "$ROOT/outputs/gpu_eval_matrix.json"
