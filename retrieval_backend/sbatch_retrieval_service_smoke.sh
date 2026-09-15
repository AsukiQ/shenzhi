#!/bin/bash
#SBATCH --job-name=shenzhi_http
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi/logs/http-smoke-%j.out

set -euo pipefail

ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
PORT=${RETRIEVAL_PORT:-18181}
NEO4J_HTTP_PORT=${NEO4J_HTTP_PORT:-17475}
NEO4J_BOLT_PORT=${NEO4J_BOLT_PORT:-17688}
mkdir -p "$ROOT/logs"
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true
module load apptainer/1.4.5
source activate /data/home/scyb713/run/miniconda3/envs/xzf
export PYTHONUNBUFFERED=1
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export RETRIEVAL_MODE=full
export RETRIEVAL_HOST=127.0.0.1
export RETRIEVAL_PORT="$PORT"
export NEO4J_HTTP_URI="http://127.0.0.1:$NEO4J_HTTP_PORT"
export NEO4J_USER=${NEO4J_USER:-neo4j}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-unused}
LOCAL_DB=${SLURM_TMPDIR:-/tmp}/shenzhi-papers-${SLURM_JOB_ID}.db
cp "$ROOT/retrieval_backend/papers_fts.db" "$LOCAL_DB"
export PAPER_SEARCH_DB="$LOCAL_DB"
LOCAL_NEO4J=${SLURM_TMPDIR:-/tmp}/shenzhi-neo4j-${SLURM_JOB_ID}
mkdir -p "$LOCAL_NEO4J/data" "$LOCAL_NEO4J/logs" "$LOCAL_NEO4J/conf"
cp -a "$ROOT/runtime/neo4j/data/." "$LOCAL_NEO4J/data/"
cp -a "$ROOT/runtime/neo4j/conf/." "$LOCAL_NEO4J/conf/"
sed -i -E '/^(server\.(http|bolt)\.(listen|advertised)_address|server\.default_(listen|advertised)_address|dbms\.security\.auth_enabled)=/d' \
  "$LOCAL_NEO4J/conf/neo4j.conf"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -n "${neo4j_pid:-}" ]]; then
    kill "$neo4j_pid" 2>/dev/null || true
    wait "$neo4j_pid" 2>/dev/null || true
  fi
  rm -f "$LOCAL_DB"
  rm -rf "$LOCAL_NEO4J"
}
trap cleanup EXIT

NEO4J_DATA_ROOT="$LOCAL_NEO4J/data" \
NEO4J_LOG_ROOT="$LOCAL_NEO4J/logs" \
NEO4J_CONF_ROOT="$LOCAL_NEO4J/conf" \
NEO4J_AUTH=none \
NEO4J_HTTP_PORT="$NEO4J_HTTP_PORT" \
NEO4J_BOLT_PORT="$NEO4J_BOLT_PORT" \
bash "$ROOT/retrieval_backend/run_neo4j_apptainer.sh" console &
neo4j_pid=$!
for _attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$NEO4J_HTTP_PORT" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$neo4j_pid" 2>/dev/null; then
    wait "$neo4j_pid"
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$NEO4J_HTTP_PORT" >/dev/null

bash "$ROOT/retrieval_backend/run_retrieval_service.sh" &
server_pid=$!
for _attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
python "$ROOT/retrieval_backend/http_smoke_test.py" \
  --base-uri "http://127.0.0.1:$PORT" \
  --expect-component bm25 \
  --expect-component stage0_dense \
  --expect-component skillrouter_reranker \
  --expect-component neo4j_filter \
  --output "$ROOT/outputs/http_service_smoke.json"
