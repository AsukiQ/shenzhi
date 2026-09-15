#!/usr/bin/env bash
set -euo pipefail

ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
CLSTR_SOURCE=${CLSTR_SOURCE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source}
STAGE0_ROOT=${STAGE0_ROOT:-$ROOT/outputs/paper_vnext_stage0_skillrouter_full}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-}
STAGE0_SKILLS=${STAGE0_SKILLS:-$STAGE0_ROOT/selected_skills.jsonl}
STAGE0_BASE_MODEL=${STAGE0_BASE_MODEL:-}
DB=${PAPER_SEARCH_DB:-$ROOT/retrieval_backend/papers_fts.db}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-$ROOT/outputs/service_model_cache}
QUERY_GLOSSARY=${QUERY_GLOSSARY:-$ROOT/retrieval_backend/data/domain_glossary.v1.json}
QUERY_TRANSLATOR_URL=${QUERY_TRANSLATOR_URL:-}
RERANKER_MODEL=${RERANKER_MODEL:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B}
HOST=${RETRIEVAL_HOST:-127.0.0.1}
PORT=${RETRIEVAL_PORT:-8080}
MODE=${RETRIEVAL_MODE:-hybrid}

case "$MODE" in
  bm25|hybrid|full) ;;
  *) echo "RETRIEVAL_MODE must be bm25, hybrid or full" >&2; exit 2 ;;
esac
for required in "$PYTHON_BIN" "$DB"; do
  if [[ ! -e "$required" ]]; then
    echo "required retrieval service input not found: $required" >&2
    exit 2
  fi
done

args=(
  --db "$DB"
  --host "$HOST"
  --port "$PORT"
  --recall-k "${RECALL_K:-1000}"
  --rerank-k "${RERANK_K:-50}"
  --fuzzy-weight "${FUZZY_WEIGHT:-0.35}"
  --graph-weight "${GRAPH_WEIGHT:-0.8}"
  --graph-seed-k "${GRAPH_SEED_K:-12}"
  --graph-expand-k "${GRAPH_EXPAND_K:-100}"
  --graph-value-limit "${GRAPH_VALUE_LIMIT:-16}"
  --max-body-bytes "${MAX_BODY_BYTES:-1048576}"
  --max-workers "${MAX_WORKERS:-16}"
  --worker-slot-wait "${WORKER_SLOT_WAIT:-0.1}"
  --request-timeout "${REQUEST_TIMEOUT:-30}"
  --translator-timeout "${TRANSLATOR_TIMEOUT:-0.8}"
)

if [[ "${ZILLIZ_DENSE_ENABLED:-}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  args+=(--zilliz-dense --zilliz-collection "${ZILLIZ_COLLECTION:-paper_embedding_chunks_v1_1024}")
fi

case "${GRAPH_INTENT_ENABLED:-1}" in
  1|true|TRUE|yes|YES) ;;
  0|false|FALSE|no|NO) args+=(--disable-graph-intent) ;;
  *) echo "GRAPH_INTENT_ENABLED must be 0/1 or true/false" >&2; exit 2 ;;
esac

if [[ -f "$QUERY_GLOSSARY" ]]; then
  args+=(--glossary "$QUERY_GLOSSARY")
fi
if [[ -n "$QUERY_TRANSLATOR_URL" ]]; then
  args+=(--translator-url "$QUERY_TRANSLATOR_URL")
fi

if [[ -n "${AUTH_TOKEN_ENV:-}" ]]; then
  args+=(--auth-token-env "$AUTH_TOKEN_ENV")
fi

if [[ "$MODE" != "bm25" ]]; then
  selection="$STAGE0_ROOT/stage0_selection.json"
  skills="$STAGE0_SKILLS"
  if [[ ! -f "$skills" ]]; then
    echo "Stage0 skills are not ready: $skills" >&2
    exit 3
  fi
  if [[ -n "$STAGE0_CHECKPOINT" ]]; then
    checkpoint="$STAGE0_CHECKPOINT"
  else
    if [[ ! -f "$selection" ]]; then
      echo "Stage0 selection is not ready: $selection" >&2
      exit 3
    fi
    checkpoint=$(
      "$PYTHON_BIN" - "$selection" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
row = json.loads(p.read_text())
if row.get("status") not in {"ok", "complete", "passed"}:
    raise SystemExit(f"Stage0 selection status is not releasable: {row.get('status')}")
print(row["selected_checkpoint_path"])
PY
    )
  fi
  for required in "$checkpoint" "$skills" "$CLSTR_SOURCE"; do
    if [[ ! -e "$required" ]]; then
      echo "required Stage0 service input not found: $required" >&2
      exit 3
    fi
  done
  args+=(
    --stage0-checkpoint "$checkpoint"
    --stage0-skills "$skills"
    --clstr-source "$CLSTR_SOURCE"
    --model-cache-dir "$MODEL_CACHE_DIR"
  )
  if [[ -n "$STAGE0_BASE_MODEL" ]]; then
    if [[ ! -f "$STAGE0_BASE_MODEL/config.json" ]]; then
      echo "Stage0 base model is not ready: $STAGE0_BASE_MODEL" >&2
      exit 3
    fi
    args+=(--stage0-base-model "$STAGE0_BASE_MODEL")
  fi
  args+=(--warmup-query "${WARMUP_QUERY:-knowledge graph retrieval}")
fi

if [[ "$MODE" == "full" ]]; then
  if [[ ! -f "$RERANKER_MODEL/config.json" ]]; then
    echo "required reranker model not found: $RERANKER_MODEL" >&2
    exit 3
  fi
  args+=(
    --reranker-model "$RERANKER_MODEL"
    --reranker-batch-size "${RERANKER_BATCH_SIZE:-4}"
  )
fi

if [[ -n "${NEO4J_HTTP_URI:-}" ]]; then
  if [[ -z "${NEO4J_USER:-}" ]]; then
    echo "NEO4J_HTTP_URI requires NEO4J_USER" >&2
    exit 2
  fi
  args+=(
    --neo4j-http-uri "$NEO4J_HTTP_URI"
    --neo4j-user "$NEO4J_USER"
    --neo4j-password-env "${NEO4J_PASSWORD_ENV:-NEO4J_PASSWORD}"
    --neo4j-database "${NEO4J_DATABASE:-neo4j}"
    --neo4j-timeout "${NEO4J_TIMEOUT:-3}"
  )
fi

mkdir -p "$MODEL_CACHE_DIR"
cd "$ROOT/retrieval_backend"
exec "$PYTHON_BIN" http_server.py "${args[@]}"
