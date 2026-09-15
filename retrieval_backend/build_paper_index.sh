#!/usr/bin/env bash
set -euo pipefail

ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
DOCUMENTS=${PAPER_DOCUMENTS:-$ROOT/derived/paper_data_v1/retrieval_documents.jsonl}
DB=${PAPER_SEARCH_DB:-$ROOT/retrieval_backend/papers_fts.db}

for required in "$PYTHON_BIN" "$DOCUMENTS"; do
  if [[ ! -e "$required" ]]; then
    echo "required paper-index input not found: $required" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$DB")"
exec "$PYTHON_BIN" "$ROOT/retrieval_backend/paper_search.py" build \
  --documents "$DOCUMENTS" \
  --db "$DB" \
  --replace
