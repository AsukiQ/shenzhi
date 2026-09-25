#!/usr/bin/env bash
set -euo pipefail

ROOT=${SHENZHI_ROOT:-/root/jiansuo}
PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE=${PAPER_SEARCH_DB:-$ROOT/retrieval_backend/papers_fts.db}
TARGET=${PAPER_INDEX_TARGET:?Set PAPER_INDEX_TARGET to a NEW database path}

for required in "$SOURCE"; do
  if [[ ! -e "$required" ]]; then
    echo "required paper-index input not found: $required" >&2
    exit 2
  fi
done

exec "$PYTHON_BIN" "$ROOT/retrieval_backend/rebuild_search_index.py" \
  --source "$SOURCE" --target "$TARGET"
