#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/jiansuo
OUT="$ROOT/MIGRATION_SHA256SUMS"
> "$OUT"
for f in \
  "$ROOT/retrieval_backend/papers_fts.db" \
  "$ROOT/neo4j_store_preserved_20260924.tar.gz" \
  "$ROOT/demo_aaai2026/papers_fts.db" \
  "$ROOT/AAAI-2026.zip"; do
  [[ -f "$f" ]] || continue
  sha256sum "$f" >> "$OUT"
done
printf 'Wrote %s\n' "$OUT"
