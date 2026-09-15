#!/usr/bin/env bash
set -euo pipefail

ARCHIVE=${ARCHIVE:-/root/jiansuo/merged_rich_graph_backend_final_20260821.tar.gz}
MEMBER=${MEMBER:-merged_rich_graph/retrieval_documents.jsonl}
TARGET=${TARGET:-/root/jiansuo/retrieval_backend/papers_fts.final.db}
WORK_DIR=$(mktemp -d /root/jiansuo/.bm25-final-build.XXXXXX)
DOCUMENT_PIPE=$WORK_DIR/retrieval_documents.pipe
FILTER_LOG=$WORK_DIR/filter.log
WRITER_PID=""

stop_writer() {
  if [[ -n "$WRITER_PID" ]] && kill -0 "$WRITER_PID" 2>/dev/null; then
    kill "$WRITER_PID" 2>/dev/null || true
    wait "$WRITER_PID" 2>/dev/null || true
  fi
}
trap stop_writer EXIT INT TERM

if [[ -e "$TARGET" || -e "$TARGET-wal" || -e "$TARGET-shm" ]]; then
  echo "target already exists: $TARGET" >&2
  exit 2
fi

mkfifo "$DOCUMENT_PIPE"
(
  set -o pipefail
  tar -xOf "$ARCHIVE" "$MEMBER" |
    node /root/jiansuo/retrieval_backend/deployment/filter_retrieval_documents.mjs \
      paper:2604.11820v1 \
      paper:2604.26488v1 \
      paper:2605.02290v1 \
      paper:2605.26442v1 > "$DOCUMENT_PIPE"
) 2> "$FILTER_LOG" &
WRITER_PID=$!

nice -n 10 python3 /root/jiansuo/retrieval_backend/paper_search.py build \
  --documents "$DOCUMENT_PIPE" \
  --db "$TARGET" \
  --replace
wait "$WRITER_PID"
WRITER_PID=""

sed -n '1,20p' "$FILTER_LOG"
ls -lh "$TARGET" "$TARGET-wal" "$TARGET-shm" 2>/dev/null || true
df -h /
echo "build work directory retained at: $WORK_DIR"
