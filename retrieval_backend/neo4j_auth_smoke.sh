#!/usr/bin/env bash
set -euo pipefail

ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
HTTP_PORT=${NEO4J_AUTH_SMOKE_HTTP_PORT:-17476}
BOLT_PORT=${NEO4J_AUTH_SMOKE_BOLT_PORT:-17689}
PASSWORD=${NEO4J_AUTH_SMOKE_PASSWORD:-ShenzhiSmoke123}
TEMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/shenzhi-neo4j-auth.XXXXXX")

cleanup() {
  if [[ -n "${neo4j_pid:-}" ]]; then
    kill "$neo4j_pid" 2>/dev/null || true
    wait "$neo4j_pid" 2>/dev/null || true
  fi
  rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$TEMP_ROOT/data" "$TEMP_ROOT/logs" "$TEMP_ROOT/conf"
cp -a "$ROOT/runtime/neo4j/data/." "$TEMP_ROOT/data/"
cp -a "$ROOT/runtime/neo4j/conf/." "$TEMP_ROOT/conf/"
sed -i -E '/^dbms\.security\.auth_enabled=/d' "$TEMP_ROOT/conf/neo4j.conf"

NEO4J_DATA_ROOT="$TEMP_ROOT/data" \
NEO4J_LOG_ROOT="$TEMP_ROOT/logs" \
NEO4J_CONF_ROOT="$TEMP_ROOT/conf" \
NEO4J_AUTH="neo4j/$PASSWORD" \
NEO4J_HTTP_PORT="$HTTP_PORT" \
NEO4J_BOLT_PORT="$BOLT_PORT" \
bash "$ROOT/retrieval_backend/run_neo4j_apptainer.sh" console &
neo4j_pid=$!

for _attempt in $(seq 1 120); do
  if "$PYTHON_BIN" - "$HTTP_PORT" "$PASSWORD" <<'PY' >/dev/null 2>&1
import sys
from urllib import request
import base64

port, password = sys.argv[1:]
token = base64.b64encode(f"neo4j:{password}".encode()).decode()
req = request.Request(
    f"http://127.0.0.1:{port}/db/neo4j/tx/commit",
    data=b'{"statements":[{"statement":"RETURN 1 AS ok"}]}',
    headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
)
with request.urlopen(req, timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
  then
    break
  fi
  if ! kill -0 "$neo4j_pid" 2>/dev/null; then
    wait "$neo4j_pid"
  fi
  sleep 1
done

"$PYTHON_BIN" "$ROOT/retrieval_backend/apply_neo4j_schema.py" \
  --http-uri "http://127.0.0.1:$HTTP_PORT" \
  --user neo4j --password "$PASSWORD" \
  --cypher "$ROOT/runtime/neo4j/import_ready/constraints.cypher" \
  --output "$ROOT/outputs/neo4j_auth_smoke.json"
