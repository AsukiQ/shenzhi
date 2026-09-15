#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-console}
SHENZHI_ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
IMAGE=${NEO4J_IMAGE:-$SHENZHI_ROOT/runtime/neo4j/images/neo4j-5.26-community.sif}
IMPORT_ROOT=${NEO4J_IMPORT_ROOT:-$SHENZHI_ROOT/runtime/neo4j/import_ready}
DATA_ROOT=${NEO4J_DATA_ROOT:-$SHENZHI_ROOT/runtime/neo4j/data}
LOG_ROOT=${NEO4J_LOG_ROOT:-$SHENZHI_ROOT/runtime/neo4j/logs}
CONF_ROOT=${NEO4J_CONF_ROOT:-$SHENZHI_ROOT/runtime/neo4j/conf}
HTTP_PORT=${NEO4J_HTTP_PORT:-17474}
BOLT_PORT=${NEO4J_BOLT_PORT:-17687}
AUTH=${NEO4J_AUTH:-}

if [[ "$ACTION" != "console" ]]; then
  echo "supported action: console" >&2
  exit 2
fi
if [[ -z "$AUTH" ]]; then
  echo "NEO4J_AUTH must be explicit: neo4j/<password> for deployment or none for local smoke" >&2
  exit 2
fi
for required in "$IMAGE" "$DATA_ROOT/databases/neo4j" "$IMPORT_ROOT/constraints.cypher"; do
  if [[ ! -e "$required" ]]; then
    echo "required Neo4j runtime input not found: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_ROOT" "$CONF_ROOT"
module load apptainer/1.4.5
if [[ ! -f "$CONF_ROOT/neo4j.conf" ]]; then
  apptainer exec \
    --bind "$CONF_ROOT:/host-conf" \
    "$IMAGE" \
    sh -lc 'cp -a /var/lib/neo4j/conf/. /host-conf/'
fi
# Authentication mode is selected per launch.  Remove a prior dev-mode value
# so NEO4J_AUTH=neo4j/<password> cannot silently inherit auth_enabled=false.
sed -i -E '/^dbms\.security\.auth_enabled=/d' "$CONF_ROOT/neo4j.conf"
export APPTAINERENV_NEO4J_AUTH="$AUTH"
export APPTAINERENV_NEO4J_server_default__listen__address=0.0.0.0
export APPTAINERENV_NEO4J_server_default__advertised__address=127.0.0.1
export APPTAINERENV_NEO4J_server_http_listen__address=":$HTTP_PORT"
export APPTAINERENV_NEO4J_server_http_advertised__address=":$HTTP_PORT"
export APPTAINERENV_NEO4J_server_bolt_listen__address=":$BOLT_PORT"
export APPTAINERENV_NEO4J_server_bolt_advertised__address=":$BOLT_PORT"
export APPTAINERENV_NEO4J_server_memory_heap_initial__size=${NEO4J_HEAP_INITIAL:-1G}
export APPTAINERENV_NEO4J_server_memory_heap_max__size=${NEO4J_HEAP_MAX:-1G}
export APPTAINERENV_NEO4J_server_memory_pagecache_size=${NEO4J_PAGECACHE:-1G}

# The official image converts every raw NEO4J_* variable into a config key.
# Remove wrapper-only variables after capturing their values, otherwise e.g.
# NEO4J_HTTP_PORT becomes the invalid setting HTTP.PORT inside the container.
unset NEO4J_IMAGE NEO4J_IMPORT_ROOT NEO4J_DATA_ROOT NEO4J_LOG_ROOT NEO4J_CONF_ROOT
unset NEO4J_HTTP_PORT NEO4J_BOLT_PORT NEO4J_HEAP_INITIAL NEO4J_HEAP_MAX NEO4J_PAGECACHE
unset NEO4J_AUTH
unset NEO4J_HTTP_URI NEO4J_URI NEO4J_USER NEO4J_PASSWORD NEO4J_PASSWORD_ENV NEO4J_DATABASE

exec apptainer exec \
  --bind "$DATA_ROOT:/data" \
  --bind "$LOG_ROOT:/logs" \
  --bind "$CONF_ROOT:/var/lib/neo4j/conf" \
  --bind "$IMPORT_ROOT:/import:ro" \
  "$IMAGE" \
  /startup/docker-entrypoint.sh neo4j console
