#!/usr/bin/env bash
set -euo pipefail

SHENZHI_ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
IMAGE=${NEO4J_IMAGE:-$SHENZHI_ROOT/runtime/neo4j/images/neo4j-5.26-community.sif}
IMPORT_ROOT=${NEO4J_IMPORT_ROOT:-$SHENZHI_ROOT/runtime/neo4j/import_ready}
DATA_ROOT=${NEO4J_DATA_ROOT:-$SHENZHI_ROOT/runtime/neo4j/data}
LOG_ROOT=${NEO4J_LOG_ROOT:-$SHENZHI_ROOT/runtime/neo4j/logs}
REPORT=${NEO4J_IMPORT_REPORT:-$SHENZHI_ROOT/runtime/neo4j/import.report}
THREADS=${NEO4J_IMPORT_THREADS:-6}
OFF_HEAP=${NEO4J_IMPORT_OFF_HEAP:-8G}

for required in \
  "$IMAGE" \
  "$IMPORT_ROOT/manifest.json" \
  "$IMPORT_ROOT/paper_kg/papers.csv" \
  "$IMPORT_ROOT/paper_kg/cites.csv"; do
  if [[ ! -e "$required" ]]; then
    echo "required Neo4j import input not found: $required" >&2
    exit 2
  fi
done

if [[ -d "$DATA_ROOT/databases/neo4j" ]]; then
  echo "Neo4j database already exists: $DATA_ROOT/databases/neo4j" >&2
  echo "Use a new NEO4J_DATA_ROOT; this script never deletes an existing database." >&2
  exit 3
fi

mkdir -p "$DATA_ROOT" "$LOG_ROOT" "$(dirname "$REPORT")"
module load apptainer/1.4.5

apptainer exec \
  --bind "$DATA_ROOT:/data" \
  --bind "$LOG_ROOT:/logs" \
  --bind "$IMPORT_ROOT:/import:ro" \
  "$IMAGE" \
  sh -lc '
    export JAVA_HOME=/opt/java/openjdk
    export PATH=/opt/java/openjdk/bin:/var/lib/neo4j/bin:$PATH
    neo4j-admin database import full neo4j \
      --id-type=string \
      --input-encoding=UTF-8 \
      --strict=true \
      --bad-tolerance=0 \
      --skip-bad-relationships=false \
      --skip-duplicate-nodes=false \
      --threads="'"$THREADS"'" \
      --max-off-heap-memory="'"$OFF_HEAP"'" \
      --report-file=/logs/import.report \
      --nodes=/import/paper_kg/papers.csv \
      --nodes=/import/paper_kg/authors.csv \
      --nodes=/import/paper_kg/venues.csv \
      --nodes=/import/paper_kg/keywords.csv \
      --nodes=/import/paper_kg/subjects.csv \
      --nodes=/import/paper_kg/years.csv \
      --relationships=/import/paper_kg/authored_by.csv \
      --relationships=/import/paper_kg/published_in.csv \
      --relationships=/import/paper_kg/published_year.csv \
      --relationships=/import/paper_kg/has_keyword.csv \
      --relationships=/import/paper_kg/has_subject.csv \
      --relationships=/import/paper_kg/cites.csv
  '

if [[ -f "$LOG_ROOT/import.report" ]]; then
  cp "$LOG_ROOT/import.report" "$REPORT"
fi
printf 'Neo4j bulk import completed: %s\n' "$DATA_ROOT/databases/neo4j"
