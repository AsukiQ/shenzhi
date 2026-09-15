#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/jiansuo
set -a
source "$ROOT/retrieval_backend/deployment/paperddl.env.bm25"
set +a
exec bash "$ROOT/retrieval_backend/run_retrieval_service.sh"
