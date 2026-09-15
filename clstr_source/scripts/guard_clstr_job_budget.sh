#!/bin/bash
set -euo pipefail

MAX_ACTIVE_JOBS=${MAX_ACTIVE_JOBS:-4}

if ! [[ "${MAX_ACTIVE_JOBS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_ACTIVE_JOBS must be an integer, got ${MAX_ACTIVE_JOBS}" >&2
  exit 2
fi

if [[ "${MAX_ACTIVE_JOBS}" -le 0 ]]; then
  echo "ERROR: MAX_ACTIVE_JOBS must be positive, got ${MAX_ACTIVE_JOBS}" >&2
  exit 2
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "WARN: squeue not available; cannot enforce active job budget" >&2
  exit 0
fi

active_jobs=$(squeue -u "${USER}" -h -o "%T" | awk '
  $1 == "RUNNING" || $1 == "PENDING" || $1 == "CONFIGURING" || $1 == "COMPLETING" { count += 1 }
  END { print count + 0 }
')

if [[ "${active_jobs}" -ge "${MAX_ACTIVE_JOBS}" ]]; then
  echo "ERROR: active Slurm jobs for ${USER} = ${active_jobs}, limit = ${MAX_ACTIVE_JOBS}; refusing to submit another job" >&2
  exit 3
fi

echo "Active Slurm jobs for ${USER}: ${active_jobs}/${MAX_ACTIVE_JOBS}"
