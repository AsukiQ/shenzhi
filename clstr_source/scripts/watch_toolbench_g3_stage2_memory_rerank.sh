#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"

SMOKE_JOB_ID=${SMOKE_JOB_ID:?SMOKE_JOB_ID is required}
LOG_PATH=${LOG_PATH:-"${PROJECT_ROOT}/.tmp/stage2_memory_rerank_watch_${SMOKE_JOB_ID}.log"}
FULL_PARTITION=${FULL_PARTITION:-gpu_h200,gpu_h100,gpu_a800}
FULL_QOS=${FULL_QOS:-gpugpu}
SMOKE_POLL_SECONDS=${SMOKE_POLL_SECONDS:-900}
FULL_POLL_SECONDS=${FULL_POLL_SECONDS:-1800}
MRR_TOLERANCE=${MRR_TOLERANCE:-0.005}
RECALL5_TOLERANCE=${RECALL5_TOLERANCE:-0.01}

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_PATH}"
}

latest_report() {
  find outputs/toolbench_g3_official_skillrouter_comparison \
    -maxdepth 2 \
    -name 'stage2_memory_rerank_eval_report.json' \
    -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

job_state() {
  sacct -j "$1" --noheader --parsable2 --format=State 2>/dev/null | head -1 | cut -d'|' -f1 || true
}

job_pending_or_running() {
  squeue -h -j "$1" >/dev/null 2>&1
}

gate_report() {
  local report_path="$1"
  python - "$report_path" "$MRR_TOLERANCE" "$RECALL5_TOLERANCE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mrr_tol = float(sys.argv[2])
recall5_tol = float(sys.argv[3])
report = json.loads(path.read_text())
strict = report.get("strict") or {}
s2 = strict.get("stage2_only") or {}
mem = strict.get("stage2_plus_memory") or {}
delta = report.get("strict_delta") or {}
source_rows = int(report.get("source_eval_rows") or 0)
retained_rows = int(report.get("retained_eval_rows") or 0)
agg = report.get("aggregate") or {}
memory_hits = float(agg.get("stage2_memory_online_memory_hit_rows") or 0.0)

s2_mrr = float(s2.get("strict_transition_skill_mrr") or 0.0)
mem_mrr = float(mem.get("strict_transition_skill_mrr") or 0.0)
s2_r5 = float(s2.get("strict_transition_skill_recall@5") or 0.0)
mem_r5 = float(mem.get("strict_transition_skill_recall@5") or 0.0)
ok = (
    report.get("status") == "ok"
    and source_rows > 0
    and retained_rows > 0
    and memory_hits > 0
    and mem_mrr >= s2_mrr - mrr_tol
    and mem_r5 >= s2_r5 - recall5_tol
)
summary = {
    "ok": ok,
    "report": str(path),
    "source_rows": source_rows,
    "retained_rows": retained_rows,
    "memory_hit_rows": memory_hits,
    "stage2_mrr": s2_mrr,
    "memory_mrr": mem_mrr,
    "delta_mrr": mem_mrr - s2_mrr,
    "stage2_recall5": s2_r5,
    "memory_recall5": mem_r5,
    "delta_recall5": mem_r5 - s2_r5,
    "strict_delta": delta,
}
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 2)
PY
}

log "watching smoke job ${SMOKE_JOB_ID}"
while job_pending_or_running "${SMOKE_JOB_ID}"; do
  log "smoke still queued/running: $(squeue -h -j "${SMOKE_JOB_ID}" -o '%.18i %.20P %.2t %.10M %R' || true)"
  sleep "${SMOKE_POLL_SECONDS}"
done

SMOKE_STATE=$(job_state "${SMOKE_JOB_ID}")
log "smoke left queue with state=${SMOKE_STATE:-unknown}"
if [[ "${SMOKE_STATE}" != COMPLETED* ]]; then
  log "smoke did not complete successfully; not submitting full"
  exit 1
fi

REPORT_PATH=$(latest_report)
if [[ -z "${REPORT_PATH}" || ! -f "${REPORT_PATH}" ]]; then
  log "no stage2 memory rerank report found; not submitting full"
  exit 1
fi

set +e
GATE_OUTPUT=$(gate_report "${REPORT_PATH}" 2>&1)
GATE_STATUS=$?
set -e
log "smoke gate: ${GATE_OUTPUT}"
if [[ "${GATE_STATUS}" -ne 0 ]]; then
  log "smoke gate failed; not submitting full"
  exit 1
fi

FULL_OUTPUT_DIR="outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_memory_rerank_full_$(date +%Y%m%d_%H%M%S)"
FULL_JOB_ID=$(
  OUTPUT_DIR="${FULL_OUTPUT_DIR}" \
  MAX_EVAL_ROWS=ALL \
  sbatch --parsable --partition="${FULL_PARTITION}" --qos="${FULL_QOS}" --time=00:30:00 \
    scripts/sbatch/run_toolbench_g3_stage2_memory_rerank_eval.sh
)
log "submitted full job ${FULL_JOB_ID} output_dir=${FULL_OUTPUT_DIR}"

while job_pending_or_running "${FULL_JOB_ID}"; do
  log "full still queued/running: $(squeue -h -j "${FULL_JOB_ID}" -o '%.18i %.20P %.2t %.10M %R' || true)"
  sleep "${FULL_POLL_SECONDS}"
done

FULL_STATE=$(job_state "${FULL_JOB_ID}")
log "full left queue with state=${FULL_STATE:-unknown}"
if [[ "${FULL_STATE}" != COMPLETED* ]]; then
  log "full did not complete successfully"
  exit 1
fi

FULL_REPORT="${FULL_OUTPUT_DIR}/stage2_memory_rerank_eval_report.json"
if [[ ! -f "${FULL_REPORT}" ]]; then
  log "full report missing at ${FULL_REPORT}"
  exit 1
fi
log "full gate summary: $(gate_report "${FULL_REPORT}" || true)"
log "watch complete"
