#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"

SMOKE_JOB_ID=${SMOKE_JOB_ID:?SMOKE_JOB_ID is required}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-14B}
TOP_K=${TOP_K:-20}
LOG_PATH=${LOG_PATH:-"${PROJECT_ROOT}/.tmp/qwen_skill_rerank_watch_${SMOKE_JOB_ID}.log"}
FULL_PARTITION=${FULL_PARTITION:-gpu_h200,gpu_h100,gpu_a800}
FULL_QOS=${FULL_QOS:-gpugpu}
SMOKE_POLL_SECONDS=${SMOKE_POLL_SECONDS:-900}
FULL_POLL_SECONDS=${FULL_POLL_SECONDS:-1800}
FULL_TIME=${FULL_TIME:-02:00:00}
FULL_BATCH_SIZE=${FULL_BATCH_SIZE:-1}

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_PATH}"
}

job_pending_or_running() {
  squeue -h -j "$1" >/dev/null 2>&1
}

job_state() {
  sacct -j "$1" --noheader --parsable2 --format=State 2>/dev/null | head -1 | cut -d'|' -f1 || true
}

latest_report() {
  find outputs/toolbench_g3_official_skillrouter_comparison \
    -maxdepth 2 \
    -name 'qwen_skill_rerank_eval_report.json' \
    -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

gate_report() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
parse = report.get("parse") or {}
source = int(report.get("source_eval_rows") or 0)
retained = int(report.get("retained_eval_rows") or 0)
empty = int(parse.get("empty_parse_rows") or 0)
mean_count = float(parse.get("mean_generated_rank_count") or 0.0)
empty_rate = empty / retained if retained else 1.0
ok = (
    report.get("status") == "ok"
    and source > 0
    and retained > 0
    and empty_rate <= 0.25
    and mean_count >= 3.0
)
summary = {
    "ok": ok,
    "report": str(path),
    "source_eval_rows": source,
    "retained_eval_rows": retained,
    "empty_parse_rows": empty,
    "empty_parse_rate": empty_rate,
    "mean_generated_rank_count": mean_count,
    "strict": report.get("strict") or {},
}
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 2)
PY
}

log "watching smoke job ${SMOKE_JOB_ID}"
while job_pending_or_running "${SMOKE_JOB_ID}"; do
  log "smoke queued/running: $(squeue -h -j "${SMOKE_JOB_ID}" -o '%.18i %.20P %.2t %.10M %R' || true)"
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
  log "smoke report missing; not submitting full"
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

MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')
FULL_OUTPUT_DIR="outputs/toolbench_g3_official_skillrouter_comparison/${MODEL_TAG}_skill_rerank_top${TOP_K}_full_$(date +%Y%m%d_%H%M%S)"
FULL_JOB_ID=$(
  OUTPUT_DIR="${FULL_OUTPUT_DIR}" \
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
  TOP_K="${TOP_K}" \
  MAX_EVAL_ROWS=ALL \
  BATCH_SIZE="${FULL_BATCH_SIZE}" \
  sbatch --parsable --partition="${FULL_PARTITION}" --qos="${FULL_QOS}" --time="${FULL_TIME}" \
    scripts/sbatch/run_toolbench_g3_qwen_skill_rerank_eval.sh
)
log "submitted full job ${FULL_JOB_ID} output_dir=${FULL_OUTPUT_DIR}"

while job_pending_or_running "${FULL_JOB_ID}"; do
  log "full queued/running: $(squeue -h -j "${FULL_JOB_ID}" -o '%.18i %.20P %.2t %.10M %R' || true)"
  sleep "${FULL_POLL_SECONDS}"
done

FULL_STATE=$(job_state "${FULL_JOB_ID}")
log "full left queue with state=${FULL_STATE:-unknown}"
if [[ "${FULL_STATE}" != COMPLETED* ]]; then
  log "full did not complete successfully"
  exit 1
fi

FULL_REPORT="${FULL_OUTPUT_DIR}/qwen_skill_rerank_eval_report.json"
if [[ ! -f "${FULL_REPORT}" ]]; then
  log "full report missing at ${FULL_REPORT}"
  exit 1
fi
log "full summary: $(gate_report "${FULL_REPORT}" || true)"
log "watch complete"
