#!/bin/bash
#SBATCH --time=00:30:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_stage0_handoff_coverage_audit/v3_smoke}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/clstr_stage0_handoff_coverage_audit/v3_smoke/report.json}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-${OUTPUT_DIR}/model_cache}
TOP_K_VALUES=${TOP_K_VALUES:-20,50,100,200,500}
QUERY_MODES=${QUERY_MODES:-skillrouter_state}
MAX_ROWS=${MAX_ROWS:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
REQUIRE_MULTI_TOP_K=${REQUIRE_MULTI_TOP_K:-1}
MIN_GLOBAL_NEXT_RECALL_AT_500=${MIN_GLOBAL_NEXT_RECALL_AT_500:-0.90}
MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500=${MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500:-0.90}
MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500:-0.88}
MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200:-0.75}

TOP_K_VALUES="${TOP_K_VALUES//:/,}"
TOP_K_VALUES="${TOP_K_VALUES//;/,}"
QUERY_MODES="${QUERY_MODES//:/,}"
QUERY_MODES="${QUERY_MODES//;/,}"

case "${MAX_ROWS}" in
  all|ALL|none|NONE|null|NULL|0)
    MAX_ROWS=""
    ;;
esac

TOP_K_VALUES_COUNT=0
IFS=',' read -r -a _TOP_K_PARTS <<< "${TOP_K_VALUES}"
for _top_k_part in "${_TOP_K_PARTS[@]}"; do
  if [[ -n "${_top_k_part//[[:space:]]/}" ]]; then
    ((TOP_K_VALUES_COUNT+=1))
  fi
done
unset _TOP_K_PARTS _top_k_part

if [[ "${REQUIRE_MULTI_TOP_K}" = "1" && "${TOP_K_VALUES_COUNT}" -lt 2 ]]; then
  echo "ERROR: TOP_K_VALUES resolved to a single value (${TOP_K_VALUES}). Use colon-separated TOP_K_VALUES when exporting through sbatch, for example TOP_K_VALUES=20:50:100:200:500; comma-separated values are safe only inside this script/defaults." >&2
  exit 2
fi

if [[ -s "${CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 handoff audit requires completed checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -s "${TRAIN_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 handoff audit requires trajectories: ${TRAIN_PATH}" >&2
  exit 2
fi

if [[ -s "${SKILLS_PATH}" ]]; then
  :
else
  echo "ERROR: Stage0 handoff audit requires skill pool: ${SKILLS_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${MODEL_CACHE_DIR}"
rm -f "${OUTPUT_PATH}"
rm -f "${OUTPUT_DIR}/handoff_audit_stdout.json"

ARGS=(
  scripts/audit_clstr_stage0_handoff_coverage.py
  --checkpoint_path "${CHECKPOINT_PATH}"
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_path "${OUTPUT_PATH}"
  --top_k_values "${TOP_K_VALUES}"
  --query_modes "${QUERY_MODES}"
  --batch_size "${BATCH_SIZE}"
  --model_cache_dir "${MODEL_CACHE_DIR}"
)

if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max_rows "${MAX_ROWS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/handoff_audit_stdout.json"

"${PYTHON_BIN}" - \
  "${OUTPUT_PATH}" \
  "${MIN_GLOBAL_NEXT_RECALL_AT_500}" \
  "${MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500}" \
  "${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500}" \
  "${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
thresholds = {
    "global.next_recall@500": float(sys.argv[2]),
    "toolbench_g3.next_recall@500": float(sys.argv[3]),
    "traject_bench.next_recall@500": float(sys.argv[4]),
    "traject_bench.next_recall@200": float(sys.argv[5]),
}
report = json.loads(report_path.read_text(encoding="utf-8"))
mode_report = (report.get("query_modes") or {}).get("skillrouter_state") or {}
global_metrics = mode_report.get("global") or {}
benchmark_metrics = mode_report.get("benchmarks") or {}

observed = {
    "global.next_recall@500": global_metrics.get("next_recall@500"),
    "toolbench_g3.next_recall@500": (benchmark_metrics.get("toolbench_g3") or {}).get("next_recall@500"),
    "traject_bench.next_recall@500": (benchmark_metrics.get("traject_bench") or {}).get("next_recall@500"),
    "traject_bench.next_recall@200": (benchmark_metrics.get("traject_bench") or {}).get("next_recall@200"),
}
blockers = []
for name, threshold in thresholds.items():
    value = observed.get(name)
    if value is None:
        blockers.append({"metric": name, "reason": "missing", "threshold": threshold})
    elif float(value) < threshold:
        blockers.append({"metric": name, "value": float(value), "threshold": threshold})

gate = {
    "status": "action_required" if blockers else "ok",
    "thresholds": thresholds,
    "observed": observed,
    "blockers": blockers,
}
gate_path = report_path.with_name("handoff_coverage_gate.json")
gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if blockers:
    print("ERROR: handoff coverage gate blockers", file=sys.stderr)
    print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(3)
print(json.dumps(gate, ensure_ascii=False, indent=2))
PY
