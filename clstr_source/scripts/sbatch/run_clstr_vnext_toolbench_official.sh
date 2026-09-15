#!/bin/bash
#SBATCH --job-name=clstr_tb_sopr
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SOURCE_PROJECT_ROOT}}
ARTIFACT_RUN_ROOT=${ARTIFACT_RUN_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1}
MATCHED_RELEASE_SELECTION_PATH=${MATCHED_RELEASE_SELECTION_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/diagnostics/matched_release_1c94bc2_v1/clstr_vnext_stage2_matched_release.json}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/checkpoints/clstr_vnext_stage2-step500.pt}
TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:-${ARTIFACT_RUN_ROOT}/stage0/selected_skills.jsonl}
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/StableToolBench}
TEST_SET=${TEST_SET:-G3_instruction}
ORIGINAL_QUERY_FILE=${ORIGINAL_QUERY_FILE:-${STABLETOOLBENCH_ROOT}/solvable_queries/test_instruction/${TEST_SET}.json}
OFFICIAL_OUTPUT_ROOT=${OFFICIAL_OUTPUT_ROOT:-outputs/official_closed_loop/toolbench_step500}
PREP_OUTPUT_DIR=${PROJECT_ROOT}/${OFFICIAL_OUTPUT_ROOT}/prepared
RAW_ANSWER_PATH=${RAW_ANSWER_PATH:-${OFFICIAL_OUTPUT_ROOT}/raw_answers}
CONVERTED_ANSWER_PATH=${CONVERTED_ANSWER_PATH:-${OFFICIAL_OUTPUT_ROOT}/converted_answers}
SAVE_PATH=${SAVE_PATH:-${OFFICIAL_OUTPUT_ROOT}/pass_rate}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_vnext_step500_top50}
RUN_EVAL=${RUN_EVAL:-0}
OFFICIAL_PROTOCOL=${OFFICIAL_PROTOCOL:-1}
TOP_K=${TOP_K:-50}
COARSE_K=${COARSE_K:-500}
DYNAMIC_EXTRA_K=${DYNAMIC_EXTRA_K:-64}
EXECUTOR_MODEL=${EXECUTOR_MODEL:-gpt-qwen3-14b}
EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE:-0.0}
MIRROR_MODEL=${MIRROR_MODEL:-gpt-4.1}
MIRROR_TEMPERATURE=${MIRROR_TEMPERATURE:-0.1}
NUM_THREAD=${NUM_THREAD:-1}
EVALUATOR=${EVALUATOR:-tooleval_gpt-3.5-turbo_default}
EVAL_MODEL=${EVAL_MODEL:-gpt-3.5-turbo-1106}
MAX_EVAL_THREADS=${MAX_EVAL_THREADS:-1}
EVALUATE_TIMES=${EVALUATE_TIMES:-3}

if [[ "${OFFICIAL_PROTOCOL}" == "1" ]] && {
  [[ "${TEST_SET}" != "G3_instruction" ]] ||
  [[ "${TOP_K}" != "50" ]] ||
  [[ "${COARSE_K}" != "500" ]] ||
  [[ "${EXECUTOR_MODEL}" != "gpt-qwen3-14b" ]] ||
  [[ "${EXECUTOR_TEMPERATURE}" != "0.0" ]] ||
  [[ "${MIRROR_MODEL}" != "gpt-4.1" ]] ||
  [[ "${NUM_THREAD}" != "1" ]] ||
  [[ "${EVALUATOR}" != "tooleval_gpt-3.5-turbo_default" ]] ||
  [[ "${EVAL_MODEL}" != "gpt-3.5-turbo-1106" ]] ||
  [[ "${MAX_EVAL_THREADS}" != "1" ]] ||
  [[ "${EVALUATE_TIMES}" != "3" ]];
}; then
  echo "ERROR: official ToolBench protocol must use G3/50-from-500, Qwen3-14B, GPT-4.1 MirrorAPI, and the matched three-trial judge" >&2
  exit 2
fi

python3 - "${PROJECT_ROOT}" "${MATCHED_RELEASE_SELECTION_PATH}" \
  "${STAGE2_CHECKPOINT_PATH}" "${TRAINING_SKILLS_PATH}" \
  "${STABLETOOLBENCH_ROOT}" "${ORIGINAL_QUERY_FILE}" <<'PY'
import ast
import json
import sys
from pathlib import Path

project, release, checkpoint, skills, repo, queries = map(Path, sys.argv[1:])
required = [
    release, checkpoint, skills, queries,
    repo / "toolbench" / "inference" / "qa_pipeline_multithread.py",
    repo / "toolbench" / "tooleval" / "eval_pass_rate.py",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing StableToolBench evaluation inputs: " + ", ".join(missing))
query_rows = json.loads(queries.read_text(encoding="utf-8"))
if not isinstance(query_rows, list) or len(query_rows) != 61:
    raise SystemExit("StableToolBench G3 solvable-query denominator must be 61")
for relative in (
    "clstr/vnext_online_selector.py",
    "scripts/run_clstr_vnext_toolbench_sopr_prepare.py",
    "scripts/run_stabletoolbench_generation.py",
    "scripts/run_stabletoolbench_mirrorapi_server.py",
):
    path = project / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("StableToolBench CLSTR closed-loop readiness: OK")
PY

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=${RUN_EVAL}; readiness only, no generation or pass-rate call was made."
  exit 0
fi

export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_qwen14_executor.sh"
PYTHON_BIN=${TOOLBENCH_PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
export PYTHON_BIN
export PYTHONPATH="${PROJECT_ROOT}:${STABLETOOLBENCH_ROOT}:${STABLETOOLBENCH_ROOT}/toolbench/inference:${PYTHONPATH:-}"
EXECUTOR_CONTEXT_LENGTH=${EXECUTOR_CONTEXT_LENGTH:-32768}
export EXECUTOR_MODEL EXECUTOR_CONTEXT_LENGTH
mkdir -p "${PREP_OUTPUT_DIR}"
prepare_args=(
  "${PYTHON_BIN}" scripts/run_clstr_vnext_toolbench_sopr_prepare.py
  --matched_release_selection_path "${MATCHED_RELEASE_SELECTION_PATH}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}"
  --training_skills_path "${TRAINING_SKILLS_PATH}"
  --original_query_file "${ORIGINAL_QUERY_FILE}"
  --output_dir "${PREP_OUTPUT_DIR}"
  --top_k "${TOP_K}"
  --coarse_k "${COARSE_K}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K}"
  --device "${DEVICE:-cuda}"
)
if [[ -n "${MAX_QUERIES:-}" ]]; then prepare_args+=(--max_queries "${MAX_QUERIES}"); fi
"${prepare_args[@]}" | tee "${PREP_OUTPUT_DIR}/stdout.log"
"${PYTHON_BIN}" - "${PREP_OUTPUT_DIR}/preparation_report.json" "${OFFICIAL_PROTOCOL}" "${MAX_QUERIES:-}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "ok":
    raise SystemExit("ToolBench preparation report is not status=ok")
if int(report.get("executable_candidate_count") or 0) != 1586:
    raise SystemExit("ToolBench executable API pool differs from canonical 1,586")
if int(report.get("top_k") or 0) != 50 or int(report.get("coarse_k") or 0) != 500:
    raise SystemExit("ToolBench preparation is not canonical top-50 from coarse-500")
if sys.argv[2] == "1" and not sys.argv[3] and int(report.get("query_count") or 0) != 61:
    raise SystemExit("ToolBench full query denominator differs from canonical 61")
PY

OUTPUT_ROOT_ABS="${PROJECT_ROOT}/${OFFICIAL_OUTPUT_ROOT}"
RAW_ROOT="${PROJECT_ROOT}/${RAW_ANSWER_PATH}"
CONVERTED_ROOT="${PROJECT_ROOT}/${CONVERTED_ANSWER_PATH}"
PASS_RATE_ROOT="${PROJECT_ROOT}/${SAVE_PATH}"
mkdir -p "${OUTPUT_ROOT_ABS}" "${RAW_ROOT}" "${CONVERTED_ROOT}" "${PASS_RATE_ROOT}"

if [[ -z "${EXECUTOR_BASE_URL:-}" ]]; then
  start_clstr_qwen14_executor "${OUTPUT_ROOT_ABS}/qwen3_14b_executor.log"
fi
MIRROR_PID=""
cleanup() {
  if [[ -n "${MIRROR_PID}" ]] && kill -0 "${MIRROR_PID}" 2>/dev/null; then
    kill "${MIRROR_PID}" 2>/dev/null || true
    wait "${MIRROR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${API_POOL_FILE:-}" ]]; then
    rm -f "${API_POOL_FILE}"
  fi
  rm -f "${OUTPUT_ROOT_ABS}/mirrorapi_runtime/config_mirrorapi.yml"
  stop_clstr_qwen14_executor
  stop_clstr_provider_proxy_tunnel
}
trap cleanup EXIT
start_clstr_provider_proxy_tunnel
load_clstr_provider_credentials

OPENAI_PROVIDER_BASE_URL=${OPENAI_PROVIDER_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
OPENAI_PROVIDER_API_KEY=${OPENAI_PROVIDER_API_KEY:-${OPENAI_API_KEY:-}}
if [[ -z "${OPENAI_PROVIDER_API_KEY}" ]]; then
  echo "ERROR: StableToolBench MirrorAPI and judge require OPENAI_API_KEY" >&2
  exit 2
fi
API_POOL_FILE=${API_POOL_FILE:-${OUTPUT_ROOT_ABS}/runtime/openai_key.json}
mkdir -p "$(dirname "${API_POOL_FILE}")"
API_POOL_FILE="${API_POOL_FILE}" \
OPENAI_PROVIDER_API_KEY="${OPENAI_PROVIDER_API_KEY}" \
OPENAI_PROVIDER_BASE_URL="${OPENAI_PROVIDER_BASE_URL}" \
"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["API_POOL_FILE"])
path.write_text(json.dumps([{
    "api_key": os.environ["OPENAI_PROVIDER_API_KEY"],
    "api_base": os.environ["OPENAI_PROVIDER_BASE_URL"],
}], indent=2) + "\n", encoding="utf-8")
path.chmod(0o600)
PY

VIRTUAL_API_PORT=${VIRTUAL_API_PORT:-12001}
SERVICE_URL=${SERVICE_URL:-http://127.0.0.1:${VIRTUAL_API_PORT}/virtual}
export SERVICE_URL
MIRROR_SERVER_API_KEY="${MIRROR_API_KEY:-${OPENAI_PROVIDER_API_KEY}}" \
"${PYTHON_BIN}" -u scripts/run_stabletoolbench_mirrorapi_server.py \
  --server_root "${STABLETOOLBENCH_ROOT}/server" \
  --work_dir "${OUTPUT_ROOT_ABS}/mirrorapi_runtime" \
  --tools_folder "${PREP_OUTPUT_DIR}/toolenv/tools" \
  --api_base "${MIRROR_MODEL_BASE_URL:-${OPENAI_PROVIDER_BASE_URL}}" \
  --model "${MIRROR_MODEL}" \
  --port "${VIRTUAL_API_PORT}" \
  --temperature "${MIRROR_TEMPERATURE}" \
  --max_attempts "${MIRROR_MAX_ATTEMPTS:-3}" \
  --initial_backoff_seconds "${MIRROR_INITIAL_BACKOFF_SECONDS:-1.0}" \
  --trace_path "${OUTPUT_ROOT_ABS}/mirrorapi_runtime/requests.jsonl" \
  >"${OUTPUT_ROOT_ABS}/virtual_api_server.log" 2>&1 &
MIRROR_PID=$!
for _attempt in $(seq 1 60); do
  if ! kill -0 "${MIRROR_PID}" 2>/dev/null; then
    tail -100 "${OUTPUT_ROOT_ABS}/virtual_api_server.log" >&2 || true
    echo "StableToolBench MirrorAPI exited before readiness" >&2
    exit 2
  fi
  if curl -fsS "http://127.0.0.1:${VIRTUAL_API_PORT}/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${VIRTUAL_API_PORT}/docs" >/dev/null

mkdir -p "${RAW_ROOT}/${CANDIDATE_MODEL}/${TEST_SET}"
pushd "${STABLETOOLBENCH_ROOT}" >/dev/null
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_stabletoolbench_generation.py" \
  --tool_root_dir "${PREP_OUTPUT_DIR}/toolenv/tools" \
  --backbone_model chatgpt_function \
  --chatgpt_model "${EXECUTOR_MODEL}" \
  --base_url "${EXECUTOR_BASE_URL}" \
  --openai_key "${EXECUTOR_API_KEY:-EMPTY}" \
  --max_observation_length "${MAX_OBSERVATION_LENGTH:-1024}" \
  --single_chain_max_step "${SINGLE_CHAIN_MAX_STEP:-50}" \
  --max_query_count "${MAX_QUERY_COUNT:-100000}" \
  --method "${METHOD:-CoT@1}" \
  --input_query_file "${PREP_OUTPUT_DIR}/routed_queries.json" \
  --output_answer_file "${RAW_ROOT}/${CANDIDATE_MODEL}/${TEST_SET}" \
  --toolbench_key "${TOOLBENCH_KEY:-}" \
  --num_thread "${NUM_THREAD}" \
  --executor_temperature "${EXECUTOR_TEMPERATURE}" \
  --request_max_attempts "${EXECUTOR_REQUEST_MAX_ATTEMPTS:-3}" \
  --request_initial_backoff_seconds "${EXECUTOR_REQUEST_INITIAL_BACKOFF_SECONDS:-1.0}" \
  | tee "${OUTPUT_ROOT_ABS}/raw_generation_stdout.txt"
popd >/dev/null

MIRROR_TRACE_PATH="${OUTPUT_ROOT_ABS}/mirrorapi_runtime/requests.jsonl"
MIRROR_AUDIT_PATH="${OUTPUT_ROOT_ABS}/mirrorapi_request_report.json"
"${PYTHON_BIN}" - "${MIRROR_TRACE_PATH}" "${MIRROR_AUDIT_PATH}" \
  "${MIRROR_MODEL}" <<'PY'
import collections
import json
import sys
from pathlib import Path

trace_path, output_path = map(Path, sys.argv[1:3])
model = sys.argv[3]
rows = []
if trace_path.is_file():
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "mirrorapi_request":
            rows.append(row)
errors = [row for row in rows if row.get("status") != "ok"]
error_types = collections.Counter(
    str(row.get("exception_type") or "unknown_error") for row in errors
)
report = {
    "status": "ok" if rows and not errors else "action_required",
    "metric_scope": "StableToolBench MirrorAPI infrastructure validity; not SoPR.",
    "model": model,
    "request_count": len(rows),
    "successful_request_count": len(rows) - len(errors),
    "error_count": len(errors),
    "error_types": dict(sorted(error_types.items())),
    "trace_path": str(trace_path),
}
output_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
if report["status"] != "ok":
    raise SystemExit(2)
PY

"${PYTHON_BIN}" scripts/convert_stabletoolbench_answers.py \
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}" \
  --raw_answer_path "${RAW_ROOT}" \
  --converted_answer_path "${CONVERTED_ROOT}" \
  --candidate_model "${CANDIDATE_MODEL}" \
  --test_set "${TEST_SET}" \
  --method "${METHOD:-CoT@1}" \
  --command_root "${PROJECT_ROOT}" \
  --output_path "${OUTPUT_ROOT_ABS}/conversion_report.json" \
  --run_convert \
  --fail_on_action_required

EVAL_RUNTIME="${OUTPUT_ROOT_ABS}/tooleval_runtime"
"${PYTHON_BIN}" scripts/prepare_stabletoolbench_tooleval_runtime.py \
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}" \
  --runtime_dir "${EVAL_RUNTIME}" \
  --api_pool_file "${API_POOL_FILE}" \
  --evaluator "${EVALUATOR}" \
  --output_path "${OUTPUT_ROOT_ABS}/tooleval_runtime_report.json"

mkdir -p "${PASS_RATE_ROOT}/${CANDIDATE_MODEL}"
export EVAL_MODEL
pushd "${EVAL_RUNTIME}" >/dev/null
"${PYTHON_BIN}" eval_pass_rate.py \
  --converted_answer_path "${CONVERTED_ROOT}" \
  --save_path "${PASS_RATE_ROOT}/${CANDIDATE_MODEL}" \
  --reference_model "${CANDIDATE_MODEL}" \
  --test_ids "${STABLETOOLBENCH_ROOT}/solvable_queries/test_query_ids" \
  --evaluator "${EVALUATOR}" \
  --max_eval_threads "${MAX_EVAL_THREADS}" \
  --evaluate_times "${EVALUATE_TIMES}" \
  --test_set "${TEST_SET}" | tee "${OUTPUT_ROOT_ABS}/eval_pass_rate_stdout.txt"
popd >/dev/null

"${PYTHON_BIN}" scripts/summarize_stabletoolbench_sopr.py \
  --labels_path "${PASS_RATE_ROOT}/${CANDIDATE_MODEL}/${TEST_SET}_${CANDIDATE_MODEL}.json" \
  --method clstr_vnext \
  --candidate_model "${CANDIDATE_MODEL}" \
  --test_set "${TEST_SET}" \
  --output_path "${OUTPUT_ROOT_ABS}/clstr_task_accuracy.json"
