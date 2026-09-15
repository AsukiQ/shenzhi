#!/bin/bash
#SBATCH --job-name=clstr_tau2_e3
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/clstr" ]]; then
    PROJECT_ROOT=${SLURM_SUBMIT_DIR}
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  fi
fi
METHOD=${METHOD:?METHOD must be static, transformer, or lstr}
if [[ "${METHOD}" != "static" && "${METHOD}" != "transformer" && "${METHOD}" != "lstr" ]]; then
  echo "ERROR: unsupported E3 METHOD=${METHOD}" >&2
  exit 2
fi

ARTIFACT_RUN_ROOT=${ARTIFACT_RUN_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1}
E1_ROOT=${E1_ROOT:-${ARTIFACT_RUN_ROOT}/matched_history_e1_v1}
MATCHED_ROOT=${MATCHED_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/e3039e2/matched_union}
FOUNDATION_CHECKPOINT_PATH=${FOUNDATION_CHECKPOINT_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/checkpoints/clstr_vnext_stage2-step0.pt}
SKILLS_PATH=${SKILLS_PATH:-${ARTIFACT_RUN_ROOT}/stage0/selected_skills.jsonl}
MATCHED_UNION_MANIFEST_PATH=${MATCHED_UNION_MANIFEST_PATH:-${MATCHED_ROOT}/matched_union_manifest.json}
TAU2_ROOT=${TAU2_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench}
CLSTR_PROVIDER_CREDENTIAL_FILE=${CLSTR_PROVIDER_CREDENTIAL_FILE:-${PROJECT_ROOT}/outputs/official_closed_loop/toolbench_step500_smoke_g3_1_retry/runtime/openai_key.json}
OUTPUT_DIR=${OUTPUT_DIR:-${E1_ROOT}/controlled_tau2_e3_v1/${METHOD}}
RUN_EVAL=${RUN_EVAL:-0}
OFFICIAL_PROTOCOL=${OFFICIAL_PROTOCOL:-1}
EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE:-0.0}
TOP_K=${TOP_K:-8}
MAX_STEPS=${MAX_STEPS:-100}
NUM_TRIALS=${NUM_TRIALS:-1}

E1_CHECKPOINT_PATH=${E1_CHECKPOINT_PATH:-}
E1_REPORT_PATH=${E1_REPORT_PATH:-}
if [[ "${METHOD}" == "transformer" ]]; then
  E1_CHECKPOINT_PATH=${E1_CHECKPOINT_PATH:-${E1_ROOT}/pilot_v2_seed23/transformer/best.pt}
  E1_REPORT_PATH=${E1_REPORT_PATH:-${E1_ROOT}/pilot_v2_seed23/transformer/train_report.json}
elif [[ "${METHOD}" == "lstr" ]]; then
  E1_CHECKPOINT_PATH=${E1_CHECKPOINT_PATH:-${E1_ROOT}/full_seed31/lstr/best.pt}
  E1_REPORT_PATH=${E1_REPORT_PATH:-${E1_ROOT}/full_seed31/lstr/train_report.json}
elif [[ -n "${E1_CHECKPOINT_PATH}" || -n "${E1_REPORT_PATH}" ]]; then
  echo "ERROR: controlled Static must not receive E1 learned artifacts" >&2
  exit 2
fi

if [[ "${OFFICIAL_PROTOCOL}" == "1" ]] && {
  [[ "${EXECUTOR_TEMPERATURE}" != "0.0" ]] ||
  [[ "${TOP_K}" != "8" ]] ||
  [[ "${MAX_STEPS}" != "100" ]] ||
  [[ "${NUM_TRIALS}" != "1" ]];
}; then
  echo "ERROR: official E3 requires temperature 0, Top-8, 100 steps, one trial" >&2
  exit 2
fi
if [[ "${RUN_EVAL}" == "1" && ! -r "${CLSTR_PROVIDER_CREDENTIAL_FILE}" ]]; then
  echo "ERROR: provider credential file is not readable" >&2
  exit 2
fi

python3 - "${PROJECT_ROOT}" "${METHOD}" "${FOUNDATION_CHECKPOINT_PATH}" \
  "${SKILLS_PATH}" "${E1_CHECKPOINT_PATH}" "${E1_REPORT_PATH}" \
  "${MATCHED_UNION_MANIFEST_PATH}" "${TAU2_ROOT}" <<'PY'
import ast
import json
import sys
from pathlib import Path

project = Path(sys.argv[1])
method = sys.argv[2]
foundation, skills, e1_checkpoint, e1_report, manifest, repo = map(Path, sys.argv[3:])
required = [foundation, skills, manifest, repo / "src" / "tau2"]
if method != "static":
    required.extend([e1_checkpoint, e1_report])
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing controlled Tau2 E3 inputs: " + ", ".join(missing))
for relative in (
    "clstr/matched_history_online.py",
    "clstr/vnext_tau2_agent.py",
    "scripts/run_clstr_matched_history_tau2_e3.py",
):
    path = project / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
payload = json.loads(manifest.read_text(encoding="utf-8"))
split = payload["tau2_split_manifest"]
records = split["splits"]["test"]
counts = {domain: 0 for domain in ("airline", "retail", "telecom")}
for record in records:
    counts[str(record).split("/", 1)[0]] += 1
if not split.get("official_test_preserved") or counts != {
    "airline": 20,
    "retail": 40,
    "telecom": 40,
}:
    raise SystemExit("matched manifest does not bind the 20/40/40 official Tau2 split")
print(f"Tau2 controlled E3 readiness: OK ({method})")
PY

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=${RUN_EVAL}; readiness only, no Tau2 simulation was executed."
  exit 0
fi

export PROJECT_ROOT
export CLSTR_PROVIDER_CREDENTIAL_FILE
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_qwen14_executor.sh"
export CLSTR_PROXY_JUMP_HOST=${CLSTR_PROXY_JUMP_HOST:-ln01}
PYTHON_BIN=${TAU2_PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
export PYTHON_BIN
export PYTHONPATH="${PROJECT_ROOT}:${TAU2_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"
if [[ -z "${EXECUTOR_BASE_URL:-}" ]]; then
  start_clstr_qwen14_executor "${OUTPUT_DIR}/qwen3_14b_executor.log"
fi
cleanup() {
  stop_clstr_qwen14_executor
  stop_clstr_provider_proxy_tunnel
}
trap cleanup EXIT
start_clstr_provider_proxy_tunnel
load_clstr_provider_credentials
USER_BASE_URL=${USER_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
USER_API_KEY=${USER_API_KEY:-${OPENAI_API_KEY:-}}
if [[ -z "${USER_API_KEY}" ]]; then
  echo "ERROR: Tau2 official user simulator requires USER_API_KEY or OPENAI_API_KEY" >&2
  exit 2
fi

args=(
  "${PYTHON_BIN}" scripts/run_clstr_matched_history_tau2_e3.py
  --method "${METHOD}"
  --foundation_checkpoint_path "${FOUNDATION_CHECKPOINT_PATH}"
  --skills_path "${SKILLS_PATH}"
  --matched_union_manifest_path "${MATCHED_UNION_MANIFEST_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --executor_model "${EXECUTOR_MODEL}"
  --executor_base_url "${EXECUTOR_BASE_URL}"
  --executor_api_key "${EXECUTOR_API_KEY:-EMPTY}"
  --executor_temperature "${EXECUTOR_TEMPERATURE}"
  --user_model "${USER_MODEL:-gpt-4.1}"
  --user_base_url "${USER_BASE_URL}"
  --num_trials "${NUM_TRIALS}"
  --max_steps "${MAX_STEPS}"
  --max_concurrency "${MAX_CONCURRENCY:-4}"
  --top_k "${TOP_K}"
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-300}"
)
if [[ "${METHOD}" != "static" ]]; then
  args+=(
    --e1_checkpoint_path "${E1_CHECKPOINT_PATH}"
    --e1_report_path "${E1_REPORT_PATH}"
  )
fi
if [[ -n "${NUM_TASKS:-}" ]]; then args+=(--num_tasks "${NUM_TASKS}"); fi
if [[ -n "${DOMAIN:-}" ]]; then args+=(--domain "${DOMAIN}"); fi
"${args[@]}" | tee "${OUTPUT_DIR}/stdout.log"
