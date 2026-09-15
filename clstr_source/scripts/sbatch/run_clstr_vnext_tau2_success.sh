#!/bin/bash
#SBATCH --job-name=clstr_tau2_success
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
ARTIFACT_RUN_ROOT=${ARTIFACT_RUN_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1}
MATCHED_ROOT=${MATCHED_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/e3039e2/matched_union}
MATCHED_RELEASE_SELECTION_PATH=${MATCHED_RELEASE_SELECTION_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/diagnostics/matched_release_1c94bc2_v1/clstr_vnext_stage2_matched_release.json}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-${ARTIFACT_RUN_ROOT}/stage2_coverage_control_p00_v1/checkpoints/clstr_vnext_stage2-step500.pt}
TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:-${ARTIFACT_RUN_ROOT}/stage0/selected_skills.jsonl}
BENCHMARK_SKILLS_PATH=${BENCHMARK_SKILLS_PATH:-${MATCHED_ROOT}/matched_splits/tau2_skills.jsonl}
MATCHED_UNION_MANIFEST_PATH=${MATCHED_UNION_MANIFEST_PATH:-${MATCHED_ROOT}/matched_union_manifest.json}
TAU2_ROOT=${TAU2_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/official_closed_loop/tau2_test_step500}
RUN_EVAL=${RUN_EVAL:-0}
OFFICIAL_PROTOCOL=${OFFICIAL_PROTOCOL:-1}
CLSTR_ROUTE_MODE=${CLSTR_ROUTE_MODE:-adaptive}
EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE:-0.0}

if [[ "${OFFICIAL_PROTOCOL}" == "1" ]] && {
  [[ "${CLSTR_ROUTE_MODE}" != "adaptive" ]] ||
  [[ "${EXECUTOR_TEMPERATURE}" != "0.0" ]];
}; then
  echo "ERROR: official Tau2 requires adaptive routing and executor temperature 0.0" >&2
  exit 2
fi

python3 - "${PROJECT_ROOT}" "${MATCHED_RELEASE_SELECTION_PATH}" \
  "${STAGE2_CHECKPOINT_PATH}" "${TRAINING_SKILLS_PATH}" \
  "${BENCHMARK_SKILLS_PATH}" "${MATCHED_UNION_MANIFEST_PATH}" \
  "${TAU2_ROOT}" <<'PY'
import ast
import json
import sys
from pathlib import Path

project, release, checkpoint, skills, benchmark_skills, manifest, repo = map(Path, sys.argv[1:])
required = [release, checkpoint, skills, benchmark_skills, manifest, repo / "src" / "tau2"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing Tau2 evaluation inputs: " + ", ".join(missing))
for relative in (
    "clstr/vnext_online_selector.py",
    "clstr/vnext_tau2_agent.py",
    "scripts/run_clstr_vnext_tau2_success.py",
):
    path = project / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
payload = json.loads(manifest.read_text(encoding="utf-8"))
split = payload["tau2_split_manifest"]
if not split.get("official_test_preserved") or len(split["splits"]["test"]) != 100:
    raise SystemExit("matched manifest does not bind the 100-task official Tau2 test split")
print("Tau2 CLSTR closed-loop readiness: OK")
PY

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=${RUN_EVAL}; readiness only, no Tau2 simulation was executed."
  exit 0
fi

export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_qwen14_executor.sh"
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
  "${PYTHON_BIN}" scripts/run_clstr_vnext_tau2_success.py
  --matched_release_selection_path "${MATCHED_RELEASE_SELECTION_PATH}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}"
  --training_skills_path "${TRAINING_SKILLS_PATH}"
  --benchmark_skills_path "${BENCHMARK_SKILLS_PATH}"
  --matched_union_manifest_path "${MATCHED_UNION_MANIFEST_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --executor_model "${EXECUTOR_MODEL}"
  --executor_base_url "${EXECUTOR_BASE_URL}"
  --executor_api_key "${EXECUTOR_API_KEY:-EMPTY}"
  --executor_temperature "${EXECUTOR_TEMPERATURE}"
  --user_model "${USER_MODEL:-gpt-4.1}"
  --user_base_url "${USER_BASE_URL}"
  --num_trials "${NUM_TRIALS:-1}"
  --max_steps "${MAX_STEPS:-100}"
  --max_concurrency "${MAX_CONCURRENCY:-4}"
  --top_k "${TOP_K:-8}"
  --coarse_k "${COARSE_K:-500}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K:-64}"
  --route_mode "${CLSTR_ROUTE_MODE}"
  --device "${DEVICE:-cuda}"
  --seed "${SEED:-300}"
)
if [[ -n "${NUM_TASKS:-}" ]]; then args+=(--num_tasks "${NUM_TASKS}"); fi
if [[ -n "${DOMAIN:-}" ]]; then args+=(--domain "${DOMAIN}"); fi
"${args[@]}" | tee "${OUTPUT_DIR}/stdout.log"
