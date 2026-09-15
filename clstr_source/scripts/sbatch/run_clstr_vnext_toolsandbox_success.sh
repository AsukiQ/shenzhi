#!/bin/bash
#SBATCH --job-name=clstr_ts_success
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=04:00:00
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
BENCHMARK_SKILLS_PATH=${BENCHMARK_SKILLS_PATH:-${MATCHED_ROOT}/matched_splits/toolsandbox_skills.jsonl}
MATCHED_UNION_MANIFEST_PATH=${MATCHED_UNION_MANIFEST_PATH:-${MATCHED_ROOT}/matched_union_manifest.json}
TOOL_SANDBOX_ROOT=${TOOL_SANDBOX_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/official_closed_loop/toolsandbox_step500}
RUN_EVAL=${RUN_EVAL:-0}
OFFICIAL_PROTOCOL=${OFFICIAL_PROTOCOL:-1}
CLSTR_ROUTE_MODE=${CLSTR_ROUTE_MODE:-adaptive}
EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE:-0.0}
EXECUTOR_MAX_TOKENS=${EXECUTOR_MAX_TOKENS:-1024}
EXECUTOR_ENABLE_THINKING=${EXECUTOR_ENABLE_THINKING:-0}
EXECUTOR_RANDOM_SEED=${EXECUTOR_RANDOM_SEED:-0}
USER_TEMPERATURE=${USER_TEMPERATURE:-0.0}
USER_MAX_TOKENS=${USER_MAX_TOKENS:-1024}
CLSTR_INTERVENTION_MODE=${CLSTR_INTERVENTION_MODE:-ranked_guidance}
GUIDANCE_TOP_K=${GUIDANCE_TOP_K:-2}

if [[ "${OFFICIAL_PROTOCOL}" == "1" ]] && {
  [[ "${CLSTR_ROUTE_MODE}" != "adaptive" ]] ||
  [[ "${EXECUTOR_TEMPERATURE}" != "0.0" ]] ||
  [[ "${EXECUTOR_MAX_TOKENS}" != "1024" ]] ||
  [[ "${EXECUTOR_ENABLE_THINKING}" != "0" ]] ||
  [[ "${EXECUTOR_RANDOM_SEED}" != "0" ]] ||
  [[ "${USER_TEMPERATURE}" != "0.0" ]] ||
  [[ "${USER_MAX_TOKENS}" != "1024" ]] ||
  [[ "${CLSTR_INTERVENTION_MODE}" != "ranked_guidance" ]];
}; then
  echo "ERROR: official ToolSandbox requires adaptive ranked_guidance, deterministic executor/user sampling, and thinking disabled" >&2
  exit 2
fi

python3 - "${PROJECT_ROOT}" "${MATCHED_RELEASE_SELECTION_PATH}" \
  "${STAGE2_CHECKPOINT_PATH}" "${TRAINING_SKILLS_PATH}" \
  "${BENCHMARK_SKILLS_PATH}" "${MATCHED_UNION_MANIFEST_PATH}" \
  "${TOOL_SANDBOX_ROOT}" <<'PY'
import ast
import json
import sys
from pathlib import Path

project, release, checkpoint, skills, benchmark_skills, manifest, repo = map(Path, sys.argv[1:])
required = [release, checkpoint, skills, benchmark_skills, manifest, repo / "tool_sandbox"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing ToolSandbox evaluation inputs: " + ", ".join(missing))
for relative in (
    "clstr/vnext_online_selector.py",
    "clstr/vnext_toolsandbox_agent.py",
    "scripts/run_clstr_vnext_toolsandbox_success.py",
):
    path = project / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
payload = json.loads(manifest.read_text(encoding="utf-8"))
split = payload["toolsandbox_split_manifest"]
count = sum(
    split["family_to_split"][record["family_id"]] == "test"
    for record in split["scenario_records"].values()
)
if count != 21:
    raise SystemExit(f"expected 21 grouped ToolSandbox test scenarios, observed {count}")
print("ToolSandbox CLSTR closed-loop readiness: OK")
PY

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=${RUN_EVAL}; readiness only, no official scenario was executed."
  exit 0
fi

export PROJECT_ROOT
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_qwen14_executor.sh"
PYTHON_BIN=${TOOLSANDBOX_PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
export PYTHON_BIN
export PYTHONPATH="${PROJECT_ROOT}:${TOOL_SANDBOX_ROOT}:${PYTHONPATH:-}"
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
  echo "ERROR: ToolSandbox official user simulator requires USER_API_KEY or OPENAI_API_KEY" >&2
  exit 2
fi
# ToolSandbox's official OpenAI user constructor validates this conventional
# variable before the runner replaces its client with the explicit provider.
export OPENAI_API_KEY="${USER_API_KEY}"
args=(
  "${PYTHON_BIN}" scripts/run_clstr_vnext_toolsandbox_success.py
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
  --executor_max_tokens "${EXECUTOR_MAX_TOKENS}"
  --executor_random_seed "${EXECUTOR_RANDOM_SEED}"
  --intervention_mode "${CLSTR_INTERVENTION_MODE}"
  --guidance_top_k "${GUIDANCE_TOP_K}"
  --user "${TOOLSANDBOX_USER:-GPT_4_o_2024_05_13}"
  --user_base_url "${USER_BASE_URL}"
  --user_temperature "${USER_TEMPERATURE}"
  --user_max_tokens "${USER_MAX_TOKENS}"
  --top_k "${TOP_K:-8}"
  --coarse_k "${COARSE_K:-500}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K:-64}"
  --route_mode "${CLSTR_ROUTE_MODE}"
  --device "${DEVICE:-cuda}"
)
if [[ "${EXECUTOR_ENABLE_THINKING}" == "1" ]]; then
  args+=(--executor_enable_thinking)
else
  args+=(--no-executor_enable_thinking)
fi
if [[ -n "${SCENARIO:-}" && -n "${SCENARIOS:-}" ]]; then
  echo "ERROR: set only one of SCENARIO or comma-separated SCENARIOS" >&2
  exit 2
fi
if [[ -n "${SCENARIOS:-}" ]]; then
  IFS=',' read -r -a scenario_names <<<"${SCENARIOS}"
  for scenario_name in "${scenario_names[@]}"; do
    [[ -n "${scenario_name}" ]] && args+=(--scenario "${scenario_name}")
  done
elif [[ -n "${SCENARIO:-}" ]]; then
  args+=(--scenario "${SCENARIO}")
fi
"${args[@]}" | tee "${OUTPUT_DIR}/stdout.log"
