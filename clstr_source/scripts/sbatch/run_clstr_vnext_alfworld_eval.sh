#!/bin/bash
#SBATCH --job-name=clstr_vnext_alf
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --exclude=d1n41a15g01
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/clstr" ]]; then
    PROJECT_ROOT=${SLURM_SUBMIT_DIR}
  else
    PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  fi
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
cd "${PROJECT_ROOT}"

EVAL_SCOPE=${EVAL_SCOPE:-smoke}
SPLIT=${SPLIT:-valid_seen}
MATCHED_RELEASE_SELECTION_PATH=${MATCHED_RELEASE_SELECTION_PATH:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/diagnostics/matched_release_1c94bc2_v1/clstr_vnext_stage2_matched_release.json}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage2_coverage_control_p00_v1/checkpoints/clstr_vnext_stage2-step500.pt}
TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/full_v1/stage0/selected_skills.jsonl}
OFFICIAL_REPO=${OFFICIAL_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo}
DATA_DIR=${DATA_DIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_data}
QWEN_MODEL_NAME_OR_PATH=${QWEN_MODEL_NAME_OR_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/models/Qwen3-14B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-performance-runs/5ec4e11/alfworld_current_release}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${EVAL_SCOPE}_${SLURM_JOB_ID:-manual}}
SMOKE_METRICS_PATH=${SMOKE_METRICS_PATH:-}
QWEN_WEIGHT=${QWEN_WEIGHT:-1.0}
CLSTR_WEIGHT=${CLSTR_WEIGHT:-0.25}
CLSTR_ROUTE_MODE=${CLSTR_ROUTE_MODE:-adaptive}
CLSTR_EXECUTOR_INTERFACE=${CLSTR_EXECUTOR_INTERFACE:-exact_prior}
SCORING_METHOD=${SCORING_METHOD:-generate}
LIKELIHOOD_BATCH_SIZE=${LIKELIHOOD_BATCH_SIZE:-16}
SKILL_GUIDANCE_TOP_K=${SKILL_GUIDANCE_TOP_K:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
LOOP_GUARD=${LOOP_GUARD:-1}
OFFICIAL_PROTOCOL=${OFFICIAL_PROTOCOL:-1}
RUN_EVAL=${RUN_EVAL:-0}

if [[ "${OFFICIAL_PROTOCOL}" == "1" ]] && {
  [[ "${CLSTR_EXECUTOR_INTERFACE}" != "exact_prior" ]] ||
  [[ "${SCORING_METHOD}" != "generate" ]] ||
  [[ "${CLSTR_ROUTE_MODE}" != "adaptive" ]] ||
  [[ "${QWEN_WEIGHT}" != "1.0" ]] ||
  [[ "${CLSTR_WEIGHT}" != "0.25" ]] ||
  [[ "${LOOP_GUARD}" != "1" ]];
}; then
  echo "ERROR: official ALFWorld requires adaptive exact_prior generate, weights 1.0/0.25, and loop guard" >&2
  exit 2
fi

if [[ "${SPLIT}" != "valid_seen" && "${SPLIT}" != "valid_unseen" ]]; then
  printf 'ERROR: SPLIT must be valid_seen or valid_unseen; got %s\n' "${SPLIT}" >&2
  exit 2
fi
if [[ "${CLSTR_ROUTE_MODE}" != "adaptive" && "${CLSTR_ROUTE_MODE}" != "static" && "${CLSTR_ROUTE_MODE}" != "dynamic" ]]; then
  printf 'ERROR: CLSTR_ROUTE_MODE must be adaptive, static, or dynamic; got %s\n' "${CLSTR_ROUTE_MODE}" >&2
  exit 2
fi
if [[ "${CLSTR_EXECUTOR_INTERFACE}" != "abstract_grounded" && "${CLSTR_EXECUTOR_INTERFACE}" != "guided_exact_prior" && "${CLSTR_EXECUTOR_INTERFACE}" != "guided_static_exact_prior" && "${CLSTR_EXECUTOR_INTERFACE}" != "legacy_guided_exact_prior" && "${CLSTR_EXECUTOR_INTERFACE}" != "skill_prompt" && "${CLSTR_EXECUTOR_INTERFACE}" != "exact_prior" ]]; then
  printf 'ERROR: unsupported CLSTR_EXECUTOR_INTERFACE: %s\n' "${CLSTR_EXECUTOR_INTERFACE}" >&2
  exit 2
fi
if [[ "${SCORING_METHOD}" != "generate" && "${SCORING_METHOD}" != "likelihood" ]]; then
  printf 'ERROR: SCORING_METHOD must be generate or likelihood; got %s\n' "${SCORING_METHOD}" >&2
  exit 2
fi
if [[ "${LOOP_GUARD}" != "0" && "${LOOP_GUARD}" != "1" ]]; then
  echo "ERROR: LOOP_GUARD must be 0 or 1" >&2
  exit 2
fi

for path in \
  "${MATCHED_RELEASE_SELECTION_PATH}" \
  "${STAGE2_CHECKPOINT_PATH}" \
  "${TRAINING_SKILLS_PATH}"; do
  if [[ ! -s "${path}" ]]; then
    printf 'ERROR: required release artifact is missing: %s\n' "${path}" >&2
    exit 2
  fi
done
for path in "${OFFICIAL_REPO}" "${DATA_DIR}" "${QWEN_MODEL_NAME_OR_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    printf 'ERROR: required directory is missing: %s\n' "${path}" >&2
    exit 2
  fi
done

if [[ "${RUN_EVAL}" != "1" ]]; then
  python3 - "${PROJECT_ROOT}" <<'PY'
import ast
import sys
from pathlib import Path

project = Path(sys.argv[1])
for relative in (
    "clstr/vnext_online_selector.py",
    "clstr/vnext_alfworld.py",
    "clstr/alfworld_qwen_clstr_gate.py",
    "scripts/run_clstr_vnext_alfworld_executor.py",
):
    path = project / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("ALFWorld CLSTR executor readiness: OK")
PY
  echo "RUN_EVAL=${RUN_EVAL}; readiness only, no ALFWorld episode was executed."
  exit 0
fi

if [[ "${EVAL_SCOPE}" == "smoke" ]]; then
  MAX_EPISODES=${MAX_EPISODES:-2}
  MAX_STEPS=${MAX_STEPS:-20}
elif [[ "${EVAL_SCOPE}" == "full" ]]; then
  if [[ "${SPLIT}" == "valid_seen" ]]; then
    MAX_EPISODES=${MAX_EPISODES:-140}
  else
    MAX_EPISODES=${MAX_EPISODES:-134}
  fi
  MAX_STEPS=${MAX_STEPS:-50}
  if [[ ! -s "${SMOKE_METRICS_PATH}" ]]; then
    echo "ERROR: full evaluation requires SMOKE_METRICS_PATH" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - "${SMOKE_METRICS_PATH}" "${STAGE2_CHECKPOINT_PATH}" "${SPLIT}" "${CLSTR_ROUTE_MODE}" "${CLSTR_EXECUTOR_INTERFACE}" "${SKILL_GUIDANCE_TOP_K}" "${SCORING_METHOD}" "${QWEN_WEIGHT}" "${CLSTR_WEIGHT}" "${LOOP_GUARD}" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics.get("status") != "ok":
    raise SystemExit("accepted ALFWorld smoke is not status=ok")
if Path(metrics.get("stage2_checkpoint_path", "")).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("ALFWorld smoke checkpoint differs from full checkpoint")
if str(metrics.get("split") or "") != sys.argv[3]:
    raise SystemExit("ALFWorld smoke split differs from full split")
if str(metrics.get("clstr_route_mode") or "") != sys.argv[4]:
    raise SystemExit("ALFWorld smoke route mode differs from full route mode")
if str(metrics.get("executor_interface") or "") != sys.argv[5]:
    raise SystemExit("ALFWorld smoke executor interface differs from full interface")
if sys.argv[5] in {"abstract_grounded", "guided_exact_prior", "guided_static_exact_prior", "legacy_guided_exact_prior", "skill_prompt"} and int(metrics.get("skill_guidance_top_k") or 0) != int(sys.argv[6]):
    raise SystemExit("ALFWorld smoke skill-guidance top-k differs from full top-k")
if str(metrics.get("scoring_method") or "") != sys.argv[7]:
    raise SystemExit("ALFWorld smoke scoring method differs from full scoring method")
if bool(metrics.get("loop_guard")) != (sys.argv[10] == "1"):
    raise SystemExit("ALFWorld smoke loop-guard protocol differs from full protocol")
if sys.argv[5] in {"abstract_grounded", "guided_exact_prior", "guided_static_exact_prior", "legacy_guided_exact_prior", "exact_prior"}:
    if float(metrics.get("qwen_weight")) != float(sys.argv[8]) or float(metrics.get("clstr_weight")) != float(sys.argv[9]):
        raise SystemExit("ALFWorld smoke fusion weights differ from full weights")
if sys.argv[5] in {"exact_prior", "legacy_guided_exact_prior"}:
    if str(metrics.get("memory_update_skill_mode") or "") != "exact_action":
        raise SystemExit("ALFWorld exact-prior smoke did not use exact-action memory updates")
    if str(metrics.get("qwen_score_mode") or "") != "proposal_bonus":
        raise SystemExit("ALFWorld exact-prior smoke did not use proposal-bonus Qwen scores")
    if str(metrics.get("score_normalization") or "") != "qwen:proposal_bonus;clstr:row_zscore":
        raise SystemExit("ALFWorld exact-prior smoke score normalization differs")
    candidate_schema = str(metrics.get("candidate_skill_schema") or "")
    accepted_schemas = {"alfworld_exact_admissible_action_v1"}
    if sys.argv[5] == "legacy_guided_exact_prior":
        # Pilot 120648 predates the top-level dual-surface provenance fix. Its
        # per-step trace already records the exact-action schema, while the
        # summary incorrectly reported only the abstract guidance surface.
        accepted_schemas.add("alfworld_mapped_abstract_skill_v1")
    if candidate_schema not in accepted_schemas:
        raise SystemExit("ALFWorld exact-prior smoke candidate schema differs")
    if not bool(metrics.get("runtime_pseudo_skills_allowed", False)):
        raise SystemExit("ALFWorld exact-prior smoke did not expose exact legal-action skills")
if sys.argv[5] in {"abstract_grounded", "guided_exact_prior", "guided_static_exact_prior"}:
    if bool(metrics.get("runtime_pseudo_skills_allowed", True)):
        raise SystemExit("ALFWorld grounded smoke allowed runtime pseudo skills")
    if int(metrics.get("runtime_appended_skill_count") or 0) != 0:
        raise SystemExit("ALFWorld grounded smoke appended runtime pseudo skills")
if sys.argv[5] in {"guided_exact_prior", "guided_static_exact_prior"}:
    if not bool(metrics.get("uses_clstr_score_fusion", False)):
        raise SystemExit("ALFWorld guided exact-prior smoke omitted score fusion")
    if not bool(metrics.get("uses_clstr_skill_guidance", False)):
        raise SystemExit("ALFWorld guided exact-prior smoke omitted skill guidance")
    if not bool(metrics.get("guidance_conditions_qwen", False)):
        raise SystemExit("ALFWorld guided exact-prior smoke did not condition Qwen")
if sys.argv[5] == "guided_static_exact_prior" and str(metrics.get("grounding_expert_mode") or "") != "static":
    raise SystemExit("ALFWorld guided static exact-prior smoke did not force static grounding")
if sys.argv[5] == "legacy_guided_exact_prior":
    if not bool(metrics.get("uses_clstr_score_fusion", False)):
        raise SystemExit("ALFWorld legacy guided exact-prior smoke omitted score fusion")
    if not bool(metrics.get("uses_clstr_skill_guidance", False)):
        raise SystemExit("ALFWorld legacy guided exact-prior smoke omitted skill guidance")
    if not bool(metrics.get("guidance_conditions_qwen", False)):
        raise SystemExit("ALFWorld legacy guided exact-prior smoke did not condition Qwen")
trace = metrics.get("trace_summary") or {}
if int(trace.get("causal_update_count") or 0) <= 0:
    raise SystemExit("ALFWorld smoke did not execute recurrent updates")
if float(trace.get("post_action_result_correction_step_rate") or 0.0) <= 0.0:
    raise SystemExit("ALFWorld smoke did not prove post-action result correction")
if sys.argv[5] in {"abstract_grounded", "guided_exact_prior", "guided_static_exact_prior"} and int(trace.get("maximum_runtime_appended_skill_count") or 0) != 0:
    raise SystemExit("ALFWorld grounded trace contains runtime pseudo skills")
PY
else
  printf 'ERROR: EVAL_SCOPE must be smoke or full; got %s\n' "${EVAL_SCOPE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

LOOP_GUARD_FLAG=--loop_guard
if [[ "${LOOP_GUARD}" == "0" ]]; then
  LOOP_GUARD_FLAG=--no-loop_guard
fi

"${PYTHON_BIN}" scripts/run_clstr_vnext_alfworld_executor.py \
  --matched_release_selection_path "${MATCHED_RELEASE_SELECTION_PATH}" \
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}" \
  --training_skills_path "${TRAINING_SKILLS_PATH}" \
  --official_repo "${OFFICIAL_REPO}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --run_name "clstr_vnext_matched_step500_qwen14b_${CLSTR_EXECUTOR_INTERFACE}_${SCORING_METHOD}_${CLSTR_ROUTE_MODE}_${SPLIT}_${EVAL_SCOPE}" \
  --qwen_model_name_or_path "${QWEN_MODEL_NAME_OR_PATH}" \
  --device cuda \
  --torch_dtype bfloat16 \
  --scoring_method "${SCORING_METHOD}" \
  --max_new_tokens 16 \
  --likelihood_batch_size "${LIKELIHOOD_BATCH_SIZE}" \
  --max_episodes "${MAX_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --qwen_weight "${QWEN_WEIGHT}" \
  --clstr_weight "${CLSTR_WEIGHT}" \
  --executor_interface "${CLSTR_EXECUTOR_INTERFACE}" \
  --skill_guidance_top_k "${SKILL_GUIDANCE_TOP_K}" \
  --clstr_route_mode "${CLSTR_ROUTE_MODE}" \
  "${LOOP_GUARD_FLAG}" \
  --local_files_only \
  | tee "${OUTPUT_DIR}/stdout.json"
