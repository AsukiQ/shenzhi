#!/bin/bash
#SBATCH --job-name=q06_clstr_alfworld
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${REQUESTED_PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

required=(
  EVAL_SCOPE
  FINAL_CHAIN_MANIFEST_PATH FINAL_CHAIN_MANIFEST_SHA256 CHECKPOINT_CHAIN_DIGEST
  STAGE0_CHECKPOINT_PATH STAGE2_CHECKPOINT_PATH STAGE4_CHECKPOINT_PATH
  TRAINING_SKILLS_PATH CORPUS_MANIFEST_PATH CORPUS_MANIFEST_SHA256
  ALFWORLD_SKILLS_PATH ALFWORLD_SKILLS_MANIFEST_PATH ALFWORLD_SKILLS_MANIFEST_SHA256
  OFFICIAL_REPO DATA_DIR SPLIT OUTPUT_DIR SAFE_MEMORY_RESIDUAL_BOUND
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is empty: %s\n' "${name}" >&2
    exit 2
  fi
done
if [[ "${EVAL_SCOPE}" == "full" ]]; then
  for name in ACCEPTED_SMOKE_GATE_PATH ACCEPTED_SMOKE_GATE_SHA256; do
    if [[ -z "${!name:-}" ]]; then
      printf 'ERROR: full evaluation requires %s\n' "${name}" >&2
      exit 2
    fi
  done
  "${PYTHON_BIN}" scripts/validate_qwen06_clstr_multibench_smoke_gate.py \
    --smoke_gate_path "${ACCEPTED_SMOKE_GATE_PATH}" \
    --expected_smoke_gate_sha256 "${ACCEPTED_SMOKE_GATE_SHA256}" \
    --expected_checkpoint_chain_digest "${CHECKPOINT_CHAIN_DIGEST}" \
    --expected_final_chain_manifest_sha256 "${FINAL_CHAIN_MANIFEST_SHA256}"
elif [[ "${EVAL_SCOPE}" != "smoke" ]]; then
  printf 'ERROR: EVAL_SCOPE must be smoke or full; got %s\n' "${EVAL_SCOPE}" >&2
  exit 2
fi

"${PYTHON_BIN}" - \
  "${FINAL_CHAIN_MANIFEST_PATH}" "${FINAL_CHAIN_MANIFEST_SHA256}" \
  "${CHECKPOINT_CHAIN_DIGEST}" "${CORPUS_MANIFEST_PATH}" \
  "${CORPUS_MANIFEST_SHA256}" "${STAGE0_CHECKPOINT_PATH}" \
  "${STAGE2_CHECKPOINT_PATH}" "${STAGE4_CHECKPOINT_PATH}" \
  "${TRAINING_SKILLS_PATH}" "${ALFWORLD_SKILLS_MANIFEST_PATH}" \
  "${ALFWORLD_SKILLS_MANIFEST_SHA256}" "${ALFWORLD_SKILLS_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from clstr.frozen_clstr_route_eval import load_final_chain_manifest
from clstr.memory_utility_records import canonical_digest

(
    chain_path,
    expected_manifest,
    expected_chain,
    corpus_path,
    expected_corpus,
    stage0,
    stage2,
    stage4,
    skills,
    action_manifest_path,
    expected_action_manifest,
    action_skills_path,
) = sys.argv[1:]
chain = load_final_chain_manifest(chain_path)
if chain["manifest_sha256"] != expected_manifest or chain["checkpoint_chain_digest"] != expected_chain:
    raise SystemExit("final-chain identity mismatch")
for role, raw in (("stage0", stage0), ("stage2", stage2), ("stage4", stage4)):
    if Path(chain["checkpoints"][role]["path"]).resolve() != Path(raw).resolve():
        raise SystemExit(f"{role} checkpoint path mismatch")
if Path(chain["skill_pool"]["path"]).resolve() != Path(skills).resolve():
    raise SystemExit("training skill-pool path mismatch")
corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
recorded = str(corpus.pop("manifest_sha256", ""))
if recorded != expected_corpus or canonical_digest(corpus) != recorded:
    raise SystemExit("corpus manifest identity mismatch")
action_manifest = json.loads(Path(action_manifest_path).read_text(encoding="utf-8"))
recorded = str(action_manifest.pop("manifest_sha256", ""))
if recorded != expected_action_manifest or canonical_digest(action_manifest) != recorded:
    raise SystemExit("ALFWorld action-skill manifest identity mismatch")
action_skills = [
    json.loads(line)
    for line in Path(action_skills_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if action_skills != action_manifest.get("canonical_skills"):
    raise SystemExit("ALFWorld action-skill file identity mismatch")
PY

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/.tmp/slurm"
args=(
  "${PYTHON_BIN}" scripts/run_alfworld_clstr_eval.py eval
  --official_repo "${OFFICIAL_REPO}"
  --data_dir "${DATA_DIR}"
  --routing_init_manifest "${FINAL_CHAIN_MANIFEST_PATH}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --checkpoint_path "${STAGE2_CHECKPOINT_PATH}"
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT_PATH}"
  --skill_rows_path_override "${TRAINING_SKILLS_PATH}"
  --benchmark_skill_rows_path "${ALFWORLD_SKILLS_PATH}"
  --scorer_mode "${SCORER_MODE:-unified_memory_admissible_action}"
  --memory_protocol "${MEMORY_PROTOCOL:-stateful_post_action_v1}"
  --reliability_mode "${RELIABILITY_MODE:-dynamic}"
  --fixed_alpha "${FIXED_ALPHA:-1.0}"
  --safe_memory_residual_bound "${SAFE_MEMORY_RESIDUAL_BOUND}"
  --feature_update_count_cap "${FEATURE_UPDATE_COUNT_CAP:-1.0}"
  --feature_candidate_count_cap "${FEATURE_CANDIDATE_COUNT_CAP:-1.0}"
  --memory_utility_gate_checkpoint_path "${MEMORY_UTILITY_GATE_CHECKPOINT_PATH:-}"
  --expected_memory_utility_gate_checkpoint_sha256 "${MEMORY_UTILITY_GATE_CHECKPOINT_SHA256:-}"
  --expected_memory_utility_gate_audit_sha256 "${MEMORY_UTILITY_GATE_AUDIT_SHA256:-}"
  --output_dir "${OUTPUT_DIR}"
  --protocol_report_path "${OUTPUT_DIR}/protocol_report.json"
  --env_report_path "${OUTPUT_DIR}/env_report.json"
  --splits "${SPLIT}"
  --run_name "qwen06_clstr_${SPLIT}"
  --max_steps "${MAX_STEPS:-50}"
  --batch_size "${BATCH_SIZE:-1}"
  --controller_mode "${CONTROLLER_MODE:-policy_plus_transition_belief_stop_loop_penalty}"
)
if [[ -n "${MAX_EPISODES:-}" && "${MAX_EPISODES}" != "ALL" ]]; then
  args+=(--max_episodes "${MAX_EPISODES}")
fi

printf '[qwen06-clstr-alfworld] split=%s scope=%s output=%s\n' \
  "${SPLIT}" "${EVAL_SCOPE:-unknown}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-alfworld] chain=%s corpus=%s\n' \
  "${CHECKPOINT_CHAIN_DIGEST}" "${CORPUS_MANIFEST_SHA256}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
