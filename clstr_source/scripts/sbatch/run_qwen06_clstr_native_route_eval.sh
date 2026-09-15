#!/bin/bash
#SBATCH --job-name=q06_clstr_native
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=02:00:00
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
  BENCHMARK OUTPUT_DIR
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
  "${TRAINING_SKILLS_PATH}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from clstr.frozen_clstr_route_eval import load_final_chain_manifest
from clstr.memory_utility_records import canonical_digest

chain_path, expected_manifest, expected_chain, corpus_path, expected_corpus, stage0, stage2, stage4, skills = sys.argv[1:]
chain = load_final_chain_manifest(chain_path)
if chain["manifest_sha256"] != expected_manifest or chain["checkpoint_chain_digest"] != expected_chain:
    raise SystemExit("final-chain identity mismatch")
expected_paths = {
    "stage0": Path(stage0).resolve(),
    "stage2": Path(stage2).resolve(),
    "stage4": Path(stage4).resolve(),
}
for role, path in expected_paths.items():
    if Path(chain["checkpoints"][role]["path"]).resolve() != path:
        raise SystemExit(f"{role} checkpoint path mismatch")
if Path(chain["skill_pool"]["path"]).resolve() != Path(skills).resolve():
    raise SystemExit("training skill-pool path mismatch")
corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
recorded = str(corpus.pop("manifest_sha256", ""))
if recorded != expected_corpus or canonical_digest(corpus) != recorded:
    raise SystemExit("corpus manifest identity mismatch")
PY

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/.tmp/slurm"
common=(
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT_PATH}"
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-8}"
  --route_scorer "${ROUTE_SCORER:-unified_memory}"
  --reliability_mode "${RELIABILITY_MODE:-dynamic}"
  --fixed_alpha "${FIXED_ALPHA:-1.0}"
  --feature_update_count_cap "${FEATURE_UPDATE_COUNT_CAP:-1.0}"
  --feature_candidate_count_cap "${FEATURE_CANDIDATE_COUNT_CAP:-1.0}"
  --memory_utility_gate_checkpoint_path "${MEMORY_UTILITY_GATE_CHECKPOINT_PATH:-}"
  --expected_memory_utility_gate_checkpoint_sha256 "${MEMORY_UTILITY_GATE_CHECKPOINT_SHA256:-}"
  --expected_memory_utility_gate_audit_sha256 "${MEMORY_UTILITY_GATE_AUDIT_SHA256:-}"
)

case "${BENCHMARK}" in
  toolbench_g3)
    args=(
      "${PYTHON_BIN}" scripts/run_toolbench_g3_full_clstr_route_eval.py
      "${common[@]}"
      --train_trajectories_path "${TRAIN_TRAJECTORIES_PATH}"
      --eval_trajectories_path "${EVAL_TRAJECTORIES_PATH}"
      --skills_path "${TRAINING_SKILLS_PATH}"
      --static_k "${STATIC_K:-500}"
      --dynamic_extra_k "${DYNAMIC_EXTRA_K:-64}"
      --final_k "${CANDIDATE_COUNT:-64}"
      --stage0_handoff_query_mode checkpoint_state_query
      --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE:-16}"
    )
    if [[ -n "${MAX_TRAIN_ROWS:-}" && "${MAX_TRAIN_ROWS}" != "ALL" ]]; then
      args+=(--max_train_rows "${MAX_TRAIN_ROWS}")
    fi
    ;;
  toolsandbox)
    args=(
      "${PYTHON_BIN}" scripts/run_toolsandbox_full_clstr_route_eval.py
      "${common[@]}"
      --scenarios_root "${SCENARIOS_ROOT}"
      --tools_root "${TOOLS_ROOT}"
      --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE:-16}"
      --candidate_count "${CANDIDATE_COUNT:-64}"
      --model_skill_pool_mode "${MODEL_SKILL_POOL_MODE:-checkpoint_faithful}"
      --toolsandbox_replay_mode "${TOOLSANDBOX_REPLAY_MODE:-corrected_causal}"
    )
    if [[ "${MODEL_SKILL_POOL_MODE:-checkpoint_faithful}" == "checkpoint_faithful" ]]; then
      args+=(--training_skills_path "${TRAINING_SKILLS_PATH}")
    fi
    if [[ -n "${MAX_SCENARIOS:-}" && "${MAX_SCENARIOS}" != "ALL" ]]; then
      args+=(--max_scenarios "${MAX_SCENARIOS}")
    fi
    ;;
  tau2)
    args=(
      "${PYTHON_BIN}" scripts/run_tau2_full_clstr_route_eval.py
      "${common[@]}"
      --data_root "${DATA_ROOT}"
      --training_skills_path "${TRAINING_SKILLS_PATH}"
      --stage0_candidate_batch_size "${STAGE0_CANDIDATE_BATCH_SIZE:-16}"
      --candidate_count "${CANDIDATE_COUNT:-64}"
      --task_split "${TASK_SPLIT:-base}"
    )
    if [[ -n "${MAX_TASKS_PER_DOMAIN:-}" && "${MAX_TASKS_PER_DOMAIN}" != "ALL" ]]; then
      args+=(--max_tasks_per_domain "${MAX_TASKS_PER_DOMAIN}")
    fi
    ;;
  *)
    printf 'ERROR: unsupported native benchmark: %s\n' "${BENCHMARK}" >&2
    exit 2
    ;;
esac
if [[ -n "${MAX_EVAL_ROWS:-}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[qwen06-clstr-native] benchmark=%s scope=%s output=%s\n' \
  "${BENCHMARK}" "${EVAL_SCOPE:-unknown}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-native] chain=%s corpus=%s\n' \
  "${CHECKPOINT_CHAIN_DIGEST}" "${CORPUS_MANIFEST_SHA256}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
