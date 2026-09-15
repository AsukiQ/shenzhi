#!/bin/bash
#SBATCH --job-name=alf_qwen_gate
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --exclude=d1n41a15g01
set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${REQUESTED_PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

OFFICIAL_REPO=${OFFICIAL_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo}
DATA_DIR=${DATA_DIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/alfworld_eval/qwen_clstr_executor_gate_smoke}
SPLITS=${SPLITS:-valid_seen}
RUN_NAME=${RUN_NAME:-qwen_clstr_executor_gate}
MAX_EPISODES=${MAX_EPISODES:-2}
MAX_STEPS=${MAX_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
ROUTING_INIT_MANIFEST=${ROUTING_INIT_MANIFEST:-outputs/clstr_native_routing_init/manifest.json}
CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-${CHECKPOINT_PATH:-}}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT_PATH:-${STAGE4_CHECKPOINT:-}}
SKILL_ROWS_PATH_OVERRIDE=${TRAINING_SKILLS_PATH:-${SKILL_ROWS_PATH_OVERRIDE:-}}
AUX_DATA_ROOT=${AUX_DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
QWEN_MODEL_NAME_OR_PATH=${QWEN_MODEL_NAME_OR_PATH:-models/Qwen3-14B}
CACHE_DIR=${CACHE_DIR:-.cache/huggingface}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
FALLBACK_STRATEGY=${FALLBACK_STRATEGY:-first_admissible}
SCORING_METHOD=${SCORING_METHOD:-generate}
LIKELIHOOD_BATCH_SIZE=${LIKELIHOOD_BATCH_SIZE:-4}
QWEN_WEIGHT=${QWEN_WEIGHT:-1.0}
CLSTR_WEIGHT=${CLSTR_WEIGHT:-0.25}
HYBRID_PRIOR=${HYBRID_PRIOR:-clstr}
SKILLROUTER_MODEL_NAME_OR_PATH=${SKILLROUTER_MODEL_NAME_OR_PATH:-.cache/hf_models/SkillRouter-Embedding-0.6B}
SKILLROUTER_ADAPTER_CHECKPOINT_PATH=${SKILLROUTER_ADAPTER_CHECKPOINT_PATH:-}
SKILLROUTER_ENCODE_BATCH_SIZE=${SKILLROUTER_ENCODE_BATCH_SIZE:-16}
SKILLROUTER_MAX_LENGTH=${SKILLROUTER_MAX_LENGTH:-2048}
QWEN_SCORE_MODE=${QWEN_SCORE_MODE:-auto}
CLSTR_PRIOR_MODE=${CLSTR_PRIOR_MODE:-unified_memory_concrete_action}
CLSTR_REPLAY_PREFIX_MAX_STEPS=${CLSTR_REPLAY_PREFIX_MAX_STEPS:-6}
RELIABILITY_MODE=${RELIABILITY_MODE:-cmc}
FIXED_ALPHA=${FIXED_ALPHA:-1.0}
SAFE_MEMORY_RESIDUAL_BOUND=${SAFE_MEMORY_RESIDUAL_BOUND:-2.0}
FEATURE_UPDATE_COUNT_CAP=${FEATURE_UPDATE_COUNT_CAP:-16.0}
FEATURE_CANDIDATE_COUNT_CAP=${FEATURE_CANDIDATE_COUNT_CAP:-256.0}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
NO_NORMALIZE_SCORES=${NO_NORMALIZE_SCORES:-0}
INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}
LOOP_GUARD=${LOOP_GUARD:-0}
RUN_QWEN_ONLY=${RUN_QWEN_ONLY:-1}
RUN_HYBRID=${RUN_HYBRID:-1}

required=(
  EVAL_SCOPE
  FINAL_CHAIN_MANIFEST_PATH FINAL_CHAIN_MANIFEST_SHA256 CHECKPOINT_CHAIN_DIGEST
  STAGE0_CHECKPOINT_PATH STAGE2_CHECKPOINT_PATH STAGE4_CHECKPOINT_PATH
  TRAINING_SKILLS_PATH CORPUS_MANIFEST_PATH CORPUS_MANIFEST_SHA256
  ALFWORLD_SKILLS_PATH ALFWORLD_SKILLS_MANIFEST_PATH
  ALFWORLD_SKILLS_MANIFEST_SHA256
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
    training_skills,
    skills_manifest_path,
    expected_skills_manifest,
    benchmark_skills,
) = sys.argv[1:]
chain = load_final_chain_manifest(chain_path)
if (
    chain["manifest_sha256"] != expected_manifest
    or chain["checkpoint_chain_digest"] != expected_chain
):
    raise SystemExit("final-chain identity mismatch")
for role, path in {
    "stage0": stage0,
    "stage2": stage2,
    "stage4": stage4,
}.items():
    if Path(chain["checkpoints"][role]["path"]).resolve() != Path(path).resolve():
        raise SystemExit(f"{role} checkpoint path mismatch")
if Path(chain["skill_pool"]["path"]).resolve() != Path(training_skills).resolve():
    raise SystemExit("training skill-pool path mismatch")
for path, expected, label in (
    (corpus_path, expected_corpus, "ALFWorld split manifest"),
    (skills_manifest_path, expected_skills_manifest, "ALFWorld skill manifest"),
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = str(payload.pop("manifest_sha256", ""))
    if recorded != expected or canonical_digest(payload) != recorded:
        raise SystemExit(f"{label} identity mismatch")
skills_payload = json.loads(Path(skills_manifest_path).read_text(encoding="utf-8"))
skill_rows = [
    json.loads(line)
    for line in Path(benchmark_skills).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if skill_rows != skills_payload.get("canonical_skills"):
    raise SystemExit("ALFWorld benchmark skills do not match manifest")
skills_path = skills_payload.get("skills_path") or skills_payload.get("source_rows_path")
if skills_path and Path(skills_path).resolve() != Path(benchmark_skills).resolve():
    raise SystemExit("ALFWorld benchmark skill path mismatch")
PY

if [[ ! -d "${OFFICIAL_REPO}" ]]; then
  echo "ERROR: missing ALFWorld official repo: ${OFFICIAL_REPO}" >&2
  exit 2
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing ALFWorld data dir: ${DATA_DIR}" >&2
  exit 2
fi

if [[ "${RUN_HYBRID}" = "1" ]]; then
  for path in \
    "${STAGE0_CHECKPOINT_PATH}" \
    "${CHECKPOINT_PATH}" \
    "${STAGE4_CHECKPOINT}" \
    "${TRAINING_SKILLS_PATH}" \
    "${ALFWORLD_SKILLS_PATH}"; do
    if [[ ! -s "${path}" ]]; then
      printf 'ERROR: missing CLSTR final-chain input: %s\n' "${path}" >&2
      exit 2
    fi
  done
fi
if [[ ! -d "${QWEN_MODEL_NAME_OR_PATH}" ]]; then
  printf 'ERROR: missing Qwen3-14B executor: %s\n' "${QWEN_MODEL_NAME_OR_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --official_repo "${OFFICIAL_REPO}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --splits ${SPLITS}
  --run_name "${RUN_NAME}"
  --max_episodes "${MAX_EPISODES}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --routing_init_manifest "${ROUTING_INIT_MANIFEST}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --training_skills_path "${TRAINING_SKILLS_PATH}"
  --benchmark_skill_rows_path "${ALFWORLD_SKILLS_PATH}"
  --aux_data_root "${AUX_DATA_ROOT}"
  --qwen_model_name_or_path "${QWEN_MODEL_NAME_OR_PATH}"
  --cache_dir "${CACHE_DIR}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --fallback_strategy "${FALLBACK_STRATEGY}"
  --scoring_method "${SCORING_METHOD}"
  --likelihood_batch_size "${LIKELIHOOD_BATCH_SIZE}"
  --qwen_weight "${QWEN_WEIGHT}"
  --clstr_weight "${CLSTR_WEIGHT}"
  --hybrid_prior "${HYBRID_PRIOR}"
  --skillrouter_model_name_or_path "${SKILLROUTER_MODEL_NAME_OR_PATH}"
  --skillrouter_encode_batch_size "${SKILLROUTER_ENCODE_BATCH_SIZE}"
  --skillrouter_max_length "${SKILLROUTER_MAX_LENGTH}"
  --qwen_score_mode "${QWEN_SCORE_MODE}"
  --clstr_prior_mode "${CLSTR_PRIOR_MODE}"
  --clstr_replay_prefix_max_steps "${CLSTR_REPLAY_PREFIX_MAX_STEPS}"
  --reliability_mode "${RELIABILITY_MODE}"
  --fixed_alpha "${FIXED_ALPHA}"
  --safe_memory_residual_bound "${SAFE_MEMORY_RESIDUAL_BOUND}"
  --feature_update_count_cap "${FEATURE_UPDATE_COUNT_CAP}"
  --feature_candidate_count_cap "${FEATURE_CANDIDATE_COUNT_CAP}"
)

if [[ -n "${STAGE4_CHECKPOINT}" ]]; then
  ARGS+=(--stage4_checkpoint_path "${STAGE4_CHECKPOINT}")
fi
if [[ -n "${SKILL_ROWS_PATH_OVERRIDE}" ]]; then
  ARGS+=(--skill_rows_path_override "${SKILL_ROWS_PATH_OVERRIDE}")
fi
if [[ -n "${SKILLROUTER_ADAPTER_CHECKPOINT_PATH}" ]]; then
  ARGS+=(--skillrouter_adapter_checkpoint_path "${SKILLROUTER_ADAPTER_CHECKPOINT_PATH}")
fi
if [[ "${LOCAL_FILES_ONLY}" = "1" ]]; then
  ARGS+=(--local_files_only)
fi
if [[ "${NO_NORMALIZE_SCORES}" = "1" ]]; then
  ARGS+=(--no_normalize_scores)
fi
if [[ "${INCLUDE_AVAILABLE_ACTIONS_IN_STATE}" = "1" ]]; then
  ARGS+=(--include_available_actions_in_state)
fi
if [[ "${LOOP_GUARD}" = "1" ]]; then
  ARGS+=(--loop_guard)
fi
if [[ "${RUN_QWEN_ONLY}" = "1" ]]; then
  ARGS+=(--run_qwen_only)
fi
if [[ "${RUN_HYBRID}" = "1" ]]; then
  ARGS+=(--run_hybrid)
fi

"${PYTHON_BIN}" scripts/run_alfworld_qwen_clstr_executor_gate.py "${ARGS[@]}" \
  | tee "${OUTPUT_DIR}/executor_gate_stdout.json"
