#!/bin/bash

# Submit only the native Stage-2 chain while reusing verified, read-only Stage0
# and candidate-compressor parents.  This launcher deliberately does not run
# precompute, Stage0, or candidate-compressor jobs.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be a new Stage2-only run directory}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:?STAGE0_CHECKPOINT_PATH is required}
CANDIDATE_CHECKPOINT_PATH=${CANDIDATE_CHECKPOINT_PATH:?CANDIDATE_CHECKPOINT_PATH is required}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/models/Qwen3-Embedding-0.6B}
STAGE2_SEGMENTS=${STAGE2_SEGMENTS:-"2000 4000 6000 8000 10000"}
CURRICULUM_TOTAL_STEPS=${CURRICULUM_TOTAL_STEPS:-10000}
ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION:-1}
SYNCHRONIZATION_PAIR_DIM=${SYNCHRONIZATION_PAIR_DIM:-96}
SYNCHRONIZATION_TRACE_LENGTH=${SYNCHRONIZATION_TRACE_LENGTH:-8}
SYNCHRONIZATION_SCALE_INITIAL=${SYNCHRONIZATION_SCALE_INITIAL:-0.05}

RUN_ROOT=$(realpath -m "${RUN_ROOT}")
STAGE0_CHECKPOINT_PATH=$(realpath "${STAGE0_CHECKPOINT_PATH}")
CANDIDATE_CHECKPOINT_PATH=$(realpath "${CANDIDATE_CHECKPOINT_PATH}")
STAGE0_OUTPUT_DIR=$(realpath -m "${STAGE0_OUTPUT_DIR:-$(dirname "$(dirname "${STAGE0_CHECKPOINT_PATH}")")}")
CANDIDATE_OUTPUT_DIR=$(realpath -m "${CANDIDATE_OUTPUT_DIR:-$(dirname "$(dirname "${CANDIDATE_CHECKPOINT_PATH}")")}")
STAGE0_SKILLS_PATH=$(realpath -m "${STAGE0_SKILLS_PATH:-${STAGE0_OUTPUT_DIR}/selected_skills.jsonl}")

if [[ -e "${RUN_ROOT}" ]]; then
  printf 'ERROR: Stage2-only run root already exists: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
for path in "${STAGE0_CHECKPOINT_PATH}" "${CANDIDATE_CHECKPOINT_PATH}" \
  "${STAGE0_SKILLS_PATH}" "${STAGE0_OUTPUT_DIR}/train_report.json" \
  "${STAGE0_OUTPUT_DIR}/stage0_selection.json" \
  "${CANDIDATE_OUTPUT_DIR}/train_report.json" \
  "${CANDIDATE_OUTPUT_DIR}/compressor_selection.json"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: required existing-parent artifact is missing: %s\n' "${path}" >&2
    exit 2
  fi
done

readarray -t PARENT_DIGESTS < <("${PYTHON_BIN:-python3}" - \
  "${STAGE0_CHECKPOINT_PATH}" "${CANDIDATE_CHECKPOINT_PATH}" \
  "${STAGE0_OUTPUT_DIR}" "${CANDIDATE_OUTPUT_DIR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

stage0_checkpoint = Path(sys.argv[1]).resolve()
candidate_checkpoint = Path(sys.argv[2]).resolve()
stage0_root = Path(sys.argv[3]).resolve()
candidate_root = Path(sys.argv[4]).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

stage0_report = json.loads((stage0_root / "train_report.json").read_text())
stage0_selection = json.loads((stage0_root / "stage0_selection.json").read_text())
if (
    stage0_report.get("status") != "ok"
    or stage0_report.get("stage") != "clstr_vnext_stage0"
    or stage0_selection.get("status") != "ok"
    or stage0_report.get("selection") != stage0_selection
):
    raise SystemExit("Stage2-only requires quality-gated Stage0 parents")
selected_stage0 = Path(str(stage0_selection.get("selected_checkpoint_path") or "")).resolve()
if selected_stage0 != stage0_checkpoint:
    raise SystemExit("STAGE0_CHECKPOINT_PATH is not the selected Stage0 checkpoint")
reported_stage0 = Path(str(stage0_report.get("checkpoint_path") or "")).resolve()
if reported_stage0 != stage0_checkpoint:
    raise SystemExit("Stage0 train report and selection disagree")

candidate_report = json.loads((candidate_root / "train_report.json").read_text())
candidate_selection = json.loads((candidate_root / "compressor_selection.json").read_text())
if (
    candidate_report.get("status") != "ok"
    or candidate_report.get("stage") != "clstr_vnext_candidate_compressor"
    or candidate_selection.get("status") != "ok"
    or candidate_report.get("selection") != candidate_selection
):
    raise SystemExit("Stage2-only requires a quality-gated candidate parent")
selected_candidate = Path(str(candidate_selection.get("selected_checkpoint_path") or "")).resolve()
if selected_candidate != candidate_checkpoint:
    raise SystemExit("CANDIDATE_CHECKPOINT_PATH is not the selected candidate checkpoint")
stage0_sha = sha256(stage0_checkpoint)
candidate_sha = sha256(candidate_checkpoint)
if candidate_sha != str(candidate_selection.get("selected_checkpoint_sha256") or ""):
    raise SystemExit("selected candidate checkpoint digest changed")
if stage0_sha != str(candidate_selection.get("parent_stage0_checkpoint_sha256") or ""):
    raise SystemExit("candidate parent does not descend from the requested Stage0 checkpoint")
print(stage0_sha)
print(candidate_sha)
PY
)
STAGE0_CHECKPOINT_SHA256=${PARENT_DIGESTS[0]}
CANDIDATE_CHECKPOINT_SHA256=${PARENT_DIGESTS[1]}

mkdir -p "${RUN_ROOT}/slurm"
CACHE_ARGS="CACHE_ROOT=${RUN_ROOT}/frozen_qwen_cache,ALLOW_EXTERNAL_CACHE_ROOT=0,FROZEN_CACHE_READ_ONLY=0"
if [[ -n "${FROZEN_CACHE_ROOT:-}" ]]; then
  FROZEN_CACHE_ROOT=$(realpath "${FROZEN_CACHE_ROOT}")
  [[ -d "${FROZEN_CACHE_ROOT}" ]] || {
    printf 'ERROR: FROZEN_CACHE_ROOT is not a directory: %s\n' "${FROZEN_CACHE_ROOT}" >&2
    exit 2
  }
  CACHE_ARGS="CACHE_ROOT=${FROZEN_CACHE_ROOT},ALLOW_EXTERNAL_CACHE_ROOT=1,FROZEN_CACHE_READ_ONLY=1"
fi

common_export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH},RUN_ROOT=${RUN_ROOT},MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH},STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR},CANDIDATE_OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR},STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH},CANDIDATE_CHECKPOINT_PATH=${CANDIDATE_CHECKPOINT_PATH},EXPECTED_STAGE0_CHECKPOINT_SHA256=${STAGE0_CHECKPOINT_SHA256},EXPECTED_CANDIDATE_CHECKPOINT_SHA256=${CANDIDATE_CHECKPOINT_SHA256},STAGE0_SKILLS_PATH=${STAGE0_SKILLS_PATH},ALLOW_EXTERNAL_PARENT_OUTPUTS=1,ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION},SYNCHRONIZATION_PAIR_DIM=${SYNCHRONIZATION_PAIR_DIM},SYNCHRONIZATION_TRACE_LENGTH=${SYNCHRONIZATION_TRACE_LENGTH},SYNCHRONIZATION_SCALE_INITIAL=${SYNCHRONIZATION_SCALE_INITIAL},FINALIZE_FULL_CHAIN=0,${CACHE_ARGS}"

dependency=""
for segment_end in ${STAGE2_SEGMENTS}; do
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  else
    dependency_args=()
  fi
  dependency=$(sbatch --parsable "${dependency_args[@]}" \
    --export="${common_export},SEGMENT_END=${segment_end},CURRICULUM_TOTAL_STEPS=${CURRICULUM_TOTAL_STEPS}" \
    --output="${RUN_ROOT}/slurm/stage2-${segment_end}-%j.out" \
    "${PROJECT_ROOT}/scripts/sbatch/run_clstr_vnext_full_stage2_segment.sh")
done

printf '%s\n' "${dependency}"
