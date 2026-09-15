#!/bin/bash
#SBATCH --job-name=clstr_vnext_s2_full
# Stage2 segmented resume compares held-out metrics at strict numerical
# tolerance.  Keep every segment on one GPU architecture so BF16 kernels do
# not create a false run-contract mismatch across A800/H100/H200.  A800 is
# the canonical production architecture because it has materially better
# queue availability on this cluster and already supports the BF16 recipe.
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:?DATA_MANIFEST_PATH is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
SEGMENT_END=${SEGMENT_END:?SEGMENT_END is required}
CURRICULUM_TOTAL_STEPS=${CURRICULUM_TOTAL_STEPS:-10000}
LAMBDA_SAFETY=${LAMBDA_SAFETY:-0.05}
LAMBDA_RAW_ROUTE=${LAMBDA_RAW_ROUTE:-0.5}
LAMBDA_ROUTE_TOPK=${LAMBDA_ROUTE_TOPK:-0.0}
ROUTE_TOPK=${ROUTE_TOPK:-5}
ROUTE_TOPK_MARGIN=${ROUTE_TOPK_MARGIN:-0.05}
ROUTE_TOPK_RECOVERABLE_WEIGHT=${ROUTE_TOPK_RECOVERABLE_WEIGHT:-4.0}
ROUTE_TOPK_LISTWISE_WEIGHT=${ROUTE_TOPK_LISTWISE_WEIGHT:-0.05}
LAMBDA_MIXTURE=${LAMBDA_MIXTURE:-0.2}
MIXTURE_UTILITY_SCALE=${MIXTURE_UTILITY_SCALE:-0.5}
LAMBDA_HISTORY=${LAMBDA_HISTORY:-0.2}
NO_REGRET_TOLERANCE=${NO_REGRET_TOLERANCE:-0.05}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
LAMBDA_RECALL=${LAMBDA_RECALL:-0.2}
LAMBDA_COMPRESSION=${LAMBDA_COMPRESSION:-0.2}
TEACHER_RETENTION_START=${TEACHER_RETENTION_START:-1.0}
TEACHER_RETENTION_END=${TEACHER_RETENTION_END:-0.0}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-500}
MAX_ORDINARY_DEV_ROWS=${MAX_ORDINARY_DEV_ROWS:-1024}
FAMILY_FIRST_HORIZON_SAMPLING=${FAMILY_FIRST_HORIZON_SAMPLING:-0}
ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION:-0}
SYNCHRONIZATION_PAIR_DIM=${SYNCHRONIZATION_PAIR_DIM:-96}
SYNCHRONIZATION_TRACE_LENGTH=${SYNCHRONIZATION_TRACE_LENGTH:-8}
SYNCHRONIZATION_SCALE_INITIAL=${SYNCHRONIZATION_SCALE_INITIAL:-0.05}
ALLOW_EXTERNAL_PARENT_OUTPUTS=${ALLOW_EXTERNAL_PARENT_OUTPUTS:-0}
ALLOW_EXTERNAL_CACHE_ROOT=${ALLOW_EXTERNAL_CACHE_ROOT:-0}
FROZEN_CACHE_READ_ONLY=${FROZEN_CACHE_READ_ONLY:-0}
family_horizon_args=()
case "${FAMILY_FIRST_HORIZON_SAMPLING}" in
  0) ;;
  1) family_horizon_args=(--family_first_horizon_sampling) ;;
  *)
    printf 'ERROR: FAMILY_FIRST_HORIZON_SAMPLING must be 0 or 1\n' >&2
    exit 2
    ;;
esac
native_synchronization_args=()
case "${ENABLE_NATIVE_SYNCHRONIZATION}" in
  0) ;;
  1)
    native_synchronization_args=(
      --enable_native_synchronization
      --synchronization_pair_dim "${SYNCHRONIZATION_PAIR_DIM}"
      --synchronization_trace_length "${SYNCHRONIZATION_TRACE_LENGTH}"
      --synchronization_scale_initial "${SYNCHRONIZATION_SCALE_INITIAL}"
    )
    ;;
  *)
    printf 'ERROR: ENABLE_NATIVE_SYNCHRONIZATION must be 0 or 1\n' >&2
    exit 2
    ;;
esac
cache_mode_args=()
if [[ "${FROZEN_CACHE_READ_ONLY}" == "1" ]]; then
  cache_mode_args=(--frozen_cache_read_only)
fi
for binary_flag in ALLOW_EXTERNAL_PARENT_OUTPUTS ALLOW_EXTERNAL_CACHE_ROOT FROZEN_CACHE_READ_ONLY; do
  case "${!binary_flag}" in
    0|1) ;;
    *)
      printf 'ERROR: %s must be 0 or 1\n' "${binary_flag}" >&2
      exit 2
      ;;
  esac
done
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
CACHE_ROOT=$(realpath -m "${CACHE_ROOT:-${RUN_ROOT}/frozen_qwen_cache}")
if [[ "${ALLOW_EXTERNAL_CACHE_ROOT}" == "0" ]]; then
  case "${CACHE_ROOT}" in
    "${RUN_ROOT}"/*) ;;
    *)
      printf 'ERROR: writable CACHE_ROOT must remain inside RUN_ROOT: cache=%s run=%s\n' \
        "${CACHE_ROOT}" "${RUN_ROOT}" >&2
      exit 2
      ;;
  esac
elif [[ ! -d "${CACHE_ROOT}" ]]; then
  printf 'ERROR: external CACHE_ROOT does not exist: %s\n' "${CACHE_ROOT}" >&2
  exit 2
fi
STAGE0_OUTPUT_DIR=$(realpath -m "${STAGE0_OUTPUT_DIR:-${RUN_ROOT}/stage0}")
CANDIDATE_OUTPUT_DIR=$(realpath -m "${CANDIDATE_OUTPUT_DIR:-${RUN_ROOT}/candidate_compressor}")
if [[ "${ALLOW_EXTERNAL_PARENT_OUTPUTS}" == "0" ]]; then
  case "${STAGE0_OUTPUT_DIR}" in
    "${RUN_ROOT}"/*) ;;
    *)
      printf 'ERROR: Stage0 output must remain inside RUN_ROOT: stage0=%s run=%s\n' \
        "${STAGE0_OUTPUT_DIR}" "${RUN_ROOT}" >&2
      exit 2
      ;;
  esac
  case "${CANDIDATE_OUTPUT_DIR}" in
    "${RUN_ROOT}"/*) ;;
    *)
      printf 'ERROR: static-reranker output must remain inside RUN_ROOT: candidate=%s run=%s\n' \
        "${CANDIDATE_OUTPUT_DIR}" "${RUN_ROOT}" >&2
      exit 2
      ;;
  esac
else
  for parent_dir in "${STAGE0_OUTPUT_DIR}" "${CANDIDATE_OUTPUT_DIR}"; do
    if [[ ! -d "${parent_dir}" ]]; then
      printf 'ERROR: external parent output directory is missing: %s\n' "${parent_dir}" >&2
      exit 2
    fi
  done
fi
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-${RUN_ROOT}/stage2}")
case "${OUTPUT_DIR}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: writable OUTPUT_DIR must remain inside RUN_ROOT: output=%s run=%s\n' \
      "${OUTPUT_DIR}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac
FINALIZE_FULL_CHAIN=${FINALIZE_FULL_CHAIN:-1}
if [[ "${FINALIZE_FULL_CHAIN}" != "0" && "${FINALIZE_FULL_CHAIN}" != "1" ]]; then
  printf 'ERROR: FINALIZE_FULL_CHAIN must be 0 or 1\n' >&2
  exit 2
fi
REQUESTED_STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-}
REQUESTED_STAGE0_SKILLS_PATH=${STAGE0_SKILLS_PATH:-}
REQUESTED_CANDIDATE_CHECKPOINT_PATH=${CANDIDATE_CHECKPOINT_PATH:-}
EXPECTED_STAGE0_CHECKPOINT_SHA256=${EXPECTED_STAGE0_CHECKPOINT_SHA256:-}
EXPECTED_CANDIDATE_CHECKPOINT_SHA256=${EXPECTED_CANDIDATE_CHECKPOINT_SHA256:-}
eval "$("${PYTHON_BIN}" scripts/resolve_clstr_vnext_full_inputs.py "${DATA_MANIFEST_PATH}" --shell)"

readarray -t STAGE0_SELECTED < <("${PYTHON_BIN}" - "${STAGE0_OUTPUT_DIR}" \
  "${REQUESTED_STAGE0_CHECKPOINT_PATH}" "${REQUESTED_STAGE0_SKILLS_PATH}" \
  "${EXPECTED_STAGE0_CHECKPOINT_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
stage0_root = Path(sys.argv[1])
requested_checkpoint = Path(sys.argv[2]).resolve() if sys.argv[2] else None
requested_skills = Path(sys.argv[3]).resolve() if sys.argv[3] else None
expected_sha = sys.argv[4]
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
report = json.loads((stage0_root / "train_report.json").read_text())
selection = json.loads((stage0_root / "stage0_selection.json").read_text())
if report.get("status") != "ok" or report.get("stage") != "clstr_vnext_stage0":
    raise SystemExit("Stage2 requires a completed canonical Stage0 report")
if selection.get("status") != "ok" or report.get("selection") != selection:
    raise SystemExit("Stage2 requires a quality-gated Stage0 selection")
step = int(report.get("step") or 0)
checkpoint = Path(str(report.get("checkpoint_path") or ""))
skills = requested_skills or (stage0_root / "selected_skills.jsonl")
if step <= 0 or checkpoint.name != f"clstr_vnext_stage0-step{step}.pt":
    raise SystemExit("Stage2 Stage0 report does not name its final checkpoint")
if not checkpoint.is_file() or not skills.is_file():
    raise SystemExit("final Stage0 handoff artifact is missing")
if requested_checkpoint is not None and checkpoint.resolve() != requested_checkpoint:
    raise SystemExit("requested Stage0 checkpoint does not match its selected report")
if expected_sha and sha256(checkpoint) != expected_sha:
    raise SystemExit("requested Stage0 checkpoint digest does not match")
print(checkpoint.resolve())
print(skills.resolve())
PY
)
STAGE0_CHECKPOINT_PATH=${STAGE0_SELECTED[0]}
STAGE0_SKILLS_PATH=${STAGE0_SELECTED[1]}

readarray -t CANDIDATE_SELECTED < <("${PYTHON_BIN}" - "${CANDIDATE_OUTPUT_DIR}" \
  "${STAGE0_CHECKPOINT_PATH}" "${REQUESTED_CANDIDATE_CHECKPOINT_PATH}" \
  "${EXPECTED_CANDIDATE_CHECKPOINT_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
stage0_checkpoint = Path(sys.argv[2])
requested_checkpoint = Path(sys.argv[3]).resolve() if sys.argv[3] else None
expected_sha = sys.argv[4]
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
selection = json.loads((root / "compressor_selection.json").read_text())
if selection.get("status") != "ok":
    raise SystemExit("Stage2 requires a quality-gated static-reranker selection")
checkpoint = Path(selection["selected_checkpoint_path"])
if not checkpoint.is_file():
    raise SystemExit("selected static-reranker checkpoint is missing")
if requested_checkpoint is not None and checkpoint.resolve() != requested_checkpoint:
    raise SystemExit("requested candidate checkpoint does not match its selected report")
observed = sha256(checkpoint)
if observed != selection.get("selected_checkpoint_sha256"):
    raise SystemExit("selected static-reranker checkpoint digest changed")
if expected_sha and observed != expected_sha:
    raise SystemExit("requested candidate checkpoint digest does not match")
parent = sha256(stage0_checkpoint)
if parent != selection.get("parent_stage0_checkpoint_sha256"):
    raise SystemExit("static-reranker parent Stage0 checkpoint changed")
print(checkpoint.resolve())
PY
)
CANDIDATE_CHECKPOINT_PATH=${CANDIDATE_SELECTED[0]}

resume_args=()
warm_start_args=()
if [[ -n "${WARM_START_CHECKPOINT_PATH:-}" ]]; then
  if [[ ! -f "${WARM_START_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: Stage2 objective warm-start checkpoint is missing: %s\n' \
      "${WARM_START_CHECKPOINT_PATH}" >&2
    exit 2
  fi
  WARM_START_CHECKPOINT_PATH=$(realpath "${WARM_START_CHECKPOINT_PATH}")
  warm_start_args=(--warm_start_checkpoint_path "${WARM_START_CHECKPOINT_PATH}")
fi
if [[ -f "${OUTPUT_DIR}/stage2_selection.json" ]]; then
  if [[ ${#warm_start_args[@]} -gt 0 ]]; then
    printf 'ERROR: objective warm start requires a fresh Stage2 output directory\n' >&2
    exit 2
  fi
  RESUME_CHECKPOINT=$("${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
selection = json.loads((root / "stage2_selection.json").read_text())
report = json.loads((root / "train_report.json").read_text())
if selection.get("status") not in {"ok", "action_required"}:
    raise SystemExit("Stage2 resume requires a mechanically complete prior segment")
if report.get("status") != selection.get("status") or report.get("stage") != "clstr_vnext_stage2":
    raise SystemExit("Stage2 resume report/selection boundary is inconsistent")
records = selection.get("validation_records") or []
if not records:
    raise SystemExit("Stage2 selection has no complete validation record")
step = int(records[-1]["step"])
if int(report.get("step") or 0) != step:
    raise SystemExit("Stage2 resume report does not end at its validation boundary")
path = root / "checkpoints" / f"clstr_vnext_stage2-step{step}.pt"
if not path.is_file():
    raise SystemExit(f"Stage2 completed-boundary checkpoint is missing: {path}")
print(path.resolve())
PY
  )
  resume_args=(--resume_checkpoint_path "${RESUME_CHECKPOINT}")
elif [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: Stage2 output exists without a completed selection boundary: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage2_train.py \
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}" \
  --candidate_checkpoint_path "${CANDIDATE_CHECKPOINT_PATH}" \
  --skills_path "${STAGE0_SKILLS_PATH}" \
  --trajectory_rows_path "${TRAJECTORY_ROWS}" \
  --trajectory_dev_rows_path "${TRAJECTORY_DEV_ROWS}" \
  --causal_pair_support_rows_path "${PAIR_SUPPORT_ROWS}" \
  --causal_pair_support_dev_rows_path "${PAIR_SUPPORT_DEV_ROWS}" \
  --inventory_catalogs_path "${INVENTORY_CATALOGS}" \
  --data_contract_path "${DATA_MANIFEST_PATH}" \
  --causal_branch_pairs_path "${CAUSAL_BRANCH_PAIRS}" \
  --causal_branch_dev_pairs_path "${CAUSAL_BRANCH_DEV_PAIRS}" \
  --causal_order_pairs_path "${CAUSAL_ORDER_PAIRS}" \
  --causal_order_dev_pairs_path "${CAUSAL_ORDER_DEV_PAIRS}" \
  --causal_outcome_pairs_path "${CAUSAL_OUTCOME_PAIRS}" \
  --causal_outcome_dev_pairs_path "${CAUSAL_OUTCOME_DEV_PAIRS}" \
  --one_error_prefix_rows_path "${ONE_ERROR_PREFIX_ROWS}" \
  --one_error_prefix_dev_rows_path "${ONE_ERROR_PREFIX_DEV_ROWS}" \
  --two_or_more_error_prefix_rows_path "${TWO_ERROR_PREFIX_ROWS}" \
  --two_or_more_error_prefix_dev_rows_path "${TWO_ERROR_PREFIX_DEV_ROWS}" \
  --recovery_prefix_rows_path "${RECOVERY_PREFIX_ROWS}" \
  --recovery_prefix_dev_rows_path "${RECOVERY_PREFIX_DEV_ROWS}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${SEGMENT_END}" \
  --curriculum_total_steps "${CURRICULUM_TOTAL_STEPS}" \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE}" \
  --transition_scale_initial 0.05 \
  --result_scale_initial 0.05 \
  --recall_scale_initial 0.10 \
  "${native_synchronization_args[@]}" \
  --max_horizon 16 \
  "${family_horizon_args[@]}" \
  --coarse_k 500 \
  --compressed_m 64 \
  --final_k 100 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --frozen_cache_dir "${CACHE_ROOT}" \
  "${cache_mode_args[@]}" \
  --candidate_learning_rate_scale 1.0 \
  --static_route_learning_rate_scale 1.0 \
  --lambda_recall "${LAMBDA_RECALL}" \
  --lambda_compression "${LAMBDA_COMPRESSION}" \
  --lambda_anchor 1.0e-4 \
  --teacher_retention_start "${TEACHER_RETENTION_START}" \
  --teacher_retention_end "${TEACHER_RETENTION_END}" \
  --lambda_safety "${LAMBDA_SAFETY}" \
  --lambda_raw_route "${LAMBDA_RAW_ROUTE}" \
  --lambda_route_topk "${LAMBDA_ROUTE_TOPK}" \
  --route_topk "${ROUTE_TOPK}" \
  --route_topk_margin "${ROUTE_TOPK_MARGIN}" \
  --route_topk_recoverable_weight "${ROUTE_TOPK_RECOVERABLE_WEIGHT}" \
  --route_topk_listwise_weight "${ROUTE_TOPK_LISTWISE_WEIGHT}" \
  --lambda_mixture "${LAMBDA_MIXTURE}" \
  --mixture_utility_scale "${MIXTURE_UTILITY_SCALE}" \
  --lambda_history "${LAMBDA_HISTORY}" \
  --lambda_order 0.0 \
  --lambda_result 0.0 \
  --counterfactual_margin 0.2 \
  --no_regret_tolerance "${NO_REGRET_TOLERANCE}" \
  --ordinary_mrr_noninferiority_tolerance 0.01 \
  --seed 23 \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --validation_interval "${VALIDATION_INTERVAL}" \
  --max_dev_pairs_per_kind 0 \
  --max_ordinary_dev_rows "${MAX_ORDINARY_DEV_ROWS}" \
  --ordinary_dev_batch_size 64 \
  --max_robust_dev_rows_per_kind 0 \
  --minimum_dev_score_gain 0.0 \
  --minimum_gradient_norm 1.0e-10 \
  --minimum_dev_clusters_per_enabled_kind 20 \
  --minimum_dev_clusters_per_source 5 \
  --minimum_ordinary_dev_clusters_per_stratum 10 \
  --minimum_ordinary_dev_stratum_coverage 0.90 \
  --minimum_full_pool_clusters 20 \
  --require_clean_source \
  "${warm_start_args[@]}" \
  "${resume_args[@]}"

if [[ "${SEGMENT_END}" -eq "${CURRICULUM_TOTAL_STEPS}" \
  && "${FINALIZE_FULL_CHAIN}" -eq 1 ]]; then
  if [[ "${OUTPUT_DIR}" != "${RUN_ROOT}/stage2" ]]; then
    printf 'ERROR: finalizer requires canonical OUTPUT_DIR=%s/stage2\n' \
      "${RUN_ROOT}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" scripts/finalize_clstr_vnext_full_chain.py \
    "${RUN_ROOT}" "${DATA_MANIFEST_PATH}"
fi
