#!/bin/bash
#SBATCH --job-name=clstr-vnext-eval
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must identify one completed canonical vNext run}
BENCHMARK=${BENCHMARK:?BENCHMARK must be toolbench_g3, toolsandbox, tau2, or trajectbench}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be a fresh benchmark output directory}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

RUN_ROOT=$(realpath -m "${RUN_ROOT}")
STAGE2_OUTPUT_DIR=$(realpath -m "${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/stage2}")
case "${STAGE2_OUTPUT_DIR}" in
  "${RUN_ROOT}"/*) ;;
  *)
    printf 'ERROR: Stage2 output must remain inside RUN_ROOT: stage2=%s run=%s\n' \
      "${STAGE2_OUTPUT_DIR}" "${RUN_ROOT}" >&2
    exit 2
    ;;
esac

if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: benchmark output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

STAGE2_RELEASE_SELECTION_PATH=${STAGE2_RELEASE_SELECTION_PATH:-}
UNIFIED_ROUTER_PATH=${UNIFIED_ROUTER_PATH:-}
SUPPORT_AWARE_ANCHOR=${SUPPORT_AWARE_ANCHOR:-0}
mapfile -t RELEASE_CONTRACT < <("${PYTHON_BIN}" - \
  "${STAGE2_OUTPUT_DIR}" \
  "${STAGE2_RELEASE_SELECTION_PATH}" \
  "${PROJECT_ROOT}" \
  "${UNIFIED_ROUTER_PATH}" \
  "${SUPPORT_AWARE_ANCHOR}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
release_arg = str(sys.argv[2]).strip()
project_root = Path(sys.argv[3]).resolve()
router_arg = str(sys.argv[4]).strip()
support_aware_anchor = str(sys.argv[5]).strip() == "1"
if support_aware_anchor and router_arg:
    raise SystemExit("support-aware anchoring may not use a learned router artifact")
if support_aware_anchor and not release_arg:
    raise SystemExit("support-aware anchoring requires an explicit release selection")
selection_path = (
    Path(release_arg).resolve()
    if release_arg
    else (root / "stage2_selection.json").resolve()
)
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if selection.get("status") != "ok":
    raise SystemExit("Stage2 selection is not release-ready")
if release_arg:
    selection_schema = str(selection.get("schema_version") or "")
    if selection_schema not in {
        "clstr_vnext_stage2_open_pool_selection_v1",
        "clstr_vnext_stage2_matched_multibench_selection_v1",
    }:
        raise SystemExit("unsupported Stage2 release-selection schema")
    if selection.get("uses_benchmark_eval_rows") or selection.get("uses_test_metrics"):
        raise SystemExit("Stage2 release selection uses benchmark test evidence")
    if selection_schema == "clstr_vnext_stage2_open_pool_selection_v1":
        if selection.get("selection_mode") != "open_pool_full_pool":
            raise SystemExit("Stage2 release selection has the wrong mode")
        if not selection.get("closed_set_dispatch_required"):
            raise SystemExit("Stage2 release selection omits closed-set dispatch")
    else:
        if selection.get("selection_mode") != "matched_multibench_unified_recurrent":
            raise SystemExit("matched Stage2 release selection has the wrong mode")
        if selection.get("closed_set_dispatch_required"):
            raise SystemExit("matched Stage2 release may not use closed-set dispatch")
        if selection.get("uses_benchmark_identity_for_dispatch") or selection.get(
            "uses_source_identity_for_dispatch"
        ):
            raise SystemExit("matched Stage2 release may not dispatch by benchmark/source")
        if router_arg or support_aware_anchor:
            raise SystemExit("matched Stage2 release may not use a deployment router")
    if Path(str(selection.get("stage2_output_dir") or "")).resolve() != root:
        raise SystemExit("Stage2 release selection names a different output root")
    original = (root / "stage2_selection.json").resolve()
    if Path(str(selection.get("original_stage2_selection_path") or "")).resolve() != original:
        raise SystemExit("Stage2 release selection names the wrong training selection")
    expected_original_sha = str(
        selection.get("original_stage2_selection_sha256") or ""
    )
    actual_original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    if not expected_original_sha or actual_original_sha != expected_original_sha:
        raise SystemExit("original Stage2 training-selection digest differs")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if selection_schema == "clstr_vnext_stage2_matched_multibench_selection_v1":
        source_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if source_status:
            raise SystemExit("matched Stage2 release requires a clean immutable source")
path = Path(str(selection.get("selected_checkpoint_path") or ""))
if not path.is_file():
    raise SystemExit(f"selected Stage2 checkpoint is missing: {path}")
resolved = path.resolve()
try:
    resolved.relative_to((root / "checkpoints").resolve())
except ValueError as error:
    raise SystemExit("selected Stage2 checkpoint escapes STAGE2_OUTPUT_DIR") from error
if release_arg:
    expected_checkpoint_sha = str(selection.get("selected_checkpoint_sha256") or "")
    actual_checkpoint_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not expected_checkpoint_sha or actual_checkpoint_sha != expected_checkpoint_sha:
        raise SystemExit("release-selected Stage2 checkpoint digest differs")
    actual_selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    if router_arg:
        router_path = Path(router_arg).resolve()
        if not router_path.is_file():
            raise SystemExit(f"unified-router artifact is missing: {router_path}")
        import torch

        router = torch.load(router_path, map_location="cpu")
        if (
            not isinstance(router, dict)
            or router.get("schema_version")
            != "clstr_vnext_unified_three_expert_router_v6"
            or router.get("routing_mode") != "sparse_top1_expert"
            or router.get("status") != "ok"
            or router.get("blockers")
        ):
            raise SystemExit("unified-router artifact is not calibration-approved")
        if str(router.get("source_commit") or "") != source_commit:
            raise SystemExit("unified-router artifact was trained from a different source commit")
        training_contract = router.get("training_contract")
        expected_training_contract = {
            "default_expert_prior": "foundation",
            "observation_grounding_feature": "observation_correction_fraction",
            "paired_calibration_protocol": (
                "same_prefix_factual_and_action_only_result_suppressed_v1"
            ),
            "paired_views_share_trajectory_split": True,
            "foundation_expert_protocol": (
                "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
            ),
            "expert_supervision_protocol": (
                "class_balanced_sparse_minimum_regret_top1_v1"
            ),
            "expert_tie_break_order": [
                "foundation",
                "adapted_static",
                "recurrent_memory",
            ],
            "expert_utility": "reciprocal_rank_plus_0.5_top5",
            "oracle_expert_recall_min_rows": 20,
            "oracle_expert_recall_floor": 0.20,
            "calibration_no_regret_tolerances": {
                "global_mrr": 0.005,
                "global_recall_at_5": 0.01,
                "family_mrr": 0.02,
                "family_recall_at_5": 0.03,
                "memory_evidence_mrr": 0.005,
                "memory_evidence_recall_at_5": 0.01,
            },
        }
        if not isinstance(training_contract, dict) or any(
            training_contract.get(key) != value
            for key, value in expected_training_contract.items()
        ):
            raise SystemExit(
                "unified-router observation-grounded calibration contract differs"
            )
        wrapper = router.get("stage2_release_wrapper")
        expected_wrapper = {
            "schema_version": "clstr_vnext_unified_router_stage2_release_wrapper_v1",
            "router_source_commit": source_commit,
            "stage2_release_selection_path": str(selection_path),
            "stage2_release_selection_sha256": actual_selection_sha,
            "stage2_release_source_commit": str(selection.get("source_commit") or ""),
            "stage2_selection_mode": str(selection.get("selection_mode") or ""),
            "stage2_checkpoint_path": str(resolved),
            "stage2_checkpoint_sha256": actual_checkpoint_sha,
            "original_stage2_selection_path": str(original),
            "original_stage2_selection_sha256": actual_original_sha,
            "uses_benchmark_eval_rows": False,
            "uses_test_metrics": False,
        }
        if not isinstance(wrapper, dict):
            raise SystemExit("unified-router artifact lacks its Stage2 release wrapper")
        for key, expected in expected_wrapper.items():
            if wrapper.get(key) != expected:
                raise SystemExit(
                    f"unified-router Stage2 release wrapper differs: {key}"
                )
        if str(router.get("stage2_checkpoint_sha256") or "") != actual_checkpoint_sha:
            raise SystemExit("unified-router Stage2 checkpoint digest differs")
    elif (
        not support_aware_anchor
        and source_commit != str(selection.get("source_commit") or "")
    ):
        raise SystemExit("release selection was finalized from a different source commit")
elif router_arg:
    raise SystemExit("unified router requires an explicit Stage2 release selection")
print(resolved)
print(selection_path if release_arg else "NONE")
print(str(selection.get("schema_version") or "NONE") if release_arg else "NONE")
PY
)
SELECTED_CHECKPOINT_PATH=${RELEASE_CONTRACT[0]}
RESOLVED_STAGE2_RELEASE_SELECTION_PATH=${RELEASE_CONTRACT[1]}
RESOLVED_STAGE2_RELEASE_SELECTION_SCHEMA=${RELEASE_CONTRACT[2]}
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH=$(realpath -m "${CHECKPOINT_PATH}")
  if [[ "${CHECKPOINT_PATH}" != "${SELECTED_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: explicit CHECKPOINT_PATH does not match the release-ready Stage2 selection\n' >&2
    exit 2
  fi
else
  CHECKPOINT_PATH=${SELECTED_CHECKPOINT_PATH}
fi

TRAINING_SKILLS_PATH=${TRAINING_SKILLS_PATH:-${RUN_ROOT}/stage0/selected_skills.jsonl}
EVAL_CACHE_ROOT=${EVAL_CACHE_ROOT:-${RUN_ROOT}/eval_frozen_qwen_cache}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-}
CANDIDATE_QUERY_MODE=${CANDIDATE_QUERY_MODE:-learned_causal}
CANDIDATE_COMPRESSION_MODE=${CANDIDATE_COMPRESSION_MODE:-learned}
COARSE_K=${COARSE_K:-500}
COMPRESSED_M=${COMPRESSED_M:-64}
DYNAMIC_EXTRA_K=${DYNAMIC_EXTRA_K:-${COMPRESSED_M}}
FINAL_K=${FINAL_K:-100}
DISABLE_VERIFIED_RESULT_CORRECTION=${DISABLE_VERIFIED_RESULT_CORRECTION:-0}
ALLOW_STAGE0_STATIC_DIAGNOSTIC=${ALLOW_STAGE0_STATIC_DIAGNOSTIC:-0}
ALLOW_STATIC_RERANKER_DIAGNOSTIC=${ALLOW_STATIC_RERANKER_DIAGNOSTIC:-0}
TOOLBENCH_POOL_SCOPE=${TOOLBENCH_POOL_SCOPE:-global}
TOOLBENCH_NATIVE_SKILLS=${TOOLBENCH_NATIVE_SKILLS:-}
CLOSED_SET_FOUNDATION_CHECKPOINT_PATH=${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH:-}
MATCHED_UNION_MANIFEST_PATH=${MATCHED_UNION_MANIFEST_PATH:-}
MATCHED_SPLIT=${MATCHED_SPLIT:-test}

if [[ "${MATCHED_SPLIT}" != "dev" && "${MATCHED_SPLIT}" != "test" ]]; then
  printf 'ERROR: MATCHED_SPLIT must be dev or test\n' >&2
  exit 2
fi

args=(
  "${PYTHON_BIN}" scripts/run_clstr_vnext_eval.py
  --benchmark "${BENCHMARK}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --training_skills_path "${TRAINING_SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --coarse_k "${COARSE_K}"
  --compressed_m "${COMPRESSED_M}"
  --dynamic_extra_k "${DYNAMIC_EXTRA_K}"
  --final_k "${FINAL_K}"
  --candidate_query_mode "${CANDIDATE_QUERY_MODE}"
  --candidate_compression_mode "${CANDIDATE_COMPRESSION_MODE}"
  --score_batch_size 32
  --cache_batch_size 128
  --belief_top_k 64
  --frozen_cache_dir "${EVAL_CACHE_ROOT}"
)
if [[ "${DISABLE_VERIFIED_RESULT_CORRECTION}" == "1" ]]; then
  args+=(--disable_verified_result_correction)
elif [[ "${DISABLE_VERIFIED_RESULT_CORRECTION}" != "0" ]]; then
  printf 'ERROR: DISABLE_VERIFIED_RESULT_CORRECTION must be 0 or 1\n' >&2
  exit 2
fi
if [[ -n "${MAX_EVAL_ROWS}" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi
if [[ "${ALLOW_STAGE0_STATIC_DIAGNOSTIC}" == "1" ]]; then
  args+=(--allow_stage0_static_diagnostic)
fi
if [[ "${ALLOW_STATIC_RERANKER_DIAGNOSTIC}" == "1" ]]; then
  args+=(--allow_static_reranker_diagnostic)
fi
if [[ -n "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}" ]]; then
  args+=(
    --closed_set_foundation_checkpoint_path
    "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}"
  )
fi
if [[ -n "${UNIFIED_ROUTER_PATH}" ]]; then
  if [[ -z "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: unified router requires the foundation checkpoint\n' >&2
    exit 2
  fi
  if [[ ! -f "${UNIFIED_ROUTER_PATH}" ]]; then
    printf 'ERROR: unified-router artifact is missing: %s\n' \
      "${UNIFIED_ROUTER_PATH}" >&2
    exit 2
  fi
  args+=(--unified_router_path "${UNIFIED_ROUTER_PATH}")
fi
if [[ "${SUPPORT_AWARE_ANCHOR}" == "1" ]]; then
  if [[ -z "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: support-aware anchoring requires the foundation checkpoint\n' >&2
    exit 2
  fi
  if [[ -n "${UNIFIED_ROUTER_PATH}" ]]; then
    printf 'ERROR: support-aware anchoring may not use UNIFIED_ROUTER_PATH\n' >&2
    exit 2
  fi
  args+=(--support_aware_anchor)
elif [[ "${SUPPORT_AWARE_ANCHOR}" != "0" ]]; then
  printf 'ERROR: SUPPORT_AWARE_ANCHOR must be 0 or 1\n' >&2
  exit 2
fi
if [[ "${RESOLVED_STAGE2_RELEASE_SELECTION_PATH}" != "NONE" ]]; then
  if [[ "${RESOLVED_STAGE2_RELEASE_SELECTION_SCHEMA}" == \
    "clstr_vnext_stage2_open_pool_selection_v1" && \
    -z "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: open-pool release selection requires the closed-set foundation\n' >&2
    exit 2
  fi
  if [[ "${RESOLVED_STAGE2_RELEASE_SELECTION_SCHEMA}" == \
    "clstr_vnext_stage2_matched_multibench_selection_v1" && \
    -n "${CLOSED_SET_FOUNDATION_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: matched unified recurrent release may not use the closed-set foundation\n' >&2
    exit 2
  fi
  args+=(
    --stage2_release_selection_path
    "${RESOLVED_STAGE2_RELEASE_SELECTION_PATH}"
  )
fi

append_matched_eval_args() {
  local benchmark=$1
  local -a matched_artifacts=()
  if [[ -z "${MATCHED_UNION_MANIFEST_PATH}" ]]; then
    printf 'ERROR: %s final evaluation requires MATCHED_UNION_MANIFEST_PATH\n' \
      "${benchmark}" >&2
    exit 2
  fi
  if [[ ! -f "${MATCHED_UNION_MANIFEST_PATH}" ]]; then
    printf 'ERROR: matched union manifest is missing: %s\n' \
      "${MATCHED_UNION_MANIFEST_PATH}" >&2
    exit 2
  fi
  mapfile -t matched_artifacts < <("${PYTHON_BIN}" - \
    "${MATCHED_UNION_MANIFEST_PATH}" "${benchmark}" "${MATCHED_SPLIT}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
benchmark = sys.argv[2]
split = sys.argv[3]
route_files = manifest.get("route_files") or {}
rows = Path(str((route_files.get(f"{benchmark}_{split}") or {}).get("path") or ""))
skills = Path(str((route_files.get(f"{benchmark}_skills") or {}).get("path") or ""))
if not rows.is_file() or not skills.is_file():
    raise SystemExit(
        f"matched union lacks bound {benchmark}/{split} rows or skills"
    )
print(rows.resolve())
print(skills.resolve())
PY
  )
  if [[ ${#matched_artifacts[@]} -ne 2 ]]; then
    printf 'ERROR: failed to resolve matched %s/%s evaluation artifacts\n' \
      "${benchmark}" "${MATCHED_SPLIT}" >&2
    exit 2
  fi
  args+=(
    --prebuilt_source_rows_path "${matched_artifacts[0]}"
    --prebuilt_skills_path "${matched_artifacts[1]}"
    --matched_union_manifest_path "${MATCHED_UNION_MANIFEST_PATH}"
    --matched_split "${MATCHED_SPLIT}"
  )
}

case "${BENCHMARK}" in
  toolbench_g3)
    TOOLBENCH_ROWS=${TOOLBENCH_ROWS:?TOOLBENCH_ROWS must name the verified v2 export}
    TOOLBENCH_SKILLS=${TOOLBENCH_SKILLS:?TOOLBENCH_SKILLS must match that export}
    args+=(
      --toolbench_eval_trajectories_path "${TOOLBENCH_ROWS}"
      --toolbench_skills_path "${TOOLBENCH_SKILLS}"
      --toolbench_pool_scope "${TOOLBENCH_POOL_SCOPE}"
      --require_verified_toolbench_results
    )
    if [[ "${TOOLBENCH_POOL_SCOPE}" == "native" ]]; then
      if [[ -z "${TOOLBENCH_NATIVE_SKILLS}" ]]; then
        printf 'ERROR: native ToolBench evaluation requires TOOLBENCH_NATIVE_SKILLS\n' >&2
        exit 2
      fi
      args+=(--toolbench_native_skills_path "${TOOLBENCH_NATIVE_SKILLS}")
    elif [[ -n "${TOOLBENCH_NATIVE_SKILLS}" ]]; then
      printf 'ERROR: TOOLBENCH_NATIVE_SKILLS is valid only for native scope\n' >&2
      exit 2
    fi
    ;;
  toolsandbox)
    append_matched_eval_args toolsandbox
    ;;
  tau2)
    append_matched_eval_args tau2
    ;;
  trajectbench|traject_bench)
    TRAJECTBENCH_PUBLIC_DATA=${TRAJECTBENCH_PUBLIC_DATA:?TRAJECTBENCH_PUBLIC_DATA must name TRAJECT-Bench public_data}
    TRAJECTBENCH_SPLIT_PARTITION=${TRAJECTBENCH_SPLIT_PARTITION:-test}
    if [[ "${TRAJECTBENCH_SPLIT_PARTITION}" != "dev" && \
      "${TRAJECTBENCH_SPLIT_PARTITION}" != "test" ]]; then
      printf 'ERROR: TRAJECTBENCH_SPLIT_PARTITION must be dev or test\n' >&2
      exit 2
    fi
    args+=(
      --trajectbench_public_data "${TRAJECTBENCH_PUBLIC_DATA}"
      --trajectbench_split_partition "${TRAJECTBENCH_SPLIT_PARTITION}"
    )
    ;;
  *)
    printf 'ERROR: unsupported benchmark: %s\n' "${BENCHMARK}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
