#!/bin/bash
#SBATCH --job-name=clstr_vnext_s2_smoke
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

BUNDLE_ROOT=${BUNDLE_ROOT:?BUNDLE_ROOT must come from the freshly audited causal manifest}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:?STAGE0_OUTPUT_DIR must be the matching Stage0 smoke}
CANDIDATE_OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:?CANDIDATE_OUTPUT_DIR must be the matching static-reranker smoke}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be a fresh Stage2 smoke directory}
FROZEN_CACHE_DIR=${FROZEN_CACHE_DIR:?FROZEN_CACHE_DIR must identify the matching smoke cache}
STAGE0_SKILLS_PATH=${STAGE0_SKILLS_PATH:-${STAGE0_OUTPUT_DIR}/selected_skills.jsonl}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-$("${PYTHON_BIN}" - "${STAGE0_OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
selection = json.loads((root / "stage0_selection.json").read_text())
if selection.get("status") != "ok":
    raise SystemExit("Stage2 smoke requires a quality-gated Stage0 smoke")
print(Path(selection["selected_checkpoint_path"]).resolve())
PY
)}
MAX_STEPS=${MAX_STEPS:-2}
CURRICULUM_TOTAL_STEPS=${CURRICULUM_TOTAL_STEPS:-${MAX_STEPS}}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-1}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-${VALIDATION_INTERVAL}}
MAX_DEV_PAIRS_PER_KIND=${MAX_DEV_PAIRS_PER_KIND:-4}
MAX_ORDINARY_DEV_ROWS=${MAX_ORDINARY_DEV_ROWS:-8}
ORDINARY_DEV_BATCH_SIZE=${ORDINARY_DEV_BATCH_SIZE:-4}
MAX_HORIZON=${MAX_HORIZON:-16}
FAMILY_FIRST_HORIZON_SAMPLING=${FAMILY_FIRST_HORIZON_SAMPLING:-0}
family_horizon_args=()
case "${FAMILY_FIRST_HORIZON_SAMPLING}" in
  0) ;;
  1) family_horizon_args=(--family_first_horizon_sampling) ;;
  *)
    printf 'ERROR: FAMILY_FIRST_HORIZON_SAMPLING must be 0 or 1\n' >&2
    exit 2
    ;;
esac
BELIEF_TOP_K=${BELIEF_TOP_K:-64}

for path in \
  "${STAGE0_CHECKPOINT_PATH}" \
  "${STAGE0_SKILLS_PATH}" \
  "${CANDIDATE_OUTPUT_DIR}/compressor_selection.json" \
  "${BUNDLE_ROOT}/trajectory_rows.jsonl" \
  "${BUNDLE_ROOT}/trajectory_dev_rows.jsonl" \
  "${BUNDLE_ROOT}/causal_pair_support_rows.jsonl" \
  "${BUNDLE_ROOT}/causal_pair_support_dev_rows.jsonl" \
  "${BUNDLE_ROOT}/causal_branch_pairs.jsonl" \
  "${BUNDLE_ROOT}/causal_branch_dev_pairs.jsonl" \
  "${BUNDLE_ROOT}/inventory_catalogs.jsonl" \
  "${BUNDLE_ROOT}/manifest.json"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: missing Stage2 smoke input: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'ERROR: Stage2 smoke output already exists: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

CANDIDATE_CHECKPOINT_PATH=$("${PYTHON_BIN}" - \
  "${CANDIDATE_OUTPUT_DIR}" "${STAGE0_CHECKPOINT_PATH}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
stage0_checkpoint = Path(sys.argv[2])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

selection = json.loads((root / "compressor_selection.json").read_text())
if selection.get("status") != "ok":
    raise SystemExit("Stage2 smoke requires a quality-gated static-reranker smoke")
checkpoint = Path(str(selection.get("selected_checkpoint_path") or ""))
if not checkpoint.is_file():
    raise SystemExit("selected static-reranker smoke checkpoint is missing")
if sha256(checkpoint) != selection.get("selected_checkpoint_sha256"):
    raise SystemExit("static-reranker smoke checkpoint digest changed")
if sha256(stage0_checkpoint) != selection.get("parent_stage0_checkpoint_sha256"):
    raise SystemExit("static-reranker smoke parent Stage0 checkpoint changed")
print(checkpoint.resolve())
PY
)

"${PYTHON_BIN}" scripts/run_clstr_vnext_stage2_train.py \
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}" \
  --candidate_checkpoint_path "${CANDIDATE_CHECKPOINT_PATH}" \
  --skills_path "${STAGE0_SKILLS_PATH}" \
  --trajectory_rows_path "${BUNDLE_ROOT}/trajectory_rows.jsonl" \
  --trajectory_dev_rows_path "${BUNDLE_ROOT}/trajectory_dev_rows.jsonl" \
  --causal_pair_support_rows_path "${BUNDLE_ROOT}/causal_pair_support_rows.jsonl" \
  --causal_pair_support_dev_rows_path "${BUNDLE_ROOT}/causal_pair_support_dev_rows.jsonl" \
  --inventory_catalogs_path "${BUNDLE_ROOT}/inventory_catalogs.jsonl" \
  --data_contract_path "${BUNDLE_ROOT}/manifest.json" \
  --causal_branch_pairs_path "${BUNDLE_ROOT}/causal_branch_pairs.jsonl" \
  --causal_branch_dev_pairs_path "${BUNDLE_ROOT}/causal_branch_dev_pairs.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --curriculum_total_steps "${CURRICULUM_TOTAL_STEPS}" \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-4 \
  --max_horizon "${MAX_HORIZON}" \
  "${family_horizon_args[@]}" \
  --coarse_k 500 \
  --compressed_m 64 \
  --final_k 100 \
  --belief_top_k "${BELIEF_TOP_K}" \
  --cache_batch_size 64 \
  --cache_shard_size 64 \
  --frozen_cache_dir "${FROZEN_CACHE_DIR}" \
  --candidate_learning_rate_scale 1.0 \
  --static_route_learning_rate_scale 1.0 \
  --transition_scale_initial 0.05 \
  --result_scale_initial 0.05 \
  --recall_scale_initial 0.10 \
  --lambda_recall 0.2 \
  --lambda_compression 0.2 \
  --lambda_anchor 1.0e-4 \
  --teacher_retention_start 1.0 \
  --teacher_retention_end 0.0 \
  --lambda_safety 0.05 \
  --lambda_raw_route 0.5 \
  --lambda_mixture 0.2 \
  --mixture_utility_scale 0.5 \
  --lambda_history 0.2 \
  --lambda_order 0.0 \
  --lambda_result 0.0 \
  --counterfactual_margin 0.2 \
  --no_regret_tolerance 0.05 \
  --ordinary_mrr_noninferiority_tolerance 0.01 \
  --seed 23 \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --validation_interval "${VALIDATION_INTERVAL}" \
  --max_dev_pairs_per_kind "${MAX_DEV_PAIRS_PER_KIND}" \
  --max_ordinary_dev_rows "${MAX_ORDINARY_DEV_ROWS}" \
  --ordinary_dev_batch_size "${ORDINARY_DEV_BATCH_SIZE}" \
  --max_robust_dev_rows_per_kind 1 \
  --minimum_dev_score_gain -1.0 \
  --minimum_gradient_norm 1.0e-14 \
  --minimum_dev_clusters_per_enabled_kind 1 \
  --minimum_dev_clusters_per_source 1 \
  --minimum_ordinary_dev_clusters_per_stratum 1 \
  --minimum_ordinary_dev_stratum_coverage 0.90 \
  --minimum_full_pool_clusters 1 \
  | tee "${OUTPUT_DIR}.stdout.log"
