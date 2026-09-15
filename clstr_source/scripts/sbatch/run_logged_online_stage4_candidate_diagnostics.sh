#!/bin/bash
#SBATCH --time=01:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/logged_online_stage4_candidate_diagnostics/toolbench_g3_top350_rows1024}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}

ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3}
MAX_ROWS=${MAX_ROWS:-1024}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-8}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-25}
TOP_K=${TOP_K:-10}

if [[ -s "${ROUTING_CHECKPOINT_PATH}" ]]; then
  :
else
  echo "ERROR: logged-online Stage4 candidate diagnostics requires Stage0 checkpoint: ${ROUTING_CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -s "${TRAJECTORIES_PATH}" ]]; then
  :
else
  echo "ERROR: trajectories file not found: ${TRAJECTORIES_PATH}" >&2
  exit 2
fi

if [[ -s "${SKILLS_PATH}" ]]; then
  :
else
  echo "ERROR: skill pool file not found: ${SKILLS_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/audit_logged_online_stage4_candidates.py \
  --routing_checkpoint_path "${ROUTING_CHECKPOINT_PATH}" \
  --trajectories_path "${TRAJECTORIES_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --allowed_benchmarks "${ALLOWED_BENCHMARKS}" \
  --max_rows "${MAX_ROWS}" \
  --stage0_top_m "${STAGE0_TOP_M}" \
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}" \
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}" \
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}" \
  --top_k "${TOP_K}" | tee "${OUTPUT_DIR}/diagnostics_stdout.json"
