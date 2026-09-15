#!/bin/bash
#SBATCH --time=01:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE_CHECKPOINT_PATH=${STAGE_CHECKPOINT_PATH:-outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/stage1_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_prior_gate_sweep/clstr_s12_clean_b16_full2_stage1_smoke}

TOP_M=${TOP_M:-500}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_ROWS=${MAX_ROWS:-128}
MAX_DIAGNOSTIC_ROWS=${MAX_DIAGNOSTIC_ROWS:-128}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"
STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-32}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-5}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-stage0_topk_trajectory_prior}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-50}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
FORMULAS=${FORMULAS:-fixed_0_00,fixed_0_10,fixed_0_15,fixed_0_25,fixed_0_50,margin_gate,entropy_gate,margin_entropy_gate}
LAMBDA_MIN=${LAMBDA_MIN:-0.05}
LAMBDA_MAX=${LAMBDA_MAX:-0.5}
MARGIN_THRESHOLD=${MARGIN_THRESHOLD:-1.0}
ENTROPY_LOW=${ENTROPY_LOW:-0.2}
ENTROPY_HIGH=${ENTROPY_HIGH:-0.8}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/sweep_clstr_prior_gate.py
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --stage_checkpoint_path "${STAGE_CHECKPOINT_PATH}"
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --top_m "${TOP_M}"
  --batch_size "${BATCH_SIZE}"
  --max_rows "${MAX_ROWS}"
  --max_diagnostic_rows "${MAX_DIAGNOSTIC_ROWS}"
  --benchmark_caps "${BENCHMARK_CAPS}"
  --stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"
  --stage0_candidate_encode_batch_size "${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE}"
  --stage0_candidate_progress_interval_batches "${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}"
  --formulas "${FORMULAS}"
  --lambda_min "${LAMBDA_MIN}"
  --lambda_max "${LAMBDA_MAX}"
  --margin_threshold "${MARGIN_THRESHOLD}"
  --entropy_low "${ENTROPY_LOW}"
  --entropy_high "${ENTROPY_HIGH}"
)

if [[ -n "${ALLOWED_BENCHMARKS}" ]]; then
  read -r -a BENCHMARK_ARRAY <<< "${ALLOWED_BENCHMARKS}"
  ARGS+=(--allowed_benchmarks "${BENCHMARK_ARRAY[@]}")
fi
if [[ -n "${SKILL_TEXT_FORMAT:-}" ]]; then
  ARGS+=(--skill_text_format "${SKILL_TEXT_FORMAT}")
fi

"${PYTHON_BIN}" "${ARGS[@]}" | tee "${OUTPUT_DIR}/gate_sweep_stdout.json"
