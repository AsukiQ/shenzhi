#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_transition_row_diag}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT_PATH=${STAGE2_CHECKPOINT_PATH:-outputs/clstr_unified_stage2_v3_mixed_top350_inject_full/checkpoints/clstr_full_base-step10000.pt}
TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
TOP_M=${TOP_M:-350}
TOP_K=${TOP_K:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000}
STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=${STAGE0_CANDIDATE_ENCODE_BATCH_SIZE:-16}
STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-100}
TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.25}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}

CMD=(
  "$PYTHON_BIN"
  scripts/run_clstr_stage2_transition_row_diagnostics.py
  --stage0_checkpoint_path "$STAGE0_CHECKPOINT_PATH"
  --stage2_checkpoint_path "$STAGE2_CHECKPOINT_PATH"
  --train_path "$TRAIN_PATH"
  --skills_path "$SKILLS_PATH"
  --output_dir "$OUTPUT_DIR"
  --top_m "$TOP_M"
  --top_k "$TOP_K"
  --batch_size "$BATCH_SIZE"
  --embedding_cache_mode "$EMBEDDING_CACHE_MODE"
  --embedding_cache_max_rows "$EMBEDDING_CACHE_MAX_ROWS"
  --stage0_candidate_encode_batch_size "$STAGE0_CANDIDATE_ENCODE_BATCH_SIZE"
  --stage0_candidate_progress_interval_batches "$STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES"
  --transition_residual_lambda "$TRANSITION_RESIDUAL_LAMBDA"
  --transition_scoring_mode "$TRANSITION_SCORING_MODE"
)

if [[ -n "${MAX_ROWS:-}" ]]; then
  CMD+=(--max_rows "$MAX_ROWS")
fi
if [[ -n "${MAX_DIAGNOSTIC_ROWS:-}" ]]; then
  CMD+=(--max_diagnostic_rows "$MAX_DIAGNOSTIC_ROWS")
fi
if [[ -n "${ALLOWED_BENCHMARKS:-}" ]]; then
  read -r -a BENCHMARK_ARRAY <<< "$ALLOWED_BENCHMARKS"
  CMD+=(--allowed_benchmarks "${BENCHMARK_ARRAY[@]}")
fi
if [[ -n "${BENCHMARK_CAPS:-}" ]]; then
  CMD+=(--benchmark_caps "$BENCHMARK_CAPS")
fi
if [[ -n "${SKILL_TEXT_FORMAT:-}" ]]; then
  CMD+=(--skill_text_format "$SKILL_TEXT_FORMAT")
fi
if [[ -n "${STAGE0_HANDOFF_QUERY_MODE:-}" ]]; then
  CMD+=(--stage0_handoff_query_mode "$STAGE0_HANDOFF_QUERY_MODE")
fi

mkdir -p "${OUTPUT_DIR}"
"${CMD[@]}" | tee "${OUTPUT_DIR}/transition_row_diagnostics_stdout.json"
