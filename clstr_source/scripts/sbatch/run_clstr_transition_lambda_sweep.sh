#!/bin/bash
#SBATCH --time=01:00:00
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_transition_lambda_sweep/stage1_inventory100_l025}
LAMBDAS=${LAMBDAS:-0,0.25,0.5,1.0,2.0}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ROWS=${MAX_ROWS:-512}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-64}
SEED=${SEED:-17}
ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-never}
EMBEDDING_CACHE_ENCODE_BATCH_SIZE=${EMBEDDING_CACHE_ENCODE_BATCH_SIZE:-16}

ARGS=(
  scripts/evaluate_clstr_transition_lambda_sweep.py
  --train_path "${TRAIN_PATH}"
  --skills_path "${SKILLS_PATH}"
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT_PATH}"
  --stage1_checkpoint_path "${STAGE1_CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --lambdas "${LAMBDAS}"
  --batch_size "${BATCH_SIZE}"
  --max_rows "${MAX_ROWS}"
  --max_eval_batches "${MAX_EVAL_BATCHES}"
  --seed "${SEED}"
  --benchmark_caps "${BENCHMARK_CAPS}"
  --sampling_strategy "${SAMPLING_STRATEGY}"
  --stage0_top_m "${STAGE0_TOP_M}"
  --transition_inventory_mask_mode "${TRANSITION_INVENTORY_MASK_MODE}"
  --transition_inventory_min_candidates "${TRANSITION_INVENTORY_MIN_CANDIDATES}"
  --transition_loss_type "${TRANSITION_LOSS_TYPE}"
  --transition_positive_mode "${TRANSITION_POSITIVE_MODE}"
  --transition_scoring_mode "${TRANSITION_SCORING_MODE}"
  --embedding_cache_mode "${EMBEDDING_CACHE_MODE}"
  --embedding_cache_encode_batch_size "${EMBEDDING_CACHE_ENCODE_BATCH_SIZE}"
)

if [[ -n "${ALLOWED_BENCHMARKS}" ]]; then
  ARGS+=(--allowed_benchmarks "${ALLOWED_BENCHMARKS}")
fi
if [[ -n "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}" ]]; then
  ARGS+=(--stage0_handoff_sample_multiplier "${STAGE0_HANDOFF_SAMPLE_MULTIPLIER}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
