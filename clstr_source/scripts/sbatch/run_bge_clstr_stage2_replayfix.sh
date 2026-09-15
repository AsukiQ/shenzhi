#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"

TS=${TS:-$(date +%Y%m%d_%H%M%S)}
export TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl}
export SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/bge_clstr_stage2_toolbench_clean_frozen_b64_unified_memory_replayfix_${TS}}
export ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/bge_clstr_stage0_toolbench_clean_full_20260709_144443/checkpoints/latest.pt}
export STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/bge_clstr_stage1_toolbench_clean_frozen_b64_unified_memory_20260709_171129/checkpoints/clstr_stage1_heads-step3000.pt}
export STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/bge_clstr_stage0_toolbench_clean_full_20260709_144443}
export STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-outputs/bge_clstr_stage0_toolbench_clean_full_20260709_144443/train_report.json}
export STAGE0_BASELINE_METRICS_PATH=${STAGE0_BASELINE_METRICS_PATH:-outputs/bge_clstr_stage0_toolbench_clean_full_20260709_144443/train_report.json}
export STAGE0_QUALITY_GATE_MODE=${STAGE0_QUALITY_GATE_MODE:-strict}
export MODEL_DIM=${MODEL_DIM:-1024}
export MAX_STEPS=${MAX_STEPS:-3000}
export BATCH_SIZE=${BATCH_SIZE:-64}
export ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}
export TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}
export TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}
export STAGE0_TOP_M=${STAGE0_TOP_M:-500}
export STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}
export STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-auto}
export STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}
export SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}
export EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}
export TRANSITION_MEMORY_MARGIN_LOSS_WEIGHT=${TRANSITION_MEMORY_MARGIN_LOSS_WEIGHT:-0.05}

echo "[bge-clstr-stage2-replayfix] output_dir=${OUTPUT_DIR}"
exec bash scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
