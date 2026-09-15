#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

TRAIN_PATH=${TRAIN_PATH:-data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak/trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage2_v4_2_nowweak_top350_inventory_listwise}
ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE0_OUTPUT_DIR=${STAGE0_OUTPUT_DIR:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}
STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_unified_stage1_v4_2_nowweak_top350_inventory_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt}
STAGE0_TOP_M=${STAGE0_TOP_M:-350}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.2}
POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}
Q_SUCCESS_LOSS_WEIGHT=${Q_SUCCESS_LOSS_WEIGHT:-0.0}
TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}
TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-1.0}
TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}
TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}
TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}
TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}
BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}

export TRAIN_PATH SKILLS_PATH OUTPUT_DIR ROUTING_CHECKPOINT_PATH
export STAGE0_OUTPUT_DIR STAGE0_EVAL_METRICS_PATH STAGE1_CHECKPOINT_PATH STAGE0_TOP_M
export POLICY_LOSS_WEIGHT POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT Q_SUCCESS_LOSS_WEIGHT
export TRANSITION_LOSS_WEIGHT TRANSITION_SKILL_CE_LOSS_WEIGHT
export TRANSITION_INVENTORY_MASK_MODE TRANSITION_INVENTORY_MIN_CANDIDATES
export TRANSITION_LOSS_TYPE TRANSITION_POSITIVE_MODE
export BENCHMARK_CAPS SAMPLING_STRATEGY

exec bash scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
