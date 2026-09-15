#!/usr/bin/env bash
set -euo pipefail

CLSTR_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
SHENZHI_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
STAGE0_OUTPUT=${STAGE0_OUTPUT:-$SHENZHI_ROOT/outputs/action_vnext_stage0_skillrouter_full}
STAGE2_DATA=${STAGE2_DATA:-$SHENZHI_ROOT/derived/action_vnext_stage2_v1}
STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-$STAGE0_OUTPUT/checkpoints/clstr_vnext_stage0-step1000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$SHENZHI_ROOT/outputs/action_vnext_stage2_skillrouter_full}
MAX_STEPS=${MAX_STEPS:-1000}

for required in "$STAGE0_CHECKPOINT" "$STAGE0_OUTPUT/selected_skills.jsonl" "$STAGE2_DATA/data_contract.json"; do
  [[ -f "$required" ]] || { echo "missing Stage2 input: $required" >&2; exit 2; }
done

cd "$CLSTR_SOURCE"
"$PYTHON_BIN" scripts/run_clstr_vnext_stage2_train.py \
  --stage0_checkpoint_path "$STAGE0_CHECKPOINT" \
  --skills_path "$STAGE0_OUTPUT/selected_skills.jsonl" \
  --trajectory_rows_path "$STAGE2_DATA/trajectory_train.jsonl" \
  --trajectory_dev_rows_path "$STAGE2_DATA/trajectory_dev.jsonl" \
  --causal_pair_support_rows_path "$STAGE2_DATA/causal_pair_support_train.jsonl" \
  --causal_pair_support_dev_rows_path "$STAGE2_DATA/causal_pair_support_dev.jsonl" \
  --inventory_catalogs_path "$SHENZHI_ROOT/derived/action_vnext_stage0_v1/inventory_catalogs.jsonl" \
  --data_contract_path "$STAGE2_DATA/data_contract.json" \
  --causal_branch_pairs_path "$STAGE2_DATA/causal_branch_pairs_train.jsonl" \
  --causal_branch_dev_pairs_path "$STAGE2_DATA/causal_branch_pairs_dev.jsonl" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  --curriculum_total_steps "$MAX_STEPS" \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_horizon 4 \
  --coarse_k 500 \
  --compressed_m 64 \
  --final_k 100 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --lambda_recall 0.2 \
  --lambda_compression 0.2 \
  --lambda_safety 0.05 \
  --lambda_raw_route 0.5 \
  --lambda_route_topk 0.0 \
  --lambda_mixture 0.2 \
  --lambda_history 0.2 \
  --lambda_order 0.0 \
  --lambda_result 0.0 \
  --checkpoint_interval 200 \
  --validation_interval 200 \
  --max_dev_pairs_per_kind 24 \
  --max_ordinary_dev_rows 192 \
  --ordinary_dev_batch_size 32 \
  --minimum_dev_clusters_per_enabled_kind 10 \
  --minimum_dev_clusters_per_source 1 \
  --minimum_ordinary_dev_clusters_per_stratum 2 \
  --minimum_ordinary_dev_stratum_coverage 0.5 \
  --minimum_full_pool_clusters 10 \
  --minimum_dev_score_gain 0.0 \
  --lambda_anchor 1e-4 \
  --seed 23
