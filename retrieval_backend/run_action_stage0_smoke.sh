#!/usr/bin/env bash
set -euo pipefail

CLSTR_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
SHENZHI_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
DATA_ROOT="$SHENZHI_ROOT/derived/action_vnext_stage0_v1"
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/envs/xzf/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/data/home/scyb713/run/miniconda3/envs/xzf/bin/python
fi
MODEL_PATH=${MODEL_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B}
OUTPUT_DIR=${OUTPUT_DIR:-$SHENZHI_ROOT/outputs/action_vnext_stage0_skillrouter_smoke}

for required in "$MODEL_PATH/config.json" "$DATA_ROOT/data_contract.json" "$DATA_ROOT/skills.jsonl"; do
  [[ -f "$required" ]] || { echo "missing Stage0 input: $required" >&2; exit 2; }
done

cd "$CLSTR_SOURCE"
"$PYTHON_BIN" scripts/run_clstr_vnext_stage0_train.py \
  --skills_path "$DATA_ROOT/skills.jsonl" \
  --retrieval_rows_path "$DATA_ROOT/retrieval_train.jsonl" \
  --retrieval_dev_rows_path "$DATA_ROOT/retrieval_dev.jsonl" \
  --static_route_rows_path "$DATA_ROOT/static_route_train.jsonl" \
  --static_route_dev_rows_path "$DATA_ROOT/static_route_dev.jsonl" \
  --inventory_catalogs_path "$DATA_ROOT/inventory_catalogs.jsonl" \
  --data_contract_path "$DATA_ROOT/data_contract.json" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps 20 \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --model_dim 1024 \
  --max_skills 515 \
  --max_retrieval_rows 4096 \
  --max_static_rows 4096 \
  --max_dev_rows_per_kind 1024 \
  --learning_rate 2e-5 \
  --torch_dtype bfloat16 \
  --max_length 1024 \
  --checkpoint_interval 20 \
  --validation_interval 20

cd "$SHENZHI_ROOT"
"$PYTHON_BIN" retrieval_backend/verify_stage0_artifacts.py \
  --output-dir "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/verification_report.json"
