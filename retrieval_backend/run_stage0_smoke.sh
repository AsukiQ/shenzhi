#!/usr/bin/env bash
set -euo pipefail

CLSTR_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
SHENZHI_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
MODEL_PATH=${MODEL_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "SkillRouter model not found: $MODEL_PATH" >&2
  echo "Set MODEL_PATH to a local SkillRouter-Embedding-0.6B directory." >&2
  exit 2
fi

DATA_ROOT="$SHENZHI_ROOT/derived/paper_vnext_stage0_v1"
if [[ ! -f "$DATA_ROOT/data_contract.json" ]]; then
  echo "vNext Stage0 data not found: $DATA_ROOT/data_contract.json" >&2
  echo "Run retrieval_backend/build_vnext_stage0_paper_data.py first." >&2
  exit 2
fi

cd "$CLSTR_SOURCE"
"$PYTHON_BIN" scripts/run_clstr_vnext_stage0_train.py \
  --skills_path "$DATA_ROOT/skills.jsonl" \
  --retrieval_rows_path "$DATA_ROOT/retrieval_train.jsonl" \
  --retrieval_dev_rows_path "$DATA_ROOT/retrieval_dev.jsonl" \
  --static_route_rows_path "$DATA_ROOT/static_route_train.jsonl" \
  --static_route_dev_rows_path "$DATA_ROOT/static_route_dev.jsonl" \
  --inventory_catalogs_path "$DATA_ROOT/inventory_catalogs.jsonl" \
  --data_contract_path "$DATA_ROOT/data_contract.json" \
  --output_dir "$SHENZHI_ROOT/outputs/paper_vnext_stage0_skillrouter_smoke" \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps 20 \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --model_dim 1024 \
  --max_skills 2000 \
  --max_retrieval_rows 512 \
  --max_static_rows 512 \
  --max_dev_rows_per_kind 256 \
  --learning_rate 2e-5 \
  --torch_dtype bfloat16 \
  --max_length 1024 \
  --checkpoint_interval 20 \
  --validation_interval 20

cd "$SHENZHI_ROOT"
"$PYTHON_BIN" retrieval_backend/verify_stage0_artifacts.py \
  --output-dir outputs/paper_vnext_stage0_skillrouter_smoke \
  --report outputs/paper_vnext_stage0_skillrouter_smoke/verification_report.json
