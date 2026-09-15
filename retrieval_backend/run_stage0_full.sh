#!/usr/bin/env bash
set -euo pipefail

CLSTR_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-native-sync-smoke-source
SHENZHI_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi
DATA_ROOT="$SHENZHI_ROOT/derived/paper_vnext_stage0_v1"
MODEL_PATH=${MODEL_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-$SHENZHI_ROOT/outputs/paper_vnext_stage0_skillrouter_full}
MAX_STEPS=${MAX_STEPS:-1000}
AUTO_RESUME=${AUTO_RESUME:-1}

for required in \
  "$MODEL_PATH/config.json" \
  "$DATA_ROOT/data_contract.json" \
  "$DATA_ROOT/skills.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "required Stage0 input not found: $required" >&2
    exit 2
  fi
done

cd "$CLSTR_SOURCE"
train_args=(
  scripts/run_clstr_vnext_stage0_train.py
  --skills_path "$DATA_ROOT/skills.jsonl" \
  --retrieval_rows_path "$DATA_ROOT/retrieval_train.jsonl" \
  --retrieval_dev_rows_path "$DATA_ROOT/retrieval_dev.jsonl" \
  --static_route_rows_path "$DATA_ROOT/static_route_train.jsonl" \
  --static_route_dev_rows_path "$DATA_ROOT/static_route_dev.jsonl" \
  --inventory_catalogs_path "$DATA_ROOT/inventory_catalogs.jsonl" \
  --data_contract_path "$DATA_ROOT/data_contract.json" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --batch_size 128 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --model_dim 1024 \
  --max_length 1024 \
  --torch_dtype bfloat16 \
  --skill_table_batch_size 64 \
  --belief_top_k 64 \
  --cache_batch_size 128 \
  --cache_shard_size 4096 \
  --skill_cache_shard_size 2048 \
  --hard_negative_loss_weight 0.2 \
  --hard_negative_margin 0.1 \
  --hard_negative_top_k 32 \
  --checkpoint_interval 200 \
  --validation_interval 200 \
  --validation_batch_size 128 \
  --max_dev_rows_per_kind 4096 \
  --minimum_dev_score_gain 0.0
  --seed 17
)

completed=$(
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_steps = int(sys.argv[2])
report = root / "train_report.json"
if not report.is_file():
    print("0")
else:
    row = json.loads(report.read_text())
    print("1" if int(row.get("step") or -1) == max_steps else "0")
PY
)

if [[ "$completed" == "1" ]]; then
  echo "Stage0 max_steps=$MAX_STEPS is already complete; skipping training"
elif [[ "$AUTO_RESUME" == "1" ]]; then
  resume=$(
    "$PYTHON_BIN" - "$OUTPUT_DIR" "$MAX_STEPS" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "checkpoints"
max_steps = int(sys.argv[2])
candidates = []
for path in root.glob("clstr_vnext_stage0-step*.pt"):
    match = re.fullmatch(r"clstr_vnext_stage0-step(\d+)\.pt", path.name)
    if match and 0 < int(match.group(1)) < max_steps:
        candidates.append((int(match.group(1)), path.resolve()))
if candidates:
    print(max(candidates)[1])
PY
  )
  if [[ -n "$resume" ]]; then
    echo "resuming Stage0 from $resume"
    train_args+=(--resume_checkpoint_path "$resume")
  fi
  "$PYTHON_BIN" "${train_args[@]}"
else
  "$PYTHON_BIN" "${train_args[@]}"
fi

cd "$SHENZHI_ROOT"
"$PYTHON_BIN" retrieval_backend/verify_stage0_artifacts.py \
  --output-dir "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/verification_report.json"
