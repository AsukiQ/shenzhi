#!/usr/bin/env bash
set -euo pipefail

ROOT=${SHENZHI_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/shenzhi}
PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}

cd "$ROOT/retrieval_backend"
for shell_script in ./*.sh; do
  bash -n "$shell_script"
done
"$PYTHON_BIN" -m unittest discover -v -p 'test_*.py'
PYTHONPATH="$ROOT/paper_data_pipeline/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m unittest discover -v \
  -s "$ROOT/paper_data_pipeline/tests" -p 'test_*.py'
"$PYTHON_BIN" verify_stage0_artifacts.py \
  --output-dir "$ROOT/outputs/paper_vnext_stage0_skillrouter_full" \
  --report "$ROOT/outputs/paper_vnext_stage0_skillrouter_full/verification_report.json" \
  >/dev/null
"$PYTHON_BIN" validate_qrels.py \
  --qrels "$ROOT/evaluation/human_qrels_annotation_v1.jsonl" \
  --output "$ROOT/outputs/human_qrels_status.json" \
  >/dev/null
"$PYTHON_BIN" build_release_manifest.py \
  --root "$ROOT" \
  --output "$ROOT/outputs/release_manifest.json" \
  >/dev/null
printf 'release verification passed: %s\n' "$ROOT/outputs/release_manifest.json"
