#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.apibank_route_eval import run_apibank_skillrouter_frozen_eval


def _csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SkillRouter-style bi-encoder on official APIBank-derived next-API routing rows."
    )
    parser.add_argument("--data_root", default=".tmp/benchmark_probe_direct/liminghao1630__API-Bank")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--files")
    parser.add_argument("--include_trivial", action="store_true")
    parser.add_argument("--max_rows_per_file", type=int)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--adapter_checkpoint_path")
    args = parser.parse_args()
    report = run_apibank_skillrouter_frozen_eval(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        files=_csv(args.files),
        include_trivial=args.include_trivial,
        max_rows_per_file=args.max_rows_per_file,
        max_eval_rows=args.max_eval_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
