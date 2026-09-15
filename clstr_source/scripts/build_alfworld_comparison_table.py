#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import build_alfworld_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ALFWorld closed-loop comparison table.")
    parser.add_argument("--eval_root", default="outputs/alfworld_eval")
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--output_table_path", default="outputs/alfworld_eval/comparison_table.md")
    parser.add_argument("--output_summary_path", default="outputs/alfworld_eval/comparison_summary.json")
    args = parser.parse_args()
    summary = build_alfworld_comparison_table(
        eval_root=Path(args.eval_root),
        run_names=args.run_names,
        output_table_path=Path(args.output_table_path),
        output_summary_path=Path(args.output_summary_path),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
