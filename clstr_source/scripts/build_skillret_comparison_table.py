#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillret_official import write_official_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKILLRET official-protocol comparison table.")
    parser.add_argument("--eval_root", default="outputs/skillret_official_eval")
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--output_table_path", default="outputs/skillret_official_eval/comparison_table.md")
    parser.add_argument("--output_summary_path", default="outputs/skillret_official_eval/comparison_summary.json")
    args = parser.parse_args()
    summary = write_official_comparison_table(
        eval_root=args.eval_root,
        run_names=args.run_names,
        output_table_path=args.output_table_path,
        output_summary_path=args.output_summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
