#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.traject_eval_import import import_traject_eval


def _optional_list(values: list[str] | None) -> list[str] | None:
    return values if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize TRAJECT-Bench public_data into CLSTR retrieval-eval JSONL files."
    )
    parser.add_argument("--public_data", required=True)
    parser.add_argument("--output_dir", default="data/traject_eval_traject_split_test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split_partition", choices=["train", "dev", "test"])
    parser.add_argument("--trajectory_types", nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--max_queries", type=int)
    args = parser.parse_args()

    manifest = import_traject_eval(
        public_data=args.public_data,
        output_dir=args.output_dir,
        split=args.split,
        split_partition=args.split_partition,
        trajectory_types=_optional_list(args.trajectory_types),
        domains=_optional_list(args.domains),
        max_queries=args.max_queries,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
