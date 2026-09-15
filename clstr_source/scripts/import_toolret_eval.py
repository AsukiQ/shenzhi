#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolret_eval_import import import_toolret_eval


def _optional_list(values: list[str] | None) -> list[str] | None:
    return values if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize ToolRet official eval queries/tools into CLSTR JSONL files."
    )
    parser.add_argument("--query_source", default=None)
    parser.add_argument("--tool_source", default=None)
    parser.add_argument("--output_dir", default="data/toolret_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_tools", type=int)
    args = parser.parse_args()

    manifest = import_toolret_eval(
        query_source=args.query_source,
        tool_source=args.tool_source,
        output_dir=args.output_dir,
        split=args.split,
        tasks=_optional_list(args.tasks),
        categories=_optional_list(args.categories),
        max_queries=args.max_queries,
        max_tools=args.max_tools,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
