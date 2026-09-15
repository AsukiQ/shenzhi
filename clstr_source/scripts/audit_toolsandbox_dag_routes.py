#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ToolSandbox DAG-aware route extraction."
    )
    parser.add_argument("--scenarios_root", required=True)
    parser.add_argument("--tools_root", required=True)
    parser.add_argument("--maximum_route_variants", type=int, default=64)
    parser.add_argument("--output_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_toolsandbox_route_corpus(
        scenarios_root=args.scenarios_root,
        tools_root=args.tools_root,
        maximum_route_variants=args.maximum_route_variants,
    )
    payload = json.dumps(corpus.report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
