#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.frozen_clstr_route_eval import run_frozen_clstr_route_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one immutable Qwen 0.6B CLSTR checkpoint chain on a frozen route corpus."
    )
    parser.add_argument("--final_chain_manifest_path", required=True)
    parser.add_argument("--benchmark_manifest_path", required=True)
    parser.add_argument("--source_rows_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_benchmark_manifest_sha256")
    parser.add_argument("--expected_final_chain_manifest_sha256")
    parser.add_argument("--expected_checkpoint_chain_digest")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--device")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return run_frozen_clstr_route_eval(
        final_chain_manifest_path=args.final_chain_manifest_path,
        benchmark_manifest_path=args.benchmark_manifest_path,
        source_rows_path=args.source_rows_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_eval_rows=args.max_eval_rows,
        device=args.device,
        expected_benchmark_manifest_sha256=args.expected_benchmark_manifest_sha256,
        expected_final_chain_manifest_sha256=args.expected_final_chain_manifest_sha256,
        expected_checkpoint_chain_digest=args.expected_checkpoint_chain_digest,
    )


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
