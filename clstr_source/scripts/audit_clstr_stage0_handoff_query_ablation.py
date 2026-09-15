#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage0_handoff_query_ablation import (  # noqa: E402
    QUERY_VARIANTS,
    audit_stage0_handoff_query_ablation_from_checkpoint,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_variants(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage0 next-skill coverage under handoff query variants.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--benchmark", default="toolbench_g3")
    parser.add_argument("--variants", default=",".join(QUERY_VARIANTS))
    parser.add_argument("--top_k_values", default="20,50,100,200,500")
    parser.add_argument("--max_rows", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_cache_dir", default=None)
    args = parser.parse_args()

    report = audit_stage0_handoff_query_ablation_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_path=args.output_path,
        benchmark=None if str(args.benchmark).lower() in {"all", "none", ""} else args.benchmark,
        variants=_parse_variants(args.variants),
        top_k_values=_parse_ints(args.top_k_values),
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        model_cache_dir=args.model_cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
