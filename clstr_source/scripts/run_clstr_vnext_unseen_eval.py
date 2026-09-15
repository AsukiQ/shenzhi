#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_eval import run_vnext_unseen_skill_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate append-only retrieval of CLSTR vNext held-out skills",
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--top_m", type=int, default=100)
    parser.add_argument("--score_batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--frozen_cache_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_vnext_unseen_skill_eval(
        checkpoint_path=args.checkpoint_path,
        data_manifest_path=args.data_manifest_path,
        output_dir=args.output_dir,
        max_eval_rows=args.max_eval_rows,
        top_m=args.top_m,
        score_batch_size=args.score_batch_size,
        cache_batch_size=args.cache_batch_size,
        belief_top_k=args.belief_top_k,
        frozen_cache_dir=args.frozen_cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
