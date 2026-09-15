from __future__ import annotations

import argparse
import json
from pathlib import Path

from clstr.qwen_clstr_eval_batch_parity import (
    SUPPORTED_BATCH_PARITY_KINDS,
    compare_qwen_clstr_eval_batches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fallback and accelerated CLSTR eval outputs.")
    parser.add_argument("--kind", choices=sorted(SUPPORTED_BATCH_PARITY_KINDS), required=True)
    parser.add_argument("--fallback_dir", required=True)
    parser.add_argument("--accelerated_dir", required=True)
    parser.add_argument("--output_path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_qwen_clstr_eval_batches(
        kind=args.kind,
        fallback_dir=args.fallback_dir,
        accelerated_dir=args.accelerated_dir,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
