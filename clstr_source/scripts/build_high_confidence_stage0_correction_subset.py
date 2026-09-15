#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_stage0_correction_audit import build_high_confidence_stage0_correction_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Stage0 correction rows into a high-confidence training subset. "
            "The default policy keeps rows whose positive skill overlaps a required write/action API."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policy", default="write_only", choices=["write_only"])
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = build_high_confidence_stage0_correction_subset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        policy=args.policy,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
