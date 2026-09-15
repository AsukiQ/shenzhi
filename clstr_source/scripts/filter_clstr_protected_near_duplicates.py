#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.protected_near_duplicate_filter import filter_protected_near_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter protected-evaluation near duplicates from unified CLSTR data",
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--preflight_report_path", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    report = filter_protected_near_duplicates(
        input_root=args.input_root,
        preflight_report_path=args.preflight_report_path,
        output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
