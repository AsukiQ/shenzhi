#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_baseline_corpus import build_matched_baseline_corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the digest-bound matched SR/ToolREx corpus from an approved vNext manifest."
    )
    parser.add_argument("--vnext_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = build_matched_baseline_corpus(
        vnext_manifest_path=args.vnext_manifest_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
