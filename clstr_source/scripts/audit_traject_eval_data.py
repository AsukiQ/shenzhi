#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.traject_eval_audit import audit_traject_eval_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit normalized TRAJECT-Bench eval data before CLSTR retrieval evaluation."
    )
    parser.add_argument("--data_dir", default="data/traject_eval_traject_split_test")
    parser.add_argument("--output_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_traject_eval_data(
        data_dir=args.data_dir,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
