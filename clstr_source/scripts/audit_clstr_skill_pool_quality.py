#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skill_pool_quality_audit import audit_clstr_skill_pool_quality


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CLSTR unified canonical skill pool and retrieval qrels before Stage0 training."
    )
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final")
    parser.add_argument("--output_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_clstr_skill_pool_quality(
        data_root=args.data_root,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
