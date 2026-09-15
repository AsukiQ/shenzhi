#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_virtual_api import audit_stabletoolbench_virtual_api


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit StableToolBench virtual API server prerequisites.")
    parser.add_argument("--stabletoolbench_root", required=True)
    parser.add_argument("--mode", default="mirrorapi", choices=["mirrorapi", "mirrorapi_cache", "gpt_cache"])
    parser.add_argument("--autodl_tmp_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp")
    parser.add_argument("--output_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stabletoolbench_virtual_api(
        stabletoolbench_root=args.stabletoolbench_root,
        mode=args.mode,
        autodl_tmp_root=args.autodl_tmp_root,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
