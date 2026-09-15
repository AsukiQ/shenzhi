#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage4_validation import finalize_stage4_selection


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize validation-selected CLSTR Stage4 reliability."
    )
    parser.add_argument("--dynamic_selection_path", required=True)
    parser.add_argument("--gate_report_path", default=None)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    payload = finalize_stage4_selection(
        dynamic_selection_path=args.dynamic_selection_path,
        gate_report_path=args.gate_report_path,
        output_path=args.output_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
