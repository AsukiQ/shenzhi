#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.leakage import run_leakage_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage0 no-leakage audit artifact generation.")
    parser.add_argument("--skillrouter_eval_root", default="data/skillrouter_eval_core")
    parser.add_argument("--skillsbench_root", default="/root/autodl-tmp/skillsbench")
    parser.add_argument("--output_dir", default="outputs/leakage_audit")
    args = parser.parse_args()

    report = run_leakage_audit(
        skillrouter_eval_root=Path(args.skillrouter_eval_root),
        skillsbench_root=Path(args.skillsbench_root),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
