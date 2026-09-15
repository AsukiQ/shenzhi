#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_multistep import build_multistep_failure_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AppWorld multi-step runs and report reference-success/focus-failure cases.")
    parser.add_argument("--focus_runs_path", required=True)
    parser.add_argument("--reference_runs_path", required=True)
    parser.add_argument("--output_dir", default="outputs/appworld_multistep_failure_diagnostics")
    parser.add_argument("--focus_name", default="focus")
    parser.add_argument("--reference_name", default="reference")
    args = parser.parse_args()
    payload = build_multistep_failure_diagnostics(
        focus_runs_path=args.focus_runs_path,
        reference_runs_path=args.reference_runs_path,
        output_dir=args.output_dir,
        focus_name=args.focus_name,
        reference_name=args.reference_name,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
