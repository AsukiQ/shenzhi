#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_clstr_multibench_jobs import (  # noqa: E402
    collect_qwen_clstr_multibench_job_evidence,
)


def run_sacct(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Slurm accounting query failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return completed.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect terminal Slurm evidence for a Qwen CLSTR multibench submission."
    )
    parser.add_argument("--registry_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return collect_qwen_clstr_multibench_job_evidence(
        registry_path=args.registry_path,
        output_path=args.output_path,
        sacct_runner=run_sacct,
    )


def main() -> int:
    evidence = run_from_args(build_parser().parse_args())
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
