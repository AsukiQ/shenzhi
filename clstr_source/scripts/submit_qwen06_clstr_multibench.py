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

from clstr.qwen_clstr_multibench_submit import (
    build_qwen_clstr_multibench_stages,
    submit_qwen_clstr_multibench_stage_plan,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation config must contain an object: {source}")
    return payload


def run_sbatch(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Slurm submission failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or submit the Qwen 0.6B CLSTR four-benchmark evaluation plan."
    )
    parser.add_argument("--evaluation_config_path", required=True)
    parser.add_argument("--registry_path", required=True)
    parser.add_argument("--scope", required=True, choices=["smoke", "full"])
    parser.add_argument("--partition", default="gpu_a800,gpu_h100,gpu_h200")
    parser.add_argument("--resume_submission", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.evaluation_config_path)
    stages = build_qwen_clstr_multibench_stages(
        config,
        scope=args.scope,
        partition=args.partition,
    )
    return submit_qwen_clstr_multibench_stage_plan(
        stages=stages,
        registry_path=args.registry_path,
        submitter=run_sbatch,
        resume=bool(args.resume_submission),
        dry_run=bool(args.dry_run),
    )


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"dry_run", "submitted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
