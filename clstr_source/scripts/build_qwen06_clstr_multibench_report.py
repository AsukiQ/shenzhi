#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_clstr_multibench_report import (  # noqa: E402
    build_qwen_clstr_multibench_report,
)
from clstr.qwen_clstr_multibench_submit import (  # noqa: E402
    build_qwen_clstr_multibench_stages,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation config must contain an object: {source}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate one Qwen 0.6B CLSTR multibench run."
    )
    parser.add_argument("--evaluation_config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--markdown_path")
    parser.add_argument("--job_evidence_path")
    parser.add_argument("--scope", required=True, choices=["smoke", "full"])
    parser.add_argument("--partition", default="gpu_a800,gpu_h100,gpu_h200")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.evaluation_config_path)
    stages = build_qwen_clstr_multibench_stages(
        config,
        scope=args.scope,
        partition=args.partition,
    )
    return build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=args.output_path,
        markdown_path=args.markdown_path,
        job_evidence_path=args.job_evidence_path,
    )


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
