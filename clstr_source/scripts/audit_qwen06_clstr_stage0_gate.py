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

from clstr.qwen_clstr_lineage import validate_lineage_manifest  # noqa: E402
from clstr.qwen_clstr_stage0_gate import (  # noqa: E402
    audit_stage0_promotion,
    audit_stage0_release,
    audit_stage0_smoke,
    select_stage0_checkpoint,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON report must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_candidate(value: str) -> dict[str, Any]:
    parts = str(value).split("=", 3)
    if len(parts) != 4:
        raise ValueError(
            "candidate must use step=report.json=checkpoint.pt=lineage.json format"
        )
    step, report_path, checkpoint_path, lineage_path = parts
    return {
        "step": int(step),
        "report": _read_json(report_path),
        "report_path": report_path,
        "checkpoint_path": checkpoint_path,
        "lineage_manifest_path": lineage_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Qwen CLSTR Stage0 continuation and release gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--train_report_path", required=True)
    smoke.add_argument("--lineage_manifest_path", required=True)
    smoke.add_argument("--output_path", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--baseline_report", required=True)
    promote.add_argument("--current_report", required=True)
    promote.add_argument("--previous_report", required=True)
    promote.add_argument("--target_step", type=int, required=True)
    promote.add_argument("--output_path", required=True)

    release = subparsers.add_parser("release")
    release.add_argument("--baseline_report", required=True)
    release.add_argument("--current_report", required=True)
    release.add_argument("--checkpoint_path", required=True)
    release.add_argument("--output_path", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--baseline_report", required=True)
    select.add_argument("--candidate", action="append", required=True)
    select.add_argument("--output_path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "smoke":
            lineage_report = validate_lineage_manifest(
                args.lineage_manifest_path,
                expected_stage="stage0",
                expected_parent_manifests={},
            )
            report = audit_stage0_smoke(
                _read_json(args.train_report_path),
                lineage_report,
            )
        elif args.command == "promote":
            report = audit_stage0_promotion(
                baseline_report=_read_json(args.baseline_report),
                current_report=_read_json(args.current_report),
                previous_report=_read_json(args.previous_report),
                target_step=args.target_step,
            )
        elif args.command == "release":
            report = audit_stage0_release(
                _read_json(args.baseline_report),
                _read_json(args.current_report),
                args.checkpoint_path,
            )
        else:
            report = select_stage0_checkpoint(
                baseline_report=_read_json(args.baseline_report),
                candidates=[_parse_candidate(value) for value in args.candidate],
            )
        _write_json(args.output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "ok" else 2
    except Exception as exc:
        report = {
            "status": "error",
            "gate": f"stage0_{args.command}",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(args.output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
