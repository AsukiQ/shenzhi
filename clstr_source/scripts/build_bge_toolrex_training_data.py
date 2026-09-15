#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolrex_training_data import build_toolrex_unified_data


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_roots(value: str) -> list[Path]:
    roots = [Path(item.strip()) for item in str(value).split(",") if item.strip()]
    if not roots:
        raise ValueError("--source_roots must contain at least one path")
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ToolRex-style unified data for BGE Tool-Embed/Tool-Rank.")
    parser.add_argument(
        "--source_roots",
        default="data/toolret_training,data/toolbench_g3",
        help="Comma-separated source roots. Supports retrieval.jsonl+skills.jsonl and queries/qrels+skills formats.",
    )
    parser.add_argument("--output_dir", default="data/bge_toolrex_unified")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(
        progress_path,
        {
            "status": "running",
            "stage": "building_toolrex_unified_data",
            "updated_at_unix": time.time(),
            "source_roots": [str(path) for path in _parse_roots(args.source_roots)],
            "output_dir": str(output_dir),
        },
    )
    try:
        report = build_toolrex_unified_data(
            source_roots=_parse_roots(args.source_roots),
            output_dir=output_dir,
            max_rows=args.max_rows,
            seed=args.seed,
        )
    except Exception as exc:
        blocker = {
            "status": "blocked",
            "stage": "building_toolrex_unified_data",
            "reason": type(exc).__name__,
            "message": str(exc),
            "updated_at_unix": time.time(),
        }
        _write_json(output_dir / "blocker_report.json", blocker)
        _write_json(progress_path, blocker)
        raise
    metrics = {**report, "stage": "building_toolrex_unified_data"}
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        progress_path,
        {
            "status": "ok",
            "stage": "complete",
            "updated_at_unix": time.time(),
            "skill_count": report["skill_count"],
            "trajectory_count": report["trajectory_count"],
            "retrieval_row_count": report["retrieval_row_count"],
            "metrics_path": str(output_dir / "metrics.json"),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

