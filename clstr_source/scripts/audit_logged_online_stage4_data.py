#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.logged_online_trajectory import (
    TRAIN_SPLIT_NAMES,
    audit_logged_online_coverage,
    iter_logged_online_steps,
    load_skill_id_set,
)


def _parse_csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit executor-free logged-online Stage4 trajectory data.")
    parser.add_argument("--trajectories_path", required=True)
    parser.add_argument("--skill_pool_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--include_splits",
        default=",".join(sorted(TRAIN_SPLIT_NAMES)),
        help="Comma-separated split names to include.",
    )
    parser.add_argument("--max_source_rows", type=int, default=None)
    parser.add_argument(
        "--stage0_handoff_report_path",
        default=None,
        help="Optional existing Stage0 top-M handoff JSON report to summarize in the audit.",
    )
    parser.add_argument(
        "--steps_output_path",
        default=None,
        help="Optional JSONL path for normalized logged-online steps.",
    )
    args = parser.parse_args()

    include_splits = _parse_csv_set(args.include_splits)
    report = audit_logged_online_coverage(
        trajectories_path=args.trajectories_path,
        skill_pool_path=args.skill_pool_path,
        include_splits=include_splits,
        max_source_rows=args.max_source_rows,
        stage0_handoff_report_path=args.stage0_handoff_report_path,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.steps_output_path:
        skill_ids = load_skill_id_set(args.skill_pool_path)
        steps_output_path = Path(args.steps_output_path)
        steps_output_path.parent.mkdir(parents=True, exist_ok=True)
        with steps_output_path.open("w", encoding="utf-8") as handle:
            for step in iter_logged_online_steps(
                args.trajectories_path,
                skill_ids,
                include_splits=include_splits,
                max_rows=args.max_source_rows,
            ):
                handle.write(json.dumps(step, ensure_ascii=False) + "\n")

    print(json.dumps({"output_path": str(output_path), "emitted_steps": report["emitted_steps"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
