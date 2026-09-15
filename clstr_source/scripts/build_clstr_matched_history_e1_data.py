#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_data import (
    MATCHED_HISTORY_BENCHMARKS,
    MATCHED_HISTORY_DATA_SCHEMA,
    MatchedHistoryTrajectorySkip,
    iter_selected_trajectories,
    prepare_matched_history_trajectory,
)
from clstr.vnext_training import file_sha256, load_inventory_catalogs


def _read_skill_ids(path: Path) -> set[str]:
    output: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            skill_id = str(
                row.get("skill_id")
                or row.get("canonical_skill_id")
                or row.get("id")
                or ""
            ).strip()
            if not skill_id or skill_id in output:
                raise ValueError("selected skill file has an empty or duplicate identity")
            output.add(skill_id)
    if not output:
        raise ValueError("selected skill file is empty")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_rows_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--report_path", required=True)
    args = parser.parse_args()

    source_path = Path(args.trajectory_rows_path).resolve()
    skills_path = Path(args.skills_path).resolve()
    catalogs_path = Path(args.inventory_catalogs_path).resolve()
    output_path = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    for path in (source_path, skills_path, catalogs_path):
        if not path.is_file():
            raise ValueError(f"matched-history input does not exist: {path}")
    if output_path == source_path or report_path in {source_path, skills_path, catalogs_path}:
        raise ValueError("matched-history outputs may not overwrite immutable inputs")

    selected_skill_ids = _read_skill_ids(skills_path)
    catalogs = load_inventory_catalogs(catalogs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_rows: Counter[str] = Counter()
    benchmark_trajectories: Counter[str] = Counter()
    quarantined: Counter[str] = Counter()
    skipped_trajectories: Counter[str] = Counter()
    skipped_rows: Counter[str] = Counter()
    trajectory_count = 0
    row_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for raw_trajectory in iter_selected_trajectories(source_path):
            try:
                compact, trajectory_quarantine = prepare_matched_history_trajectory(
                    raw_trajectory,
                    selected_skill_ids=selected_skill_ids,
                    catalogs=catalogs,
                )
            except MatchedHistoryTrajectorySkip as error:
                skipped_trajectories[error.reason] += 1
                skipped_rows[error.reason] += len(raw_trajectory)
                continue
            benchmark = str(compact[0]["benchmark"])
            trajectory_count += 1
            row_count += len(compact)
            benchmark_trajectories[benchmark] += 1
            benchmark_rows[benchmark] += len(compact)
            quarantined.update(trajectory_quarantine)
            for row in compact:
                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    report = {
        "schema_version": MATCHED_HISTORY_DATA_SCHEMA,
        "status": "ok",
        "benchmarks": sorted(MATCHED_HISTORY_BENCHMARKS),
        "trajectory_count": trajectory_count,
        "row_count": row_count,
        "trajectory_count_by_benchmark": dict(sorted(benchmark_trajectories.items())),
        "row_count_by_benchmark": dict(sorted(benchmark_rows.items())),
        "result_quarantine_by_reason": dict(sorted(quarantined.items())),
        "skipped_trajectory_count_by_reason": dict(
            sorted(skipped_trajectories.items())
        ),
        "skipped_row_count_by_reason": dict(sorted(skipped_rows.items())),
        "inputs": {
            "trajectory_rows_path": str(source_path),
            "trajectory_rows_sha256": file_sha256(source_path),
            "skills_path": str(skills_path),
            "skills_sha256": file_sha256(skills_path),
            "inventory_catalogs_path": str(catalogs_path),
            "inventory_catalogs_sha256": file_sha256(catalogs_path),
        },
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "leakage_contract": {
            "representation_prefix_uses_rows_strictly_before_decision": True,
            "current_action_is_output_only_and_never_an_input_event": True,
            "event_states_are_joined_from_the_corresponding_earlier_row": True,
            "persisted_prefix_count_digest_and_content_verified": True,
            "maximum_horizon_applied_by_the_training_collator": 16,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
