from __future__ import annotations

import json
from pathlib import Path

from clstr.clean_training_preflight import _row_sha256
from clstr.protected_near_duplicate_filter import filter_protected_near_duplicates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_filter_removes_matched_query_and_complete_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    trajectory_rows = [
        {"trajectory_id": "remove-t", "task_id": "remove-t::0", "state_text": "x"},
        {"trajectory_id": "remove-t", "task_id": "remove-t::1", "state_text": "y"},
        {"trajectory_id": "keep-t", "task_id": "keep-t::0", "state_text": "z"},
    ]
    retrieval_rows = [
        {"query_id": "remove-q", "query_text": "x"},
        {"query_id": "keep-q", "query_text": "z"},
    ]
    _write_jsonl(source / "trajectories.jsonl", trajectory_rows)
    _write_jsonl(source / "retrieval.jsonl", retrieval_rows)
    _write_jsonl(source / "skill_pool.jsonl", [{"skill_id": "s"}])
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "error",
                "preflight_contract": {"protected_near_duplicate_blocking": True},
                "leakage_policy": {
                    "has_exact_leakage": False,
                    "has_near_duplicate": True,
                },
                "leakage": {
                    "fixture": {
                        "trajectory_content_overlap": {
                            "matched_train_rows": [
                                {
                                    "train_row_sha256": _row_sha256(trajectory_rows[0]),
                                    "trajectory_id": "remove-t",
                                    "query_id": "",
                                }
                            ]
                        },
                        "retrieval_content_overlap": {
                            "matched_train_rows": [
                                {
                                    "train_row_sha256": _row_sha256(retrieval_rows[0]),
                                    "trajectory_id": "",
                                    "query_id": "remove-q",
                                }
                            ]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "filtered"
    result = filter_protected_near_duplicates(
        input_root=source,
        preflight_report_path=report_path,
        output_root=output,
    )
    assert result["trajectories"]["removed_rows"] == 2
    assert result["retrieval"]["removed_rows"] == 1
    assert [json.loads(line)["trajectory_id"] for line in (output / "trajectories.jsonl").read_text().splitlines()] == ["keep-t"]
    assert [json.loads(line)["query_id"] for line in (output / "retrieval.jsonl").read_text().splitlines()] == ["keep-q"]
