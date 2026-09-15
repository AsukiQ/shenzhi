from __future__ import annotations

import json
from pathlib import Path

from clstr.trajectbench_visible_inventory_eval import build_trajectbench_visible_inventory_eval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_trajectbench_visible_inventory_eval_removes_oracle_inventory(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skill_pool.jsonl"
    output = tmp_path / "traject_visible.jsonl"
    report_path = tmp_path / "report.json"

    _write_jsonl(
        skills,
        [
            {"skill_id": "traject/search-api/query", "name": "Search"},
            {"skill_id": "toolbench-g3/weather/current", "name": "Weather"},
            {"skill_id": "traject/maps-api/route", "name": "Route"},
        ],
    )
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "tr-1::0",
                "skill_id": "traject/search-api/query",
                "next_skill_id": "traject/maps-api/route",
                "tool_inventory_skill_ids": ["traject/search-api/query"],
                "visible_inventory_skill_ids": ["stale/oracle"],
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-1::0",
                "skill_id": "toolbench-g3/weather/current",
                "tool_inventory_skill_ids": ["toolbench-g3/weather/current"],
            },
        ],
    )

    report = build_trajectbench_visible_inventory_eval(
        trajectories_path=trajectories,
        skills_path=skills,
        output_path=output,
        report_path=report_path,
    )

    assert report["status"] == "ok"
    assert report["source_rows"] == 2
    assert report["written_rows"] == 1
    assert report["skipped_non_traject_rows"] == 1
    assert report["visible_inventory_count"] == 2
    assert report["removed_tool_inventory_rows"] == 1
    assert report["current_skill_visible_rows"] == 1
    assert report["next_skill_visible_rows"] == 1

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert "tool_inventory_skill_ids" not in rows[0]
    assert rows[0]["visible_inventory_skill_ids"] == [
        "traject/maps-api/route",
        "traject/search-api/query",
    ]
    assert rows[0]["provenance"]["visible_inventory_source"] == "skill_pool_traject_global"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_build_trajectbench_visible_inventory_eval_can_use_domain_local_inventory(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skill_pool.jsonl"
    output = tmp_path / "traject_visible.jsonl"

    _write_jsonl(
        skills,
        [
            {
                "skill_id": "traject/education/dictionary",
                "name": "Dictionary",
                "environment": "Education",
            },
            {
                "skill_id": "traject/education/kanji",
                "name": "Kanji",
                "source_files": ["/data/TRAJECT-Bench/public_data/sequential/Education/traj_query.json"],
            },
            {
                "skill_id": "traject/music/radio",
                "name": "Radio",
                "environment": "Music",
            },
        ],
    )
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "traject_bench",
                "trajectory_id": "traject::sequential::Education::traj_query::0",
                "skill_id": "traject/education/dictionary",
                "next_skill_id": "traject/education/kanji",
                "tool_inventory_skill_ids": ["traject/education/dictionary"],
                "provenance": {"domain": "Education"},
            }
        ],
    )

    report = build_trajectbench_visible_inventory_eval(
        trajectories_path=trajectories,
        skills_path=skills,
        output_path=output,
        visible_inventory_mode="domain_local",
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["visible_inventory_source"] == "skill_pool_traject_domain_local"
    assert report["visible_inventory_count_mean"] == 2
    assert report["current_skill_visible_rows"] == 1
    assert report["next_skill_visible_rows"] == 1
    assert "tool_inventory_skill_ids" not in rows[0]
    assert rows[0]["visible_inventory_skill_ids"] == [
        "traject/education/dictionary",
        "traject/education/kanji",
    ]
    assert rows[0]["provenance"]["visible_inventory_source"] == "skill_pool_traject_domain_local"
    assert rows[0]["provenance"]["visible_inventory_domain"] == "Education"
