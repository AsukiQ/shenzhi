from __future__ import annotations

from clstr.trajectory_inventory import backfill_tool_inventory_from_trajectory_rows


def test_backfill_tool_inventory_from_trajectory_rows_uses_same_trajectory_steps():
    rows = [
        {
            "benchmark": "traject_bench",
            "trajectory_id": "traj-1",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "provenance": {"split": "train"},
        },
        {
            "benchmark": "traject_bench",
            "trajectory_id": "traj-1",
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
            "provenance": {"split": "train"},
        },
    ]

    backfilled, report = backfill_tool_inventory_from_trajectory_rows(rows)

    assert report["backfilled_rows"] == 2
    assert backfilled[0]["tool_inventory_skill_ids"] == ["skill/a", "skill/b", "skill/c"]
    assert backfilled[1]["tool_inventory_skill_ids"] == ["skill/a", "skill/b", "skill/c"]
    assert backfilled[0]["equivalent_next_skill_ids"] == []
    assert backfilled[0]["provenance"]["tool_inventory_source"] == "trajectory_row_backfill"
