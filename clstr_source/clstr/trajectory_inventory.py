from __future__ import annotations

from collections import defaultdict
from typing import Any


def _append_unique(output: list[str], value: Any) -> None:
    text = str(value or "")
    if text and text not in output:
        output.append(text)


def infer_tool_inventory_by_trajectory(
    rows: list[dict[str, Any]],
    *,
    benchmark: str = "traject_bench",
) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if str(row.get("benchmark") or "") != benchmark:
            continue
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        _append_unique(inventory[trajectory_id], row.get("skill_id"))
        _append_unique(inventory[trajectory_id], row.get("next_skill_id"))
    return {key: values for key, values in inventory.items() if values}


def backfill_tool_inventory_from_trajectory_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str = "traject_bench",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory_by_trajectory = infer_tool_inventory_by_trajectory(rows, benchmark=benchmark)
    backfilled: list[dict[str, Any]] = []
    backfilled_rows = 0
    existing_inventory_rows = 0
    missing_inventory_rows = 0
    for row in rows:
        if str(row.get("benchmark") or "") != benchmark:
            backfilled.append(row)
            continue
        if row.get("tool_inventory_skill_ids"):
            existing_inventory_rows += 1
            backfilled.append(row)
            continue
        trajectory_id = str(row.get("trajectory_id") or "")
        inventory = inventory_by_trajectory.get(trajectory_id) or []
        if not inventory:
            missing_inventory_rows += 1
            backfilled.append(row)
            continue
        copied = dict(row)
        copied["tool_inventory_skill_ids"] = list(inventory)
        copied.setdefault("equivalent_next_skill_ids", [])
        provenance = copied.get("provenance")
        if isinstance(provenance, dict):
            provenance = dict(provenance)
            provenance["tool_inventory_source"] = "trajectory_row_backfill"
            copied["provenance"] = provenance
        backfilled.append(copied)
        backfilled_rows += 1
    return backfilled, {
        "benchmark": benchmark,
        "trajectory_inventory_count": len(inventory_by_trajectory),
        "backfilled_rows": backfilled_rows,
        "existing_inventory_rows": existing_inventory_rows,
        "missing_inventory_rows": missing_inventory_rows,
    }
