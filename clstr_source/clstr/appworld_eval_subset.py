from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

from clstr.appworld_routing import write_json, write_jsonl


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def required_apps_group_key(row: dict[str, Any]) -> str:
    apps = [
        str(app)
        for app in row.get("required_apps", [])
        if str(app) and str(app) != "supervisor"
    ]
    return "+".join(apps) if apps else "unknown"


def _round_robin_groups(groups: OrderedDict[str, list[dict[str, Any]]], max_tasks: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in groups}
    while len(selected) < int(max_tasks):
        added = False
        for key, rows in groups.items():
            offset = offsets[key]
            if offset >= len(rows):
                continue
            selected.append(rows[offset])
            offsets[key] = offset + 1
            added = True
            if len(selected) >= int(max_tasks):
                break
        if not added:
            break
    return selected


def _ordered_groups(rows: Iterable[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(required_apps_group_key(row), []).append(row)
    return groups


def build_stratified_task_subset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    max_tasks: int = 10,
) -> dict[str, Any]:
    if int(max_tasks) <= 0:
        raise ValueError("max_tasks must be positive")
    rows = _read_jsonl(input_path)
    groups = _ordered_groups(rows)
    selected = _round_robin_groups(groups, int(max_tasks))
    written = write_jsonl(output_path, selected)

    group_counts = Counter(required_apps_group_key(row) for row in rows)
    selected_group_counts = Counter(required_apps_group_key(row) for row in selected)
    manifest = {
        "status": "ok" if written else "empty",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_task_count": len(rows),
        "selected_task_count": written,
        "max_tasks": int(max_tasks),
        "strategy": "required_apps_round_robin",
        "group_counts": dict(sorted(group_counts.items())),
        "selected_group_counts": dict(sorted(selected_group_counts.items())),
        "selected_task_ids": [str(row.get("task_id") or row.get("query_id")) for row in selected],
    }
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest
