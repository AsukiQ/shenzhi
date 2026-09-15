from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


def handoff_sampling_key(row: dict[str, Any]) -> tuple[str, str]:
    benchmark = str(row.get("benchmark") or "<missing>")
    loss_mask = row.get("loss_mask") or {}
    has_current = bool(loss_mask.get("routing") and row.get("skill_id"))
    has_next = bool(loss_mask.get("L_trans_skill_ce") and row.get("next_skill_id"))
    if has_current and has_next:
        label = "current_next"
    elif has_current:
        label = "current"
    elif has_next:
        label = "next"
    else:
        label = "unsupervised"
    return benchmark, label


def select_rows_round_robin(
    rows: list[dict[str, Any]],
    max_rows: int | None,
    *,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...] | str],
) -> list[dict[str, Any]]:
    if max_rows is None or int(max_rows) >= len(rows):
        return list(rows)
    limit = max(0, int(max_rows))
    if limit == 0:
        return []

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        buckets[key if isinstance(key, tuple) else (key,)].append(row)

    active_keys = sorted(buckets)
    offsets = {key: 0 for key in active_keys}
    selected: list[dict[str, Any]] = []
    while active_keys and len(selected) < limit:
        next_active: list[tuple[str, str]] = []
        for key in active_keys:
            offset = offsets[key]
            bucket = buckets[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[key] = offset + 1
            if offsets[key] < len(bucket):
                next_active.append(key)
            if len(selected) >= limit:
                break
        active_keys = next_active
    return selected


def select_audit_rows(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    return select_rows_round_robin(rows, max_rows, key_fn=handoff_sampling_key)
