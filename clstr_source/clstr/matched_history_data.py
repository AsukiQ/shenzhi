from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from clstr.history_channel import canonical_causal_event


MATCHED_HISTORY_DATA_SCHEMA = "clstr_matched_history_e1_data_v1"
MATCHED_HISTORY_BENCHMARKS = frozenset({"toolbench_g3", "tau2"})
_CATALOG_SKILL_SET_CACHE: dict[
    tuple[int, str, str],
    tuple[dict[str, dict[str, Any]], frozenset[str]],
] = {}


class MatchedHistoryTrajectorySkip(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


def normalized_history_benchmark(row: dict[str, Any]) -> str:
    return str(row.get("_unified_source") or row.get("benchmark") or "").strip()


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("target_skill_id") or row.get("skill_id") or "").strip()


def _catalog_skill_id_set(
    catalogs: dict[str, dict[str, Any]],
    catalog_id: str,
) -> frozenset[str]:
    catalog = catalogs[catalog_id]
    digest = str(catalog.get("inventory_catalog_digest") or "")
    key = (id(catalogs), catalog_id, digest)
    cached = _CATALOG_SKILL_SET_CACHE.get(key)
    if cached is not None and cached[0] is catalogs:
        return cached[1]
    legal = frozenset(
        str(value).strip()
        for value in catalog.get("runtime_visible_skill_ids") or []
        if str(value).strip()
    )
    if not legal:
        raise ValueError("matched-history runtime catalog is empty")
    _CATALOG_SKILL_SET_CACHE[key] = (catalogs, legal)
    return legal


def _positive_skill_ids(
    row: dict[str, Any],
    *,
    selected_skill_ids: set[str],
) -> list[str]:
    target = _skill_id(row)
    positives = {
        target,
        *(
            str(value).strip()
            for value in row.get("equivalent_next_skill_ids") or []
            if str(value).strip()
        ),
    }
    unknown = sorted(positives - selected_skill_ids)
    if unknown:
        raise ValueError(
            f"matched-history positive is outside the selected skill table: {unknown[:3]}"
        )
    return sorted(positives)


def _verified_result(row: dict[str, Any]) -> tuple[str, bool, str]:
    capabilities = row.get("capabilities")
    capability = bool(
        isinstance(capabilities, dict)
        and capabilities.get("actual_execution_result")
    )
    text = str(row.get("actual_result_text") or "").strip()
    executed = bool(row.get("actual_result_executed"))
    if capability and not (executed and text):
        raise ValueError(
            "matched-history executed-result capability lacks an aligned result"
        )
    if executed and not text:
        raise ValueError("matched-history executed-result flag lacks result text")
    if text and not capability:
        reason = (
            "executed_flag_without_actual_execution_capability"
            if executed
            else "result_text_without_actual_execution_capability"
        )
        return "", False, reason
    return (text, bool(executed and text), "")


def _canonical_row_event(row: dict[str, Any]) -> dict[str, Any]:
    result_text, result_executed, _reason = _verified_result(row)
    return canonical_causal_event(
        {
            "step_index": int(row.get("step_index") or 0),
            "skill_id": _skill_id(row),
            "action_text": str(row.get("action_text") or ""),
            "result_text": result_text,
            "result_executed": result_executed,
        }
    )


def _validate_persisted_prefix(
    row: dict[str, Any],
    earlier_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_events = row.get("causal_prefix_events")
    if not isinstance(raw_events, list):
        raise ValueError("matched-history row lacks persisted causal-prefix events")
    canonical = [canonical_causal_event(event) for event in raw_events]
    declared_count = int(row.get("causal_prefix_event_count") or 0)
    if declared_count != len(canonical):
        raise ValueError("matched-history causal-prefix count differs")
    declared_digest = str(row.get("causal_prefix_sha256") or "")
    if not declared_digest or declared_digest != _json_digest(canonical):
        raise ValueError("matched-history causal-prefix digest differs")
    decision_step = int(row.get("step_index") or 0)
    steps = [event.get("step_index") for event in canonical]
    if any(step is None for step in steps):
        raise ValueError("matched-history prefix event lacks a step index")
    integer_steps = [int(step) for step in steps]
    if integer_steps != sorted(integer_steps) or len(set(integer_steps)) != len(
        integer_steps
    ):
        raise ValueError("matched-history prefix steps are not strictly ordered")
    if any(step >= decision_step for step in integer_steps):
        raise ValueError("matched-history prefix contains the current or a future event")
    expected_steps = list(range(decision_step))
    if integer_steps != expected_steps:
        raise ValueError("matched-history prefix is not contiguous from step zero")
    result_visibility: list[dict[str, Any]] = []
    for event, step in zip(canonical, integer_steps):
        previous = earlier_rows.get(step)
        if previous is None:
            raise ValueError("matched-history prefix cannot be joined to an earlier row")
        expected = _canonical_row_event(previous)
        if any(
            event[key] != expected[key]
            for key in ("step_index", "skill_id", "action_text")
        ):
            raise ValueError(
                "matched-history prefix event differs from its earlier trajectory row"
            )
        if bool(event["result_executed"]):
            if not bool(expected["result_executed"]) or str(
                event["result_text"]
            ) != str(expected["result_text"]):
                raise ValueError(
                    "matched-history visible prefix result differs from its verified result"
                )
            result_sha256 = hashlib.sha256(
                str(event["result_text"]).encode("utf-8")
            ).hexdigest()
        else:
            result_sha256 = ""
        result_visibility.append(
            {
                "step_index": step,
                "result_executed": bool(event["result_executed"]),
                "result_sha256": result_sha256,
            }
        )
    return result_visibility


def prepare_matched_history_trajectory(
    rows: list[dict[str, Any]],
    *,
    selected_skill_ids: set[str],
    catalogs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not rows:
        return [], {}
    ordered = sorted(rows, key=lambda item: int(item.get("step_index") or 0))
    if len(ordered) < 2:
        raise MatchedHistoryTrajectorySkip("trajectory_too_short")
    trajectory_id = str(
        ordered[0].get("trajectory_id") or ordered[0].get("task_id") or ""
    ).strip()
    if not trajectory_id:
        raise ValueError("matched-history trajectory lacks an identity")
    if any(
        str(row.get("trajectory_id") or row.get("task_id") or "").strip()
        != trajectory_id
        for row in ordered
    ):
        raise ValueError("matched-history trajectory contains mixed identities")
    benchmark = normalized_history_benchmark(ordered[0])
    if benchmark not in MATCHED_HISTORY_BENCHMARKS:
        raise ValueError(f"unsupported matched-history benchmark: {benchmark}")
    if any(normalized_history_benchmark(row) != benchmark for row in ordered):
        raise ValueError("matched-history trajectory contains mixed benchmarks")
    steps = [int(row.get("step_index") or 0) for row in ordered]
    if steps != list(range(len(ordered))):
        raise MatchedHistoryTrajectorySkip(
            "trajectory_not_contiguous_from_initial_decision"
        )
    ordered_flags = [
        bool((row.get("capabilities") or {}).get("ordered_next_tool"))
        for row in ordered
    ]
    if ordered_flags[0] or not all(ordered_flags[1:]):
        raise MatchedHistoryTrajectorySkip(
            "trajectory_not_fully_ordered_after_initial_event"
        )

    earlier_rows: dict[int, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    quarantined: Counter[str] = Counter()
    for row, step in zip(ordered, steps):
        if row.get("decision_step_index") in (None, "") or int(
            row["decision_step_index"]
        ) != step:
            raise ValueError("matched-history decision index differs from trajectory step")
        state = str(row.get("state_text_current") or "").strip()
        action = str(row.get("action_text") or "").strip()
        target = _skill_id(row)
        if not state or not action or not target:
            raise ValueError("matched-history row lacks state, action, or target")
        if target not in selected_skill_ids:
            raise ValueError("matched-history target is outside the selected skill table")
        catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
        catalog = catalogs.get(catalog_id)
        if not isinstance(catalog, dict):
            raise ValueError(f"matched-history row names an unknown catalog: {catalog_id}")
        expected_catalog_digest = str(row.get("inventory_catalog_digest") or "")
        observed_catalog_digest = str(catalog.get("inventory_catalog_digest") or "")
        if not expected_catalog_digest or expected_catalog_digest != observed_catalog_digest:
            raise ValueError("matched-history runtime catalog digest differs")
        positives = _positive_skill_ids(row, selected_skill_ids=selected_skill_ids)
        legal = _catalog_skill_id_set(catalogs, catalog_id)
        if not set(positives).issubset(legal):
            raise ValueError("matched-history positive lies outside the runtime catalog")
        history_result_visibility = _validate_persisted_prefix(row, earlier_rows)
        result_text, result_executed, quarantine_reason = _verified_result(row)
        if quarantine_reason:
            quarantined[quarantine_reason] += 1
        persisted_event = _canonical_row_event(row)
        if bool(persisted_event["result_executed"]) != bool(result_executed):
            raise ValueError("matched-history canonical result execution flag differs")
        compact = {
            "schema_version": MATCHED_HISTORY_DATA_SCHEMA,
            "benchmark": benchmark,
            "data_split": str(row.get("data_split") or ""),
            "trajectory_id": trajectory_id,
            "step_index": step,
            "state_text_current": state,
            "executed_skill_id": target,
            "action_text": str(persisted_event["action_text"]),
            "result_text": str(persisted_event["result_text"]),
            "result_executed": bool(persisted_event["result_executed"]),
            "result_quarantine_reason": quarantine_reason,
            "target_skill_id": target,
            "positive_skill_ids": positives,
            "runtime_visible_catalog_id": catalog_id,
            "inventory_catalog_digest": expected_catalog_digest,
            "causal_prefix_event_count": int(
                row.get("causal_prefix_event_count") or 0
            ),
            "causal_prefix_sha256": str(row.get("causal_prefix_sha256") or ""),
            "history_result_visibility": history_result_visibility,
        }
        compact["source_row_sha256"] = _json_digest(row)
        output.append(compact)
        earlier_rows[step] = row
    return output, dict(sorted(quarantined.items()))


def iter_selected_trajectories(
    path: str | Path,
    *,
    benchmarks: set[str] | frozenset[str] = MATCHED_HISTORY_BENCHMARKS,
) -> Iterator[list[dict[str, Any]]]:
    selected = set(benchmarks)
    current_id = ""
    current_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if normalized_history_benchmark(row) not in selected:
                continue
            trajectory_id = str(
                row.get("trajectory_id") or row.get("task_id") or ""
            ).strip()
            if not trajectory_id:
                raise ValueError(
                    f"matched-history row {line_number} lacks a trajectory identity"
                )
            if not current_id:
                current_id = trajectory_id
            if trajectory_id != current_id:
                completed.add(current_id)
                yield current_rows
                if trajectory_id in completed:
                    raise ValueError(
                        "matched-history source is not contiguous by trajectory"
                    )
                current_id = trajectory_id
                current_rows = []
            current_rows.append(row)
    if current_rows:
        yield current_rows


def history_events_for_decision(
    trajectory_rows: list[dict[str, Any]],
    decision_index: int,
    *,
    max_horizon: int = 16,
) -> list[dict[str, Any]]:
    if int(max_horizon) <= 0:
        raise ValueError("matched-history horizon must be positive")
    index = int(decision_index)
    if index < 0 or index >= len(trajectory_rows):
        raise IndexError("matched-history decision index is outside the trajectory")
    current = trajectory_rows[index]
    if int(current.get("step_index") or 0) != index:
        raise ValueError("matched-history compact trajectory is not index aligned")
    start = max(0, index - int(max_horizon))
    visibility = current.get("history_result_visibility")
    if not isinstance(visibility, list) or len(visibility) != index:
        raise ValueError("matched-history row has an invalid result-visibility contract")
    visibility_by_step = {
        int(item["step_index"]): item
        for item in visibility
        if isinstance(item, dict) and item.get("step_index") not in (None, "")
    }
    if sorted(visibility_by_step) != list(range(index)):
        raise ValueError("matched-history result visibility is not contiguous")
    events: list[dict[str, Any]] = []
    for row in trajectory_rows[start:index]:
        if row.get("schema_version") != MATCHED_HISTORY_DATA_SCHEMA:
            raise ValueError("matched-history compact row has the wrong schema")
        step = int(row["step_index"])
        result_contract = visibility_by_step[step]
        result_executed = bool(result_contract.get("result_executed"))
        result_text = str(row.get("result_text") or "") if result_executed else ""
        if result_executed:
            expected_sha256 = str(result_contract.get("result_sha256") or "")
            observed_sha256 = hashlib.sha256(result_text.encode("utf-8")).hexdigest()
            if not result_text or not expected_sha256 or observed_sha256 != expected_sha256:
                raise ValueError(
                    "matched-history visible result differs from its compact-row contract"
                )
        events.append(
            {
                "step_index": step,
                "state_text": str(row["state_text_current"]),
                "skill_id": str(row["executed_skill_id"]),
                "action_text": str(row["action_text"]),
                "result_text": result_text,
                "result_executed": result_executed,
            }
        )
    return events


def serialize_factual_history(events: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        step = int(event["step_index"])
        state = " ".join(str(event.get("state_text") or "").split())
        skill = " ".join(str(event.get("skill_id") or "").split())
        action = " ".join(str(event.get("action_text") or "").split())
        result = " ".join(str(event.get("result_text") or "").split())
        executed = bool(event.get("result_executed"))
        if not state or not skill or not action:
            raise ValueError("serialized matched history has an incomplete event")
        if result and not executed:
            raise ValueError("serialized matched history has an unverified result")
        lines.extend(
            (
                f"event[{step}].state: {state}",
                f"event[{step}].skill_id: {skill}",
                f"event[{step}].action: {action}",
            )
        )
        if executed:
            lines.append(f"event[{step}].result: {result}")
    return "\n".join(lines)


def load_matched_history_trajectories(
    path: str | Path,
) -> list[list[dict[str, Any]]]:
    trajectories: list[list[dict[str, Any]]] = []
    current_id = ""
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != MATCHED_HISTORY_DATA_SCHEMA:
                raise ValueError(
                    f"matched-history compact row {line_number} has the wrong schema"
                )
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            if not trajectory_id:
                raise ValueError("matched-history compact row lacks a trajectory identity")
            if not current_id:
                current_id = trajectory_id
            if trajectory_id != current_id:
                seen.add(current_id)
                trajectories.append(current)
                if trajectory_id in seen:
                    raise ValueError(
                        "matched-history compact data is not contiguous by trajectory"
                    )
                current_id = trajectory_id
                current = []
            if int(row.get("step_index") or 0) != len(current):
                raise ValueError(
                    "matched-history compact trajectory is not contiguous from step zero"
                )
            visibility = row.get("history_result_visibility")
            if not isinstance(visibility, list) or len(visibility) != len(current):
                raise ValueError(
                    "matched-history compact row has an invalid visibility contract"
                )
            current.append(row)
    if current:
        trajectories.append(current)
    if not trajectories:
        raise ValueError("matched-history compact dataset is empty")
    return trajectories
