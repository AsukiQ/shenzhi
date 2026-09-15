from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable


HISTORY_SECTION_MARKERS = frozenset(
    {
        "history",
        "previous_actions",
        "previous_tools",
        "previous_tool_calls",
        "prior_actions",
        "prior_tool_calls",
        "tool_history",
        "action_history",
        "conversation_history",
        "dialogue_history",
    }
)
NONCAUSAL_OBSERVATION_PREFIXES = (
    "oracle_next_action_arguments:",
    "oracle_next_tool_arguments:",
)
NONCAUSAL_OBSERVATION_SOURCES = frozenset(
    {
        "next_state_without_tool_result",
        "action_only_no_tool_result",
        "oracle_next_action_arguments",
        "oracle_next_tool_arguments",
    }
)

CURRENT_STATE_CONTRACT = "clstr_structured_current_state_v1"
CAUSAL_STATE_CONTRACT = "clstr_structured_causal_state_v1"
CURRENT_STATE_COMPONENT_KEYS = (
    "goal_text",
    "task_text",
    "current_observation_text",
)


def _single_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def canonical_causal_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return one decision-time event with no inferred future information."""

    skill_id = _single_line(
        event.get("skill_id")
        or event.get("target_skill_id")
        or event.get("executed_skill_id")
    )
    action_text = _single_line(
        event.get("action_text")
        or event.get("executed_action_text")
    )
    result_text = _single_line(
        event.get("result_text")
        or event.get("actual_result_text")
    )
    result_executed = bool(
        event.get("result_executed")
        or event.get("actual_result_executed")
    )
    if not skill_id or not action_text:
        raise ValueError("causal prefix event requires an executed skill and action")
    if result_text and not result_executed:
        raise ValueError("causal prefix result lacks an executed-result contract")
    raw_step = event.get("step_index")
    step_index = None if raw_step in (None, "") else int(raw_step)
    if step_index is not None and step_index < 0:
        raise ValueError("causal prefix event step must be nonnegative")
    return {
        "step_index": step_index,
        "skill_id": skill_id,
        "action_text": action_text,
        "result_text": result_text if result_executed else "",
        "result_executed": bool(result_executed and result_text),
    }


def serialize_causal_state(
    current_state_text: str,
    events: Iterable[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Serialize current state plus events completed before this decision.

    An empty prefix returns the current-state string byte-for-byte.  This is an
    architectural invariant: the causal retriever has an exact zero-history
    endpoint rather than an approximate learned cancellation.
    """

    current = str(current_state_text or "").strip()
    if not current:
        raise ValueError("causal state requires a nonempty current-state channel")
    canonical = [canonical_causal_event(event) for event in events]
    if not canonical:
        return current, canonical
    lines = [current, "causal_prefix:"]
    for index, event in enumerate(canonical):
        step = event["step_index"]
        label = str(index) if step is None else str(step)
        lines.append(f"event[{label}].skill_id: {event['skill_id']}")
        lines.append(f"event[{label}].action: {event['action_text']}")
        if event["result_executed"]:
            lines.append(f"event[{label}].result: {event['result_text']}")
    return "\n".join(lines).strip(), canonical


def _bounded_event_text(value: Any, max_chars: int) -> str:
    text = _single_line(value)
    limit = max(16, int(max_chars))
    if len(text) <= limit:
        return text
    head = max(8, (limit - 5) // 2)
    tail = max(8, limit - head - 5)
    return f"{text[:head]} ... {text[-tail:]}"


def serialize_compact_causal_state(
    current_state_text: str,
    events: Iterable[dict[str, Any]],
    *,
    max_events: int = 8,
    max_action_chars: int = 256,
) -> tuple[str, list[dict[str, Any]]]:
    """Current state plus bounded skill/action history, never raw results.

    Full executed results remain in the canonical events for recurrent-memory
    correction.  The static retrieval query receives only a compact causal
    control-flow summary and retains an exact current-only zero-history path.
    """

    current = str(current_state_text or "").strip()
    if not current:
        raise ValueError("compact causal state requires a nonempty current state")
    canonical = [canonical_causal_event(event) for event in events]
    if not canonical:
        return current, canonical
    limit = max(1, int(max_events))
    selected = canonical[-limit:]
    lines = [current, "causal_skill_action_prefix:"]
    for event in selected:
        step = event["step_index"]
        label = "?" if step is None else str(step)
        lines.append(f"event[{label}].skill_id: {event['skill_id']}")
        lines.append(
            f"event[{label}].action: "
            f"{_bounded_event_text(event['action_text'], max_action_chars)}"
        )
    return "\n".join(lines).strip(), canonical


def compact_causal_state_from_row(
    row: dict[str, Any],
    *,
    max_events: int = 8,
    max_action_chars: int = 256,
) -> str:
    current = str(row.get("state_text_current") or "").strip()
    events = row.get("causal_prefix_events")
    if not current or not isinstance(events, list):
        raise ValueError("compact causal row requires current state and persisted events")
    text, _canonical = serialize_compact_causal_state(
        current,
        events,
        max_events=max_events,
        max_action_chars=max_action_chars,
    )
    return text


def materialize_structured_causal_state(
    row: dict[str, Any],
    events: Iterable[dict[str, Any]],
    *,
    replace_state_text: bool = False,
) -> dict[str, Any]:
    """Persist separate current, causal-prefix, and future-label channels."""

    prepared = materialize_structured_current_state(
        row,
        replace_state_text=replace_state_text,
    )
    causal, canonical_events = serialize_causal_state(
        str(prepared["state_text_current"]),
        events,
    )
    prepared["state_text_causal"] = causal
    prepared["causal_state_contract"] = CAUSAL_STATE_CONTRACT
    prepared["causal_prefix_events"] = canonical_events
    prepared["causal_prefix_event_count"] = len(canonical_events)
    prepared["causal_prefix_sha256"] = _text_digest(
        json.dumps(
            canonical_events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    prepared["causal_state_sha256"] = _text_digest(causal)
    prepared["zero_history_causal_equals_current"] = bool(
        not canonical_events and causal == str(prepared["state_text_current"])
    )
    return prepared


def _marker_name(line: str) -> str | None:
    head, separator, _tail = str(line).partition(":")
    if not separator:
        return None
    marker = head.strip().lower().replace(" ", "_")
    return marker if marker in HISTORY_SECTION_MARKERS else None


def strip_history_sections(text: str) -> str:
    """Remove appended execution-history sections from a router query."""

    lines = str(text or "").splitlines()
    cut = len(lines)
    for index, line in enumerate(lines):
        if _marker_name(line) is not None:
            cut = index
            break
    return "\n".join(lines[:cut]).strip()


def _text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _labeled_current_observation(state_text: str) -> str:
    """Extract a labeled current observation from a legacy structured prompt.

    This is a migration parser, not the canonical router-state definition.  It
    accepts only an explicit ``observation:`` field before the first declared
    replay/history field.  The canonical row persists the extracted component
    and its source digest so later consumers never need to repeat the parsing.
    """

    lines = str(state_text or "").splitlines()
    history_index = len(lines)
    for index, line in enumerate(lines):
        if _marker_name(line) is not None:
            history_index = index
            break
    observation_index: int | None = None
    first_value = ""
    for index, line in enumerate(lines[:history_index]):
        head, separator, tail = str(line).partition(":")
        if separator and head.strip().lower().replace(" ", "_") in {
            "observation",
            "current_observation",
        }:
            observation_index = index
            first_value = tail.strip()
            break
    if observation_index is None:
        return ""
    body = [first_value] if first_value else []
    body.extend(lines[observation_index + 1 : history_index])
    return "\n".join(body).strip()


def serialize_current_state_components(components: dict[str, Any]) -> str:
    goal = str(components.get("goal_text") or "").strip()
    task = str(components.get("task_text") or "").strip()
    observation = str(components.get("current_observation_text") or "").strip()
    if not (goal or task or observation):
        raise ValueError("structured current state has no goal, task, or observation")
    lines: list[str] = []
    if goal:
        lines.append(f"goal: {goal}")
    if task and task != goal:
        lines.append(f"task: {task}")
    if observation:
        lines.append(f"observation: {observation}")
    return "\n".join(lines).strip()


def structured_current_state_components(row: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Return canonical current-only components and migration provenance.

    Explicit canonical components win.  Otherwise the function reconstructs
    the decision boundary from source-level goal/task fields and, where
    necessary, an explicitly labeled observation section in the legacy prompt.
    It never treats ``next_observation_text`` or replay fields as current state.
    """

    declared = row.get("current_state_components")
    if isinstance(declared, dict):
        components = {
            key: str(declared.get(key) or "").strip()
            for key in CURRENT_STATE_COMPONENT_KEYS
        }
        serialize_current_state_components(components)
        return components, "declared_structured_components"

    goal = _first_text(row, ("goal_text", "instruction_text", "query_text"))
    task = _first_text(row, ("task_text", "current_task_text"))
    observation = _first_text(
        row,
        ("current_observation_text", "observation_text", "current_observation"),
    )
    source = "explicit_structured_fields"
    if not observation:
        observation = _labeled_current_observation(str(row.get("state_text") or ""))
        if observation:
            source = "labeled_observation_migration"
    if not (goal or task or observation):
        raise ValueError(
            "cannot reconstruct current state from structured goal/task/observation fields"
        )
    components = {
        "goal_text": goal,
        "task_text": task,
        "current_observation_text": observation,
    }
    serialize_current_state_components(components)
    return components, source


def materialize_structured_current_state(
    row: dict[str, Any],
    *,
    replace_state_text: bool = False,
) -> dict[str, Any]:
    """Materialize the canonical router state from structured components."""

    prepared = dict(row)
    full = str(row.get("state_text_full") or row.get("state_text") or "").strip()
    components, source = structured_current_state_components(row)
    current = serialize_current_state_components(components)
    if any(_marker_name(line) is not None for line in current.splitlines()):
        raise ValueError("structured current-state serialization contains replay history")
    prepared["state_text_full"] = full
    prepared["state_text_current"] = current
    prepared["current_state_components"] = components
    prepared["current_state_contract"] = CURRENT_STATE_CONTRACT
    prepared["current_state_materialization_source"] = source
    prepared["current_state_full_text_sha256"] = _text_digest(full)
    prepared["current_state_components_sha256"] = _text_digest(
        json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    if replace_state_text:
        prepared["state_text"] = current
    return prepared


def materialize_structured_retrieval_state(
    row: dict[str, Any],
    *,
    replace_state_text: bool = False,
) -> dict[str, Any]:
    """Canonicalize a static retrieval query without legacy replay suffixes.

    Unified retrieval sources use both ``query_text`` and ``query``.  Some
    trajectory-derived rows serialize an empty or nonempty replay section in
    that field.  Static Stage0 must see only the query before the first
    declared history marker, while provenance consumers retain the exact full
    text in ``state_text_full``.
    """

    full = _first_text(row, ("query_text", "query", "state_text"))
    if not full:
        raise ValueError("retrieval row lacks query_text/query/state_text")
    explicit = str(row.get("state_text_current") or "").strip()
    current = strip_history_sections(explicit or full)
    if not current:
        raise ValueError("retrieval query has no current-only text")
    goal = current
    first, _separator, tail = current.partition("\n")
    head, label_separator, first_value = first.partition(":")
    if label_separator and head.strip().lower().replace(" ", "_") in {
        "goal",
        "query",
        "instruction",
    }:
        goal = "\n".join(
            value for value in (first_value.strip(), tail.strip()) if value
        ).strip()
    materialized = materialize_structured_current_state(
        {
            "goal_text": goal,
            "state_text": full,
        },
        replace_state_text=replace_state_text,
    )
    prepared = dict(row)
    prepared.update(materialized)
    prepared["retrieval_query_materialization_source"] = (
        "explicit_state_text_current"
        if explicit
        else (
            "legacy_history_section_removed"
            if current != full
            else "raw_query_is_current_only"
        )
    )
    return prepared


def router_state_text(row: dict[str, Any]) -> str:
    explicit = str(row.get("state_text_current") or "").strip()
    if explicit:
        return strip_history_sections(explicit)
    return strip_history_sections(
        str(row.get("state_text") or row.get("observation_text") or "")
    )


def materialize_history_free_state(
    row: dict[str, Any],
    *,
    replace_state_text: bool = False,
) -> dict[str, Any]:
    prepared = dict(row)
    full = str(row.get("state_text_full") or row.get("state_text") or "").strip()
    current = router_state_text(row)
    prepared["state_text_full"] = full
    prepared["state_text_current"] = current
    if replace_state_text:
        prepared["state_text"] = current
    if "next_state_text" in row:
        prepared["next_state_text"] = strip_history_sections(
            str(row.get("next_state_text") or "")
        )
    prefix = row.get("replay_prefix")
    if isinstance(prefix, list):
        clean_prefix: list[Any] = []
        for step in prefix:
            if not isinstance(step, dict):
                clean_prefix.append(step)
                continue
            clean_step = dict(step)
            clean_step["observation_text"] = strip_history_sections(
                str(step.get("observation_text") or "")
            )
            clean_prefix.append(clean_step)
        prepared["replay_prefix"] = clean_prefix
    return prepared


def actual_causal_observation(step: dict[str, Any]) -> bool:
    source = str(step.get("observation_source") or "").strip().lower()
    if source in NONCAUSAL_OBSERVATION_SOURCES:
        return False
    observation = str(step.get("next_observation_text") or "").strip()
    if not observation:
        return False
    lowered = observation.lower()
    if any(lowered.startswith(prefix) for prefix in NONCAUSAL_OBSERVATION_PREFIXES):
        return False
    return True


def actual_causal_replay_step(step: dict[str, Any]) -> bool:
    action = str(step.get("action_text") or "").strip()
    skill = str(step.get("skill_id") or "").strip()
    return bool(action and skill and actual_causal_observation(step))


def audit_history_channel_rows(
    rows: Iterable[dict[str, Any]],
    *,
    require_explicit_current: bool = False,
    require_actual_replay_observation: bool = False,
    require_structured_current: bool = False,
    require_explicit_causal: bool = False,
) -> dict[str, Any]:
    row_count = 0
    post_initial_rows = 0
    explicit_current_rows = 0
    leaked_rows = 0
    leakage_reasons: Counter[str] = Counter()
    replay_steps = 0
    invalid_replay_action_steps = 0
    invalid_replay_observation_steps = 0
    structured_current_rows = 0
    explicit_causal_rows = 0
    structured_causal_rows = 0
    invalid_causal_rows = 0
    causal_reasons: Counter[str] = Counter()
    leakage_examples: list[dict[str, Any]] = []

    for row in rows:
        row_count += 1
        try:
            post_initial = int(row.get("step_index") or 0) > 0
        except (TypeError, ValueError):
            post_initial = False
        post_initial_rows += int(post_initial)
        explicit = bool(str(row.get("state_text_current") or "").strip())
        explicit_current_rows += int(explicit)
        current = str(row.get("state_text_current") or "").strip()
        history_overlap_scope = current
        row_reasons: set[str] = set()
        if require_explicit_current and not explicit:
            row_reasons.add("missing_state_text_current")
        if str(row.get("current_state_contract") or "") == CURRENT_STATE_CONTRACT:
            try:
                components, _source = structured_current_state_components(row)
                expected = serialize_current_state_components(components)
                # Goal/task text can legitimately name an already executed
                # action or, for parallel plans, enumerate every requested
                # tool.  Treating any such lexical overlap as replay leakage
                # produced false blockers (for example ``look`` in an
                # ALFWorld goal).  Once the structured-current contract is
                # present, exact replay containment is meaningful only in the
                # current observation component.  Explicit history markers
                # remain blocking across the complete serialization below.
                history_overlap_scope = str(
                    components.get("current_observation_text") or ""
                ).strip()
                if expected != current:
                    row_reasons.add("structured_current_serialization_mismatch")
                else:
                    structured_current_rows += 1
            except ValueError:
                row_reasons.add("invalid_structured_current_components")
        elif require_structured_current:
            row_reasons.add("missing_structured_current_contract")
        if current and any(_marker_name(line) is not None for line in current.splitlines()):
            row_reasons.add("structured_history_marker")
        history = str(row.get("history_text") or "").strip()
        if history_overlap_scope and history and history in history_overlap_scope:
            row_reasons.add("history_text_exact_substring")
        if row_reasons:
            leaked_rows += 1
            leakage_reasons.update(row_reasons)
            if len(leakage_examples) < 20:
                leakage_examples.append(
                    {
                        "trajectory_id": str(row.get("trajectory_id") or ""),
                        "task_id": str(row.get("task_id") or ""),
                        "query_id": str(row.get("query_id") or ""),
                        "step_index": int(row.get("step_index") or 0),
                        "reasons": sorted(row_reasons),
                        "state_text_current": current[:600],
                        "history_text": history[:600],
                    }
                )

        causal = str(row.get("state_text_causal") or "").strip()
        causal_contract = str(row.get("causal_state_contract") or "").strip()
        explicit_causal_rows += int(bool(causal))
        causal_row_reasons: set[str] = set()
        if require_explicit_causal and not causal:
            causal_row_reasons.add("missing_state_text_causal")
        if causal and not causal_contract:
            causal_row_reasons.add("missing_causal_state_contract")
        declared_causal_digest = str(row.get("causal_state_sha256") or "").strip()
        if causal and (
            not declared_causal_digest
            or declared_causal_digest != _text_digest(causal)
        ):
            causal_row_reasons.add("causal_state_digest_mismatch")
        if causal_contract == CAUSAL_STATE_CONTRACT:
            raw_events = row.get("causal_prefix_events")
            if not isinstance(raw_events, list):
                causal_row_reasons.add("missing_causal_prefix_events")
            else:
                try:
                    expected_causal, canonical_events = serialize_causal_state(
                        current,
                        raw_events,
                    )
                except (TypeError, ValueError):
                    causal_row_reasons.add("invalid_causal_prefix_event")
                else:
                    if expected_causal != causal:
                        causal_row_reasons.add("causal_state_serialization_mismatch")
                    if int(row.get("causal_prefix_event_count") or 0) != len(
                        canonical_events
                    ):
                        causal_row_reasons.add("causal_prefix_event_count_mismatch")
                    raw_decision_step = row.get("decision_step_index")
                    if raw_decision_step not in (None, ""):
                        decision_step = int(raw_decision_step)
                        if any(
                            event["step_index"] is not None
                            and int(event["step_index"]) >= decision_step
                            for event in canonical_events
                        ):
                            causal_row_reasons.add("causal_event_not_before_decision")
                    if not canonical_events and causal != current:
                        causal_row_reasons.add("zero_history_not_exact_current")
                    if not causal_row_reasons:
                        structured_causal_rows += 1
        if causal_row_reasons:
            invalid_causal_rows += 1
            causal_reasons.update(causal_row_reasons)

        prefix = row.get("replay_prefix")
        if not isinstance(prefix, list):
            continue
        for step in prefix:
            if not isinstance(step, dict):
                continue
            replay_steps += 1
            invalid_replay_action_steps += int(
                not str(step.get("action_text") or "").strip()
                or not str(step.get("skill_id") or "").strip()
            )
            invalid_replay_observation_steps += int(
                not actual_causal_observation(step)
            )

    blockers: list[str] = []
    if leaked_rows:
        blockers.append("router_state_contains_execution_history")
    if require_actual_replay_observation:
        if invalid_replay_action_steps:
            blockers.append("replay_lacks_executed_action_or_skill")
        if invalid_replay_observation_steps:
            blockers.append("replay_lacks_actual_causal_observation")
    if invalid_causal_rows:
        blockers.append("invalid_causal_state_channel")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "row_count": int(row_count),
        "post_initial_row_count": int(post_initial_rows),
        "explicit_current_state_rows": int(explicit_current_rows),
        "structured_current_state_rows": int(structured_current_rows),
        "explicit_causal_state_rows": int(explicit_causal_rows),
        "structured_causal_state_rows": int(structured_causal_rows),
        "invalid_causal_state_rows": int(invalid_causal_rows),
        "causal_state_reasons": dict(sorted(causal_reasons.items())),
        "leaked_row_count": int(leaked_rows),
        "leakage_reasons": dict(sorted(leakage_reasons.items())),
        "leakage_examples": leakage_examples,
        "replay_step_count": int(replay_steps),
        "invalid_replay_action_steps": int(invalid_replay_action_steps),
        "invalid_replay_observation_steps": int(
            invalid_replay_observation_steps
        ),
        "require_explicit_current": bool(require_explicit_current),
        "require_actual_replay_observation": bool(
            require_actual_replay_observation
        ),
        "require_structured_current": bool(require_structured_current),
    }
