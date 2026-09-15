from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

import torch

from clstr.history_channel import actual_causal_observation, materialize_structured_current_state


VNEXT_SCHEMA_VERSION = "clstr_vnext_semantic_v1"
CANONICAL_INVENTORY_KEY = "runtime_visible_skill_ids"
RESULT_VISIBLE_OVERLAP_THRESHOLD = 0.8


# Full-v2 contains hundreds of thousands of rows that reference the same
# 67k-skill catalog.  Resolving/deduplicating that catalog, rebuilding its set,
# and mapping all IDs to tensor columns once per *row* is mathematically
# redundant and makes both trainer startup and every training step Python-bound.
# These process-local caches keep the owning dictionaries alive so an object ID
# cannot be recycled underneath a stale entry.  Catalog and skill-index
# dictionaries are immutable for the lifetime of one trainer/evaluator.
_CATALOG_SKILL_IDS_CACHE: dict[
    tuple[int, str, str],
    tuple[dict[str, dict[str, Any]], list[str]],
] = {}
_CATALOG_SKILL_SET_CACHE: dict[
    tuple[int, str, str],
    tuple[dict[str, dict[str, Any]], frozenset[str]],
] = {}
_CATALOG_INDEX_TENSOR_CACHE: dict[
    tuple[int, int, str, str, str],
    tuple[
        dict[str, dict[str, Any]],
        dict[str, int],
        torch.Tensor,
    ],
] = {}


@dataclass(frozen=True)
class CapabilityFlags:
    required_tool_set: bool
    current_state_route_set: bool
    ordered_next_tool: bool
    actual_execution_result: bool
    causal_branch_pair: bool
    verified_order_effect_pair: bool
    eligible_outcome_pair: bool


@dataclass(frozen=True)
class SemanticViews:
    retrieval_rows: list[dict[str, Any]]
    static_route_rows: list[dict[str, Any]]
    ordered_transition_rows: list[dict[str, Any]]
    result_correction_rows: list[dict[str, Any]]
    causal_branch_pairs: list[dict[str, Any]]
    causal_order_pairs: list[dict[str, Any]]
    no_call_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("skill_id", "canonical_skill_id", "id", "name"):
            if str(value.get(key) or "").strip():
                return [str(value[key]).strip()]
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            output.extend(_string_list(item))
        return output
    text = str(value).strip()
    return [text] if text else []


def _unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def is_no_call_row(row: dict[str, Any]) -> bool:
    return str(row.get("route_target") or "").strip().upper() in {"STOP", "NO_CALL"} or bool(
        row.get("no_tool_action_required")
    )


def runtime_visible_skill_ids(
    row: dict[str, Any],
    inventory_catalogs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    declared = _unique(_string_list(row.get(CANONICAL_INVENTORY_KEY)))
    if declared:
        return declared
    catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
    if catalog_id:
        catalog = (inventory_catalogs or {}).get(catalog_id)
        if not isinstance(catalog, dict):
            raise ValueError(f"missing runtime inventory catalog: {catalog_id}")
        expected_digest = str(row.get("inventory_catalog_digest") or "").strip()
        observed_digest = str(catalog.get("inventory_catalog_digest") or "").strip()
        if not expected_digest or expected_digest != observed_digest:
            raise ValueError(f"runtime inventory catalog digest mismatch: {catalog_id}")
        if inventory_catalogs is None:
            raise ValueError(f"missing runtime inventory catalog: {catalog_id}")
        cache_key = (id(inventory_catalogs), catalog_id, observed_digest)
        cached = _CATALOG_SKILL_IDS_CACHE.get(cache_key)
        if cached is not None and cached[0] is inventory_catalogs:
            return cached[1]
        resolved = _unique(_string_list(catalog.get(CANONICAL_INVENTORY_KEY)))
        if not resolved:
            raise ValueError(f"runtime inventory catalog is empty: {catalog_id}")
        _CATALOG_SKILL_IDS_CACHE[cache_key] = (inventory_catalogs, resolved)
        return resolved
    protocol = str(row.get("inventory_protocol") or "").strip().lower()
    catalog_digest = str(row.get("inventory_catalog_digest") or "").strip()
    global_ids = _unique(_string_list(row.get("public_global_catalog_skill_ids")))
    if protocol == "public_global" and catalog_digest and global_ids:
        return global_ids
    raise ValueError(
        "vNext row lacks runtime_visible_skill_ids or a hashed public-global catalog"
    )


def runtime_visible_skill_id_set(
    row: dict[str, Any],
    inventory_catalogs: dict[str, dict[str, Any]] | None = None,
) -> frozenset[str]:
    """Resolve one legal inventory as a reusable immutable membership set."""

    catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
    if not catalog_id:
        return frozenset(runtime_visible_skill_ids(row, inventory_catalogs))
    expected_digest = str(row.get("inventory_catalog_digest") or "").strip()
    if inventory_catalogs is None:
        raise ValueError(f"missing runtime inventory catalog: {catalog_id}")
    cache_key = (id(inventory_catalogs), catalog_id, expected_digest)
    cached = _CATALOG_SKILL_SET_CACHE.get(cache_key)
    if cached is not None and cached[0] is inventory_catalogs:
        return cached[1]
    resolved = frozenset(runtime_visible_skill_ids(row, inventory_catalogs))
    if not resolved:
        raise ValueError(f"runtime inventory catalog is empty: {catalog_id}")
    _CATALOG_SKILL_SET_CACHE[cache_key] = (inventory_catalogs, resolved)
    return resolved


def _catalog_index_tensor(
    row: dict[str, Any],
    inventory_catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
    expected_digest = str(row.get("inventory_catalog_digest") or "").strip()
    cache_key = (
        id(inventory_catalogs),
        id(skill_id_to_idx),
        str(device),
        catalog_id,
        expected_digest,
    )
    cached = _CATALOG_INDEX_TENSOR_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is inventory_catalogs
        and cached[1] is skill_id_to_idx
    ):
        return cached[2]
    declared = runtime_visible_skill_ids(row, inventory_catalogs)
    unknown = [skill_id for skill_id in declared if skill_id not in skill_id_to_idx]
    if unknown:
        raise ValueError(f"runtime inventory contains unknown skills: {unknown[:3]}")
    indices = torch.tensor(
        [skill_id_to_idx[skill_id] for skill_id in declared],
        dtype=torch.long,
        device=device,
    )
    if not int(indices.numel()):
        raise ValueError("runtime inventory must contain at least one known skill")
    _CATALOG_INDEX_TENSOR_CACHE[cache_key] = (
        inventory_catalogs,
        skill_id_to_idx,
        indices,
    )
    return indices


def runtime_visible_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device | str | None = None,
    inventory_catalogs: dict[str, dict[str, Any]] | None = None,
) -> torch.Tensor:
    mask = torch.zeros(
        len(rows),
        len(skill_id_to_idx),
        dtype=torch.bool,
        device=device,
    )
    groups: dict[tuple[Any, ...], tuple[torch.Tensor, list[int]]] = {}
    direct_indices: dict[tuple[str, ...], torch.Tensor] = {}
    for row_idx, row in enumerate(rows):
        catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
        if catalog_id:
            if inventory_catalogs is None:
                raise ValueError(f"missing runtime inventory catalog: {catalog_id}")
            digest = str(row.get("inventory_catalog_digest") or "").strip()
            group_key = ("catalog", id(inventory_catalogs), catalog_id, digest)
            indices = _catalog_index_tensor(
                row,
                inventory_catalogs,
                skill_id_to_idx,
                device=mask.device,
            )
        else:
            declared = tuple(runtime_visible_skill_ids(row, inventory_catalogs))
            group_key = ("direct", declared)
            indices = direct_indices.get(declared)
            if indices is None:
                unknown = [
                    skill_id for skill_id in declared if skill_id not in skill_id_to_idx
                ]
                if unknown:
                    raise ValueError(
                        f"runtime inventory contains unknown skills: {unknown[:3]}"
                    )
                indices = torch.tensor(
                    [skill_id_to_idx[skill_id] for skill_id in declared],
                    dtype=torch.long,
                    device=mask.device,
                )
                if not int(indices.numel()):
                    raise ValueError("runtime inventory must contain at least one known skill")
                direct_indices[declared] = indices
        grouped = groups.get(group_key)
        if grouped is None:
            groups[group_key] = (indices, [row_idx])
        else:
            grouped[1].append(row_idx)
    for indices, row_indices in groups.values():
        row_tensor = torch.tensor(row_indices, dtype=torch.long, device=mask.device)
        mask[row_tensor.unsqueeze(1), indices.unsqueeze(0)] = True
    return mask


def capability_flags(row: dict[str, Any]) -> CapabilityFlags:
    raw = row.get("capabilities")
    capabilities = raw if isinstance(raw, dict) else {}

    def enabled(name: str) -> bool:
        if name in capabilities:
            return bool(capabilities[name])
        return bool(row.get(name))

    return CapabilityFlags(
        required_tool_set=enabled("required_tool_set"),
        current_state_route_set=enabled("current_state_route_set"),
        ordered_next_tool=enabled("ordered_next_tool"),
        actual_execution_result=enabled("actual_execution_result"),
        causal_branch_pair=enabled("causal_branch_pair"),
        verified_order_effect_pair=enabled("verified_order_effect_pair"),
        eligible_outcome_pair=enabled("eligible_outcome_pair"),
    )


_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")


def _normalized_text(value: Any) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").lower()))


def _token_recall(needle: str, haystack: str) -> float:
    needle_tokens = set(_TOKEN_RE.findall(needle.lower()))
    if not needle_tokens:
        return 0.0
    haystack_tokens = set(_TOKEN_RE.findall(haystack.lower()))
    return float(len(needle_tokens & haystack_tokens) / len(needle_tokens))


def actual_result_text(row: dict[str, Any]) -> str:
    explicit = str(row.get("actual_result_text") or row.get("actual_result") or "").strip()
    if explicit:
        return explicit
    return str(row.get("next_observation_text") or "").strip() if actual_causal_observation(row) else ""


def result_visibility(
    result_text: str,
    next_state_current: str,
    *,
    result_event_id: str | None = None,
    next_state_event_id: str | None = None,
) -> dict[str, Any]:
    result = _normalized_text(result_text)
    state = _normalized_text(next_state_current)
    same_event = bool(result_event_id and next_state_event_id and result_event_id == next_state_event_id)
    containment = bool(result and state and (result in state or state in result))
    overlap = _token_recall(result, state) if result and state else 0.0
    known = bool(result and state) or same_event
    visible = bool(
        same_event
        or containment
        or (result and state and overlap >= RESULT_VISIBLE_OVERLAP_THRESHOLD)
    )
    return {
        "result_visibility_known": known,
        "result_visible_in_next_state": visible,
        "result_overlap_score": float(overlap),
        "result_novel_for_memory": bool(known and not visible),
        "result_event_identity_match": same_event,
        "result_text_containment_match": containment,
        "result_visible_overlap_threshold": RESULT_VISIBLE_OVERLAP_THRESHOLD,
    }


def annotate_result_visibility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = [
        materialize_structured_current_state(row, replace_state_text=True)
        for row in rows
    ]
    prepared.sort(
        key=lambda row: (
            str(row.get("trajectory_id") or row.get("task_id") or ""),
            int(row.get("step_index") or 0),
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in prepared:
        trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "")
        grouped.setdefault(trajectory_id, []).append(row)
    output: list[dict[str, Any]] = []
    for trajectory_rows in grouped.values():
        for index, row in enumerate(trajectory_rows):
            copied = dict(row)
            result = actual_result_text(row)
            copied["actual_result_text"] = result
            copied["actual_result_executed"] = bool(result)
            copied["actual_result_source"] = str(row.get("observation_source") or "")
            next_row = trajectory_rows[index + 1] if index + 1 < len(trajectory_rows) else None
            next_state = str((next_row or {}).get("state_text_current") or row.get("next_state_text") or "")
            visibility = result_visibility(
                result,
                next_state,
                result_event_id=str(row.get("actual_result_event_id") or "") or None,
                next_state_event_id=str((next_row or {}).get("state_event_id") or "") or None,
            )
            copied.update(visibility)
            first_hidden: int | None = None
            if result:
                for future_index in range(index + 1, len(trajectory_rows)):
                    future_state = str(trajectory_rows[future_index].get("state_text_current") or "")
                    future_visibility = result_visibility(result, future_state)
                    if future_visibility["result_visibility_known"] and not future_visibility[
                        "result_visible_in_next_state"
                    ]:
                        first_hidden = int(trajectory_rows[future_index].get("step_index") or future_index)
                        break
            copied["first_future_step_without_result_visibility"] = first_hidden
            output.append(copied)
    return output


def cross_legal_branch_pair(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    *,
    inventory_catalogs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_a = str(row_a.get("next_skill_id") or row_a.get("target_skill_id") or "").strip()
    target_b = str(row_b.get("next_skill_id") or row_b.get("target_skill_id") or "").strip()
    inventory_a = set(runtime_visible_skill_ids(row_a, inventory_catalogs))
    inventory_b = set(runtime_visible_skill_ids(row_b, inventory_catalogs))
    cross_legal = bool(
        target_a
        and target_b
        and target_a != target_b
        and target_a in inventory_a
        and target_a in inventory_b
        and target_b in inventory_a
        and target_b in inventory_b
    )
    return {
        "cross_legal": cross_legal,
        "target_a": target_a,
        "target_b": target_b,
        "target_a_legal_in_a": target_a in inventory_a,
        "target_a_legal_in_b": target_a in inventory_b,
        "target_b_legal_in_a": target_b in inventory_a,
        "target_b_legal_in_b": target_b in inventory_b,
    }


def materialize_semantic_views(
    rows: list[dict[str, Any]],
    *,
    inventory_catalogs: dict[str, dict[str, Any]] | None = None,
) -> SemanticViews:
    annotated = annotate_result_visibility(rows)
    retrieval_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    branch_pairs: list[dict[str, Any]] = []
    order_pairs: list[dict[str, Any]] = []
    no_call_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for row in annotated:
        copied = dict(row)
        copied["vnext_schema_version"] = VNEXT_SCHEMA_VERSION
        if is_no_call_row(copied):
            no_call_rows.append(copied)
            continue
        try:
            legal = set(runtime_visible_skill_ids(copied, inventory_catalogs))
        except ValueError as exc:
            blockers.append({"row": str(copied.get("task_id") or ""), "reason": str(exc)})
            continue
        flags = capability_flags(copied)
        required = _unique(_string_list(copied.get("required_tool_set_skill_ids")))
        current_targets = _unique(
            _string_list(copied.get("current_state_route_set_skill_ids"))
        )
        if not current_targets:
            current_targets = _unique(
                _string_list(copied.get("skill_id") or copied.get("target_skill_id"))
            )
        ordered_target = str(
            copied.get("target_skill_id") or copied.get("skill_id") or ""
        ).strip()
        if flags.required_tool_set and required:
            if any(skill_id not in legal for skill_id in required):
                blockers.append(
                    {
                        "row": str(copied.get("task_id") or ""),
                        "reason": "required_tool_not_runtime_visible",
                    }
                )
            else:
                retrieval_rows.append(copied)
        if flags.current_state_route_set and current_targets:
            if all(current_target in legal for current_target in current_targets):
                static_rows.append(copied)
            else:
                blockers.append(
                    {
                        "row": str(copied.get("task_id") or ""),
                        "reason": "static_route_set_not_runtime_visible",
                    }
                )
        if flags.ordered_next_tool and int(copied.get("step_index") or 0) > 0 and ordered_target:
            if ordered_target in legal:
                transition_rows.append(copied)
            else:
                blockers.append(
                    {
                        "row": str(copied.get("task_id") or ""),
                        "reason": "ordered_target_not_runtime_visible",
                    }
                )
        if flags.actual_execution_result and copied.get("actual_result_executed"):
            correction_rows.append(copied)
        if flags.causal_branch_pair:
            branch_pairs.append(copied)
        if flags.verified_order_effect_pair:
            order_pairs.append(copied)
    report = {
        "schema_version": VNEXT_SCHEMA_VERSION,
        "source_row_count": len(rows),
        "retrieval_row_count": len(retrieval_rows),
        "static_route_row_count": len(static_rows),
        "ordered_transition_row_count": len(transition_rows),
        "result_correction_row_count": len(correction_rows),
        "causal_branch_pair_count": len(branch_pairs),
        "causal_order_pair_count": len(order_pairs),
        "no_call_row_count": len(no_call_rows),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "status": "ok" if not blockers else "action_required",
    }
    return SemanticViews(
        retrieval_rows=retrieval_rows,
        static_route_rows=static_rows,
        ordered_transition_rows=transition_rows,
        result_correction_rows=correction_rows,
        causal_branch_pairs=branch_pairs,
        causal_order_pairs=order_pairs,
        no_call_rows=no_call_rows,
        report=report,
    )
