#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.history_channel import (
    CAUSAL_STATE_CONTRACT,
    CURRENT_STATE_CONTRACT,
    actual_causal_observation,
    audit_history_channel_rows,
    materialize_structured_causal_state,
    materialize_structured_current_state,
    materialize_structured_retrieval_state,
)
from clstr.matched_multibench_data import LOCKED_SPLIT_SCHEMA
from clstr.vnext_data import VNEXT_SCHEMA_VERSION, annotate_result_visibility


EXECUTED_RESULT_SOURCE_IDS = frozenset(
    {
        "agentgym_agenttraj_l",
        "alfworld_official_train_replay",
        "clstr_dagger_expert_corrected_train_enriched",
        "hf_alfworld_admissible_success",
        "toolbench_g3",
        "tau2_official_successful_rollout_v1",
        "traject_bench",
        "webshop_expert_trajectories",
    }
)
VNEXT_SPLIT_SCHEMA = "clstr_task_group_hash_split_v1"
UNORDERED_BRANCH_SOURCE_ID = "verified_unordered_required_set"
SOURCE_CAUSAL_QUERY_CONTRACT = "clstr_source_causal_query_v1"


def _read_jsonl(path: Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= int(max_rows):
                break
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_clean_preflight(
    report_path: Path,
    *,
    skills_path: Path,
    retrieval_path: Path,
    trajectories_path: Path,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"missing clean preflight report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contract = report.get("preflight_contract") or {}
    if report.get("status") != "ok":
        raise ValueError("vNext view build requires a clean preflight with status=ok")
    if not bool(contract.get("structured_current_state_required")):
        raise ValueError("clean preflight did not require structured current state")
    if not bool(contract.get("protected_near_duplicate_blocking")):
        raise ValueError("clean preflight did not block protected near duplicates")
    expected_root = trajectories_path.resolve().parent
    observed_root = Path(
        str((report.get("composition") or {}).get("data_root") or "")
    ).resolve()
    if observed_root != expected_root:
        raise ValueError("clean preflight data root does not match vNext source root")
    expected_files = {
        "skill_pool.jsonl": skills_path,
        "retrieval.jsonl": retrieval_path,
        "trajectories.jsonl": trajectories_path,
    }
    observed_hashes = (report.get("composition") or {}).get("files_sha256") or {}
    mismatches = [
        name
        for name, path in expected_files.items()
        if str(observed_hashes.get(name) or "") != _file_digest(path)
    ]
    if mismatches:
        raise ValueError(f"clean preflight source digest mismatch: {mismatches}")
    benchmark_counts = (report.get("composition") or {}).get(
        "trajectories_by_benchmark"
    ) or {}
    protected_inputs = report.get("protected_eval_inputs") or {}
    leakage = report.get("leakage") or {}
    for benchmark, protected_key, leakage_key in (
        ("tau2", "tau2_test_rows_sha256", "tau2_test"),
        ("toolsandbox", "toolsandbox_test_rows_sha256", "toolsandbox_test"),
    ):
        if int(benchmark_counts.get(benchmark) or 0) <= 0:
            continue
        if not str(protected_inputs.get(protected_key) or ""):
            raise ValueError(
                f"clean preflight did not protect matched {benchmark} test rows"
            )
        protected_report = leakage.get(leakage_key) or {}
        if (
            int(protected_report.get("group_overlap_count") or 0) != 0
            or int(protected_report.get("trajectory_overlap_count") or 0) != 0
        ):
            raise ValueError(f"clean preflight reports matched {benchmark} leakage")
    return {
        "path": str(report_path.resolve()),
        "sha256": _file_digest(report_path),
        "schema_version": str(contract.get("schema_version") or ""),
        "source_files_sha256": {
            name: str(observed_hashes[name]) for name in sorted(expected_files)
        },
        "protected_eval_inputs": protected_inputs,
        "status": "ok",
    }


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _load_skills(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    skills = list(_read_jsonl(path))
    by_prefix: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in skills:
        skill_id = _skill_id(row)
        if not skill_id:
            raise ValueError("skill pool contains an empty skill_id")
        if skill_id in seen:
            raise ValueError(f"skill pool contains a duplicate skill_id: {skill_id}")
        seen.add(skill_id)
        prefix = skill_id.split("/", 1)[0]
        by_prefix[prefix].append(skill_id)
    return skills, {key: sorted(set(values)) for key, values in by_prefix.items()}


def _catalog_row(catalog_id: str, skill_ids: list[str], *, source: str) -> dict[str, Any]:
    ordered = sorted(set(skill_ids))
    digest = _json_digest(ordered)
    return {
        "inventory_catalog_id": catalog_id,
        "inventory_catalog_digest": digest,
        "inventory_source": source,
        "inventory_available_before_decision": True,
        "inventory_pool_size": len(ordered),
        "runtime_visible_skill_ids": ordered,
    }


def _build_catalogs(
    skills: list[dict[str, Any]],
    by_prefix: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    all_ids = [_skill_id(row) for row in skills]
    catalogs = {
        "public_global_67k": _catalog_row(
            "public_global_67k",
            all_ids,
            source="immutable_public_global_skill_pool",
        )
    }
    for prefix in ("alfworld", "webshop"):
        if by_prefix.get(prefix):
            catalogs[f"local_{prefix}"] = _catalog_row(
                f"local_{prefix}",
                by_prefix[prefix],
                source=f"runtime_environment_{prefix}_catalog",
            )
    return catalogs


def _catalog_for_trajectory(
    row: dict[str, Any],
    catalogs: dict[str, dict[str, Any]],
    known_skill_ids: set[str],
) -> str:
    declared = sorted(
        {
            str(item).strip()
            for item in row.get("tool_inventory_skill_ids") or []
            if str(item).strip()
        }
    )
    if declared:
        unknown = sorted(set(declared) - known_skill_ids)
        if unknown:
            raise ValueError(f"runtime-declared inventory contains unknown skills: {unknown[:4]}")
        digest = _json_digest(declared)
        catalog_id = f"runtime_declared_{digest[:20]}"
        if catalog_id not in catalogs:
            catalogs[catalog_id] = _catalog_row(
                catalog_id,
                declared,
                source="source_declared_runtime_tool_inventory",
            )
        return catalog_id
    benchmark = str(row.get("benchmark") or "").strip().lower()
    if benchmark == "alfworld" and "local_alfworld" in catalogs:
        return "local_alfworld"
    if benchmark == "webshop" and "local_webshop" in catalogs:
        return "local_webshop"
    return "public_global_67k"


def _with_catalog(
    row: dict[str, Any],
    catalog_id: str,
    catalogs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    catalog = catalogs[catalog_id]
    copied = dict(row)
    copied["runtime_visible_catalog_id"] = catalog_id
    copied["inventory_catalog_digest"] = catalog["inventory_catalog_digest"]
    copied["inventory_source"] = catalog["inventory_source"]
    copied["inventory_available_before_decision"] = True
    copied["inventory_pool_size"] = int(catalog["inventory_pool_size"])
    copied["inventory_protocol"] = (
        "public_global" if catalog_id == "public_global_67k" else "environment_local"
    )
    return copied


def _trajectory_type(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return str(row.get("trajectory_type") or provenance.get("trajectory_type") or "").strip().lower()


def _source_identity(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    value = str(
        provenance.get("source_id")
        or row.get("source")
        or row.get("benchmark")
        or ""
    ).strip()
    if not value:
        raise ValueError("semantic data materialization requires a source identity")
    return value


def _retrieval_requires_canonical_cross_binding(row: dict[str, Any]) -> bool:
    source = _source_identity(row).lower()
    return bool(source.startswith("trajectory_derived_") or source == "traject_bench")


def _provenance_mapping(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("provenance")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _trajectory_causal_event(row: dict[str, Any]) -> dict[str, Any] | None:
    skill_id = str(row.get("target_skill_id") or row.get("skill_id") or "").strip()
    action_text = str(row.get("action_text") or "").strip()
    if not skill_id or not action_text:
        return None
    result_text = str(row.get("actual_result_text") or "").strip()
    result_executed = bool(row.get("actual_result_executed") and result_text)
    # When the successor current observation already contains this result, do
    # not duplicate it in the prefix.  The executed skill/action still records
    # the causal event and delayed-hidden results remain available to memory.
    if bool(row.get("result_visible_in_next_state")):
        result_text = ""
        result_executed = False
    return {
        "step_index": int(row.get("step_index") or 0),
        "skill_id": skill_id,
        "action_text": action_text,
        "result_text": result_text,
        "result_executed": result_executed,
    }


def _attach_trajectory_causal_states(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError("causal-state materialization requires trajectory_id")
        grouped[trajectory_id].append(row)
    output: list[dict[str, Any]] = []
    dropped_events = 0
    for trajectory_id in sorted(grouped):
        trajectory_rows = sorted(
            grouped[trajectory_id],
            key=lambda row: int(row.get("step_index") or 0),
        )
        prefix: list[dict[str, Any]] = []
        for row in trajectory_rows:
            prepared = materialize_structured_causal_state(row, prefix)
            prepared["decision_step_index"] = int(row.get("step_index") or 0)
            prepared["causal_query_materialization_source"] = (
                "canonical_executed_trajectory_prefix"
            )
            prepared["future_label_skill_ids"] = [
                str(row.get("target_skill_id") or row.get("skill_id") or "").strip()
            ]
            output.append(prepared)
            event = _trajectory_causal_event(row)
            if event is None:
                dropped_events += 1
            else:
                prefix.append(event)
    return output, {
        "protocol": CAUSAL_STATE_CONTRACT,
        "trajectory_count": len(grouped),
        "row_count": len(output),
        "event_count": sum(int(row.get("causal_prefix_event_count") or 0) for row in output),
        "missing_executed_event_count": int(dropped_events),
        "zero_history_exact_current_count": sum(
            int(
                int(row.get("causal_prefix_event_count") or 0) == 0
                and str(row.get("state_text_causal") or "")
                == str(row.get("state_text_current") or "")
            )
            for row in output
        ),
    }


def _credible_executed_result(row: dict[str, Any]) -> tuple[bool, str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source_id = str(provenance.get("source_id") or "").strip()
    action = str(row.get("action_text") or "").strip()
    skill_id = str(row.get("skill_id") or "").strip()
    result = str(row.get("next_observation_text") or "").strip()
    credible = bool(
        source_id in EXECUTED_RESULT_SOURCE_IDS
        and action
        and skill_id
        and result
        and actual_causal_observation(row)
    )
    return credible, f"executed_trace:{source_id}" if credible else ""


def _canonical_trajectory_rows(
    raw_rows: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    known_skill_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sources_by_raw_trajectory_id: dict[str, set[str]] = defaultdict(set)
    for raw in raw_rows:
        row = materialize_structured_current_state(raw, replace_state_text=True)
        trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "").strip()
        if not trajectory_id:
            raise ValueError("trajectory row lacks trajectory_id/task_id")
        source = _source_identity(row)
        grouped[(source, trajectory_id)].append(row)
        sources_by_raw_trajectory_id[trajectory_id].add(source)
    colliding_raw_ids = {
        trajectory_id
        for trajectory_id, sources in sources_by_raw_trajectory_id.items()
        if len(sources) > 1
    }
    canonical: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    counts["trajectory_id_namespace_collision_count"] = len(colliding_raw_ids)
    for (source, raw_trajectory_id), rows in grouped.items():
        sources = {_source_identity(row) for row in rows}
        steps = [int(row.get("step_index") or 0) for row in rows]
        goals = {
            str(row.get("goal_text") or row.get("task_text") or "").strip()
            for row in rows
        }
        if len(sources) != 1:
            raise ValueError("one source-qualified trajectory crosses source identities")
        if len(steps) != len(set(steps)):
            raise ValueError("one trajectory contains a duplicate step index")
        if len(goals) != 1:
            raise ValueError("one trajectory_id crosses multiple task/goal boundaries")
        rows.sort(key=lambda row: int(row.get("step_index") or 0))
        trajectory_id = (
            f"{source}::{raw_trajectory_id}"
            if raw_trajectory_id in colliding_raw_ids
            else raw_trajectory_id
        )
        required_tools = sorted(
            {
                str(row.get("skill_id") or "").strip()
                for row in rows
                if str(row.get("skill_id") or "").strip()
            }
        )
        for index, row in enumerate(rows):
            namespaced = raw_trajectory_id in colliding_raw_ids
            row = dict(row)
            row["source_trajectory_id"] = raw_trajectory_id
            row["trajectory_id"] = trajectory_id
            row["trajectory_namespace_source"] = source
            row["trajectory_id_namespaced"] = bool(namespaced)
            copied = _with_catalog(
                row,
                _catalog_for_trajectory(row, catalogs, known_skill_ids),
                catalogs,
            )
            copied["vnext_schema_version"] = VNEXT_SCHEMA_VERSION
            copied["target_skill_id"] = str(row.get("skill_id") or "").strip()
            copied["required_tool_set_skill_ids"] = required_tools if index == 0 else []
            previous = rows[index - 1] if index > 0 else None
            adjacency_ok = bool(
                previous is not None
                and int(row.get("step_index") or index)
                == int(previous.get("step_index") or index - 1) + 1
                and (
                    not str(previous.get("next_skill_id") or "").strip()
                    or str(previous.get("next_skill_id") or "").strip()
                    == copied["target_skill_id"]
                )
            )
            parallel = _trajectory_type(row) == "parallel"
            result_ok, result_source = _credible_executed_result(row)
            copied["capabilities"] = {
                "required_tool_set": bool(index == 0 and required_tools),
                "current_state_route_set": False,
                "ordered_next_tool": bool(adjacency_ok and not parallel),
                "actual_execution_result": bool(result_ok),
                "causal_branch_pair": False,
                "verified_order_effect_pair": False,
                "eligible_outcome_pair": False,
            }
            copied["temporal_adjacency_verified"] = adjacency_ok
            copied["actual_result_text"] = (
                str(row.get("next_observation_text") or "").strip() if result_ok else ""
            )
            copied["actual_result_executed"] = bool(result_ok)
            copied["actual_result_source"] = result_source
            copied["observation_source"] = result_source
            copied.pop("loss_mask", None)
            canonical.append(copied)
            counts[f"benchmark:{str(row.get('benchmark') or 'unknown')}"] += 1
            counts["ordered"] += int(adjacency_ok and not parallel)
            counts["parallel"] += int(parallel)
            counts["actual_result"] += int(result_ok)
            counts["trajectory_rows_namespaced"] += int(namespaced)
    annotated = annotate_result_visibility(canonical)
    causal_rows, causal_report = _attach_trajectory_causal_states(annotated)
    for name, value in causal_report.items():
        if isinstance(value, int):
            counts[f"causal_state:{name}"] = int(value)
    return causal_rows, dict(sorted(counts.items()))


def _group_retrieval_rows(
    rows: Iterable[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    trajectories: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    catalog = catalogs["public_global_67k"]
    trajectory_index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for trajectory_row in trajectories or []:
        for trajectory_id in {
            str(trajectory_row.get("source_trajectory_id") or "").strip(),
            str(trajectory_row.get("trajectory_id") or "").strip(),
        }:
            if trajectory_id:
                trajectory_index[
                    (trajectory_id, int(trajectory_row.get("step_index") or 0))
                ].append(trajectory_row)
    for index, row in enumerate(rows):
        source = str(row.get("source") or "unknown")
        query_id = str(row.get("query_id") or f"row-{index}")
        try:
            structured = materialize_structured_retrieval_state(
                row,
                replace_state_text=True,
            )
        except ValueError:
            continue
        positives = []
        positives.extend(row.get("positive_skill_ids") or [])
        if row.get("positive_skill_id"):
            positives.append(row["positive_skill_id"])
        positive_ids = {str(item) for item in positives if str(item)}
        provenance = _provenance_mapping(row)
        raw_trajectory_id = str(provenance.get("trajectory_id") or "").strip()
        raw_step = provenance.get("step_index")
        target_kind = str(provenance.get("target") or "current").strip().lower()
        if target_kind not in {"current", "next"}:
            raise ValueError(f"unsupported retrieval temporal target: {target_kind}")
        source_step = None if raw_step in (None, "") else int(raw_step)
        decision_step = (
            None
            if source_step is None
            else int(source_step) + int(target_kind == "next")
        )
        bound: dict[str, Any] | None = None
        if raw_trajectory_id and decision_step is not None:
            candidates = trajectory_index.get((raw_trajectory_id, decision_step), [])
            target_matches = [
                candidate
                for candidate in candidates
                if str(candidate.get("target_skill_id") or "") in positive_ids
            ]
            if len(target_matches) > 1:
                raise ValueError(
                    "retrieval causal query maps to multiple target-aligned trajectory rows"
                )
            bound = target_matches[0] if target_matches else None
        if bound is not None:
            current_text = str(bound.get("state_text_current") or "").strip()
            causal_text = str(bound.get("state_text_causal") or "").strip()
            current_components = dict(bound.get("current_state_components") or {})
            current_materialization_source = str(
                bound.get("current_state_materialization_source") or ""
            )
            current_components_sha256 = str(
                bound.get("current_state_components_sha256") or ""
            )
            causal_contract = str(bound.get("causal_state_contract") or "")
            causal_events = list(bound.get("causal_prefix_events") or [])
            causal_prefix_sha256 = str(bound.get("causal_prefix_sha256") or "")
            zero_history_exact = bool(
                bound.get("zero_history_causal_equals_current")
            )
            causal_source = "cross_bound_canonical_trajectory_decision"
            decision_trajectory_id = str(bound.get("trajectory_id") or "")
            decision_step = int(bound.get("step_index") or 0)
        else:
            current_text = str(structured["state_text_current"])
            causal_text = str(structured["state_text_full"] or current_text).strip()
            current_components = dict(structured["current_state_components"])
            current_materialization_source = str(
                structured["current_state_materialization_source"]
            )
            current_components_sha256 = str(
                structured["current_state_components_sha256"]
            )
            causal_contract = SOURCE_CAUSAL_QUERY_CONTRACT
            causal_events = []
            causal_prefix_sha256 = hashlib.sha256(b"[]").hexdigest()
            zero_history_exact = bool(causal_text == current_text)
            causal_source = "source_declared_retrieval_decision_query"
            decision_trajectory_id = raw_trajectory_id
        if not causal_text:
            raise ValueError("retrieval row lacks a causal decision query")
        key = (source, query_id, causal_text)
        target = grouped.setdefault(
            key,
            {
                "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                "source": source,
                "query_id": query_id,
                "state_text": current_text,
                "state_text_current": current_text,
                "state_text_full": structured["state_text_full"],
                "state_text_causal": causal_text,
                "causal_state_contract": causal_contract,
                "causal_state_sha256": hashlib.sha256(
                    causal_text.encode("utf-8")
                ).hexdigest(),
                "causal_prefix_events": causal_events,
                "causal_prefix_event_count": len(causal_events),
                "causal_prefix_sha256": causal_prefix_sha256,
                "zero_history_causal_equals_current": zero_history_exact,
                "causal_query_materialization_source": causal_source,
                "source_step_index": source_step,
                "decision_step_index": decision_step,
                "decision_trajectory_id": decision_trajectory_id,
                "retrieval_temporal_target": target_kind,
                "current_state_components": current_components,
                "current_state_contract": CURRENT_STATE_CONTRACT,
                "current_state_materialization_source": current_materialization_source,
                "current_state_full_text_sha256": structured[
                    "current_state_full_text_sha256"
                ],
                "current_state_components_sha256": current_components_sha256,
                "retrieval_query_materialization_source": structured[
                    "retrieval_query_materialization_source"
                ],
                "runtime_visible_catalog_id": "public_global_67k",
                "inventory_catalog_digest": catalog["inventory_catalog_digest"],
                "inventory_source": catalog["inventory_source"],
                "inventory_available_before_decision": True,
                "inventory_pool_size": int(catalog["inventory_pool_size"]),
                "inventory_protocol": "public_global",
                "required_tool_set_skill_ids": [],
                "explicit_negative_skill_ids": [],
                "capabilities": {
                    "required_tool_set": True,
                    "current_state_route_set": False,
                    "ordered_next_tool": False,
                    "actual_execution_result": False,
                    "causal_branch_pair": False,
                    "verified_order_effect_pair": False,
                    "eligible_outcome_pair": False,
                },
                "provenance": row.get("provenance"),
            },
        )
        target["required_tool_set_skill_ids"] = sorted(
            set(target["required_tool_set_skill_ids"]) | {str(item) for item in positives if str(item)}
        )
        target["explicit_negative_skill_ids"] = sorted(
            set(target["explicit_negative_skill_ids"])
            | {str(item) for item in row.get("negative_skill_ids") or [] if str(item)}
        )
    known_skills = {
        str(item)
        for item in catalog.get("runtime_visible_skill_ids") or []
        if str(item)
    }
    output: list[dict[str, Any]] = []
    for row in grouped.values():
        positives = {
            str(item)
            for item in row.get("required_tool_set_skill_ids") or []
            if str(item)
        }
        negatives = {
            str(item)
            for item in row.get("explicit_negative_skill_ids") or []
            if str(item)
        }
        unknown = sorted((positives | negatives) - known_skills)
        if unknown:
            raise ValueError(f"retrieval row references unknown skills: {unknown[:4]}")
        collisions = positives & negatives
        row["required_tool_set_skill_ids"] = sorted(positives)
        row["future_label_skill_ids"] = sorted(positives)
        row["future_label_contract"] = "separate_supervision_only_v1"
        row["explicit_negative_skill_ids"] = sorted(negatives - positives)
        row["removed_positive_negative_collision_count"] = len(collisions)
        if positives:
            output.append(row)
    return output


def _exclude_unbound_trajectory_retrieval_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        bound = (
            str(row.get("causal_query_materialization_source") or "")
            == "cross_bound_canonical_trajectory_decision"
        )
        if _retrieval_requires_canonical_cross_binding(row) and not bound:
            exclusions.append(
                {
                    "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                    "exclusion_kind": "retrieval",
                    "exclusion_reason": "trajectory_retrieval_lacks_canonical_cross_binding",
                    "source_id": _source_identity(row),
                    "query_id": str(row.get("query_id") or ""),
                    "decision_trajectory_id": str(
                        row.get("decision_trajectory_id") or ""
                    ),
                    "decision_step_index": row.get("decision_step_index"),
                }
            )
            continue
        kept.append(row)
    return kept, exclusions


def _normalized_audit_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _trajectory_audit_signature(rows: list[dict[str, Any]], *, normalized: bool) -> str:
    def text(value: Any) -> str:
        return _normalized_audit_text(value) if normalized else str(value or "").strip()

    return _json_digest(
        [
            {
                "goal": text(row.get("goal_text") or row.get("task_text")),
                "state": text(row.get("state_text_current")),
                "skill": str(row.get("target_skill_id") or ""),
                "action": text(row.get("action_text")),
                "result": text(row.get("actual_result_text")),
                "inventory": str(row.get("inventory_catalog_digest") or ""),
            }
            for row in sorted(rows, key=lambda item: int(item.get("step_index") or 0))
        ]
    )


def _deduplicate_semantic_rows(
    trajectories: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            raise ValueError("deduplication requires trajectory identities")
        by_trajectory[trajectory_id].append(row)

    normalized_groups: dict[str, list[str]] = defaultdict(list)
    exact_signatures: dict[str, str] = {}
    for trajectory_id, rows in by_trajectory.items():
        exact_signatures[trajectory_id] = _trajectory_audit_signature(
            rows,
            normalized=False,
        )
        normalized_groups[
            _trajectory_audit_signature(rows, normalized=True)
        ].append(trajectory_id)

    excluded_trajectories: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    exact_duplicate_trajectories = 0
    normalized_duplicate_trajectories = 0
    for group_digest, trajectory_ids in sorted(normalized_groups.items()):
        ordered = sorted(
            trajectory_ids,
            key=lambda trajectory_id: (
                _source_identity(by_trajectory[trajectory_id][0]),
                trajectory_id,
            ),
        )
        kept = ordered[0]
        for duplicate in ordered[1:]:
            reason = (
                "exact_duplicate_trajectory"
                if exact_signatures[duplicate] == exact_signatures[kept]
                else "normalized_duplicate_trajectory"
            )
            exact_duplicate_trajectories += int(reason == "exact_duplicate_trajectory")
            normalized_duplicate_trajectories += int(
                reason == "normalized_duplicate_trajectory"
            )
            excluded_trajectories.add(duplicate)
            exclusions.append(
                {
                    "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                    "exclusion_kind": "trajectory",
                    "exclusion_reason": reason,
                    "source_id": _source_identity(by_trajectory[duplicate][0]),
                    "trajectory_id": duplicate,
                    "kept_trajectory_id": kept,
                    "row_count": len(by_trajectory[duplicate]),
                    "duplicate_group_digest": group_digest,
                }
            )
    deduplicated_trajectories = [
        row
        for row in trajectories
        if str(row.get("trajectory_id") or "") not in excluded_trajectories
    ]

    retrieval_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval_rows:
        signature = _json_digest(
            {
                "source_id": _source_identity(row),
                "state_text_current": _normalized_audit_text(
                    row.get("state_text_current")
                ),
                "positives": sorted(
                    str(item)
                    for item in row.get("required_tool_set_skill_ids") or []
                    if str(item)
                ),
                "inventory_catalog_digest": str(
                    row.get("inventory_catalog_digest") or ""
                ),
            }
        )
        retrieval_groups[signature].append(row)
    deduplicated_retrieval: list[dict[str, Any]] = []
    duplicate_retrieval_rows = 0
    for group_digest, rows in sorted(retrieval_groups.items()):
        ordered = sorted(rows, key=lambda row: str(row.get("query_id") or ""))
        deduplicated_retrieval.append(ordered[0])
        for duplicate in ordered[1:]:
            duplicate_retrieval_rows += 1
            exclusions.append(
                {
                    "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                    "exclusion_kind": "retrieval",
                    "exclusion_reason": "normalized_duplicate_retrieval_query",
                    "source_id": _source_identity(duplicate),
                    "query_id": str(duplicate.get("query_id") or ""),
                    "kept_query_id": str(ordered[0].get("query_id") or ""),
                    "duplicate_group_digest": group_digest,
                }
            )

    template_counts: Counter[str] = Counter()
    for trajectory_id, rows in by_trajectory.items():
        if trajectory_id in excluded_trajectories:
            continue
        first = sorted(rows, key=lambda row: int(row.get("step_index") or 0))[0]
        template_digest = _json_digest(
            {
                "source_id": _source_identity(first),
                "goal": _normalized_audit_text(
                    first.get("goal_text") or first.get("task_text")
                ),
                "skill_sequence": [
                    str(row.get("target_skill_id") or "")
                    for row in sorted(
                        rows,
                        key=lambda row: int(row.get("step_index") or 0),
                    )
                ],
            }
        )
        template_counts[template_digest] += 1
    exclusions.sort(
        key=lambda row: (
            str(row.get("exclusion_kind") or ""),
            str(row.get("trajectory_id") or row.get("query_id") or ""),
        )
    )
    return deduplicated_trajectories, deduplicated_retrieval, exclusions, {
        "protocol": "deterministic_normalized_dedup_v1",
        "input_trajectory_count": len(by_trajectory),
        "retained_trajectory_count": len(by_trajectory) - len(excluded_trajectories),
        "exact_duplicate_trajectory_count": int(exact_duplicate_trajectories),
        "normalized_duplicate_trajectory_count": int(
            normalized_duplicate_trajectories
        ),
        "input_trajectory_row_count": len(trajectories),
        "retained_trajectory_row_count": len(deduplicated_trajectories),
        "input_retrieval_row_count": len(retrieval_rows),
        "retained_retrieval_row_count": len(deduplicated_retrieval),
        "duplicate_retrieval_row_count": int(duplicate_retrieval_rows),
        "exclusion_count": len(exclusions),
        "template_group_count": len(template_counts),
        "repeated_template_group_count": sum(
            int(count > 1) for count in template_counts.values()
        ),
        "maximum_template_repetition": max(template_counts.values(), default=0),
        "automatic_template_deletion": False,
    }


def _shortcut_predictability_report(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train = [
        row
        for row in train_rows
        if str(row.get("target_skill_id") or "")
        and bool((row.get("capabilities") or {}).get("ordered_next_tool"))
    ]
    dev = [
        row
        for row in dev_rows
        if str(row.get("target_skill_id") or "")
        and bool((row.get("capabilities") or {}).get("ordered_next_tool"))
    ]
    if not train or not dev:
        return {
            "protocol": "train_fit_dev_source_step_majority_v1",
            "status": "action_required",
            "reason": "empty_ordered_train_or_dev_rows",
            "train_row_count": len(train),
            "dev_row_count": len(dev),
        }

    feature_functions = {
        "source_only": lambda row: (_source_identity(row),),
        "source_step": lambda row: (
            _source_identity(row),
            int(row.get("step_index") or 0),
        ),
        "source_type_step": lambda row: (
            _source_identity(row),
            _trajectory_type(row),
            int(row.get("step_index") or 0),
        ),
    }
    global_counts = Counter(str(row.get("target_skill_id") or "") for row in train)
    global_target = min(
        (
            (-count, target)
            for target, count in global_counts.items()
        )
    )[1]
    metrics: dict[str, Any] = {}
    for name, feature in feature_functions.items():
        counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        for row in train:
            counts[feature(row)][str(row.get("target_skill_id") or "")] += 1
        predictors = {
            key: min((-count, target) for target, count in target_counts.items())[1]
            for key, target_counts in counts.items()
        }
        known = 0
        correct = 0
        for row in dev:
            key = feature(row)
            known += int(key in predictors)
            predicted = predictors.get(key, global_target)
            correct += int(predicted == str(row.get("target_skill_id") or ""))
        metrics[name] = {
            "accuracy": float(correct) / len(dev),
            "known_feature_coverage": float(known) / len(dev),
            "known_feature_count": int(known),
            "category_count": len(predictors),
        }
    return {
        "protocol": "train_fit_dev_source_step_majority_v1",
        "status": "ok",
        "train_row_count": len(train),
        "dev_row_count": len(dev),
        "global_majority_target": global_target,
        "metrics": metrics,
    }


def _verified_robust_prefix_kind(row: dict[str, Any]) -> tuple[str | None, str | None]:
    annotation = (
        row.get("robust_prefix_annotation")
        if isinstance(row.get("robust_prefix_annotation"), dict)
        else {}
    )
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    raw_error_count = annotation.get(
        "executed_error_count",
        row.get("executed_error_count", provenance.get("executed_error_count")),
    )
    if raw_error_count in (None, ""):
        return None, None
    try:
        error_count = int(raw_error_count)
    except (TypeError, ValueError):
        return None, "invalid_executed_error_count"
    if error_count <= 0:
        return None, None
    executed = bool(
        annotation.get(
            "perturbation_executed",
            row.get("perturbation_executed", provenance.get("perturbation_executed")),
        )
    )
    state_verified = bool(
        annotation.get(
            "resulting_state_verified",
            row.get(
                "resulting_state_verified",
                provenance.get("resulting_state_verified"),
            ),
        )
    )
    target_verified = bool(
        annotation.get(
            "recovery_target_verified",
            row.get(
                "recovery_target_verified",
                provenance.get("recovery_target_verified"),
            ),
        )
    )
    verification_id = str(
        annotation.get("verification_id")
        or row.get("recovery_verification_id")
        or provenance.get("recovery_verification_id")
        or ""
    ).strip()
    raw_error_steps = annotation.get(
        "executed_error_step_indices",
        row.get(
            "executed_error_step_indices",
            provenance.get("executed_error_step_indices"),
        ),
    )
    if not isinstance(raw_error_steps, list):
        return None, "missing_executed_error_step_indices"
    try:
        error_steps = sorted({int(item) for item in raw_error_steps})
    except (TypeError, ValueError):
        return None, "invalid_executed_error_step_indices"
    current_step = int(row.get("step_index") or 0)
    if (
        len(error_steps) < error_count
        or any(step < 0 or step >= current_step for step in error_steps)
    ):
        return None, "misaligned_executed_error_step_indices"
    if not (executed and state_verified and target_verified and verification_id):
        return None, "incomplete_executed_recovery_verification"
    if bool(annotation.get("recovery_prefix") or row.get("recovery_prefix")):
        return "recovery_prefix", None
    if error_count == 1:
        return "one_error_prefix", None
    return "two_or_more_error_prefix", None


def _capability_coverage_report(
    trajectories: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prefixes, _by_step = _prefix_records(trajectories)
    source_rows: Counter[str] = Counter()
    source_trajectories: dict[str, set[str]] = defaultdict(set)
    capability_combinations: Counter[str] = Counter()
    prefix_lengths: Counter[int] = Counter()
    result_delay: Counter[str] = Counter()
    robust_prefix_counts: Counter[str] = Counter()
    robust_prefix_downgrades: Counter[str] = Counter()
    for row in trajectories:
        source = _source_identity(row)
        source_rows[source] += 1
        source_trajectories[source].add(str(row.get("trajectory_id") or ""))
        capabilities = row.get("capabilities") or {}
        enabled = sorted(
            name
            for name in (
                "required_tool_set",
                "ordered_next_tool",
                "actual_execution_result",
            )
            if bool(capabilities.get(name))
        )
        capability_combinations["+".join(enabled) if enabled else "none"] += 1
        prefix = prefixes[
            (
                str(row.get("trajectory_id") or ""),
                int(row.get("step_index") or 0),
            )
        ]
        prefix_lengths[int(prefix["length"])] += 1
        if bool(row.get("actual_result_executed")):
            if bool(row.get("result_novel_for_memory")):
                result_delay["novel"] += 1
            if bool(row.get("result_visible_in_next_state")):
                result_delay["visible_in_next_state"] += 1
            hidden_step = row.get("first_future_step_without_result_visibility")
            if hidden_step is None:
                result_delay["no_aligned_hidden_future"] += 1
            else:
                delay = int(hidden_step) - int(row.get("step_index") or 0)
                result_delay[f"hidden_after_{max(delay, 0)}_steps"] += 1
        robust_kind, downgrade = _verified_robust_prefix_kind(row)
        if robust_kind:
            robust_prefix_counts[robust_kind] += 1
        if downgrade:
            robust_prefix_downgrades[downgrade] += 1
    retrieval_source_rows = Counter(_source_identity(row) for row in retrieval_rows)
    retrieval_causal_sources = Counter(
        str(row.get("causal_query_materialization_source") or "missing")
        for row in retrieval_rows
    )
    retrieval_decision_steps = Counter(
        "unknown"
        if row.get("decision_step_index") in (None, "")
        else str(int(row["decision_step_index"]))
        for row in retrieval_rows
    )
    trajectory_like_unbound = sum(
        int(
            _retrieval_requires_canonical_cross_binding(row)
            and str(row.get("causal_query_materialization_source") or "")
            != "cross_bound_canonical_trajectory_decision"
        )
        for row in retrieval_rows
    )
    return {
        "protocol": "capability_prefix_result_coverage_v1",
        "trajectory_row_count": len(trajectories),
        "retrieval_row_count": len(retrieval_rows),
        "trajectory_rows_by_source": dict(sorted(source_rows.items())),
        "trajectories_by_source": {
            source: len(ids) for source, ids in sorted(source_trajectories.items())
        },
        "retrieval_rows_by_source": dict(sorted(retrieval_source_rows.items())),
        "retrieval_causal_materialization": dict(
            sorted(retrieval_causal_sources.items())
        ),
        "retrieval_decision_step_counts": dict(sorted(retrieval_decision_steps.items())),
        "retrieval_unique_current_query_count": len(
            {str(row.get("state_text_current") or "") for row in retrieval_rows}
        ),
        "retrieval_unique_causal_query_count": len(
            {str(row.get("state_text_causal") or "") for row in retrieval_rows}
        ),
        "trajectory_like_unbound_retrieval_row_count": int(trajectory_like_unbound),
        "capability_combination_counts": dict(sorted(capability_combinations.items())),
        "prefix_length_counts": {
            str(length): int(count) for length, count in sorted(prefix_lengths.items())
        },
        "result_delay_counts": dict(sorted(result_delay.items())),
        "verified_robust_prefix_counts": dict(sorted(robust_prefix_counts.items())),
        "robust_prefix_downgrade_counts": dict(
            sorted(robust_prefix_downgrades.items())
        ),
    }


def _causal_retrieval_release_blockers(
    coverage_report: dict[str, Any],
) -> list[str]:
    """Reject trajectory-like retrieval rows without an audited time boundary."""

    blockers: list[str] = []
    if int(coverage_report.get("trajectory_like_unbound_retrieval_row_count") or 0):
        blockers.append("trajectory_like_retrieval_row_lacks_canonical_cross_binding")
    return blockers


def _hash_split(identity: dict[str, Any], *, dev_fraction: float, seed: str) -> str:
    fraction = float(dev_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("vNext dev_fraction must be in (0, 0.5)")
    digest = _json_digest({"seed": str(seed), "identity": identity})
    value = int(digest[:16], 16) / float(16**16)
    return "dev" if value < fraction else "train"


def _locked_split_assignment(row: dict[str, Any]) -> tuple[str, str, str] | None:
    split = str(row.get("locked_data_split") or "").strip().lower()
    group = str(row.get("locked_split_group_identity") or "").strip()
    manifest = str(row.get("locked_split_manifest_sha256") or "").strip()
    schema = str(row.get("locked_split_schema_version") or "").strip()
    if not any((split, group, manifest, schema)):
        return None
    if not all((split, group, manifest, schema)):
        raise ValueError("locked semantic split fields must be complete")
    if split not in {"train", "dev"}:
        raise ValueError("locked semantic split must be train or dev")
    if schema != LOCKED_SPLIT_SCHEMA:
        raise ValueError("locked semantic split uses an unsupported schema")
    if not re.fullmatch(r"[0-9a-f]{64}", group):
        raise ValueError("locked semantic group identity is not SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest):
        raise ValueError("locked semantic manifest identity is not SHA256")
    return split, group, manifest


def _split_semantic_rows(
    trajectories: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    *,
    dev_fraction: float,
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {
        "trajectory_train": [],
        "trajectory_dev": [],
        "retrieval_train": [],
        "retrieval_dev": [],
    }
    trajectory_group_splits: dict[str, str] = {}
    retrieval_group_splits: dict[str, str] = {}
    locked_group_splits: dict[str, str] = {}
    locked_group_manifests: dict[str, str] = {}
    locked_manifests: set[str] = set()
    locked_trajectory_rows = 0
    locked_retrieval_rows = 0
    for row in trajectories:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        identity = {
            "kind": "trajectory_task_group",
            "benchmark": str(row.get("benchmark") or ""),
            "goal_text": str(row.get("goal_text") or row.get("task_text") or ""),
            "trajectory_type": _trajectory_type(row),
            "domain": str(provenance.get("domain") or ""),
        }
        locked = _locked_split_assignment(row)
        if locked is None:
            group_id = _json_digest(identity)
            split = trajectory_group_splits.setdefault(
                group_id,
                _hash_split(identity, dev_fraction=dev_fraction, seed=seed),
            )
            split_schema = VNEXT_SPLIT_SCHEMA
            assignment_source = "deterministic_hash"
        else:
            split, group_id, manifest_sha256 = locked
            if locked_group_splits.setdefault(group_id, split) != split:
                raise ValueError("one locked semantic group crosses train/dev")
            if locked_group_manifests.setdefault(group_id, manifest_sha256) != manifest_sha256:
                raise ValueError("one locked semantic group crosses split manifests")
            locked_manifests.add(manifest_sha256)
            locked_trajectory_rows += 1
            split_schema = LOCKED_SPLIT_SCHEMA
            assignment_source = "locked_manifest"
        copied = dict(row)
        copied["data_split"] = split
        copied["split_group_identity"] = group_id
        copied["split_schema_version"] = split_schema
        copied["split_assignment_source"] = assignment_source
        splits[f"trajectory_{split}"].append(copied)
    for row in retrieval_rows:
        identity = {
            "kind": "retrieval_query_group",
            "source": str(row.get("source") or ""),
            "state_text_current": str(row.get("state_text_current") or ""),
        }
        locked = _locked_split_assignment(row)
        if locked is None:
            group_id = _json_digest(identity)
            split = retrieval_group_splits.setdefault(
                group_id,
                _hash_split(identity, dev_fraction=dev_fraction, seed=seed),
            )
            split_schema = VNEXT_SPLIT_SCHEMA
            assignment_source = "deterministic_hash"
        else:
            split, group_id, manifest_sha256 = locked
            if locked_group_splits.setdefault(group_id, split) != split:
                raise ValueError("one locked semantic group crosses train/dev")
            if locked_group_manifests.setdefault(group_id, manifest_sha256) != manifest_sha256:
                raise ValueError("one locked semantic group crosses split manifests")
            locked_manifests.add(manifest_sha256)
            locked_retrieval_rows += 1
            split_schema = LOCKED_SPLIT_SCHEMA
            assignment_source = "locked_manifest"
        copied = dict(row)
        copied["data_split"] = split
        copied["split_group_identity"] = group_id
        copied["split_schema_version"] = split_schema
        copied["split_assignment_source"] = assignment_source
        splits[f"retrieval_{split}"].append(copied)

    train_trajectory_groups = {
        str(row["split_group_identity"]) for row in splits["trajectory_train"]
    }
    dev_trajectory_groups = {
        str(row["split_group_identity"]) for row in splits["trajectory_dev"]
    }
    train_retrieval_groups = {
        str(row["split_group_identity"]) for row in splits["retrieval_train"]
    }
    dev_retrieval_groups = {
        str(row["split_group_identity"]) for row in splits["retrieval_dev"]
    }
    trajectory_overlap = train_trajectory_groups & dev_trajectory_groups
    retrieval_overlap = train_retrieval_groups & dev_retrieval_groups
    if trajectory_overlap or retrieval_overlap:
        raise RuntimeError("vNext group split leaked one identity across train/dev")
    return splits, {
        "schema_version": VNEXT_SPLIT_SCHEMA,
        "seed": str(seed),
        "dev_fraction": float(dev_fraction),
        "trajectory_train_rows": len(splits["trajectory_train"]),
        "trajectory_dev_rows": len(splits["trajectory_dev"]),
        "retrieval_train_rows": len(splits["retrieval_train"]),
        "retrieval_dev_rows": len(splits["retrieval_dev"]),
        "trajectory_train_group_count": len(train_trajectory_groups),
        "trajectory_dev_group_count": len(dev_trajectory_groups),
        "retrieval_train_group_count": len(train_retrieval_groups),
        "retrieval_dev_group_count": len(dev_retrieval_groups),
        "trajectory_group_overlap_count": len(trajectory_overlap),
        "retrieval_group_overlap_count": len(retrieval_overlap),
        "locked_split_schema_version": LOCKED_SPLIT_SCHEMA,
        "locked_trajectory_row_count": locked_trajectory_rows,
        "locked_retrieval_row_count": locked_retrieval_rows,
        "locked_group_count": len(locked_group_splits),
        "locked_manifest_sha256s": sorted(locked_manifests),
    }


def _materialize_dynamic_skill_holdout(
    skills: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    *,
    fraction: float,
    minimum_queries: int,
    maximum_skills: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trajectory_skills = {
        str(row.get("target_skill_id") or row.get("skill_id") or "")
        for row in trajectories
        if str(row.get("target_skill_id") or row.get("skill_id") or "")
    }
    query_rows_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval_rows:
        for skill_id in row.get("required_tool_set_skill_ids") or []:
            value = str(skill_id)
            if value and value not in trajectory_skills:
                query_rows_by_skill[value].append(row)
    candidates = sorted(
        skill_id
        for skill_id, rows in query_rows_by_skill.items()
        if len(rows) >= int(minimum_queries)
    )
    target_count = min(
        int(maximum_skills),
        max(1, int(round(len(candidates) * float(fraction)))) if candidates else 0,
    )
    selected = sorted(
        candidates,
        key=lambda skill_id: _json_digest(
            {"seed": str(seed), "skill_id": skill_id}
        ),
    )[:target_count]
    selected_set = set(selected)
    skills_by_id = {_skill_id(row): row for row in skills}
    heldout_skills: list[dict[str, Any]] = []
    for skill_id in selected:
        copied = dict(skills_by_id[skill_id])
        copied["dynamic_holdout_protocol"] = "unseen_skill_append_v2"
        copied["dynamic_holdout_query_count"] = len(query_rows_by_skill[skill_id])
        copied["dynamic_holdout_seed"] = str(seed)
        heldout_skills.append(copied)
    heldout_queries: list[dict[str, Any]] = []
    for row in retrieval_rows:
        positives = {
            str(item) for item in row.get("required_tool_set_skill_ids") or [] if str(item)
        }
        heldout_positives = sorted(positives & selected_set)
        if not heldout_positives:
            continue
        copied = dict(row)
        copied["dynamic_holdout_positive_skill_ids"] = heldout_positives
        copied["dynamic_holdout_all_positive_skill_ids"] = sorted(positives)
        copied["dynamic_holdout_protocol"] = "unseen_skill_append_v2"
        copied["excluded_from_dynamic_ablation_training"] = True
        heldout_queries.append(copied)
    return heldout_skills, heldout_queries, {
        "protocol": "unseen_skill_append_v2",
        "seed": str(seed),
        "fraction": float(fraction),
        "minimum_queries": int(minimum_queries),
        "maximum_skills": int(maximum_skills),
        "candidate_skill_count": len(candidates),
        "heldout_skill_count": len(heldout_skills),
        "heldout_query_count": len(heldout_queries),
        "trajectory_skill_exclusion_count": len(trajectory_skills),
    }


def _derive_holdout_free_catalogs(
    catalogs: dict[str, dict[str, Any]],
    heldout_skill_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    heldout = {str(item) for item in heldout_skill_ids if str(item)}
    if not heldout:
        raise ValueError("dynamic holdout catalog derivation requires held-out skills")
    output: dict[str, dict[str, Any]] = {}
    mapping: dict[str, str] = {}
    removed_memberships = 0
    for catalog_id, row in sorted(catalogs.items()):
        original = [
            str(item)
            for item in row.get("runtime_visible_skill_ids") or []
            if str(item)
        ]
        filtered = [item for item in original if item not in heldout]
        removed_memberships += len(original) - len(filtered)
        if not filtered:
            raise ValueError(
                f"dynamic holdout removed every skill from runtime catalog: {catalog_id}"
            )
        if filtered == original:
            copied = dict(row)
            output[catalog_id] = copied
            mapping[catalog_id] = catalog_id
            continue
        digest = _json_digest(sorted(set(filtered)))
        derived_id = f"{catalog_id}__training_prefix_{digest[:16]}"
        copied = dict(row)
        copied["inventory_catalog_id"] = derived_id
        copied["inventory_catalog_digest"] = digest
        copied["runtime_visible_skill_ids"] = sorted(set(filtered))
        copied["inventory_pool_size"] = len(copied["runtime_visible_skill_ids"])
        copied["inventory_parent_catalog_id"] = catalog_id
        copied["inventory_parent_catalog_digest"] = str(
            row.get("inventory_catalog_digest") or ""
        )
        copied["inventory_derivation"] = "dynamic_skill_training_antijoin_v1"
        output[derived_id] = copied
        mapping[catalog_id] = derived_id
    return output, mapping, {
        "protocol": "dynamic_skill_training_antijoin_v1",
        "input_catalog_count": len(catalogs),
        "training_catalog_count": len(output),
        "derived_catalog_count": sum(
            int(parent != child) for parent, child in mapping.items()
        ),
        "removed_catalog_memberships": int(removed_memberships),
    }


def _rewrite_rows_to_training_catalogs(
    rows: list[dict[str, Any]],
    mapping: dict[str, str],
    training_catalogs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        parent_id = str(row.get("runtime_visible_catalog_id") or "")
        training_id = mapping.get(parent_id)
        if not training_id or training_id not in training_catalogs:
            raise ValueError(
                f"row references a catalog absent from holdout-free training views: {parent_id}"
            )
        catalog = training_catalogs[training_id]
        copied = dict(row)
        copied["runtime_visible_catalog_id"] = training_id
        copied["inventory_catalog_digest"] = str(
            catalog.get("inventory_catalog_digest") or ""
        )
        copied["inventory_source"] = str(catalog.get("inventory_source") or "")
        copied["inventory_available_before_decision"] = True
        copied["inventory_pool_size"] = int(catalog.get("inventory_pool_size") or 0)
        copied["inventory_parent_catalog_id"] = parent_id
        copied["inventory_parent_catalog_digest"] = str(
            row.get("inventory_catalog_digest") or ""
        )
        copied["inventory_derivation"] = "dynamic_skill_training_antijoin_v1"
        output.append(copied)
    return output


def _anti_join_dynamic_holdout(
    skills: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    heldout_skills: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    heldout = {_skill_id(row) for row in heldout_skills if _skill_id(row)}
    if not heldout:
        raise ValueError("dynamic holdout anti-join requires at least one held-out skill")
    training_skills = [row for row in skills if _skill_id(row) not in heldout]
    if len(training_skills) + len(heldout) != len(skills):
        raise ValueError("dynamic holdout skill identities are not unique in the source pool")

    training_retrieval: list[dict[str, Any]] = []
    excluded_query_ids: list[str] = []
    removed_negative_references = 0
    for row in retrieval_rows:
        positives = {
            str(item)
            for item in row.get("required_tool_set_skill_ids") or []
            if str(item)
        }
        if positives & heldout:
            excluded_query_ids.append(str(row.get("query_id") or ""))
            continue
        copied = dict(row)
        original_negatives = [
            str(item)
            for item in row.get("explicit_negative_skill_ids") or []
            if str(item)
        ]
        copied["explicit_negative_skill_ids"] = [
            item for item in original_negatives if item not in heldout
        ]
        removed_negative_references += len(original_negatives) - len(
            copied["explicit_negative_skill_ids"]
        )
        training_retrieval.append(copied)

    training_catalogs, catalog_mapping, catalog_report = _derive_holdout_free_catalogs(
        catalogs,
        heldout,
    )
    training_retrieval = _rewrite_rows_to_training_catalogs(
        training_retrieval,
        catalog_mapping,
        training_catalogs,
    )
    training_trajectories = _rewrite_rows_to_training_catalogs(
        trajectories,
        catalog_mapping,
        training_catalogs,
    )

    leak_locations: Counter[str] = Counter()
    for row in training_retrieval:
        referenced = {
            str(item)
            for field in ("required_tool_set_skill_ids", "explicit_negative_skill_ids")
            for item in row.get(field) or []
            if str(item)
        }
        leak_locations["retrieval_rows"] += len(referenced & heldout)
    for row in training_trajectories:
        referenced = {
            str(row.get("target_skill_id") or ""),
            str(row.get("skill_id") or ""),
            *(
                str(item)
                for item in row.get("required_tool_set_skill_ids") or []
                if str(item)
            ),
            *(
                str(item)
                for item in row.get("equivalent_next_skill_ids") or []
                if str(item)
            ),
        }
        leak_locations["trajectory_rows"] += len(referenced & heldout)
    for catalog in training_catalogs.values():
        referenced = {
            str(item)
            for item in catalog.get("runtime_visible_skill_ids") or []
            if str(item)
        }
        leak_locations["training_catalogs"] += len(referenced & heldout)
    training_skill_ids = {_skill_id(row) for row in training_skills}
    leak_locations["training_skills"] += len(training_skill_ids & heldout)
    leak_count = sum(leak_locations.values())
    if leak_count:
        raise ValueError(f"dynamic holdout leaked into training artifacts: {dict(leak_locations)}")

    return (
        training_skills,
        training_retrieval,
        training_trajectories,
        training_catalogs,
        {
            "protocol": "unseen_skill_append_v2",
            "heldout_skill_ids_sha256": _json_digest(sorted(heldout)),
            "source_skill_count": len(skills),
            "training_skill_count": len(training_skills),
            "heldout_skill_count": len(heldout),
            "source_retrieval_row_count": len(retrieval_rows),
            "training_retrieval_row_count": len(training_retrieval),
            "excluded_holdout_query_count": len(excluded_query_ids),
            "excluded_holdout_query_ids_sha256": _json_digest(
                sorted(excluded_query_ids)
            ),
            "removed_negative_reference_count": int(removed_negative_references),
            "training_trajectory_row_count": len(training_trajectories),
            "leak_count": int(leak_count),
            "leak_locations": dict(sorted(leak_locations.items())),
            "catalogs": catalog_report,
        },
    )


def _group_static_route_rows(
    trajectories: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    for row in trajectories:
        if str(row.get("route_target") or "").upper() in {"STOP", "NO_CALL"}:
            skipped["no_call"] += 1
            continue
        if _trajectory_type(row) == "parallel":
            skipped["parallel_retrieval_only"] += 1
            continue
        catalog_id = str(row.get("runtime_visible_catalog_id") or "")
        catalog = catalogs.get(catalog_id)
        if not isinstance(catalog, dict):
            skipped["missing_catalog"] += 1
            continue
        legal = {
            str(item)
            for item in catalog.get("runtime_visible_skill_ids") or []
            if str(item)
        }
        target_values = [str(row.get("target_skill_id") or "").strip()]
        target_values.extend(
            str(item).strip()
            for item in row.get("equivalent_next_skill_ids") or []
        )
        targets = sorted({item for item in target_values if item and item in legal})
        if not targets:
            skipped["no_legal_target"] += 1
            continue
        state_text = str(row.get("state_text_current") or "").strip()
        inventory_digest = str(row.get("inventory_catalog_digest") or "").strip()
        goal_boundary = str(row.get("goal_text") or row.get("task_text") or "").strip()
        key = (state_text, inventory_digest, goal_boundary)
        entry = grouped.setdefault(
            key,
            {
                "row": dict(row),
                "targets": set(),
                "factual_target_counts": Counter(),
                "trajectory_ids": set(),
                "source_rows": 0,
            },
        )
        entry["targets"].update(targets)
        factual_target = str(row.get("target_skill_id") or "").strip()
        if factual_target and factual_target in legal:
            entry["factual_target_counts"][factual_target] += 1
        entry["trajectory_ids"].add(str(row.get("trajectory_id") or ""))
        entry["source_rows"] += 1

    output: list[dict[str, Any]] = []
    ambiguous = 0
    for entry in grouped.values():
        copied = dict(entry["row"])
        positives = sorted(entry["targets"])
        capabilities = dict(copied.get("capabilities") or {})
        capabilities["current_state_route_set"] = True
        copied["capabilities"] = capabilities
        copied["current_state_route_set_skill_ids"] = positives
        copied["current_state_route_factual_target_counts"] = dict(
            sorted(entry["factual_target_counts"].items())
        )
        copied["current_state_route_ambiguous"] = len(positives) > 1
        copied["current_state_route_group_row_count"] = int(entry["source_rows"])
        copied["current_state_route_group_trajectory_count"] = len(
            {item for item in entry["trajectory_ids"] if item}
        )
        copied["current_state_route_group_identity"] = _json_digest(
            {
                "state_text_current": copied.get("state_text_current"),
                "inventory_catalog_digest": copied.get("inventory_catalog_digest"),
                "goal_boundary": str(copied.get("goal_text") or copied.get("task_text") or ""),
            }
        )
        ambiguous += int(len(positives) > 1)
        output.append(copied)
    output.sort(key=lambda row: str(row.get("current_state_route_group_identity") or ""))
    return output, {
        "group_count": len(output),
        "ambiguous_group_count": int(ambiguous),
        "singleton_group_count": int(len(output) - ambiguous),
        "skipped": dict(sorted(skipped.items())),
    }


def _causal_row_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": str(row.get("trajectory_id") or ""),
        "step_index": int(row.get("step_index") or 0),
        "target_skill_id": str(row.get("target_skill_id") or ""),
        "inventory_catalog_digest": str(row.get("inventory_catalog_digest") or ""),
    }


def _event_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": str(row.get("target_skill_id") or row.get("skill_id") or ""),
        "action_text": str(row.get("action_text") or ""),
        "actual_result_text": str(row.get("actual_result_text") or ""),
        "actual_result_executed": bool(row.get("actual_result_executed")),
    }


def _prefix_records(
    trajectories: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[tuple[str, int], dict[str, Any]] = {}
    for row in trajectories:
        trajectory_id = str(row.get("trajectory_id") or "")
        step_index = int(row.get("step_index") or 0)
        if not trajectory_id or (trajectory_id, step_index) in by_step:
            raise ValueError("causal mining requires unique trajectory/step identities")
        grouped[trajectory_id].append(row)
        by_step[(trajectory_id, step_index)] = row
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("step_index") or 0))
        prefix_event_contents: list[dict[str, Any]] = []
        prefix_event_refs: list[dict[str, Any]] = []
        for row in rows:
            step_index = int(row.get("step_index") or 0)
            event_digests = [_json_digest(event) for event in prefix_event_contents]
            skill_counts = Counter(
                str(event.get("skill_id") or "")
                for event in prefix_event_contents
                if str(event.get("skill_id") or "")
            )
            output[(trajectory_id, step_index)] = {
                "length": len(prefix_event_contents),
                "digest": _json_digest(prefix_event_contents),
                "event_digests": event_digests,
                "event_refs": list(prefix_event_refs),
                "skill_counts": dict(sorted(skill_counts.items())),
            }
            prefix_event_contents.append(_event_identity(row))
            prefix_event_refs.append(
                {
                    "trajectory_id": trajectory_id,
                    "step_index": step_index,
                }
            )
    return output, by_step


def _first_prefix_divergence(
    left: dict[str, Any],
    right: dict[str, Any],
) -> int | None:
    left_events = list(left.get("event_digests") or [])
    right_events = list(right.get("event_digests") or [])
    for index, (event_a, event_b) in enumerate(zip(left_events, right_events)):
        if event_a != event_b:
            return index
    if len(left_events) != len(right_events):
        return min(len(left_events), len(right_events))
    return None


def _targets_are_semantically_distinct(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    target_a: str,
    target_b: str,
) -> bool:
    equivalents_a = {
        str(item)
        for item in row_a.get("equivalent_next_skill_ids") or []
        if str(item)
    }
    equivalents_b = {
        str(item)
        for item in row_b.get("equivalent_next_skill_ids") or []
        if str(item)
    }
    return target_b not in equivalents_a and target_a not in equivalents_b


def _completion_contrast_evidence(
    prefix_a: dict[str, Any],
    prefix_b: dict[str, Any],
    target_a: str,
    target_b: str,
) -> dict[str, Any] | None:
    counts_a = Counter(prefix_a.get("skill_counts") or {})
    counts_b = Counter(prefix_b.get("skill_counts") or {})
    a_target_a = int(counts_a[target_a])
    b_target_a = int(counts_b[target_a])
    a_target_b = int(counts_a[target_b])
    b_target_b = int(counts_b[target_b])
    if not (a_target_a < b_target_a and b_target_b < a_target_b):
        return None
    return {
        "evidence_type": "symmetric_target_completion_contrast_v1",
        "history_a_target_a_count": a_target_a,
        "history_b_target_a_count": b_target_a,
        "history_a_target_b_count": a_target_b,
        "history_b_target_b_count": b_target_b,
        "interpretation": "each factual history selects the relatively less-completed target",
    }


def _history_diagnostic_views(
    trajectories: list[dict[str, Any]],
    prefixes: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    masked: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    for row in trajectories:
        key = (
            str(row.get("trajectory_id") or ""),
            int(row.get("step_index") or 0),
        )
        prefix = prefixes[key]
        if int(prefix["length"]) <= 0:
            continue
        identity = {
            "row": _causal_row_ref(row),
            "history_digest": str(prefix["digest"]),
        }
        masked.append(
            {
                "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                "diagnostic_id": _json_digest({**identity, "kind": "masked"}),
                "diagnostic_kind": "masked",
                "row": _causal_row_ref(row),
                "history_digest": str(prefix["digest"]),
                "history_length": int(prefix["length"]),
                "trainable_causal_label": False,
            }
        )
        event_digests = list(prefix.get("event_digests") or [])
        event_refs = list(prefix.get("event_refs") or [])
        reversed_indices = list(reversed(range(len(event_digests))))
        intervened_event_digests = [event_digests[index] for index in reversed_indices]
        if (
            len(event_digests) >= 2
            and reversed_indices != list(range(len(event_digests)))
            and intervened_event_digests != event_digests
        ):
            shuffled.append(
                {
                    "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                    "diagnostic_id": _json_digest(
                        {**identity, "kind": "order_shuffle", "order": reversed_indices}
                    ),
                    "diagnostic_kind": "order_shuffle",
                    "row": _causal_row_ref(row),
                    "history_digest": str(prefix["digest"]),
                    "history_length": int(prefix["length"]),
                    "original_event_refs": event_refs,
                    "intervened_event_order": reversed_indices,
                    "intervened_history_digest": _json_digest(
                        intervened_event_digests
                    ),
                    "trainable_causal_label": False,
                }
            )
    masked.sort(key=lambda row: str(row["diagnostic_id"]))
    shuffled.sort(key=lambda row: str(row["diagnostic_id"]))
    return {
        "masked_history_diagnostics": masked,
        "order_shuffle_diagnostics": shuffled,
    }


def _causal_cluster_counts_by_source(
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = str(row.get("source_id") or "")
        cluster = str(row.get("causal_cluster_id") or "")
        if source and cluster:
            grouped[source].add(cluster)
    return {source: len(clusters) for source, clusters in sorted(grouped.items())}


def _refresh_branch_pair_report(
    report: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    refreshed = dict(report)
    refreshed["causal_branch_pair_count"] = len(pairs)
    refreshed["causal_branch_cluster_count"] = len(
        {str(row.get("causal_cluster_id") or "") for row in pairs}
    )
    refreshed["causal_branch_clusters_by_source"] = (
        _causal_cluster_counts_by_source(pairs)
    )
    outcome_sources = dict(refreshed.get("causal_outcome_clusters_by_source") or {})
    combined_sources = dict(refreshed["causal_branch_clusters_by_source"])
    for source, count in outcome_sources.items():
        combined_sources[source] = max(int(combined_sources.get(source, 0)), int(count))
    refreshed["causal_clusters_by_source"] = dict(sorted(combined_sources.items()))
    return refreshed


def _select_supported_branch_sources(
    split_views: dict[str, dict[str, list[dict[str, Any]]]],
    reports: dict[str, dict[str, Any]],
    *,
    minimum_clusters_per_source: int,
) -> dict[str, Any]:
    train_counts = _causal_cluster_counts_by_source(
        split_views["train"]["causal_branch_pairs"]
    )
    dev_counts = _causal_cluster_counts_by_source(
        split_views["dev"]["causal_branch_pairs"]
    )
    sources = sorted(set(train_counts) | set(dev_counts))
    enabled = {
        source
        for source in sources
        if int(train_counts.get(source, 0)) >= int(minimum_clusters_per_source)
        and int(dev_counts.get(source, 0)) >= int(minimum_clusters_per_source)
    }
    demoted_pair_counts: dict[str, int] = {}
    for split in ("train", "dev"):
        original = split_views[split]["causal_branch_pairs"]
        kept = [row for row in original if str(row.get("source_id") or "") in enabled]
        kept_pair_ids = {str(row.get("causal_pair_id") or "") for row in kept}
        support = split_views[split]["causal_pair_support_rows"]
        split_views[split]["causal_pair_support_rows"] = [
            row
            for row in support
            if str(row.get("causal_pair_id") or "") in kept_pair_ids
        ]
        split_views[split]["causal_branch_pairs"] = kept
        demoted_pair_counts[split] = len(original) - len(kept)
        reports[split] = _refresh_branch_pair_report(reports[split], kept)
    return {
        "protocol": "train_dev_cluster_supported_causal_sources_v1",
        "minimum_train_clusters_per_source": int(minimum_clusters_per_source),
        "minimum_dev_clusters_per_source": int(minimum_clusters_per_source),
        "train_clusters_by_source_before_selection": train_counts,
        "dev_clusters_by_source_before_selection": dev_counts,
        "enabled_sources": sorted(enabled),
        "demoted_sources": sorted(set(sources) - enabled),
        "demoted_pair_counts": demoted_pair_counts,
        "demoted_pairs_remain_in_matched_history_diagnostics": True,
    }


def _clone_pair_support_sequence(
    rows: list[dict[str, Any]],
    *,
    trajectory_id: str,
    parent_trajectory_id: str,
    pair_id: str,
    branch: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for step_index, row in enumerate(rows):
        copied = dict(row)
        provenance = (
            dict(row.get("provenance"))
            if isinstance(row.get("provenance"), dict)
            else {}
        )
        provenance.update(
            {
                "source_id": UNORDERED_BRANCH_SOURCE_ID,
                "parent_source_id": _source_identity(row),
                "parent_trajectory_id": parent_trajectory_id,
                "causal_pair_id": pair_id,
                "causal_branch": branch,
                "causal_transform": "unordered_required_set_last_two_swap_v1",
            }
        )
        capabilities = dict(row.get("capabilities") or {})
        capabilities["causal_branch_pair"] = True
        capabilities["ordered_next_tool"] = False
        copied.update(
            {
                "trajectory_id": trajectory_id,
                "source_trajectory_id": parent_trajectory_id,
                "task_id": f"{trajectory_id}:{step_index}",
                "step_index": step_index,
                "pair_support_only": True,
                "causal_pair_id": pair_id,
                "causal_branch": branch,
                "capabilities": capabilities,
                "provenance": provenance,
            }
        )
        output.append(copied)
    refreshed, _causal_report = _attach_trajectory_causal_states(output)
    return refreshed


def _verified_unordered_remaining_tool_pairs(
    trajectories: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    *,
    maximum_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build verified complementary branches from unordered required-tool sets.

    The original unordered rows remain retrieval-only.  Pair-support rows are
    isolated from ordinary Stage2 sampling and exist only so the causal loss
    can replay two audited histories: common+A -> B versus common+B -> A.
    Both swapped events must have actually executed results, every required
    skill must occur exactly once, and the last two calls must be the only
    complementary remaining tools.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if _trajectory_type(row) == "parallel":
            grouped[str(row.get("trajectory_id") or "")].append(row)

    skipped: Counter[str] = Counter()
    candidates: list[
        tuple[str, str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    ] = []
    for parent_trajectory_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("step_index") or 0))
        if len(rows) < 2:
            skipped["fewer_than_two_events"] += 1
            continue
        if len({str(row.get("state_text_current") or "") for row in rows}) != 1:
            skipped["current_state_changes_across_unordered_set"] += 1
            continue
        targets = [str(row.get("target_skill_id") or "") for row in rows]
        target_counts = Counter(targets)
        if not targets or any(not target for target in targets):
            skipped["missing_target_skill"] += 1
            continue
        if any(count != 1 for count in target_counts.values()):
            skipped["required_skill_not_executed_exactly_once"] += 1
            continue
        required = {
            str(item)
            for item in rows[0].get("required_tool_set_skill_ids") or []
            if str(item)
        }
        if required != set(targets):
            skipped["executed_set_differs_from_required_set"] += 1
            continue
        left, right = rows[-2], rows[-1]
        left_target = str(left.get("target_skill_id") or "")
        right_target = str(right.get("target_skill_id") or "")
        if left_target == right_target:
            skipped["last_two_targets_identical"] += 1
            continue
        if not all(
            bool(row.get("actual_result_executed"))
            and str(row.get("actual_result_text") or "")
            and str(row.get("action_text") or "")
            for row in (left, right)
        ):
            skipped["last_two_lack_executed_action_result"] += 1
            continue
        catalog_id = str(left.get("runtime_visible_catalog_id") or "")
        catalog = catalogs.get(catalog_id) or {}
        legal = {
            str(item) for item in catalog.get("runtime_visible_skill_ids") or []
        }
        if (
            not catalog_id
            or str(right.get("runtime_visible_catalog_id") or "") != catalog_id
            or left_target not in legal
            or right_target not in legal
        ):
            skipped["last_two_targets_not_cross_legal"] += 1
            continue
        pair_id = _json_digest(
            {
                "kind": "verified_unordered_remaining_tool_branch_v1",
                "parent_trajectory_id": parent_trajectory_id,
                "common_prefix": [
                    _event_identity(row) for row in rows[:-2]
                ],
                "left": _event_identity(left),
                "right": _event_identity(right),
            }
        )
        candidates.append((pair_id, parent_trajectory_id, rows[:-2], left, right))

    candidates.sort(key=lambda item: item[0])
    eligible_candidate_count = len(candidates)
    if int(maximum_pairs) > 0:
        candidates = candidates[: int(maximum_pairs)]
    support_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for pair_id, parent_trajectory_id, common, left, right in candidates:
        trajectory_a = f"{parent_trajectory_id}::unordered_branch::{pair_id[:16]}::a"
        trajectory_b = f"{parent_trajectory_id}::unordered_branch::{pair_id[:16]}::b"
        rows_a = _clone_pair_support_sequence(
            [*common, left, right],
            trajectory_id=trajectory_a,
            parent_trajectory_id=parent_trajectory_id,
            pair_id=pair_id,
            branch="a",
        )
        rows_b = _clone_pair_support_sequence(
            [*common, right, left],
            trajectory_id=trajectory_b,
            parent_trajectory_id=parent_trajectory_id,
            pair_id=pair_id,
            branch="b",
        )
        prefixes, _by_step = _prefix_records([*rows_a, *rows_b])
        row_a = rows_a[-1]
        row_b = rows_b[-1]
        prefix_a = prefixes[(trajectory_a, int(row_a["step_index"]))]
        prefix_b = prefixes[(trajectory_b, int(row_b["step_index"]))]
        target_a = str(row_a.get("target_skill_id") or "")
        target_b = str(row_b.get("target_skill_id") or "")
        evidence = _completion_contrast_evidence(
            prefix_a,
            prefix_b,
            target_a,
            target_b,
        )
        divergence = _first_prefix_divergence(prefix_a, prefix_b)
        if evidence is None or divergence is None:
            raise RuntimeError("verified unordered branch failed its own replay audit")
        required = sorted(
            {
                str(row.get("target_skill_id") or "")
                for row in rows_a
                if str(row.get("target_skill_id") or "")
            }
        )
        verification = {
            **evidence,
            "evidence_type": "verified_unordered_remaining_tool_branch_v1",
            "unordered_required_set_verified": True,
            "actual_swapped_event_results": True,
            "shared_prefix_event_count": len(common),
            "required_tool_set_sha256": _json_digest(required),
            "interpretation": (
                "after the shared completed set, executing A leaves only B; "
                "executing B leaves only A"
            ),
        }
        pairs.append(
            {
                "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                "causal_pair_id": pair_id,
                "causal_cluster_id": _json_digest(
                    {
                        "kind": "unordered_required_set_parent",
                        "parent_trajectory_id": parent_trajectory_id,
                    }
                ),
                "causal_pair_kind": "history_branch",
                "history_pair_id": pair_id,
                "history_pair_kind": "verified_unordered_remaining_tool",
                "pair_mining_source": (
                    "verified_unordered_required_set_last_two_swap_v1"
                ),
                "source_id": UNORDERED_BRANCH_SOURCE_ID,
                "benchmark": str(left.get("benchmark") or ""),
                "domain": str(
                    (left.get("provenance") or {}).get("domain")
                    if isinstance(left.get("provenance"), dict)
                    else ""
                ),
                "row_a": _causal_row_ref(row_a),
                "row_b": _causal_row_ref(row_b),
                "history_a_digest": str(prefix_a["digest"]),
                "history_b_digest": str(prefix_b["digest"]),
                "history_length": int(prefix_a["length"]),
                "first_history_divergence_index": int(divergence),
                "same_prefix_length": True,
                "same_goal": True,
                "exact_current_state_alias": True,
                "different_target": True,
                "different_verified_target": True,
                "cross_legal": True,
                "semantic_target_equivalence_excluded": True,
                "causal_label_verified": True,
                "trainable_causal_label": True,
                "causal_verification": verification,
            }
        )
        support_rows.extend(rows_a)
        support_rows.extend(rows_b)
    return support_rows, pairs, {
        "protocol": "verified_unordered_required_set_last_two_swap_v1",
        "parallel_trajectory_count": len(grouped),
        "eligible_candidate_count": int(eligible_candidate_count),
        "emitted_pair_count": len(pairs),
        "support_row_count": len(support_rows),
        "maximum_pairs": int(maximum_pairs),
        "skipped": dict(sorted(skipped.items())),
        "model_generated_labels": False,
        "executed_results_required": True,
    }


def _mine_exact_causal_pairs(
    trajectories: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    *,
    max_pairs_per_group: int = 32,
    max_unordered_branch_pairs: int = 512,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Separate matched history contrasts from evidence-bearing causal labels."""

    prefixes, by_trajectory_step = _prefix_records(trajectories)
    grouped: dict[
        tuple[str, str, str, str, str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in trajectories:
        trajectory_id = str(row.get("trajectory_id") or "")
        step_index = int(row.get("step_index") or 0)
        prefix = prefixes[(trajectory_id, step_index)]
        if int(prefix["length"]) <= 0 or not str(row.get("target_skill_id") or ""):
            continue
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        key = (
            str(row.get("benchmark") or ""),
            _source_identity(row),
            str(provenance.get("domain") or row.get("domain") or ""),
            str(row.get("goal_text") or row.get("task_text") or ""),
            str(row.get("state_text_current") or ""),
            str(row.get("inventory_catalog_digest") or ""),
            int(prefix["length"]),
        )
        grouped[key].append(row)

    branch_pairs: list[dict[str, Any]] = []
    outcome_pairs: list[dict[str, Any]] = []
    matched_pairs: list[dict[str, Any]] = []
    group_count = 0
    candidate_comparisons = 0
    skipped: Counter[str] = Counter()
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            targets[str(row.get("target_skill_id") or "")].append(row)
        target_ids = sorted(target for target in targets if target)
        if len(target_ids) < 2:
            continue
        group_count += 1
        emitted = 0
        for left_index, target_a in enumerate(target_ids):
            for target_b in target_ids[left_index + 1 :]:
                for row_a in sorted(
                    targets[target_a],
                    key=lambda row: str(row.get("trajectory_id") or ""),
                ):
                    row_b = None
                    prefix_a = prefixes[
                        (
                            str(row_a.get("trajectory_id") or ""),
                            int(row_a.get("step_index") or 0),
                        )
                    ]
                    for candidate in sorted(
                        targets[target_b],
                        key=lambda row: str(row.get("trajectory_id") or ""),
                    ):
                        candidate_comparisons += 1
                        if str(candidate.get("trajectory_id") or "") == str(
                            row_a.get("trajectory_id") or ""
                        ):
                            continue
                        candidate_prefix = prefixes[
                            (
                                str(candidate.get("trajectory_id") or ""),
                                int(candidate.get("step_index") or 0),
                            )
                        ]
                        if str(candidate_prefix["digest"]) == str(prefix_a["digest"]):
                            skipped["identical_replay_prefix"] += 1
                            continue
                        row_b = candidate
                        break
                    if row_b is None:
                        skipped["no_distinct_cross_trajectory_prefix"] += 1
                        continue
                    prefix_b = prefixes[
                        (
                            str(row_b.get("trajectory_id") or ""),
                            int(row_b.get("step_index") or 0),
                        )
                    ]
                    catalog = catalogs.get(str(row_a.get("runtime_visible_catalog_id") or ""))
                    legal = {
                        str(item)
                        for item in (catalog or {}).get("runtime_visible_skill_ids") or []
                        if str(item)
                    }
                    if target_a not in legal or target_b not in legal:
                        skipped["not_cross_legal"] += 1
                        continue
                    if not _targets_are_semantically_distinct(
                        row_a,
                        row_b,
                        target_a,
                        target_b,
                    ):
                        skipped["semantic_target_equivalence"] += 1
                        continue
                    divergence = _first_prefix_divergence(prefix_a, prefix_b)
                    if divergence is None:
                        skipped["missing_prefix_divergence"] += 1
                        continue
                    identity = {
                        "benchmark": key[0],
                        "source_id": key[1],
                        "domain": key[2],
                        "goal_text": key[3],
                        "state_text_current": key[4],
                        "inventory_catalog_digest": key[5],
                        "prefix_length": key[6],
                        "target_a": target_a,
                        "target_b": target_b,
                        "history_a_digest": str(prefix_a["digest"]),
                        "history_b_digest": str(prefix_b["digest"]),
                    }
                    cluster_identity = {
                        "benchmark": key[0],
                        "source_id": key[1],
                        "domain": key[2],
                        "goal_text": key[3],
                        "state_text_current": key[4],
                        "inventory_catalog_digest": key[5],
                        "prefix_length": key[6],
                    }
                    pair_id = _json_digest(identity)
                    matched = {
                        "vnext_schema_version": VNEXT_SCHEMA_VERSION,
                        "history_pair_id": pair_id,
                        "history_pair_kind": "matched_history_contrast",
                        "causal_cluster_id": _json_digest(cluster_identity),
                        "benchmark": key[0],
                        "source_id": key[1],
                        "domain": key[2],
                        "row_a": _causal_row_ref(row_a),
                        "row_b": _causal_row_ref(row_b),
                        "cross_legal": True,
                        "same_goal": True,
                        "exact_current_state_alias": True,
                        "same_prefix_length": True,
                        "different_target": True,
                        "semantic_target_equivalence_excluded": True,
                        "history_a_digest": str(prefix_a["digest"]),
                        "history_b_digest": str(prefix_b["digest"]),
                        "history_length": int(prefix_a["length"]),
                        "first_history_divergence_index": int(divergence),
                        "pair_mining_source": "exact_structured_alias_with_distinct_prefix_v2",
                        "trainable_causal_label": False,
                    }
                    matched_pairs.append(matched)

                    completion_evidence = _completion_contrast_evidence(
                        prefix_a,
                        prefix_b,
                        target_a,
                        target_b,
                    )
                    if completion_evidence is not None:
                        pair = dict(matched)
                        pair["causal_pair_id"] = pair_id
                        pair["causal_pair_kind"] = "history_branch"
                        pair["different_verified_target"] = True
                        pair["causal_label_verified"] = True
                        pair["causal_verification"] = completion_evidence
                        pair["trainable_causal_label"] = True
                        branch_pairs.append(pair)
                    else:
                        skipped["matched_without_decision_evidence"] += 1

                    previous_a = by_trajectory_step.get(
                        (
                            str(row_a.get("trajectory_id") or ""),
                            int(row_a.get("step_index") or 0) - 1,
                        )
                    )
                    previous_b = by_trajectory_step.get(
                        (
                            str(row_b.get("trajectory_id") or ""),
                            int(row_b.get("step_index") or 0) - 1,
                        )
                    )
                    if previous_a is not None and previous_b is not None:
                        previous_prefix_a = prefixes[
                            (
                                str(previous_a.get("trajectory_id") or ""),
                                int(previous_a.get("step_index") or 0),
                            )
                        ]
                        previous_prefix_b = prefixes[
                            (
                                str(previous_b.get("trajectory_id") or ""),
                                int(previous_b.get("step_index") or 0),
                            )
                        ]
                        same_executed_action = bool(
                            str(previous_a.get("skill_id") or "")
                            == str(previous_b.get("skill_id") or "")
                            and str(previous_a.get("action_text") or "")
                            == str(previous_b.get("action_text") or "")
                        )
                        same_event_boundary = bool(
                            str(previous_a.get("state_text_current") or "")
                            == str(previous_b.get("state_text_current") or "")
                            and str(previous_a.get("goal_text") or previous_a.get("task_text") or "")
                            == str(previous_b.get("goal_text") or previous_b.get("task_text") or "")
                            and str(previous_a.get("inventory_catalog_digest") or "")
                            == str(previous_b.get("inventory_catalog_digest") or "")
                        )
                        result_a = str(previous_a.get("actual_result_text") or "")
                        result_b = str(previous_b.get("actual_result_text") or "")
                        novel = bool(
                            previous_a.get("result_novel_for_memory")
                            and previous_b.get("result_novel_for_memory")
                        )
                        if (
                            str(previous_prefix_a["digest"])
                            == str(previous_prefix_b["digest"])
                            and same_executed_action
                            and same_event_boundary
                            and result_a
                            and result_b
                            and result_a != result_b
                            and novel
                        ):
                            outcome = dict(matched)
                            outcome["causal_pair_kind"] = "result_outcome"
                            outcome["causal_pair_id"] = _json_digest(
                                {**identity, "result_a": result_a, "result_b": result_b}
                            )
                            outcome["event_a"] = _causal_row_ref(previous_a)
                            outcome["event_b"] = _causal_row_ref(previous_b)
                            outcome["same_pre_action_state"] = True
                            outcome["same_pre_event_history"] = True
                            outcome["causal_label_verified"] = True
                            outcome["causal_verification"] = {
                                "evidence_type": "same_action_different_executed_result_v1",
                                "pre_event_history_digest": str(previous_prefix_a["digest"]),
                            }
                            outcome["trainable_causal_label"] = True
                            outcome_pairs.append(outcome)
                    emitted += 1
                    break
                if emitted >= int(max_pairs_per_group):
                    break
            if emitted >= int(max_pairs_per_group):
                break

    (
        unordered_support_rows,
        unordered_branch_pairs,
        unordered_branch_report,
    ) = _verified_unordered_remaining_tool_pairs(
        trajectories,
        catalogs,
        maximum_pairs=int(max_unordered_branch_pairs),
    )
    branch_pairs.extend(unordered_branch_pairs)
    branch_pairs.sort(key=lambda row: str(row["causal_pair_id"]))
    outcome_pairs.sort(key=lambda row: str(row["causal_pair_id"]))
    matched_pairs.sort(key=lambda row: str(row["history_pair_id"]))
    diagnostic_views = _history_diagnostic_views(trajectories, prefixes)
    return {
        "causal_branch_pairs": branch_pairs,
        "causal_pair_support_rows": unordered_support_rows,
        "causal_order_pairs": [],
        "causal_outcome_pairs": outcome_pairs,
        "matched_history_contrast_pairs": matched_pairs,
        **diagnostic_views,
    }, {
        "exact_alias_group_count": int(group_count),
        "candidate_comparison_count": int(candidate_comparisons),
        "causal_branch_pair_count": len(branch_pairs),
        "causal_branch_cluster_count": len(
            {str(row.get("causal_cluster_id") or "") for row in branch_pairs}
        ),
        "causal_order_pair_count": 0,
        "causal_outcome_pair_count": len(outcome_pairs),
        "causal_outcome_cluster_count": len(
            {str(row.get("causal_cluster_id") or "") for row in outcome_pairs}
        ),
        "matched_history_contrast_pair_count": len(matched_pairs),
        "matched_history_contrast_cluster_count": len(
            {str(row.get("causal_cluster_id") or "") for row in matched_pairs}
        ),
        "causal_clusters_by_source": _causal_cluster_counts_by_source(
            [*branch_pairs, *outcome_pairs]
        ),
        "causal_branch_clusters_by_source": _causal_cluster_counts_by_source(
            branch_pairs
        ),
        "causal_outcome_clusters_by_source": _causal_cluster_counts_by_source(
            outcome_pairs
        ),
        "masked_history_diagnostic_count": len(
            diagnostic_views["masked_history_diagnostics"]
        ),
        "order_shuffle_diagnostic_count": len(
            diagnostic_views["order_shuffle_diagnostics"]
        ),
        "max_pairs_per_group": int(max_pairs_per_group),
        "model_based_pair_labels": False,
        "causal_label_contract": "evidence_bearing_history_pair_v2",
        "verified_unordered_branch": unordered_branch_report,
        "skipped": dict(sorted(skipped.items())),
    }


def _view_rows(
    trajectories: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    views: dict[str, list[dict[str, Any]]] = {
        "ordered_transition": [],
        "result_correction": [],
        "no_call": [],
        "one_error_prefix": [],
        "two_or_more_error_prefix": [],
        "recovery_prefix": [],
        "robust_prefix_downgrade": [],
    }
    for row in trajectories:
        if str(row.get("route_target") or "").upper() in {"STOP", "NO_CALL"}:
            views["no_call"].append(row)
            continue
        capabilities = row.get("capabilities") or {}
        if bool(capabilities.get("ordered_next_tool")):
            views["ordered_transition"].append(row)
        if bool(capabilities.get("actual_execution_result")):
            views["result_correction"].append(row)
        robust_kind, downgrade = _verified_robust_prefix_kind(row)
        if robust_kind and not bool(capabilities.get("ordered_next_tool")):
            robust_kind = None
            downgrade = "verified_robust_prefix_without_ordered_target"
        if robust_kind:
            copied = dict(row)
            copied_capabilities = dict(capabilities)
            copied_capabilities["verified_robust_prefix"] = True
            copied["capabilities"] = copied_capabilities
            copied["robust_prefix_kind"] = robust_kind
            views[robust_kind].append(copied)
        elif downgrade:
            copied = dict(row)
            copied["robust_prefix_downgrade_reason"] = downgrade
            views["robust_prefix_downgrade"].append(copied)
    return views


def build_views(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skills_path = Path(args.skills_path)
    retrieval_path = Path(args.retrieval_path)
    trajectories_path = Path(args.trajectories_path)
    clean_preflight = _bind_clean_preflight(
        Path(args.clean_preflight_report),
        skills_path=skills_path,
        retrieval_path=retrieval_path,
        trajectories_path=trajectories_path,
    )
    skills, by_prefix = _load_skills(skills_path)
    source_skill_count = len(skills)
    catalogs = _build_catalogs(skills, by_prefix)
    raw_trajectories = list(
        _read_jsonl(
            trajectories_path,
            max_rows=None if args.max_trajectory_rows <= 0 else args.max_trajectory_rows,
        )
    )
    trajectories, trajectory_counts = _canonical_trajectory_rows(
        raw_trajectories,
        catalogs,
        {_skill_id(row) for row in skills},
    )
    retrieval_rows = _group_retrieval_rows(
        _read_jsonl(
            retrieval_path,
            max_rows=None if args.max_retrieval_rows <= 0 else args.max_retrieval_rows,
        ),
        catalogs,
        trajectories,
    )
    retrieval_rows, temporal_retrieval_exclusions = (
        _exclude_unbound_trajectory_retrieval_rows(retrieval_rows)
    )
    (
        trajectories,
        retrieval_rows,
        data_exclusions,
        deduplication_report,
    ) = _deduplicate_semantic_rows(trajectories, retrieval_rows)
    data_exclusions = [*temporal_retrieval_exclusions, *data_exclusions]
    dynamic_holdout_skills, dynamic_holdout_queries, holdout_selection_report = (
        _materialize_dynamic_skill_holdout(
            skills,
            retrieval_rows,
            trajectories,
            fraction=float(getattr(args, "dynamic_holdout_fraction", 0.01)),
            minimum_queries=int(getattr(args, "dynamic_holdout_minimum_queries", 2)),
            maximum_skills=int(getattr(args, "dynamic_holdout_maximum_skills", 256)),
            seed=str(getattr(args, "dynamic_holdout_seed", "clstr-dynamic-v1")),
        )
    )
    append_catalogs = {key: dict(value) for key, value in catalogs.items()}
    (
        skills,
        retrieval_rows,
        trajectories,
        catalogs,
        holdout_antijoin_report,
    ) = _anti_join_dynamic_holdout(
        skills,
        retrieval_rows,
        trajectories,
        catalogs,
        dynamic_holdout_skills,
    )
    dynamic_holdout_report = {
        **holdout_selection_report,
        "protocol": "unseen_skill_append_v2",
        "training_antijoin": holdout_antijoin_report,
    }
    full_coverage_report = _capability_coverage_report(
        trajectories,
        retrieval_rows,
    )
    splits, split_report = _split_semantic_rows(
        trajectories,
        retrieval_rows,
        dev_fraction=float(getattr(args, "dev_fraction", 0.1)),
        seed=str(getattr(args, "split_seed", "clstr-vnext-v1")),
    )
    shortcut_report = _shortcut_predictability_report(
        splits["trajectory_train"],
        splits["trajectory_dev"],
    )
    split_coverage_reports = {
        split: _capability_coverage_report(
            splits[f"trajectory_{split}"],
            splits[f"retrieval_{split}"],
        )
        for split in ("train", "dev")
    }
    split_views: dict[str, dict[str, list[dict[str, Any]]]] = {}
    static_route_reports: dict[str, dict[str, Any]] = {}
    causal_pair_reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        split_trajectories = splits[f"trajectory_{split}"]
        views = _view_rows(split_trajectories)
        static_rows, static_report = _group_static_route_rows(
            split_trajectories,
            catalogs,
        )
        causal_views, causal_report = _mine_exact_causal_pairs(
            split_trajectories,
            catalogs,
            max_pairs_per_group=int(getattr(args, "max_causal_pairs_per_group", 32)),
            max_unordered_branch_pairs=int(
                getattr(args, "max_unordered_branch_pairs_per_split", 512)
            ),
        )
        split_views[split] = {
            **views,
            "static_route": static_rows,
            **causal_views,
        }
        static_route_reports[split] = static_report
        causal_pair_reports[split] = causal_report
    causal_source_selection_report = _select_supported_branch_sources(
        split_views,
        causal_pair_reports,
        minimum_clusters_per_source=int(
            getattr(args, "minimum_dev_clusters_per_source", 5)
        ),
    )
    history_channel_report = audit_history_channel_rows(
        [
            *trajectories,
            *retrieval_rows,
            *dynamic_holdout_queries,
            *split_views["train"]["causal_pair_support_rows"],
            *split_views["dev"]["causal_pair_support_rows"],
        ],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    files = {
        "training_skills": output_dir / "training_skills.jsonl",
        "data_exclusions": output_dir / "data_exclusions.jsonl",
        "inventory_catalogs": output_dir / "inventory_catalogs.jsonl",
        "retrieval_rows": output_dir / "retrieval_rows.jsonl",
        "retrieval_dev_rows": output_dir / "retrieval_dev_rows.jsonl",
        "trajectory_rows": output_dir / "trajectory_rows.jsonl",
        "trajectory_dev_rows": output_dir / "trajectory_dev_rows.jsonl",
        "static_route_rows": output_dir / "static_route_rows.jsonl",
        "static_route_dev_rows": output_dir / "static_route_dev_rows.jsonl",
        "ordered_transition_rows": output_dir / "ordered_transition_rows.jsonl",
        "ordered_transition_dev_rows": output_dir / "ordered_transition_dev_rows.jsonl",
        "result_correction_rows": output_dir / "result_correction_rows.jsonl",
        "result_correction_dev_rows": output_dir / "result_correction_dev_rows.jsonl",
        "no_call_rows": output_dir / "no_call_rows.jsonl",
        "no_call_dev_rows": output_dir / "no_call_dev_rows.jsonl",
        "one_error_prefix_rows": output_dir / "one_error_prefix_rows.jsonl",
        "one_error_prefix_dev_rows": output_dir / "one_error_prefix_dev_rows.jsonl",
        "two_or_more_error_prefix_rows": output_dir
        / "two_or_more_error_prefix_rows.jsonl",
        "two_or_more_error_prefix_dev_rows": output_dir
        / "two_or_more_error_prefix_dev_rows.jsonl",
        "recovery_prefix_rows": output_dir / "recovery_prefix_rows.jsonl",
        "recovery_prefix_dev_rows": output_dir / "recovery_prefix_dev_rows.jsonl",
        "robust_prefix_downgrade_rows": output_dir
        / "robust_prefix_downgrade_rows.jsonl",
        "robust_prefix_downgrade_dev_rows": output_dir
        / "robust_prefix_downgrade_dev_rows.jsonl",
        "causal_branch_pairs": output_dir / "causal_branch_pairs.jsonl",
        "causal_branch_dev_pairs": output_dir / "causal_branch_dev_pairs.jsonl",
        "causal_pair_support_rows": output_dir / "causal_pair_support_rows.jsonl",
        "causal_pair_support_dev_rows": output_dir
        / "causal_pair_support_dev_rows.jsonl",
        "causal_order_pairs": output_dir / "causal_order_pairs.jsonl",
        "causal_order_dev_pairs": output_dir / "causal_order_dev_pairs.jsonl",
        "causal_outcome_pairs": output_dir / "causal_outcome_pairs.jsonl",
        "causal_outcome_dev_pairs": output_dir / "causal_outcome_dev_pairs.jsonl",
        "matched_history_contrast_pairs": output_dir
        / "matched_history_contrast_pairs.jsonl",
        "matched_history_contrast_dev_pairs": output_dir
        / "matched_history_contrast_dev_pairs.jsonl",
        "masked_history_diagnostics": output_dir / "masked_history_diagnostics.jsonl",
        "masked_history_dev_diagnostics": output_dir
        / "masked_history_dev_diagnostics.jsonl",
        "order_shuffle_diagnostics": output_dir / "order_shuffle_diagnostics.jsonl",
        "order_shuffle_dev_diagnostics": output_dir
        / "order_shuffle_dev_diagnostics.jsonl",
        "dynamic_skill_holdout_skills": output_dir / "dynamic_skill_holdout_skills.jsonl",
        "dynamic_skill_holdout_queries": output_dir / "dynamic_skill_holdout_queries.jsonl",
        "dynamic_skill_holdout_catalogs": output_dir
        / "dynamic_skill_holdout_catalogs.jsonl",
    }
    counts = {
        "training_skills": _write_jsonl(files["training_skills"], skills),
        "data_exclusions": _write_jsonl(files["data_exclusions"], data_exclusions),
        "inventory_catalogs": _write_jsonl(files["inventory_catalogs"], catalogs.values()),
        "retrieval_rows": _write_jsonl(files["retrieval_rows"], splits["retrieval_train"]),
        "retrieval_dev_rows": _write_jsonl(
            files["retrieval_dev_rows"], splits["retrieval_dev"]
        ),
        "trajectory_rows": _write_jsonl(files["trajectory_rows"], splits["trajectory_train"]),
        "trajectory_dev_rows": _write_jsonl(
            files["trajectory_dev_rows"], splits["trajectory_dev"]
        ),
        "static_route_rows": _write_jsonl(
            files["static_route_rows"], split_views["train"]["static_route"]
        ),
        "static_route_dev_rows": _write_jsonl(
            files["static_route_dev_rows"], split_views["dev"]["static_route"]
        ),
        "ordered_transition_rows": _write_jsonl(
            files["ordered_transition_rows"], split_views["train"]["ordered_transition"]
        ),
        "ordered_transition_dev_rows": _write_jsonl(
            files["ordered_transition_dev_rows"], split_views["dev"]["ordered_transition"]
        ),
        "result_correction_rows": _write_jsonl(
            files["result_correction_rows"], split_views["train"]["result_correction"]
        ),
        "result_correction_dev_rows": _write_jsonl(
            files["result_correction_dev_rows"], split_views["dev"]["result_correction"]
        ),
        "no_call_rows": _write_jsonl(
            files["no_call_rows"], split_views["train"]["no_call"]
        ),
        "no_call_dev_rows": _write_jsonl(
            files["no_call_dev_rows"], split_views["dev"]["no_call"]
        ),
        "one_error_prefix_rows": _write_jsonl(
            files["one_error_prefix_rows"], split_views["train"]["one_error_prefix"]
        ),
        "one_error_prefix_dev_rows": _write_jsonl(
            files["one_error_prefix_dev_rows"], split_views["dev"]["one_error_prefix"]
        ),
        "two_or_more_error_prefix_rows": _write_jsonl(
            files["two_or_more_error_prefix_rows"],
            split_views["train"]["two_or_more_error_prefix"],
        ),
        "two_or_more_error_prefix_dev_rows": _write_jsonl(
            files["two_or_more_error_prefix_dev_rows"],
            split_views["dev"]["two_or_more_error_prefix"],
        ),
        "recovery_prefix_rows": _write_jsonl(
            files["recovery_prefix_rows"], split_views["train"]["recovery_prefix"]
        ),
        "recovery_prefix_dev_rows": _write_jsonl(
            files["recovery_prefix_dev_rows"], split_views["dev"]["recovery_prefix"]
        ),
        "robust_prefix_downgrade_rows": _write_jsonl(
            files["robust_prefix_downgrade_rows"],
            split_views["train"]["robust_prefix_downgrade"],
        ),
        "robust_prefix_downgrade_dev_rows": _write_jsonl(
            files["robust_prefix_downgrade_dev_rows"],
            split_views["dev"]["robust_prefix_downgrade"],
        ),
        "causal_branch_pairs": _write_jsonl(
            files["causal_branch_pairs"], split_views["train"]["causal_branch_pairs"]
        ),
        "causal_branch_dev_pairs": _write_jsonl(
            files["causal_branch_dev_pairs"], split_views["dev"]["causal_branch_pairs"]
        ),
        "causal_pair_support_rows": _write_jsonl(
            files["causal_pair_support_rows"],
            split_views["train"]["causal_pair_support_rows"],
        ),
        "causal_pair_support_dev_rows": _write_jsonl(
            files["causal_pair_support_dev_rows"],
            split_views["dev"]["causal_pair_support_rows"],
        ),
        "causal_order_pairs": _write_jsonl(
            files["causal_order_pairs"], split_views["train"]["causal_order_pairs"]
        ),
        "causal_order_dev_pairs": _write_jsonl(
            files["causal_order_dev_pairs"], split_views["dev"]["causal_order_pairs"]
        ),
        "causal_outcome_pairs": _write_jsonl(
            files["causal_outcome_pairs"], split_views["train"]["causal_outcome_pairs"]
        ),
        "causal_outcome_dev_pairs": _write_jsonl(
            files["causal_outcome_dev_pairs"], split_views["dev"]["causal_outcome_pairs"]
        ),
        "matched_history_contrast_pairs": _write_jsonl(
            files["matched_history_contrast_pairs"],
            split_views["train"]["matched_history_contrast_pairs"],
        ),
        "matched_history_contrast_dev_pairs": _write_jsonl(
            files["matched_history_contrast_dev_pairs"],
            split_views["dev"]["matched_history_contrast_pairs"],
        ),
        "masked_history_diagnostics": _write_jsonl(
            files["masked_history_diagnostics"],
            split_views["train"]["masked_history_diagnostics"],
        ),
        "masked_history_dev_diagnostics": _write_jsonl(
            files["masked_history_dev_diagnostics"],
            split_views["dev"]["masked_history_diagnostics"],
        ),
        "order_shuffle_diagnostics": _write_jsonl(
            files["order_shuffle_diagnostics"],
            split_views["train"]["order_shuffle_diagnostics"],
        ),
        "order_shuffle_dev_diagnostics": _write_jsonl(
            files["order_shuffle_dev_diagnostics"],
            split_views["dev"]["order_shuffle_diagnostics"],
        ),
        "dynamic_skill_holdout_skills": _write_jsonl(
            files["dynamic_skill_holdout_skills"], dynamic_holdout_skills
        ),
        "dynamic_skill_holdout_queries": _write_jsonl(
            files["dynamic_skill_holdout_queries"], dynamic_holdout_queries
        ),
        "dynamic_skill_holdout_catalogs": _write_jsonl(
            files["dynamic_skill_holdout_catalogs"], append_catalogs.values()
        ),
    }
    blockers: list[str] = []
    if not counts["retrieval_rows"]:
        blockers.append("empty_retrieval_view")
    if not counts["static_route_rows"]:
        blockers.append("empty_static_route_view")
    if not counts["ordered_transition_rows"]:
        blockers.append("empty_ordered_transition_view")
    if not counts["result_correction_rows"]:
        blockers.append("empty_result_correction_view")
    if not counts["retrieval_dev_rows"] or not counts["static_route_dev_rows"]:
        blockers.append("empty_stage0_causal_dev_view")
    if not counts["trajectory_dev_rows"]:
        blockers.append("empty_stage2_causal_dev_trajectory_view")
    if not counts["result_correction_dev_rows"]:
        blockers.append("empty_result_correction_dev_view")
    if history_channel_report["status"] != "ok":
        blockers.append("structured_current_or_causal_state_audit_failed")
    blockers.extend(_causal_retrieval_release_blockers(full_coverage_report))
    if shortcut_report.get("status") != "ok":
        blockers.append("source_step_shortcut_audit_unavailable")
    else:
        maximum_shortcut_accuracy = float(
            getattr(args, "maximum_source_step_shortcut_accuracy", 0.8)
        )
        minimum_shortcut_coverage = float(
            getattr(args, "minimum_shortcut_feature_coverage", 0.5)
        )
        if not 0.0 <= maximum_shortcut_accuracy <= 1.0:
            raise ValueError("maximum shortcut accuracy must be in [0, 1]")
        if not 0.0 <= minimum_shortcut_coverage <= 1.0:
            raise ValueError("minimum shortcut coverage must be in [0, 1]")
        for feature_name in ("source_step", "source_type_step"):
            metric = shortcut_report["metrics"][feature_name]
            if (
                float(metric["known_feature_coverage"])
                >= minimum_shortcut_coverage
                and float(metric["accuracy"]) > maximum_shortcut_accuracy
            ):
                blockers.append(f"source_step_label_shortcut:{feature_name}")
    if not counts["dynamic_skill_holdout_skills"] or not counts[
        "dynamic_skill_holdout_queries"
    ]:
        blockers.append("empty_dynamic_skill_holdout_view")
    if int(holdout_antijoin_report.get("leak_count") or 0) != 0:
        blockers.append("dynamic_skill_holdout_training_leak")
    if counts["training_skills"] + counts["dynamic_skill_holdout_skills"] != int(
        source_skill_count
    ):
        blockers.append("dynamic_skill_holdout_partition_mismatch")
    if bool(getattr(args, "require_robust_prefix_views", False)):
        minimum_error_train = int(getattr(args, "minimum_verified_error_prefix_rows", 1000))
        minimum_error_dev = int(
            getattr(args, "minimum_verified_error_prefix_dev_rows", 100)
        )
        minimum_recovery_train = int(
            getattr(args, "minimum_verified_recovery_prefix_rows", 1000)
        )
        minimum_recovery_dev = int(
            getattr(args, "minimum_verified_recovery_prefix_dev_rows", 100)
        )
        if min(
            minimum_error_train,
            minimum_error_dev,
            minimum_recovery_train,
            minimum_recovery_dev,
        ) <= 0:
            raise ValueError("verified robust-prefix release thresholds must be positive")
        train_error_rows = counts["one_error_prefix_rows"] + counts[
            "two_or_more_error_prefix_rows"
        ]
        dev_error_rows = counts["one_error_prefix_dev_rows"] + counts[
            "two_or_more_error_prefix_dev_rows"
        ]
        if train_error_rows < minimum_error_train:
            blockers.append("insufficient_verified_error_prefix_train_rows")
        if dev_error_rows < minimum_error_dev:
            blockers.append("insufficient_verified_error_prefix_dev_rows")
        if counts["recovery_prefix_rows"] < minimum_recovery_train:
            blockers.append("insufficient_verified_recovery_prefix_train_rows")
        if counts["recovery_prefix_dev_rows"] < minimum_recovery_dev:
            blockers.append("insufficient_verified_recovery_prefix_dev_rows")
    if bool(getattr(args, "require_causal_branch_pairs", True)) and not counts[
        "causal_branch_pairs"
    ]:
        blockers.append("empty_memory_critical_causal_branch_view")
    if bool(getattr(args, "require_causal_branch_pairs", True)) and not counts[
        "causal_branch_dev_pairs"
    ]:
        blockers.append("empty_memory_critical_causal_branch_dev_view")
    train_pair_report = causal_pair_reports["train"]
    dev_pair_report = causal_pair_reports["dev"]
    minimum_train_clusters = int(
        getattr(args, "minimum_train_clusters_per_enabled_kind", 20)
    )
    minimum_dev_clusters = int(
        getattr(args, "minimum_dev_clusters_per_enabled_kind", 20)
    )
    minimum_dev_source_clusters = int(
        getattr(args, "minimum_dev_clusters_per_source", 5)
    )
    if min(
        minimum_train_clusters,
        minimum_dev_clusters,
        minimum_dev_source_clusters,
    ) <= 0:
        raise ValueError("causal data-release cluster thresholds must be positive")
    required_pair_kinds: list[str] = []
    if bool(getattr(args, "require_causal_branch_pairs", True)):
        required_pair_kinds.append("branch")
    if bool(getattr(args, "require_causal_outcome_pairs", False)):
        required_pair_kinds.append("outcome")
    for kind in required_pair_kinds:
        train_clusters = int(train_pair_report[f"causal_{kind}_cluster_count"])
        dev_clusters = int(dev_pair_report[f"causal_{kind}_cluster_count"])
        if train_clusters < minimum_train_clusters:
            blockers.append(f"insufficient_causal_{kind}_train_clusters")
        if dev_clusters < minimum_dev_clusters:
            blockers.append(f"insufficient_causal_{kind}_dev_clusters")
        for source, cluster_count in sorted(
            (
                dev_pair_report.get(f"causal_{kind}_clusters_by_source") or {}
            ).items()
        ):
            if int(cluster_count) < minimum_dev_source_clusters:
                blockers.append(
                    f"insufficient_causal_{kind}_dev_clusters_for_source:{source}"
                )
    manifest = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "schema_version": VNEXT_SCHEMA_VERSION,
        "inputs": {
            "skills_path": str(skills_path.resolve()),
            "skills_sha256": _file_digest(skills_path),
            "retrieval_path": str(retrieval_path.resolve()),
            "retrieval_sha256": _file_digest(retrieval_path),
            "trajectories_path": str(trajectories_path.resolve()),
            "trajectories_sha256": _file_digest(trajectories_path),
        },
        "clean_preflight": clean_preflight,
        "counts": counts,
        "trajectory_counts": trajectory_counts,
        "data_quality": {
            "deduplication": deduplication_report,
            "shortcut_predictability": shortcut_report,
            "maximum_source_step_shortcut_accuracy": float(
                getattr(args, "maximum_source_step_shortcut_accuracy", 0.8)
            ),
            "minimum_shortcut_feature_coverage": float(
                getattr(args, "minimum_shortcut_feature_coverage", 0.5)
            ),
            "capability_coverage": full_coverage_report,
            "split_capability_coverage": split_coverage_reports,
            "robust_prefix_release": {
                "required": bool(getattr(args, "require_robust_prefix_views", False)),
                "minimum_verified_error_prefix_rows": int(
                    getattr(args, "minimum_verified_error_prefix_rows", 1000)
                ),
                "minimum_verified_error_prefix_dev_rows": int(
                    getattr(args, "minimum_verified_error_prefix_dev_rows", 100)
                ),
                "minimum_verified_recovery_prefix_rows": int(
                    getattr(args, "minimum_verified_recovery_prefix_rows", 1000)
                ),
                "minimum_verified_recovery_prefix_dev_rows": int(
                    getattr(args, "minimum_verified_recovery_prefix_dev_rows", 100)
                ),
            },
        },
        "split_report": split_report,
        "static_route_report": static_route_reports,
        "history_channel": history_channel_report,
        "causal_pair_report": causal_pair_reports,
        "causal_source_selection": causal_source_selection_report,
        "dynamic_skill_holdout_report": dynamic_holdout_report,
        "model_input_contract": {
            "history_free_current_state_channel": True,
            "route_query_channel": (
                "compact_causal_skill_action_v1"
            ),
            "route_query_history_max_events": 8,
            "raw_tool_results_in_route_query": False,
            "explicit_benchmark_or_source_label_added_to_query": False,
            "executed_skill_ids_in_route_query": True,
            "executed_skill_ids_may_be_namespaced": True,
            "static_dynamic_route_query_identical": True,
            "recurrent_memory_carries_result_correction": True,
            "training_target_semantics": (
                "current_skill_before_memory_update_v1"
            ),
        },
        "data_release_thresholds": {
            "required_pair_kinds": required_pair_kinds,
            "minimum_train_clusters_per_enabled_kind": minimum_train_clusters,
            "minimum_dev_clusters_per_enabled_kind": minimum_dev_clusters,
            "minimum_dev_clusters_per_source": minimum_dev_source_clusters,
        },
        "training_prefix": {
            "skills_path": str(files["training_skills"].resolve()),
            "skills_sha256": _file_digest(files["training_skills"]),
            "skill_count": counts["training_skills"],
            "checkpoint_skill_order_contract": "source_order_minus_holdout_v1",
        },
        "catalogs": {
            key: {
                "digest": value["inventory_catalog_digest"],
                "pool_size": value["inventory_pool_size"],
                "source": value["inventory_source"],
            }
            for key, value in sorted(catalogs.items())
        },
        "dynamic_append_catalogs": {
            key: {
                "digest": value["inventory_catalog_digest"],
                "pool_size": value["inventory_pool_size"],
                "source": value["inventory_source"],
            }
            for key, value in sorted(append_catalogs.items())
        },
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": _file_digest(path),
            }
            for key, path in files.items()
        },
        "max_trajectory_rows": args.max_trajectory_rows,
        "max_retrieval_rows": args.max_retrieval_rows,
        "dev_fraction": float(args.dev_fraction),
        "split_seed": str(args.split_seed),
        "dynamic_holdout_fraction": float(args.dynamic_holdout_fraction),
        "dynamic_holdout_seed": str(args.dynamic_holdout_seed),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited CLSTR vNext semantic views")
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--retrieval_path", required=True)
    parser.add_argument("--trajectories_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clean_preflight_report", required=True)
    parser.add_argument("--max_trajectory_rows", type=int, default=0)
    parser.add_argument("--max_retrieval_rows", type=int, default=0)
    parser.add_argument("--max_causal_pairs_per_group", type=int, default=32)
    parser.add_argument(
        "--max_unordered_branch_pairs_per_split",
        type=int,
        default=512,
    )
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", default="clstr-vnext-v1")
    parser.add_argument("--dynamic_holdout_fraction", type=float, default=0.01)
    parser.add_argument("--dynamic_holdout_minimum_queries", type=int, default=2)
    parser.add_argument("--dynamic_holdout_maximum_skills", type=int, default=256)
    parser.add_argument("--dynamic_holdout_seed", default="clstr-dynamic-v1")
    parser.add_argument(
        "--minimum_train_clusters_per_enabled_kind",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--minimum_dev_clusters_per_enabled_kind",
        type=int,
        default=20,
    )
    parser.add_argument("--minimum_dev_clusters_per_source", type=int, default=5)
    parser.add_argument(
        "--maximum_source_step_shortcut_accuracy",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--minimum_shortcut_feature_coverage",
        type=float,
        default=0.5,
    )
    parser.add_argument("--minimum_verified_error_prefix_rows", type=int, default=1000)
    parser.add_argument(
        "--minimum_verified_error_prefix_dev_rows",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--minimum_verified_recovery_prefix_rows",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--minimum_verified_recovery_prefix_dev_rows",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--require_robust_prefix_views",
        action="store_true",
        help="Promote verified executed-error/recovery views to a data-release gate",
    )
    parser.set_defaults(require_robust_prefix_views=False)
    parser.add_argument(
        "--allow_empty_causal_branch_pairs",
        action="store_false",
        dest="require_causal_branch_pairs",
    )
    parser.set_defaults(require_causal_branch_pairs=True)
    parser.add_argument(
        "--require_causal_outcome_pairs",
        action="store_true",
        help="Require verified same-action/different-result pairs for an optional ablation",
    )
    parser.set_defaults(require_causal_outcome_pairs=False)
    return parser.parse_args()


def main() -> None:
    manifest = build_views(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if manifest["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
