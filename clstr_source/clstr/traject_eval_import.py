from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from clstr.traject_split import assign_traject_split


@dataclass(frozen=True)
class TrajectBenchRouteCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def _tool_name_key(value: str) -> str:
    return _slug(value)


def traject_skill_id(tool: dict[str, Any]) -> str:
    parent = tool.get("parent tool name") or tool.get("parent_tool_name") or ""
    api = tool.get("API name") or tool.get("api_name") or ""
    tool_name = tool.get("tool name") or tool.get("tool_name") or ""
    parent_slug = _slug(str(parent))
    api_slug = _slug(str(api))
    if (parent or api) and not (parent_slug == "unknown" and api_slug == "unknown" and tool_name):
        return f"traject/{parent_slug}/{api_slug}"
    return f"traject/{_slug(str(tool_name))}"


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("tool name") or tool.get("tool_name") or "").strip()


def _tool_description(tool: dict[str, Any]) -> str:
    return str(
        tool.get("tool description")
        or tool.get("tool_description")
        or tool.get("parent tool description")
        or tool.get("parent_tool_description")
        or ""
    )


def _tool_input_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_parameters": tool.get("required_parameters") or tool.get("required parameters") or [],
        "optional_parameters": tool.get("optional_parameters") or tool.get("optional parameters") or [],
    }


def _skill_record(
    tool: dict[str, Any],
    source_path: Path,
    *,
    dedup_method: str = "traject_tool_identity_v2",
) -> dict[str, Any]:
    skill_id = traject_skill_id(tool)
    description = _tool_description(tool)
    return {
        "skill_id": skill_id,
        "canonical_skill_id": skill_id,
        "name": _tool_name(tool) or skill_id,
        "description": description,
        "source": "TRAJECT-Bench",
        "environment": tool.get("domain name") or tool.get("domain_name"),
        "executor_desc": description,
        "input_schema": _tool_input_schema(tool),
        "output_schema": tool.get("output_info") or {},
        "body": str(tool.get("code") or ""),
        "provenance": {
            "source_dataset": "TRAJECT-Bench public_data",
            "source_file": str(source_path),
            "parent_tool_name": tool.get("parent tool name") or tool.get("parent_tool_name"),
            "api_name": tool.get("API name") or tool.get("api_name"),
            "dedup_method": dedup_method,
        },
    }


def _action_text_for_tool(tool: dict[str, Any]) -> str:
    params: list[dict[str, Any]] = []
    for key in ("required parameters", "required_parameters", "optional parameters", "optional_parameters"):
        value = tool.get(key) or []
        if isinstance(value, list):
            params.extend(item for item in value if isinstance(item, dict))
    if not params:
        return _tool_name(tool)
    param_text = ", ".join(
        f"{item.get('name', '')}={item.get('value', item.get('default', ''))}"
        for item in params
    )
    return f"{_tool_name(tool)}: {param_text}"


def _result_text_for_tool(tool: dict[str, Any]) -> str:
    value = tool.get("executed_output")
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _state_text(
    *,
    query: str,
    traj_type: str,
    domain: str,
    task_name: str,
    task_description: str,
    history: list[str],
) -> str:
    previous_tools = ", ".join(history) if history else "<empty>"
    return "\n".join(
        [
            f"goal: {query}",
            "benchmark: TRAJECT-Bench",
            f"trajectory_type: {traj_type}",
            f"domain: {domain}",
            f"task_name: {task_name}",
            f"task_description: {task_description}",
            f"previous_tools: {previous_tools}",
        ]
    )


def _current_state_text(
    *,
    query: str,
    traj_type: str,
    domain: str,
    task_name: str,
    task_description: str,
) -> str:
    return "\n".join(
        [
            f"goal: {query}",
            "benchmark: TRAJECT-Bench",
            f"trajectory_type: {traj_type}",
            f"domain: {domain}",
            f"task_name: {task_name}",
            f"task_description: {task_description}",
        ]
    )


def iter_traject_query_files(public_data: str | Path) -> Iterable[tuple[str, str, Path]]:
    public_data = Path(public_data)
    for traj_type in ("parallel", "sequential"):
        root = public_data / traj_type
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.json")):
            yield traj_type, path.parent.name, path


def _allowed(value: str, allowed_values: set[str] | None) -> bool:
    return allowed_values is None or value in allowed_values


def _optional_set(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed = {str(item).strip() for item in values if str(item).strip()}
    return parsed or None


def _load_tool_index(public_data: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    tools_path = public_data / "tools" / "all_tools.json"
    payload = _read_json(tools_path)
    if not isinstance(payload, list):
        raise ValueError(f"TRAJECT tools file must be a list: {tools_path}")
    skill_rows_by_id: dict[str, dict[str, Any]] = {}
    tools_by_name: dict[str, dict[str, Any]] = {}
    ambiguous_tool_names: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        skill_id = traject_skill_id(row)
        if skill_id not in skill_rows_by_id:
            skill_rows_by_id[skill_id] = _skill_record(row, tools_path)
        name = _tool_name(row)
        name_key = _tool_name_key(name)
        if not name:
            continue
        existing = tools_by_name.get(name_key)
        if existing is None:
            tools_by_name[name_key] = row
        elif traject_skill_id(existing) != skill_id:
            ambiguous_tool_names.add(name_key)
    for name in ambiguous_tool_names:
        tools_by_name.pop(name, None)
    return list(skill_rows_by_id.values()), tools_by_name, sorted(ambiguous_tool_names)


def _resolve_tool_for_qrel(tool: dict[str, Any], tools_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return tools_by_name.get(_tool_name_key(_tool_name(tool))) or tool


def load_trajectbench_route_corpus(
    public_data: str | Path,
    *,
    split_partition: str = "test",
) -> TrajectBenchRouteCorpus:
    """Build a held-out current-action route corpus from TRAJECT public data.

    The legal pool is the global public ``traject/*`` inventory rather than a
    row's oracle tool list. Sequential trajectories retain their declared
    order; parallel trajectories treat every still-unexecuted required tool as
    an equivalent positive because the source list is not a causal order.
    """

    if split_partition not in {"dev", "test"}:
        raise ValueError("TrajectBench route evaluation requires dev or test split")
    public_data = Path(public_data).resolve()
    skill_rows, tools_by_name, ambiguous_tool_names = _load_tool_index(public_data)
    skills_by_id = {str(row["skill_id"]): row for row in skill_rows}
    query_files = list(iter_traject_query_files(public_data))

    normalized_queries: list[
        tuple[str, str, Path, int, dict[str, Any], list[dict[str, Any]]]
    ] = []
    split_counts = {"train": 0, "dev": 0, "test": 0}
    skipped_empty = 0
    skipped_ambiguous = 0
    for traj_type, domain, path in query_files:
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for query_idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            tools = item.get("tool list") or item.get("tool_list") or []
            valid_tools = (
                [tool for tool in tools if isinstance(tool, dict)]
                if isinstance(tools, list)
                else []
            )
            if not valid_tools:
                skipped_empty += 1
                continue
            trajectory_id = f"traject::{traj_type}::{domain}::{path.stem}::{query_idx}"
            assigned_split = assign_traject_split(trajectory_id)
            split_counts[assigned_split] += 1
            resolved_tools: list[dict[str, Any]] = []
            for tool in valid_tools:
                if _tool_name_key(_tool_name(tool)) in ambiguous_tool_names:
                    skipped_ambiguous += 1
                    continue
                resolved = _resolve_tool_for_qrel(tool, tools_by_name)
                skill_id = traject_skill_id(resolved)
                if skill_id not in skills_by_id:
                    skills_by_id[skill_id] = _skill_record(
                        tool,
                        path,
                        dedup_method="traject_query_tool_identity_v2",
                    )
                resolved_tools.append({"raw": tool, "skill_id": skill_id})
            if resolved_tools:
                normalized_queries.append(
                    (traj_type, domain, path, query_idx, item, resolved_tools)
                )

    skills = [skills_by_id[skill_id] for skill_id in sorted(skills_by_id)]
    global_skill_ids = [str(row["skill_id"]) for row in skills]
    source_rows: list[dict[str, Any]] = []
    selected_trajectory_ids: list[str] = []
    parallel_row_count = 0
    multi_positive_row_count = 0
    actual_result_row_count = 0
    for traj_type, domain, path, query_idx, item, tools in normalized_queries:
        trajectory_id = f"traject::{traj_type}::{domain}::{path.stem}::{query_idx}"
        if assign_traject_split(trajectory_id) != split_partition:
            continue
        selected_trajectory_ids.append(trajectory_id)
        query = str(item.get("query") or "").strip()
        task_name = str(item.get("task_name") or item.get("task name") or "").strip()
        task_description = str(
            item.get("task_description") or item.get("task description") or ""
        ).strip()
        current_state = _current_state_text(
            query=query,
            traj_type=traj_type,
            domain=domain,
            task_name=task_name,
            task_description=task_description,
        )
        for step_idx, entry in enumerate(tools):
            target = str(entry["skill_id"])
            remaining = list(
                dict.fromkeys(str(value["skill_id"]) for value in tools[step_idx:])
            )
            equivalents = (
                [skill_id for skill_id in remaining if skill_id != target]
                if traj_type == "parallel"
                else []
            )
            result_text = _result_text_for_tool(entry["raw"])
            parallel_row_count += int(traj_type == "parallel")
            multi_positive_row_count += int(bool(equivalents))
            actual_result_row_count += int(bool(result_text))
            action_text = _action_text_for_tool(entry["raw"])
            source_rows.append(
                {
                    "benchmark": "trajectbench",
                    "task_id": f"{trajectory_id}::{step_idx}",
                    "trajectory_id": trajectory_id,
                    "step_index": step_idx,
                    "goal_text": query,
                    "task_text": task_name,
                    "state_text": current_state,
                    "state_text_current": current_state,
                    "state_text_full": current_state,
                    "history_text": "",
                    "skill_id": (
                        "traject/__start__"
                        if step_idx == 0
                        else str(tools[step_idx - 1]["skill_id"])
                    ),
                    "next_skill_id": target,
                    "equivalent_next_skill_ids": equivalents,
                    "candidate_next_skill_ids": global_skill_ids,
                    "target_action_text": action_text,
                    "next_action_text": action_text,
                    "route_target": "TOOL",
                    "actual_result_skill_id": target if result_text else "",
                    "actual_result_text": result_text,
                    "actual_result_executed": bool(result_text),
                    "next_observation_text": result_text,
                    "observation_source": (
                        "traject_public_executed_output"
                        if result_text
                        else "action_only_no_tool_result"
                    ),
                    "done": step_idx + 1 == len(tools),
                    "provenance": {
                        "source_id": "traject_bench",
                        "source_dataset": "TRAJECT-Bench public_data",
                        "split": split_partition,
                        "split_policy": "deterministic_hash_trajectory_level_v1",
                        "path": str(path.resolve()),
                        "trajectory_type": traj_type,
                        "domain": domain,
                        "parallel_positive_semantics": (
                            "remaining_required_tool_set_v1"
                            if traj_type == "parallel"
                            else "ordered_unique_next_tool_v1"
                        ),
                    },
                }
            )
    if not source_rows:
        raise ValueError(f"TrajectBench {split_partition} split has no route rows")
    source_files = [public_data / "tools" / "all_tools.json"] + [
        path.resolve() for _traj_type, _domain, path in query_files
    ]
    return TrajectBenchRouteCorpus(
        skills=skills,
        source_rows=source_rows,
        report={
            "benchmark": "trajectbench",
            "public_data": str(public_data),
            "split_partition": split_partition,
            "split_policy": "deterministic_hash_trajectory_level_v1",
            "split_counts": split_counts,
            "source_files": [str(path) for path in source_files],
            "source_query_file_count": len(query_files),
            "source_skill_count": len(skills),
            "source_row_count": len(source_rows),
            "trajectory_count": len(set(selected_trajectory_ids)),
            "global_visible_inventory_count": len(global_skill_ids),
            "candidate_source": "global_public_traject_inventory",
            "metric_scope": "heldout_next_skill_routing_diagnostic",
            "official_metric_caveat": (
                "This corpus evaluates held-out skill routing only; it does not "
                "compute TRAJECT-Bench executor Usage, Traj-Satisfy, or Acc."
            ),
            "parallel_row_count": parallel_row_count,
            "multi_positive_row_count": multi_positive_row_count,
            "actual_result_row_count": actual_result_row_count,
            "skipped_empty_tool_list_count": skipped_empty,
            "skipped_ambiguous_tool_count": skipped_ambiguous,
        },
    )


def _iter_eval_records(
    *,
    public_data: Path,
    split: str,
    split_partition: str | None,
    trajectory_types: set[str] | None,
    domains: set[str] | None,
    max_queries: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[str], dict[str, dict[str, Any]]]:
    _skill_rows, tools_by_name, ambiguous_tool_names = _load_tool_index(public_data)
    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    query_only_skill_rows: dict[str, dict[str, Any]] = {}
    counts = {
        "source_query_count": 0,
        "used_query_count": 0,
        "step_query_count": 0,
        "skipped_empty_tool_list": 0,
        "skipped_ambiguous_tool_name": 0,
        "skipped_by_split_partition": 0,
        "query_only_skill_count": 0,
    }
    for traj_type, domain, path in iter_traject_query_files(public_data):
        if not _allowed(traj_type, trajectory_types) or not _allowed(domain, domains):
            continue
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for query_idx, item in enumerate(payload):
            if max_queries is not None and counts["used_query_count"] >= max_queries:
                break
            if not isinstance(item, dict):
                continue
            counts["source_query_count"] += 1
            tools = item.get("tool list") or item.get("tool_list") or []
            valid_tools = [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []
            if not valid_tools:
                counts["skipped_empty_tool_list"] += 1
                continue
            query = str(item.get("query") or "")
            task_name = str(item.get("task_name") or item.get("task name") or "")
            task_description = str(item.get("task_description") or item.get("task description") or "")
            trajectory_id = f"traject::{traj_type}::{domain}::{path.stem}::{query_idx}"
            assigned_split = assign_traject_split(trajectory_id)
            if split_partition is not None and assigned_split != split_partition:
                counts["skipped_by_split_partition"] += 1
                continue
            history: list[str] = []
            used_steps = 0
            for step_idx, tool in enumerate(valid_tools):
                tool_name = _tool_name(tool)
                if _tool_name_key(tool_name) in ambiguous_tool_names:
                    counts["skipped_ambiguous_tool_name"] += 1
                    history.append(tool_name)
                    continue
                resolved_tool = _resolve_tool_for_qrel(tool, tools_by_name)
                resolved_skill_id = traject_skill_id(resolved_tool)
                if resolved_tool is tool and resolved_skill_id not in query_only_skill_rows:
                    query_only_skill_rows[resolved_skill_id] = _skill_record(
                        tool,
                        path,
                        dedup_method="traject_query_tool_identity_v2",
                    )
                    counts["query_only_skill_count"] += 1
                query_id = f"{trajectory_id}::{step_idx}"
                text = _state_text(
                    query=query,
                    traj_type=traj_type,
                    domain=domain,
                    task_name=task_name,
                    task_description=task_description,
                    history=history,
                )
                query_rows.append(
                    {
                        "query_id": query_id,
                        "query_text": text,
                        "query": query,
                        "task": domain,
                        "task_name": task_name,
                        "task_description": task_description,
                        "source": "TRAJECT-Bench",
                        "split": split,
                        "assigned_traject_split": assigned_split,
                        "split_policy": "deterministic_hash_trajectory_level_v1",
                        "trajectory_id": trajectory_id,
                        "trajectory_type": traj_type,
                        "domain": domain,
                        "step_index": step_idx,
                        "history_text": " -> ".join(history),
                    }
                )
                qrel_rows.append(
                    {
                        "query_id": query_id,
                        "skill_id": resolved_skill_id,
                        "relevance": 1,
                        "task": domain,
                        "source": "TRAJECT-Bench",
                        "split": split,
                        "assigned_traject_split": assigned_split,
                        "split_policy": "deterministic_hash_trajectory_level_v1",
                        "trajectory_type": traj_type,
                        "step_index": step_idx,
                    }
                )
                counts["step_query_count"] += 1
                used_steps += 1
                history.append(_tool_name(tool) or _action_text_for_tool(tool))
            if used_steps:
                counts["used_query_count"] += 1
    return query_rows, qrel_rows, counts, ambiguous_tool_names, query_only_skill_rows


def import_traject_eval(
    *,
    public_data: str | Path,
    output_dir: str | Path = "data/traject_eval_traject_split_test",
    split: str = "test",
    split_partition: str | None = None,
    trajectory_types: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    public_data = Path(public_data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    skill_rows, _tools_by_name, ambiguous_tool_names = _load_tool_index(public_data)
    query_rows, qrel_rows, counts, skipped_ambiguous_tool_names, query_only_skill_rows = _iter_eval_records(
        public_data=public_data,
        split=split,
        split_partition=split_partition,
        trajectory_types=_optional_set(trajectory_types),
        domains=_optional_set(domains),
        max_queries=max_queries,
    )
    ambiguous_tool_names = sorted(set(ambiguous_tool_names) | set(skipped_ambiguous_tool_names))
    skill_rows_by_id = {row["skill_id"]: row for row in skill_rows}
    for skill_id, row in query_only_skill_rows.items():
        skill_rows_by_id.setdefault(skill_id, row)
    skill_rows = [skill_rows_by_id[skill_id] for skill_id in sorted(skill_rows_by_id)]

    _write_jsonl(output / "queries.jsonl", query_rows)
    _write_jsonl(output / "skills.jsonl", skill_rows)
    _write_jsonl(output / "qrels.jsonl", qrel_rows)

    skill_ids = {row["skill_id"] for row in skill_rows}
    qrel_skill_ids = {row["skill_id"] for row in qrel_rows if int(row.get("relevance", 0)) > 0}
    manifest = {
        "status": "ok",
        "source_dataset": "TRAJECT-Bench public_data",
        "public_data": str(public_data),
        "output_dir": str(output),
        "split": split,
        "query_count": len(query_rows),
        "skill_count": len(skill_rows),
        "qrel_count": len(qrel_rows),
        "missing_qrel_skill_ids": sorted(qrel_skill_ids - skill_ids),
        "ambiguous_tool_names": ambiguous_tool_names,
        "counts": counts,
        "filters": {
            "trajectory_types": sorted(_optional_set(trajectory_types) or []),
            "domains": sorted(_optional_set(domains) or []),
            "split_partition": split_partition,
            "max_queries": max_queries,
        },
        "files": {
            "queries": "queries.jsonl",
            "skills": "skills.jsonl",
            "qrels": "qrels.jsonl",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
