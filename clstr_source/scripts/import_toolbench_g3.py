#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


SOURCE_DATASET = "ToolBench-G3"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def _action_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _strip_action_prefix(value: str) -> str:
    for prefix in ("get_",):
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix):]
    return value


def _action_alias(tool_name: str, api_name: str) -> str:
    api = _action_token(api_name)
    tool = _action_token(tool_name)
    return f"{api}_for_{tool}"


def _action_aliases(tool_name: str, api_name: str) -> list[str]:
    api = _action_token(api_name)
    tool = _action_token(tool_name)
    api_variants = {api, _strip_action_prefix(api)}
    tool_variants = {tool}
    if tool and not tool.startswith("get_"):
        tool_variants.add(f"get_{tool}")
    aliases = {
        f"{api_variant}_for_{tool_variant}"
        for api_variant in api_variants
        for tool_variant in tool_variants
        if api_variant and tool_variant
    }
    return sorted(aliases)


def toolbench_skill_id(tool_name: str, api_name: str) -> str:
    return f"toolbench-g3/{_slug(tool_name)}/{_slug(api_name)}"


def _resolve_data_root(source_root: str | Path) -> Path:
    root = Path(source_root)
    if (root / "instruction" / "G3_query.json").is_file():
        return root
    if (root / "data" / "instruction" / "G3_query.json").is_file():
        return root / "data"
    raise FileNotFoundError(f"ToolBench data root missing instruction/G3_query.json: {root}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _skill_record(api: dict[str, Any], source_path: Path) -> dict[str, Any]:
    tool_name = str(api.get("tool_name") or "")
    api_name = str(api.get("api_name") or "")
    sid = toolbench_skill_id(tool_name, api_name)
    return {
        "skill_id": sid,
        "canonical_skill_id": sid,
        "name": api_name or sid,
        "description": str(api.get("api_description") or api_name or sid),
        "alternate_descriptions": [],
        "environment": api.get("category_name"),
        "source": SOURCE_DATASET,
        "source_files": [str(source_path)],
        "executor_desc": api.get("api_description") or "",
        "input_schema": {
            "required_parameters": api.get("required_parameters") or [],
            "optional_parameters": api.get("optional_parameters") or [],
            "method": api.get("method"),
        },
        "output_schema": api.get("template_response") or {},
        "body": "",
        "provenance": {
            "dedup_method": "toolbench_g3_tool_api_identity_v2",
            "tool_name": tool_name,
            "api_name": api_name,
            "action_alias": _action_alias(tool_name, api_name),
            "action_aliases": _action_aliases(tool_name, api_name),
        },
    }


def _query_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("query_id") or row.get("id") or index)


def _answer_path(data_root: Path, qid: str) -> Path | None:
    answer_dir = data_root / "answer" / "G3_answer"
    candidates = sorted(answer_dir.glob(f"{qid}_*.json"))
    return candidates[0] if candidates else None


def _is_finish_action(action_name: str) -> bool:
    return str(action_name or "").strip().lower() == "finish"


def _finish_success_score(node: dict[str, Any]) -> int:
    text_parts = []
    for child in node.get("children") or []:
        if isinstance(child, dict) and str(child.get("node_type") or "") == "Action Input":
            text_parts.append(str(child.get("description") or ""))
            text_parts.append(str(child.get("observation") or ""))
    text = " ".join(text_parts).lower()
    if "give_up" in text or "restart" in text:
        return 0
    if "give_answer" in text or "success" in text:
        return 2
    return 1


def _resolve_action_skill(action_name: str, alias_to_skill: dict[str, str]) -> str | None:
    action = _action_token(action_name)
    sid = alias_to_skill.get(action)
    return sid or None


def _extract_branch_candidates(tree_root: dict[str, Any]) -> list[tuple[int, list[dict[str, str]]]]:
    candidates: list[tuple[int, list[dict[str, str]]]] = []

    def dfs(node: dict[str, Any], path: list[dict[str, str]]) -> None:
        children = [child for child in (node.get("children") or []) if isinstance(child, dict)]
        if not children:
            candidates.append((0, list(path)))
            return
        expanded = False
        for child in children:
            node_type = str(child.get("node_type") or "")
            if node_type == "Action":
                action_name = str(child.get("description") or "")
                if _is_finish_action(action_name):
                    candidates.append((_finish_success_score(child), list(path)))
                    expanded = True
                    continue
                action_inputs = [
                    grandchild
                    for grandchild in (child.get("children") or [])
                    if isinstance(grandchild, dict) and str(grandchild.get("node_type") or "") == "Action Input"
                ]
                if not action_inputs:
                    continue
                for action_input in action_inputs:
                    step = {
                        "action_name": action_name,
                        "action_input": str(action_input.get("description") or ""),
                        "observation": str(action_input.get("observation") or ""),
                    }
                    dfs(action_input, path + [step])
                    expanded = True
            elif node_type in {"Action Input", "Thought"}:
                dfs(child, path)
                expanded = True
        if not expanded:
            candidates.append((0, list(path)))

    if isinstance(tree_root, dict):
        dfs(tree_root, [])
    return [(score, path) for score, path in candidates if path]


def _extract_first_branch_steps(tree_root: dict[str, Any]) -> list[dict[str, str]]:
    candidates = _extract_branch_candidates(tree_root)
    if not candidates:
        return []
    return max(candidates, key=lambda item: (item[0], len(item[1])))[1]


def _state_text(
    query: str,
    previous_tools: list[str],
    *,
    include_history: bool = True,
) -> str:
    previous = ", ".join(previous_tools) if previous_tools else "<empty>"
    parts = [
        f"goal: {query}",
        "benchmark: ToolBench-G3",
        "trajectory_type: intra_collection_multi_tool",
    ]
    if include_history:
        parts.append(f"previous_tools: {previous}")
    return "\n".join(parts)


def _trajectory_rows(
    query: dict[str, Any],
    query_index: int,
    alias_to_skill: dict[str, str],
    answer_path: Path,
    answer: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    answer = answer if answer is not None else _read_json(answer_path)
    tree_root = ((answer.get("tree") or {}).get("tree") or {}) if isinstance(answer, dict) else {}
    candidates = _extract_branch_candidates(tree_root)
    if not candidates:
        return [], []
    scored_candidates: list[tuple[bool, int, int, int, list[dict[str, str]], list[str], list[str | None]]] = []
    for score, path in candidates:
        resolved = [_resolve_action_skill(step["action_name"], alias_to_skill) for step in path]
        unresolved = [
            step["action_name"]
            for step, sid in zip(path, resolved)
            if not sid
        ]
        scored_candidates.append((not unresolved, score, -len(unresolved), len(path), path, unresolved, resolved))
    _all_resolved, _score, _neg_unresolved, _length, steps, unresolved_actions, resolved_skill_ids = max(
        scored_candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    if unresolved_actions:
        return [], unresolved_actions
    qid = _query_id(query, query_index)
    query_text = str(query.get("query") or "")
    trajectory_id = f"toolbench-g3-{qid}"
    previous_tools: list[str] = []
    rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        sid = str(resolved_skill_ids[step_index])
        next_sid = ""
        if step_index + 1 < len(steps):
            next_sid = str(resolved_skill_ids[step_index + 1])
        state_text_current = _state_text(
            query_text,
            previous_tools,
            include_history=False,
        )
        row = {
            "benchmark": "toolbench_g3",
            "task_id": f"{trajectory_id}::{step_index}",
            "trajectory_id": trajectory_id,
            "step_index": step_index,
            "goal_text": query_text,
            "task_text": query_text,
            "state_text": state_text_current,
            "state_text_current": state_text_current,
            "state_text_full": _state_text(
                query_text,
                previous_tools,
                include_history=True,
            ),
            "history_text": " -> ".join(previous_tools),
            "action_text": f"{step['action_name']}: {step['action_input']}",
            "expert_action": f"{step['action_name']}: {step['action_input']}",
            "admissible_actions": [],
            "next_action_text": steps[step_index + 1]["action_name"] if next_sid else "",
            "next_observation_text": step["observation"],
            "observation_source": "actual_tool_result",
            "skill_id": sid,
            "next_skill_id": next_sid,
            "done": step_index + 1 >= len(steps),
            "reward": 1.0 if answer.get("win") else 0.0,
            "loss_mask": {
                "L_policy": True,
                "L_trans": True,
                "L_trans_skill_ce": bool(next_sid),
                "belief": False,
                "STOP": True,
                "routing": True,
            },
            "source_quality": "toolbench_g3_dfs_tree",
            "candidate_source": "toolbench_g3_api_list",
            "on_policy_rollout": False,
            "m_t_source": "serialized_tool_history",
            "provenance": {
                "source_id": "toolbench_g3",
                "source_dataset": SOURCE_DATASET,
                "split": "train_or_released_g3",
                "answer_path": str(answer_path),
            },
            "_unified_source": "toolbench_g3",
        }
        rows.append(row)
        previous_tools.append(step["action_name"])
    return rows, []


def import_toolbench_g3(
    source_root: str | Path,
    output_dir: str | Path = "data/toolbench_g3",
    max_queries: int | None = None,
) -> dict[str, Any]:
    data_root = _resolve_data_root(source_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    instruction_path = data_root / "instruction" / "G3_query.json"
    queries = _read_json(instruction_path)
    if not isinstance(queries, list):
        raise ValueError(f"Expected list in {instruction_path}")
    if max_queries is not None:
        queries = queries[:max_queries]

    skills: dict[str, dict[str, Any]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    answer_files = 0
    successful_answer_files = 0
    failed_answer_files = 0
    skipped_failed_answer_files = 0
    skipped_unresolved_answer_files = 0
    unresolved_action_count = 0
    unresolved_action_samples: list[dict[str, str]] = []

    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            continue
        qid = _query_id(query, query_index)
        query_text = str(query.get("query") or "")
        api_list = [api for api in (query.get("api_list") or []) if isinstance(api, dict)]
        alias_to_skill: dict[str, str] = {}
        all_skill_ids: list[str] = []
        for api in api_list:
            record = _skill_record(api, instruction_path)
            skills.setdefault(record["skill_id"], record)
            for alias in record["provenance"]["action_aliases"]:
                alias_to_skill.setdefault(alias, record["skill_id"])
            all_skill_ids.append(record["skill_id"])

        relevant = query.get("relevant APIs") or query.get("relevant_apis") or []
        for pair_index, item in enumerate(relevant):
            if not isinstance(item, list) or len(item) < 2:
                continue
            sid = toolbench_skill_id(str(item[0]), str(item[1]))
            if sid not in all_skill_ids:
                continue
            retrieval_rows.append(
                {
                    "source": "toolbench_g3",
                    "query_id": f"toolbench-g3-{qid}",
                    "query_text": query_text,
                    "positive_skill_id": sid,
                    "negative_skill_ids": [candidate for candidate in all_skill_ids if candidate != sid],
                    "provenance": {
                        "source_dataset": SOURCE_DATASET,
                        "source_path": str(instruction_path),
                        "query_index": query_index,
                        "positive_index": pair_index,
                        "split": "train_or_released_g3",
                    },
                }
            )

        answer_path = _answer_path(data_root, qid)
        if answer_path is not None:
            answer_files += 1
            answer = _read_json(answer_path)
            if isinstance(answer, dict) and answer.get("win") is True:
                successful_answer_files += 1
                rows, unresolved_actions = _trajectory_rows(
                    query,
                    query_index,
                    alias_to_skill,
                    answer_path,
                    answer=answer,
                )
                if unresolved_actions:
                    skipped_unresolved_answer_files += 1
                    unresolved_action_count += len(unresolved_actions)
                    for action_name in unresolved_actions:
                        if len(unresolved_action_samples) < 100:
                            unresolved_action_samples.append(
                                {
                                    "query_id": str(qid),
                                    "action_name": str(action_name),
                                }
                            )
                else:
                    trajectory_rows.extend(rows)
            else:
                failed_answer_files += 1
                skipped_failed_answer_files += 1

    retrieval_count = _write_jsonl(output / "retrieval.jsonl", retrieval_rows)
    trajectory_count = _write_jsonl(output / "trajectories.jsonl", trajectory_rows)
    skill_count = _write_jsonl(output / "skills.jsonl", (skills[sid] for sid in sorted(skills)))

    manifest = {
        "status": "ok",
        "source_dataset": SOURCE_DATASET,
        "source_root": str(data_root),
        "output_dir": str(output),
        "files": {
            "retrieval": "retrieval.jsonl",
            "trajectories": "trajectories.jsonl",
            "skills": "skills.jsonl",
        },
        "queries": len(queries),
        "answer_files": answer_files,
        "successful_answer_files": successful_answer_files,
        "failed_answer_files": failed_answer_files,
        "skipped_failed_answer_files": skipped_failed_answer_files,
        "skipped_unresolved_answer_files": skipped_unresolved_answer_files,
        "unresolved_action_count": unresolved_action_count,
        "unresolved_action_samples": unresolved_action_samples,
        "retrieval_pairs": retrieval_count,
        "trajectory_rows": trajectory_count,
        "skills": skill_count,
        "max_queries": max_queries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ToolBench-G3 into CLSTR unified v2 JSONL inputs.")
    parser.add_argument("--source_root", required=True, help="Path to ToolBench/data or ToolBench root.")
    parser.add_argument("--output_dir", default="data/toolbench_g3")
    parser.add_argument("--max_queries", type=int)
    args = parser.parse_args()
    manifest = import_toolbench_g3(args.source_root, args.output_dir, max_queries=args.max_queries)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
