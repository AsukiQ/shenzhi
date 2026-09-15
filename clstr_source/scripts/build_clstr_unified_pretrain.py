#!/usr/bin/env python
"""Build a unified CLSTR pretrain corpus that merges trajectory-style data from
multiple benchmarks into a single JSONL with a shared schema. Performs a
leakage audit against AppWorld dev/test/test_challenge tasks before writing.

This script does NOT train. It produces:
  data/clstr_unified_pretrain_v2/trajectories.jsonl      (Stream A: POMDP rows)
  data/clstr_unified_pretrain_v2/retrieval.jsonl         (Stream B: query-skill pairs)
  data/clstr_unified_pretrain_v2/skill_pool.jsonl        (used canonical skills)
  data/clstr_unified_pretrain_v2/source_inventory.jsonl  (available/missing sources)
  data/clstr_unified_pretrain_v2/leakage_audit.json      (post-build audit)
  data/clstr_unified_pretrain_v2/manifest.json           (counts + file map)

Stream A schema (matches clstr/full_base_train fields):
  benchmark, task_id, trajectory_id, step_index, goal_text, state_text,
  state_text_current, state_text_full, history_text, replay_prefix,
  action_text, next_action_text, next_observation_text, skill_id,
  next_skill_id, admissible_actions, expert_action, done, reward,
  loss_mask, source_quality, candidate_source, on_policy_rollout,
  m_t_source, provenance

Stream B schema:
  source, query_id, query_text, positive_skill_id, negative_skill_ids,
  provenance

Leakage policy:
  - AppWorld dev / test_normal / test_challenge task_ids EXCLUDED from training
  - SKILLRET test split EXCLUDED
  - Only train splits + train-only synthetic data enter the corpus
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.history_channel import (
    audit_history_channel_rows,
    materialize_history_free_state,
    router_state_text,
)
from clstr.traject_split import assign_traject_split


DEFAULT_REPO_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr")
DEFAULT_TRAJECTORY_RETRIEVAL_CAPS = {
    "toolbench_g3": -1,
    "traject_bench": -1,
    "alfworld": 10000,
    "scienceworld": 10000,
}
DEFAULT_V4_1_TRAJECTORY_RETRIEVAL_CAPS = {
    "toolbench_g3": -1,
    "traject_bench": -1,
    "alfworld": 10000,
    "scienceworld": 0,
    "webshop": 10000,
}
V4_2_ALFWORLD_HF_RECIPE = "v4_2_alfworld_hf_quality"
V4_2_PROGRESSIVE_FINAL_RECIPE = "v4_2_progressive_final"
V4_CLEAN_RECIPES = {"v4_1", "v4_1b", V4_2_ALFWORLD_HF_RECIPE, V4_2_PROGRESSIVE_FINAL_RECIPE}
DEFAULT_PROGRESSIVE_FILTERED_WEAK_POLICY_CAP = 12000
HF_ALFWORLD_ADMISSIBLE_SUCCESS_DATASET = "kuririrn/sft_alfworld_trajectory_dataset_v3to5_admissible_success"
WEBSHOP_HF_GPT4_FILENAME = "webshop_gpt4_0613.json"
WEBSHOP_ACTION_SKILLS = {
    "think": {
        "skill_id": "webshop/webshop-reasoning-planner",
        "name": "WebShop Reasoning Planner",
        "description": "Plan the next WebShop interaction step from the user goal, page observation, and previous actions.",
    },
    "search": {
        "skill_id": "webshop/webshop-search-executor",
        "name": "WebShop Search Executor",
        "description": "Issue a product search query in the WebShop environment.",
    },
    "click": {
        "skill_id": "webshop/webshop-click-executor",
        "name": "WebShop Click Executor",
        "description": "Click a visible product, option, navigation button, or purchase control in the WebShop environment.",
    },
    "finish": {
        "skill_id": "webshop/webshop-finish-executor",
        "name": "WebShop Finish Executor",
        "description": "Finish the WebShop task with the selected item or final response.",
    },
}


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_benchmark_caps(value: str | dict[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): int(cap) for key, cap in value.items() if str(key).strip()}
    caps: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"benchmark cap must use name=value format: {item}")
        name, raw_cap = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"benchmark cap has empty benchmark name: {item}")
        caps[name] = int(raw_cap.strip())
    return caps


def parse_string_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_appworld_dev_test_ids(repo_root: str | Path | None = None) -> set[str]:
    """All task_ids in AppWorld dev / test_normal / test_challenge — must
    never appear in training stream."""
    root = _repo_root(repo_root)
    excluded: set[str] = set()
    for name in ("dev_tasks.jsonl", "test_normal_tasks.jsonl", "test_challenge_tasks.jsonl"):
        path = root / "data" / "appworld_routing" / name
        for row in read_jsonl(path):
            tid = row.get("task_id")
            if tid:
                excluded.add(str(tid))
    return excluded


def load_skillret_test_query_ids(repo_root: str | Path | None = None) -> set[str]:
    root = _repo_root(repo_root)
    excluded: set[str] = set()
    for row in read_jsonl(root / "data" / "skillret" / "qrels.jsonl"):
        if str(row.get("split", "")).lower() == "test":
            qid = row.get("query_id")
            if qid:
                excluded.add(str(qid))
    return excluded


def normalize_trajectory_row(row: dict, default_source: str) -> dict:
    """Project a row from full_base_train / agentgym schema into the canonical
    Stream A schema. Missing fields are filled with sentinel defaults so
    downstream loaders can rely on the keys existing."""
    normalized = {
        "benchmark": row.get("benchmark", "unknown"),
        "task_id": row.get("task_id", ""),
        "trajectory_id": row.get("trajectory_id", ""),
        "step_index": int(row.get("step_index", 0)),
        "goal_text": row.get("goal_text", ""),
        "task_text": row.get("task_text", ""),
        "state_text": row.get("state_text", ""),
        "state_text_current": row.get("state_text_current", ""),
        "state_text_full": row.get("state_text_full", row.get("state_text", "")),
        "history_text": row.get("history_text", ""),
        "replay_prefix": row.get("replay_prefix", []),
        "action_text": row.get("action_text", ""),
        "expert_action": row.get("expert_action", row.get("action_text", "")),
        "admissible_actions": row.get("admissible_actions", "[]"),
        "next_action_text": row.get("next_action_text", ""),
        "next_observation_text": row.get("next_observation_text", ""),
        "observation_source": row.get("observation_source", ""),
        "skill_id": row.get("skill_id", ""),
        "next_skill_id": row.get("next_skill_id", ""),
        "done": bool(row.get("done", False)),
        "reward": row.get("reward", "None"),
        "loss_mask": row.get("loss_mask", "{}"),
        "source_quality": row.get("source_quality", "unknown"),
        "candidate_source": row.get("candidate_source", "unknown"),
        "on_policy_rollout": bool(row.get("on_policy_rollout", False)),
        "m_t_source": row.get("m_t_source", ""),
        "provenance": row.get("provenance", ""),
        "_unified_source": default_source,
    }
    return materialize_history_free_state(normalized, replace_state_text=True)


def _write_trajectory_row(fout: Any, row: dict[str, Any]) -> None:
    prepared = materialize_history_free_state(row, replace_state_text=True)
    fout.write(json.dumps(prepared, ensure_ascii=False) + "\n")


def _provenance_dict(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        return provenance
    if isinstance(provenance, str):
        try:
            parsed = json.loads(provenance)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalized_string_list(value: Any) -> list[str]:
    parsed = _parse_maybe_json(value)
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _loss_mask_dict(row: dict[str, Any]) -> dict[str, bool]:
    parsed = _parse_maybe_json(row.get("loss_mask"))
    if not isinstance(parsed, dict):
        parsed = {}
    return {str(key): bool(value) for key, value in parsed.items()}


def _v4_1_full_base_skip_reason(row: dict[str, Any]) -> str | None:
    benchmark = str(row.get("benchmark") or "unknown")
    source_quality = str(row.get("source_quality") or "")
    provenance = _provenance_dict(row)
    if source_quality == "aux_only" or provenance.get("source_id") == "aux_skillnet_rebuilt":
        return f"v4_1_aux_only_full_base_filtered:{benchmark}"
    return None


def _trajectory_skip_reason(row: dict[str, Any], dataset_recipe: str) -> str | None:
    if dataset_recipe in V4_CLEAN_RECIPES:
        return _v4_1_full_base_skip_reason(row)
    return None


def _trajectory_step_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("benchmark") or "unknown"),
        str(row.get("trajectory_id") or row.get("task_id") or ""),
        int(row.get("step_index", 0)),
    )


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def _parse_maybe_json(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("from") or "").lower()


def _message_content(message: dict[str, Any]) -> str:
    return str(message.get("content") or message.get("value") or "")


def _extract_alfworld_action(content: str) -> str:
    text = str(content or "")
    match = re.search(r"(?:Action|Act)\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text.strip()
    action = match.group(1).strip()
    action = action.splitlines()[0].strip()
    return action.strip().strip("`")


def _extract_alfworld_task(content: str) -> str:
    text = str(content or "")
    for pattern in (
        r"Your task is to:\s*(.+?)(?:\n|$)",
        r"task is to:\s*(.+?)(?:\n|$)",
        r"Task:\s*(.+?)(?:\n|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _extract_admissible_actions(content: str) -> list[str]:
    text = str(content or "")
    match = re.search(r"Admissible actions:\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"AVAILABLE ACTIONS:\s*(.+)", text, flags=re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1)
    else:
        raw = match.group(1)
    return [item.strip().strip("'\"") for item in str(raw).split(",") if item.strip()]


def _alfworld_skill_id_for_action(action: str) -> str:
    text = str(action or "").strip().lower()
    if text.startswith("go to "):
        return "alfworld/alfworld-receptacle-navigator"
    if text.startswith("take "):
        return "alfworld/alfworld-object-picker"
    if text.startswith("put "):
        return "alfworld/alfworld-object-placer"
    if text.startswith("open "):
        return "alfworld/alfworld-receptacle-opener"
    if text.startswith("close "):
        return "alfworld/alfworld-receptacle-closer"
    if text.startswith("clean "):
        return "alfworld/alfworld-clean-object"
    if text.startswith("heat "):
        return "alfworld/alfworld-object-heater"
    if text.startswith("cool "):
        return "alfworld/alfworld-object-cooler"
    if text.startswith("toggle ") or text.startswith("use "):
        return "alfworld/alfworld-device-operator"
    if text.startswith("look") or text.startswith("inventory"):
        return "alfworld/alfworld-environment-scanner"
    return "alfworld/alfworld-task-verifier"


def traject_skill_id(tool: dict[str, Any]) -> str:
    parent = tool.get("parent tool name") or tool.get("parent_tool_name") or ""
    api = tool.get("API name") or tool.get("api_name") or ""
    tool_name = tool.get("tool name") or tool.get("tool_name") or ""
    parent_slug = _slug(parent)
    api_slug = _slug(api)
    if (parent or api) and not (parent_slug == "unknown" and api_slug == "unknown" and tool_name):
        return f"traject/{parent_slug}/{api_slug}"
    return f"traject/{_slug(tool_name)}"


def _parameters_for_state(tool: dict[str, Any]) -> list[dict[str, Any]]:
    params = []
    for key in ("required parameters", "required_parameters", "optional parameters", "optional_parameters"):
        value = tool.get(key) or []
        if isinstance(value, list):
            params.extend(item for item in value if isinstance(item, dict))
    return params


def _action_text_for_tool(tool: dict[str, Any]) -> str:
    params = _parameters_for_state(tool)
    if not params:
        return str(tool.get("tool name") or tool.get("tool_name") or "")
    param_text = ", ".join(
        f"{item.get('name', '')}={item.get('value', item.get('default', ''))}"
        for item in params
    )
    return f"{tool.get('tool name') or tool.get('tool_name')}: {param_text}"


def _traject_state_text(
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


def find_traject_public_data(repo_root: str | Path | None = None) -> Path | None:
    root = _repo_root(repo_root)
    candidates = [
        root / "data" / "traject_bench" / "public_data",
        root / "data" / "traject_bench",
        root.parent / "TRAJECT-Bench" / "public_data",
        root.parent / "TRAJECT_Bench" / "public_data",
        root.parent / "TRAJECT-Bench",
    ]
    for candidate in candidates:
        if (candidate / "tools").exists() and ((candidate / "parallel").exists() or (candidate / "sequential").exists()):
            return candidate
    return None


def iter_traject_query_files(public_data: Path) -> Iterable[tuple[str, str, Path]]:
    for traj_type in ("parallel", "sequential"):
        root = public_data / traj_type
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.json")):
            domain = path.parent.name
            yield traj_type, domain, path


def _iter_traject_tool_entries(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    if payload.get("tool name") or payload.get("tool_name"):
        yield payload
        return
    for value in payload.values():
        yield from _iter_traject_tool_entries(value)


def _traject_visible_inventory_by_domain(public_data: Path) -> dict[str, list[str]]:
    tools_dir = public_data / "tools"
    by_domain: dict[str, set[str]] = defaultdict(set)
    all_skill_ids: set[str] = set()
    if tools_dir.exists():
        for path in sorted(tools_dir.glob("*.json")):
            try:
                payload = read_json(path)
            except Exception:
                continue
            file_domain = path.stem.removesuffix("_tool")
            for tool in _iter_traject_tool_entries(payload):
                skill_id = traject_skill_id(tool)
                if not skill_id:
                    continue
                all_skill_ids.add(skill_id)
                if file_domain and file_domain != "all_tools":
                    by_domain[file_domain].add(skill_id)
                tool_domain = str(tool.get("domain name") or tool.get("domain_name") or "").strip()
                if tool_domain:
                    by_domain[tool_domain].add(skill_id)
    for _traj_type, domain, query_path in iter_traject_query_files(public_data):
        try:
            payload = read_json(query_path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            for tool in item.get("tool list") or item.get("tool_list") or []:
                if not isinstance(tool, dict):
                    continue
                skill_id = traject_skill_id(tool)
                if not skill_id:
                    continue
                all_skill_ids.add(skill_id)
                by_domain[domain].add(skill_id)
    if all_skill_ids:
        by_domain["__all__"] = all_skill_ids
    return {key: sorted(values) for key, values in by_domain.items()}


def build_traject_bench_trajectories(
    fout,
    public_data: Path | None,
    counts: Counter,
    used_skill_ids: set[str],
) -> dict[str, Any]:
    if public_data is None:
        return {"available": False, "total_rows": 0, "query_count": 0}
    row_count = 0
    query_count = 0
    visible_inventory_by_domain = _traject_visible_inventory_by_domain(public_data)
    for traj_type, domain, path in iter_traject_query_files(public_data):
        payload = read_json(path)
        if not isinstance(payload, list):
            continue
        for query_idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            tools = item.get("tool list") or item.get("tool_list") or []
            if not isinstance(tools, list) or not tools:
                continue
            valid_tools = [tool for tool in tools if isinstance(tool, dict)]
            if not valid_tools:
                continue
            query_count += 1
            query = str(item.get("query") or "")
            task_name = str(item.get("task_name") or item.get("task name") or "")
            task_description = str(item.get("task_description") or item.get("task description") or "")
            trajectory_id = f"traject::{traj_type}::{domain}::{path.stem}::{query_idx}"
            split = assign_traject_split(trajectory_id)
            if split != "train":
                counts[f"traject_split_skipped:{split}"] += len(valid_tools)
                continue
            trajectory_skill_ids = [traject_skill_id(tool) for tool in valid_tools]
            tool_inventory_skill_ids = _unique_preserve_order(trajectory_skill_ids)
            visible_inventory_skill_ids = visible_inventory_by_domain.get("__all__") or []
            if not visible_inventory_skill_ids:
                counts["traject_visible_inventory_missing"] += len(valid_tools)
            history: list[str] = []
            for step_idx, tool in enumerate(valid_tools):
                sid = trajectory_skill_ids[step_idx]
                next_sid = ""
                if step_idx + 1 < len(valid_tools):
                    next_sid = trajectory_skill_ids[step_idx + 1]
                used_skill_ids.add(sid)
                if next_sid:
                    used_skill_ids.add(next_sid)
                state_text = _traject_state_text(
                    query=query,
                    traj_type=traj_type,
                    domain=domain,
                    task_name=task_name,
                    task_description=task_description,
                    history=history,
                )
                action_text = _action_text_for_tool(tool)
                normalized = {
                    "benchmark": "traject_bench",
                    "task_id": f"{trajectory_id}::{step_idx}",
                    "trajectory_id": trajectory_id,
                    "step_index": step_idx,
                    "goal_text": query,
                    "task_text": task_name,
                    "state_text": state_text,
                    "history_text": " -> ".join(history),
                    "action_text": action_text,
                    "expert_action": action_text,
                    "admissible_actions": [],
                    "next_action_text": _action_text_for_tool(valid_tools[step_idx + 1]) if next_sid else "",
                    "next_observation_text": str(tool.get("executed_output") or ""),
                    "observation_source": "actual_tool_result",
                    "skill_id": sid,
                    "next_skill_id": next_sid,
                    "tool_inventory_skill_ids": tool_inventory_skill_ids,
                    "visible_inventory_skill_ids": visible_inventory_skill_ids,
                    "equivalent_next_skill_ids": [],
                    "done": step_idx + 1 >= len(valid_tools),
                    "reward": item.get("reward"),
                    "loss_mask": {
                        "L_policy": True,
                        "L_trans": True,
                        "L_trans_skill_ce": bool(next_sid),
                        "belief": False,
                        "STOP": True,
                        "routing": True,
                    },
                    "source_quality": "traject_public_data",
                    "candidate_source": "traject_tool_list",
                    "on_policy_rollout": False,
                    "m_t_source": "serialized_tool_history",
                    "provenance": {
                        "source_id": "traject_bench",
                        "source_dataset": "TRAJECT-Bench public_data",
                        "split": split,
                        "split_policy": "deterministic_hash_trajectory_level_v1",
                        "path": str(path),
                        "trajectory_type": traj_type,
                        "domain": domain,
                    },
                    "_unified_source": "traject_bench",
                }
                _write_trajectory_row(fout, normalized)
                counts["benchmark:traject_bench"] += 1
                counts[f"traject_type:{traj_type}"] += 1
                counts[f"traject_domain:{domain}"] += 1
                row_count += 1
                history.append(str(tool.get("tool name") or tool.get("tool_name") or sid))
    return {"available": True, "total_rows": row_count, "query_count": query_count}


def find_webshop_expert_data(repo_root: str | Path | None = None) -> Path | None:
    root = _repo_root(repo_root)
    candidates = [
        root / "data" / "webshop_expert_trajectories" / WEBSHOP_HF_GPT4_FILENAME,
        root / ".tmp" / "hf_probe" / WEBSHOP_HF_GPT4_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _webshop_reward(row: dict[str, Any]) -> float:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    try:
        return float(info.get("reward", row.get("reward", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _webshop_action_type(content: str) -> str:
    match = re.search(r"Action:([A-Za-z_]+)", str(content))
    if not match:
        return ""
    return match.group(1).strip().lower()


def _webshop_skill_id(action_type: str) -> str:
    record = WEBSHOP_ACTION_SKILLS.get(action_type)
    return str(record["skill_id"]) if record else ""


def _webshop_next_user_message(conversations: list[dict[str, Any]], start_idx: int) -> str:
    for message in conversations[start_idx + 1:]:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _webshop_state_text(goal: str, observation: str, history: list[str]) -> str:
    previous_actions = "\n".join(history[-8:]) if history else "<empty>"
    return "\n".join(
        [
            f"goal: {goal}",
            "benchmark: WebShop",
            f"observation: {observation}",
            f"previous_actions: {previous_actions}",
        ]
    )


def _webshop_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    conversations = row.get("conversations") or []
    if not isinstance(conversations, list):
        return []
    goal = str(row.get("question") or "").strip()
    steps: list[dict[str, Any]] = []
    last_user_message = ""
    for idx, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user":
            last_user_message = content
            if not goal and "Task:" in content:
                goal = content.split("Task:", 1)[-1].strip()
            continue
        if role != "assistant":
            continue
        action_type = _webshop_action_type(content)
        skill_id = _webshop_skill_id(action_type)
        if not skill_id:
            continue
        steps.append(
            {
                "action_type": action_type,
                "skill_id": skill_id,
                "observation": last_user_message,
                "action_text": content.strip(),
                "next_observation": _webshop_next_user_message(conversations, idx),
            }
        )
    if not goal:
        goal = "webshop task"
    for step in steps:
        step["goal"] = goal
    return steps


def build_webshop_expert_trajectories(
    fout,
    repo_root: str | Path | None,
    counts: Counter,
    used_skill_ids: set[str],
    min_reward: float,
) -> dict[str, Any]:
    data_path = find_webshop_expert_data(repo_root)
    if data_path is None:
        return {
            "available": False,
            "total_rows": 0,
            "trajectory_count": 0,
            "retained_trajectory_count": 0,
            "filtered_low_reward_count": 0,
            "min_reward": min_reward,
        }
    payload = read_json(data_path)
    if not isinstance(payload, list):
        return {
            "available": False,
            "total_rows": 0,
            "trajectory_count": 0,
            "retained_trajectory_count": 0,
            "filtered_low_reward_count": 0,
            "min_reward": min_reward,
            "error": "expected_json_list",
            "path": str(data_path),
        }
    row_count = 0
    retained_trajectories = 0
    low_reward_count = 0
    malformed_count = 0
    inventory_skill_ids = [record["skill_id"] for record in WEBSHOP_ACTION_SKILLS.values()]
    for item in payload:
        if not isinstance(item, dict):
            malformed_count += 1
            continue
        reward = _webshop_reward(item)
        if reward < min_reward:
            low_reward_count += 1
            continue
        steps = _webshop_steps(item)
        if not steps:
            malformed_count += 1
            continue
        retained_trajectories += 1
        raw_id = str(item.get("id") or retained_trajectories - 1)
        trajectory_id = f"webshop_hf_gpt4::{raw_id}"
        history: list[str] = []
        for step_idx, step in enumerate(steps):
            next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None
            sid = str(step["skill_id"])
            next_sid = str(next_step["skill_id"]) if next_step is not None else ""
            used_skill_ids.add(sid)
            if next_sid:
                used_skill_ids.add(next_sid)
            normalized = {
                "benchmark": "webshop",
                "task_id": f"{trajectory_id}::{step_idx}",
                "trajectory_id": trajectory_id,
                "step_index": step_idx,
                "goal_text": step["goal"],
                "task_text": step["goal"],
                "state_text": _webshop_state_text(step["goal"], str(step["observation"]), history),
                "history_text": "\n".join(history),
                "action_text": step["action_text"],
                "expert_action": step["action_text"],
                "admissible_actions": [],
                "next_action_text": str(next_step["action_text"]) if next_step is not None else "",
                "next_observation_text": str(step["next_observation"]),
                "observation_source": "recorded_environment_observation",
                "skill_id": sid,
                "next_skill_id": next_sid,
                "tool_inventory_skill_ids": inventory_skill_ids,
                "equivalent_next_skill_ids": [],
                "done": step_idx + 1 >= len(steps) or step["action_type"] == "finish",
                "reward": reward,
                "loss_mask": {
                    "L_policy": True,
                    "L_trans": True,
                    "L_trans_skill_ce": bool(next_sid),
                    "belief": False,
                    "STOP": True,
                    "routing": True,
                },
                "source_quality": f"webshop_hf_gpt4_reward_ge_{min_reward:g}",
                "candidate_source": "webshop_hf_conversation_actions",
                "on_policy_rollout": False,
                "m_t_source": "serialized_webshop_action_history",
                "provenance": {
                    "source_id": "webshop_expert_trajectories",
                    "source_dataset": "lclan/webshop_expert_trajectories",
                    "split": "train",
                    "reward": reward,
                    "min_reward": min_reward,
                    "raw_id": raw_id,
                    "path": str(data_path),
                },
                "_unified_source": "webshop_expert_trajectories",
            }
            _write_trajectory_row(fout, normalized)
            counts["benchmark:webshop"] += 1
            counts[f"source_quality:{normalized['source_quality']}"] += 1
            row_count += 1
            history.append(f"{step['action_type']}: {step['action_text']}")
    return {
        "available": True,
        "total_rows": row_count,
        "trajectory_count": len(payload),
        "retained_trajectory_count": retained_trajectories,
        "filtered_low_reward_count": low_reward_count,
        "malformed_trajectory_count": malformed_count,
        "min_reward": min_reward,
        "path": str(data_path),
    }


def build_clean_alfworld_additional_trajectories(
    fout,
    repo_root: str | Path | None,
    counts: Counter,
    used_skill_ids: set[str],
    written_step_keys: set[tuple[str, str, int]],
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    sources = [
        {
            "source_id": "clstr_dagger_expert_corrected_train_enriched",
            "path": root / "data" / "clstr_dagger_expert_corrected_train_enriched" / "train.jsonl",
        },
        {
            "source_id": "clstr_qwen3_structured_train",
            "path": root / "data" / "clstr_qwen3_structured_train" / "train.jsonl",
        },
    ]
    total_rows = 0
    skipped_overlap_count = 0
    skipped_non_alfworld_count = 0
    source_stats: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source["source_id"])
        path = Path(source["path"])
        stats = {
            "available": path.exists(),
            "path": str(path),
            "added_rows": 0,
            "skipped_overlap_count": 0,
            "skipped_non_alfworld_count": 0,
        }
        if not path.exists():
            source_stats[source_id] = stats
            continue
        for row in read_jsonl(path):
            if str(row.get("benchmark") or "") != "alfworld":
                skipped_non_alfworld_count += 1
                stats["skipped_non_alfworld_count"] += 1
                continue
            key = _trajectory_step_key(row)
            if key in written_step_keys:
                skipped_overlap_count += 1
                stats["skipped_overlap_count"] += 1
                continue
            normalized = normalize_trajectory_row(row, default_source=source_id)
            normalized["_unified_source"] = source_id
            provenance = _provenance_dict(normalized)
            provenance.setdefault("source_id", source_id)
            provenance.setdefault("source_dataset", source_id)
            provenance["non_overlap_filter"] = {
                "method": "benchmark_trajectory_id_step_index_v1",
                "reason": "avoid_conflicting_labels_with_existing_clean_rows",
            }
            normalized["provenance"] = provenance
            for skill_key in ("skill_id", "next_skill_id"):
                skill_id = str(normalized.get(skill_key) or "")
                if skill_id:
                    used_skill_ids.add(skill_id)
            _write_trajectory_row(fout, normalized)
            written_step_keys.add(key)
            counts[f"benchmark:{normalized['benchmark']}"] += 1
            counts[f"source_quality:{normalized['source_quality']}"] += 1
            total_rows += 1
            stats["added_rows"] += 1
        source_stats[source_id] = stats
    return {
        "enabled": True,
        "total_rows": total_rows,
        "skipped_overlap_count": skipped_overlap_count,
        "skipped_non_alfworld_count": skipped_non_alfworld_count,
        "sources": source_stats,
    }


def find_hf_alfworld_admissible_success_files(repo_root: str | Path | None = None) -> list[Path]:
    root = _repo_root(repo_root)
    candidates = [
        root / "data" / "hf_alfworld_admissible_success",
        root / ".tmp" / "hf_alfworld_admissible_success",
    ]
    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".parquet":
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(sorted(candidate.rglob("*.parquet")))
    return files


def _hf_alfworld_message_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    last_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _message_role(message)
        content = _message_content(message)
        if role in {"user", "human"}:
            last_user = content
            continue
        if role not in {"assistant", "gpt"}:
            continue
        action = _extract_alfworld_action(content)
        if not action or action.upper() == "OK":
            continue
        steps.append(
            {
                "observation": last_user,
                "action_text": action,
                "admissible_actions": _extract_admissible_actions(last_user),
                "skill_id": _alfworld_skill_id_for_action(action),
            }
        )
    return steps


def _hf_alfworld_state_text(goal: str, observation: str, history: list[str]) -> str:
    return "\n".join(
        [
            f"goal: {goal}",
            "task_type: alfworld_hf_admissible_success",
            f"observation: {observation}",
            f"history: {' | '.join(history[-8:]) if history else '<empty>'}",
        ]
    )


def build_hf_alfworld_admissible_success_trajectories(
    fout,
    repo_root: str | Path | None,
    counts: Counter,
    used_skill_ids: set[str],
    written_step_keys: set[tuple[str, str, int]],
) -> dict[str, Any]:
    paths = find_hf_alfworld_admissible_success_files(repo_root)
    stats = {
        "enabled": True,
        "available": bool(paths),
        "paths": [str(path) for path in paths],
        "total_rows": 0,
        "retained_trajectory_count": 0,
        "filtered_non_success_count": 0,
        "filtered_missing_admissible_count": 0,
        "filtered_malformed_count": 0,
    }
    if not paths:
        return stats

    try:
        import pandas as pd
    except ImportError:
        stats["error"] = "missing_pandas_or_parquet_backend"
        return stats

    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            stats.setdefault("read_errors", []).append({"path": str(path), "error": repr(exc)})
            continue
        for raw_idx, row in enumerate(frame.to_dict(orient="records")):
            metadata = _parse_maybe_json(row.get("metadata"))
            if not isinstance(metadata, dict):
                metadata = {}
            outcome = str(metadata.get("trajectory_outcome") or row.get("trajectory_outcome") or "").lower()
            if outcome and outcome != "success":
                stats["filtered_non_success_count"] += 1
                continue
            if metadata.get("has_admissible") is False:
                stats["filtered_missing_admissible_count"] += 1
                continue
            messages = _parse_maybe_json(row.get("messages"))
            if not isinstance(messages, list):
                stats["filtered_malformed_count"] += 1
                continue
            steps = _hf_alfworld_message_steps(messages)
            if not steps:
                stats["filtered_malformed_count"] += 1
                continue
            if not any(step.get("admissible_actions") for step in steps):
                stats["filtered_missing_admissible_count"] += 1
                continue
            goal = str(metadata.get("description") or "")
            if not goal:
                for message in messages:
                    if isinstance(message, dict) and _message_role(message) in {"user", "human"}:
                        goal = _extract_alfworld_task(_message_content(message))
                        if goal:
                            break
            if not goal:
                goal = "alfworld task"
            trajectory_id = f"hf_alfworld_admissible_success::{path.stem}::{raw_idx}"
            history: list[str] = []
            written_any = False
            for step_idx, step in enumerate(steps):
                next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None
                sid = str(step["skill_id"])
                next_sid = str(next_step["skill_id"]) if next_step is not None else ""
                normalized = {
                    "benchmark": "alfworld",
                    "task_id": f"{trajectory_id}::{step_idx}",
                    "trajectory_id": trajectory_id,
                    "step_index": step_idx,
                    "goal_text": goal,
                    "task_text": goal,
                    "state_text": _hf_alfworld_state_text(goal, str(step["observation"]), history),
                    "history_text": "\n".join(history),
                    "action_text": step["action_text"],
                    "expert_action": step["action_text"],
                    "admissible_actions": step["admissible_actions"],
                    "next_action_text": str(next_step["action_text"]) if next_step is not None else "",
                    "next_observation_text": str(next_step["observation"]) if next_step is not None else "",
                    "observation_source": "recorded_environment_observation",
                    "skill_id": sid,
                    "next_skill_id": next_sid,
                    "done": step_idx + 1 >= len(steps),
                    "reward": 1.0,
                    "loss_mask": {
                        "L_policy": bool(step["admissible_actions"]),
                        "L_trans": True,
                        "L_trans_skill_ce": bool(next_sid),
                        "belief": True,
                        "STOP": True,
                        "routing": True,
                    },
                    "source_quality": "hf_alfworld_admissible_success",
                    "candidate_source": "hf_alfworld_admissible_actions",
                    "on_policy_rollout": False,
                    "m_t_source": "hf_alfworld_success_history",
                    "provenance": {
                        "source_id": "hf_alfworld_admissible_success",
                        "source_dataset": HF_ALFWORLD_ADMISSIBLE_SUCCESS_DATASET,
                        "split": "train",
                        "dataset_version": metadata.get("dataset_version") or row.get("dataset_version"),
                        "task_type": metadata.get("task_type"),
                        "trajectory_outcome": outcome or "success",
                        "has_admissible": metadata.get("has_admissible", True),
                        "path": str(path),
                        "raw_index": raw_idx,
                    },
                    "_unified_source": "hf_alfworld_admissible_success",
                }
                key = _trajectory_step_key(normalized)
                if key in written_step_keys:
                    continue
                _write_trajectory_row(fout, normalized)
                written_step_keys.add(key)
                used_skill_ids.add(sid)
                if next_sid:
                    used_skill_ids.add(next_sid)
                counts["benchmark:alfworld"] += 1
                counts["source_quality:hf_alfworld_admissible_success"] += 1
                stats["total_rows"] += 1
                written_any = True
                history.append(step["action_text"])
            if written_any:
                stats["retained_trajectory_count"] += 1
    return stats


def _weak_policy_split(row: dict[str, Any]) -> str:
    provenance = _provenance_dict(row)
    return str(row.get("split") or provenance.get("split") or "").strip().lower()


def _progressive_weak_policy_candidate(
    row: dict[str, Any],
    *,
    default_source: str = "agentgym_agenttraj_l",
) -> tuple[dict[str, Any] | None, str, int]:
    split = _weak_policy_split(row)
    if split and split != "train":
        return None, f"split:{split}", 99
    if not str(row.get("state_text") or "").strip():
        return None, "empty_state_text", 99
    if not str(row.get("action_text") or row.get("expert_action") or "").strip():
        return None, "empty_action_text", 99
    skill_id = str(row.get("skill_id") or "").strip()
    if not skill_id:
        return None, "missing_skill_id", 99

    loss_mask = _loss_mask_dict(row)
    admissible_actions = _normalized_string_list(row.get("admissible_actions"))
    has_next_observation = bool(str(row.get("next_observation_text") or "").strip())
    policy_priority = bool(loss_mask.get("L_policy")) and bool(admissible_actions)
    belief_fill = has_next_observation
    if not policy_priority and not belief_fill:
        return None, "no_policy_or_belief_signal", 99

    normalized = normalize_trajectory_row(row, default_source=default_source)
    normalized["benchmark"] = "alfworld"
    normalized["admissible_actions"] = admissible_actions
    normalized["expert_action"] = str(row.get("expert_action") or row.get("action_text") or "")
    normalized["next_skill_id"] = ""
    normalized["loss_mask"] = {
        "L_policy": policy_priority,
        "L_trans": has_next_observation,
        "L_trans_skill_ce": False,
        "belief": has_next_observation,
        "STOP": True,
        "routing": False,
    }
    normalized["source_quality"] = "weak_policy_filtered"
    normalized["candidate_source"] = "agentgym_filtered_weak_policy"
    normalized["on_policy_rollout"] = False
    normalized["m_t_source"] = "filtered_weak_policy_replay"
    provenance = _provenance_dict(row)
    provenance.setdefault("source_id", "agentgym_agenttraj_l")
    provenance.setdefault("source_dataset", "AgentGym/AgentTraj-L")
    provenance.setdefault("split", "train")
    provenance["augmentation"] = "filtered_weak_policy_v4_2_progressive_final"
    provenance["weak_policy_filter"] = {
        "policy_priority": policy_priority,
        "belief_fill": belief_fill,
        "transition_skill_ce_disabled": True,
    }
    normalized["provenance"] = provenance
    priority = 0 if policy_priority else 1
    return normalized, "retained_candidate", priority


def build_progressive_filtered_weak_policy_trajectories(
    fout,
    root: Path,
    counts: Counter,
    used_skill_ids: set[str],
    written_step_keys: set[tuple[str, str, int]],
    excluded_appworld_ids: set[str],
    *,
    cap: int,
) -> dict[str, Any]:
    ag_path = root / "data" / "agentgym_agenttraj_l" / "alfworld" / "converted_train.jsonl"
    stats: dict[str, Any] = {
        "enabled": True,
        "available": ag_path.exists(),
        "cap": int(cap),
        "source_rows": 0,
        "candidate_rows": 0,
        "retained_rows": 0,
        "skipped_rows": 0,
        "skipped_by_reason": {},
    }
    if int(cap) <= 0:
        stats["available"] = ag_path.exists()
        return stats
    if not ag_path.exists():
        return stats

    candidates: list[tuple[int, int, dict[str, Any]]] = []
    skipped_by_reason: Counter[str] = Counter()
    for raw_idx, row in enumerate(read_jsonl(ag_path)):
        stats["source_rows"] += 1
        tid = str(row.get("task_id", ""))
        if tid in excluded_appworld_ids:
            skipped_by_reason["appworld_leakage_filtered"] += 1
            continue
        normalized, reason, priority = _progressive_weak_policy_candidate(row)
        if normalized is None:
            skipped_by_reason[reason] += 1
            continue
        key = _trajectory_step_key(normalized)
        if key in written_step_keys:
            skipped_by_reason["duplicate_step_key"] += 1
            continue
        candidates.append((priority, raw_idx, normalized))

    candidates.sort(key=lambda item: (item[0], item[1]))
    retained_limit = min(max(0, int(cap)), len(candidates))
    retained = candidates[:retained_limit]
    skipped_by_reason["cap_exceeded"] += max(0, len(candidates) - retained_limit)
    for _priority, _raw_idx, normalized in retained:
        _write_trajectory_row(fout, normalized)
        written_step_keys.add(_trajectory_step_key(normalized))
        skill_id = str(normalized.get("skill_id") or "")
        if skill_id:
            used_skill_ids.add(skill_id)
        counts["benchmark:alfworld"] += 1
        counts["source_quality:weak_policy_filtered"] += 1

    stats["candidate_rows"] = len(candidates)
    stats["retained_rows"] = len(retained)
    stats["skipped_rows"] = max(0, int(stats["source_rows"]) - len(retained))
    stats["skipped_by_reason"] = dict(sorted((key, value) for key, value in skipped_by_reason.items() if value))
    return stats


def build_trajectories(
    out_path: Path,
    excluded_appworld_ids: set[str],
    repo_root: str | Path | None = None,
    dataset_recipe: str = "legacy",
    webshop_min_reward: float = 0.8,
    progressive_filtered_weak_policy_cap: int = DEFAULT_PROGRESSIVE_FILTERED_WEAK_POLICY_CAP,
) -> dict:
    """Stream A: merge ALFWorld + ScienceWorld + AgentGym trajectory rows.
    AppWorld trajectories are NOT included from this builder because the
    `appworld_act/verified_pairs_train.jsonl` uses a different (POMDP belief
    pair) schema; we expose that as Stream A2 separately if needed."""
    root = _repo_root(repo_root)
    traject_public_data = find_traject_public_data(root)
    counts = Counter()
    filtered_counts = Counter()
    leakage_filtered = 0
    used_skill_ids: set[str] = set()
    written_step_keys: set[tuple[str, str, int]] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        # full_base_train: ALFWorld + ScienceWorld
        for row in read_jsonl(root / "data" / "clstr_full_base_train" / "train.jsonl"):
            tid = str(row.get("task_id", ""))
            if tid in excluded_appworld_ids:
                leakage_filtered += 1
                continue
            skip_reason = _trajectory_skip_reason(row, dataset_recipe)
            if skip_reason:
                filtered_counts[skip_reason] += 1
                continue
            normalized = normalize_trajectory_row(row, default_source="clstr_full_base_train")
            for key in ("skill_id", "next_skill_id"):
                skill_id = str(normalized.get(key) or "")
                if skill_id:
                    used_skill_ids.add(skill_id)
            _write_trajectory_row(fout, normalized)
            written_step_keys.add(_trajectory_step_key(normalized))
            counts[f"benchmark:{normalized['benchmark']}"] += 1
            counts[f"source_quality:{normalized['source_quality']}"] += 1

        clean_alfworld_stats = (
            build_clean_alfworld_additional_trajectories(
                fout,
                root,
                counts,
                used_skill_ids,
                written_step_keys,
            )
            if dataset_recipe in {"v4_1b", V4_2_ALFWORLD_HF_RECIPE, V4_2_PROGRESSIVE_FINAL_RECIPE}
            else {"enabled": False, "total_rows": 0}
        )

        hf_alfworld_stats = (
            build_hf_alfworld_admissible_success_trajectories(
                fout,
                root,
                counts,
                used_skill_ids,
                written_step_keys,
            )
            if dataset_recipe in {V4_2_ALFWORLD_HF_RECIPE, V4_2_PROGRESSIVE_FINAL_RECIPE}
            else {"enabled": False, "available": False, "total_rows": 0}
        )

        # AgentGym AgentTraj-L ALFWorld converted train (weak policy supervision)
        ag_path = root / "data" / "agentgym_agenttraj_l" / "alfworld" / "converted_train.jsonl"
        if dataset_recipe == V4_2_PROGRESSIVE_FINAL_RECIPE:
            filtered_weak_policy_stats = build_progressive_filtered_weak_policy_trajectories(
                fout,
                root,
                counts,
                used_skill_ids,
                written_step_keys,
                excluded_appworld_ids,
                cap=progressive_filtered_weak_policy_cap,
            )
        else:
            filtered_weak_policy_stats = {"enabled": False, "available": ag_path.exists(), "retained_rows": 0}
            for row in read_jsonl(ag_path):
                tid = str(row.get("task_id", ""))
                if tid in excluded_appworld_ids:
                    leakage_filtered += 1
                    continue
                normalized = normalize_trajectory_row(row, default_source="agentgym_agenttraj_l")
                for key in ("skill_id", "next_skill_id"):
                    skill_id = str(normalized.get(key) or "")
                    if skill_id:
                        used_skill_ids.add(skill_id)
                _write_trajectory_row(fout, normalized)
                written_step_keys.add(_trajectory_step_key(normalized))
                counts["benchmark:alfworld_agentgym"] += 1
                counts[f"source_quality:{normalized['source_quality']}"] += 1

        webshop_stats = (
            build_webshop_expert_trajectories(
                fout,
                root,
                counts,
                used_skill_ids,
                min_reward=webshop_min_reward,
            )
            if dataset_recipe in V4_CLEAN_RECIPES
            else {"available": False, "enabled": False, "total_rows": 0}
        )

        traject_stats = build_traject_bench_trajectories(
            fout,
            traject_public_data,
            counts,
            used_skill_ids,
        )

        toolbench_path = root / "data" / "toolbench_g3" / "trajectories.jsonl"
        toolbench_rows = 0
        for row in read_jsonl(toolbench_path):
            tid = str(row.get("task_id", ""))
            if tid in excluded_appworld_ids:
                leakage_filtered += 1
                continue
            normalized = normalize_trajectory_row(row, default_source="toolbench_g3")
            for key in ("skill_id", "next_skill_id"):
                skill_id = str(normalized.get(key) or "")
                if skill_id:
                    used_skill_ids.add(skill_id)
            _write_trajectory_row(fout, normalized)
            counts[f"benchmark:{normalized['benchmark']}"] += 1
            counts[f"source_quality:{normalized['source_quality']}"] += 1
            toolbench_rows += 1

    return {
        "total_rows": sum(v for k, v in counts.items() if k.startswith("benchmark:")),
        "leakage_filtered": leakage_filtered,
        "used_skill_count": len(used_skill_ids),
        "used_skill_ids": sorted(used_skill_ids),
        "quality_filters": {
            "dataset_recipe": dataset_recipe,
            "filtered_rows": sum(filtered_counts.values()),
            "filtered_rows_by_reason": dict(sorted(filtered_counts.items())),
        },
        "clean_alfworld": clean_alfworld_stats,
        "hf_alfworld_admissible_success": hf_alfworld_stats,
        "filtered_weak_policy": filtered_weak_policy_stats,
        "webshop_expert_trajectories": webshop_stats,
        "traject_bench": traject_stats,
        "toolbench_g3": {
            "available": toolbench_path.exists(),
            "total_rows": toolbench_rows,
        },
        "counts": dict(counts),
    }


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def build_traject_bench_retrieval_pairs(
    fout,
    public_data: Path | None,
    counts: Counter,
    used_skill_ids: set[str],
) -> dict[str, Any]:
    if public_data is None:
        return {"available": False, "total_pairs": 0, "query_count": 0}
    pair_count = 0
    query_count = 0
    for traj_type, domain, path in iter_traject_query_files(public_data):
        payload = read_json(path)
        if not isinstance(payload, list):
            continue
        for query_idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            tools = item.get("tool list") or item.get("tool_list") or []
            if not isinstance(tools, list) or not tools:
                continue
            valid_tools = [tool for tool in tools if isinstance(tool, dict)]
            if not valid_tools:
                continue
            query_count += 1
            query = str(item.get("query") or "")
            task_name = str(item.get("task_name") or item.get("task name") or "")
            task_description = str(item.get("task_description") or item.get("task description") or "")
            trajectory_id = f"traject::{traj_type}::{domain}::{path.stem}::{query_idx}"
            split = assign_traject_split(trajectory_id)
            if split != "train":
                counts[f"traject_bench_pair_skipped:{split}"] += len(valid_tools)
                continue
            trajectory_skill_ids = [traject_skill_id(tool) for tool in valid_tools]
            history: list[str] = []
            for step_idx, tool in enumerate(valid_tools):
                sid = trajectory_skill_ids[step_idx]
                negative_skill_ids = _unique_preserve_order(
                    other_sid
                    for other_idx, other_sid in enumerate(trajectory_skill_ids)
                    if other_idx != step_idx and other_sid != sid
                )
                used_skill_ids.add(sid)
                used_skill_ids.update(negative_skill_ids)
                state_text = _traject_state_text(
                    query=query,
                    traj_type=traj_type,
                    domain=domain,
                    task_name=task_name,
                    task_description=task_description,
                    history=history,
                )
                record = {
                    "source": "traject_bench",
                    "query_id": f"{trajectory_id}::{step_idx}",
                    "query_text": router_state_text({"state_text": state_text}),
                    "positive_skill_id": sid,
                    "negative_skill_ids": negative_skill_ids,
                    "provenance": json.dumps(
                        {
                            "source_dataset": "TRAJECT-Bench public_data",
                            "split": split,
                            "split_policy": "deterministic_hash_trajectory_level_v1",
                            "path": str(path),
                            "trajectory_type": traj_type,
                            "domain": domain,
                            "trajectory_id": trajectory_id,
                            "step_index": step_idx,
                        },
                        ensure_ascii=False,
                    ),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["traject_bench_pair"] += 1
                pair_count += 1
                history.append(str(tool.get("tool name") or tool.get("tool_name") or sid))
    return {"available": True, "total_pairs": pair_count, "query_count": query_count}


def _trajectory_row_key(row: dict[str, Any], fallback_idx: int) -> str:
    return str(row.get("trajectory_id") or row.get("task_id") or f"row-{fallback_idx}")


def _trajectory_query_text(row: dict[str, Any], target: str) -> str:
    goal = str(row.get("goal_text") or row.get("task_text") or "").strip()
    task = str(row.get("task_text") or "").strip()
    state = router_state_text(row)
    action = str(row.get("action_text") or row.get("expert_action") or "").strip()
    next_observation = str(row.get("next_observation_text") or "").strip()
    lowered_state = state.lower()
    parts: list[str] = []
    if goal and "goal:" not in lowered_state:
        parts.append(f"goal: {goal}")
    if task and task != goal and not any(
        marker in lowered_state for marker in ("task:", "task_type:", "task_description:")
    ):
        parts.append(f"task: {task}")
    if state:
        parts.append(
            state
            if ":" in state.splitlines()[0]
            else f"observation: {state}"
        )
    if target == "current":
        pass
    elif target == "next":
        if action:
            parts.append(f"previous_action: {action}")
        if next_observation:
            parts.append(f"next_observation: {next_observation}")
    else:
        raise ValueError(f"unsupported trajectory retrieval target: {target}")
    return "\n".join(parts)


def _trajectory_negative_skill_ids(row: dict[str, Any], positive: str) -> list[str]:
    candidates = [str(row.get("skill_id") or ""), str(row.get("next_skill_id") or "")]
    return _unique_preserve_order(item for item in candidates if item and item != positive)


def build_trajectory_derived_retrieval_pairs(
    fout,
    trajectories_path: Path,
    counts: Counter,
    used_skill_ids: set[str],
    row_caps: dict[str, int],
) -> dict[str, Any]:
    if not trajectories_path.exists() or not row_caps:
        return {
            "enabled": False,
            "total_pairs": 0,
            "row_caps": dict(sorted(row_caps.items())),
            "retained_rows_by_benchmark": {},
            "skipped_rows_by_benchmark": {},
        }
    retained_by_benchmark: Counter[str] = Counter()
    skipped_by_benchmark: Counter[str] = Counter()
    pair_count = 0
    for row_idx, row in enumerate(read_jsonl(trajectories_path)):
        benchmark = str(row.get("benchmark") or "")
        if benchmark not in row_caps:
            continue
        if str(row.get("source_quality") or "") == "weak_policy_filtered":
            skipped_by_benchmark[benchmark] += 1
            continue
        cap = int(row_caps[benchmark])
        if cap >= 0 and retained_by_benchmark[benchmark] >= cap:
            skipped_by_benchmark[benchmark] += 1
            continue
        retained_by_benchmark[benchmark] += 1
        row_key = _trajectory_row_key(row, row_idx)
        step_index = int(row.get("step_index", row_idx))
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        for target, skill_id in (("current", current_skill_id), ("next", next_skill_id)):
            if not skill_id:
                continue
            query_text = _trajectory_query_text(row, target=target)
            if not query_text.strip():
                continue
            negative_skill_ids = _trajectory_negative_skill_ids(row, skill_id)
            used_skill_ids.add(skill_id)
            used_skill_ids.update(negative_skill_ids)
            record = {
                "source": f"trajectory_derived_{benchmark}",
                "query_id": f"trajectory-derived::{benchmark}::{row_key}::{step_index}::{target}",
                "query_text": query_text,
                "positive_skill_id": skill_id,
                "negative_skill_ids": negative_skill_ids,
                "provenance": json.dumps(
                    {
                        "source_dataset": "clstr_unified_trajectory_stream",
                        "source": "trajectory_derived",
                        "benchmark": benchmark,
                        "split": "train",
                        "trajectory_id": row.get("trajectory_id"),
                        "task_id": row.get("task_id"),
                        "step_index": step_index,
                        "target": target,
                    },
                    ensure_ascii=False,
                ),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[f"trajectory_derived_{benchmark}_{target}_pair"] += 1
            pair_count += 1
    return {
        "enabled": True,
        "total_pairs": pair_count,
        "row_caps": dict(sorted(row_caps.items())),
        "retained_rows_by_benchmark": dict(sorted(retained_by_benchmark.items())),
        "skipped_rows_by_benchmark": dict(sorted(skipped_by_benchmark.items())),
    }


def _normalize_path_list(value: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(item) for item in parse_string_list(str(value))]
    return [Path(item) for item in value]


def _extra_retrieval_skip_reason(
    row: dict[str, Any],
    *,
    excluded_skillret_qids: set[str],
    excluded_appworld_ids: set[str],
) -> str | None:
    provenance = _provenance_dict(row)
    qid = str(row.get("query_id") or row.get("id") or "")
    task_id = str(row.get("task_id") or provenance.get("task_id") or "")
    if qid in excluded_skillret_qids:
        return "skillret_test_query_id"
    if qid in excluded_appworld_ids or task_id in excluded_appworld_ids:
        return "appworld_dev_test_task_id"
    return None


def append_extra_retrieval_pairs(
    fout,
    extra_retrieval_paths: str | Path | Iterable[str | Path] | None,
    counts: Counter,
    used_skill_ids: set[str],
    *,
    repo_root: str | Path | None,
    excluded_skillret_qids: set[str],
    excluded_appworld_ids: set[str],
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    paths = _normalize_path_list(extra_retrieval_paths)
    if not paths:
        return {"enabled": False, "total_pairs": 0, "leakage_filtered": 0, "paths": []}
    total_pairs = 0
    leakage_filtered = 0
    skipped_missing_field = 0
    missing_files: list[str] = []
    read_paths: list[str] = []
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        read_paths.append(str(path))
        if not path.exists():
            missing_files.append(str(path))
            continue
        for row in read_jsonl(path):
            if _extra_retrieval_skip_reason(
                row,
                excluded_skillret_qids=excluded_skillret_qids,
                excluded_appworld_ids=excluded_appworld_ids,
            ):
                leakage_filtered += 1
                continue
            qid = str(row.get("query_id") or row.get("id") or "")
            sid = row.get("positive_skill_id") or row.get("skill_id")
            if not qid or not sid:
                skipped_missing_field += 1
                continue
            sid = str(sid)
            negative_skill_ids = [str(item) for item in row.get("negative_skill_ids", []) if item]
            source = str(row.get("source") or "extra_retrieval")
            provenance = row.get("provenance") or {
                "source_dataset": "extra_retrieval",
                "path": str(path),
            }
            record = {
                "source": source,
                "query_id": qid,
                "query_text": router_state_text(
                    {"state_text": str(row.get("query_text") or row.get("query") or "")}
                ),
                "positive_skill_id": sid,
                "negative_skill_ids": negative_skill_ids,
                "provenance": provenance if isinstance(provenance, str) else json.dumps(provenance, ensure_ascii=False),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            used_skill_ids.add(sid)
            used_skill_ids.update(negative_skill_ids)
            counts[f"extra_retrieval_pair:{source}"] += 1
            total_pairs += 1
    return {
        "enabled": True,
        "total_pairs": total_pairs,
        "leakage_filtered": leakage_filtered,
        "skipped_missing_field": skipped_missing_field,
        "paths": read_paths,
        "missing_files": missing_files,
    }


def build_retrieval_pairs(
    out_path: Path,
    excluded_skillret_qids: set[str],
    repo_root: str | Path | None = None,
    excluded_appworld_ids: set[str] | None = None,
    trajectories_path: Path | None = None,
    trajectory_retrieval_caps: str | dict[str, int] | None = None,
    extra_retrieval_paths: str | Path | Iterable[str | Path] | None = None,
) -> dict:
    """Stream B: train query-skill pairs.

    SKILLRET test split is excluded. ToolRet-Training-20w enters only after it
    has been normalized into data/toolret_training/retrieval.jsonl.
    """
    root = _repo_root(repo_root)
    qrels_path = root / "data" / "skillret" / "qrels.jsonl"
    queries_path = root / "data" / "skillret" / "queries.jsonl"
    toolret_path = root / "data" / "toolret_training" / "retrieval.jsonl"
    toolbench_path = root / "data" / "toolbench_g3" / "retrieval.jsonl"
    traject_public_data = find_traject_public_data(root)
    caps = parse_benchmark_caps(trajectory_retrieval_caps)
    excluded_appworld_ids = excluded_appworld_ids or set()

    # Cache query_id -> query text
    queries: dict[str, str] = {}
    for row in read_jsonl(queries_path):
        qid = row.get("query_id") or row.get("id")
        if qid:
            queries[str(qid)] = row.get("query_text") or row.get("query") or row.get("text") or ""

    counts = Counter()
    pair_count = 0
    leakage_filtered = 0
    used_skill_ids: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for row in read_jsonl(qrels_path):
            qid = str(row.get("query_id", ""))
            if qid in excluded_skillret_qids:
                leakage_filtered += 1
                continue
            if int(row.get("relevance", 0)) <= 0:
                continue
            sid = row.get("skill_id")
            if not sid:
                continue
            used_skill_ids.add(str(sid))
            record = {
                "source": "skillret",
                "query_id": qid,
                "query_text": queries.get(qid, ""),
                "positive_skill_id": str(sid),
                "negative_skill_ids": [],
                "provenance": json.dumps({
                    "source_dataset": "ThakiCloud/SKILLRET",
                    "split": row.get("split", "train"),
                    "relevance": row.get("relevance", 1),
                }),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["skillret_train_pair"] += 1
            pair_count += 1

        traject_stats = build_traject_bench_retrieval_pairs(
            fout,
            traject_public_data,
            counts,
            used_skill_ids,
        )
        pair_count += int(traject_stats.get("total_pairs", 0))

        for row in read_jsonl(toolret_path):
            qid = str(row.get("query_id") or row.get("id") or "")
            query_text = row.get("query_text") or row.get("query") or row.get("prompt") or ""
            sid = row.get("positive_skill_id") or row.get("skill_id")
            if not qid or not sid:
                continue
            sid = str(sid)
            negative_skill_ids = [
                str(item)
                for item in row.get("negative_skill_ids", [])
                if item
            ]
            used_skill_ids.add(sid)
            used_skill_ids.update(negative_skill_ids)
            provenance = row.get("provenance") or {
                "source_dataset": "mangopy/ToolRet-Training-20w",
                "split": row.get("split", "train"),
            }
            record = {
                "source": "toolret_training",
                "query_id": qid,
                "query_text": str(query_text),
                "positive_skill_id": sid,
                "negative_skill_ids": negative_skill_ids,
                "provenance": provenance if isinstance(provenance, str) else json.dumps(provenance, ensure_ascii=False),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["toolret_training_pair"] += 1
            pair_count += 1

        for row in read_jsonl(toolbench_path):
            qid = str(row.get("query_id") or row.get("id") or "")
            query_text = row.get("query_text") or row.get("query") or ""
            sid = row.get("positive_skill_id") or row.get("skill_id")
            if not qid or not sid:
                continue
            sid = str(sid)
            negative_skill_ids = [
                str(item)
                for item in row.get("negative_skill_ids", [])
                if item
            ]
            used_skill_ids.add(sid)
            used_skill_ids.update(negative_skill_ids)
            provenance = row.get("provenance") or {
                "source_dataset": "ToolBench-G3",
                "split": row.get("split", "train"),
            }
            record = {
                "source": "toolbench_g3",
                "query_id": qid,
                "query_text": str(query_text),
                "positive_skill_id": sid,
                "negative_skill_ids": negative_skill_ids,
                "provenance": provenance if isinstance(provenance, str) else json.dumps(provenance, ensure_ascii=False),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["toolbench_g3_pair"] += 1
            pair_count += 1
        trajectory_stats = build_trajectory_derived_retrieval_pairs(
            fout,
            trajectories_path or (root / "data" / "clstr_unified_pretrain_v2" / "trajectories.jsonl"),
            counts,
            used_skill_ids,
            caps,
        )
        pair_count += int(trajectory_stats.get("total_pairs", 0))
        extra_stats = append_extra_retrieval_pairs(
            fout,
            extra_retrieval_paths,
            counts,
            used_skill_ids,
            repo_root=root,
            excluded_skillret_qids=excluded_skillret_qids,
            excluded_appworld_ids=excluded_appworld_ids,
        )
        pair_count += int(extra_stats.get("total_pairs", 0))
    return {
        "total_pairs": pair_count,
        "leakage_filtered": leakage_filtered,
        "used_skill_count": len(used_skill_ids),
        "used_skill_ids": sorted(used_skill_ids),
        "traject_bench": traject_stats,
        "trajectory_derived": trajectory_stats,
        "extra_retrieval": extra_stats,
        "counts": dict(counts),
    }


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def _compact_text(value: Any) -> str:
    return " ".join(str(value).split())


def _schema_parameter_names(schema: Any) -> list[str]:
    schema = _parse_maybe_json(schema)
    names: list[str] = []
    seen: set[str] = set()

    def add_name(value: Any) -> None:
        name = str(value or "").strip()
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(name)

    def add_param(item: Any) -> None:
        if isinstance(item, dict):
            add_name(item.get("name") or item.get("param_name") or item.get("key"))

    if isinstance(schema, dict):
        has_grouped_schema = any(
            key in schema
            for key in ("required_parameters", "required", "optional_parameters", "optional")
        )
        if has_grouped_schema:
            for key in ("required_parameters", "required", "optional_parameters", "optional"):
                values = schema.get(key) or []
                for item in values if isinstance(values, list) else []:
                    add_param(item)
        else:
            for name in schema:
                if name not in {"method", "http_method"}:
                    add_name(name)
    elif isinstance(schema, list):
        for item in schema:
            add_param(item)
    return names


def _skill_description_fallback(row: dict[str, Any]) -> str:
    name = _compact_text(row.get("name") or "")
    sid = _compact_text(row.get("skill_id") or row.get("id") or "")
    schema = row.get("input_schema") or row.get("args_schema") or row.get("parameters") or {}
    param_names = _schema_parameter_names(schema)
    display_name = name or sid
    if display_name and param_names:
        shown_params = param_names[:12]
        suffix = ""
        if len(param_names) > len(shown_params):
            suffix = f", and {len(param_names) - len(shown_params)} more"
        return f"{display_name} API with parameters: {', '.join(shown_params)}{suffix}."
    return display_name


def _skill_description(row: dict[str, Any]) -> str:
    for key in ("description", "executor_desc", "body"):
        value = row.get(key)
        text = _compact_text(value) if value is not None else ""
        if text:
            return text
    return _skill_description_fallback(row)


def _dedup_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _param_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("schema_type") or value.get("data_type") or "").lower()
    return str(value or "").lower()


def _param_signature(schema: Any) -> str:
    params: list[tuple[str, str, str]] = []

    def add_param(item: Any, required: str) -> None:
        if not isinstance(item, dict):
            return
        name = str(item.get("name") or item.get("param_name") or item.get("key") or "").strip().lower()
        if not name:
            return
        params.append((name, _param_type(item), required))

    if isinstance(schema, dict):
        has_grouped_schema = any(
            key in schema
            for key in ("required_parameters", "required", "optional_parameters", "optional")
        )
        if has_grouped_schema:
            required = schema.get("required_parameters") or schema.get("required") or []
            optional = schema.get("optional_parameters") or schema.get("optional") or []
            for item in required if isinstance(required, list) else []:
                add_param(item, "required")
            for item in optional if isinstance(optional, list) else []:
                add_param(item, "optional")
        else:
            for name, spec in schema.items():
                if name in {"method", "http_method"}:
                    continue
                param_type = _param_type(spec)
                params.append((str(name).strip().lower(), param_type, "unknown"))
    elif isinstance(schema, list):
        for item in schema:
            add_param(item, "unknown")
    return json.dumps(sorted(params), ensure_ascii=False, separators=(",", ":"))


def _param_names(schema: Any) -> set[str]:
    names: set[str] = set()

    def add_name(item: Any) -> None:
        if not isinstance(item, dict):
            return
        name = str(item.get("name") or item.get("param_name") or item.get("key") or "").strip().lower()
        if name:
            names.add(name)

    if isinstance(schema, dict):
        has_grouped_schema = any(
            key in schema
            for key in ("required_parameters", "required", "optional_parameters", "optional")
        )
        if has_grouped_schema:
            for key in ("required_parameters", "required", "optional_parameters", "optional"):
                values = schema.get(key) or []
                for item in values if isinstance(values, list) else []:
                    add_name(item)
        else:
            for name in schema:
                if name not in {"method", "http_method"}:
                    names.add(str(name).strip().lower())
    elif isinstance(schema, list):
        for item in schema:
            add_name(item)
    return {name for name in names if name}


def _dedup_key(row: dict[str, Any]) -> tuple[str, str, str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    if provenance.get("missing_source_record"):
        return ("missing", str(row.get("skill_id") or ""), "")
    name = _dedup_name(str(row.get("name") or row.get("skill_id") or ""))
    if not name:
        return ("id", str(row.get("skill_id") or ""), "")
    return ("name_param", name, _param_signature(row.get("input_schema") or {}))


def _canonical_priority(skill_id: str) -> tuple[int, str]:
    sid = str(skill_id)
    priorities = [
        ("toolbench-g3/", 0),
        ("traject/", 1),
        ("toolret/", 2),
        ("skillret", 3),
        ("webshop/", 4),
        ("appworld", 5),
    ]
    for prefix, priority in priorities:
        if sid.startswith(prefix):
            return priority, sid
    return 9, sid


def _merge_list_unique(base: list[Any], values: Iterable[Any]) -> list[Any]:
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in base}
    merged = list(base)
    for item in values:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _partial_dedup_skill_records(records: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    for sid, record in records.items():
        groups.setdefault(_dedup_key(record), []).append(sid)

    canonical_records: dict[str, dict[str, Any]] = {}
    alias_map: dict[str, str] = {}
    alias_rows: list[dict[str, Any]] = []
    merged_group_count = 0
    for key, raw_ids in groups.items():
        canonical_id = sorted(raw_ids, key=_canonical_priority)[0]
        aliases = sorted(raw_ids)
        if len(raw_ids) > 1:
            merged_group_count += 1
        record = deepcopy(records[canonical_id])
        record["skill_id"] = canonical_id
        record["canonical_skill_id"] = canonical_id
        record["alias_skill_ids"] = aliases

        descriptions: list[str] = []
        source_files: list[str] = list(record.get("source_files") or [])
        sources: list[str] = []
        for raw_id in aliases:
            raw = records[raw_id]
            desc_values = []
            desc = _skill_description(raw)
            if desc:
                desc_values.append(desc)
            desc_values.extend(raw.get("alternate_descriptions") or [])
            descriptions = _merge_list_unique(descriptions, [str(item) for item in desc_values if item])
            source_files = _merge_list_unique(source_files, raw.get("source_files") or [])
            source = raw.get("source")
            if source:
                sources = _merge_list_unique(sources, [str(source)])
            for extra_key in ("body", "executor_desc", "input_schema", "output_schema", "failure_modes", "environment"):
                if extra_key not in record and extra_key in raw:
                    record[extra_key] = raw[extra_key]

        primary_desc = str(record.get("description") or "")
        record["alternate_descriptions"] = [
            desc for desc in descriptions if desc and desc != primary_desc
        ]
        record["source_files"] = source_files
        if sources:
            record["source_aliases"] = sources
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        provenance["partial_dedup"] = {
            "method": "partial_rule_name_param_v2",
            "dedup_key": list(key),
            "raw_skill_ids": aliases,
            "merged_count": len(aliases),
        }
        record["provenance"] = provenance
        canonical_records[canonical_id] = record

        for raw_id in aliases:
            alias_map[raw_id] = canonical_id
            alias_rows.append(
                {
                    "raw_skill_id": raw_id,
                    "canonical_skill_id": canonical_id,
                    "dedup_method": "partial_rule_name_param_v2",
                    "dedup_key": list(key),
                }
            )

    stats = {
        "raw_skill_count": len(records),
        "canonical_skill_count": len(canonical_records),
        "merged_group_count": merged_group_count,
        "alias_count": len(alias_rows),
        "changed_alias_count": sum(1 for raw_id, canonical_id in alias_map.items() if raw_id != canonical_id),
    }
    return canonical_records, alias_map, sorted(alias_rows, key=lambda row: row["raw_skill_id"]), stats


def _name_similarity(left: str, right: str) -> float:
    left_norm = _dedup_name(left)
    right_norm = _dedup_name(right)
    if not left_norm or not right_norm:
        return 0.0
    return round(SequenceMatcher(None, left_norm, right_norm).ratio(), 4)


def _param_overlap(left_schema: Any, right_schema: Any) -> float:
    left_names = _param_names(left_schema)
    right_names = _param_names(right_schema)
    if not left_names or not right_names:
        return 0.0
    return round(len(left_names & right_names) / max(len(left_names), len(right_names)), 4)


def _candidate_bucket_tokens(record: dict[str, Any]) -> set[str]:
    name_tokens = {
        token
        for token in _dedup_name(str(record.get("name") or record.get("skill_id") or "")).split()
        if len(token) >= 4
    }
    return name_tokens | _param_names(record.get("input_schema") or {})


def _borderline_reasons(name_similarity: float, param_overlap: float) -> list[str]:
    reasons: list[str] = []
    if 0.6 <= name_similarity < 0.95:
        reasons.append("fuzzy_name_similarity")
    if 0.0 < param_overlap < 1.0:
        reasons.append("param_signature_partial_overlap")
    return reasons


def write_borderline_dedup_candidates(
    path: Path,
    records: dict[str, dict[str, Any]],
    alias_map: dict[str, str],
    max_bucket_size: int = 500,
    max_candidates: int = 20000,
) -> dict[str, Any]:
    buckets: dict[str, list[str]] = {}
    for sid, record in records.items():
        for token in _candidate_bucket_tokens(record):
            buckets.setdefault(token, []).append(sid)

    seen_pairs: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    for token in sorted(buckets):
        raw_ids = sorted(set(buckets[token]))
        if len(raw_ids) < 2 or len(raw_ids) > max_bucket_size:
            continue
        for left_idx, left_id in enumerate(raw_ids):
            for right_id in raw_ids[left_idx + 1:]:
                if alias_map.get(left_id, left_id) == alias_map.get(right_id, right_id):
                    continue
                pair_key = (left_id, right_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                left = records[left_id]
                right = records[right_id]
                similarity = _name_similarity(str(left.get("name") or left_id), str(right.get("name") or right_id))
                overlap = _param_overlap(left.get("input_schema") or {}, right.get("input_schema") or {})
                reasons = _borderline_reasons(similarity, overlap)
                if not reasons:
                    continue
                candidates.append(
                    {
                        "left_skill_id": left_id,
                        "right_skill_id": right_id,
                        "left_name": str(left.get("name") or left_id),
                        "right_name": str(right.get("name") or right_id),
                        "name_similarity": similarity,
                        "param_overlap": overlap,
                        "reasons": reasons,
                        "review_status": "pending_manual_or_llm",
                    }
                )
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    candidates.sort(
        key=lambda row: (
            -len(row["reasons"]),
            -float(row["name_similarity"]),
            -float(row["param_overlap"]),
            row["left_skill_id"],
            row["right_skill_id"],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in candidates:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "status": "pending_manual_or_llm_review" if candidates else "no_borderline_candidates",
        "candidate_count": len(candidates),
        "max_candidates": max_candidates,
        "method": "fuzzy_name_or_partial_param_overlap_v1",
    }


def _merge_skill_record(existing: dict[str, Any], row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    desc = _skill_description(row)
    if desc and desc != existing.get("description") and desc not in existing["alternate_descriptions"]:
        existing["alternate_descriptions"].append(desc)
    source = str(source_path)
    if source not in existing["source_files"]:
        existing["source_files"].append(source)
    return existing


def _traject_skill_record(row: dict[str, Any], source_path: Path, dedup_method: str = "traject_tool_identity_v2") -> dict[str, Any]:
    sid = traject_skill_id(row)
    return {
        "skill_id": sid,
        "canonical_skill_id": sid,
        "name": str(row.get("tool name") or sid),
        "description": str(row.get("tool description") or row.get("parent tool description") or sid),
        "alternate_descriptions": [
            str(row.get("parent tool description"))
        ] if row.get("parent tool description") else [],
        "environment": row.get("domain name"),
        "source": "TRAJECT-Bench",
        "source_files": [str(source_path)],
        "executor_desc": row.get("tool description") or row.get("parent tool description") or "",
        "input_schema": {
            "required_parameters": row.get("required_parameters") or row.get("required parameters") or [],
            "optional_parameters": row.get("optional_parameters") or row.get("optional parameters") or [],
        },
        "output_schema": row.get("output_info") or {},
        "body": row.get("code") or "",
        "provenance": {
            "dedup_method": dedup_method,
            "parent_tool_name": row.get("parent tool name"),
            "api_name": row.get("API name"),
            "domain_name": row.get("domain name"),
        },
    }


def _webshop_skill_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for action_type, item in WEBSHOP_ACTION_SKILLS.items():
        sid = str(item["skill_id"])
        records[sid] = {
            "skill_id": sid,
            "canonical_skill_id": sid,
            "name": str(item["name"]),
            "description": str(item["description"]),
            "alternate_descriptions": [],
            "environment": "webshop",
            "source": "lclan/webshop_expert_trajectories",
            "source_files": [f"data/webshop_expert_trajectories/{WEBSHOP_HF_GPT4_FILENAME}"],
            "executor_desc": str(item["description"]),
            "input_schema": {"action_type": {"type": action_type}},
            "output_schema": {"action_text": {"type": "string"}},
            "body": str(item["description"]),
            "provenance": {
                "dedup_method": "webshop_action_type_identity_v1",
                "source_dataset": "lclan/webshop_expert_trajectories",
                "action_type": action_type,
            },
        }
    return records


def build_skill_pool(out_path: Path, used_skill_ids: set[str], repo_root: str | Path | None = None) -> dict:
    root = _repo_root(repo_root)
    source_paths = [
        root / "data" / "clstr_full_base_train" / "skills.jsonl",
        root / "data" / "clstr_full_base_train_trans_skill_ce" / "skills.jsonl",
        root / "data" / "clstr_dagger_expert_corrected_train_enriched" / "skills.jsonl",
        root / "data" / "clstr_qwen3_structured_train" / "skills.jsonl",
        root / "data" / "aux_skillnet_rebuilt" / "skills.jsonl",
        root / "data" / "aux_trajectories" / "pseudo_skills.jsonl",
        root / "data" / "appworld_skill_pool" / "skill_pool.jsonl",
        root / "data" / "skillret" / "skills.jsonl",
        root / "data" / "toolret_training" / "skills.jsonl",
        root / "data" / "toolbench_g3" / "skills.jsonl",
    ]
    records: dict[str, dict[str, Any]] = {}
    for source_path in source_paths:
        for row in read_jsonl(source_path):
            sid = _skill_id(row)
            if not sid or sid not in used_skill_ids:
                continue
            if sid in records:
                _merge_skill_record(records[sid], row, source_path)
                continue
            records[sid] = {
                "skill_id": sid,
                "canonical_skill_id": sid,
                "name": str(row.get("name") or sid),
                "description": _skill_description(row),
                "alternate_descriptions": [],
                "environment": row.get("environment"),
                "source": row.get("source") or row.get("source_dataset") or "unknown",
                "source_files": [str(source_path)],
                "provenance": {
                    "dedup_method": "identity_skill_id_v2",
                    "source_record_id": row.get("id") or row.get("skill_id"),
                },
            }
            for key in ("body", "executor_desc", "input_schema", "output_schema", "failure_modes"):
                if key in row:
                    records[sid][key] = row[key]
    traject_public_data = find_traject_public_data(root)
    traject_tools_path = traject_public_data / "tools" / "all_tools.json" if traject_public_data is not None else None
    traject_tools = read_json(traject_tools_path) if traject_tools_path is not None else None
    if isinstance(traject_tools, list) and traject_tools_path is not None:
        for row in traject_tools:
            if not isinstance(row, dict):
                continue
            sid = traject_skill_id(row)
            if sid not in used_skill_ids:
                continue
            if sid in records:
                _merge_skill_record(records[sid], row, traject_tools_path)
                continue
            records[sid] = _traject_skill_record(row, traject_tools_path)
    if traject_public_data is not None:
        for _traj_type, _domain, query_path in iter_traject_query_files(traject_public_data):
            payload = read_json(query_path)
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                tools = item.get("tool list") or item.get("tool_list") or []
                if not isinstance(tools, list):
                    continue
                for row in tools:
                    if not isinstance(row, dict):
                        continue
                    sid = traject_skill_id(row)
                    if sid not in used_skill_ids or sid in records:
                        continue
                    records[sid] = _traject_skill_record(
                        row,
                        query_path,
                        dedup_method="traject_query_tool_identity_v2",
                    )
    for sid, row in _webshop_skill_records().items():
        if sid in used_skill_ids and sid not in records:
            records[sid] = row
    missing = sorted(used_skill_ids - set(records))
    for sid in missing:
        records[sid] = {
            "skill_id": sid,
            "canonical_skill_id": sid,
            "name": sid,
            "description": sid,
            "alternate_descriptions": [],
            "environment": None,
            "source": "placeholder_from_training_reference",
            "source_files": [],
            "provenance": {
                "dedup_method": "identity_skill_id_v2",
                "missing_source_record": True,
            },
        }
    canonical_records, alias_map, alias_rows, dedup_stats = _partial_dedup_skill_records(records)
    alias_path = out_path.with_name("skill_aliases.jsonl")
    borderline_path = out_path.with_name("skill_dedup_borderline_candidates.jsonl")
    borderline_stats = write_borderline_dedup_candidates(borderline_path, records, alias_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for sid in sorted(canonical_records):
            fout.write(json.dumps(canonical_records[sid], ensure_ascii=False) + "\n")
    with alias_path.open("w") as fout:
        for row in alias_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "total_skills": len(canonical_records),
        "used_skill_count": len(used_skill_ids),
        "missing_source_record_count": len(missing),
        "missing_source_record_skill_ids": missing[:100],
        "dedup_method": "partial_rule_name_param_v2",
        "raw_skill_count": dedup_stats["raw_skill_count"],
        "canonical_skill_count": dedup_stats["canonical_skill_count"],
        "merged_group_count": dedup_stats["merged_group_count"],
        "alias_count": dedup_stats["alias_count"],
        "changed_alias_count": dedup_stats["changed_alias_count"],
        "borderline_review": borderline_stats,
        "__alias_map": alias_map,
    }


def _canonicalize_skill_value(value: Any, alias_map: dict[str, str]) -> tuple[Any, bool, bool]:
    if value is None:
        return value, False, False
    sid = str(value)
    if not sid or sid not in alias_map:
        return value, False, False
    canonical = alias_map[sid]
    return canonical, True, canonical != sid


def _canonicalize_jsonl_skill_refs(path: Path, alias_map: dict[str, str], scalar_fields: tuple[str, ...], list_fields: tuple[str, ...]) -> dict[str, int]:
    if not path.exists():
        return {"rows": 0, "rewritten_rows": 0, "changed_skill_refs": 0}
    rows = list(read_jsonl(path))
    rewritten_rows = 0
    changed_refs = 0
    with path.open("w") as fout:
        for row in rows:
            row_touched = False
            for field in scalar_fields:
                value, touched, changed = _canonicalize_skill_value(row.get(field), alias_map)
                if touched:
                    row_touched = True
                    row[field] = value
                    changed_refs += int(changed)
            for field in list_fields:
                values = row.get(field)
                if not isinstance(values, list):
                    continue
                new_values = []
                for item in values:
                    value, touched, changed = _canonicalize_skill_value(item, alias_map)
                    row_touched = row_touched or touched
                    changed_refs += int(changed)
                    new_values.append(value)
                row[field] = new_values
            replay_prefix = row.get("replay_prefix")
            if isinstance(replay_prefix, list):
                remapped_prefix: list[Any] = []
                for step in replay_prefix:
                    if not isinstance(step, dict):
                        remapped_prefix.append(step)
                        continue
                    remapped_step = dict(step)
                    value, touched, changed = _canonicalize_skill_value(
                        step.get("skill_id"),
                        alias_map,
                    )
                    if touched:
                        row_touched = True
                        remapped_step["skill_id"] = value
                        changed_refs += int(changed)
                    remapped_prefix.append(remapped_step)
                row["replay_prefix"] = remapped_prefix
            rewritten_rows += int(row_touched)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "rows": len(rows),
        "rewritten_rows": rewritten_rows,
        "changed_skill_refs": changed_refs,
    }


def canonicalize_unified_streams(out_dir: Path, alias_map: dict[str, str]) -> dict[str, Any]:
    trajectory_stats = _canonicalize_jsonl_skill_refs(
        out_dir / "trajectories.jsonl",
        alias_map,
        scalar_fields=("skill_id", "next_skill_id"),
        list_fields=("tool_inventory_skill_ids", "visible_inventory_skill_ids", "equivalent_next_skill_ids"),
    )
    retrieval_stats = _canonicalize_jsonl_skill_refs(
        out_dir / "retrieval.jsonl",
        alias_map,
        scalar_fields=("positive_skill_id",),
        list_fields=("negative_skill_ids",),
    )
    return {
        "trajectory_rows": trajectory_stats["rows"],
        "trajectory_rewritten_rows": trajectory_stats["rewritten_rows"],
        "trajectory_changed_skill_refs": trajectory_stats["changed_skill_refs"],
        "retrieval_rows": retrieval_stats["rows"],
        "retrieval_rewritten_rows": retrieval_stats["rewritten_rows"],
        "retrieval_changed_skill_refs": retrieval_stats["changed_skill_refs"],
    }


def apply_trajectory_source_quality_filter(
    path: Path,
    traj_stats: dict[str, Any],
    allowlist: str | Iterable[str] | None,
) -> dict[str, Any]:
    allowed = parse_string_list(allowlist)
    if not allowed:
        traj_stats["source_quality_filter"] = {"enabled": False, "allowlist": []}
        return traj_stats
    allowed_set = set(allowed)
    rows = list(read_jsonl(path))
    retained: list[dict[str, Any]] = []
    skipped_by_quality: Counter[str] = Counter()
    for row in rows:
        source_quality = str(row.get("source_quality") or "")
        if source_quality in allowed_set:
            retained.append(row)
        else:
            skipped_by_quality[source_quality or "missing"] += 1
    with path.open("w") as fout:
        for row in retained:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter()
    used_skill_ids: set[str] = set()
    for row in retained:
        benchmark = str(row.get("benchmark") or "unknown")
        source_quality = str(row.get("source_quality") or "unknown")
        counts[f"benchmark:{benchmark}"] += 1
        counts[f"source_quality:{source_quality}"] += 1
        for key in ("skill_id", "next_skill_id"):
            skill_id = str(row.get(key) or "")
            if skill_id:
                used_skill_ids.add(skill_id)
        for field in ("tool_inventory_skill_ids", "visible_inventory_skill_ids"):
            for skill_id in row.get(field) or []:
                if skill_id:
                    used_skill_ids.add(str(skill_id))
        for skill_id in row.get("equivalent_next_skill_ids") or []:
            if skill_id:
                used_skill_ids.add(str(skill_id))

    traj_stats = dict(traj_stats)
    traj_stats["total_rows"] = len(retained)
    traj_stats["used_skill_count"] = len(used_skill_ids)
    traj_stats["used_skill_ids"] = sorted(used_skill_ids)
    traj_stats["counts"] = dict(counts)
    traj_stats["source_quality_filter"] = {
        "enabled": True,
        "allowlist": allowed,
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "skipped_rows": len(rows) - len(retained),
        "skipped_by_source_quality": dict(sorted(skipped_by_quality.items())),
    }
    return traj_stats


def _resolve_first_existing(repo_root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            return path
    return None


def _contains_file(path: Path, relative_candidates: list[str]) -> bool:
    return any((path / candidate).is_file() for candidate in relative_candidates)


def _contains_any_glob(path: Path, patterns: list[str]) -> bool:
    return any(next(path.glob(pattern), None) is not None for pattern in patterns)


def _is_traject_public_data(path: Path) -> bool:
    public_data = path / "public_data" if (path / "public_data").exists() else path
    return (
        (public_data / "tools" / "all_tools.json").is_file()
        and (
            _contains_any_glob(public_data / "parallel", ["*/*.json"])
            or _contains_any_glob(public_data / "sequential", ["*/*.json"])
        )
    )


def _is_toolbench_g3_training_data(path: Path) -> bool:
    """ToolBench code/raw data is an import input, not a unified training source.
    Mark available only after the normalized CLSTR export has been created."""
    normalized_files = ["trajectories.jsonl", "retrieval.jsonl", "skills.jsonl"]
    return all((path / filename).is_file() for filename in normalized_files)


def _is_toolret_training_data(path: Path) -> bool:
    """The benchmark repo is code/leaderboard only. Training is available only
    when the HF ToolRet-Training-20w data or a normalized local export exists."""
    return (
        _contains_file(path, ["retrieval.jsonl", "data/toolret_training/retrieval.jsonl"])
        or _contains_any_glob(path, ["*.jsonl", "*.json", "*.parquet", "*.arrow"])
        or _contains_any_glob(path, ["toolret_training/*.jsonl", "ToolRet-Training-20w/*"])
        or _contains_any_glob(path, ["data/*.jsonl", "data/*.parquet", "data/**/*.jsonl", "data/**/*.parquet"])
    )


def _resolve_first_valid_source(
    repo_root: Path,
    candidates: list[str],
    validator,
) -> tuple[Path | None, Path | None]:
    first_existing = None
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            continue
        if first_existing is None:
            first_existing = path
        if validator(path):
            return path, first_existing
    return None, first_existing


def build_source_inventory(out_path: Path, repo_root: str | Path | None = None) -> dict:
    root = _repo_root(repo_root)
    specs = [
        {
            "source_id": "clstr_full_base_train",
            "benchmark": "alfworld_scienceworld_webshop",
            "role": "trajectory",
            "split_policy": "train_only",
            "candidates": ["data/clstr_full_base_train/train.jsonl"],
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "agentgym_agenttraj_l",
            "benchmark": "alfworld",
            "role": "weak_trajectory",
            "split_policy": "train_only",
            "candidates": ["data/agentgym_agenttraj_l/alfworld/converted_train.jsonl"],
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "hf_alfworld_admissible_success",
            "benchmark": "alfworld",
            "role": "high_quality_success_trajectory_v4_2",
            "split_policy": "train_only_success_and_admissible_filtered",
            "candidates": [
                "data/hf_alfworld_admissible_success",
                ".tmp/hf_alfworld_admissible_success",
            ],
            "remote_url": f"https://huggingface.co/datasets/{HF_ALFWORLD_ADMISSIBLE_SUCCESS_DATASET}",
            "validator": lambda path: path.is_file() or (path.is_dir() and any(path.rglob("*.parquet"))),
        },
        {
            "source_id": "clstr_dagger_expert_corrected_train_enriched",
            "benchmark": "alfworld",
            "role": "clean_trajectory_non_overlap_v4_1b",
            "split_policy": "train_only_non_overlap_with_existing_clean_rows",
            "candidates": ["data/clstr_dagger_expert_corrected_train_enriched/train.jsonl"],
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "clstr_qwen3_structured_train",
            "benchmark": "alfworld",
            "role": "clean_trajectory_non_overlap_v4_1b",
            "split_policy": "train_only_non_overlap_with_existing_clean_rows",
            "candidates": ["data/clstr_qwen3_structured_train/train.jsonl"],
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "webshop_expert_trajectories",
            "benchmark": "webshop",
            "role": "auxiliary_high_reward_trajectory",
            "split_policy": "train_only_high_reward_filtered",
            "candidates": [
                f"data/webshop_expert_trajectories/{WEBSHOP_HF_GPT4_FILENAME}",
                f".tmp/hf_probe/{WEBSHOP_HF_GPT4_FILENAME}",
            ],
            "remote_url": "https://huggingface.co/datasets/lclan/webshop_expert_trajectories",
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "skillret",
            "benchmark": "skillret",
            "role": "retrieval",
            "split_policy": "train_only_test_excluded",
            "candidates": ["data/skillret/qrels.jsonl"],
            "validator": lambda path: path.is_file(),
        },
        {
            "source_id": "traject_bench",
            "benchmark": "TRAJECT-Bench",
            "role": "target_trajectory_retrieval",
            "split_policy": "train_only_dev_test_excluded",
            "candidates": [
                "data/traject_bench",
                "../TRAJECT-Bench/public_data",
                "../TRAJECT-Bench",
                "../TRAJECT_Bench",
            ],
            "remote_url": "https://github.com/PengfeiHePower/TRAJECT-Bench",
            "validator": _is_traject_public_data,
        },
        {
            "source_id": "toolbench_g3",
            "benchmark": "ToolBench-G3",
            "role": "target_trajectory_retrieval",
            "split_policy": "train_only_dev_test_excluded",
            "candidates": [
                "data/toolbench_g3",
                "../ToolBench",
                "../toolbench",
            ],
            "remote_url": "https://github.com/OpenBMB/ToolBench",
            "validator": _is_toolbench_g3_training_data,
        },
        {
            "source_id": "toolret_training",
            "benchmark": "ToolRet",
            "role": "target_retrieval",
            "split_policy": "train_only_test_excluded",
            "candidates": [
                "data/toolret_training",
                "data/toolret",
                "../tool-retrieval-benchmark",
                "../ToolRet",
                "../toolret",
            ],
            "remote_url": "https://github.com/mangopy/tool-retrieval-benchmark",
            "validator": _is_toolret_training_data,
        },
    ]
    rows = []
    for spec in specs:
        validator = spec.get("validator", lambda path: path.exists())
        resolved, first_existing = _resolve_first_valid_source(root, spec["candidates"], validator)
        row = dict(spec)
        row.pop("candidates")
        row.pop("validator")
        row["available"] = resolved is not None
        row["resolved_path"] = str(resolved) if resolved is not None else None
        row["first_existing_path"] = str(first_existing) if first_existing is not None else None
        row["train_allowed"] = resolved is not None and "test" not in str(row["split_policy"])
        if resolved is not None and row["split_policy"] in {"train_only_test_excluded", "train_only_dev_test_excluded"}:
            row["train_allowed"] = True
        row["mirror_hint"] = {
            "github": "https://ghfast.top",
            "huggingface": "https://hf-mirror.com",
        }
        if resolved is not None:
            row["reason_if_missing"] = None
        elif first_existing is not None:
            row["reason_if_missing"] = "source_exists_but_required_training_files_missing"
        else:
            row["reason_if_missing"] = "source_path_missing_or_not_downloaded"
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "source_count": len(rows),
        "available_source_count": sum(1 for row in rows if row["available"]),
        "missing_required_target_sources": [
            row["source_id"]
            for row in rows
            if row["source_id"] in {"traject_bench", "toolbench_g3", "toolret_training"} and not row["available"]
        ],
    }


def write_leakage_audit(
    output_path: Path,
    trajectories_path: Path,
    retrieval_path: Path,
    excluded_appworld_ids: set[str],
    excluded_skillret_qids: set[str],
) -> dict:
    leaked_tasks = sorted(
        {
            str(row.get("task_id"))
            for row in read_jsonl(trajectories_path)
            if str(row.get("task_id")) in excluded_appworld_ids
        }
    )
    leaked_queries = sorted(
        {
            str(row.get("query_id"))
            for row in read_jsonl(retrieval_path)
            if str(row.get("query_id")) in excluded_skillret_qids
        }
    )
    audit = {
        "status": "ok" if not leaked_tasks and not leaked_queries else "failed",
        "leaked_appworld_task_ids_in_trajectories": leaked_tasks,
        "leaked_skillret_query_ids_in_retrieval": leaked_queries,
        "excluded_appworld_task_id_count": len(excluded_appworld_ids),
        "excluded_skillret_query_id_count": len(excluded_skillret_qids),
        "policy": "train split only; AppWorld dev/test/challenge and SKILLRET test ids must not appear in outputs",
    }
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def build_unified_pretrain(
    repo_root: str | Path | None = None,
    output_dir: str | Path = "data/clstr_unified_pretrain_v2",
    schema_version: str = "v2",
    trajectory_retrieval_caps: str | dict[str, int] | None = None,
    dataset_recipe: str = "legacy",
    webshop_min_reward: float = 0.8,
    trajectory_source_quality_allowlist: str | Iterable[str] | None = None,
    progressive_filtered_weak_policy_cap: int = DEFAULT_PROGRESSIVE_FILTERED_WEAK_POLICY_CAP,
    extra_retrieval_paths: str | Path | Iterable[str | Path] | None = None,
) -> dict:
    root = _repo_root(repo_root)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_appworld = load_appworld_dev_test_ids(root)
    excluded_skillret = load_skillret_test_query_ids(root)
    if trajectory_retrieval_caps is None:
        trajectory_retrieval_caps = (
            DEFAULT_V4_1_TRAJECTORY_RETRIEVAL_CAPS
            if dataset_recipe in V4_CLEAN_RECIPES
            else DEFAULT_TRAJECTORY_RETRIEVAL_CAPS
        )
    traj_stats = build_trajectories(
        out_dir / "trajectories.jsonl",
        excluded_appworld,
        repo_root=root,
        dataset_recipe=dataset_recipe,
        webshop_min_reward=webshop_min_reward,
        progressive_filtered_weak_policy_cap=progressive_filtered_weak_policy_cap,
    )
    traj_stats = apply_trajectory_source_quality_filter(
        out_dir / "trajectories.jsonl",
        traj_stats,
        trajectory_source_quality_allowlist,
    )
    retr_stats = build_retrieval_pairs(
        out_dir / "retrieval.jsonl",
        excluded_skillret,
        repo_root=root,
        excluded_appworld_ids=excluded_appworld,
        trajectories_path=out_dir / "trajectories.jsonl",
        trajectory_retrieval_caps=trajectory_retrieval_caps,
        extra_retrieval_paths=extra_retrieval_paths,
    )
    used_skill_ids = set(traj_stats.pop("used_skill_ids", [])) | set(retr_stats.pop("used_skill_ids", []))
    skill_stats = build_skill_pool(out_dir / "skill_pool.jsonl", used_skill_ids, repo_root=root)
    alias_map = skill_stats.pop("__alias_map", {})
    stream_remap_stats = canonicalize_unified_streams(out_dir, alias_map)
    skill_stats["stream_remap"] = stream_remap_stats
    inventory_stats = build_source_inventory(out_dir / "source_inventory.jsonl", repo_root=root)
    leakage_audit = write_leakage_audit(
        out_dir / "leakage_audit.json",
        out_dir / "trajectories.jsonl",
        out_dir / "retrieval.jsonl",
        excluded_appworld,
        excluded_skillret,
    )
    history_channel_audit = {
        "trajectory_stream": audit_history_channel_rows(
            read_jsonl(out_dir / "trajectories.jsonl"),
            require_explicit_current=True,
            require_actual_replay_observation=True,
        ),
        "retrieval_stream": audit_history_channel_rows(
            (
                {
                    "state_text_current": str(row.get("query_text") or ""),
                    "step_index": 1,
                }
                for row in read_jsonl(out_dir / "retrieval.jsonl")
            ),
            require_explicit_current=True,
        ),
    }
    history_channel_audit["status"] = (
        "ok"
        if all(
            report["status"] == "ok"
            for report in history_channel_audit.values()
            if isinstance(report, dict)
        )
        else "action_required"
    )
    (out_dir / "history_channel_audit.json").write_text(
        json.dumps(history_channel_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": (
            "ok"
            if leakage_audit["status"] == "ok"
            and history_channel_audit["status"] == "ok"
            else "failed"
        ),
        "schema_version": schema_version,
        "dataset_recipe": dataset_recipe,
        "output_dir": str(out_dir.relative_to(root)) if out_dir.is_relative_to(root) else str(out_dir),
        "files": {
            "trajectories": "trajectories.jsonl",
            "retrieval": "retrieval.jsonl",
            "skill_pool": "skill_pool.jsonl",
            "skill_aliases": "skill_aliases.jsonl",
            "skill_dedup_borderline_candidates": "skill_dedup_borderline_candidates.jsonl",
            "source_inventory": "source_inventory.jsonl",
            "leakage_audit": "leakage_audit.json",
            "history_channel_audit": "history_channel_audit.json",
        },
        "trajectory_stream": traj_stats,
        "retrieval_stream": retr_stats,
        "skill_pool": skill_stats,
        "source_inventory": inventory_stats,
        "leakage_audit": leakage_audit,
        "history_channel_audit": history_channel_audit,
        "sources": {
            "clstr_full_base_train": "existing unified v1 trajectory source",
            "agentgym_agenttraj_l": "weak ALFWorld policy source when present",
            "hf_alfworld_admissible_success": "HF ALFWorld success/admissible trajectories used by v4.2 when present",
            "clstr_dagger_expert_corrected_train_enriched": "clean ALFWorld DAgger rows used by v4.1b only when non-overlapping",
            "clstr_qwen3_structured_train": "clean ALFWorld verified teacher rows used by v4.1b only when non-overlapping",
            "webshop_expert_trajectories": "HF WebShop GPT-4 trajectories filtered by reward for v4.1 when present",
            "skillret": "train split retrieval pairs with test qids excluded",
            "extra_retrieval": "optional leakage-filtered retrieval rows passed through --extra_retrieval_paths",
        },
        "not_yet_included": inventory_stats["missing_required_target_sources"],
        "note": "This is data preparation only. No training is launched by this script.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--output_dir", default="data/clstr_unified_pretrain_v2",
                        help="Output directory (relative to repo root).")
    parser.add_argument("--schema_version", default="v2")
    parser.add_argument(
        "--dataset_recipe",
        default="legacy",
        choices=["legacy", "v4_1", "v4_1b", V4_2_ALFWORLD_HF_RECIPE, V4_2_PROGRESSIVE_FINAL_RECIPE],
        help=(
            "Dataset recipe. v4_1 filters aux-only full-base rows and adds high-reward WebShop trajectories. "
            "v4_1b additionally adds non-overlapping clean ALFWorld rows. "
            "v4_2_alfworld_hf_quality additionally adds HF ALFWorld admissible-success rows before weak AgentGym rows. "
            "v4_2_progressive_final keeps v4.2 high-quality data and adds capped filtered weak-policy augmentation."
        ),
    )
    parser.add_argument(
        "--progressive_filtered_weak_policy_cap",
        type=int,
        default=DEFAULT_PROGRESSIVE_FILTERED_WEAK_POLICY_CAP,
        help=(
            "Maximum filtered AgentGym weak_policy rows retained by "
            "--dataset_recipe=v4_2_progressive_final. These rows never enable L_trans_skill_ce."
        ),
    )
    parser.add_argument(
        "--webshop_min_reward",
        type=float,
        default=0.8,
        help="Minimum reward for lclan WebShop GPT-4 trajectories when --dataset_recipe=v4_1.",
    )
    parser.add_argument(
        "--trajectory_retrieval_caps",
        default=None,
        help=(
            "Comma-separated benchmark=row_cap controls for trajectory-derived Stage0 retrieval pairs. "
            "Use -1 for unlimited and 0 to disable a benchmark. If omitted, defaults depend on --dataset_recipe."
        ),
    )
    parser.add_argument(
        "--trajectory_source_quality_allowlist",
        default=None,
        help=(
            "Optional comma-separated source_quality allowlist applied to trajectories before "
            "trajectory-derived retrieval and skill-pool construction."
        ),
    )
    parser.add_argument(
        "--extra_retrieval_paths",
        default=None,
        help=(
            "Optional comma-separated JSONL paths with Stream-B retrieval rows "
            "(query_id, query_text, positive_skill_id, negative_skill_ids). "
            "Rows are leakage-filtered against AppWorld dev/test task ids and SkillRET test qids."
        ),
    )
    args = parser.parse_args()

    print(f"Loading leakage exclusion sets...")
    repo_root = _repo_root(args.repo_root)
    excluded_appworld = load_appworld_dev_test_ids(repo_root)
    excluded_skillret = load_skillret_test_query_ids(repo_root)
    print(f"  AppWorld dev/test/challenge task_ids: {len(excluded_appworld)} (must NOT appear in training)")
    print(f"  SKILLRET test query_ids: {len(excluded_skillret)} (must NOT appear in training)")

    manifest = build_unified_pretrain(
        repo_root=repo_root,
        output_dir=args.output_dir,
        schema_version=args.schema_version,
        trajectory_retrieval_caps=args.trajectory_retrieval_caps,
        dataset_recipe=args.dataset_recipe,
        webshop_min_reward=args.webshop_min_reward,
        trajectory_source_quality_allowlist=args.trajectory_source_quality_allowlist,
        progressive_filtered_weak_policy_cap=args.progressive_filtered_weak_policy_cap,
        extra_retrieval_paths=args.extra_retrieval_paths,
    )
    print(f"Manifest written to {repo_root / args.output_dir / 'manifest.json'}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
