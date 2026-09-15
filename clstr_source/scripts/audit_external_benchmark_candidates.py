#!/usr/bin/env python3
"""Audit external benchmark candidates for CLSTR routing experiments.

This script is intentionally CPU-only and local-file-only. It summarizes
whether candidate datasets expose enough structure for state/history -> next
tool or skill routing, without launching training or evaluation jobs.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _valid_nonempty_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 32:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:64].lower()
    except OSError:
        return False
    return "not found" not in head and "entry not found" not in head


def _mean(values: list[int | float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _pctl(values: list[int | float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def _counter_top(counter: collections.Counter[Any], limit: int = 20) -> list[list[Any]]:
    return [[k, v] for k, v in counter.most_common(limit)]


def _extract_tau_action_name(action: Any) -> str | None:
    if isinstance(action, str):
        return action
    if not isinstance(action, dict):
        return None
    for key in ("name", "action", "tool", "tool_name", "function", "function_name"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(action.get("request"), dict):
        return _extract_tau_action_name(action["request"])
    return None


def _extract_tau_actions(task: dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in (
        "actions",
        "oracle_actions",
        "oracle_action_sequence",
        "action_sequence",
        "solution",
        "steps",
    ):
        value = task.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    evaluation_criteria = task.get("evaluation_criteria")
    if isinstance(evaluation_criteria, dict) and isinstance(evaluation_criteria.get("actions"), list):
        candidates.extend(evaluation_criteria["actions"])
    names: list[str] = []
    for item in candidates:
        name = _extract_tau_action_name(item)
        if name:
            names.append(name)
    return names


def audit_tau2(probe_root: Path) -> dict[str, Any]:
    root = probe_root / "HuggingFaceH4__tau2-bench-data" / "domains"
    domain_reports: dict[str, Any] = {}
    all_actions = collections.Counter()
    total_tasks = 0
    total_steps = 0
    for task_path in sorted(root.glob("*/tasks.json")):
        domain = task_path.parent.name
        tasks = _load_json(task_path)
        if isinstance(tasks, dict):
            if isinstance(tasks.get("tasks"), list):
                tasks = tasks["tasks"]
            else:
                tasks = list(tasks.values())
        if not isinstance(tasks, list):
            domain_reports[domain] = {"error": f"unexpected tasks type {type(tasks).__name__}"}
            continue
        action_lens: list[int] = []
        action_counter: collections.Counter[str] = collections.Counter()
        field_counter: collections.Counter[str] = collections.Counter()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            field_counter.update(task.keys())
            actions = _extract_tau_actions(task)
            action_lens.append(len(actions))
            action_counter.update(actions)
        total_tasks += len(tasks)
        total_steps += sum(action_lens)
        all_actions.update(action_counter)
        domain_reports[domain] = {
            "tasks": len(tasks),
            "oracle_action_steps": sum(action_lens),
            "mean_actions_per_task": _mean(action_lens),
            "p50_actions_per_task": _pctl(action_lens, 0.5),
            "p90_actions_per_task": _pctl(action_lens, 0.9),
            "unique_action_names": len(action_counter),
            "top_action_names": _counter_top(action_counter, 15),
            "task_fields_top": _counter_top(field_counter, 20),
            "policy_md_present": _valid_nonempty_file(task_path.parent / "policy.md"),
            "db_json_present": _valid_nonempty_file(task_path.parent / "db.json"),
        }
    return {
        "status": "ok" if domain_reports else "missing",
        "path": str(root),
        "domains": domain_reports,
        "total_tasks": total_tasks,
        "total_oracle_action_steps": total_steps,
        "unique_action_names_all_domains": len(all_actions),
        "top_action_names_all_domains": _counter_top(all_actions, 30),
        "clstr_fit": {
            "state_history_to_next_tool": bool(total_steps),
            "multi_step": bool(total_steps and total_steps > total_tasks),
            "notes": [
                "Good fit if oracle action names are complete and policy/db context is retained as public state.",
                "Tool pool is domain-local and small; useful for stateful multi-step routing, less for very-large pool stress.",
            ],
        },
    }


def _iter_unitool_rows(path: Path) -> list[dict[str, Any]]:
    obj = _load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "rows", "instances", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, str):
        return tool
    if not isinstance(tool, dict):
        return None
    for key in ("name", "tool_name"):
        value = tool.get(key)
        if isinstance(value, str) and value:
            return value
    function = tool.get("function")
    if isinstance(function, dict):
        value = function.get("name")
        if isinstance(value, str) and value:
            return value
    return None


def _parse_tools_field(tools: Any) -> list[Any]:
    if isinstance(tools, list):
        return tools
    if isinstance(tools, dict):
        return list(tools.values())
    if isinstance(tools, str):
        try:
            parsed = json.loads(tools)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return list(parsed.values())
    return []


def _conversation_call_names(row: dict[str, Any]) -> list[str]:
    calls: list[str] = []
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return calls
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        if "function_call" in role or role in {"tool", "assistant"}:
            for key in ("name", "tool_name", "function_name"):
                value = turn.get(key)
                if isinstance(value, str) and value:
                    calls.append(value)
                    break
            else:
                value = turn.get("function_call")
                if isinstance(value, dict):
                    name = value.get("name")
                    if isinstance(name, str) and name:
                        calls.append(name)
                elif isinstance(value, str) and value:
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                            calls.append(parsed["name"])
                    except json.JSONDecodeError:
                        pass
        value = turn.get("value") or turn.get("content")
        if isinstance(value, str) and role == "function_call":
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                    calls.append(parsed["name"])
            except json.JSONDecodeError:
                pass
    return calls


def audit_unitoolcall(probe_root: Path) -> dict[str, Any]:
    root = probe_root / "EIT-NLP__UniToolCall"
    file_reports: dict[str, Any] = {}
    total_rows = 0
    unique_tools = set()
    call_counter: collections.Counter[str] = collections.Counter()
    missing_call_in_candidates = 0
    total_calls = 0
    for path in sorted(root.glob("*/*.json")):
        rel = path.relative_to(root).as_posix()
        rows = _iter_unitool_rows(path)
        tool_counts: list[int] = []
        call_counts: list[int] = []
        file_unique_tools = set()
        file_call_counter: collections.Counter[str] = collections.Counter()
        file_missing = 0
        file_calls = 0
        for row in rows:
            tools = _parse_tools_field(row.get("tools"))
            tool_names = {_tool_name(t) for t in tools}
            tool_names.discard(None)
            calls = _conversation_call_names(row)
            file_unique_tools.update(tool_names)
            unique_tools.update(tool_names)
            call_counter.update(calls)
            file_call_counter.update(calls)
            tool_counts.append(len(tool_names))
            call_counts.append(len(calls))
            file_calls += len(calls)
            file_missing += sum(1 for name in calls if name not in tool_names)
        total_rows += len(rows)
        total_calls += file_calls
        missing_call_in_candidates += file_missing
        file_reports[rel] = {
            "rows": len(rows),
            "mean_candidate_tools": _mean(tool_counts),
            "p50_candidate_tools": _pctl(tool_counts, 0.5),
            "p90_candidate_tools": _pctl(tool_counts, 0.9),
            "unique_candidate_tools": len(file_unique_tools),
            "function_calls": file_calls,
            "mean_calls_per_row": _mean(call_counts),
            "multi_call_rows": sum(1 for x in call_counts if x > 1),
            "call_not_in_candidate_tools": file_missing,
            "top_call_names": _counter_top(file_call_counter, 15),
        }
    return {
        "status": "ok" if file_reports else "missing",
        "path": str(root),
        "files": file_reports,
        "total_rows": total_rows,
        "total_function_calls": total_calls,
        "unique_candidate_tools_all_files": len(unique_tools),
        "call_not_in_candidate_tools": missing_call_in_candidates,
        "top_function_call_names": _counter_top(call_counter, 30),
        "clstr_fit": {
            "state_history_to_next_tool": bool(total_calls),
            "topk_candidate_rerank": True,
            "notes": [
                "Very good fit for fair top-K reranking because many rows ship candidate tool lists.",
                "Need row-level split hygiene because converted datasets mix multiple original sources.",
            ],
        },
    }


def audit_webshop(repo_root: Path) -> dict[str, Any]:
    dataset_dirs = [
        repo_root / "data" / "clstr_unified_pretrain_v4_1b",
        repo_root / "data" / "clstr_unified_pretrain_v4_2_progressive_final",
    ]
    data_reports: dict[str, Any] = {}
    for data_dir in dataset_dirs:
        path = data_dir / "trajectories.jsonl"
        if not path.exists():
            continue
        counts = collections.Counter()
        source_quality = collections.Counter()
        skill_counts = collections.Counter()
        rewards: list[float] = []
        total = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                benchmark = row.get("benchmark")
                counts[benchmark] += 1
                if benchmark == "webshop":
                    source_quality[row.get("source_quality")] += 1
                    skill_counts[row.get("skill_id")] += 1
                    reward = row.get("reward")
                    if isinstance(reward, (int, float)):
                        rewards.append(float(reward))
        data_reports[data_dir.name] = {
            "total_rows": total,
            "benchmark_counts": dict(counts),
            "webshop_rows": counts.get("webshop", 0),
            "webshop_source_quality": dict(source_quality),
            "webshop_skill_counts": dict(skill_counts),
            "webshop_reward_min": min(rewards) if rewards else None,
            "webshop_reward_max": max(rewards) if rewards else None,
        }
    harness = repo_root / "outputs" / "webshop_eval" / "harness_smoke_report.json"
    closed_loop = repo_root / "outputs" / "webshop_eval" / "clstr_controller_gate" / "metrics.json"
    return {
        "status": "ok" if data_reports else "missing",
        "datasets": data_reports,
        "harness_smoke": _load_json(harness) if harness.exists() else {"status": "missing"},
        "closed_loop_gate": _load_json(closed_loop) if closed_loop.exists() else {"status": "missing"},
        "clstr_fit": {
            "logged_training_source": True,
            "clean_official_eval_ready": bool(closed_loop.exists()),
            "notes": [
                "Existing WebShop rows are high-reward logged trajectories used in CLSTR training; avoid reporting same-source training rows as held-out benchmark evidence.",
                "Official harness smoke is available, but historical closed-loop CLSTR run used only one episode and got zero reward, so it is not yet a strong evaluation surface.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--probe_root", default=".tmp/benchmark_probe_direct")
    parser.add_argument("--output", default=".tmp/benchmark_probe_direct/external_benchmark_audit_report.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    probe_root = (repo_root / args.probe_root).resolve()
    report = {
        "webshop": audit_webshop(repo_root),
        "tau2_bench": audit_tau2(probe_root),
        "unitoolcall": audit_unitoolcall(probe_root),
        "recommendation": [
            "Use ToolBench-G3 as main result.",
            "Use WebShop only as logged/source-gated auxiliary evidence unless we build a clean held-out official eval.",
            "Prioritize UniToolCall for a clean top-20/top-K routing benchmark and fair SkillRouter/Qwen-reranker comparison.",
            "Prioritize tau2/tau-bench for a stateful domain-policy benchmark after downloading all domain task files.",
        ],
    }
    output = (repo_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
