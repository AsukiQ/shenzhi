#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_executor import load_appworld_api_refs
from clstr.appworld_official_executor import _DEFAULT_OFFICIAL_API_REFS, _api_call_refs_from_code


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _task_key(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("query_id") or "")


def _api_ref_text(ref: tuple[str, str]) -> str:
    return f"apis.{ref[0]}.{ref[1]}"


def _is_supervisor_complete_task_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "complete_task"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "supervisor"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "apis"
    )


def _complete_task_positional_calls(code: str) -> int:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_supervisor_complete_task_call(node) and node.args:
            count += 1
    return count


def audit_official_executor_runs(
    *,
    runs_path: str | Path,
    tasks_path: str | Path,
    appworld_root: str | Path,
) -> dict[str, Any]:
    tasks = {_task_key(row): row for row in _read_jsonl(tasks_path)}
    rows = _read_jsonl(runs_path)
    invalid_api_refs: Counter[str] = Counter()
    selected_skill_counts: Counter[str] = Counter()
    complete_task_positional_count = 0
    preflight_failed_steps = 0
    execution_failed_steps = 0
    step_count = 0

    valid_refs_by_task: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        task_id = _task_key(row)
        task = tasks.get(task_id, row)
        if task_id not in valid_refs_by_task:
            valid_refs_by_task[task_id] = load_appworld_api_refs(
                appworld_root=appworld_root,
                required_apps=[str(item) for item in task.get("required_apps", [])],
                api_refs=[str(item) for item in task.get("api_refs", [])],
            )
        valid_refs = valid_refs_by_task[task_id] | set(_DEFAULT_OFFICIAL_API_REFS)
        for step in row.get("steps", []):
            step_count += 1
            code = str(step.get("code") or "")
            complete_task_positional_count += _complete_task_positional_calls(code)
            for ref in sorted(_api_call_refs_from_code(code) - valid_refs):
                invalid_api_refs[_api_ref_text(ref)] += 1
            if not bool(step.get("preflight_ok", True)):
                preflight_failed_steps += 1
            output = str(step.get("execute_output") or "")
            if (not bool(step.get("execution_attempted", True))) or output.startswith("Execution failed"):
                execution_failed_steps += 1
            evidence = step.get("skill_evidence") or {}
            for skill_id in evidence.get("selected_skill_ids") or []:
                selected_skill_counts[str(skill_id)] += 1

    task_count = len(rows)
    success_count = sum(1 for row in rows if row.get("success"))
    task_completed_count = sum(1 for row in rows if row.get("task_completed"))
    evaluation_success_count = sum(1 for row in rows if row.get("evaluation_success"))
    return {
        "runs_path": str(runs_path),
        "tasks_path": str(tasks_path),
        "appworld_root": str(appworld_root),
        "task_count": task_count,
        "step_count": step_count,
        "success_count": success_count,
        "task_completed_count": task_completed_count,
        "evaluation_success_count": evaluation_success_count,
        "invalid_api_ref_count": sum(invalid_api_refs.values()),
        "invalid_api_refs": dict(invalid_api_refs),
        "complete_task_positional_count": complete_task_positional_count,
        "preflight_failed_steps": preflight_failed_steps,
        "execution_failed_steps": execution_failed_steps,
        "selected_skill_counts": dict(selected_skill_counts),
        "top_selected_skills": selected_skill_counts.most_common(20),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit official-style AppWorld executor JSONL traces.")
    parser.add_argument("--runs_path", required=True)
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument(
        "--appworld_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root",
    )
    parser.add_argument("--output_path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_official_executor_runs(
        runs_path=args.runs_path,
        tasks_path=args.tasks_path,
        appworld_root=args.appworld_root,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
