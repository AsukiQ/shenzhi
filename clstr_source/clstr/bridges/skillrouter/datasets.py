from __future__ import annotations

from pathlib import Path

from clstr.data import Skill, Task, read_jsonl


def load_eval_tasks(path: Path) -> list[Task]:
    rows = read_jsonl(path)
    return [
        Task(
            task_id=row["task_id"],
            query=row.get("query", row.get("instruction_text", "")),
            meta=row,
        )
        for row in rows
    ]


def load_eval_pool(path: Path) -> list[Skill]:
    rows = read_jsonl(path)
    return [
        Skill(
            name=row["name"],
            description=row.get("description", ""),
            input_schema=row.get("input_schema", {}),
            output_schema=row.get("output_schema", {}),
            executor_desc=row.get("executor_desc", ""),
            failure_modes=row.get("failure_modes", []),
            skill_id=row.get("skill_id") or row.get("id"),
        )
        for row in rows
    ]
