from __future__ import annotations

import json
from pathlib import Path

from clstr.toolbench_skillrouter_finetune import (
    _build_task_query_text,
    EvalCoreTask,
    load_eval_core_split,
    select_skill_pool_with_required_positives,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_load_eval_core_split_reads_official_layout(tmp_path):
    root = tmp_path / "eval_core"
    _write_jsonl(
        root / "train" / "tasks.jsonl",
        [
            {"task_id": "q1", "instruction_text": "state one", "source": "toolbench"},
            {"task_id": "q2", "instruction_text": "state two", "source": "toolbench"},
            {"task_id": "q3", "instruction_text": "no positive in pool", "source": "toolbench"},
        ],
    )
    _write_json(
        root / "train" / "relevance.json",
        {
            "q1": {"gt_skill_ids": ["s1"], "relevance": {"s1": 1}},
            "q2": {"core_gt_ids": ["s2"], "relevance": {"s2": 1}},
            "q3": {"gt_skill_ids": ["missing"], "relevance": {"missing": 1}},
        },
    )
    _write_jsonl(
        root / "train" / "easy" / "skills.jsonl",
        [
            {"skill_id": "s1", "name": "Skill One", "description": "one", "body": "body one"},
            {"skill_id": "s2", "name": "Skill Two", "description": "two", "body": "body two"},
        ],
    )

    split = load_eval_core_split(root, "train", max_tasks=2)

    assert [task.task_id for task in split.tasks] == ["q1", "q2"]
    assert split.tasks[0].positive_skill_ids == ["s1"]
    assert split.tasks[1].positive_skill_ids == ["s2"]
    assert [skill["skill_id"] for skill in split.skills] == ["s1", "s2"]


def test_select_skill_pool_with_required_positives_keeps_required_ids_first():
    skills = [
        {"skill_id": "a"},
        {"skill_id": "b"},
        {"skill_id": "c"},
        {"skill_id": "d"},
    ]

    selected = select_skill_pool_with_required_positives(skills, {"d", "b"}, max_skills=3)

    assert {row["skill_id"] for row in selected} >= {"b", "d"}
    assert len(selected) == 3
    assert [row["skill_id"] for row in selected][:2] == ["b", "d"]


def test_build_task_query_text_can_strip_toolbench_history_for_static_baseline():
    task = EvalCoreTask(
        task_id="toolbench-g3-1::1",
        instruction_text=(
            "goal: find weather and alerts\n"
            "benchmark: ToolBench-G3\n"
            "trajectory_type: intra_collection_multi_tool\n"
            "previous_tools: geocode_location"
        ),
        positive_skill_ids=["weather"],
        relevance={"weather": 1.0},
    )

    history_query = _build_task_query_text(task, query_variant="history")
    static_query = _build_task_query_text(task, query_variant="goal_only")

    assert "previous_tools: geocode_location" in history_query
    assert "previous_tools" not in static_query
    assert static_query == "goal: find weather and alerts"
