from __future__ import annotations

import json
from pathlib import Path

from clstr.toolbench_skillrouter_official import (
    export_toolbench_g3_skillrouter_eval_core,
    export_toolbench_g3_trajectory_skillrouter_eval_core,
    export_verified_toolbench_existing_eval,
    verify_toolbench_executed_results,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_verify_toolbench_executed_results_requires_alias_and_exact_tree_event(tmp_path):
    answer_path = tmp_path / "answer.json"
    _write_json(
        answer_path,
        {
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "search_for_music",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": '{"artist":"Miles Davis"}',
                                    "observation": '{"response":"Kind of Blue"}',
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            }
        },
    )
    skill_id = "toolbench-g3/music/search"
    rows, report = verify_toolbench_executed_results(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-1::0",
                "skill_id": skill_id,
                "action_text": 'search_for_music: {"artist":"Miles Davis"}',
                "next_observation_text": '{"response":"Kind of Blue"}',
                "provenance": {
                    "source_dataset": "ToolBench-G3",
                    "answer_path": str(answer_path),
                },
            }
        ],
        [
            {
                "skill_id": skill_id,
                "provenance": {
                    "action_alias": "search_for_music",
                    "action_aliases": ["search_for_music"],
                },
            }
        ],
    )

    assert report["aligned_row_count"] == 1
    assert report["nonempty_result_row_count"] == 1
    assert rows[0]["observation_source"] == "executed_trace:toolbench_g3"
    assert rows[0]["actual_result_executed"] is True
    assert rows[0]["actual_result_skill_id"] == skill_id
    assert rows[0]["actual_result_text"] == '{"response":"Kind of Blue"}'
    assert len(rows[0]["actual_result_event_id"]) == 64


def test_verify_toolbench_maps_raw_action_alias_to_deduplicated_skill(tmp_path):
    answer_path = tmp_path / "answer.json"
    _write_json(
        answer_path,
        {
            "tree": {
                "tree": {
                    "node_type": "Action",
                    "description": "raw-alias",
                    "children": [
                        {
                            "node_type": "Action Input",
                            "description": '{"x":1}',
                            "observation": "done",
                        }
                    ],
                }
            }
        },
    )
    canonical_id = "toolbench-g3/canonical/tool"
    raw_id = "toolbench-g3/raw/tool"
    rows, report = verify_toolbench_executed_results(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-dedup::0",
                "skill_id": canonical_id,
                "action_text": 'raw-alias: {"x":1}',
                "next_observation_text": "done",
                "provenance": {
                    "source_dataset": "ToolBench-G3",
                    "answer_path": str(answer_path),
                },
            }
        ],
        [
            {
                "skill_id": canonical_id,
                "alias_skill_ids": [canonical_id, raw_id],
            }
        ],
        [
            {
                "skill_id": raw_id,
                "provenance": {
                    "action_alias": "raw_alias",
                    "action_aliases": ["raw_alias"],
                },
            }
        ],
    )
    assert rows[0]["actual_result_skill_id"] == canonical_id
    assert report["verification_skill_count"] == 1


def test_verified_toolbench_copy_preserves_existing_split_membership(tmp_path):
    answer_path = tmp_path / "answer.json"
    _write_json(
        answer_path,
        {
            "tree": {
                "tree": {
                    "node_type": "Action",
                    "description": "raw_alias",
                    "children": [
                        {
                            "node_type": "Action Input",
                            "description": "{}",
                            "observation": "done",
                        }
                    ],
                }
            }
        },
    )
    canonical_id = "toolbench-g3/canonical/tool"
    raw_id = "toolbench-g3/raw/tool"
    source_rows = tmp_path / "source.jsonl"
    canonical_skills = tmp_path / "canonical.jsonl"
    raw_skills = tmp_path / "raw.jsonl"
    _write_jsonl(
        source_rows,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "kept::0",
                "trajectory_id": "kept",
                "skill_id": canonical_id,
                "action_text": "raw_alias: {}",
                "next_observation_text": "done",
                "provenance": {
                    "source_dataset": "ToolBench-G3",
                    "answer_path": str(answer_path),
                },
            }
        ],
    )
    _write_jsonl(
        canonical_skills,
        [{"skill_id": canonical_id, "alias_skill_ids": [raw_id]}],
    )
    _write_jsonl(
        raw_skills,
        [
            {
                "skill_id": raw_id,
                "provenance": {
                    "action_alias": "raw_alias",
                    "action_aliases": ["raw_alias"],
                },
            }
        ],
    )
    report = export_verified_toolbench_existing_eval(
        source_eval_trajectories_path=source_rows,
        skills_path=canonical_skills,
        verification_skills_path=raw_skills,
        output_dir=tmp_path / "verified",
    )
    copied = [
        json.loads(line)
        for line in Path(report["eval_trajectories_path"]).read_text().splitlines()
    ]
    assert [row["task_id"] for row in copied] == ["kept::0"]
    assert report["row_membership_and_order_preserved"] is True


def test_export_toolbench_g3_skillrouter_eval_core_writes_official_layout(tmp_path):
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    skills = tmp_path / "skills.jsonl"
    out = tmp_path / "official_eval"
    _write_jsonl(
        queries,
        [
            {"query_id": "toolbench-g3-1", "query_text": "find cocktails", "source": "toolbench_g3"},
            {"query_id": "toolbench-g3-2", "query_text": "find stocks", "source": "toolbench_g3"},
            {"query_id": "toolbench-g3-3", "query_text": "find weather", "source": "toolbench_g3"},
            {"query_id": "toolbench-g3-4", "query_text": "find songs", "source": "toolbench_g3"},
        ],
    )
    _write_jsonl(
        qrels,
        [
            {"query_id": "toolbench-g3-1", "skill_id": "toolbench-g3/cocktail/list", "relevance": 1},
            {"query_id": "toolbench-g3-1", "skill_id": "toolbench-g3/search/news", "relevance": 1},
            {"query_id": "toolbench-g3-2", "skill_id": "toolbench-g3/stocks/quote", "relevance": 1},
            {"query_id": "toolbench-g3-3", "skill_id": "toolbench-g3/weather/current", "relevance": 1},
            {"query_id": "toolbench-g3-4", "skill_id": "toolbench-g3/music/search", "relevance": 1},
        ],
    )
    _write_jsonl(
        skills,
        [
            {"skill_id": "toolbench-g3/cocktail/list", "name": "List Cocktails", "description": "List cocktails"},
            {"skill_id": "toolbench-g3/search/news", "name": "News Search", "description": "Search news"},
            {"skill_id": "toolbench-g3/stocks/quote", "name": "Stock Quote", "description": "Get stock quote"},
            {"skill_id": "toolbench-g3/weather/current", "name": "Weather", "description": "Get weather"},
            {"skill_id": "toolbench-g3/music/search", "name": "Music Search", "description": "Search music"},
        ],
    )

    report = export_toolbench_g3_skillrouter_eval_core(
        queries_path=queries,
        qrels_path=qrels,
        skills_path=skills,
        output_dir=out,
        eval_fraction=0.5,
        seed=7,
        max_eval_queries=2,
    )

    assert report["status"] == "ok"
    assert report["train_query_count"] == 2
    assert report["eval_query_count"] == 2
    assert (out / "train" / "tasks.jsonl").exists()
    assert (out / "train" / "relevance.json").exists()
    assert (out / "train" / "easy").is_dir()
    assert (out / "train" / "easy" / "skills.jsonl").exists()
    assert (out / "eval" / "tasks.jsonl").exists()
    assert (out / "eval" / "relevance.json").exists()
    assert (out / "eval" / "easy").is_dir()
    assert (out / "eval" / "easy" / "skills.jsonl").exists()
    assert (out / "eval" / "hard").is_dir()
    assert (out / "eval" / "hard" / "skills.jsonl").exists()

    eval_tasks = [json.loads(line) for line in (out / "eval" / "tasks.jsonl").read_text().splitlines()]
    assert len(eval_tasks) == 2
    assert sorted(eval_tasks[0]) == ["instruction_text", "source", "task_id"]
    relevance = json.loads((out / "eval" / "relevance.json").read_text())
    assert set(relevance) == {row["task_id"] for row in eval_tasks}
    for task_id, rel in relevance.items():
        assert rel["task_type"] == "toolbench_g3"
        assert rel["gt_skill_ids"]
        assert rel["core_gt_ids"] == rel["gt_skill_ids"]
        assert all(score == 1 for score in rel["relevance"].values())


def test_export_toolbench_g3_trajectory_skillrouter_eval_core_splits_by_trajectory(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skill_pool.jsonl"
    out = tmp_path / "trajectory_eval"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-1::0",
                "trajectory_id": "tb-1",
                "step_index": 0,
                "state_text": "goal: search music\nprevious_tools: <empty>",
                "skill_id": "toolbench-g3/music/search",
                "next_skill_id": "toolbench-g3/music/search",
                "loss_mask": {"L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-1::1",
                "trajectory_id": "tb-1",
                "step_index": 1,
                "state_text": "goal: search music\nprevious_tools: music",
                "skill_id": "toolbench-g3/music/search",
                "next_skill_id": "toolbench-g3/music/lyrics",
                "loss_mask": {"L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-2::0",
                "trajectory_id": "tb-2",
                "step_index": 0,
                "state_text": "goal: get weather",
                "skill_id": "toolbench-g3/weather/current",
                "next_skill_id": "toolbench-g3/weather/current",
                "loss_mask": {"L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "alf-1::0",
                "trajectory_id": "alf-1",
                "state_text": "not toolbench",
                "next_skill_id": "alfworld/skill",
                "loss_mask": {"L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
        ],
    )
    _write_jsonl(
        skills,
        [
            {"skill_id": "toolbench-g3/music/search", "name": "Music Search", "description": "Search music"},
            {"skill_id": "toolbench-g3/music/lyrics", "name": "Lyrics", "description": "Find lyrics"},
            {"skill_id": "toolbench-g3/weather/current", "name": "Weather", "description": "Get weather"},
        ],
    )

    report = export_toolbench_g3_trajectory_skillrouter_eval_core(
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=out,
        eval_fraction=0.5,
        seed=3,
    )

    assert report["status"] == "ok"
    assert report["usable_row_count"] == 3
    assert report["skill_count"] == 3
    assert (out / "train" / "tasks.jsonl").exists()
    assert (out / "eval" / "tasks.jsonl").exists()
    assert (out / "eval" / "easy" / "skills.jsonl").exists()
    assert (out / "eval_trajectories.jsonl").exists()

    train_rows = [json.loads(line) for line in (out / "train_trajectories.jsonl").read_text().splitlines()]
    eval_rows = [json.loads(line) for line in (out / "eval_trajectories.jsonl").read_text().splitlines()]
    assert {row["trajectory_id"] for row in train_rows}.isdisjoint({row["trajectory_id"] for row in eval_rows})
    assert all(row.get("split") != "eval" for row in eval_rows)

    eval_tasks = [json.loads(line) for line in (out / "eval" / "tasks.jsonl").read_text().splitlines()]
    eval_task_ids = {row["task_id"] for row in eval_tasks}
    assert eval_task_ids == {row["task_id"] for row in eval_rows}
    eval_relevance = json.loads((out / "eval" / "relevance.json").read_text())
    for row in eval_rows:
        assert eval_relevance[row["task_id"]]["gt_skill_ids"] == [row["next_skill_id"]]
