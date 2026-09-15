from __future__ import annotations

import json
from pathlib import Path

from clstr.toolbench_clean_training_export import export_toolbench_clean_unified_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_toolbench_clean_unified_data_filters_eval_rows_from_trajectories_and_retrieval(tmp_path):
    data_root = tmp_path / "unified"
    eval_rows_path = tmp_path / "eval_trajectories.jsonl"
    out = tmp_path / "clean"

    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/current"},
            {"skill_id": "toolbench-g3/music/search"},
            {"skill_id": "alfworld/go-to-sink"},
        ],
    )
    _write_jsonl(
        eval_rows_path,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-10018::0",
                "trajectory_id": "toolbench-g3-10018",
                "state_text": "eval state",
                "skill_id": "toolbench-g3/weather/current",
                "next_skill_id": "toolbench-g3/weather/current",
                "provenance": {
                    "answer_path": "/tmp/ToolBench/data/answer/G3_answer/10018_ChatGPT_DFS_woFilter_w2.json"
                },
            }
        ],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-10018::0",
                "trajectory_id": "toolbench-g3-10018",
                "state_text": "eval state",
                "skill_id": "toolbench-g3/weather/current",
                "next_skill_id": "toolbench-g3/weather/current",
                "provenance": {
                    "split": "train_or_released_g3",
                    "answer_path": "/tmp/ToolBench/data/answer/G3_answer/10018_ChatGPT_DFS_woFilter_w2.json",
                },
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-3::0",
                "trajectory_id": "toolbench-g3-3",
                "state_text": "safe toolbench state",
                "skill_id": "toolbench-g3/music/search",
                "next_skill_id": "toolbench-g3/music/search",
                "provenance": {"split": "train_or_released_g3"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "alf-1::0",
                "trajectory_id": "alf-1",
                "state_text": "safe alfworld state",
                "skill_id": "alfworld/go-to-sink",
                "next_skill_id": "alfworld/go-to-sink",
                "provenance": {"split": "train"},
            },
        ],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-10018",
                "query_text": "eval query",
                "positive_skill_id": "toolbench-g3/weather/current",
            },
            {
                "source": "trajectory_derived_toolbench_g3",
                "query_id": "trajectory-derived::toolbench_g3::toolbench-g3-10018::0::next",
                "query_text": "eval trajectory derived query",
                "positive_skill_id": "toolbench-g3/weather/current",
                "provenance": json.dumps(
                    {
                        "benchmark": "toolbench_g3",
                        "trajectory_id": "toolbench-g3-10018",
                        "task_id": "toolbench-g3-10018::0",
                    }
                ),
            },
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-3",
                "query_text": "safe static query",
                "positive_skill_id": "toolbench-g3/music/search",
            },
            {
                "source": "trajectory_derived_toolbench_g3",
                "query_id": "trajectory-derived::toolbench_g3::toolbench-g3-3::0::next",
                "query_text": "safe trajectory derived query",
                "positive_skill_id": "toolbench-g3/music/search",
                "provenance": json.dumps(
                    {
                        "benchmark": "toolbench_g3",
                        "trajectory_id": "toolbench-g3-3",
                        "task_id": "toolbench-g3-3::0",
                    }
                ),
            },
            {
                "source": "trajectory_derived_alfworld",
                "query_id": "trajectory-derived::alfworld::alf-1::0::next",
                "query_text": "safe alfworld query",
                "positive_skill_id": "alfworld/go-to-sink",
            },
        ],
    )

    report = export_toolbench_clean_unified_data(
        data_root=data_root,
        eval_trajectories_path=eval_rows_path,
        output_dir=out,
    )

    assert report["status"] == "ok"
    assert report["exclusion"]["query_ids"] == 1
    assert report["trajectories"]["removed_rows"] == 1
    assert report["retrieval"]["removed_rows"] == 2

    clean_trajectories = _read_jsonl(out / "trajectories.jsonl")
    clean_retrieval = _read_jsonl(out / "retrieval.jsonl")
    assert {row["task_id"] for row in clean_trajectories} == {"toolbench-g3-3::0", "alf-1::0"}
    assert {row["query_id"] for row in clean_retrieval} == {
        "toolbench-g3-3",
        "trajectory-derived::toolbench_g3::toolbench-g3-3::0::next",
        "trajectory-derived::alfworld::alf-1::0::next",
    }
    assert (out / "skill_pool.jsonl").read_text(encoding="utf-8") == (
        data_root / "skill_pool.jsonl"
    ).read_text(encoding="utf-8")

