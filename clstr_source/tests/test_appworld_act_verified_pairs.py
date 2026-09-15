from __future__ import annotations

import json
from pathlib import Path

from clstr.appworld_act_verified_pairs import build_appworld_act_verified_pairs
from clstr.data import load_verified_pairs


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_appworld_act_verified_pairs_from_train_oracle_api_trace(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    task_dir = appworld_root / "data" / "tasks" / "task_1"
    _write_json(task_dir / "specs.json", {"instruction": "Use app foo then app bar."})
    _write_json(task_dir / "ground_truth" / "required_apps.json", ["demo"])
    _write_json(
        task_dir / "ground_truth" / "api_calls.json",
        [
            {"method": "get", "url": "/demo/foo", "data": {}},
            {"method": "post", "url": "/demo/bar", "data": {"value": 1}},
        ],
    )

    tasks_path = tmp_path / "train_tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "split": "train",
                "query": "Instruction: Use app foo then app bar.",
                "instruction_text": "Use app foo then app bar.",
                "required_apps": ["demo"],
                "positive_skill_ids": ["skill/foo", "skill/bar"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    skill_pool_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "skill_id": "skill/foo",
                        "name": "demo foo",
                        "description": "Call foo.",
                        "body": "result = apis.demo.foo()",
                        "executor_desc": "apis.demo.foo",
                        "failure_modes": [],
                    }
                ),
                json.dumps(
                    {
                        "skill_id": "skill/bar",
                        "name": "demo bar",
                        "description": "Call bar.",
                        "body": "result = apis.demo.bar(value=value)",
                        "executor_desc": "apis.demo.bar",
                        "failure_modes": [],
                    }
                ),
                json.dumps(
                    {
                        "skill_id": "skill/negative",
                        "name": "demo negative",
                        "description": "A distractor.",
                        "body": "result = apis.demo.baz()",
                        "executor_desc": "apis.demo.baz",
                        "failure_modes": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_jsonl = tmp_path / "verified.jsonl"
    manifest_path = tmp_path / "manifest.json"
    report = build_appworld_act_verified_pairs(
        appworld_root=appworld_root,
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_jsonl=output_jsonl,
        manifest_path=manifest_path,
        top_k=3,
    )

    assert report["status"] == "ok"
    assert report["task_count"] == 1
    assert report["tasks_with_skill_sequences"] == 1
    assert report["verified_pair_count"] == 1
    assert report["dev_test_used_for_training"] is False

    pairs = load_verified_pairs(output_jsonl)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.task_id == "task_1"
    assert pair.action_at_t == 0
    assert pair.a_next_plus == 1
    assert 1 in pair.candidates_next
    assert pair.replay_prefix == []
    assert "demo.foo" in pair.obs_at_t
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["verified_pair_count"] == 1
