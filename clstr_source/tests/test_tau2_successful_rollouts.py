from __future__ import annotations

import json
from pathlib import Path

from clstr.tau2_successful_rollouts import (
    TAU2_SUCCESSFUL_ROLLOUT_OBSERVATION_SOURCE,
    TAU2_SUCCESSFUL_ROLLOUT_SOURCE_ID,
    load_tau2_successful_rollout_corpus,
)


def _tool_call(call_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "name": name,
                "arguments": {"value": name},
                "requestor": "assistant",
            }
        ],
    }


def _simulation(simulation_id: str, second_tool: str) -> dict:
    return {
        "id": simulation_id,
        "task_id": "train-task",
        "trial": 0,
        "seed": 7,
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "assistant", "content": "Hello", "tool_calls": None},
            {"role": "user", "content": "Please resolve my visible request."},
            _tool_call(f"{simulation_id}-lookup", "lookup_user"),
            {
                "role": "tool",
                "id": f"{simulation_id}-lookup",
                "content": "visible lookup result",
                "error": False,
            },
            _tool_call(f"{simulation_id}-second", second_tool),
            {
                "role": "tool",
                "id": f"{simulation_id}-second",
                "content": f"visible {second_tool} result",
                "error": False,
            },
        ],
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    data_root = root / "data" / "tau2"
    domain_root = data_root / "domains" / "airline"
    results_root = data_root / "results" / "final"
    domain_root.mkdir(parents=True)
    results_root.mkdir(parents=True)
    (domain_root / "split_tasks.json").write_text(
        json.dumps({"train": ["train-task"], "test": ["test-task"]}),
        encoding="utf-8",
    )
    (domain_root / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "train-task",
                    "user_scenario": {
                        "instructions": {"known_info": "PRIVATE_ORACLE_DO_NOT_COPY"}
                    },
                    "evaluation_criteria": {
                        "actions": [
                            {"name": "lookup_user"},
                            {"name": "cancel_order"},
                            {"name": "refund_order"},
                        ]
                    },
                },
                {"id": "test-task", "evaluation_criteria": {"actions": []}},
            ]
        ),
        encoding="utf-8",
    )
    (domain_root / "policy.md").write_text("Visible airline policy.", encoding="utf-8")
    (results_root / "fixture-model_airline_default_fixture_4trials.json").write_text(
        json.dumps(
            {
                "tasks": [],
                "simulations": [
                    _simulation("sim-a", "cancel_order"),
                    _simulation("sim-b", "refund_order"),
                    {
                        **_simulation("failed", "cancel_order"),
                        "reward_info": {"reward": 0.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return data_root, results_root


def test_tau2_successful_rollouts_are_visible_result_grounded_and_multi_positive(
    tmp_path: Path,
) -> None:
    data_root, results_root = _write_fixture(tmp_path)
    corpus = load_tau2_successful_rollout_corpus(
        data_root,
        results_root=results_root,
        domains=("airline",),
        task_split="train",
        agent_models=("fixture-model",),
    )

    assert corpus.report["successful_tool_trajectory_count"] == 2
    assert corpus.report["successful_tool_event_count"] == 4
    assert corpus.report["multi_positive_row_count"] == 2
    assert corpus.report["input_visibility_contract"][
        "task_user_scenario_excluded"
    ] is True
    serialized = json.dumps(corpus.source_rows, ensure_ascii=False)
    assert "PRIVATE_ORACLE_DO_NOT_COPY" not in serialized
    assert "visible lookup result" in serialized
    second_rows = [row for row in corpus.source_rows if row["step_index"] == 1]
    assert all(
        "visible lookup result" not in row["state_text_current"]
        for row in second_rows
    )
    assert all(
        row["current_state_components"]["current_observation_text"]
        == "user: Please resolve my visible request."
        for row in second_rows
    )
    first_rows = [row for row in corpus.source_rows if row["step_index"] == 0]
    assert all(
        row["next_observation_text"] == "visible lookup result"
        for row in first_rows
    )
    assert {tuple(row["equivalent_next_skill_ids"]) for row in second_rows} == {
        ("tau2/airline/cancel_order", "tau2/airline/refund_order")
    }
    assert all(
        row["observation_source"] == TAU2_SUCCESSFUL_ROLLOUT_OBSERVATION_SOURCE
        for row in corpus.source_rows
    )
    assert all(
        row["provenance"]["source_id"] == TAU2_SUCCESSFUL_ROLLOUT_SOURCE_ID
        for row in corpus.source_rows
    )
    assert {
        row["current_state_components"]["goal_text"] for row in corpus.source_rows
    } == {"Please resolve my visible request."}
    visibility = corpus.report["input_visibility_contract"]
    assert visibility["route_state_external_message_roles"] == ["user"]
    assert visibility["raw_tool_results_excluded_from_state_text_current"] is True
    assert visibility["actual_tool_results_preserved_for_memory_correction"] is True
