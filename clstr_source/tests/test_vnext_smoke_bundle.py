from __future__ import annotations

from scripts.build_clstr_vnext_smoke_bundle import (
    _select_ordinary_trajectories,
    _select_stage0_rows,
)


def _route_row(benchmark: str, index: int) -> dict:
    return {
        "benchmark": benchmark,
        "source": f"matched_{benchmark}",
        "state_text_current": f"goal: {benchmark} {index}",
        "state_text_causal": f"goal: {benchmark} {index}",
        "runtime_visible_catalog_id": "global",
        "current_state_route_set_skill_ids": [f"{benchmark}/tool"],
    }


def _trajectory(benchmark: str, trajectory_id: str, *, action_only: bool) -> list[dict]:
    rows = []
    for step in range(3):
        rows.append(
            {
                "benchmark": benchmark,
                "provenance": {"source_id": f"matched_{benchmark}_action_only_v1"},
                "trajectory_id": trajectory_id,
                "task_id": f"{trajectory_id}::{step}",
                "step_index": step,
                "capabilities": {"ordered_next_tool": step > 0},
                "actual_result_text": "" if action_only else f"result {step}",
                "observation_source": (
                    "action_only_no_tool_result" if action_only else "executed_result"
                ),
            }
        )
    return rows


def test_smoke_selection_requires_matched_benchmark_coverage():
    rows = [
        *[_route_row("legacy", index) for index in range(20)],
        _route_row("tau2", 0),
        _route_row("toolsandbox", 0),
    ]
    selected = _select_stage0_rows(
        rows,
        positive_field="current_state_route_set_skill_ids",
        limit=6,
        required_benchmarks=("tau2", "toolsandbox"),
    )
    assert {row["benchmark"] for row in selected} >= {"tau2", "toolsandbox"}


def test_smoke_selection_accepts_action_only_matched_trajectories():
    rows = [
        *_trajectory("legacy", "legacy/0", action_only=False),
        *_trajectory("tau2", "tau2/0", action_only=True),
        *_trajectory("toolsandbox", "toolsandbox/0", action_only=True),
    ]
    selected = _select_ordinary_trajectories(
        rows,
        limit=3,
        minimum_length=3,
        required_benchmarks=("tau2", "toolsandbox"),
    )
    assert {row["benchmark"] for row in selected} >= {"tau2", "toolsandbox"}
