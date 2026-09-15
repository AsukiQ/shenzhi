from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from clstr.state_probe import (
    ProbeExample,
    binary_auroc,
    fit_regularized_linear_probe,
    load_toolbench_probe_trajectories,
    macro_f1,
    post_transition_events,
    result_has_error,
    split_report,
    trajectory_stratified_split,
)


def _example(trajectory_id: str, step: int, cumulative: int) -> ProbeExample:
    return ProbeExample(
        example_id=f"{trajectory_id}::{step}",
        trajectory_id=trajectory_id,
        step_index=step,
        state_text=f"state {trajectory_id} {step}",
        skill_id="skill/a",
        action_text=f"action {step}",
        result_text='{"error": "", "response": "ok"}',
        latest_result_label=0,
        cumulative_error_label=cumulative,
    )


def test_result_error_parser_handles_truncated_response_tail() -> None:
    assert not result_has_error('{"error": "", "response": "truncated...')
    assert result_has_error('{"error": "HTTP 500", "response": "truncated...')
    assert not result_has_error('{"error": null, "response": "x"}')
    assert result_has_error('{"error": {"type": "timeout"}, "response": "x"}')


def test_post_transition_events_includes_current_event_and_bounds_horizon() -> None:
    trajectory = [_example("t", step, 0) for step in range(20)]
    events = post_transition_events(trajectory, 19, max_horizon=16)
    assert len(events) == 16
    assert events[0]["step_index"] == 4
    assert events[-1]["step_index"] == 19


def test_trajectory_split_is_disjoint_deterministic_and_exact_size() -> None:
    trajectories: list[list[ProbeExample]] = []
    for stratum, count in ((0, 300), (1, 150), (2, 112)):
        for index in range(count):
            trajectories.append([_example(f"s{stratum}-{index}", 0, stratum)])
    first = trajectory_stratified_split(trajectories, seed=17)
    second = trajectory_stratified_split(trajectories, seed=17)
    assert first == second
    assert len(first) == 562
    assert list(first.values()).count("train") == 337
    assert list(first.values()).count("dev") == 113
    assert list(first.values()).count("test") == 112
    report = split_report(trajectories, first)
    assert report["train"]["trajectory_count"] == 337
    assert report["dev"]["trajectory_count"] == 113
    assert report["test"]["trajectory_count"] == 112


def test_load_probe_rows_rejects_misaligned_result(tmp_path: Path) -> None:
    row = {
        "benchmark": "toolbench_g3",
        "task_id": "t::0",
        "trajectory_id": "t",
        "step_index": 0,
        "state_text": "state",
        "skill_id": "skill/a",
        "action_text": "act",
        "actual_result_text": '{"error": ""}',
        "actual_result_executed": True,
        "actual_result_skill_id": "skill/b",
        "result_event_skill_id": "skill/a",
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aligned"):
        load_toolbench_probe_trajectories(path, enforce_canonical=False)


def test_macro_f1_and_binary_auroc_known_values() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])
    assert macro_f1(labels, predictions, num_classes=2) == pytest.approx(
        (2.0 / 3.0 + 0.8) / 2.0
    )
    assert binary_auroc(labels, torch.tensor([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert binary_auroc(labels, torch.ones(4)) == pytest.approx(0.5)


def test_regularized_linear_probe_fits_separable_features() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(120, 8, generator=generator)
    labels = (features[:, 0] + 0.7 * features[:, 1] > 0).to(torch.long)
    split_names = ["train"] * 72 + ["dev"] * 24 + ["test"] * 24
    trajectory_ids = [f"trajectory-{index}" for index in range(120)]
    metrics, predictions = fit_regularized_linear_probe(
        features=features,
        labels=labels,
        split_names=split_names,
        trajectory_ids=trajectory_ids,
        num_classes=2,
        device="cpu",
        l2_grid=(1.0e-3, 1.0e-2),
        max_iter=40,
        bootstrap_draws=50,
        bootstrap_seed=5,
    )
    assert metrics["test_macro_f1"] > 0.8
    assert metrics["test_auroc"] > 0.9
    assert predictions["probabilities"].shape == (24, 2)
