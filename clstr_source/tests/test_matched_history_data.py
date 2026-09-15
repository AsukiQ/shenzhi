from __future__ import annotations

import hashlib
import json

import pytest

from clstr.history_channel import canonical_causal_event
from clstr.matched_history_data import (
    MATCHED_HISTORY_DATA_SCHEMA,
    MatchedHistoryTrajectorySkip,
    history_events_for_decision,
    normalized_history_benchmark,
    prepare_matched_history_trajectory,
    serialize_factual_history,
)


SKILLS = {"a", "b", "c"}
CATALOGS = {
    "pool": {
        "inventory_catalog_digest": "pool-digest",
        "runtime_visible_skill_ids": sorted(SKILLS),
    }
}


def _digest(events: list[dict]) -> str:
    payload = json.dumps(
        events,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row(step: int, target: str, *, source_key: str = "benchmark") -> dict:
    prior = [
        canonical_causal_event(
            {
                "step_index": index,
                "skill_id": ("a", "b", "c")[index],
                "action_text": f"action-{index}",
                "result_text": f"result-{index}",
                "result_executed": True,
            }
        )
        for index in range(step)
    ]
    row = {
        source_key: "toolbench_g3",
        "trajectory_id": "trajectory-1",
        "data_split": "train",
        "step_index": step,
        "decision_step_index": step,
        "state_text_current": f"state-{step}",
        "skill_id": target,
        "target_skill_id": target,
        "action_text": f"action-{step}",
        "actual_result_text": f"result-{step}",
        "actual_result_executed": True,
        "equivalent_next_skill_ids": [target],
        "capabilities": {
            "actual_execution_result": True,
            "ordered_next_tool": step > 0,
        },
        "runtime_visible_catalog_id": "pool",
        "inventory_catalog_digest": "pool-digest",
        "causal_prefix_events": prior,
        "causal_prefix_event_count": len(prior),
        "causal_prefix_sha256": _digest(prior),
    }
    return row


def test_source_normalization_accepts_unified_and_benchmark_fields() -> None:
    assert normalized_history_benchmark({"_unified_source": "toolbench_g3"}) == "toolbench_g3"
    assert normalized_history_benchmark({"benchmark": "tau2"}) == "tau2"


def test_preparation_restores_only_strictly_earlier_factual_events() -> None:
    rows = [_row(0, "a"), _row(1, "b"), _row(2, "c")]
    compact, quarantine = prepare_matched_history_trajectory(
        rows,
        selected_skill_ids=SKILLS,
        catalogs=CATALOGS,
    )
    assert quarantine == {}
    assert all(row["schema_version"] == MATCHED_HISTORY_DATA_SCHEMA for row in compact)
    events = history_events_for_decision(compact, 2, max_horizon=16)
    assert [event["step_index"] for event in events] == [0, 1]
    text = serialize_factual_history(events)
    assert "state-0" in text and "state-1" in text
    assert "action-0" in text and "action-1" in text
    assert "result-0" in text and "result-1" in text
    assert "state-2" not in text
    assert "action-2" not in text
    assert "result-2" not in text


def test_history_horizon_keeps_the_latest_events() -> None:
    skills = {f"s{index}" for index in range(20)}
    catalogs = {
        "pool": {
            "inventory_catalog_digest": "pool-digest",
            "runtime_visible_skill_ids": sorted(skills),
        }
    }
    rows = []
    for step in range(20):
        prior = [
            canonical_causal_event(
                {
                    "step_index": index,
                    "skill_id": f"s{index}",
                    "action_text": f"action-{index}",
                    "result_text": f"result-{index}",
                    "result_executed": True,
                }
            )
            for index in range(step)
        ]
        row = _row(0, "a")
        row.update(
            {
                "trajectory_id": "long",
                "step_index": step,
                "decision_step_index": step,
                "state_text_current": f"state-{step}",
                "skill_id": f"s{step}",
                "target_skill_id": f"s{step}",
                "action_text": f"action-{step}",
                "actual_result_text": f"result-{step}",
                "equivalent_next_skill_ids": [f"s{step}"],
                "capabilities": {
                    "actual_execution_result": True,
                    "ordered_next_tool": step > 0,
                },
                "causal_prefix_events": prior,
                "causal_prefix_event_count": len(prior),
                "causal_prefix_sha256": _digest(prior),
            }
        )
        rows.append(row)
    compact, _ = prepare_matched_history_trajectory(
        rows,
        selected_skill_ids=skills,
        catalogs=catalogs,
    )
    events = history_events_for_decision(compact, 19, max_horizon=16)
    assert [event["step_index"] for event in events] == list(range(3, 19))


def test_prefix_content_mismatch_fails_closed() -> None:
    rows = [_row(0, "a"), _row(1, "b")]
    rows[1]["causal_prefix_events"][0]["action_text"] = "future-corruption"
    rows[1]["causal_prefix_sha256"] = _digest(rows[1]["causal_prefix_events"])
    with pytest.raises(ValueError, match="differs from its earlier"):
        prepare_matched_history_trajectory(
            rows,
            selected_skill_ids=SKILLS,
            catalogs=CATALOGS,
        )


def test_prefix_digest_mismatch_fails_closed() -> None:
    rows = [_row(0, "a"), _row(1, "b")]
    rows[1]["causal_prefix_sha256"] = "bad"
    with pytest.raises(ValueError, match="digest differs"):
        prepare_matched_history_trajectory(
            rows,
            selected_skill_ids=SKILLS,
            catalogs=CATALOGS,
        )


def test_unverified_result_is_quarantined_from_all_history_arms() -> None:
    rows = [_row(0, "a"), _row(1, "b")]
    rows[0]["capabilities"] = {"actual_execution_result": False}
    rows[0]["causal_prefix_events"][0:0] = []
    event = canonical_causal_event(
        {
            "step_index": 0,
            "skill_id": "a",
            "action_text": "action-0",
            "result_text": "",
            "result_executed": False,
        }
    )
    rows[1]["causal_prefix_events"] = [event]
    rows[1]["causal_prefix_event_count"] = 1
    rows[1]["causal_prefix_sha256"] = _digest([event])
    compact, quarantine = prepare_matched_history_trajectory(
        rows,
        selected_skill_ids=SKILLS,
        catalogs=CATALOGS,
    )
    assert quarantine == {"executed_flag_without_actual_execution_capability": 1}
    events = history_events_for_decision(compact, 1)
    assert events[0]["result_text"] == ""
    assert events[0]["result_executed"] is False


def test_verified_but_prefix_hidden_result_stays_hidden() -> None:
    rows = [_row(0, "a"), _row(1, "b")]
    hidden = canonical_causal_event(
        {
            "step_index": 0,
            "skill_id": "a",
            "action_text": "action-0",
            "result_text": "",
            "result_executed": False,
        }
    )
    rows[1]["causal_prefix_events"] = [hidden]
    rows[1]["causal_prefix_event_count"] = 1
    rows[1]["causal_prefix_sha256"] = _digest([hidden])
    compact, quarantine = prepare_matched_history_trajectory(
        rows,
        selected_skill_ids=SKILLS,
        catalogs=CATALOGS,
    )
    assert quarantine == {}
    assert compact[0]["result_text"] == "result-0"
    events = history_events_for_decision(compact, 1)
    assert events[0]["result_text"] == ""
    assert events[0]["result_executed"] is False


def test_noncontiguous_and_singleton_trajectories_are_explicitly_skipped() -> None:
    with pytest.raises(MatchedHistoryTrajectorySkip, match="trajectory_too_short"):
        prepare_matched_history_trajectory(
            [_row(0, "a")],
            selected_skill_ids=SKILLS,
            catalogs=CATALOGS,
        )
    rows = [_row(0, "a"), _row(1, "b")]
    rows[1]["step_index"] = 2
    rows[1]["decision_step_index"] = 2
    with pytest.raises(
        MatchedHistoryTrajectorySkip,
        match="trajectory_not_contiguous_from_initial_decision",
    ):
        prepare_matched_history_trajectory(
            rows,
            selected_skill_ids=SKILLS,
            catalogs=CATALOGS,
        )


def test_compact_event_text_matches_persisted_single_line_canonicalization() -> None:
    rows = [_row(0, "a"), _row(1, "b")]
    rows[0]["action_text"] = "action-0\nwith spacing"
    rows[0]["actual_result_text"] = "result-0\nwith spacing"
    event = canonical_causal_event(
        {
            "step_index": 0,
            "skill_id": "a",
            "action_text": rows[0]["action_text"],
            "result_text": rows[0]["actual_result_text"],
            "result_executed": True,
        }
    )
    rows[1]["causal_prefix_events"] = [event]
    rows[1]["causal_prefix_event_count"] = 1
    rows[1]["causal_prefix_sha256"] = _digest([event])
    compact, _ = prepare_matched_history_trajectory(
        rows,
        selected_skill_ids=SKILLS,
        catalogs=CATALOGS,
    )
    events = history_events_for_decision(compact, 1)
    assert events[0]["action_text"] == "action-0 with spacing"
    assert events[0]["result_text"] == "result-0 with spacing"
