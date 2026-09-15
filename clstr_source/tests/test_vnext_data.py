from __future__ import annotations

import pytest
import torch

from clstr.vnext_data import (
    annotate_result_visibility,
    cross_legal_branch_pair,
    materialize_semantic_views,
    result_visibility,
    runtime_visible_mask,
    runtime_visible_skill_id_set,
    runtime_visible_skill_ids,
)


def test_vnext_inventory_rejects_legacy_fail_open_aliases() -> None:
    with pytest.raises(ValueError, match="runtime_visible_skill_ids"):
        runtime_visible_skill_ids({"candidate_next_skill_ids": ["a", "b"]})


def test_vnext_inventory_accepts_hashed_public_global_catalog() -> None:
    assert runtime_visible_skill_ids(
        {
            "inventory_protocol": "public_global",
            "inventory_catalog_digest": "sha256:catalog",
            "public_global_catalog_skill_ids": ["a", "b", "a"],
        }
    ) == ["a", "b"]


def test_vnext_inventory_resolves_hashed_catalog_reference() -> None:
    catalogs = {
        "global": {
            "inventory_catalog_digest": "sha256:catalog",
            "runtime_visible_skill_ids": ["a", "b"],
        }
    }
    assert runtime_visible_skill_ids(
        {
            "runtime_visible_catalog_id": "global",
            "inventory_catalog_digest": "sha256:catalog",
        },
        catalogs,
    ) == ["a", "b"]


def test_vnext_inventory_reuses_catalog_membership_and_column_indices() -> None:
    class CountingSkillMap(dict):
        def __init__(self, values):
            super().__init__(values)
            self.getitem_count = 0

        def __getitem__(self, key):
            self.getitem_count += 1
            return super().__getitem__(key)

    catalogs = {
        "global": {
            "inventory_catalog_digest": "sha256:global",
            "runtime_visible_skill_ids": ["a", "b", "c"],
        }
    }
    rows = [
        {
            "runtime_visible_catalog_id": "global",
            "inventory_catalog_digest": "sha256:global",
        },
        {
            "runtime_visible_catalog_id": "global",
            "inventory_catalog_digest": "sha256:global",
        },
    ]
    first_set = runtime_visible_skill_id_set(rows[0], catalogs)
    assert runtime_visible_skill_id_set(rows[1], catalogs) is first_set

    skill_map = CountingSkillMap({"a": 0, "b": 1, "c": 2, "d": 3})
    expected = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    assert torch.equal(
        runtime_visible_mask(rows, skill_map, inventory_catalogs=catalogs),
        expected,
    )
    assert skill_map.getitem_count == 3
    assert torch.equal(
        runtime_visible_mask(rows, skill_map, inventory_catalogs=catalogs),
        expected,
    )
    assert skill_map.getitem_count == 3


def test_vnext_inventory_groups_equal_direct_inventories_without_semantic_change() -> None:
    rows = [
        {"runtime_visible_skill_ids": ["b", "a", "b"]},
        {"runtime_visible_skill_ids": ["b", "a"]},
    ]
    mask = runtime_visible_mask(rows, {"a": 0, "b": 1, "c": 2})
    assert torch.equal(mask, torch.tensor([[True, True, False], [True, True, False]]))


def test_result_visibility_separates_duplicate_and_novel_evidence() -> None:
    duplicate = result_visibility(
        "reservation Q69X3R confirmed",
        "observation: reservation Q69X3R confirmed",
    )
    novel = result_visibility(
        "reservation Q69X3R confirmed",
        "observation: waiting for tool response",
    )
    assert duplicate["result_visible_in_next_state"] is True
    assert duplicate["result_novel_for_memory"] is False
    assert novel["result_visible_in_next_state"] is False
    assert novel["result_novel_for_memory"] is True


def test_result_visibility_treats_high_overlap_paraphrase_as_current_evidence() -> None:
    visibility = result_visibility(
        "reservation Q69X3R confirmed for Alice on flight 17",
        "observation Alice reservation on flight 17 was confirmed successfully code Q69X3R",
    )
    assert visibility["result_text_containment_match"] is False
    assert visibility["result_overlap_score"] >= 0.8
    assert visibility["result_visible_in_next_state"] is True
    assert visibility["result_novel_for_memory"] is False


def test_result_visibility_annotation_finds_delayed_retention_step() -> None:
    rows = [
        {
            "trajectory_id": "t",
            "step_index": 0,
            "goal_text": "reserve",
            "next_observation_text": "reservation Q69X3R confirmed",
            "observation_source": "actual_tool_result",
        },
        {
            "trajectory_id": "t",
            "step_index": 1,
            "goal_text": "reserve",
            "current_observation_text": "reservation Q69X3R confirmed",
        },
        {
            "trajectory_id": "t",
            "step_index": 2,
            "goal_text": "reserve",
            "current_observation_text": "confirmation screen closed",
        },
    ]
    annotated = annotate_result_visibility(rows)
    assert annotated[0]["result_visible_in_next_state"] is True
    assert annotated[0]["first_future_step_without_result_visibility"] == 2


def test_cross_legal_branch_pair_requires_both_targets_in_both_pools() -> None:
    row_a = {
        "target_skill_id": "a",
        "runtime_visible_skill_ids": ["a", "b", "c"],
    }
    row_b = {
        "target_skill_id": "b",
        "runtime_visible_skill_ids": ["a", "b", "d"],
    }
    assert cross_legal_branch_pair(row_a, row_b)["cross_legal"] is True
    row_b["runtime_visible_skill_ids"] = ["b", "d"]
    assert cross_legal_branch_pair(row_a, row_b)["cross_legal"] is False


def test_semantic_views_separate_no_call_and_capability_rows() -> None:
    rows = [
        {
            "task_id": "tool-row",
            "trajectory_id": "tool",
            "step_index": 1,
            "goal_text": "g",
            "skill_id": "a",
            "next_skill_id": "b",
            "next_observation_text": "tool result",
            "observation_source": "actual_tool_result",
            "runtime_visible_skill_ids": ["a", "b", "c"],
            "required_tool_set_skill_ids": ["a", "b"],
            "capabilities": {
                "required_tool_set": True,
                "current_state_route_set": True,
                "ordered_next_tool": True,
                "actual_execution_result": True,
            },
        },
        {
            "task_id": "no-call",
            "trajectory_id": "none",
            "step_index": 0,
            "goal_text": "no tool needed",
            "route_target": "STOP",
        },
    ]
    views = materialize_semantic_views(rows)
    assert views.report["status"] == "ok"
    assert len(views.retrieval_rows) == 1
    assert len(views.static_route_rows) == 1
    assert len(views.ordered_transition_rows) == 1
    assert len(views.result_correction_rows) == 1
    assert len(views.no_call_rows) == 1


def test_semantic_views_require_explicit_current_state_route_capability() -> None:
    base = {
        "task_id": "ambiguous",
        "trajectory_id": "ambiguous",
        "step_index": 0,
        "goal_text": "g",
        "runtime_visible_skill_ids": ["a", "b", "c"],
        "current_state_route_set_skill_ids": ["a", "b"],
        "capabilities": {"current_state_route_set": False},
    }
    assert materialize_semantic_views([base]).static_route_rows == []
    enabled = dict(base)
    enabled["capabilities"] = {"current_state_route_set": True}
    views = materialize_semantic_views([enabled])
    assert len(views.static_route_rows) == 1


def test_semantic_views_reject_static_route_positive_outside_inventory() -> None:
    views = materialize_semantic_views(
        [
            {
                "task_id": "bad-static",
                "trajectory_id": "bad-static",
                "step_index": 0,
                "goal_text": "g",
                "runtime_visible_skill_ids": ["a", "c"],
                "current_state_route_set_skill_ids": ["a", "b"],
                "capabilities": {"current_state_route_set": True},
            }
        ]
    )
    assert views.report["status"] == "action_required"
    assert views.static_route_rows == []
