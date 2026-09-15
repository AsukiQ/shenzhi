from __future__ import annotations

from copy import deepcopy

import pytest

from clstr.matched_multibench_data import (
    LOCKED_SPLIT_SCHEMA,
    build_tau2_split_manifest,
    build_toolsandbox_split_manifest,
    canonical_digest,
    canonical_toolsandbox_family_name,
    route_rows_to_training_trajectories,
    tau2_row_split,
)


def _toolsandbox_row(name: str, step: int = 0, target: str = "tool") -> dict:
    return {
        "task_id": f"toolsandbox/{name}::{step}",
        "trajectory_id": f"toolsandbox/{name}",
        "step_index": step,
        "state_text_current": f"user request: {name}",
        "target_action_text": f"tool: {target}\narguments: {{}}",
        "skill_id": "toolsandbox/__start__",
        "next_skill_id": f"toolsandbox/{target}",
        "candidate_next_skill_ids": [f"toolsandbox/{target}", "toolsandbox/other"],
        "split_semantic_text": f"user request: {name}",
        "provenance": {
            "scenario_name": name,
            "scenario_group": "multiple_tool_call",
        },
    }


def test_toolsandbox_family_canonicalization_collapses_known_variants():
    variants = {
        "find_days_till_holiday",
        "find_days_till_holiday_alt",
        "find_days_till_holiday_wifi_off_multiple_user_turn",
    }
    assert {canonical_toolsandbox_family_name(name) for name in variants} == {
        "find_days_till_holiday"
    }
    reminder_variants = {
        "search_reminder_with_recency_upcoming_implicit",
        "search_reminder_with_creation_recency_yesterday",
    }
    assert {
        canonical_toolsandbox_family_name(name) for name in reminder_variants
    } == {"search_reminder_with_recency"}


def test_toolsandbox_manifest_keeps_one_family_in_one_split():
    names = [
        "add_reminder_content_and_date_and_time",
        "convert_currency",
        "find_address_with_lat_lon",
        "find_current_city_low_battery_mode",
        "find_days_till_holiday",
        "find_distance_with_location_name",
        "find_phone_number_with_location_name",
        "find_stock_symbol_with_company_name",
        "find_temperature",
        "find_temperature_f_with_location",
        "find_thanksgiving_timestamp",
        "get_cellular",
        "get_wifi",
        "modify_contact_with_message_recency",
        "remove_contact_by_phone",
        "search_message_with_recency_latest",
        "search_name_with_relationship",
        "search_phone_number_with_name",
        "search_relationship_with_phone_number",
        "search_reminder_with_recency_upcoming",
        "search_sender_phone_number_with_content",
        "send_message_with_contact_content_cellular_off",
        "update_contact_relationship_with_relationship",
        "find_days_till_holiday_alt",
    ]
    manifest = build_toolsandbox_split_manifest(
        [_toolsandbox_row(name) for name in names]
    )
    assert manifest["scenario_to_split"]["find_days_till_holiday"] == manifest[
        "scenario_to_split"
    ]["find_days_till_holiday_alt"]
    assert all(manifest["splits"][split] for split in ("train", "dev", "test"))


def _tau2_row(domain: str, task: str) -> dict:
    return {"domain": domain, "provenance": {"domain": domain, "raw_task_id": task}}


def test_tau2_manifest_preserves_official_test_and_splits_train_only():
    train = [
        _tau2_row(domain, f"train-{index}")
        for domain in ("airline", "retail", "telecom")
        for index in range(30)
    ]
    test = [
        _tau2_row(domain, f"test-{index}")
        for domain in ("airline", "retail", "telecom")
        for index in range(10)
    ]
    manifest = build_tau2_split_manifest(train_rows=train, test_rows=test)
    assert tau2_row_split(manifest, test[0]) == "test"
    assert set(manifest["splits"]["test"]) == {
        f"{domain}/test-{index}"
        for domain in ("airline", "retail", "telecom")
        for index in range(10)
    }


def test_tau2_manifest_rejects_official_train_test_overlap():
    row = _tau2_row("airline", "shared")
    with pytest.raises(ValueError, match="overlap"):
        build_tau2_split_manifest(train_rows=[row], test_rows=[deepcopy(row)])


def test_tau2_manifest_can_protect_official_tasks_without_successful_rows():
    train_rows = [
        _tau2_row(domain, "observed-train")
        for domain in ("airline", "retail", "telecom")
    ]
    test_rows = [
        _tau2_row(domain, "observed-test")
        for domain in ("airline", "retail", "telecom")
    ]
    manifest = build_tau2_split_manifest(
        train_rows=train_rows,
        test_rows=test_rows,
        train_task_keys=[
            f"{domain}/missing-success-train-{index}"
            for domain in ("airline", "retail", "telecom")
            for index in range(30)
        ],
        test_task_keys=[
            f"{domain}/missing-success-test"
            for domain in ("airline", "retail", "telecom")
        ],
    )
    assert all(
        f"{domain}/missing-success-test" in manifest["splits"]["test"]
        for domain in ("airline", "retail", "telecom")
    )


def test_route_conversion_uses_current_target_and_action_only_memory():
    rows = [
        _toolsandbox_row("scenario", 0, "first"),
        _toolsandbox_row("scenario", 1, "second"),
    ]
    trajectory = "toolsandbox/scenario"
    converted = route_rows_to_training_trajectories(
        rows,
        benchmark="toolsandbox",
        split_by_trajectory={trajectory: "dev"},
        group_identity_by_trajectory={
            trajectory: canonical_digest({"family": "scenario"})
        },
        split_manifest_sha256=canonical_digest({"manifest": "fixture"}),
    )
    assert [row["skill_id"] for row in converted] == [
        "toolsandbox/first",
        "toolsandbox/second",
    ]
    assert converted[0]["next_skill_id"] == "toolsandbox/second"
    assert converted[1]["next_skill_id"] == ""
    assert all(row["history_text"] == "" for row in converted)
    assert all(row["next_observation_text"] == "" for row in converted)
    assert all(row["observation_source"] == "action_only_no_tool_result" for row in converted)
    assert all(row["locked_split_schema_version"] == LOCKED_SPLIT_SCHEMA for row in converted)


def test_route_conversion_preserves_verified_actual_result_and_multi_positive_targets():
    rows = [
        {
            **_toolsandbox_row("scenario", 0, "first"),
            "next_observation_text": "actual result",
            "observation_source": "tau2_official_successful_rollout_tool_result",
            "actual_result_event_id": "call-1",
            "equivalent_next_skill_ids": ["toolsandbox/first", "toolsandbox/other"],
            "current_state_components": {
                "goal_text": "stable visible goal",
                "task_text": "domain",
                "current_observation_text": "current visible turn",
            },
            "provenance": {
                **_toolsandbox_row("scenario")["provenance"],
                "source_id": "tau2_official_successful_rollout_v1",
            },
        },
        _toolsandbox_row("scenario", 1, "second"),
    ]
    trajectory = "toolsandbox/scenario"
    converted = route_rows_to_training_trajectories(
        rows,
        benchmark="tau2",
        split_by_trajectory={trajectory: "train"},
        group_identity_by_trajectory={
            trajectory: canonical_digest({"family": "scenario"})
        },
        split_manifest_sha256=canonical_digest({"manifest": "fixture"}),
    )

    assert converted[0]["next_observation_text"] == "actual result"
    assert converted[0]["observation_source"] == (
        "tau2_official_successful_rollout_tool_result"
    )
    assert converted[0]["matched_data_protocol"] == "current_action_actual_result_v1"
    assert converted[0]["training_target_semantics"] == (
        "current_skill_before_memory_update_v1"
    )
    assert converted[0]["equivalent_positive_semantics"] == (
        "current_target_equivalents_v1"
    )
    assert converted[0]["goal_text"] == "stable visible goal"
    assert converted[0]["equivalent_next_skill_ids"] == [
        "toolsandbox/first",
        "toolsandbox/other",
    ]


def test_route_conversion_supports_disjoint_test_materialization_without_training_locks():
    rows = [
        _toolsandbox_row("scenario", 0, "first"),
        _toolsandbox_row("scenario", 1, "second"),
    ]
    converted = route_rows_to_training_trajectories(
        rows,
        benchmark="tau2",
        split_by_trajectory={},
        group_identity_by_trajectory={},
        split_manifest_sha256=canonical_digest({"manifest": "test-fixture"}),
        evaluation_split="test",
    )
    assert [row["skill_id"] for row in converted] == [
        "toolsandbox/first",
        "toolsandbox/second",
    ]
    assert all(row["provenance"]["matched_split"] == "test" for row in converted)
    assert all("locked_data_split" not in row for row in converted)
    assert all("locked_split_group_identity" not in row for row in converted)


def test_route_conversion_rejects_non_test_evaluation_split():
    with pytest.raises(ValueError, match="evaluation split must be test"):
        route_rows_to_training_trajectories(
            [_toolsandbox_row("scenario")],
            benchmark="tau2",
            split_by_trajectory={},
            group_identity_by_trajectory={},
            split_manifest_sha256=canonical_digest({"manifest": "fixture"}),
            evaluation_split="dev",
        )


def test_toolsandbox_manifest_accepts_multiple_dag_routes_for_one_scenario():
    first = [_toolsandbox_row("scenario", step, target) for step, target in enumerate(("a", "b"))]
    second = [_toolsandbox_row("scenario", step, target) for step, target in enumerate(("b", "a"))]
    for route_index, route in enumerate((first, second)):
        for row in route:
            row["trajectory_id"] = f"toolsandbox/scenario::route{route_index:03d}"
            row["task_id"] = f"{row['trajectory_id']}::{row['step_index']}"
    filler = [
        _toolsandbox_row(f"family-{index}")
        for index in range(30)
    ]
    manifest = build_toolsandbox_split_manifest([*first, *second, *filler])
    assert manifest["scenario_records"]["scenario"]["route_variant_count"] == 2
    assert manifest["scenario_records"]["scenario"]["route_row_count"] == 4
