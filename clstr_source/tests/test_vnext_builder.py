from __future__ import annotations

import json

import pytest

from clstr.matched_multibench_data import (
    LOCKED_SPLIT_SCHEMA,
    canonical_digest,
)

from scripts.build_clstr_vnext_views import (
    _anti_join_dynamic_holdout,
    _bind_clean_preflight,
    _build_catalogs,
    _causal_retrieval_release_blockers,
    _canonical_trajectory_rows,
    _deduplicate_semantic_rows,
    _exclude_unbound_trajectory_retrieval_rows,
    _group_static_route_rows,
    _group_retrieval_rows,
    _mine_exact_causal_pairs,
    _materialize_dynamic_skill_holdout,
    _file_digest,
    _split_semantic_rows,
    _shortcut_predictability_report,
    _verified_robust_prefix_kind,
    _verified_unordered_remaining_tool_pairs,
    _view_rows,
)


def _skills():
    return [
        {"skill_id": "traject/a"},
        {"skill_id": "traject/b"},
        {"skill_id": "traject/c"},
        {"skill_id": "alfworld/x"},
    ]


def _traject_row(step: int, skill: str, next_skill: str, *, trajectory_type: str):
    previous = "<empty>" if step == 0 else ", ".join(["old"] * step)
    return {
        "benchmark": "traject_bench",
        "task_id": f"task-{trajectory_type}",
        "trajectory_id": f"trajectory-{trajectory_type}",
        "step_index": step,
        "goal_text": "complete the three requests",
        "state_text": (
            "goal: complete the three requests\n"
            f"trajectory_type: {trajectory_type}\n"
            f"previous_tools: {previous}"
        ),
        "history_text": previous if step else "",
        "skill_id": skill,
        "next_skill_id": next_skill,
        "action_text": f"call {skill}",
        "next_observation_text": f"result {step}",
        "observation_source": "actual_tool_result",
        "tool_inventory_skill_ids": ["traject/a", "traject/b", "traject/c"],
        "provenance": {"trajectory_type": trajectory_type},
    }


def test_builder_groups_history_aliased_sequential_states_as_route_multi_positive() -> None:
    skills = _skills()
    by_prefix = {
        "traject": ["traject/a", "traject/b", "traject/c"],
        "alfworld": ["alfworld/x"],
    }
    catalogs = _build_catalogs(skills, by_prefix)
    raw = [
        _traject_row(0, "traject/a", "traject/b", trajectory_type="sequential"),
        _traject_row(1, "traject/b", "traject/c", trajectory_type="sequential"),
        _traject_row(2, "traject/c", "", trajectory_type="sequential"),
    ]
    trajectories, _report = _canonical_trajectory_rows(
        raw,
        catalogs,
        {row["skill_id"] for row in skills},
    )
    static_rows, static_report = _group_static_route_rows(trajectories, catalogs)
    assert len(static_rows) == 1
    assert static_rows[0]["state_text_current"] == "goal: complete the three requests"
    assert static_rows[0]["current_state_route_set_skill_ids"] == [
        "traject/a",
        "traject/b",
        "traject/c",
    ]
    assert static_rows[0]["current_state_route_factual_target_counts"] == {
        "traject/a": 1,
        "traject/b": 1,
        "traject/c": 1,
    }
    assert static_rows[0]["current_state_route_ambiguous"] is True
    assert static_report["ambiguous_group_count"] == 1


def test_builder_keeps_parallel_traject_rows_retrieval_only() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    raw = [
        _traject_row(0, "traject/a", "traject/b", trajectory_type="parallel"),
        _traject_row(1, "traject/b", "", trajectory_type="parallel"),
    ]
    trajectories, _report = _canonical_trajectory_rows(
        raw,
        catalogs,
        {row["skill_id"] for row in skills},
    )
    static_rows, static_report = _group_static_route_rows(trajectories, catalogs)
    assert static_rows == []
    assert static_report["skipped"]["parallel_retrieval_only"] == 2
    assert not any(row["capabilities"]["ordered_next_tool"] for row in trajectories)


def test_builder_materializes_verified_unordered_remaining_tool_branch_support() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    raw = [
        _traject_row(0, "traject/a", "traject/b", trajectory_type="parallel"),
        _traject_row(1, "traject/b", "traject/c", trajectory_type="parallel"),
        _traject_row(2, "traject/c", "", trajectory_type="parallel"),
    ]
    for row in raw:
        row["provenance"] = {
            "source_id": "traject_bench",
            "trajectory_type": "parallel",
        }
    trajectories, _report = _canonical_trajectory_rows(
        raw,
        catalogs,
        {row["skill_id"] for row in skills},
    )

    support, pairs, report = _verified_unordered_remaining_tool_pairs(
        trajectories,
        catalogs,
        maximum_pairs=10,
    )

    assert len(pairs) == 1
    assert len(support) == 6
    assert report["eligible_candidate_count"] == 1
    assert pairs[0]["source_id"] == "verified_unordered_required_set"
    assert (
        pairs[0]["causal_verification"]["evidence_type"]
        == "verified_unordered_remaining_tool_branch_v1"
    )
    assert {pairs[0]["row_a"]["target_skill_id"], pairs[0]["row_b"]["target_skill_id"]} == {
        "traject/b",
        "traject/c",
    }
    assert all(row["pair_support_only"] for row in support)


def test_builder_binds_source_declared_runtime_inventory_instead_of_global_pool() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    rows, _report = _canonical_trajectory_rows(
        [_traject_row(0, "traject/a", "", trajectory_type="sequential")],
        catalogs,
        {row["skill_id"] for row in skills},
    )
    catalog_id = rows[0]["runtime_visible_catalog_id"]
    assert catalog_id.startswith("runtime_declared_")
    assert catalogs[catalog_id]["runtime_visible_skill_ids"] == [
        "traject/a",
        "traject/b",
        "traject/c",
    ]
    assert rows[0]["inventory_source"] == "source_declared_runtime_tool_inventory"


def test_builder_rejects_unknown_runtime_inventory_skill() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    row = _traject_row(0, "traject/a", "", trajectory_type="sequential")
    row["tool_inventory_skill_ids"] = ["traject/a", "unknown/x"]
    try:
        _canonical_trajectory_rows(
            [row],
            catalogs,
            {item["skill_id"] for item in skills},
        )
    except ValueError as exc:
        assert "unknown skills" in str(exc)
    else:
        raise AssertionError("unknown runtime inventory must fail closed")


def test_builder_rejects_duplicate_trajectory_step_identity() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    row_a = _traject_row(0, "traject/a", "", trajectory_type="sequential")
    row_b = dict(row_a)
    try:
        _canonical_trajectory_rows(
            [row_a, row_b],
            catalogs,
            {row["skill_id"] for row in skills},
        )
    except ValueError as exc:
        assert "duplicate step index" in str(exc)
    else:
        raise AssertionError("duplicate trajectory/step identity must fail closed")


def test_builder_namespaces_trajectory_ids_reused_across_sources() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    row_a = _traject_row(0, "traject/a", "", trajectory_type="sequential")
    row_b = dict(row_a)
    row_a["provenance"] = {"source_id": "source-a", "trajectory_type": "sequential"}
    row_b["provenance"] = {"source_id": "source-b", "trajectory_type": "sequential"}

    rows, report = _canonical_trajectory_rows(
        [row_a, row_b],
        catalogs,
        {row["skill_id"] for row in skills},
    )

    assert len(rows) == 2
    assert len({row["trajectory_id"] for row in rows}) == 2
    assert {row["source_trajectory_id"] for row in rows} == {
        "trajectory-sequential"
    }
    assert all(row["trajectory_id_namespaced"] for row in rows)
    assert report["trajectory_id_namespace_collision_count"] == 1
    assert report["trajectory_rows_namespaced"] == 2


def test_builder_removes_required_tools_from_explicit_negatives() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    rows = _group_retrieval_rows(
        [
            {
                "source": "fixture",
                "query_id": "q",
                "query_text": "find the tool",
                "positive_skill_ids": ["traject/a"],
                "negative_skill_ids": ["traject/a", "traject/b"],
            }
        ],
        catalogs,
    )
    assert rows[0]["required_tool_set_skill_ids"] == ["traject/a"]
    assert rows[0]["explicit_negative_skill_ids"] == ["traject/b"]
    assert rows[0]["removed_positive_negative_collision_count"] == 1


def test_builder_keeps_query_alias_rows_and_removes_legacy_history_suffix() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    rows = _group_retrieval_rows(
        [
            {
                "source": "trajectory_derived_traject_bench",
                "query_id": "q-alias",
                "query": "find the tools\nprevious_tools: traject/a",
                "positive_skill_id": "traject/b",
            }
        ],
        catalogs,
    )

    assert len(rows) == 1
    assert rows[0]["state_text_current"] == "goal: find the tools"
    assert "previous_tools" not in rows[0]["state_text_current"]
    assert rows[0]["state_text_causal"] == (
        "find the tools\nprevious_tools: traject/a"
    )
    assert rows[0]["causal_state_contract"] == "clstr_source_causal_query_v1"
    assert rows[0]["required_tool_set_skill_ids"] == ["traject/b"]


def test_builder_cross_binds_next_target_to_successor_causal_decision() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    trajectories, _report = _canonical_trajectory_rows(
        [
            _traject_row(0, "traject/a", "traject/b", trajectory_type="sequential"),
            _traject_row(1, "traject/b", "", trajectory_type="sequential"),
        ],
        catalogs,
        {row["skill_id"] for row in skills},
    )
    rows = _group_retrieval_rows(
        [
            {
                "source": "trajectory_derived_traject_bench",
                "query_id": "trajectory-sequential::0::next",
                "query": "goal: complete the three requests\nprevious_action: call a",
                "positive_skill_id": "traject/b",
                "provenance": {
                    "trajectory_id": "trajectory-sequential",
                    "step_index": 0,
                    "target": "next",
                },
            }
        ],
        catalogs,
        trajectories,
    )
    assert len(rows) == 1
    assert rows[0]["decision_step_index"] == 1
    assert rows[0]["retrieval_temporal_target"] == "next"
    assert rows[0]["causal_query_materialization_source"] == (
        "cross_bound_canonical_trajectory_decision"
    )
    assert rows[0]["causal_prefix_event_count"] == 1
    assert rows[0]["causal_prefix_events"][0]["skill_id"] == "traject/a"
    assert "event[0].skill_id: traject/a" in rows[0]["state_text_causal"]
    assert "event[1].skill_id: traject/b" not in rows[0]["state_text_causal"]


def test_builder_blocks_unbound_trajectory_like_retrieval_release() -> None:
    assert _causal_retrieval_release_blockers(
        {"trajectory_like_unbound_retrieval_row_count": 1}
    ) == ["trajectory_like_retrieval_row_lacks_canonical_cross_binding"]
    assert _causal_retrieval_release_blockers(
        {"trajectory_like_unbound_retrieval_row_count": 0}
    ) == []


def test_builder_excludes_only_unbound_trajectory_retrieval_rows() -> None:
    kept, excluded = _exclude_unbound_trajectory_retrieval_rows(
        [
            {
                "source": "trajectory_derived_toolbench_g3",
                "query_id": "unbound",
                "causal_query_materialization_source": (
                    "source_declared_retrieval_decision_query"
                ),
            },
            {
                "source": "toolbench_g3",
                "query_id": "ordinary-query",
                "causal_query_materialization_source": (
                    "source_declared_retrieval_decision_query"
                ),
            },
            {
                "source": "trajectory_derived_toolbench_g3",
                "query_id": "bound",
                "causal_query_materialization_source": (
                    "cross_bound_canonical_trajectory_decision"
                ),
            },
        ]
    )
    assert [row["query_id"] for row in kept] == ["ordinary-query", "bound"]
    assert [row["query_id"] for row in excluded] == ["unbound"]
    assert excluded[0]["exclusion_reason"] == (
        "trajectory_retrieval_lacks_canonical_cross_binding"
    )


def test_builder_mines_only_evidence_bearing_exact_alias_branch_pairs() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    a0 = _traject_row(0, "traject/c", "traject/a", trajectory_type="sequential")
    a1 = _traject_row(1, "traject/a", "traject/b", trajectory_type="sequential")
    a2 = _traject_row(2, "traject/b", "", trajectory_type="sequential")
    b0 = _traject_row(0, "traject/b", "traject/a", trajectory_type="sequential")
    b1 = _traject_row(1, "traject/a", "traject/c", trajectory_type="sequential")
    b2 = _traject_row(2, "traject/c", "", trajectory_type="sequential")
    for row in (a0, a1, a2):
        row["task_id"] = "task-a"
        row["trajectory_id"] = "trajectory-a"
    for row in (b0, b1, b2):
        row["task_id"] = "task-b"
        row["trajectory_id"] = "trajectory-b"
    trajectories, _report = _canonical_trajectory_rows(
        [a0, a1, a2, b0, b1, b2],
        catalogs,
        {row["skill_id"] for row in skills},
    )
    views, report = _mine_exact_causal_pairs(trajectories, catalogs)
    assert report["causal_branch_pair_count"] == 1
    pair = views["causal_branch_pairs"][0]
    assert pair["same_prefix_length"] is True
    assert pair["exact_current_state_alias"] is True
    assert pair["history_a_digest"] != pair["history_b_digest"]
    assert pair["causal_label_verified"] is True
    assert (
        pair["causal_verification"]["evidence_type"]
        == "symmetric_target_completion_contrast_v1"
    )
    assert {pair["row_a"]["target_skill_id"], pair["row_b"]["target_skill_id"]} == {
        "traject/b",
        "traject/c",
    }
    assert pair["row_a"]["trajectory_id"] != pair["row_b"]["trajectory_id"]


def test_builder_keeps_unverified_different_choices_as_diagnostics_only() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    a0 = _traject_row(0, "traject/a", "traject/b", trajectory_type="sequential")
    a1 = _traject_row(1, "traject/b", "", trajectory_type="sequential")
    b0 = _traject_row(0, "traject/a", "traject/c", trajectory_type="sequential")
    b0["action_text"] = "different rendering of the same skill"
    b1 = _traject_row(1, "traject/c", "", trajectory_type="sequential")
    for row in (a0, a1):
        row["task_id"] = "task-a"
        row["trajectory_id"] = "trajectory-a"
    for row in (b0, b1):
        row["task_id"] = "task-b"
        row["trajectory_id"] = "trajectory-b"
    trajectories, _report = _canonical_trajectory_rows(
        [a0, a1, b0, b1],
        catalogs,
        {row["skill_id"] for row in skills},
    )
    views, report = _mine_exact_causal_pairs(trajectories, catalogs)
    assert views["causal_branch_pairs"] == []
    assert len(views["matched_history_contrast_pairs"]) == 1
    assert views["matched_history_contrast_pairs"][0]["trainable_causal_label"] is False
    assert report["skipped"]["matched_without_decision_evidence"] == 1


def test_builder_verifies_result_outcome_only_with_same_pre_event_history() -> None:
    skills = _skills()
    catalogs = _build_catalogs(
        skills,
        {"traject": ["traject/a", "traject/b", "traject/c"], "alfworld": ["alfworld/x"]},
    )
    rows = []
    for trajectory_id, final_target, result in (
        ("trajectory-a", "traject/b", "success-result"),
        ("trajectory-b", "traject/c", "failure-result"),
    ):
        row0 = _traject_row(0, "traject/a", "traject/a", trajectory_type="sequential")
        row1 = _traject_row(1, "traject/a", final_target, trajectory_type="sequential")
        row2 = _traject_row(2, final_target, "", trajectory_type="sequential")
        row1["next_observation_text"] = result
        for row in (row0, row1, row2):
            row["task_id"] = trajectory_id
            row["trajectory_id"] = trajectory_id
            row["provenance"] = {
                "trajectory_type": "sequential",
                "source_id": "toolbench_g3",
            }
        rows.extend((row0, row1, row2))
    trajectories, _report = _canonical_trajectory_rows(
        rows,
        catalogs,
        {row["skill_id"] for row in skills},
    )
    views, _report = _mine_exact_causal_pairs(trajectories, catalogs)
    assert len(views["causal_outcome_pairs"]) == 1
    outcome = views["causal_outcome_pairs"][0]
    assert outcome["same_pre_event_history"] is True
    assert (
        outcome["causal_verification"]["evidence_type"]
        == "same_action_different_executed_result_v1"
    )


def test_builder_binds_exact_clean_preflight_source_digests(tmp_path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    skills = root / "skill_pool.jsonl"
    retrieval = root / "retrieval.jsonl"
    trajectories = root / "trajectories.jsonl"
    for path, content in (
        (skills, '{"skill_id":"a"}\n'),
        (retrieval, '{"query_text":"q"}\n'),
        (trajectories, '{"goal_text":"g"}\n'),
    ):
        path.write_text(content, encoding="utf-8")
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "preflight_contract": {
                    "schema_version": "clstr_clean_training_preflight_v2",
                    "structured_current_state_required": True,
                    "protected_near_duplicate_blocking": True,
                },
                "composition": {
                    "data_root": str(root),
                    "files_sha256": {
                        "skill_pool.jsonl": _file_digest(skills),
                        "retrieval.jsonl": _file_digest(retrieval),
                        "trajectories.jsonl": _file_digest(trajectories),
                    },
                },
                "protected_eval_inputs": {"toolbench_eval_trajectories_sha256": "eval"},
            }
        ),
        encoding="utf-8",
    )
    binding = _bind_clean_preflight(
        report_path,
        skills_path=skills,
        retrieval_path=retrieval,
        trajectories_path=trajectories,
    )
    assert binding["status"] == "ok"
    assert binding["protected_eval_inputs"]["toolbench_eval_trajectories_sha256"] == "eval"

    trajectories.write_text('{"goal_text":"changed"}\n', encoding="utf-8")
    try:
        _bind_clean_preflight(
            report_path,
            skills_path=skills,
            retrieval_path=retrieval,
            trajectories_path=trajectories,
        )
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("stale clean preflight must fail closed")


def test_task_group_split_keeps_exact_alias_trajectories_together() -> None:
    trajectories = []
    for trajectory_id in ("ta", "tb"):
        row = _traject_row(1, "traject/b", "", trajectory_type="sequential")
        row["trajectory_id"] = trajectory_id
        row["task_id"] = trajectory_id
        trajectories.append(row)
    retrieval = [
        {"source": "s", "state_text_current": "goal: same", "query_id": "q1"},
        {"source": "s", "state_text_current": "goal: same", "query_id": "q2"},
    ]
    splits, report = _split_semantic_rows(
        trajectories,
        retrieval,
        dev_fraction=0.1,
        seed="fixture",
    )
    assert not (splits["trajectory_train"] and splits["trajectory_dev"])
    assert not (splits["retrieval_train"] and splits["retrieval_dev"])
    assert report["trajectory_group_overlap_count"] == 0
    assert report["retrieval_group_overlap_count"] == 0


def test_task_group_split_respects_locked_manifest_assignments() -> None:
    manifest = canonical_digest({"manifest": "fixture"})
    rows = []
    for trajectory_id, split in (("locked-train", "train"), ("locked-dev", "dev")):
        row = _traject_row(0, "traject/a", "", trajectory_type="sequential")
        row.update(
            {
                "trajectory_id": trajectory_id,
                "task_id": trajectory_id,
                "locked_data_split": split,
                "locked_split_group_identity": canonical_digest(
                    {"group": trajectory_id}
                ),
                "locked_split_manifest_sha256": manifest,
                "locked_split_schema_version": LOCKED_SPLIT_SCHEMA,
            }
        )
        rows.append(row)
    splits, report = _split_semantic_rows(
        rows, [], dev_fraction=0.1, seed="ignored-for-locked"
    )
    assert [row["trajectory_id"] for row in splits["trajectory_train"]] == [
        "locked-train"
    ]
    assert [row["trajectory_id"] for row in splits["trajectory_dev"]] == [
        "locked-dev"
    ]
    assert report["locked_trajectory_row_count"] == 2
    assert report["locked_manifest_sha256s"] == [manifest]


def test_task_group_split_rejects_locked_group_crossing_train_dev() -> None:
    manifest = canonical_digest({"manifest": "fixture"})
    group = canonical_digest({"group": "shared"})
    rows = []
    for index, split in enumerate(("train", "dev")):
        row = _traject_row(index, "traject/a", "", trajectory_type="sequential")
        row.update(
            {
                "trajectory_id": f"trajectory-{index}",
                "task_id": f"trajectory-{index}",
                "locked_data_split": split,
                "locked_split_group_identity": group,
                "locked_split_manifest_sha256": manifest,
                "locked_split_schema_version": LOCKED_SPLIT_SCHEMA,
            }
        )
        rows.append(row)
    with pytest.raises(ValueError, match="crosses train/dev"):
        _split_semantic_rows(rows, [], dev_fraction=0.1, seed="fixture")


def test_dynamic_skill_holdout_excludes_trajectory_skills_and_marks_queries() -> None:
    skills = [
        {"skill_id": "trajectory-skill"},
        {"skill_id": "unseen-a"},
        {"skill_id": "unseen-b"},
    ]
    catalogs = _build_catalogs(skills, {"trajectory-skill": ["trajectory-skill"]})
    catalog = catalogs["public_global_67k"]
    trajectories = [
        {
            "target_skill_id": "trajectory-skill",
            "skill_id": "trajectory-skill",
            "runtime_visible_catalog_id": "public_global_67k",
            "inventory_catalog_digest": catalog["inventory_catalog_digest"],
        }
    ]
    retrieval = [
        {
            "query_id": f"a-{index}",
            "required_tool_set_skill_ids": ["unseen-a"],
            "explicit_negative_skill_ids": [],
            "runtime_visible_catalog_id": "public_global_67k",
            "inventory_catalog_digest": catalog["inventory_catalog_digest"],
        }
        for index in range(3)
    ] + [
        {
            "query_id": f"trajectory-{index}",
            "required_tool_set_skill_ids": ["trajectory-skill"],
            "explicit_negative_skill_ids": ["unseen-a"],
            "runtime_visible_catalog_id": "public_global_67k",
            "inventory_catalog_digest": catalog["inventory_catalog_digest"],
        }
        for index in range(3)
    ]
    heldout_skills, heldout_queries, report = _materialize_dynamic_skill_holdout(
        skills,
        retrieval,
        trajectories,
        fraction=1.0,
        minimum_queries=2,
        maximum_skills=8,
        seed="fixture",
    )
    assert [row["skill_id"] for row in heldout_skills] == ["unseen-a"]
    assert len(heldout_queries) == 3
    assert all(row["excluded_from_dynamic_ablation_training"] for row in heldout_queries)
    assert report["heldout_skill_count"] == 1
    (
        training_skills,
        training_retrieval,
        training_trajectories,
        training_catalogs,
        antijoin_report,
    ) = _anti_join_dynamic_holdout(
        skills,
        retrieval,
        trajectories,
        catalogs,
        heldout_skills,
    )
    assert [row["skill_id"] for row in training_skills] == [
        "trajectory-skill",
        "unseen-b",
    ]
    assert len(training_retrieval) == 3
    assert all(row["required_tool_set_skill_ids"] == ["trajectory-skill"] for row in training_retrieval)
    assert all(row["explicit_negative_skill_ids"] == [] for row in training_retrieval)
    assert training_trajectories[0]["target_skill_id"] == "trajectory-skill"
    assert not any(
        "unseen-a" in catalog_row["runtime_visible_skill_ids"]
        for catalog_row in training_catalogs.values()
    )
    assert antijoin_report["excluded_holdout_query_count"] == 3
    assert antijoin_report["removed_negative_reference_count"] == 3
    assert antijoin_report["leak_count"] == 0


def test_builder_deduplicates_normalized_trajectories_and_retrieval_queries() -> None:
    trajectories = []
    for trajectory_id, goal in (("ta", "Complete Task"), ("tb", " complete   task ")):
        trajectories.append(
            {
                "benchmark": "fixture",
                "trajectory_id": trajectory_id,
                "step_index": 0,
                "goal_text": goal,
                "state_text_current": goal,
                "target_skill_id": "skill-a",
                "action_text": "Call A",
                "actual_result_text": "Done",
                "inventory_catalog_digest": "pool",
            }
        )
    retrieval = [
        {
            "source": "fixture",
            "query_id": query_id,
            "state_text_current": query,
            "required_tool_set_skill_ids": ["skill-a"],
            "inventory_catalog_digest": "pool",
        }
        for query_id, query in (("q1", "Find A"), ("q2", " find   a "))
    ]
    kept_trajectories, kept_retrieval, exclusions, report = (
        _deduplicate_semantic_rows(trajectories, retrieval)
    )
    assert {row["trajectory_id"] for row in kept_trajectories} == {"ta"}
    assert [row["query_id"] for row in kept_retrieval] == ["q1"]
    assert report["normalized_duplicate_trajectory_count"] == 1
    assert report["duplicate_retrieval_row_count"] == 1
    assert {row["exclusion_reason"] for row in exclusions} == {
        "normalized_duplicate_trajectory",
        "normalized_duplicate_retrieval_query",
    }


def test_builder_reports_source_step_only_label_shortcut_on_dev() -> None:
    def row(source: str, step: int, target: str):
        return {
            "benchmark": source,
            "step_index": step,
            "target_skill_id": target,
            "capabilities": {"ordered_next_tool": True},
        }

    train = [
        row("source-a", 0, "a"),
        row("source-a", 1, "b"),
        row("source-b", 0, "c"),
        row("source-b", 1, "a"),
    ]
    dev = list(train)
    report = _shortcut_predictability_report(train, dev)
    assert report["status"] == "ok"
    assert report["metrics"]["source_step"]["accuracy"] == 1.0
    assert report["metrics"]["source_step"]["known_feature_coverage"] == 1.0


def test_builder_requires_executed_verified_recovery_prefix_annotations() -> None:
    verified = {
        "target_skill_id": "skill-a",
        "step_index": 1,
        "capabilities": {"ordered_next_tool": True},
        "robust_prefix_annotation": {
            "executed_error_count": 1,
            "executed_error_step_indices": [0],
            "perturbation_executed": True,
            "resulting_state_verified": True,
            "recovery_target_verified": True,
            "verification_id": "environment-trace-1",
        },
    }
    assert _verified_robust_prefix_kind(verified) == ("one_error_prefix", None)
    views = _view_rows([verified])
    assert len(views["one_error_prefix"]) == 1
    assert views["one_error_prefix"][0]["capabilities"]["verified_robust_prefix"] is True

    incomplete = dict(verified)
    incomplete["robust_prefix_annotation"] = {
        **verified["robust_prefix_annotation"],
        "resulting_state_verified": False,
    }
    kind, downgrade = _verified_robust_prefix_kind(incomplete)
    assert kind is None
    assert downgrade == "incomplete_executed_recovery_verification"
    views = _view_rows([incomplete])
    assert len(views["robust_prefix_downgrade"]) == 1
