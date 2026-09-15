from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import random

import pytest
import torch

from clstr.vnext_stage2_train import (
    _best_positive_rank,
    _batch_memory_at_start,
    _build_ordinary_dev_anchors,
    _causal_route_state_text,
    _choose_family_first_horizon,
    _choose_horizon,
    _empty_gradient_interval,
    _evaluate_stage2_causal_dev,
    _evaluate_stage2_ordinary_dev,
    _family_horizon_support,
    _finalize_gradient_interval,
    _gradient_record_gate,
    _multi_m_coarse_recall_gate,
    _ordinary_metric_report,
    _ordinary_safety_gate,
    _load_causal_pairs,
    _load_robust_prefix_anchors,
    _prepare_trajectories,
    _paired_mean_ci,
    _prepared_prefix_record,
    _required_state_cache_texts,
    _require_selected_skill_prefix,
    _require_weighted_causal_views,
    _sample_segments,
    _select_stage2_validation,
    _stage2_validation_gates,
    _stable_cap_pairs,
    _stable_cap_anchors,
    _trajectory_index,
    _update_gradient_interval,
    train_vnext_stage2,
)


SKILLS = {"a", "b", "c"}
CATALOGS = {
    "pool": {
        "inventory_catalog_digest": "pool-digest",
        "runtime_visible_skill_ids": ["a", "b", "c"],
    }
}


def test_stage2_method_contract_names_recurrent_recall_compression_and_route() -> None:
    source = inspect.getsource(train_vnext_stage2)
    assert "frozen_static_top500_plus_recurrent_extra64" in source
    assert "legal_static_top500_dynamic_top500_diff64_union" in source
    assert "full_pool_extra_candidate_recall_and_query_level_dynamic_expert" in source
    assert '"memory_query_state": "structured_current_only"' in source
    assert '"memory_update_state": "structured_current_only"' in source
    assert '"route_fusion": "soft_train_hard_inference_query_level_selection_between_frozen_static_and_unbounded_dynamic_experts"' in source
    assert '"route_reliability_features": "detached_normalized_route_state_current_state_memory_delta_semantics_plus_static_dynamic_margin_confidence_residual_rms_and_log_normalized_absolute_history_depth"' in source
    assert '"route_reliability_source_labels": False' in source
    assert '"route_selector_reports_soft_probability": True' in source
    assert "mixed_ce_raw_dynamic_ce_no_regret_utility_calibration_plus_raw_causal" in source
    assert "training_teacher_retained_rows" in source
    assert "mixed_ce_raw_dynamic_ce_natural_support_topk_boundary_" in source
    assert '"route_topk_supervision"' in source


def test_stage2_family_horizon_support_preserves_each_family_scale() -> None:
    trajectories = []
    for benchmark, length in (("short", 4), ("long", 8)):
        rows = []
        for step in range(length):
            row = _row(f"{benchmark}-trajectory", step, "a")
            row["benchmark"] = benchmark
            rows.append(row)
        trajectories.append(rows)
    assert _family_horizon_support(
        trajectories,
        max_horizon=16,
    ) == {
        "long": (2, 3, 4, 5, 6, 7, 8),
        "short": (2, 3, 4),
    }


def test_stage2_family_first_horizon_rotates_without_common_minimum_collapse() -> None:
    class FixedRandom:
        def choice(self, values):
            return max(values)

    support = {
        "long": (2, 3, 4, 5, 6, 7, 8),
        "short": (2, 3, 4),
    }
    long_horizon, long_family = _choose_family_first_horizon(
        0.9,
        16,
        FixedRandom(),
        family_horizon_support=support,
        family_position=0,
    )
    short_horizon, short_family = _choose_family_first_horizon(
        0.9,
        16,
        FixedRandom(),
        family_horizon_support=support,
        family_position=1,
    )
    assert (long_family, long_horizon) == ("long", 8)
    assert (short_family, short_horizon) == ("short", 4)
    assert _choose_horizon(0.9, 16, FixedRandom()) == 16


def test_stage2_route_mixture_receives_absolute_history_depth_everywhere() -> None:
    source = Path(__file__).parents[1].joinpath(
        "clstr", "vnext_stage2_train.py"
    ).read_text(encoding="utf-8")
    assert 'float(pair["_index"])' in source
    assert '[float(anchor["target_index"]) for anchor in batch]' in source
    assert '[float(anchor["target_index"]) for anchor in anchors]' in source
    assert "history_depth=selected_history_depth.index_select(" in source
    assert '"route_expert_mixture": "vnext.route_expert_mixture."' in source
    assert '"route_expert_mixture_feature_dim": 6' in source
    assert '"route_expert_mixture_semantic_dim": int(model.vnext.d)' in source
    assert '"semantic_per_sample_route_selector": True' in source
    assert '"candidate_semantic_expert_selector": True' in source
    assert '"legacy_candidate_route_gate_frozen": True' in source
    assert '"legacy_candidate_route_gate_reset_for_rng_compatibility": True' in source
    assert '"depth_selector_reset_after_raw_expert": True' in source
    assert '"route_expert_selection_threshold": 0.55' in source
    assert "Stage2 resume checkpoint predates the depth-aware route mixture" in source


def test_stage2_full_launcher_exposes_safety_and_causal_weight_overrides() -> None:
    launcher = Path(__file__).parents[1].joinpath(
        "scripts", "sbatch", "run_clstr_vnext_full_stage2_segment.sh"
    ).read_text(encoding="utf-8")
    assert "LAMBDA_SAFETY=${LAMBDA_SAFETY:-0.05}" in launcher
    assert "LAMBDA_RAW_ROUTE=${LAMBDA_RAW_ROUTE:-0.5}" in launcher
    assert "LAMBDA_ROUTE_TOPK=${LAMBDA_ROUTE_TOPK:-0.0}" in launcher
    assert "LAMBDA_MIXTURE=${LAMBDA_MIXTURE:-0.2}" in launcher
    assert "MIXTURE_UTILITY_SCALE=${MIXTURE_UTILITY_SCALE:-0.5}" in launcher
    assert "LAMBDA_HISTORY=${LAMBDA_HISTORY:-0.2}" in launcher
    assert "NO_REGRET_TOLERANCE=${NO_REGRET_TOLERANCE:-0.05}" in launcher
    assert '--lambda_safety "${LAMBDA_SAFETY}"' in launcher
    assert '--lambda_raw_route "${LAMBDA_RAW_ROUTE}"' in launcher
    assert '--lambda_route_topk "${LAMBDA_ROUTE_TOPK}"' in launcher
    assert '--route_topk "${ROUTE_TOPK}"' in launcher
    assert '--lambda_mixture "${LAMBDA_MIXTURE}"' in launcher
    assert '--mixture_utility_scale "${MIXTURE_UTILITY_SCALE}"' in launcher
    assert '--lambda_history "${LAMBDA_HISTORY}"' in launcher
    assert '--no_regret_tolerance "${NO_REGRET_TOLERANCE}"' in launcher
    assert "ENABLE_NATIVE_SYNCHRONIZATION=${ENABLE_NATIVE_SYNCHRONIZATION:-0}" in launcher
    assert "native_synchronization_args=()" in launcher
    assert "--enable_native_synchronization" in launcher
    assert '"${native_synchronization_args[@]}"' in launcher
    assert "ALLOW_EXTERNAL_PARENT_OUTPUTS=${ALLOW_EXTERNAL_PARENT_OUTPUTS:-0}" in launcher
    assert "FROZEN_CACHE_READ_ONLY=${FROZEN_CACHE_READ_ONLY:-0}" in launcher
    assert "requested Stage0 checkpoint does not match its selected report" in launcher
    assert "requested candidate checkpoint digest does not match" in launcher


def test_stage2_only_parent_reuse_launcher_is_explicit_and_read_only() -> None:
    root = Path(__file__).parents[1]
    launcher = root.joinpath(
        "scripts", "submit_clstr_vnext_stage2_from_existing_parents.sh"
    ).read_text(encoding="utf-8")
    assert "STAGE0_CHECKPOINT_PATH=${STAGE0_CHECKPOINT_PATH:?" in launcher
    assert "CANDIDATE_CHECKPOINT_PATH=${CANDIDATE_CHECKPOINT_PATH:?" in launcher
    assert "parent does not descend from the requested Stage0 checkpoint" in launcher
    assert "selected candidate checkpoint digest changed" in launcher
    assert "candidate_report.get(\"selection\") != candidate_selection" in launcher
    assert "ALLOW_EXTERNAL_PARENT_OUTPUTS=1" in launcher
    assert "FROZEN_CACHE_READ_ONLY=1" in launcher
    assert "run_clstr_vnext_full_precompute.sh" not in launcher
    assert "run_clstr_vnext_full_stage0_segment.sh" not in launcher
    assert "run_clstr_vnext_candidate_compressor.sh" not in launcher
    assert "FINALIZE_FULL_CHAIN=0" in launcher


def test_full_chain_exports_native_synchronization_configuration() -> None:
    root = Path(__file__).parents[1]
    launcher = root.joinpath(
        "scripts", "submit_clstr_vnext_full_chain.sh"
    ).read_text(encoding="utf-8")
    for name, default in (
        ("ENABLE_NATIVE_SYNCHRONIZATION", "0"),
        ("SYNCHRONIZATION_PAIR_DIM", "96"),
        ("SYNCHRONIZATION_TRACE_LENGTH", "8"),
        ("SYNCHRONIZATION_SCALE_INITIAL", "0.05"),
    ):
        assert f"{name}=${{{name}:-{default}}}" in launcher
        assert f"{name}=${{{name}}}" in launcher


def test_stage2_topk_route_supervision_is_natural_support_only() -> None:
    source = Path(__file__).parents[1].joinpath(
        "clstr", "vnext_stage2_train.py"
    ).read_text(encoding="utf-8")
    natural_block = source.index(
        "natural_route = model.vnext_safe_candidate_route_scores"
    )
    teacher_block = source.index(
        "teacher_support = training_only_teacher_retained_support"
    )
    assert natural_block > teacher_block
    coverage_block = source[natural_block : source.index(
        "if bool(teacher_labels.eligible_mask.any().item())", natural_block
    )]
    assert "candidate_path.support.candidate_ids" in coverage_block
    assert "natural_route_labels.positive_mask" in coverage_block
    assert "teacher_support.candidate_ids" not in coverage_block
    assert '"candidate_membership_protocol": "immutable_natural_support"' in source
    assert '"positive_injection_count": 0' in source


def test_stage2_objective_warm_start_is_distinct_from_exact_resume() -> None:
    signature = inspect.signature(train_vnext_stage2)
    assert "warm_start_checkpoint_path" in signature.parameters
    source = inspect.getsource(train_vnext_stage2)
    assert "Stage2 exact resume and objective warm start are mutually exclusive" in source
    assert "stage2_objective_refinement_reset_optimizer_v1" in source
    assert '"optimizer_restored": False' in source
    assert '"objective_warm_start_checkpoint_sha256"' in source


def test_stage2_route_state_is_fail_closed_and_causal() -> None:
    text = _causal_route_state_text(
        {
            "state_text_current": "current only",
            "state_text_causal": "current plus causal history",
            "causal_prefix_events": [
                {
                    "step_index": 0,
                    "skill_id": "prior",
                    "action_text": "prior action",
                    "result_text": "result excluded",
                    "result_executed": True,
                }
            ],
        }
    )
    assert "current only" in text and "prior action" in text
    assert "result excluded" not in text
    with pytest.raises(ValueError, match="persisted events"):
        _causal_route_state_text({"state_text_current": "current only"})


def test_stage2_state_cache_collects_exact_compact_route_queries() -> None:
    row = {
        "state_text_current": "goal: current only",
        "state_text_causal": "stale serialized causal state with raw result",
        "causal_prefix_events": [
            {
                "step_index": 0,
                "skill_id": "prior",
                "action_text": "prior action",
                "result_text": "raw result must stay out of route state",
                "result_executed": True,
            }
        ],
    }
    texts = _required_state_cache_texts([[row]])
    assert texts == ["goal: current only", _causal_route_state_text(row)]
    assert row["state_text_causal"] not in texts
    assert "prior action" in texts[1]
    assert "raw result must stay out of route state" not in texts[1]


def test_stage2_fixed_recall_budgets_belong_to_ordinary_dev_only() -> None:
    ordinary = inspect.signature(_evaluate_stage2_ordinary_dev)
    causal = inspect.signature(_evaluate_stage2_causal_dev)
    assert ordinary.parameters["recall_ms"].default == (100, 500)
    assert "coarse_k" in ordinary.parameters
    assert "compressed_m" in ordinary.parameters
    assert "coarse_k" in causal.parameters
    assert "compressed_m" in causal.parameters
    assert "recall_ms" not in causal.parameters


def test_ordinary_metrics_decompose_same_support_route_and_candidate_extension() -> None:
    records = []
    for index, values in enumerate(((0.2, -0.05), (-0.1, 0.03))):
        same_support_delta, extension_delta = values
        records.append(
            {
                "cluster_id": f"cluster-{index}",
                "route_mrr_delta": same_support_delta + extension_delta,
                "raw_route_mrr_delta": same_support_delta + extension_delta,
                "same_support_route_mrr_delta": same_support_delta,
                "candidate_extension_mrr_delta": extension_delta,
                "static_route_mrr": 0.4,
                "same_support_dynamic_route_mrr": 0.4 + same_support_delta,
                "dynamic_route_mrr": 0.4 + same_support_delta + extension_delta,
                "raw_dynamic_route_mrr": (
                    0.4 + same_support_delta + extension_delta
                ),
                "mixture_probability": 0.25 + 0.25 * index,
                "fixed_causal_recall_rank": 10,
                "dynamic_recall_rank": 9,
                "dynamic_route_rank": 4,
                "raw_dynamic_route_rank": 4,
                "same_support_dynamic_route_rank": 5,
                "static_route_rank": 6,
                "candidate_hit@500": 1.0,
                "candidate_hit@union": 1.0,
                "candidate_support_changed": 1.0,
                "pool_size": 1000,
                "fixed_hit@100": 1.0,
                "dynamic_hit@100": 1.0,
                "fixed_hit@500": 1.0,
                "dynamic_hit@500": 1.0,
            }
        )
    report = _ordinary_metric_report(records, seed=7, recall_ms=(100, 500))
    assert report["same_support_route_mrr_delta"]["mean"] == pytest.approx(0.05)
    assert report["candidate_extension_mrr_delta"]["mean"] == pytest.approx(-0.01)
    assert report["same_support_dynamic_route_mean_rank"] == pytest.approx(5.0)
    assert report["static_route_mrr"]["mean"] == pytest.approx(0.4)
    assert report["dynamic_route_mrr"]["mean"] == pytest.approx(0.44)
    assert report["static_route_recall_at_1"]["mean"] == 0.0
    assert report["raw_dynamic_route_recall_at_5"]["mean"] == 1.0
    assert report["raw_route_recall_at_5_delta"]["mean"] == 1.0
    assert report["same_support_route_recall_at_5_delta"]["mean"] == 1.0


def _row(trajectory: str, step: int, target: str, *, state: str = "goal: shared"):
    return {
        "trajectory_id": trajectory,
        "task_id": trajectory,
        "step_index": step,
        "goal_text": "shared",
        "benchmark": "fixture-source",
        "state_text_current": state,
        "state_text_causal": (
            state
            if step == 0
            else f"{state}\ncausal_prefix:\nevent[0].skill_id: prior-{trajectory}"
        ),
        "causal_prefix_events": (
            []
            if step == 0
            else [
                {
                    "step_index": 0,
                    "skill_id": f"prior-{trajectory}",
                    "action_text": "prior action",
                    "result_text": "",
                    "result_executed": False,
                }
            ]
        ),
        "decision_step_index": step,
        "target_skill_id": target,
        "action_text": f"call {target}",
        "runtime_visible_catalog_id": "pool",
        "inventory_catalog_digest": "pool-digest",
        "capabilities": {
            "ordered_next_tool": bool(step > 0),
            "actual_execution_result": False,
            "causal_branch_pair": False,
        },
    }


def _trajectories():
    rows = [
        _row("ta", 0, "a"),
        _row("ta", 1, "b"),
        {**_row("tb", 0, "a"), "action_text": "different verified history event"},
        _row("tb", 1, "c"),
    ]
    trajectories, _report = _prepare_trajectories(rows, SKILLS, CATALOGS)
    return _trajectory_index(trajectories)


def _write_pair(tmp_path, pair):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    return path


def test_stage2_requires_exact_selected_skill_prefix(tmp_path) -> None:
    skills = tmp_path / "selected_skills.jsonl"
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")
    digest = hashlib.sha256(skills.read_bytes()).hexdigest()
    payload = {
        "run_contract": {
            "inputs": {"selected_skills_sha256": digest},
        }
    }
    assert _require_selected_skill_prefix(payload, skills) == digest

    skills.write_text('{"skill_id":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exact selected skill prefix"):
        _require_selected_skill_prefix(payload, skills)


def test_stage2_ordinary_sequences_exclude_unordered_parallel_trajectories() -> None:
    rows = [_row("parallel", 0, "a"), _row("parallel", 1, "b")]
    for row in rows:
        row["capabilities"]["ordered_next_tool"] = False
    trajectories, report = _prepare_trajectories(
        rows,
        SKILLS,
        CATALOGS,
        sequence_role="ordinary",
    )
    assert trajectories == []
    assert report["skipped"][
        "trajectory_not_fully_ordered_after_initial_event"
    ] == 2


def test_stage2_pair_support_cannot_enter_ordinary_sampling() -> None:
    rows = [_row("support", 0, "a"), _row("support", 1, "b")]
    rows[0]["pair_support_only"] = True
    with pytest.raises(ValueError, match="entered ordinary Stage2 sampling"):
        _prepare_trajectories(rows, SKILLS, CATALOGS, sequence_role="ordinary")


def test_stage2_quarantines_unverified_result_text_without_dropping_route_row() -> None:
    rows = [_row("result", 0, "a"), _row("result", 1, "b")]
    rows[0]["actual_result_text"] = "unbound output"
    rows[0]["actual_result_executed"] = True

    trajectories, report = _prepare_trajectories(
        rows,
        SKILLS,
        CATALOGS,
        sequence_role="ordinary",
    )

    assert len(trajectories) == 1
    assert len(trajectories[0]) == 2
    prepared = trajectories[0][0]
    assert prepared["_vnext_target_skill_id"] == "a"
    assert prepared["action_text"] == "call a"
    assert prepared["actual_result_text"] == ""
    assert prepared["actual_result_executed"] is False
    assert prepared["_vnext_result_correction_eligible"] is False
    assert prepared["_vnext_result_text_quarantined"] is True
    assert rows[0]["actual_result_text"] == "unbound output"
    assert rows[0]["actual_result_executed"] is True
    quarantine = report["result_text_quarantine"]
    assert quarantine["detected_input_row_count"] == 1
    assert quarantine["retained_prepared_row_count"] == 1
    assert quarantine["retained_prepared_rows_by_reason"] == {
        "executed_flag_without_actual_execution_capability": 1
    }
    assert quarantine["excluded_from_memory_correction"] is True
    assert quarantine["route_and_transition_supervision_retained"] is True

    clean_rows = [_row("result", 0, "a"), _row("result", 1, "b")]
    clean_trajectories, _ = _prepare_trajectories(
        clean_rows,
        SKILLS,
        CATALOGS,
        sequence_role="ordinary",
    )
    assert _prepared_prefix_record(trajectories[0], 1) == _prepared_prefix_record(
        clean_trajectories[0], 1
    )


def test_stage2_preserves_verified_result_text_for_correction() -> None:
    rows = [_row("result", 0, "a"), _row("result", 1, "b")]
    rows[0]["actual_result_text"] = "verified output"
    rows[0]["actual_result_executed"] = True
    rows[0]["capabilities"]["actual_execution_result"] = True

    trajectories, report = _prepare_trajectories(rows, SKILLS, CATALOGS)

    prepared = trajectories[0][0]
    assert prepared["actual_result_text"] == "verified output"
    assert prepared["actual_result_executed"] is True
    assert prepared["_vnext_result_correction_eligible"] is True
    assert report["result_text_quarantine"]["detected_input_row_count"] == 0


def test_stage2_rejects_execution_capability_without_aligned_result() -> None:
    rows = [_row("result", 0, "a"), _row("result", 1, "b")]
    rows[0]["capabilities"]["actual_execution_result"] = True
    with pytest.raises(ValueError, match="capability lacks an aligned"):
        _prepare_trajectories(rows, SKILLS, CATALOGS)


def test_stage2_rejects_executed_result_flag_without_text() -> None:
    rows = [_row("result", 0, "a"), _row("result", 1, "b")]
    rows[0]["actual_result_executed"] = True
    with pytest.raises(ValueError, match="flag lacks actual_result_text"):
        _prepare_trajectories(rows, SKILLS, CATALOGS)


def test_stage2_pair_loader_requires_symmetric_exact_alias_contract(tmp_path) -> None:
    trajectories = _trajectories()
    prefix_a = _prepared_prefix_record(trajectories["ta"], 1)
    prefix_b = _prepared_prefix_record(trajectories["tb"], 1)
    pair = {
        "causal_pair_id": "p",
        "causal_pair_kind": "history_branch",
        "causal_cluster_id": "cluster-shared",
        "source_id": "fixture-source",
        "causal_label_verified": True,
        "trainable_causal_label": True,
        "causal_verification": {
            "evidence_type": "trusted_expert_branch_annotation_v1"
        },
        "history_a_digest": prefix_a["digest"],
        "history_b_digest": prefix_b["digest"],
        "first_history_divergence_index": 0,
        "row_a": {"trajectory_id": "ta", "step_index": 1, "target_skill_id": "b"},
        "row_b": {"trajectory_id": "tb", "step_index": 1, "target_skill_id": "c"},
    }
    loaded = _load_causal_pairs(
        _write_pair(tmp_path, pair),
        expected_kind="history_branch",
        trajectories=trajectories,
        catalogs=CATALOGS,
    )
    assert len(loaded) == 1
    assert loaded[0]["_index"] == 1
    assert loaded[0]["_row_a"]["_vnext_target_skill_id"] == "b"
    assert loaded[0]["_row_b"]["_vnext_target_skill_id"] == "c"


def test_stage2_pair_loader_accepts_verified_unordered_remaining_tool_branch(
    tmp_path,
) -> None:
    rows = [
        _row("ta", 0, "a"),
        _row("ta", 1, "b"),
        _row("ta", 2, "c"),
        _row("tb", 0, "a"),
        _row("tb", 1, "c"),
        _row("tb", 2, "b"),
    ]
    for row in rows:
        row["pair_support_only"] = True
        row["capabilities"]["ordered_next_tool"] = False
        row["capabilities"]["causal_branch_pair"] = True
    prepared, _report = _prepare_trajectories(
        rows,
        SKILLS,
        CATALOGS,
        sequence_role="pair_support",
    )
    trajectories = _trajectory_index(prepared)
    prefix_a = _prepared_prefix_record(trajectories["ta"], 2)
    prefix_b = _prepared_prefix_record(trajectories["tb"], 2)
    pair = {
        "causal_pair_id": "unordered",
        "causal_pair_kind": "history_branch",
        "causal_cluster_id": "unordered-cluster",
        "source_id": "fixture-source",
        "causal_label_verified": True,
        "trainable_causal_label": True,
        "causal_verification": {
            "evidence_type": "verified_unordered_remaining_tool_branch_v1",
            "unordered_required_set_verified": True,
            "actual_swapped_event_results": True,
            "shared_prefix_event_count": 1,
        },
        "history_a_digest": prefix_a["digest"],
        "history_b_digest": prefix_b["digest"],
        "first_history_divergence_index": 1,
        "row_a": {"trajectory_id": "ta", "step_index": 2, "target_skill_id": "c"},
        "row_b": {"trajectory_id": "tb", "step_index": 2, "target_skill_id": "b"},
    }

    loaded = _load_causal_pairs(
        _write_pair(tmp_path, pair),
        expected_kind="history_branch",
        trajectories=trajectories,
        catalogs=CATALOGS,
    )

    assert len(loaded) == 1


def test_stage2_pair_loader_rejects_current_state_change(tmp_path) -> None:
    trajectories = _trajectories()
    prefix_a = _prepared_prefix_record(trajectories["ta"], 1)
    prefix_b = _prepared_prefix_record(trajectories["tb"], 1)
    trajectories["tb"][1]["state_text_current"] = "goal: donor shortcut"
    pair = {
        "causal_pair_id": "p",
        "causal_pair_kind": "history_branch",
        "causal_cluster_id": "cluster-shared",
        "source_id": "fixture-source",
        "causal_label_verified": True,
        "trainable_causal_label": True,
        "causal_verification": {
            "evidence_type": "trusted_expert_branch_annotation_v1"
        },
        "history_a_digest": prefix_a["digest"],
        "history_b_digest": prefix_b["digest"],
        "first_history_divergence_index": 0,
        "row_a": {"trajectory_id": "ta", "step_index": 1, "target_skill_id": "b"},
        "row_b": {"trajectory_id": "tb", "step_index": 1, "target_skill_id": "c"},
    }
    with pytest.raises(ValueError, match="exact aliases"):
        _load_causal_pairs(
            _write_pair(tmp_path, pair),
            expected_kind="history_branch",
            trajectories=trajectories,
            catalogs=CATALOGS,
        )


def test_stage2_outcome_pair_holds_action_fixed_and_requires_novel_results(tmp_path) -> None:
    trajectories = _trajectories()
    for trajectory_id, result in (("ta", "outcome A"), ("tb", "outcome B")):
        event = trajectories[trajectory_id][0]
        event["action_text"] = "same call"
        event["actual_result_text"] = result
        event["result_novel_for_memory"] = True
        event["actual_result_executed"] = True
        event["capabilities"]["actual_execution_result"] = True
    prefix_a = _prepared_prefix_record(trajectories["ta"], 1)
    prefix_b = _prepared_prefix_record(trajectories["tb"], 1)
    pre_event = _prepared_prefix_record(trajectories["ta"], 0)
    pair = {
        "causal_pair_id": "outcome",
        "causal_pair_kind": "result_outcome",
        "causal_cluster_id": "cluster-shared",
        "source_id": "fixture-source",
        "causal_label_verified": True,
        "trainable_causal_label": True,
        "causal_verification": {
            "evidence_type": "same_action_different_executed_result_v1",
            "pre_event_history_digest": pre_event["digest"],
        },
        "same_pre_event_history": True,
        "history_a_digest": prefix_a["digest"],
        "history_b_digest": prefix_b["digest"],
        "first_history_divergence_index": 0,
        "row_a": {"trajectory_id": "ta", "step_index": 1, "target_skill_id": "b"},
        "row_b": {"trajectory_id": "tb", "step_index": 1, "target_skill_id": "c"},
        "event_a": {"trajectory_id": "ta", "step_index": 0, "target_skill_id": "a"},
        "event_b": {"trajectory_id": "tb", "step_index": 0, "target_skill_id": "a"},
    }
    loaded = _load_causal_pairs(
        _write_pair(tmp_path, pair),
        expected_kind="result_outcome",
        trajectories=trajectories,
        catalogs=CATALOGS,
    )
    assert loaded[0]["_event_index"] == 0

    trajectories["tb"][0]["action_text"] = "different call"
    pair["history_b_digest"] = _prepared_prefix_record(trajectories["tb"], 1)[
        "digest"
    ]
    with pytest.raises(ValueError, match="executed action fixed"):
        _load_causal_pairs(
            _write_pair(tmp_path, pair),
            expected_kind="result_outcome",
            trajectories=trajectories,
            catalogs=CATALOGS,
        )


def test_stage2_positive_rank_uses_deterministic_skill_id_tie_break() -> None:
    logits = torch.tensor([[1.0, 2.0, 2.0, 0.0]])
    positive = torch.tensor([[False, False, True, False]])
    legal = torch.ones_like(positive)
    assert _best_positive_rank(logits, positive, legal).tolist() == [2]


def test_stage2_pair_loader_rejects_unverified_matched_history(tmp_path) -> None:
    pair = {
        "causal_pair_id": "diagnostic-only",
        "causal_pair_kind": "history_branch",
        "causal_cluster_id": "cluster-shared",
        "source_id": "fixture-source",
        "causal_label_verified": False,
        "trainable_causal_label": False,
        "row_a": {"trajectory_id": "ta", "step_index": 1, "target_skill_id": "b"},
        "row_b": {"trajectory_id": "tb", "step_index": 1, "target_skill_id": "c"},
    }
    with pytest.raises(ValueError, match="verified trainable label"):
        _load_causal_pairs(
            _write_pair(tmp_path, pair),
            expected_kind="history_branch",
            trajectories=_trajectories(),
            catalogs=CATALOGS,
        )


def test_stage2_positive_causal_weight_fails_closed_on_empty_view() -> None:
    with pytest.raises(ValueError, match="nonempty verified train view: order_effect"):
        _require_weighted_causal_views(
            {
                "history_branch": [{}],
                "order_effect": [],
                "result_outcome": [{}],
            },
            {
                "history_branch": [{}],
                "order_effect": [],
                "result_outcome": [{}],
            },
            weights={
                "history_branch": 0.2,
                "order_effect": 0.1,
                "result_outcome": 0.2,
            },
        )
    _require_weighted_causal_views(
        {
            "history_branch": [{}],
            "order_effect": [],
            "result_outcome": [{}],
        },
        {
            "history_branch": [{}],
            "order_effect": [],
            "result_outcome": [{}],
        },
        weights={
            "history_branch": 0.2,
            "order_effect": 0.0,
            "result_outcome": 0.2,
        },
    )


def test_stage2_paired_bootstrap_is_seed_stable() -> None:
    first = _paired_mean_ci([1.0, 2.0, 3.0, 4.0], seed=17, samples=200)
    second = _paired_mean_ci([1.0, 2.0, 3.0, 4.0], seed=17, samples=200)
    assert first == second
    assert first["mean"] == 2.5
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_stage2_bootstrap_aggregates_pairs_by_task_cluster() -> None:
    report = _paired_mean_ci(
        [1.0, 3.0, 10.0],
        seed=17,
        samples=200,
        cluster_ids=["task-a", "task-a", "task-b"],
    )
    assert report["count"] == 3
    assert report["cluster_count"] == 2
    assert report["mean"] == 6.0
    assert report["pair_mean"] == pytest.approx(14.0 / 3.0)


def test_stage2_validation_cap_is_source_stratified() -> None:
    pairs = [
        {"causal_pair_id": f"a-{index}", "_row_a": {"benchmark": "source-a"}}
        for index in range(4)
    ] + [
        {"causal_pair_id": "b-0", "_row_a": {"benchmark": "source-b"}}
    ]
    selected = _stable_cap_pairs(pairs, 2)
    assert {pair["_row_a"]["benchmark"] for pair in selected} == {
        "source-a",
        "source-b",
    }


def test_stage2_robust_dev_cap_is_source_stratified() -> None:
    anchors = [
        {"identity": f"a-{index}", "source": "source-a"}
        for index in range(4)
    ] + [{"identity": "b-0", "source": "source-b"}]
    selected = _stable_cap_anchors(anchors, 2)
    assert {anchor["source"] for anchor in selected} == {"source-a", "source-b"}


def test_stage2_segment_sampling_balances_eligible_sources() -> None:
    trajectories = [
        [
            {"benchmark": source, "trajectory_id": f"{source}-t", "step_index": step}
            for step in range(3)
        ]
        for source in ("source-a", "source-b")
    ]
    segments, report = _sample_segments(
        trajectories,
        batch_size=4,
        horizon=2,
        rng=random.Random(7),
        anchored=False,
        source_position=0,
    )
    assert len(segments) == 4
    assert report["sampled_sources"] == {"source-a": 2, "source-b": 2}
    assert report["mode"] == "ordinary"


def test_stage2_segment_sampling_balances_benchmark_before_source_count() -> None:
    trajectories = []
    for source in ("toolbench-a", "toolbench-b", "toolbench-c"):
        trajectories.append(
            [
                {
                    "benchmark": "toolbench_g3",
                    "provenance": {"source_id": source},
                    "trajectory_id": f"{source}-t",
                    "step_index": step,
                }
                for step in range(3)
            ]
        )
    trajectories.append(
        [
            {
                "benchmark": "tau2",
                "provenance": {"source_id": "tau2-success"},
                "trajectory_id": "tau2-t",
                "step_index": step,
            }
            for step in range(3)
        ]
    )

    _segments, report = _sample_segments(
        trajectories,
        batch_size=4,
        horizon=2,
        rng=random.Random(11),
        anchored=False,
        source_position=0,
    )

    assert report["protocol"] == "benchmark_family_source_trajectory_round_robin_v2"
    assert report["sampled_families"] == {"tau2": 2, "toolbench": 2}


def test_stage2_segment_sampling_honors_required_family_exactly() -> None:
    trajectories = [
        [
            {
                "benchmark": family,
                "provenance": {"source_id": f"{family}-source"},
                "trajectory_id": f"{family}-trajectory",
                "step_index": step,
            }
            for step in range(4)
        ]
        for family in ("long", "short")
    ]

    segments, report = _sample_segments(
        trajectories,
        batch_size=3,
        horizon=3,
        rng=random.Random(17),
        anchored=False,
        source_position=0,
        required_family="short",
    )

    assert len(segments) == 3
    assert {
        segment[0][0]["benchmark"]
        for segment in segments
    } == {"short"}
    assert report["sampled_families"] == {"short": 3}
    assert report["required_family"] == "short"
    assert report["required_family_honored"] is True


def test_stage2_required_family_keeps_forced_anchor_fallback_local() -> None:
    trajectories = [
        [
            {
                "benchmark": family,
                "provenance": {"source_id": f"{family}-source"},
                "trajectory_id": f"{family}-trajectory",
                "step_index": step,
            }
            for step in range(3)
        ]
        for family in ("long", "short")
    ]
    forced = [
        {
            "rows": trajectories[0],
            "event_index": 0,
            "target_index": 1,
            "kind": "one_error_prefix",
        }
    ]

    segments, report = _sample_segments(
        trajectories,
        batch_size=2,
        horizon=2,
        rng=random.Random(19),
        anchored=False,
        source_position=0,
        forced_anchors=forced,
        required_family="short",
    )

    assert report["forced_anchor_requested"] is True
    assert report["forced_anchor_eligible_count"] == 0
    assert report["mode"] == "ordinary"
    assert report["sampled_families"] == {"short": 2}
    assert all(segment[3] is False for segment in segments)


def test_stage2_robust_prefix_anchor_keeps_executed_error_inside_bptt(tmp_path) -> None:
    trajectories = _trajectories()
    view_path = tmp_path / "one-error.jsonl"
    view_path.write_text(
        json.dumps(
            {
                "trajectory_id": "ta",
                "step_index": 1,
                "target_skill_id": "b",
                "robust_prefix_kind": "one_error_prefix",
                "capabilities": {"verified_robust_prefix": True},
                "robust_prefix_annotation": {
                    "executed_error_step_indices": [0]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    anchors = _load_robust_prefix_anchors(
        view_path,
        expected_kind="one_error_prefix",
        trajectories=trajectories,
    )
    assert len(anchors) == 1
    segments, report = _sample_segments(
        list(trajectories.values()),
        batch_size=1,
        horizon=2,
        rng=random.Random(13),
        anchored=False,
        source_position=0,
        forced_anchors=anchors,
    )
    assert report["mode"] == "robust_prefix"
    assert report["sampled_anchor_kinds"] == {"one_error_prefix": 1}
    _rows, start, end, anchored = segments[0]
    assert (start, end, anchored) == (0, 2, True)


def test_stage2_ordinary_dev_anchor_cap_covers_sources_and_horizons() -> None:
    trajectories = []
    for source in ("source-a", "source-b"):
        rows = []
        for step in range(5):
            row = _row(f"{source}-trajectory", step, "a")
            row["benchmark"] = source
            row["split_group_identity"] = f"{source}-task"
            rows.append(row)
        trajectories.append(rows)
    anchors, report = _build_ordinary_dev_anchors(
        trajectories,
        max_rows=4,
        max_horizon=4,
    )
    assert len(anchors) == 4
    assert {anchor["source"] for anchor in anchors} == {"source-a", "source-b"}
    assert {anchor["horizon_bucket"] for anchor in anchors} == {"1", "2-3", "4-7"}
    assert report["protocol"] == "source_stratum_task_round_robin_v1"
    assert len(report["anchor_identity_sha256"]) == 64


def test_stage2_ordinary_dev_anchor_includes_long_prefix_drift_buckets() -> None:
    rows = []
    for step in range(35):
        row = _row("long-trajectory", step, "a")
        row["benchmark"] = "agentgym_agenttraj_l"
        row["inventory_pool_size"] = 100
        rows.append(row)
    anchors, report = _build_ordinary_dev_anchors(
        [rows],
        max_rows=None,
        max_horizon=16,
    )
    assert {anchor["horizon_bucket"] for anchor in anchors} >= {
        "8-16",
        "17-32",
        "33+",
    }
    assert report["maximum_evaluated_prefix_index"] == 34


def _fixed_causal_coarse_report(
    *,
    clusters: int,
    hit_ci_low: float,
    hit_ci_low_100: float | None = None,
    memory_invariant: bool = True,
    dynamic_hit_ci_low_500: float | None = None,
    dynamic_delta_ci_low_500: float | None = None,
    union_delta_ci_low: float = 0.0,
) -> dict:
    count = clusters
    low_by_m = {
        100: hit_ci_low if hit_ci_low_100 is None else hit_ci_low_100,
        500: hit_ci_low,
    }
    return {
        "candidate_support_memory_invariant": bool(memory_invariant),
        "overall": {
            "candidate_recall_at_500": {
                "count": count,
                "cluster_count": clusters,
                "mean": max(hit_ci_low, 0.0),
                "ci_low": hit_ci_low,
                "ci_high": max(hit_ci_low, 0.0),
            },
            "candidate_union_recall": {
                "count": count,
                "cluster_count": clusters,
                "mean": max(hit_ci_low, 0.0),
                "ci_low": hit_ci_low,
                "ci_high": max(hit_ci_low, 0.0),
            },
            "candidate_union_minus_static_recall": {
                "count": count,
                "cluster_count": clusters,
                "mean": max(union_delta_ci_low, 0.0),
                "ci_low": union_delta_ci_low,
                "ci_high": max(union_delta_ci_low, 0.0),
            },
            "candidate_support_change_rate": {
                "count": count,
                "cluster_count": clusters,
                "mean": 0.25 if count else 0.0,
                "ci_low": 0.0,
                "ci_high": 0.5 if count else 0.0,
            },
            "candidate_recall": {
                str(m): {
                    "deployment_full_pool": {
                        "fixed_causal_hit_rate": {
                            "count": count,
                            "cluster_count": clusters,
                            "mean": max(low_by_m[m], 0.0),
                            "ci_low": low_by_m[m],
                            "ci_high": max(low_by_m[m], 0.0),
                        },
                        "dynamic_hit_rate": {
                            "count": count,
                            "cluster_count": clusters,
                            "mean": max(low_by_m[m], 0.0),
                            "ci_low": (
                                dynamic_hit_ci_low_500
                                if m == 500 and dynamic_hit_ci_low_500 is not None
                                else low_by_m[m]
                            ),
                            "ci_high": max(
                                dynamic_hit_ci_low_500
                                if m == 500 and dynamic_hit_ci_low_500 is not None
                                else low_by_m[m],
                                0.0,
                            ),
                        },
                        "dynamic_minus_static": {
                            "count": count,
                            "cluster_count": clusters,
                            "mean": 0.0,
                            "ci_low": (
                                dynamic_delta_ci_low_500
                                if m == 500 and dynamic_delta_ci_low_500 is not None
                                else 0.0
                            ),
                            "ci_high": 0.0,
                        },
                        "deployed_union_hit_rate": {
                            "count": count,
                            "cluster_count": clusters,
                            "mean": max(hit_ci_low, 0.0),
                            "ci_low": hit_ci_low,
                            "ci_high": max(hit_ci_low, 0.0),
                        },
                        "deployed_union_minus_fixed": {
                            "count": count,
                            "cluster_count": clusters,
                            "mean": max(union_delta_ci_low, 0.0),
                            "ci_low": union_delta_ci_low,
                            "ci_high": max(union_delta_ci_low, 0.0),
                        },
                    }
                }
                for m in (100, 500)
            }
        }
    }


def test_stage2_coarse_recall_gate_fails_closed_without_full_pool_support() -> None:
    report = _fixed_causal_coarse_report(clusters=0, hit_ci_low=0.0)
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=2,
    )
    assert gate["pass"] is False
    assert gate["status"] == "not_testable"


def test_stage2_coarse_recall_gate_requires_static_and_union_absolute_recall() -> None:
    report = _fixed_causal_coarse_report(clusters=8, hit_ci_low=0.75)
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=8,
    )
    assert gate["pass"] is True
    assert gate["memory_invariant_support"] is False
    assert gate["deployment_pass"] is True
    assert gate["deployment_m"] == 500


def test_stage2_coarse_gate_ignores_weak_dynamic_only_top500_for_safe_union() -> None:
    report = _fixed_causal_coarse_report(
        clusters=20,
        hit_ci_low=0.75,
        dynamic_hit_ci_low_500=0.60,
        dynamic_delta_ci_low_500=-0.20,
    )
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is True
    assert gate["deployment_pass"] is True
    assert gate["dynamic_only_deployment_noninferiority_pass"] is False
    assert gate["dynamic_only_deployment_is_diagnostic"] is True


def test_stage2_coarse_gate_accepts_legacy_preserved_union_metrics() -> None:
    report = _fixed_causal_coarse_report(
        clusters=20,
        hit_ci_low=0.75,
        dynamic_hit_ci_low_500=0.60,
        dynamic_delta_ci_low_500=-0.20,
    )
    del report["overall"]["candidate_union_minus_static_recall"]
    for values in report["overall"]["candidate_recall"].values():
        deployment = values["deployment_full_pool"]
        del deployment["deployed_union_hit_rate"]
        del deployment["deployed_union_minus_fixed"]
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is True
    assert gate["candidate_union_legacy_contract_fallback"] is True


def test_stage2_coarse_recall_gate_rejects_union_membership_regression() -> None:
    report = _fixed_causal_coarse_report(
        clusters=20,
        hit_ci_low=0.75,
        union_delta_ci_low=-0.01,
    )
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is False
    assert gate["candidate_union_preservation_pass"] is False


def test_stage2_coarse_recall_gate_matches_direct_top500_deployment() -> None:
    report = _fixed_causal_coarse_report(
        clusters=20,
        hit_ci_low=0.75,
        hit_ci_low_100=0.49,
    )
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is True
    assert gate["deployment_m"] == 500
    assert gate["support_by_m"]["100"]["fixed_causal_hit_rate"]["ci_low"] == 0.49


def test_stage2_coarse_recall_gate_marks_insufficient_cluster_support_not_testable() -> None:
    report = _fixed_causal_coarse_report(clusters=8, hit_ci_low=0.75)
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=9,
    )
    assert gate["pass"] is False
    assert gate["status"] == "not_testable"
    assert gate["support_by_m"]["100"]["support_sufficient"] is False


def test_stage2_coarse_recall_gate_accepts_memory_dependent_support() -> None:
    report = _fixed_causal_coarse_report(
        clusters=20,
        hit_ci_low=0.75,
        memory_invariant=False,
    )
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is True
    assert gate["status"] == "ok"
    assert gate["memory_invariant_support"] is False


def test_stage2_coarse_recall_gate_requires_observed_candidate_membership_change() -> None:
    report = _fixed_causal_coarse_report(clusters=20, hit_ci_low=0.75)
    report["overall"]["candidate_support_change_rate"]["mean"] = 0.0
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is False
    assert gate["candidate_support_change_pass"] is False


def test_stage2_coarse_recall_gate_rejects_weak_absolute_candidate_recall() -> None:
    report = _fixed_causal_coarse_report(clusters=20, hit_ci_low=0.49)
    gate = _multi_m_coarse_recall_gate(
        report,
        minimum_full_pool_clusters=20,
    )
    assert gate["pass"] is False
    assert gate["deployment_pass"] is False


def test_stage2_ordinary_gate_reports_rare_source_without_blocking_family() -> None:
    def metric(clusters: int, rows: int = 20) -> dict:
        return {
            "row_count": rows,
            "cluster_count": clusters,
            "route_mrr_delta": {"ci_low": 0.0},
        }

    gate = _ordinary_safety_gate(
        {
            "overall": metric(20, 40),
            "per_family": {"alfworld": metric(20, 40)},
            "per_horizon": {"2-3": metric(20, 40)},
            "per_source": {
                "alfworld-large": metric(13, 30),
                "alfworld-rare": metric(7, 10),
            },
        },
        mrr_noninferiority_tolerance=0.01,
        minimum_clusters_per_stratum=10,
        minimum_stratum_coverage=0.90,
    )
    rare = next(item for item in gate["checked"] if item["name"] == "alfworld-rare")
    assert gate["pass"] is True
    assert rare["status"] == "not_testable"
    assert rare["decisive"] is False


def test_stage2_gradient_gate_aggregates_across_validation_interval() -> None:
    interval = _empty_gradient_interval(1, synchronization_enabled=True)
    names = tuple(interval["modules"])
    assert "synchronization" in names
    first = {
        name: {"tensor_count": 1, "finite": True, "norm": 1.0e-4}
        for name in names
    }
    first["correction_delta"] = {"tensor_count": 0, "finite": False, "norm": 0.0}
    first["correction_gate"] = {"tensor_count": 0, "finite": False, "norm": 0.0}
    interval = _update_gradient_interval(interval, first, step=1)
    second = {
        name: {"tensor_count": 0, "finite": False, "norm": 0.0}
        for name in names
    }
    second["correction_delta"] = {"tensor_count": 1, "finite": True, "norm": 2.0e-4}
    second["correction_gate"] = {"tensor_count": 1, "finite": True, "norm": 3.0e-4}
    interval = _update_gradient_interval(interval, second, step=2)
    record = _finalize_gradient_interval(interval, end_step=2)
    assert record["batch_count"] == 2
    assert all(module["finite"] for module in record["modules"].values())
    assert _gradient_record_gate(
        record,
        minimum_gradient_norm=1.0e-10,
        synchronization_required=True,
    )
    without_synchronization = {
        **record,
        "modules": {
            name: values
            for name, values in record["modules"].items()
            if name != "synchronization"
        },
    }
    assert not _gradient_record_gate(
        without_synchronization,
        minimum_gradient_norm=1.0e-10,
        synchronization_required=True,
    )


def test_stage2_prefix_replay_batches_active_trajectories_without_reset() -> None:
    class Cache:
        def batch(self, texts, *, device):
            return torch.tensor(
                [[float(text)] for text in texts],
                dtype=torch.float32,
                device=device,
            )

    class Core:
        synchronization_enabled = True
        synchronization_trace_length = 2

        def append_latent_trace(self, memory, latent_trace):
            return torch.cat((latent_trace, memory.unsqueeze(1)), dim=1)[:, -2:]

        def predict_memory(self, memory, state, skill, action):
            predicted = (memory + state + skill + action).to(torch.bfloat16)
            return (
                predicted,
                predicted - memory.to(dtype=predicted.dtype),
                action.to(dtype=predicted.dtype),
            )

        def correct_memory(
            self,
            predicted,
            state,
            skill,
            action,
            result,
            *,
            action_is_adapted,
        ):
            assert action_is_adapted
            corrected = (predicted + result).to(torch.bfloat16)
            return (
                corrected,
                result.to(dtype=corrected.dtype),
                torch.ones_like(corrected),
            )

    class Model:
        vnext = Core()

        def vnext_initial_belief(self, state, legal, *, top_k):
            assert top_k == 2
            assert bool(legal.all())
            return state

        def vnext_normalized_skill_embeddings(self, *, dtype):
            return torch.tensor([[1.0], [2.0]], dtype=dtype)

    rows_a = [
        {
            "state_text_current": "1",
            "state_text_causal": "1",
            "causal_prefix_events": [],
            "action_text": "0",
            "actual_result_text": "",
            "_vnext_target_skill_id": "a",
            "runtime_visible_catalog_id": "pool",
            "inventory_catalog_digest": "pool-digest",
        },
        {
            "state_text_current": "2",
            "state_text_causal": "2",
            "causal_prefix_events": [],
            "action_text": "0",
            "actual_result_text": "",
            "_vnext_target_skill_id": "b",
            "runtime_visible_catalog_id": "pool",
            "inventory_catalog_digest": "pool-digest",
        },
        {
            "state_text_current": "3",
            "state_text_causal": "3",
            "causal_prefix_events": [],
            "action_text": "0",
            "actual_result_text": "",
            "_vnext_target_skill_id": "a",
            "runtime_visible_catalog_id": "pool",
            "inventory_catalog_digest": "pool-digest",
        },
    ]
    rows_b = [dict(row) for row in rows_a]
    memories = _batch_memory_at_start(
        Model(),
        [(rows_a, 2, 3, False), (rows_b, 1, 2, False)],
        state_cache=Cache(),
        action_cache=Cache(),
        result_cache=None,
        catalogs=CATALOGS,
        skill_id_to_idx={"a": 0, "b": 1, "c": 2},
        device=torch.device("cpu"),
        belief_top_k=2,
    )
    assert memories.dtype == torch.bfloat16
    assert torch.equal(
        memories,
        torch.tensor([[7.0], [3.0]], dtype=torch.bfloat16),
    )
    traced_memories, latent_trace = _batch_memory_at_start(
        Model(),
        [(rows_a, 2, 3, False), (rows_b, 1, 2, False)],
        state_cache=Cache(),
        action_cache=Cache(),
        result_cache=None,
        catalogs=CATALOGS,
        skill_id_to_idx={"a": 0, "b": 1, "c": 2},
        device=torch.device("cpu"),
        belief_top_k=2,
        with_trace=True,
    )
    assert torch.equal(traced_memories, memories)
    assert latent_trace.shape == (2, 2, 1)
    assert torch.equal(
        latent_trace,
        torch.tensor(
            [[[3.0], [7.0]], [[0.0], [3.0]]],
            dtype=torch.bfloat16,
        ),
    )


def _passing_validation(step: int, score: float, worst_low: float) -> dict:
    ci = {"cluster_count": 20, "ci_low": worst_low, "mean": worst_low + 0.1}
    ordinary_metric = {
        "row_count": 20,
        "cluster_count": 20,
        "route_mrr_delta": {"ci_low": 0.0},
        "candidate_recall": {},
    }
    modules = {
        name: {"finite": True, "tensor_count": 1, "norm": 1.0e-3}
        for name in (
            "transition_delta",
            "correction_delta",
            "memory_recall_query",
            "unified_route_query",
            "route_skill_adapter",
            "route_expert_mixture",
            "correction_gate",
        )
    }
    validation = {
        "step": step,
        "selection_score": score,
        "by_kind": {
            "history_branch": {
                "metrics": {
                    "route_advantage": ci,
                    "raw_route_advantage": ci,
                }
            }
        },
        "per_source": {
            "source": {
                "causal_advantage": {"cluster_count": 20},
                "dynamic_static_advantage": {"ci_low": 0.0},
            }
        },
        "dynamic_static_advantage": {"ci_low": 0.0},
        "ordinary_dev": {
            "candidate_support_memory_invariant": False,
            "overall": ordinary_metric,
            "per_family": {"family": ordinary_metric},
            "per_source": {"source": ordinary_metric},
            "per_horizon": {"2-3": ordinary_metric},
        },
        "exact_zero_history_fallback": True,
        "gradient_health": {"step": step, "modules": modules},
        "robust_prefix": {"by_kind": {}, "per_source": {}},
    }
    validation["ordinary_dev"]["overall"]["candidate_recall_at_500"] = {
        "count": 20,
        "cluster_count": 20,
        "mean": 0.8,
        "ci_low": 0.7,
    }
    validation["ordinary_dev"]["overall"]["candidate_union_recall"] = {
        "count": 20,
        "cluster_count": 20,
        "mean": 0.8,
        "ci_low": 0.7,
    }
    validation["ordinary_dev"]["overall"]["candidate_support_change_rate"] = {
        "count": 20,
        "cluster_count": 20,
        "mean": 0.25,
        "ci_low": 0.05,
    }
    validation["ordinary_dev"]["overall"]["candidate_recall"].update(
        {
            "100": {
                "deployment_full_pool": {
                    "fixed_causal_hit_rate": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.85,
                        "ci_low": 0.75,
                    },
                    "dynamic_hit_rate": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.86,
                        "ci_low": 0.76,
                    },
                    "dynamic_minus_static": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.01,
                        "ci_low": 0.0,
                    },
                },
            },
            "500": {
                "deployment_full_pool": {
                    "fixed_causal_hit_rate": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.95,
                        "ci_low": 0.9,
                    },
                    "dynamic_hit_rate": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.96,
                        "ci_low": 0.91,
                    },
                    "dynamic_minus_static": {
                        "count": 20,
                        "cluster_count": 20,
                        "mean": 0.01,
                        "ci_low": 0.0,
                    },
                },
            },
        }
    )
    validation["gates"] = _stage2_validation_gates(
        validation,
        enabled_kind_weights={"history_branch": 0.2},
        minimum_dev_clusters_per_enabled_kind=20,
        minimum_dev_clusters_per_source=5,
        no_regret_tolerance=0.01,
        ordinary_mrr_noninferiority_tolerance=0.01,
        minimum_ordinary_dev_clusters_per_stratum=10,
        minimum_ordinary_dev_stratum_coverage=0.90,
        minimum_full_pool_clusters=20,
        minimum_gradient_norm=1.0e-10,
        require_robust_prefix_curriculum=False,
        robust_mrr_noninferiority_tolerance=0.01,
    )
    return validation


def test_stage2_selector_uses_best_gate_passing_heldout_score() -> None:
    unstable = _passing_validation(100, score=1.0, worst_low=0.02)
    unstable["gates"]["pass"] = False
    first_safe = _passing_validation(200, score=0.5, worst_low=0.05)
    later_stronger = _passing_validation(300, score=0.9, worst_low=0.30)
    selected, reason = _select_stage2_validation(
        [unstable, first_safe, later_stronger]
    )
    assert selected["step"] == 300
    assert reason == "best_gate_passing_heldout_score"


def test_stage2_selector_breaks_equal_safe_scores_toward_earlier_step() -> None:
    earlier = _passing_validation(200, score=0.9, worst_low=0.05)
    later = _passing_validation(300, score=0.9, worst_low=0.30)
    selected, reason = _select_stage2_validation([later, earlier])
    assert selected["step"] == 200
    assert reason == "best_gate_passing_heldout_score"
