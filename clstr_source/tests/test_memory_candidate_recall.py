import pytest
import torch

from clstr.memory_candidate_recall import (
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_UNION_VERSION,
    TIE_BREAK_POLICY,
    build_static_dynamic_union,
    candidate_provenance_mask,
    candidate_recall_applicability,
    candidate_recall_protocol_report,
    candidate_recall_report,
    explicit_inventory_skill_ids_ordered,
    full_pool_positive_mask,
    legal_skill_pool_mask,
    stable_masked_topk_rows,
)
from scripts.benchmark_memory_candidate_selection import benchmark_candidate_selection


def test_stable_masked_topk_rows_breaks_boundary_ties_by_declared_index():
    logits = torch.tensor([[4.0, 3.0, 3.0, 3.0, 1.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    selected = stable_masked_topk_rows(logits, valid, k=3)

    assert selected == [[0, 1, 2]]
    assert CANDIDATE_SELECTION_VERSION == "stable_declared_pool_v1"
    assert TIE_BREAK_POLICY == "declared_pool_index_ascending"


def test_stable_masked_topk_rows_respects_sparse_valid_mask_and_short_rows():
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0], [4.0, 3.0, 2.0, 1.0]])
    valid = torch.tensor([[False, True, False, True], [False, False, False, True]])

    selected = stable_masked_topk_rows(logits, valid, k=3)

    assert selected == [[1, 3], [3]]


def test_stable_masked_topk_rows_rejects_nonfinite_valid_logits():
    logits = torch.tensor([[1.0, float("nan"), 0.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    with pytest.raises(ValueError, match="finite"):
        stable_masked_topk_rows(logits, valid, k=2)


def test_candidate_selection_benchmark_reports_selector_contract():
    report = benchmark_candidate_selection(
        batch_size=2,
        skill_count=16,
        hidden_size=4,
        static_k=4,
        dynamic_extra_k=2,
        warmup=0,
        repeats=2,
        device=torch.device("cpu"),
    )

    assert report["candidate_selection_version"] == CANDIDATE_SELECTION_VERSION
    assert report["tie_break_policy"] == TIE_BREAK_POLICY
    assert report["selector_latency_seconds"] >= 0.0
    assert report["topk_latency_seconds"] >= 0.0
    assert report["end_to_end_overhead_fraction"] >= 0.0


def test_explicit_inventory_skill_ids_ordered_flattens_and_deduplicates():
    row = {
        "visible_inventory_skill_ids": ["skill/a", {"skill_id": "skill/b"}],
        "available_skills": [{"id": "skill/b"}, ["skill/c", "skill/a"]],
    }

    assert explicit_inventory_skill_ids_ordered(row) == ["skill/a", "skill/b", "skill/c"]


def test_full_pool_masks_use_aliases_and_explicit_inventory_without_target_inference():
    rows = [
        {
            "next_skill_id": "skill/gold",
            "visible_inventory_skill_ids": ["skill/gold-alias", "skill/negative"],
        },
        {
            "next_skill_id": "skill/gold",
            "visible_inventory_skill_ids": ["skill/unknown"],
        },
    ]
    mapping = {"skill/gold": 0, "skill/gold-alias": 1, "skill/negative": 2}

    positives = full_pool_positive_mask(
        rows,
        mapping,
        {"skill/gold": ["skill/gold-alias"]},
        skill_count=3,
        device=torch.device("cpu"),
    )
    legal = legal_skill_pool_mask(
        rows,
        mapping,
        skill_count=3,
        device=torch.device("cpu"),
    )

    assert positives.tolist() == [[True, True, False], [True, True, False]]
    assert legal.mask.tolist() == [[False, True, True], [False, False, False]]
    assert legal.explicit_inventory_no_known_skill_mask.tolist() == [False, True]


def test_full_pool_positive_mask_unions_row_local_equivalent_targets():
    rows = [
        {
            "next_skill_id": "skill/a",
            "equivalent_next_skill_ids": ["skill/c", "skill/missing"],
        }
    ]
    mapping = {"skill/a": 0, "skill/b": 1, "skill/c": 2}

    positives = full_pool_positive_mask(
        rows,
        mapping,
        None,
        skill_count=3,
        device=torch.device("cpu"),
    )

    assert positives.tolist() == [[True, False, True]]


def test_legal_pool_uses_row_candidates_when_explicit_inventory_is_absent():
    rows = [
        {
            "next_skill_id": "skill/b",
            "candidate_next_skill_ids": ["skill/b", "skill/c"],
            "candidate_pool_protocol": "benchmark_local",
        }
    ]
    mapping = {"skill/a": 0, "skill/b": 1, "skill/c": 2, "skill/d": 3}

    legal = legal_skill_pool_mask(
        rows,
        mapping,
        skill_count=4,
        device=torch.device("cpu"),
    )

    assert legal.mask.tolist() == [[False, True, True, False]]
    assert legal.explicit_inventory_no_known_skill_mask.tolist() == [False]


def test_static_dynamic_union_preserves_static_and_adds_dynamic_only_candidates():
    static = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    dynamic = torch.tensor([[1.0, 2.0, 3.0, 10.0, 11.0]])
    valid = torch.ones_like(static, dtype=torch.bool)

    union = build_static_dynamic_union(
        static,
        dynamic,
        valid,
        static_k=2,
        dynamic_extra_k=2,
    )

    assert union.static_rows == [[0, 1]]
    assert union.dynamic_top_rows == [[4, 3]]
    assert union.dynamic_extra_rows == [[4, 3]]
    assert union.candidate_rows == [[0, 1, 4, 3]]
    assert union.static_equal_budget_rows == [[0, 1, 2, 3]]
    assert len(union.candidate_rows[0]) == len(set(union.candidate_rows[0]))
    assert CANDIDATE_UNION_VERSION == "memory_union_v1"


def test_candidate_provenance_mask_tracks_dynamic_extras_after_reordering():
    valid = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )

    mask = candidate_provenance_mask(
        candidate_rows=[[4, 0, 3, 1], [2, 1]],
        dynamic_extra_rows=[[4, 3], []],
        valid_mask=valid,
    )

    assert mask.tolist() == [
        [True, False, True, False],
        [False, False, False, False],
    ]


def test_equal_static_dynamic_scores_reproduce_equal_budget_static_topk_exactly():
    logits = torch.tensor(
        [
            [4.0, 3.0, 3.0, 3.0, 1.0],
            [5.0, 5.0, 4.0, 4.0, 3.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, True, True],
            [False, True, True, True, False],
        ]
    )

    union = build_static_dynamic_union(
        logits,
        logits,
        valid,
        static_k=2,
        dynamic_extra_k=2,
    )

    assert union.candidate_rows == union.static_equal_budget_rows
    assert union.candidate_rows == [[0, 1, 3, 4], [1, 2, 3]]


def test_static_dynamic_union_respects_invalid_entries_and_small_pools():
    static = torch.tensor([[100.0, 3.0, 2.0, 1.0]])
    dynamic = torch.tensor([[100.0, 1.0, 4.0, 3.0]])
    valid = torch.tensor([[False, True, True, False]])

    union = build_static_dynamic_union(
        static,
        dynamic,
        valid,
        static_k=3,
        dynamic_extra_k=4,
    )

    assert union.static_rows == [[1, 2]]
    assert union.dynamic_top_rows == [[2, 1]]
    assert union.dynamic_extra_rows == [[]]
    assert union.candidate_rows == [[1, 2]]
    assert 0 not in union.candidate_rows[0]
    assert 3 not in union.candidate_rows[0]


def test_candidate_recall_report_keeps_unknown_and_illegal_targets_in_strict_denominator():
    static = torch.tensor(
        [
            [9.0, 2.0, 1.0, 0.0],
            [1.0, 9.0, 2.0, 0.0],
            [9.0, 2.0, 1.0, 0.0],
            [9.0, 2.0, 1.0, 0.0],
        ]
    )
    dynamic = torch.tensor(
        [
            [1.0, 2.0, 10.0, 0.0],
            [1.0, 8.0, 7.0, 0.0],
            [1.0, 10.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 10.0],
        ]
    )
    legal = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    positives = torch.tensor(
        [
            [False, False, True, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, True],
        ]
    )
    union = build_static_dynamic_union(
        static,
        dynamic,
        legal,
        static_k=1,
        dynamic_extra_k=1,
    )

    report = candidate_recall_report(
        union,
        positives,
        legal,
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )

    all_rows = report["all_eligible_source_strict"]
    active = report["memory_active_strict"]
    assert all_rows["source_rows"] == 4.0
    assert all_rows["static_recall"] == pytest.approx(0.25)
    assert all_rows["union_recall"] == pytest.approx(0.5)
    assert all_rows["static_hit_rows"] == 1.0
    assert all_rows["union_hit_rows"] == 2.0
    assert all_rows["static_miss_rows"] == 3.0
    assert all_rows["dynamic_rescue_rows"] == 1.0
    assert all_rows["static_miss_recovery_rate"] == pytest.approx(1.0 / 3.0)
    assert active["source_rows"] == 3.0
    assert active["union_recall"] == pytest.approx(2.0 / 3.0)
    assert active["union_hit_rows"] == 2.0
    assert report["memory_active_coverage"] == pytest.approx(0.75)
    assert report["positive_not_in_declared_pool_rows"] == 1.0
    assert report["positive_outside_legal_pool_rows"] == 1.0
    assert report["candidate_union_version"] == CANDIDATE_UNION_VERSION


def test_candidate_recall_applicability_distinguishes_global_local_and_environment_protocols():
    global_report = candidate_recall_applicability(
        pool_protocol="known_global",
        legal_pool_size=1000,
        static_k=100,
        dynamic_extra_k=20,
        causal_sequential=True,
    )
    saturated = candidate_recall_applicability(
        pool_protocol="benchmark_local",
        legal_pool_size=80,
        static_k=64,
        dynamic_extra_k=16,
        causal_sequential=True,
    )
    environment = candidate_recall_applicability(
        pool_protocol="environment_candidates",
        legal_pool_size=12,
        static_k=8,
        dynamic_extra_k=4,
        causal_sequential=True,
    )

    assert global_report == {
        "candidate_recall_applicable": True,
        "candidate_recall_applicability_reason": "known_global_sequential",
        "candidate_recall_saturated": False,
    }
    assert saturated == {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
        "candidate_recall_saturated": True,
    }
    assert environment == {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "environment_admissible_actions",
        "candidate_recall_saturated": False,
    }


def _retained_candidate_recall_metrics() -> dict[str, float | str]:
    return {
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "candidate_union_version": CANDIDATE_UNION_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "candidate_tie_break_policy": TIE_BREAK_POLICY,
        "candidate_recall_all_source_rows": 2.0,
        "candidate_recall_all_static_hit_rows": 1.0,
        "candidate_recall_all_dynamic_top_hit_rows": 1.0,
        "candidate_recall_all_dynamic_extra_hit_rows": 1.0,
        "candidate_recall_all_union_hit_rows": 2.0,
        "candidate_recall_all_static_equal_budget_hit_rows": 1.0,
        "candidate_recall_all_static_miss_rows": 1.0,
        "candidate_recall_all_dynamic_rescue_rows": 1.0,
        "candidate_recall_all_static_dynamic_overlap_sum": 0.5,
        "candidate_recall_all_dynamic_only_candidate_total": 4.0,
        "candidate_recall_memory_active_source_rows": 1.0,
        "candidate_recall_memory_active_static_hit_rows": 0.0,
        "candidate_recall_memory_active_dynamic_top_hit_rows": 1.0,
        "candidate_recall_memory_active_dynamic_extra_hit_rows": 1.0,
        "candidate_recall_memory_active_union_hit_rows": 1.0,
        "candidate_recall_memory_active_static_equal_budget_hit_rows": 0.0,
        "candidate_recall_memory_active_static_miss_rows": 1.0,
        "candidate_recall_memory_active_dynamic_rescue_rows": 1.0,
        "candidate_recall_memory_active_static_dynamic_overlap_sum": 0.0,
        "candidate_recall_memory_active_dynamic_only_candidate_total": 2.0,
    }


def test_candidate_recall_protocol_report_rebases_primary_metrics_to_source_denominator():
    report = candidate_recall_protocol_report(
        _retained_candidate_recall_metrics(),
        pool_protocol="known_global",
        candidate_source="declared_legal_full_skill_pool",
        legal_pool_size=1000,
        source_rows=4,
        static_k=100,
        dynamic_extra_k=20,
        final_k=20,
        causal_sequential=True,
    )

    strict = report["all_eligible_source_strict"]
    active = report["memory_active_strict"]
    assert report["candidate_recall_applicable"] is True
    assert strict["source_rows"] == 4.0
    assert strict["retained_evaluator_rows"] == 2.0
    assert strict["static_recall"] == pytest.approx(0.25)
    assert strict["union_recall"] == pytest.approx(0.5)
    assert strict["static_miss_rows"] == 3.0
    assert strict["static_miss_recovery_rate"] == pytest.approx(1.0 / 3.0)
    assert strict["static_dynamic_overlap"] == pytest.approx(0.125)
    assert strict["dynamic_only_candidate_count"] == pytest.approx(1.0)
    assert active["source_rows"] == 1.0
    assert active["union_recall"] == 1.0
    assert report["memory_active_coverage"] == pytest.approx(0.25)
    assert report["equal_budget_comparator_mode"] == "same_final_scorer_static_top_m_plus_d"


def test_candidate_recall_protocol_report_keeps_target_outside_pool_as_strict_zero():
    report = candidate_recall_protocol_report(
        _retained_candidate_recall_metrics(),
        pool_protocol="known_global",
        candidate_source="declared_legal_full_skill_pool",
        legal_pool_size=1000,
        source_rows=3,
        static_k=100,
        dynamic_extra_k=20,
        final_k=20,
        causal_sequential=True,
        target_outside_declared_legal_pool_rows=1,
    )

    assert report["all_eligible_source_strict"]["union_recall"] == pytest.approx(2.0 / 3.0)
    assert report["positive_not_in_declared_pool_rows"] == 1.0
    assert report["positive_outside_legal_pool_rows"] == 1.0
    assert report["protocol_blockers"] == ["target_outside_declared_legal_pool"]
