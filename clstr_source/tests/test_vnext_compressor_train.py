from __future__ import annotations

from pathlib import Path

from clstr.vnext_compressor_train import (
    _compressor_validation_gates,
    _initialize_static_reranker_scale,
    _natural_support_listwise_nll,
    _prepare_successor_route_rows,
    _schedule_compressor_batches,
    _strict_summary,
)
from clstr.vnext_core import CandidateCompressor, StaticRouteQueryDelta
import torch
from types import SimpleNamespace


def test_compressor_schedule_balances_sources_deterministically() -> None:
    rows = [
        {"source": source, "row_id": f"{source}-{index}"}
        for source, count in (("a", 1), ("b", 5), ("c", 2))
        for index in range(count)
    ]
    first, report = _schedule_compressor_batches(
        rows,
        max_steps=2,
        gradient_accumulation_steps=1,
        batch_size=3,
        seed=37,
    )
    second, _ = _schedule_compressor_batches(
        rows,
        max_steps=2,
        gradient_accumulation_steps=1,
        batch_size=3,
        seed=37,
    )
    assert first == second
    assert report["sampled_exposures"] == {"a": 2, "b": 2, "c": 2}


def test_compressor_strict_summary_counts_natural_misses_as_zero() -> None:
    report = _strict_summary(
        [
            {"full_pool_rank": 80, "direct_rank": 4, "compressed_rank": 2},
            {"full_pool_rank": 300, "direct_rank": None, "compressed_rank": 5},
            {"full_pool_rank": 900, "direct_rank": None, "compressed_rank": None},
        ]
    )
    assert report["full_pool_recall_at_100"] == 1 / 3
    assert report["full_pool_recall_at_500"] == 2 / 3
    assert report["direct_recall_at_64"] == 1 / 3
    assert report["compressed_recall_at_64"] == 2 / 3
    assert report["compressed_mrr"] == (0.5 + 0.2) / 3
    assert report["recovered_at_64"] == 1
    assert report["lost_at_64"] == 0
    assert report["net_recovered_at_64"] == 1
    assert report["rank_improved_rows"] == 1
    assert report["rank_worsened_rows"] == 0


def test_static_reranker_scale_reset_preserves_identity_scores() -> None:
    compressor = CandidateCompressor(8, hidden_dim=4)
    model = SimpleNamespace(vnext=SimpleNamespace(candidate_compressor=compressor))
    causal = torch.randn(2, 8)
    current = torch.randn(2, 8)
    candidates = torch.randn(2, 5, 8)
    base = torch.randn(2, 5)
    valid = torch.ones(2, 5, dtype=torch.bool)
    before = compressor(causal, current, candidates, base, valid).logits
    report = _initialize_static_reranker_scale(model, 1.0)
    after = compressor(causal, current, candidates, base, valid).logits
    assert report["identity_output_verified"] is True
    assert abs(float(report["realized"]) - 1.0) < 1.0e-6
    assert torch.equal(before, after)
    assert torch.equal(after, base)

    with torch.no_grad():
        compressor.output.bias.fill_(0.1)
    try:
        _initialize_static_reranker_scale(model, 1.0)
    except ValueError as exc:
        assert "exact identity output" in str(exc)
    else:
        raise AssertionError("non-identity reranker scale reset must fail closed")


def test_static_route_query_scale_reset_preserves_identity_query() -> None:
    residual = StaticRouteQueryDelta(8, hidden_dim=4)
    model = SimpleNamespace(
        vnext=SimpleNamespace(static_route_query_delta=residual)
    )
    state = torch.randn(2, 8)
    memory = torch.randn(2, 8)
    before = residual(state, memory)
    report = _initialize_static_reranker_scale(
        model,
        1.0,
        objective_mode="static_route_query_residual",
    )
    after = residual(state, memory)
    assert report["identity_output_verified"] is True
    assert report["component"] == "route_query_residual"
    assert abs(float(report["realized"]) - 1.0) < 1.0e-6
    assert torch.equal(before, after)
    assert torch.count_nonzero(after) == 0

    with torch.no_grad():
        residual.adapter.output.bias.fill_(0.1)
    try:
        _initialize_static_reranker_scale(
            model,
            1.0,
            objective_mode="static_route_query_residual",
        )
    except ValueError as exc:
        assert "exact identity output" in str(exc)
    else:
        raise AssertionError("non-identity route adapter reset must fail closed")


def test_compressor_gate_requires_toolbench_noninferiority() -> None:
    good = {
        "row_count": 10,
        "direct_recall_at_64": 0.6,
        "compressed_recall_at_64": 0.7,
        "direct_mrr": 0.2,
        "compressed_mrr": 0.3,
    }
    harmful = {**good, "compressed_mrr": 0.1}
    gates = _compressor_validation_gates(
        {"overall": good, "trajectory": good, "toolbench": harmful}
    )
    assert gates["noninferior_to_direct_top64"] is True
    assert gates["toolbench_noninferior_to_direct_top64"] is False
    assert gates["noninferior_to_base"] is True
    assert gates["toolbench_noninferior_to_base"] is False
    assert gates["pass"] is False


def test_static_reranker_listwise_uses_only_natural_support() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [3.0, 2.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[False, True, False], [False, False, False]])
    valid = torch.ones_like(positive)
    loss, report = _natural_support_listwise_nll(logits, positive, valid)
    assert torch.isfinite(loss)
    assert report["eligible_rows"] == 1
    assert report["natural_miss_rows"] == 1
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_static_reranker_uses_ordered_successor_route_rows() -> None:
    common = {
        "trajectory_id": "trajectory-a",
        "runtime_visible_catalog_id": "global",
        "state_text_current": "goal: use a then b",
        "actual_result_text": "",
        "actual_result_executed": False,
    }
    rows = [
        {
            **common,
            "step_index": 0,
            "decision_step_index": 0,
            "target_skill_id": "a",
            "skill_id": "a",
            "action_text": "call a",
            "causal_prefix_events": [],
            "capabilities": {"ordered_next_tool": False},
        },
        {
            **common,
            "step_index": 1,
            "decision_step_index": 1,
            "target_skill_id": "b",
            "skill_id": "b",
            "action_text": "call b",
            "causal_prefix_events": [
                {
                    "step_index": 0,
                    "skill_id": "a",
                    "action_text": "call a",
                    "result_text": "",
                    "result_executed": False,
                }
            ],
            "capabilities": {"ordered_next_tool": True},
        },
    ]
    prepared, report = _prepare_successor_route_rows(
        rows,
        {"a", "b"},
        {"global": {"runtime_visible_skill_ids": ["a", "b"]}},
    )
    assert report["trajectory_count"] == 1
    assert [row["_vnext_target_skill_id"] for row in prepared] == ["a", "b"]
    assert prepared[0]["_vnext_causal_query"] == "goal: use a then b"
    assert "event[0].skill_id: a" in prepared[1]["_vnext_causal_query"]
    assert prepared[1]["_vnext_positive_skill_ids"] == ["b"]


def test_mainline_is_natural_top500_static_route_query_without_gt_injection() -> None:
    root = Path(__file__).parents[1]
    source = root.joinpath("clstr/vnext_compressor_train.py").read_text(
        encoding="utf-8"
    )
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_candidate_compressor.sh"
    ).read_text(encoding="utf-8")
    assert "route_training_candidates" not in source
    assert "common_candidate_support" not in source
    assert "label_natural_support" in source
    assert '"positive_injection_count": 0' in source
    assert '"selected_checkpoint_sha256"' in source
    assert '"selected_candidate_foundation_digest"' in source
    assert '"toolbench_noninferior_to_base"' in source
    assert "natural_candidate_topk_coverage_loss" in source
    assert "natural_top500_static_route_query_listwise_v1" in source
    assert 'objective_mode == "static_route_query_residual"' in source
    assert "zero-initialized static route adapter changed step-zero logits" in source
    assert "base_logits + route_delta_logits" in source
    assert "exact candidate kernel deployed by" in source
    assert "--coarse_k 500" in launcher
    assert "--compressed_m 500" in launcher
    assert '--trajectory_rows_path "${TRAJECTORY_ROWS}"' in launcher
    assert '--trajectory_dev_rows_path "${TRAJECTORY_DEV_ROWS}"' in launcher
    assert '--retrieval_rows_path "${RETRIEVAL_ROWS}"' not in launcher
    assert "--objective_mode static_route_query_residual" in launcher
    assert "--static_reranker_scale_initial" in launcher
    assert "STATIC_RERANKER_SCALE_INITIAL=${STATIC_RERANKER_SCALE_INITIAL:-1.0}" in launcher
    assert 'report = json.loads((root / "train_report.json").read_text())' in launcher
    assert 'checkpoint = Path(str(report.get("checkpoint_path") or ""))' in launcher
    assert "Stage0 report does not name its final checkpoint" in launcher
    assert "gpu_a800,gpu_h100,gpu_h200" in launcher
    assert "--require_clean_source" in launcher


def test_stage2_and_eval_share_static_top500_memory_extra_union() -> None:
    root = Path(__file__).parents[1]
    stage2 = root.joinpath("clstr/vnext_stage2_train.py").read_text(
        encoding="utf-8"
    )
    evaluator = root.joinpath("clstr/vnext_eval.py").read_text(encoding="utf-8")
    stage2_launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_full_stage2_segment.sh"
    ).read_text(encoding="utf-8")
    smoke_launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_stage2_smoke.sh"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "route_training_candidates",
        "common_candidate_support",
        "maximum_route_positive_injection_rate",
    ):
        assert forbidden not in stage2
    assert "_memory_conditioned_natural_support" in stage2
    assert "vnext_natural_candidate_union_path" in stage2
    assert "vnext_natural_candidate_union_path" in evaluator
    assert "training_only_teacher_retained_support" in stage2
    assert "CANDIDATE_OUTPUT_DIR must be the matching static-reranker smoke" in smoke_launcher
    assert 'compressor_selection.json' in smoke_launcher
    assert '--candidate_checkpoint_path "${CANDIDATE_CHECKPOINT_PATH}"' in smoke_launcher
    assert "candidate_foundation=static_scored" in evaluator
    assert '"candidate_foundation_reused_across_memory_alternatives": (' in evaluator
    assert "None if foundation_selected else False" in evaluator
    assert '"memory_changes_candidate_support": bool' in evaluator
    assert "candidate IDs cannot be frozen or cached" in stage2
    assert '"candidate_foundation_digest"' in stage2
    assert 'inputs.get("training_rows_kind")' in stage2
    assert "natural_top500_static_route_query_residual_v1" in stage2
    assert "--stage0_checkpoint_path" in stage2_launcher
    assert "--candidate_checkpoint_path" in stage2_launcher
    assert "compressor_selection.json" in stage2_launcher
    assert "--coarse_k 500" in stage2_launcher
    assert "--compressed_m 64" in stage2_launcher
    assert "--final_k 100" in stage2_launcher
    assert '--lambda_recall "${LAMBDA_RECALL}"' in stage2_launcher
    assert '--lambda_compression "${LAMBDA_COMPRESSION}"' in stage2_launcher
    assert "--candidate_learning_rate_scale 1.0" in stage2_launcher
    assert "--static_route_learning_rate_scale 1.0" in stage2_launcher
    assert '"group_role": "recurrent"' in stage2
    assert '"group_role": "memory_candidate_proposal"' in stage2
    assert '"group_role": "unbounded_unified_memory_residual"' in stage2
    assert '"static_route_foundation_digest"' in stage2


def test_static_reranker_smoke_matches_full_candidate_contract() -> None:
    root = Path(__file__).parents[1]
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_candidate_compressor_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "BUNDLE_ROOT must come from the freshly audited causal manifest" in launcher
    assert "STAGE0_OUTPUT_DIR must be the matching Stage0 smoke" in launcher
    assert "FROZEN_CACHE_DIR must identify the matching smoke cache" in launcher
    assert "--belief_top_k 64" in launcher
    assert "--coarse_k 500" in launcher
    assert "--compressed_m 500" in launcher
    assert "--objective_mode static_route_query_residual" in launcher
    assert "--require_clean_source" in launcher
