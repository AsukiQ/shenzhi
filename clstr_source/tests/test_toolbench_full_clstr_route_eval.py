from __future__ import annotations

from pathlib import Path

import pytest
import torch

from clstr.toolbench_full_clstr_route_eval import (
    apply_transition_inventory_filter_to_stage4_rows,
    build_toolbench_full_clstr_route_report,
    run_toolbench_full_clstr_route_eval,
    strict_stage4_metrics,
)


def test_strict_stage4_metrics_counts_missing_rows_as_zero():
    metrics = {
        "stage4_next_skill_recall@1": 0.25,
        "stage4_next_skill_recall@5": 0.75,
        "stage4_next_skill_mrr": 0.5,
        "stage4_candidate_count": 48.0,
    }

    strict = strict_stage4_metrics(metrics, retained_rows=80, source_rows=100)

    assert strict["strict_stage4_next_skill_recall@1"] == 0.2
    assert strict["strict_stage4_next_skill_recall@5"] == pytest.approx(0.6)
    assert strict["strict_stage4_next_skill_mrr"] == 0.4
    assert strict["retained_row_fraction"] == 0.8
    assert strict["retained_stage4_candidate_count"] == 48.0


def test_build_report_records_full_route_and_clean_split_inputs():
    report = build_toolbench_full_clstr_route_report(
        output_dir="outputs/run",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        train_trajectories_path="train.jsonl",
        eval_trajectories_path="eval.jsonl",
        skills_path="skills.jsonl",
        source_eval_rows=10,
        retained_eval_rows=8,
        train_feedback_rows=20,
        target_data_report={"skipped_reasons": {"missing": 2}},
        train_data_report={"stage4_rows": 20},
        memory_selection={"selected_mode_by_benchmark": {"toolbench_g3": "latest_exact"}},
        prior_eval={"stage4_next_skill_recall@1": 0.25, "stage4_next_skill_mrr": 0.5},
        stage4_eval={"stage4_next_skill_recall@1": 0.5, "stage4_next_skill_mrr": 0.625},
        config={"top_m": 350},
    )

    assert report["status"] == "ok"
    assert report["stage4_route"] == "logged_online_memory"
    assert report["stage4_checkpoint_path"] == "stage4.pt"
    assert report["train_trajectories_path"] == "train.jsonl"
    assert report["eval_trajectories_path"] == "eval.jsonl"
    assert report["strict"]["stage4"]["strict_stage4_next_skill_recall@1"] == 0.4
    assert report["strict"]["prior"]["strict_stage4_next_skill_mrr"] == 0.4
    assert report["strict_delta"]["strict_stage4_next_skill_recall@1"] == 0.2
    assert report["metric_contract"] == {
        "primary_metric_scope": "strict",
        "retained_metrics_scope": "diagnostic_only",
        "dropped_candidate_handoff_rows": "counted_as_zero_in_strict_metrics",
        "main_table_required_prefix": "strict_",
    }


def test_full_route_inventory_filter_preserves_gold_and_backfills_stage0_order():
    skill_id_to_idx = {
        "toolbench-g3/search": 0,
        "other/noise_a": 1,
        "toolbench-g3/gold": 2,
        "other/noise_b": 3,
        "toolbench-g3/refine": 4,
    }
    rows = [
        {
            "source_benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/gold",
            "positive_next_skill_idx": 2,
            "positive_next_skill_position": 2,
            "candidate_next_skill_ids": [
                "toolbench-g3/search",
                "other/noise_a",
                "toolbench-g3/gold",
                "other/noise_b",
                "toolbench-g3/refine",
            ],
            "candidate_next_skill_indices": [0, 1, 2, 3, 4],
            "candidate_next_prior_scores": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    ]

    filtered, report = apply_transition_inventory_filter_to_stage4_rows(
        rows,
        skill_id_to_idx,
        transition_inventory_mask_mode="auto",
        transition_inventory_min_candidates=4,
    )

    assert report["enabled"] is True
    assert report["mode"] == "auto"
    assert report["min_candidates"] == 4
    assert report["audit"]["inventory_mask_applied_rows"] == 1
    assert report["audit"]["inventory_mask_backfilled_rows"] == 1
    assert filtered[0]["candidate_next_skill_ids"] == [
        "toolbench-g3/search",
        "toolbench-g3/gold",
        "toolbench-g3/refine",
        "other/noise_a",
    ]
    assert filtered[0]["candidate_next_skill_indices"] == [0, 2, 4, 1]
    assert filtered[0]["positive_next_skill_position"] == 1
    assert len(filtered[0]["candidate_next_prior_scores"]) == 4


def test_eval_inventory_filter_does_not_repair_gold_with_label():
    skill_id_to_idx = {
        "toolbench-g3/search": 0,
        "other/gold": 1,
        "toolbench-g3/refine": 2,
    }
    rows = [
        {
            "source_benchmark": "toolbench_g3",
            "next_skill_id": "other/gold",
            "positive_next_skill_idx": 1,
            "positive_next_skill_position": 1,
            "candidate_next_skill_ids": [
                "toolbench-g3/search",
                "other/gold",
                "toolbench-g3/refine",
            ],
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [3.0, 2.0, 1.0],
        }
    ]

    filtered, report = apply_transition_inventory_filter_to_stage4_rows(
        rows,
        skill_id_to_idx,
        transition_inventory_mask_mode="auto",
        transition_inventory_min_candidates=0,
        preserve_positive=False,
    )

    assert filtered == []
    assert report["positive_removed_rows"] == 1
    assert report["output_rows"] == 0


def test_toolbench_full_route_cli_and_sbatch_expose_inventory_filter():
    cli = Path("scripts/run_toolbench_g3_full_clstr_route_eval.py").read_text(encoding="utf-8")
    sbatch = Path("scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh").read_text(encoding="utf-8")

    assert "--transition_inventory_mask_mode" in cli
    assert "--transition_inventory_min_candidates" in cli
    assert "transition_inventory_mask_mode=args.transition_inventory_mask_mode" in cli
    assert "transition_inventory_min_candidates=args.transition_inventory_min_candidates" in cli
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}" in sbatch
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in sbatch
    assert "--transition_inventory_mask_mode" in sbatch
    assert "--transition_inventory_min_candidates" in sbatch
    assert "checkpoint_state_query" in cli
    assert "--reliability_mode" in cli
    assert "--fixed_alpha" in cli
    assert "reliability_mode=args.reliability_mode" in cli
    assert "fixed_alpha=args.fixed_alpha" in cli
    assert "RELIABILITY_MODE=${RELIABILITY_MODE:-dynamic}" in sbatch
    assert "FIXED_ALPHA=${FIXED_ALPHA:-1.0}" in sbatch
    assert "--reliability_mode" in sbatch
    assert "--fixed_alpha" in sbatch


def test_toolbench_full_route_threads_known_pool_candidate_union(tmp_path, monkeypatch):
    import clstr.toolbench_full_clstr_route_eval as route_eval

    skills_path = tmp_path / "skills.jsonl"
    skills_path.write_text(
        '{"skill_id":"skill/a"}\n'
        '{"skill_id":"skill/b","alias_skill_ids":["skill/b_alias"]}\n'
        '{"skill_id":"skill/c"}\n',
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        '{"benchmark":"toolbench_g3","trajectory_id":"traj-1","step_index":1,'
        '"state_text":"state","action_text":"call a","next_observation_text":"next",'
        '"next_state_text":"next","skill_id":"skill/a","next_skill_id":"skill/b"}\n',
        encoding="utf-8",
    )
    stage0_checkpoint_path = tmp_path / "stage0.pt"
    stage2_checkpoint_path = tmp_path / "stage2.pt"
    stage0_checkpoint_path.write_bytes(b"stage0")
    stage2_checkpoint_path.write_bytes(b"stage2")
    route_records_path = tmp_path / "memory_utility_routes.jsonl"

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        route_eval,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (FakeModel(), {}, {}),
    )
    monkeypatch.setattr(
        route_eval,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {},
    )
    row_calls = []

    def fake_stage4_rows(**kwargs):
        row_calls.append(dict(kwargs))
        rows = [dict(kwargs["source_rows"][0], source_benchmark="toolbench_g3")]
        return rows, {
            "stage4_rows": 1,
            "positive_injected_rows": 0,
            "candidate_source": "declared_legal_full_skill_pool",
            "next_skill_pool_mode": kwargs["next_skill_pool_mode"],
        }

    monkeypatch.setattr(route_eval, "_stage4_rows_from_source_rows", fake_stage4_rows)
    monkeypatch.setattr(
        route_eval,
        "apply_transition_inventory_filter_to_stage4_rows",
        lambda rows, *_args, **_kwargs: (list(rows), {"enabled": False}),
    )
    monkeypatch.setattr(
        route_eval,
        "attach_trajectory_prefix_online_memory_scores",
        lambda rows, **_kwargs: (list(rows), {}),
    )
    eval_calls = []

    def fake_eval(_model, _rows, **kwargs):
        eval_calls.append(dict(kwargs))
        return {
            "marker": "prior" if len(eval_calls) == 1 else "stage4",
            "stage4_act_count": 1.0,
            "stage4_candidate_count": 2.0,
            "stage4_next_skill_recall@1": 1.0,
            "stage4_next_skill_recall@5": 1.0,
            "stage4_next_skill_mrr": 1.0,
            "candidate_recall_mode": "static_plus_dynamic_extra",
            "candidate_union_version": "memory_union_v1",
            "candidate_selection_version": "stable_declared_pool_v1",
            "candidate_tie_break_policy": "declared_pool_index_ascending",
            "candidate_recall_all_source_rows": 1.0,
            "candidate_recall_all_static_hit_rows": 1.0,
            "candidate_recall_all_dynamic_top_hit_rows": 1.0,
            "candidate_recall_all_dynamic_extra_hit_rows": 0.0,
            "candidate_recall_all_union_hit_rows": 1.0,
            "candidate_recall_all_static_equal_budget_hit_rows": 1.0,
            "candidate_recall_all_dynamic_rescue_rows": 0.0,
            "candidate_recall_all_static_dynamic_overlap_sum": 1.0,
            "candidate_recall_all_dynamic_only_candidate_total": 1.0,
            "candidate_recall_memory_active_source_rows": 1.0,
            "candidate_recall_memory_active_static_hit_rows": 1.0,
            "candidate_recall_memory_active_dynamic_top_hit_rows": 1.0,
            "candidate_recall_memory_active_dynamic_extra_hit_rows": 0.0,
            "candidate_recall_memory_active_union_hit_rows": 1.0,
            "candidate_recall_memory_active_static_equal_budget_hit_rows": 1.0,
            "candidate_recall_memory_active_dynamic_rescue_rows": 0.0,
            "candidate_recall_memory_active_static_dynamic_overlap_sum": 1.0,
            "candidate_recall_memory_active_dynamic_only_candidate_total": 1.0,
        }

    def fail_grouped_eval(*_args, **_kwargs):
        raise AssertionError("single-benchmark ToolBench must not rescore by benchmark")

    monkeypatch.setattr(route_eval, "evaluate_logged_online_stage4_rows", fake_eval)
    monkeypatch.setattr(
        route_eval,
        "evaluate_logged_online_stage4_rows_by_benchmark",
        fail_grouped_eval,
        raising=False,
    )

    sentinel_gate = torch.nn.Identity()
    monkeypatch.setattr(
        route_eval,
        "resolve_reliability_gate",
        lambda **_kwargs: (
            sentinel_gate,
            {
                "feature_update_count_cap": 4.0,
                "feature_candidate_count_cap": 64.0,
            },
        ),
        raising=False,
    )
    report = run_toolbench_full_clstr_route_eval(
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        train_trajectories_path=tmp_path / "unused_train.jsonl",
        eval_trajectories_path=eval_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        top_m=1,
        dynamic_extra_k=1,
        candidate_count=1,
        batch_size=1,
        feedback_mode="eval_prefix",
        reliability_mode="learned",
        fixed_alpha=1.0,
        memory_utility_gate_checkpoint_path=tmp_path / "gate.pt",
        expected_memory_utility_gate_checkpoint_sha256="gate-sha",
        expected_memory_utility_gate_audit_sha256="audit-sha",
        route_records_path=route_records_path,
    )

    assert row_calls[0]["next_skill_pool_mode"] == "full_pool"
    assert row_calls[0]["require_next_state_text"] is False
    assert len(eval_calls) == 2
    for call in eval_calls:
        assert call["candidate_recall_mode"] == "static_plus_dynamic_extra"
        assert call["skill_id_to_idx"] == {"skill/a": 0, "skill/b": 1, "skill/c": 2}
        assert call["equivalent_skill_ids_by_skill_id"]["skill/b"] == ["skill/b_alias"]
        assert call["static_k"] == 1
        assert call["dynamic_extra_k"] == 1
        assert call["final_k"] == 1
    assert [call["reliability_mode"] for call in eval_calls] == ["static", "learned"]
    assert [call["fixed_alpha"] for call in eval_calls] == [1.0, 1.0]
    assert eval_calls[1]["memory_utility_gate"] is sentinel_gate
    assert eval_calls[1]["feature_update_count_cap"] == pytest.approx(4.0)
    assert eval_calls[1]["feature_candidate_count_cap"] == pytest.approx(64.0)
    record_calls = [call for call in eval_calls if "route_records_path" in call]
    assert len(record_calls) == 1
    assert record_calls[0]["route_records_path"] == route_records_path
    assert record_calls[0]["route_record_pool_protocol"] == "known_global"
    assert record_calls[0]["route_record_model_digest"]
    assert record_calls[0]["route_record_sequential_benchmarks"] == {"toolbench_g3"}
    assert report["candidate_recall"]["candidate_recall_applicable"] is True
    assert report["candidate_recall"]["candidate_recall_applicability_reason"] == "known_global_sequential"
    assert report["prior_eval_by_benchmark"] == {
        "toolbench_g3": report["prior_eval"]
    }
    assert report["stage4_eval_by_benchmark"] == {
        "toolbench_g3": report["stage4_eval"]
    }
    assert report["prior_eval"]["marker"] == "prior"
    assert report["stage4_eval"]["marker"] == "stage4"
    assert report["config"]["reliability_mode"] == "learned"
    assert report["config"]["fixed_alpha"] == pytest.approx(1.0)


def test_toolbench_candidate_union_rejects_namespace_inventory_filter(tmp_path):
    with pytest.raises(ValueError, match="transition_inventory_mask_mode=off"):
        run_toolbench_full_clstr_route_eval(
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            train_trajectories_path=tmp_path / "train.jsonl",
            eval_trajectories_path=tmp_path / "eval.jsonl",
            skills_path=tmp_path / "skills.jsonl",
            output_dir=tmp_path / "out",
            transition_inventory_mask_mode="auto",
        )


@pytest.mark.parametrize(
    ("reliability_mode", "fixed_alpha", "message"),
    [
        ("learned", 0.5, "learned reliability requires a gate checkpoint"),
        ("fixed_alpha", 1.1, "fixed_alpha must be finite and in \\[0, 1\\]"),
    ],
)
def test_toolbench_outer_eval_rejects_unsafe_reliability_modes(
    tmp_path,
    reliability_mode,
    fixed_alpha,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_toolbench_full_clstr_route_eval(
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            train_trajectories_path=tmp_path / "train.jsonl",
            eval_trajectories_path=tmp_path / "eval.jsonl",
            skills_path=tmp_path / "skills.jsonl",
            output_dir=tmp_path / "out",
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
        )
