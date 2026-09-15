from __future__ import annotations

from pathlib import Path

import pytest
import torch
import clstr.global_pool_route_eval as global_pool_route_eval

from clstr.global_pool_route_eval import (
    GlobalPoolCorpus,
    _eval_by_benchmark_with_single_group_reuse,
    load_prebuilt_global_pool_corpus,
    merge_global_skill_pool,
    run_global_pool_clstr_route_eval,
)
from clstr.global_pool_skillrouter_eval import load_toolbench_g3_global_pool_corpus
from scripts.run_global_pool_skillrouter_eval import _build_corpus


def test_merge_global_skill_pool_preserves_base_prefix_and_appends_missing():
    base = [
        {"skill_id": "base/a", "name": "A"},
        {"skill_id": "bench/existing", "name": "Existing"},
    ]
    benchmark = [
        {"skill_id": "bench/existing", "name": "Existing updated"},
        {"skill_id": "bench/new", "name": "New"},
    ]

    merged, report = merge_global_skill_pool(base, benchmark, dynamic_source_label="bench")

    assert [row["skill_id"] for row in merged] == ["base/a", "bench/existing", "bench/new"]
    assert merged[1]["name"] == "Existing"
    assert merged[2]["name"] == "New"
    assert report["base_skill_count"] == 2
    assert report["benchmark_skill_count"] == 2
    assert report["appended_skill_count"] == 1
    assert report["existing_skill_count"] == 1
    assert report["dynamic_source_label"] == "bench"


def test_eval_by_benchmark_single_group_reuses_overall_without_expensive_call():
    rows = [{"benchmark": "bfcl"}, {"benchmark": "bfcl"}]
    overall = {"stage4_next_skill_mrr": 0.25, "stage4_act_count": 2.0}

    def expensive_eval(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("single benchmark should reuse the already computed overall eval")

    result = _eval_by_benchmark_with_single_group_reuse(
        rows,
        overall,
        expensive_eval,
        model=object(),
        batch_size=8,
    )

    assert result == {"bfcl": overall}
    assert result["bfcl"] is not overall


def test_load_toolbench_g3_global_pool_corpus_keeps_state_and_next_skill(tmp_path):
    eval_rows = tmp_path / "eval.jsonl"
    skills = tmp_path / "skills.jsonl"
    eval_rows.write_text(
        '{"state_text":"goal: use weather","next_skill_id":"toolbench-g3/weather/get","skill_id":"toolbench-g3/weather/search"}\n'
        '{"state_text":"","next_skill_id":"toolbench-g3/weather/missing"}\n',
        encoding="utf-8",
    )
    skills.write_text(
        '{"skill_id":"toolbench-g3/weather/get","name":"get","description":"get weather"}\n',
        encoding="utf-8",
    )

    corpus = load_toolbench_g3_global_pool_corpus(
        eval_trajectories_path=eval_rows,
        skills_path=skills,
    )

    assert corpus.benchmark == "toolbench_g3"
    assert len(corpus.skills) == 1
    assert len(corpus.source_rows) == 1
    assert corpus.source_rows[0]["state_text"] == "goal: use weather"
    assert corpus.source_rows[0]["next_skill_id"] == "toolbench-g3/weather/get"
    assert corpus.report["skipped_rows"] == 1


def test_load_prebuilt_global_pool_corpus_preserves_frozen_rows_and_skills(tmp_path):
    rows_path = tmp_path / "bfcl_source_rows.jsonl"
    skills_path = tmp_path / "bfcl_skills.jsonl"
    rows_path.write_text(
        '{"benchmark":"bfcl","source_benchmark":"bfcl","state_text":"state","next_skill_id":"bfcl/tool"}\n',
        encoding="utf-8",
    )
    skills_path.write_text(
        '{"skill_id":"bfcl/tool","name":"tool"}\n',
        encoding="utf-8",
    )

    corpus = load_prebuilt_global_pool_corpus(
        benchmark="bfcl",
        source_rows_path=rows_path,
        skills_path=skills_path,
    )

    assert corpus.benchmark == "bfcl"
    assert corpus.source_rows[0]["next_skill_id"] == "bfcl/tool"
    assert corpus.skills[0]["skill_id"] == "bfcl/tool"
    assert corpus.report["source"] == "prebuilt_jsonl"
    assert corpus.report["source_row_count"] == 1
    assert corpus.report["skill_count"] == 1


def test_skillrouter_cli_build_corpus_prefers_prebuilt_rows_and_skills(tmp_path):
    rows_path = tmp_path / "bfcl_source_rows.jsonl"
    skills_path = tmp_path / "bfcl_skills.jsonl"
    rows_path.write_text(
        '{"benchmark":"bfcl","state_text":"state","next_skill_id":"bfcl/tool"}\n',
        encoding="utf-8",
    )
    skills_path.write_text('{"skill_id":"bfcl/tool","name":"tool"}\n', encoding="utf-8")

    class Args:
        benchmark = "bfcl"
        prebuilt_source_rows_path = str(rows_path)
        prebuilt_skills_path = str(skills_path)

    corpus = _build_corpus(Args())

    assert corpus.benchmark == "bfcl"
    assert corpus.source_rows[0]["next_skill_id"] == "bfcl/tool"
    assert corpus.skills[0]["skill_id"] == "bfcl/tool"
    assert corpus.report["source"] == "prebuilt_jsonl"


def test_global_pool_clstr_cli_and_sbatch_expose_safe_reliability_controls():
    cli = Path("scripts/run_global_pool_clstr_route_eval.py").read_text(encoding="utf-8")
    sbatch = Path("scripts/sbatch/run_global_pool_clstr_route_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}" in sbatch
    assert "--stage0_handoff_query_mode" in sbatch
    assert "--reliability_mode" in cli
    assert "--fixed_alpha" in cli
    assert "reliability_mode=args.reliability_mode" in cli
    assert "fixed_alpha=args.fixed_alpha" in cli
    assert "RELIABILITY_MODE=${RELIABILITY_MODE:-dynamic}" in sbatch
    assert "FIXED_ALPHA=${FIXED_ALPHA:-1.0}" in sbatch
    assert "--reliability_mode" in sbatch
    assert "--fixed_alpha" in sbatch


def test_global_pool_clstr_run_threads_full_pool_union_and_strict_source_denominator(
    tmp_path,
    monkeypatch,
):
    import clstr.full_base_train as full_base_train
    import clstr.logged_online_stage4_train as logged_online_stage4_train
    import clstr.stage4_act_train as stage4_act_train
    import clstr.stage_checkpoint_init as stage_checkpoint_init

    base_skills_path = tmp_path / "base_skills.jsonl"
    base_skills_path.write_text(
        '{"skill_id":"skill/a"}\n'
        '{"skill_id":"skill/b"}\n'
        '{"skill_id":"skill/c"}\n',
        encoding="utf-8",
    )
    stage0_checkpoint_path = tmp_path / "stage0.pt"
    stage2_checkpoint_path = tmp_path / "stage2.pt"
    stage0_checkpoint_path.write_bytes(b"stage0")
    stage2_checkpoint_path.write_bytes(b"stage2")
    route_records_path = tmp_path / "memory_utility_routes.jsonl"
    source_rows = [
        {
            "benchmark": "toolbench_g3",
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "action_text": "call a",
            "next_observation_text": "state 1",
            "next_state_text": "state 1",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
        },
        {
            "benchmark": "toolbench_g3",
            "trajectory_id": "traj-1",
            "step_index": 1,
            "state_text": "state 1",
            "action_text": "call missing",
            "next_observation_text": "state 2",
            "next_state_text": "state 2",
            "skill_id": "skill/b",
            "next_skill_id": "skill/missing",
        },
    ]
    corpus = GlobalPoolCorpus(
        benchmark="toolbench_g3",
        skills=[{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}],
        source_rows=source_rows,
        report={"status": "ok"},
    )

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        stage_checkpoint_init,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (FakeModel(), {}, {}),
    )
    monkeypatch.setattr(
        stage_checkpoint_init,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {},
    )
    handoff_calls = []

    def fake_handoff(_model, rows, _skills, _mapping, **kwargs):
        handoff_calls.append(dict(kwargs))
        return list(rows), {"enabled": True, "next_skill_pool_mode": kwargs["next_skill_pool_mode"]}

    monkeypatch.setattr(full_base_train, "_attach_stage0_topm_candidates", fake_handoff)
    stage4_calls = []

    def fake_stage4_rows(rows, _mapping, **kwargs):
        stage4_calls.append(dict(kwargs))
        retained = [dict(rows[0], source_benchmark="toolbench_g3")]
        return retained, {
            "stage4_rows": 1,
            "positive_injected_rows": 0,
            "candidate_source": "declared_legal_full_skill_pool",
            "next_skill_pool_mode": kwargs["next_skill_pool_mode"],
            "stage0_next_static_miss_retained_rows": 1,
        }

    monkeypatch.setattr(
        stage4_act_train,
        "_build_stage4_next_skill_rows_from_source_rows",
        fake_stage4_rows,
    )
    monkeypatch.setattr(
        logged_online_stage4_train,
        "attach_trajectory_prefix_online_memory_scores",
        lambda rows, **_kwargs: (list(rows), {}),
    )
    eval_calls = []

    def fake_eval(_model, _rows, **kwargs):
        eval_calls.append(dict(kwargs))
        return {
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
            "candidate_recall_all_static_hit_rows": 0.0,
            "candidate_recall_all_dynamic_top_hit_rows": 1.0,
            "candidate_recall_all_dynamic_extra_hit_rows": 1.0,
            "candidate_recall_all_union_hit_rows": 1.0,
            "candidate_recall_all_static_equal_budget_hit_rows": 0.0,
            "candidate_recall_all_dynamic_rescue_rows": 1.0,
            "candidate_recall_all_static_dynamic_overlap_sum": 0.0,
            "candidate_recall_all_dynamic_only_candidate_total": 1.0,
            "candidate_recall_memory_active_source_rows": 1.0,
            "candidate_recall_memory_active_static_hit_rows": 0.0,
            "candidate_recall_memory_active_dynamic_top_hit_rows": 1.0,
            "candidate_recall_memory_active_dynamic_extra_hit_rows": 1.0,
            "candidate_recall_memory_active_union_hit_rows": 1.0,
            "candidate_recall_memory_active_static_equal_budget_hit_rows": 0.0,
            "candidate_recall_memory_active_dynamic_rescue_rows": 1.0,
            "candidate_recall_memory_active_static_dynamic_overlap_sum": 0.0,
            "candidate_recall_memory_active_dynamic_only_candidate_total": 1.0,
        }

    monkeypatch.setattr(logged_online_stage4_train, "evaluate_logged_online_stage4_rows", fake_eval)
    monkeypatch.setattr(
        logged_online_stage4_train,
        "evaluate_logged_online_stage4_rows_by_benchmark",
        lambda *_args, **_kwargs: {},
    )

    sentinel_gate = torch.nn.Identity()
    monkeypatch.setattr(
        global_pool_route_eval,
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
    report = run_global_pool_clstr_route_eval(
        corpus=corpus,
        base_skills_path=base_skills_path,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=None,
        output_dir=tmp_path / "out",
        stage0_top_m=1,
        dynamic_extra_k=1,
        candidate_count=1,
        batch_size=1,
        reliability_mode="learned",
        fixed_alpha=1.0,
        memory_utility_gate_checkpoint_path=tmp_path / "gate.pt",
        expected_memory_utility_gate_checkpoint_sha256="gate-sha",
        expected_memory_utility_gate_audit_sha256="audit-sha",
        route_records_path=route_records_path,
    )

    assert handoff_calls[0]["next_skill_pool_mode"] == "full_pool"
    assert stage4_calls[0]["next_skill_pool_mode"] == "full_pool"
    assert len(eval_calls) == 2
    for call in eval_calls:
        assert call["candidate_recall_mode"] == "static_plus_dynamic_extra"
        assert call["skill_id_to_idx"] == {"skill/a": 0, "skill/b": 1, "skill/c": 2}
        assert call["static_k"] == 1
        assert call["dynamic_extra_k"] == 1
        assert call["final_k"] == 1
    assert eval_calls[0]["reliability_mode"] == "static"
    assert eval_calls[0]["fixed_alpha"] == 1.0
    assert eval_calls[1]["reliability_mode"] == "learned"
    assert eval_calls[1]["fixed_alpha"] == 1.0
    assert eval_calls[1]["memory_utility_gate"] is sentinel_gate
    assert eval_calls[1]["feature_update_count_cap"] == pytest.approx(4.0)
    assert eval_calls[1]["feature_candidate_count_cap"] == pytest.approx(64.0)
    assert "route_records_path" not in eval_calls[0]
    assert eval_calls[1]["route_records_path"] == route_records_path
    assert eval_calls[1]["route_record_pool_protocol"] == "known_global"
    assert eval_calls[1]["route_record_model_digest"]
    assert eval_calls[1]["route_record_sequential_benchmarks"] == {"toolbench_g3"}
    assert report["candidate_recall"]["candidate_recall_applicable"] is True
    assert report["candidate_recall"]["all_eligible_source_strict"]["source_rows"] == 2.0
    assert report["candidate_recall"]["all_eligible_source_strict"]["union_recall"] == 0.5
    assert report["candidate_recall"]["positive_not_in_declared_pool_rows"] == 1.0
    assert "target_outside_declared_legal_pool" in report["blockers"]
    assert report["config"]["reliability_mode"] == "learned"
    assert report["config"]["fixed_alpha"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reliability_mode", "fixed_alpha", "message"),
    [
        ("learned", 0.5, "learned reliability requires a gate checkpoint"),
        ("fixed_alpha", -0.1, "fixed_alpha must be finite and in \\[0, 1\\]"),
    ],
)
def test_global_pool_outer_eval_rejects_unsafe_reliability_modes(
    tmp_path,
    reliability_mode,
    fixed_alpha,
    message,
):
    corpus = GlobalPoolCorpus(benchmark="bfcl", skills=[], source_rows=[], report={})
    with pytest.raises(ValueError, match=message):
        run_global_pool_clstr_route_eval(
            corpus=corpus,
            base_skills_path=tmp_path / "skills.jsonl",
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            stage4_checkpoint_path=None,
            output_dir=tmp_path / "out",
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
        )
