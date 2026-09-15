from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.toolsandbox_route_eval import (
    _attach_toolsandbox_causal_replay_prefixes,
    _build_toolsandbox_report,
    _load_toolsandbox_model_for_pool_mode,
    _stage4_rows_from_ranked_toolsandbox,
    _tool_route_variants,
    _toolsandbox_causal_replay_identity,
    _toolsandbox_skillrouter_query_text,
    build_toolsandbox_paired_protocol_manifest,
    load_toolsandbox_route_corpus,
    run_toolsandbox_full_clstr_route_eval,
    toolsandbox_start_skill_id,
    toolsandbox_tool_skill_id,
)
import clstr.toolsandbox_route_eval as toolsandbox_route_eval


def test_toolsandbox_dag_builds_valid_topological_routes_with_multi_positive_frontier():
    variants, report = _tool_route_variants(
        [
            [{"tool_name": "first", "arguments": {}}],
            [{"tool_name": "second", "arguments": {}}],
            [{"tool_name": "final", "arguments": {}}],
        ],
        [(0, 2), (1, 2)],
        maximum_variants=8,
    )

    assert report["explicit_milestone_dag"] is True
    assert report["route_variant_count"] == 2
    assert {
        tuple(trace["tool_name"] for trace in variant["required_tool_traces"])
        for variant in variants
    } == {("first", "second", "final"), ("second", "first", "final")}
    assert all(
        variant["equivalent_tool_names_by_step"][0] == ["first", "second"]
        for variant in variants
    )
    assert all(
        variant["equivalent_tool_names_by_step"][-1] == ["final"]
        for variant in variants
    )


def test_toolsandbox_multi_tool_milestone_is_unordered_not_source_flattened():
    variants, report = _tool_route_variants(
        [
            [
                {"tool_name": "alpha", "arguments": {}},
                {"tool_name": "beta", "arguments": {}},
            ],
            [{"tool_name": "final", "arguments": {}}],
        ],
        None,
        maximum_variants=8,
    )

    assert report["multi_tool_milestone_count"] == 1
    assert report["route_variant_count"] == 2
    assert all(
        variant["equivalent_tool_names_by_step"][0] == ["alpha", "beta"]
        for variant in variants
    )


def test_toolsandbox_causal_replay_executes_prior_next_skill_not_previous_skill():
    rows = [
        {
            "trajectory_id": "toolsandbox/example",
            "step_index": 0,
            "state_text": "state before timestamp",
            "action_text": "previous_tool: START",
            "next_observation_text": "oracle_next_tool_arguments: {}",
            "skill_id": "toolsandbox/__start__",
            "next_skill_id": "toolsandbox/get_current_timestamp",
        },
        {
            "trajectory_id": "toolsandbox/example",
            "step_index": 1,
            "state_text": "state after timestamp\nhistory:\n1. get_current_timestamp({})",
            "action_text": "previous_tool: get_current_timestamp",
            "next_observation_text": "oracle_next_tool_arguments: {}",
            "skill_id": "toolsandbox/get_current_timestamp",
            "next_skill_id": "toolsandbox/search_messages",
        },
    ]

    prepared, report = _attach_toolsandbox_causal_replay_prefixes(
        rows,
        skill_id_to_idx={
            "toolsandbox/__start__": 0,
            "toolsandbox/get_current_timestamp": 1,
            "toolsandbox/search_messages": 2,
        },
        max_steps=3,
    )

    assert "replay_prefix" not in prepared[0]
    assert len(prepared[1]["replay_prefix"]) == 1
    replay = prepared[1]["replay_prefix"][0]
    assert replay["skill_id"] == "toolsandbox/get_current_timestamp"
    assert replay["skill_idx"] == 1
    assert "get_current_timestamp" in replay["action_text"]
    assert "START" not in replay["action_text"]
    assert replay["next_observation_text"] == ""
    assert replay["observation_source"] == "action_only_no_tool_result"
    assert report["rows_with_causal_replay_prefix"] == 1
    assert report["eligible_claim"] == "action_history_memory_only"


def test_toolsandbox_skillrouter_baseline_retains_full_history_context():
    query = _toolsandbox_skillrouter_query_text(
        {
            "state_text": "scenario: s",
            "state_text_full": "scenario: s\nhistory:\n1. search_messages({})",
        }
    )

    assert "history:" in query
    assert "search_messages" in query


def test_toolsandbox_report_exposes_local_candidate_recall_saturation():
    candidate_recall = {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
        "candidate_recall_saturated": True,
    }
    report = _build_toolsandbox_report(
        output_dir="outputs/toolsandbox",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        skills_path="skills.jsonl",
        source_eval_rows=2,
        retained_eval_rows=2,
        corpus_report={"status": "ok"},
        route_data_report={"positive_injected_rows": 0},
        memory_report={},
        stage0_prior_eval={},
        base_eval={},
        stage4_eval={},
        config={},
        stage0_prior_report={},
        candidate_recall=candidate_recall,
    )

    assert report["candidate_recall"] == candidate_recall


def test_toolsandbox_stage4_rows_mark_row_candidates_as_benchmark_local():
    ranked = [
        {
            "task_id": "task-1",
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state",
            "action_text": "start",
            "next_observation_text": "obs",
            "skill_id": "skill/start",
            "next_skill_id": "skill/tool",
            "candidate_next_skill_ids": ["skill/tool", "skill/other"],
            "candidate_next_prior_scores": [2.0, 1.0],
        }
    ]

    rows, _report = _stage4_rows_from_ranked_toolsandbox(
        ranked,
        {"skill/start": 0, "skill/tool": 1, "skill/other": 2},
        candidate_count=None,
    )

    assert rows[0]["candidate_pool_protocol"] == "benchmark_local"


def test_toolsandbox_stage4_rows_keep_dag_frontier_multi_positive_targets():
    ranked = [
        {
            "task_id": "task-1",
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state",
            "action_text": "start",
            "next_observation_text": "obs",
            "skill_id": "skill/start",
            "next_skill_id": "skill/tool-a",
            "equivalent_next_skill_ids": ["skill/tool-b"],
            "candidate_next_skill_ids": ["skill/tool-b", "skill/tool-a", "skill/other"],
            "candidate_next_prior_scores": [3.0, 2.0, 1.0],
        }
    ]

    rows, report = _stage4_rows_from_ranked_toolsandbox(
        ranked,
        {
            "skill/start": 0,
            "skill/tool-a": 1,
            "skill/tool-b": 2,
            "skill/other": 3,
        },
        candidate_count=None,
    )

    assert report["stage4_rows"] == 1
    assert rows[0]["positive_next_skill_ids"] == ["skill/tool-b", "skill/tool-a"]
    assert rows[0]["positive_next_skill_positions"] == [0, 1]


def _write_toolsandbox_fixture(root: Path) -> tuple[Path, Path]:
    scenarios_root = root / "tool_sandbox" / "scenarios"
    tools_root = root / "tool_sandbox" / "tools"
    scenarios_root.mkdir(parents=True)
    tools_root.mkdir(parents=True)
    (tools_root / "messaging.py").write_text(
        '''
def search_messages():
    """Search the user's messages."""


def get_current_timestamp():
    """Return the current timestamp."""
''',
        encoding="utf-8",
    )
    (scenarios_root / "multiple_tool_call_scenarios.py").write_text(
        '''
import json
import polars as pl

def get_extensions(base_scenarios):
    return [
        ScenarioExtension(
            name="search_message_with_recency",
            base_scenario=base_scenarios["base"],
            messages=[
                {"sender": RoleType.USER, "recipient": RoleType.AGENT, "content": "What did I text most recently?"},
            ],
            tool_allow_list=["search_messages", "get_current_timestamp"],
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            target_dataframe=pl.DataFrame(
                                {"tool_trace": json.dumps({"tool_name": "search_messages", "arguments": {}})}
                            )
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            target_dataframe=pl.DataFrame(
                                {"tool_trace": json.dumps({"tool_name": "get_current_timestamp", "arguments": {}})}
                            )
                        )
                    ]
                ),
            ],
        )
    ]
''',
        encoding="utf-8",
    )
    return scenarios_root, tools_root


def test_toolsandbox_route_corpus_extracts_required_tool_sequence_from_milestones(tmp_path):
    scenarios_root, tools_root = _write_toolsandbox_fixture(tmp_path)

    corpus = load_toolsandbox_route_corpus(scenarios_root=scenarios_root, tools_root=tools_root)

    assert corpus.report["benchmark"] == "toolsandbox"
    assert corpus.report["scenario_count"] == 1
    assert corpus.report["source_row_count"] == 2
    assert toolsandbox_start_skill_id() in {row["skill_id"] for row in corpus.skills}
    assert toolsandbox_tool_skill_id("search_messages") in {row["skill_id"] for row in corpus.skills}
    assert "Search the user's messages." in next(
        row["description"] for row in corpus.skills if row["skill_id"] == toolsandbox_tool_skill_id("search_messages")
    )

    first, second = corpus.source_rows
    assert first["skill_id"] == toolsandbox_start_skill_id()
    assert first["next_skill_id"] == toolsandbox_tool_skill_id("search_messages")
    assert first["candidate_next_skill_ids"] == [
        toolsandbox_tool_skill_id("search_messages"),
        toolsandbox_tool_skill_id("get_current_timestamp"),
    ]
    assert "What did I text most recently?" in first["state_text"]
    assert second["skill_id"] == toolsandbox_tool_skill_id("search_messages")
    assert second["next_skill_id"] == toolsandbox_tool_skill_id("get_current_timestamp")
    assert "search_messages" in second["history_text"]
    assert "history:" not in second["state_text"]
    assert "search_messages" not in second["state_text"]
    assert "search_messages" in second["state_text_full"]
    assert second["state_text_current"] == second["state_text"]
    assert second["observation_source"] == "oracle_next_tool_arguments"
    assert corpus.report["history_channel"]["status"] == "ok"
    assert corpus.report["causal_memory_observation_eligible"] is False


def test_toolsandbox_max_scenarios_limits_usable_scenarios_after_skipping_empty(tmp_path):
    scenarios_root, tools_root = _write_toolsandbox_fixture(tmp_path)
    (scenarios_root / "insufficient_information_scenarios.py").write_text(
        '''
def get_extensions(base_scenarios):
    return [
        ScenarioExtension(
            name="no_required_tool",
            base_scenario=base_scenarios["base"],
            messages=[{"content": "I do not have enough information."}],
            tool_allow_list=["search_messages"],
            milestones=[],
        )
    ]
''',
        encoding="utf-8",
    )

    corpus = load_toolsandbox_route_corpus(
        scenarios_root=scenarios_root,
        tools_root=tools_root,
        max_scenarios=1,
    )

    assert corpus.report["raw_scenario_count"] == 2
    assert corpus.report["scenario_count"] == 1
    assert corpus.report["source_row_count"] == 2
    assert corpus.source_rows[0]["trajectory_id"] == "toolsandbox/search_message_with_recency"


def test_toolsandbox_route_corpus_collects_required_tools_from_each_scenario(tmp_path):
    scenarios_root, tools_root = _write_toolsandbox_fixture(tmp_path)
    (scenarios_root / "extra_scenarios.py").write_text(
        '''
import json
import polars as pl

def get_extensions(base_scenarios):
    return [
        ScenarioExtension(
            name="requires_send_email",
            base_scenario=base_scenarios["base"],
            messages=[{"content": "Send the note."}],
            tool_allow_list=[],
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            target_dataframe=pl.DataFrame(
                                {"tool_trace": json.dumps({"tool_name": "send_email", "arguments": {}})}
                            )
                        )
                    ]
                ),
            ],
        )
    ]
''',
        encoding="utf-8",
    )

    corpus = load_toolsandbox_route_corpus(scenarios_root=scenarios_root, tools_root=tools_root)

    assert toolsandbox_tool_skill_id("send_email") in {row["skill_id"] for row in corpus.skills}


def test_toolsandbox_native_eval_restores_training_pool_before_appending_local_skills(
    tmp_path,
    monkeypatch,
):
    scenarios_root, tools_root = _write_toolsandbox_fixture(tmp_path)
    expected_corpus = load_toolsandbox_route_corpus(
        scenarios_root=scenarios_root,
        tools_root=tools_root,
    )
    training_skills_path = tmp_path / "training_skills.jsonl"
    training_skills_path.write_text(
        json.dumps({"skill_id": "training/base", "name": "base"}) + "\n",
        encoding="utf-8",
    )
    adapter_calls = []

    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def _fake_restore(**kwargs):
        adapter_calls.append(kwargs)
        mapping = {"training/base": 0}
        mapping.update(
            {
                str(row["skill_id"]): index
                for index, row in enumerate(kwargs["benchmark_skills"], start=1)
            }
        )
        return (
            _FakeModel(),
            {"d": 2, "freeze_backbone": True},
            mapping,
            {
                "stage0": {"stage0_loaded": True},
                "stage2": {"loaded": True},
                "stage4": {"loaded": True},
                "skill_append": {
                    "appended_count": len(kwargs["benchmark_skills"]),
                    "appended_skill_ids": [
                        str(row["skill_id"]) for row in kwargs["benchmark_skills"]
                    ],
                },
            },
        )

    monkeypatch.setattr(
        toolsandbox_route_eval,
        "restore_native_benchmark_checkpoint_chain",
        _fake_restore,
        raising=False,
    )
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "rank_tau2_candidates_with_stage0_prior",
        lambda _model, rows, *_args, **_kwargs: (
            list(rows),
            {"ranked_rows": len(rows), "ranked_rows_with_positive": len(rows)},
        ),
    )
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "_stage4_rows_from_ranked_toolsandbox",
        lambda rows, *_args, **_kwargs: (
            list(rows),
            {
                "source_rows": len(rows),
                "stage4_rows": len(rows),
                "positive_injected_rows": 0,
                "candidate_count_mean": 2.0,
            },
        ),
    )
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "attach_trajectory_prefix_online_memory_scores",
        lambda rows, **_kwargs: (rows, {"online_memory_rows": len(rows)}),
    )
    fake_metrics = {
        "stage4_next_skill_recall@1": 1.0,
        "stage4_next_skill_recall@5": 1.0,
        "stage4_next_skill_mrr": 1.0,
    }
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "_stage0_prior_eval_from_ranked_rows",
        lambda _rows: dict(fake_metrics),
    )
    eval_calls = []
    evaluated_rows = []

    def fake_eval(*args, **kwargs):
        eval_calls.append(dict(kwargs))
        evaluated_rows.append(args[1])
        return {
            **fake_metrics,
            "marker": "base" if len(eval_calls) == 1 else "stage4",
        }

    monkeypatch.setattr(
        toolsandbox_route_eval,
        "evaluate_logged_online_stage4_rows",
        fake_eval,
    )
    sentinel_gate = torch.nn.Identity()
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "resolve_reliability_gate",
        lambda **_kwargs: (
            sentinel_gate,
            {"feature_update_count_cap": 4.0, "feature_candidate_count_cap": 64.0},
        ),
        raising=False,
    )

    def fail_grouped_eval(*_args, **_kwargs):
        raise AssertionError("discarded ToolSandbox grouped pass must not run")

    monkeypatch.setattr(
        toolsandbox_route_eval,
        "evaluate_logged_online_stage4_rows_by_benchmark",
        fail_grouped_eval,
        raising=False,
    )

    report = run_toolsandbox_full_clstr_route_eval(
        scenarios_root=scenarios_root,
        tools_root=tools_root,
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        training_skills_path=training_skills_path,
        output_dir=tmp_path / "out",
        model_skill_pool_mode="checkpoint_faithful",
        reliability_mode="learned",
        memory_utility_gate_checkpoint_path=tmp_path / "gate.pt",
        toolsandbox_replay_mode="corrected_causal",
    )

    assert report["status"] == "ok"
    assert len(eval_calls) == 2
    assert report["base_eval"]["marker"] == "base"
    assert report["stage4_eval"]["marker"] == "stage4"
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["training_skills_path"] == training_skills_path
    assert adapter_calls[0]["benchmark_skills"] == expected_corpus.skills
    assert adapter_calls[0]["require_safe_memory_delta"] is True
    assert eval_calls[1]["reliability_mode"] == "learned"
    assert eval_calls[1]["memory_utility_gate"] is sentinel_gate
    assert eval_calls[1]["feature_update_count_cap"] == pytest.approx(4.0)
    assert eval_calls[1]["feature_candidate_count_cap"] == pytest.approx(64.0)
    for call in eval_calls:
        assert call["candidate_recall_mode"] == "static_plus_dynamic_extra"
        assert call["skill_id_to_idx"]
        assert call["static_k"] == 2
        assert call["dynamic_extra_k"] == 64
        assert call["final_k"] == 2
    assert report["config"]["training_skills_path"] == str(training_skills_path)
    assert report["config"]["model_skill_pool_mode"] == "checkpoint_faithful"
    assert report["eligible_for_checkpoint_selection"] is True
    assert report["config"]["toolsandbox_replay_mode"] == "action_only"
    assert report["config"]["toolsandbox_replay_mode_requested"] == "corrected_causal"
    assert report["config"]["corrected_causal_compatibility_alias_used"] is True
    assert report["causal_replay"]["rows_with_causal_replay_prefix"] == 1
    assert evaluated_rows[0][1]["replay_prefix"][0]["skill_id"] == toolsandbox_tool_skill_id(
        "search_messages"
    )
    assert report["model_load"]["skill_pool_adapter"]["skill_append"]["appended_count"] == len(
        expected_corpus.skills
    )


def test_toolsandbox_pool_modes_are_explicit_and_local_rebuild_is_ablation_only(
    tmp_path,
    monkeypatch,
):
    benchmark_skills = [
        {"skill_id": "toolsandbox/__start__"},
        {"skill_id": "toolsandbox/search_messages"},
    ]
    local_skills_path = tmp_path / "toolsandbox_skills.jsonl"
    local_skills_path.write_text(
        "".join(json.dumps(row) + "\n" for row in benchmark_skills),
        encoding="utf-8",
    )
    training_skills_path = tmp_path / "training_skills.jsonl"
    training_skills_path.write_text('{"skill_id":"training/base"}\n', encoding="utf-8")
    calls = []

    class _Model:
        pass

    monkeypatch.setattr(
        toolsandbox_route_eval,
        "restore_native_benchmark_checkpoint_chain",
        lambda **kwargs: (
            calls.append(("checkpoint_faithful", kwargs))
            or (
                _Model(),
                {"freeze_backbone": True},
                {
                    "training/base": 0,
                    "toolsandbox/__start__": 1,
                    "toolsandbox/search_messages": 2,
                },
                {"stage0": {}, "stage2": {}, "stage4": {}, "skill_append": {}},
            )
        ),
    )
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **kwargs: (
            calls.append(("local_table_rebuild", kwargs))
            or (_Model(), {"freeze_backbone": True}, {"stage0_loaded": True})
        ),
    )
    monkeypatch.setattr(
        toolsandbox_route_eval,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {"loaded": True},
    )

    *_checkpoint, checkpoint_report = _load_toolsandbox_model_for_pool_mode(
        model_skill_pool_mode="checkpoint_faithful",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        training_skills_path=training_skills_path,
        benchmark_skills=benchmark_skills,
        benchmark_skills_path=local_skills_path,
        model_cache_dir=tmp_path / "cache-checkpoint",
        device=torch.device("cpu"),
    )
    *_local, local_report = _load_toolsandbox_model_for_pool_mode(
        model_skill_pool_mode="local_table_rebuild",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        training_skills_path=None,
        benchmark_skills=benchmark_skills,
        benchmark_skills_path=local_skills_path,
        model_cache_dir=tmp_path / "cache-local",
        device=torch.device("cpu"),
    )

    assert [kind for kind, _kwargs in calls] == [
        "checkpoint_faithful",
        "local_table_rebuild",
    ]
    assert checkpoint_report["model_skill_pool_mode"] == "checkpoint_faithful"
    assert checkpoint_report["eligible_for_checkpoint_selection"] is True
    assert local_report["model_skill_pool_mode"] == "local_table_rebuild"
    assert local_report["eligible_for_checkpoint_selection"] is False
    assert local_report["final_model_skill_ids"] == [
        "toolsandbox/__start__",
        "toolsandbox/search_messages",
    ]

    with pytest.raises(ValueError, match="must not receive training_skills_path"):
        _load_toolsandbox_model_for_pool_mode(
            model_skill_pool_mode="local_table_rebuild",
            stage0_checkpoint_path="stage0.pt",
            stage2_checkpoint_path="stage2.pt",
            stage4_checkpoint_path="stage4.pt",
            training_skills_path=training_skills_path,
            benchmark_skills=benchmark_skills,
            benchmark_skills_path=local_skills_path,
            model_cache_dir=tmp_path / "cache-invalid",
            device=torch.device("cpu"),
        )


def test_toolsandbox_paired_manifest_requires_identical_candidates_and_replay() -> None:
    checkpoint_report = {
        "status": "ok",
        "config": {"model_skill_pool_mode": "checkpoint_faithful"},
        "candidate_protocol_identity_sha256": "candidate-sha",
        "causal_replay_identity_sha256": "replay-sha",
        "stage4_eval": {"stage4_next_skill_mrr": 0.62},
        "eligible_for_checkpoint_selection": True,
    }
    local_report = {
        "status": "ok",
        "config": {"model_skill_pool_mode": "local_table_rebuild"},
        "candidate_protocol_identity_sha256": "candidate-sha",
        "causal_replay_identity_sha256": "replay-sha",
        "stage4_eval": {"stage4_next_skill_mrr": 0.60},
        "eligible_for_checkpoint_selection": False,
    }

    manifest = build_toolsandbox_paired_protocol_manifest(
        checkpoint_report=checkpoint_report,
        local_rebuild_report=local_report,
    )

    assert manifest["status"] == "ok"
    assert manifest["checkpoint_selection_mode"] == "checkpoint_faithful"
    assert manifest["local_rebuild_ablation_only"] is True
    assert manifest["delta_local_rebuild_minus_checkpoint_faithful_mrr"] == pytest.approx(-0.02)

    local_report["causal_replay_identity_sha256"] = "different"
    with pytest.raises(ValueError, match="causal replay identity mismatch"):
        build_toolsandbox_paired_protocol_manifest(
            checkpoint_report=checkpoint_report,
            local_rebuild_report=local_report,
        )


def test_toolsandbox_replay_identity_ignores_pool_specific_skill_indices() -> None:
    base = {
        "task_id": "task-1",
        "trajectory_id": "traj-1",
        "step_index": 1,
        "replay_prefix": [
            {
                "skill_id": "toolsandbox/search_messages",
                "skill_idx": 67558,
                "action_text": "executed_tool: search_messages",
                "observation_text": "before",
                "next_observation_text": "after",
                "replay_protocol": "toolsandbox_causal_next_skill_v1",
            }
        ],
    }
    local = json.loads(json.dumps(base))
    local["replay_prefix"][0]["skill_idx"] = 1

    assert _toolsandbox_causal_replay_identity([base]) == (
        _toolsandbox_causal_replay_identity([local])
    )


def test_toolsandbox_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_toolsandbox_full_clstr_route_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "toolsandbox" in result.stdout.lower()
    assert "--stage4_checkpoint_path" in result.stdout
    assert "--training_skills_path" in result.stdout
    assert "--toolsandbox_replay_mode" in result.stdout
