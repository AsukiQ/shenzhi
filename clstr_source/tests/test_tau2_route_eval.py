from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.tau2_route_eval import (
    _attach_tau2_action_only_replay_prefixes,
    _stage4_rows_from_ranked_tau2,
    build_tau2_full_clstr_route_report,
    load_tau2_route_corpus,
    rank_tau2_candidates_with_stage0_prior,
    run_tau2_full_clstr_route_eval,
    tau2_action_skill_id,
    tau2_start_skill_id,
)
import clstr.tau2_route_eval as tau2_route_eval
from clstr.tau2_skillrouter_eval import run_tau2_skillrouter_frozen_eval


def _write_tau2_domain(root: Path, domain: str, tasks: list[dict], policy: str = "Use valid domain APIs.") -> None:
    domain_dir = root / "domains" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (domain_dir / "policy.md").write_text(policy, encoding="utf-8")


def _write_tau2_splits(root: Path, domain: str, splits: dict[str, list[str]]) -> None:
    domain_dir = root / "domains" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "split_tasks.json").write_text(
        json.dumps(splits),
        encoding="utf-8",
    )


def _task(task_id: str, domain: str, actions: list[dict]) -> dict:
    return {
        "id": task_id,
        "description": {"purpose": f"purpose {task_id}"},
        "user_scenario": {
            "instructions": {
                "domain": domain,
                "reason_for_call": f"reason {task_id}",
                "known_info": "known user info",
                "unknown_info": "unknown info",
                "task_instructions": "follow policy",
            }
        },
        "evaluation_criteria": {"actions": actions, "communicate_info": [], "nl_assertions": None},
    }


def test_tau2_base_split_filters_tasks_and_keeps_no_tool_stop_rows(tmp_path: Path) -> None:
    actions = [
        {"name": "lookup", "arguments": {"id": "1"}},
        {"name": "refund", "arguments": {"id": "1"}},
    ]
    tasks = [
        _task("task-tool", "airline", actions),
        _task("task-stop", "airline", []),
        _task("task-test", "airline", [{"name": "cancel", "arguments": {}}]),
    ]
    _write_tau2_domain(tmp_path, "airline", list(reversed(tasks)))
    _write_tau2_splits(
        tmp_path,
        "airline",
        {
            "base": ["task-tool", "task-stop"],
            "train": ["task-tool"],
            "test": ["task-test"],
        },
    )

    corpus = load_tau2_route_corpus(
        tmp_path,
        domains=["airline"],
        task_split="base",
    )

    assert corpus.report["task_split"] == "base"
    assert corpus.report["task_count"] == 2
    assert corpus.report["tool_action_row_count"] == 2
    assert corpus.report["no_tool_row_count"] == 1
    assert corpus.report["source_row_count"] == 3
    assert corpus.source_rows[-1]["route_target"] == "STOP"
    assert corpus.source_rows[-1]["candidate_next_skill_ids"] == []
    assert corpus.source_rows[-1]["split"] == "base"

    _write_tau2_domain(tmp_path, "airline", tasks)
    reordered = load_tau2_route_corpus(
        tmp_path,
        domains=["airline"],
        task_split="base",
    )
    assert reordered.source_rows == corpus.source_rows


def test_tau2_split_membership_fails_closed_on_missing_task_id(tmp_path: Path) -> None:
    _write_tau2_domain(
        tmp_path,
        "airline",
        [_task("task-1", "airline", [{"name": "lookup", "arguments": {}}])],
    )
    _write_tau2_splits(
        tmp_path,
        "airline",
        {"base": ["task-1", "missing-task"], "train": [], "test": []},
    )

    with pytest.raises(ValueError, match="missing task IDs"):
        load_tau2_route_corpus(
            tmp_path,
            domains=["airline"],
            task_split="base",
        )


def test_tau2_stage4_rows_mark_row_candidates_as_benchmark_local():
    ranked = [
        {
            "task_id": "task-1",
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state",
            "action_text": "start",
            "next_observation_text": "obs",
            "skill_id": "tau2/retail/__start__",
            "next_skill_id": "tau2/retail/refund",
            "candidate_next_skill_ids": ["tau2/retail/refund", "tau2/retail/lookup"],
            "candidate_next_prior_scores": [2.0, 1.0],
        }
    ]

    rows, _report = _stage4_rows_from_ranked_tau2(
        ranked,
        {
            "tau2/retail/__start__": 0,
            "tau2/retail/refund": 1,
            "tau2/retail/lookup": 2,
        },
        candidate_count=None,
    )

    assert rows[0]["candidate_pool_protocol"] == "benchmark_local"


def _action(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments, "action_id": f"{name}-0", "info": None}


def test_tau2_route_corpus_keeps_empty_action_tasks_as_stop_and_builds_domain_candidates(tmp_path):
    _write_tau2_domain(
        tmp_path,
        "airline",
        [
            _task("empty", "airline", []),
            _task(
                "route",
                "airline",
                [
                    _action("get_user_details", user_id="u1"),
                    _action("get_reservation_details", reservation_id="r1"),
                ],
            ),
        ],
    )
    _write_tau2_domain(
        tmp_path,
        "retail",
        [_task("retail-0", "retail", [_action("get_user_details", user_id="u2")])],
    )

    corpus = load_tau2_route_corpus(tmp_path, domains=["airline", "retail"])

    assert corpus.report["task_count"] == 3
    assert corpus.report["non_routing_refusal_task_count"] == 1
    assert corpus.report["source_row_count"] == 4
    assert corpus.report["tool_action_row_count"] == 3
    assert corpus.report["no_tool_row_count"] == 1
    assert sum(row.get("route_target") == "STOP" for row in corpus.source_rows) == 1
    assert corpus.report["action_step_count"] == 3
    assert tau2_start_skill_id("airline") in {row["skill_id"] for row in corpus.skills}
    assert tau2_action_skill_id("airline", "get_user_details") in {row["skill_id"] for row in corpus.skills}
    assert tau2_action_skill_id("retail", "get_user_details") in {row["skill_id"] for row in corpus.skills}

    tool_rows = [row for row in corpus.source_rows if row.get("route_target") == "TOOL"]
    first = tool_rows[0]
    assert first["benchmark"] == "tau2"
    assert first["source_benchmark"] == "tau2"
    assert first["split"] == "full"
    assert first["trajectory_id"] == "tau2/airline/route"
    assert first["step_index"] == 0
    assert first["skill_id"] == tau2_start_skill_id("airline")
    assert first["next_skill_id"] == tau2_action_skill_id("airline", "get_user_details")
    assert set(first["candidate_next_skill_ids"]) == {
        tau2_action_skill_id("airline", "get_user_details"),
        tau2_action_skill_id("airline", "get_reservation_details"),
    }
    assert all(not item.startswith("tau2/retail/") for item in first["candidate_next_skill_ids"])
    assert "reason route" in first["state_text"]

    second = tool_rows[1]
    assert second["skill_id"] == tau2_action_skill_id("airline", "get_user_details")
    assert second["next_skill_id"] == tau2_action_skill_id("airline", "get_reservation_details")
    assert "get_user_details" in second["history_text"]
    assert "history:" not in second["state_text"]
    assert "get_user_details" not in second["state_text"]
    assert "get_user_details" in second["state_text_full"]
    assert second["state_text_current"] == second["state_text"]
    assert second["observation_source"] == "oracle_next_action_arguments"
    assert "user_id" in second["action_text"]
    assert corpus.report["history_channel"]["status"] == "ok"
    assert corpus.report["causal_memory_observation_eligible"] is False


def test_tau2_replay_uses_executed_action_but_not_oracle_args_as_observation():
    rows = [
        {
            "trajectory_id": "tau2/airline/t",
            "step_index": 0,
            "state_text": "goal: g",
            "next_skill_id": "tau2/airline/get_user_details",
            "next_observation_text": "oracle_next_action_arguments: {user_id: 7}",
        },
        {
            "trajectory_id": "tau2/airline/t",
            "step_index": 1,
            "state_text": "goal: g",
            "next_skill_id": "tau2/airline/get_reservation_details",
            "next_observation_text": "oracle_next_action_arguments: {pnr: A1}",
        },
    ]
    mapping = {
        "tau2/airline/get_user_details": 0,
        "tau2/airline/get_reservation_details": 1,
    }

    prepared, report = _attach_tau2_action_only_replay_prefixes(
        rows,
        skill_id_to_idx=mapping,
        max_steps=3,
    )

    replay = prepared[1]["replay_prefix"][0]
    assert replay["skill_id"] == "tau2/airline/get_user_details"
    assert "user_id" in replay["action_text"]
    assert replay["next_observation_text"] == ""
    assert replay["observation_source"] == "action_only_no_tool_result"
    assert report["eligible_claim"] == "action_history_memory_only"


def test_tau_family_corpus_can_load_tau3_main_policy_and_prefix_ids(tmp_path):
    domain_dir = tmp_path / "domains" / "telecom"
    domain_dir.mkdir(parents=True)
    (domain_dir / "main_policy.md").write_text("Use the telecom workflow.", encoding="utf-8")
    (domain_dir / "tasks.json").write_text(
        json.dumps(
            [
                _task(
                    "route",
                    "telecom",
                    [
                        _action("toggle_airplane_mode", enabled=False),
                        _action("grant_app_permission", app="maps"),
                    ],
                )
            ]
        ),
        encoding="utf-8",
    )

    corpus = load_tau2_route_corpus(tmp_path, domains=["telecom"], benchmark_name="tau3")

    assert corpus.report["benchmark"] == "tau3"
    assert corpus.report["source_row_count"] == 2
    first = corpus.source_rows[0]
    assert first["benchmark"] == "tau3"
    assert first["trajectory_id"] == "tau3/telecom/route"
    assert first["skill_id"] == "tau3/telecom/__start__"
    assert first["next_skill_id"] == "tau3/telecom/toggle_airplane_mode"
    assert set(first["candidate_next_skill_ids"]) == {
        "tau3/telecom/toggle_airplane_mode",
        "tau3/telecom/grant_app_permission",
    }
    assert "Use the telecom workflow." in first["state_text"]


class _FakeSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3), requires_grad=False)

    def retrieval_logits(self, encoded):
        return encoded @ self.E.t()


class _FakeStage0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skill_table = _FakeSkillTable()

    @property
    def device(self):
        return torch.device("cpu")

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            if "prefer-b" in text:
                rows.append(torch.tensor([0.0, 3.0, 1.0]))
            else:
                rows.append(torch.tensor([2.0, 0.0, 1.0]))
        return torch.stack(rows)


def test_stage0_prior_ranking_reorders_candidates_and_updates_prior_scores():
    rows = [
        {
            "task_id": "t0",
            "state_text": "prefer-b",
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "next_skill_id": "skill/c",
        }
    ]

    ranked, report = rank_tau2_candidates_with_stage0_prior(
        _FakeStage0Model(),
        rows,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        batch_size=1,
    )

    assert report["ranked_rows"] == 1
    assert ranked[0]["candidate_next_skill_ids"] == ["skill/b", "skill/c", "skill/a"]
    assert ranked[0]["candidate_next_prior_scores"][0] > ranked[0]["candidate_next_prior_scores"][1]
    assert ranked[0]["candidate_next_prior_scores"][1] > ranked[0]["candidate_next_prior_scores"][2]
    assert ranked[0]["stage0_candidate_prior_report"]["positive_stage0_domain_rank"] == 2


def test_stage0_prior_ranking_uses_best_row_local_positive():
    rows = [
        {
            "task_id": "t0",
            "state_text": "prefer-b",
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "next_skill_id": "skill/c",
            "equivalent_next_skill_ids": ["skill/b"],
        }
    ]

    ranked, report = rank_tau2_candidates_with_stage0_prior(
        _FakeStage0Model(),
        rows,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        batch_size=1,
    )

    assert report["positive_stage0_domain_recall@1"] == 1.0
    assert ranked[0]["positive_next_skill_ids"] == ["skill/b", "skill/c"]
    assert ranked[0]["positive_next_skill_positions"] == [0, 1]
    assert ranked[0]["positive_next_skill_idx"] == 1


def test_tau2_stage4_rows_retain_when_equivalent_positive_survives_truncation():
    ranked = [
        {
            "task_id": "task-1",
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state",
            "action_text": "start",
            "next_observation_text": "obs",
            "skill_id": "skill/start",
            "next_skill_id": "skill/canonical",
            "equivalent_next_skill_ids": ["skill/equivalent"],
            "candidate_next_skill_ids": ["skill/equivalent", "skill/negative", "skill/canonical"],
            "candidate_next_prior_scores": [3.0, 2.0, 1.0],
        }
    ]

    rows, report = _stage4_rows_from_ranked_tau2(
        ranked,
        {
            "skill/start": 0,
            "skill/canonical": 1,
            "skill/equivalent": 2,
            "skill/negative": 3,
        },
        candidate_count=2,
    )

    assert report["stage4_rows"] == 1
    assert rows[0]["next_skill_id"] == "skill/canonical"
    assert rows[0]["positive_next_skill_ids"] == ["skill/equivalent"]
    assert rows[0]["positive_next_skill_positions"] == [0]
    assert rows[0]["positive_next_skill_idx"] == 2


def test_build_tau2_report_marks_action_required_when_no_rows_retained():
    report = build_tau2_full_clstr_route_report(
        output_dir="outputs/tau2",
        data_root="tau2-data",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        skills_path="skills.jsonl",
        source_eval_rows=5,
        retained_eval_rows=0,
        corpus_report={"source_row_count": 5},
        route_data_report={"stage4_rows": 0},
        memory_report={"online_memory_rows": 0},
        stage0_prior_eval={"stage4_next_skill_mrr": 0.1},
        base_eval={"stage4_next_skill_mrr": 0.2},
        stage4_eval={"stage4_next_skill_mrr": 0.3},
        config={"candidate_mode": "domain_local_stage0_ranked"},
    )

    assert report["status"] == "action_required"
    assert "no_retained_stage4_eval_rows" in report["blockers"]
    assert report["benchmark"] == "tau2"
    assert report["stage4_checkpoint_path"] == "stage4.pt"
    assert report["strict"]["stage4"]["source_rows"] == 5.0
    assert report["strict_delta_stage4_vs_stage0_prior"]["strict_stage4_next_skill_mrr"] == pytest.approx(0.0)


def test_run_tau2_eval_can_reuse_prebuilt_rows_and_skills_without_data_root(tmp_path, monkeypatch):
    skills_path = tmp_path / "prebuilt_skills.jsonl"
    source_rows_path = tmp_path / "prebuilt_rows.jsonl"
    training_skills_path = tmp_path / "training_skills.jsonl"
    skill_id = tau2_start_skill_id("airline")
    next_skill_id = tau2_action_skill_id("airline", "book_flight")
    skills = [
        {"skill_id": skill_id, "name": "__start__"},
        {"skill_id": next_skill_id, "name": "book_flight"},
    ]
    source_rows = [
        {
            "benchmark": "tau2",
            "source_benchmark": "tau2",
            "split": "eval",
            "task_id": "tau2/airline/t0::0",
            "trajectory_id": "tau2/airline/t0",
            "step_index": 0,
            "domain": "airline",
            "state_text": "book a flight",
            "action_text": "previous_action: START",
            "next_observation_text": "oracle_next_action_arguments: {}",
            "skill_id": skill_id,
            "next_skill_id": next_skill_id,
            "candidate_next_skill_ids": [next_skill_id],
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        }
    ]
    skills_path.write_text("".join(json.dumps(row) + "\n" for row in skills), encoding="utf-8")
    source_rows_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    training_skills_path.write_text(
        json.dumps({"skill_id": "training/base", "name": "base"}) + "\n",
        encoding="utf-8",
    )

    def _should_not_parse_data_root(*_args, **_kwargs):
        raise AssertionError("data_root parser should not run for prebuilt tau2 eval inputs")

    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    stage4_rows = [
        {
            **source_rows[0],
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "positive_next_skill_position": 0,
            "candidate_next_skill_indices": [1],
            "candidate_next_prior_scores": [1.0],
        }
    ]

    monkeypatch.setattr(tau2_route_eval, "load_tau2_route_corpus", _should_not_parse_data_root)
    adapter_calls = []

    def _fake_restore(**kwargs):
        adapter_calls.append(kwargs)
        return (
            _FakeModel(),
            {"d": 2, "freeze_backbone": True},
            {
                "training/base": 0,
                skill_id: 1,
                next_skill_id: 2,
            },
            {
                "stage0": {"stage0_loaded": True},
                "stage2": {"loaded": True},
                "stage4": {"loaded": True},
                "skill_append": {
                    "appended_count": 2,
                    "appended_skill_ids": [skill_id, next_skill_id],
                },
            },
        )

    monkeypatch.setattr(
        tau2_route_eval,
        "restore_native_benchmark_checkpoint_chain",
        _fake_restore,
        raising=False,
    )
    monkeypatch.setattr(
        tau2_route_eval,
        "rank_tau2_candidates_with_stage0_prior",
        lambda *_args, **_kwargs: (source_rows, {"ranked_rows": 1, "ranked_rows_with_positive": 1}),
    )
    monkeypatch.setattr(
        tau2_route_eval,
        "_stage4_rows_from_ranked_tau2",
        lambda *_args, **_kwargs: (stage4_rows, {"source_rows": 1, "stage4_rows": 1, "candidate_count_mean": 1.0}),
    )
    monkeypatch.setattr(
        tau2_route_eval,
        "attach_trajectory_prefix_online_memory_scores",
        lambda rows, **_kwargs: (rows, {"online_memory_rows": 0}),
    )
    monkeypatch.setattr(
        tau2_route_eval,
        "_stage0_prior_eval_from_ranked_rows",
        lambda _rows: {"stage4_next_skill_recall@1": 1.0, "stage4_next_skill_recall@5": 1.0, "stage4_next_skill_mrr": 1.0},
    )
    fake_metrics = {"stage4_next_skill_recall@1": 1.0, "stage4_next_skill_recall@5": 1.0, "stage4_next_skill_mrr": 1.0}
    eval_calls = []
    monkeypatch.setattr(
        tau2_route_eval,
        "evaluate_logged_online_stage4_rows",
        lambda *_args, **kwargs: (eval_calls.append(dict(kwargs)) or dict(fake_metrics)),
    )
    monkeypatch.setattr(tau2_route_eval, "evaluate_logged_online_stage4_rows_by_benchmark", lambda *_args, **_kwargs: {})
    sentinel_gate = torch.nn.Identity()
    monkeypatch.setattr(
        tau2_route_eval,
        "resolve_reliability_gate",
        lambda **_kwargs: (
            sentinel_gate,
            {"feature_update_count_cap": 4.0, "feature_candidate_count_cap": 64.0},
        ),
        raising=False,
    )

    report = run_tau2_full_clstr_route_eval(
        data_root=tmp_path / "missing_data_root",
        prebuilt_source_rows_path=source_rows_path,
        prebuilt_skills_path=skills_path,
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        training_skills_path=training_skills_path,
        output_dir=tmp_path / "out",
        reliability_mode="learned",
        memory_utility_gate_checkpoint_path=tmp_path / "gate.pt",
    )

    assert report["status"] == "ok"
    assert report["corpus_report"]["source"] == "prebuilt_jsonl"
    assert report["corpus_report"]["source_row_count"] == 1
    assert report["corpus_report"]["skill_count"] == 2
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["training_skills_path"] == training_skills_path
    assert adapter_calls[0]["benchmark_skills"] == skills
    assert adapter_calls[0]["require_safe_memory_delta"] is True
    assert eval_calls[1]["reliability_mode"] == "learned"
    assert eval_calls[1]["memory_utility_gate"] is sentinel_gate
    assert eval_calls[1]["feature_update_count_cap"] == pytest.approx(4.0)
    assert eval_calls[1]["feature_candidate_count_cap"] == pytest.approx(64.0)
    for call in eval_calls:
        assert call["candidate_recall_mode"] == "static_plus_dynamic_extra"
        assert call["skill_id_to_idx"]
        assert call["static_k"] == 1
        assert call["dynamic_extra_k"] == 64
        assert call["final_k"] == 1
    assert report["config"]["prebuilt_source_rows_path"] == str(source_rows_path)
    assert report["config"]["training_skills_path"] == str(training_skills_path)
    assert report["model_load"]["skill_pool_adapter"]["skill_append"]["appended_count"] == 2
    assert report["candidate_recall"] == {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
        "candidate_recall_saturated": True,
        "pool_protocol": "benchmark_local",
        "candidate_source": "tau2_domain_local_stage0_ranked",
        "declared_legal_pool_size": 1,
        "requested_static_m": 1,
        "requested_dynamic_extra_d": 64,
        "final_k": 1,
        "equal_budget_comparator_mode": "same_final_scorer_static_top_m_plus_d",
    }
    assert (tmp_path / "out" / "tau2_source_rows.jsonl").exists()


def test_tau2_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_tau2_full_clstr_route_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "tau2" in result.stdout.lower()
    assert "--max_eval_rows" in result.stdout
    assert "--stage4_checkpoint_path" in result.stdout
    assert "--training_skills_path" in result.stdout


def test_tau3_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_tau3_full_clstr_route_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "tau3" in result.stdout.lower()
    assert "--domains" in result.stdout
    assert "--stage4_checkpoint_path" in result.stdout


def test_tau2_skillrouter_frozen_eval_uses_same_domain_candidates(tmp_path, monkeypatch):
    _write_tau2_domain(
        tmp_path,
        "retail",
        [
            _task(
                "route",
                "retail",
                [
                    _action("cancel_order", order_id="o1"),
                    _action("refund_order", order_id="o1"),
                ],
            )
        ],
    )

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            if "cancel_order" in text and "Action name:" in text:
                rows.append(torch.tensor([1.0, 0.0]))
            elif "refund_order" in text and "Action name:" in text:
                rows.append(torch.tensor([0.0, 1.0]))
            elif "history:" in text:
                rows.append(torch.tensor([0.0, 1.0]))
            elif "refund_order" in text:
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.tau2_skillrouter_eval._encode_skillrouter_texts", fake_encode)

    report = run_tau2_skillrouter_frozen_eval(
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        model_name_or_path="fake-skillrouter",
        domains=["retail"],
        batch_size=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_frozen_biencoder"
    assert report["source_eval_rows"] == 2
    assert report["ranking_report"]["candidate_skill_missing_count"] == 0
    assert report["metrics"]["next_tool_recall@1"] == pytest.approx(1.0)
    ranked = [
        json.loads(line)
        for line in (tmp_path / "out" / "tau2_skillrouter_ranked_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(tau2_start_skill_id("retail") not in row["candidate_next_skill_ids"] for row in ranked)
    assert set(ranked[0]["candidate_next_skill_ids"]) == {
        tau2_action_skill_id("retail", "cancel_order"),
        tau2_action_skill_id("retail", "refund_order"),
    }


def test_tau2_skillrouter_eval_can_apply_finetuned_adapter(tmp_path, monkeypatch):
    _write_tau2_domain(
        tmp_path,
        "retail",
        [
            _task(
                "route",
                "retail",
                [
                    _action("cancel_order", order_id="o1"),
                    _action("refund_order", order_id="o1"),
                ],
            )
        ],
    )

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            if "cancel_order" in text and "Action name:" in text:
                rows.append(torch.tensor([1.0, 0.0]))
            elif "refund_order" in text and "Action name:" in text:
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.tau2_skillrouter_eval._encode_skillrouter_texts", fake_encode)
    adapter_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter_state_dict": {
                "q_proj.weight": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "d_proj.weight": torch.eye(2),
            },
            "config": {"train_doc_projection": False},
            "uses_clstr_heads": False,
        },
        adapter_path,
    )

    report = run_tau2_skillrouter_frozen_eval(
        data_root=tmp_path,
        output_dir=tmp_path / "out_adapter",
        model_name_or_path="fake-skillrouter",
        adapter_checkpoint_path=adapter_path,
        domains=["retail"],
        batch_size=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_finetuned_biencoder_adapter"
    assert report["adapter_checkpoint_path"] == str(adapter_path)
    ranked = [
        json.loads(line)
        for line in (tmp_path / "out_adapter" / "tau2_skillrouter_ranked_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ranked[0]["candidate_next_skill_ids"][0] == tau2_action_skill_id("retail", "refund_order")
