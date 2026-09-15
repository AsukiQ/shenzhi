from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

import scripts.run_logged_online_stage4_train as run_logged_online_stage4_train_cli
import clstr.logged_online_stage4_train as logged_online_stage4_module
from clstr.logged_online_stage4_train import (
    attach_trajectory_prefix_online_memory_scores,
    decide_online_memory_weight_from_calibration,
    decide_online_memory_weight_by_benchmark_from_calibration,
    decide_online_memory_variant_by_benchmark_from_calibration,
    _resolve_logged_online_device,
    _attach_selected_variant_online_memory_scores,
    _zero_online_memory_scores_for_disabled_benchmarks,
    attach_replay_prefixes_from_source_rows,
    build_logged_online_stage4_rows,
    build_logged_online_stage4_rows_with_stage0_handoff,
    evaluate_logged_online_stage4_rows,
    evaluate_logged_online_stage4_rows_by_benchmark,
    run_logged_online_stage4_adaptation_on_rows,
    run_logged_online_stage4_adaptation_with_model,
    split_logged_online_stage4_rows,
)
from scripts.run_logged_online_stage4_train import build_parser, materialize_logged_steps


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _TinySkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3), requires_grad=False)

    def retrieval_logits(self, h):
        return torch.zeros(h.size(0), 3, device=h.device)


class _RoutingSkillTable(_TinySkillTable):
    def retrieval_logits(self, h):
        return h @ self.E.t()


class _CandidateOnlyTransHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, pred, candidate_embs):
        del pred
        return torch.einsum("bcd,d->bc", candidate_embs.float(), self.weight.float())


class _TinyLoggedStage4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skill_table = _TinySkillTable()
        self.trans_head = _CandidateOnlyTransHead()
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        self.transition = torch.nn.Linear(6, 3, bias=False)
        self.unified_route_calls = []
        with torch.no_grad():
            self.action_proj.weight.copy_(torch.eye(3))
            self.transition.weight.zero_()

    @property
    def device(self):
        return torch.device("cpu")

    def encode_observations(self, texts):
        return torch.zeros(1 if not texts else len(texts), 3)

    def action_embeddings(self, labels):
        labels = labels.to(dtype=torch.long)
        return self.skill_table.E.index_select(0, labels.reshape(-1)).view(*labels.shape, -1)

    def initial_belief(self, h_t, top_k=None):
        del top_k
        logits = self.skill_table.retrieval_logits(h_t)
        return torch.softmax(logits, dim=-1) @ self.skill_table.E.to(device=h_t.device, dtype=h_t.dtype)

    def unified_route_full_logits(self, h_t, m_t):
        del h_t
        return m_t @ self.skill_table.E.to(device=m_t.device, dtype=m_t.dtype).t()

    def gather_unified_route_logits(self, full_logits, candidate_rows):
        width = max((len(row) for row in candidate_rows), default=0)
        output = torch.full(
            (len(candidate_rows), width),
            torch.finfo(full_logits.dtype).min,
            dtype=full_logits.dtype,
            device=full_logits.device,
        )
        for row_idx, row in enumerate(candidate_rows):
            if row:
                ids = torch.tensor(row, dtype=torch.long, device=full_logits.device)
                output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = m_t @ self.skill_table.E.to(device=m_t.device, dtype=m_t.dtype).t()
        if candidate_rows is None:
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output


class _TinyBatchNormLoggedStage4Model(_TinyLoggedStage4Model):
    def __init__(self):
        super().__init__()
        self.encoder_bn = torch.nn.BatchNorm1d(3, affine=False, momentum=1.0)

    def encode_observations(self, texts):
        rows = []
        for idx, text in enumerate(texts):
            offset = float((idx % 2) + 1)
            if "next" in str(text).lower():
                rows.append(torch.tensor([offset, 0.0, 1.0]))
            else:
                rows.append(torch.tensor([offset, 1.0, 0.0]))
        return self.encoder_bn(torch.stack(rows))


class _TinyStage0HandoffModel(_TinyLoggedStage4Model):
    def __init__(self):
        super().__init__()
        self.skill_table = _RoutingSkillTable()

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "next-b" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0]))
            elif "state-a" in lowered:
                rows.append(torch.tensor([1.0, 0.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 0.0, 1.0]))
        return torch.stack(rows)


class _ForbiddenLoggedTransition(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("legacy transition path should not run in unified_memory eval")


class _TinyUnifiedLoggedStage4Model(_TinyLoggedStage4Model):
    def __init__(self):
        super().__init__()
        self.transition = _ForbiddenLoggedTransition()
        self.unified_route_calls = []

    def initial_belief(self, h_t, top_k=None):
        del top_k
        return h_t

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = torch.tensor([[0.0, 5.0, -1.0]], dtype=h_t.dtype, device=h_t.device).expand(h_t.size(0), -1)
        if candidate_rows is None:
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output


def _logged_step(step_idx: int, trajectory_id: str = "traj-1") -> dict:
    return {
        "trajectory_id": trajectory_id,
        "task_id": f"{trajectory_id}::{step_idx}",
        "benchmark": "toolbench_g3",
        "split": "train",
        "step_idx": step_idx,
        "goal": "Use the right tool.",
        "history_text": "previous tool",
        "observation_text": "current observation",
        "action_text": "call current",
        "next_observation_text": "next observation",
        "candidate_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "gt_skill_ids": ["skill/a"],
        "gt_next_skill_ids": ["skill/b"],
        "reward_type": "logged_gt",
    }


def test_logged_online_device_prefers_cuda_when_model_starts_on_cpu(monkeypatch):
    class _CpuModel:
        device = torch.device("cpu")

        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = torch.device(device)
            return self

    model = _CpuModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = _resolve_logged_online_device(model)

    assert device == torch.device("cuda")
    assert model.moved_to == torch.device("cuda")


def test_logged_online_eval_uses_auto_replay_prefix_for_recurrent_belief():
    model = _TinyLoggedStage4Model()
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "action_text": "action 0",
            "next_observation_text": "state 1",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, 1.0, 0.0],
        },
        {
            "trajectory_id": "traj-1",
            "step_index": 1,
            "state_text": "state 1",
            "action_text": "action 1",
            "next_observation_text": "state 2",
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
            "skill_idx": 1,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, 0.0, 1.0],
        },
    ]

    metrics = evaluate_logged_online_stage4_rows(
        model,
        rows,
        batch_size=2,
        device=torch.device("cpu"),
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        transition_residual_lambda=0.0,
    )

    assert metrics["stage4_auto_replay_prefix_rows_with_auto_replay_prefix"] == 1.0
    assert metrics["stage4_replay_prefix_used_count"] == 1.0


def test_logged_current_state_evaluators_default_to_unified_memory():
    assert inspect.signature(evaluate_logged_online_stage4_rows).parameters["route_scorer"].default == "unified_memory"
    assert (
        inspect.signature(evaluate_logged_online_stage4_rows_by_benchmark).parameters["route_scorer"].default
        == "unified_memory"
    )


def test_attach_replay_prefixes_from_source_rows_uses_prior_same_trajectory_steps():
    source_rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "action_text": "action 0",
            "next_observation_text": "state 1",
            "skill_id": "skill/a",
            "skill_idx": 0,
        },
        {
            "trajectory_id": "traj-1",
            "step_index": 1,
            "state_text": "state 1",
            "action_text": "action 1",
            "next_observation_text": "state 2",
            "skill_id": "skill/b",
            "skill_idx": 1,
        },
        {
            "trajectory_id": "traj-2",
            "step_index": 0,
            "state_text": "other state",
            "action_text": "other action",
            "next_observation_text": "other next",
            "skill_id": "skill/c",
            "skill_idx": 2,
        },
    ]
    target_rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 1,
            "state_text": "state 1",
            "action_text": "action 1",
            "next_observation_text": "state 2",
            "skill_id": "skill/b",
            "skill_idx": 1,
        },
        {
            "trajectory_id": "traj-2",
            "step_index": 0,
            "state_text": "other state",
            "action_text": "other action",
            "next_observation_text": "other next",
            "skill_id": "skill/c",
            "skill_idx": 2,
        },
    ]

    rows, report = attach_replay_prefixes_from_source_rows(
        target_rows,
        source_rows=source_rows,
        max_steps=3,
    )

    assert report["target_rows"] == 2
    assert report["rows_with_replay_prefix"] == 1
    assert report["total_prefix_steps"] == 1
    assert rows[0]["replay_prefix"] == [
        {
            "observation_text": "state 0",
            "action_text": "action 0",
                "next_observation_text": "state 1",
                "skill_id": "skill/a",
                "observation_source": "row_next_observation_text",
                "trajectory_id": "traj-1",
            "step_index": 0,
            "skill_idx": 0,
        }
    ]
    assert "replay_prefix" not in rows[1]


def test_logged_online_eval_threads_unified_route_scorer():
    model = _TinyUnifiedLoggedStage4Model()
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "action_text": "action 0",
            "next_observation_text": "state 1",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [1.0, 0.0, 0.0],
        }
    ]

    metrics = evaluate_logged_online_stage4_rows(
        model,
        rows,
        batch_size=1,
        device=torch.device("cpu"),
        route_scorer="unified_memory",
    )

    assert metrics["route_scorer"] == "unified_memory"
    assert metrics["uses_stage0_prior_at_inference"] is False
    assert metrics["transition_skill_head_type"] == "unified_memory_retriever"
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert "candidate_recall_all_union_recall" not in metrics
    assert len(model.unified_route_calls) == 2


def test_logged_online_eval_threads_candidate_union_configuration(monkeypatch):
    calls = []

    def recording_current_state_loss(model, batch, device, **kwargs):
        del model, device
        calls.append((list(batch), dict(kwargs)))
        return torch.tensor(0.0), {
            "stage4_act_count": float(len(batch)),
            "candidate_recall_all_source_rows": float(len(batch)),
            "candidate_recall_all_union_hit_rows": float(len(batch)),
            "candidate_recall_all_union_recall": 1.0,
            "candidate_recall_memory_active_source_rows": 0.0,
            "candidate_recall_memory_active_union_hit_rows": 0.0,
            "route_scorer": "unified_memory",
            "transition_skill_head_type": "unified_memory_retriever_candidate_union",
            "uses_stage0_prior_at_inference": False,
        }

    monkeypatch.setattr(
        logged_online_stage4_module,
        "_compute_current_state_route_loss",
        recording_current_state_loss,
    )
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]

    metrics = evaluate_logged_online_stage4_rows(
        _TinyUnifiedLoggedStage4Model(),
        rows,
        batch_size=1,
        device=torch.device("cpu"),
        route_scorer="unified_memory",
        candidate_recall_mode="static_plus_dynamic_extra",
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        equivalent_skill_ids_by_skill_id={"skill/b": ["skill/c"]},
        static_k=2,
        dynamic_extra_k=1,
        final_k=1,
        reliability_mode="fixed_alpha",
        fixed_alpha=0.25,
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=3.0,
    )

    assert metrics["candidate_recall_all_union_recall"] == 1.0
    assert len(calls) == 1
    assert calls[0][1] == {
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "skill_id_to_idx": {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        "equivalent_skill_ids_by_skill_id": {"skill/b": ["skill/c"]},
        "static_k": 2,
        "dynamic_extra_k": 1,
        "final_k": 1,
        "reliability_mode": "fixed_alpha",
        "fixed_alpha": 0.25,
        "memory_utility_gate": None,
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 3.0,
    }


def _route_record_eval_row(
    *,
    trajectory_id: str,
    next_skill_id: str,
    state_text: str,
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "task_id": f"task/{trajectory_id}",
        "row_id": f"row/{trajectory_id}",
        "step_index": 0,
        "source_benchmark": "toolbench_g3",
        "state_text": state_text,
        "skill_id": "skill/a",
        "skill_idx": 0,
        "next_skill_id": next_skill_id,
        "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
        "candidate_next_skill_indices": [0, 1, 2],
        "replay_prefix": [],
    }


def _evaluate_with_route_records(tmp_path, rows, **overrides):
    records_path = tmp_path / "memory_utility_routes.jsonl"
    kwargs = {
        "model": _TinyUnifiedLoggedStage4Model(),
        "rows": rows,
        "batch_size": 2,
        "device": torch.device("cpu"),
        "route_scorer": "unified_memory",
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "skill_id_to_idx": {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        "static_k": 2,
        "dynamic_extra_k": 1,
        "final_k": 1,
        "route_records_path": records_path,
        "route_record_pool_protocol": "known_global",
        "route_record_model_digest": "model-a",
        "route_record_sequential_benchmarks": {"toolbench_g3"},
    }
    kwargs.update(overrides)
    return records_path, evaluate_logged_online_stage4_rows(**kwargs)


def test_logged_online_eval_persists_one_route_record_per_source_row_including_strict_zero(tmp_path):
    rows = [
        _route_record_eval_row(
            trajectory_id="traj-positive",
            next_skill_id="skill/b",
            state_text="positive state",
        ),
        _route_record_eval_row(
            trajectory_id="traj-strict-zero",
            next_skill_id="skill/missing",
            state_text="strict zero state",
        ),
    ]

    records_path, metrics = _evaluate_with_route_records(tmp_path, rows)

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    manifest_path = Path(metrics["route_record_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == len(rows) == 2
    assert metrics["route_record_written_count"] == 2
    assert metrics["route_record_total_count"] == 2
    assert {record["trajectory_id"] for record in records} == {
        "traj-positive",
        "traj-strict-zero",
    }
    strict_zero = next(record for record in records if record["trajectory_id"] == "traj-strict-zero")
    assert strict_zero["positive_mask"] == [False]
    assert len(strict_zero["candidate_indices"]) == 1
    assert strict_zero["static_rank"] == 0
    assert strict_zero["dynamic_rank"] == 0
    assert strict_zero["sequential_benchmark"] is True
    assert all(record["schema_version"] == "memory_utility_route_record_v1" for record in records)
    assert all(len(record["features"]) == 11 for record in records)
    assert all(record["row_digest"] for record in records)
    assert manifest["schema_version"] == "memory_utility_route_manifest_v1"
    assert manifest["record_count"] == 2
    assert manifest["records_sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()
    assert manifest["candidate_union_version"] == "memory_union_v1"
    assert manifest["candidate_selection_version"] == "stable_declared_pool_v1"
    assert manifest["pool_protocol"] == "known_global"
    assert manifest["static_k"] == 2
    assert manifest["dynamic_extra_k"] == 1
    assert manifest["final_k"] == 1
    assert manifest["model_checkpoint_chain_digest"] == "model-a"
    assert manifest["skill_mapping_digest"]
    assert manifest["source_rows_digest"]


def test_logged_online_route_records_append_distinct_rows_with_matching_identity(tmp_path):
    first = _route_record_eval_row(
        trajectory_id="traj-a",
        next_skill_id="skill/b",
        state_text="state a",
    )
    second = _route_record_eval_row(
        trajectory_id="traj-b",
        next_skill_id="skill/c",
        state_text="state b",
    )

    records_path, first_metrics = _evaluate_with_route_records(tmp_path, [first])
    _records_path, second_metrics = _evaluate_with_route_records(tmp_path, [second])

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert first_metrics["route_record_total_count"] == 1
    assert second_metrics["route_record_written_count"] == 1
    assert second_metrics["route_record_total_count"] == 2
    assert [record["trajectory_id"] for record in records] == ["traj-a", "traj-b"]


def test_logged_online_route_records_reject_identity_mismatch_and_duplicate_rows(
    tmp_path,
    monkeypatch,
):
    row = _route_record_eval_row(
        trajectory_id="traj-a",
        next_skill_id="skill/b",
        state_text="state a",
    )
    records_path, _metrics = _evaluate_with_route_records(tmp_path, [row])
    original_bytes = records_path.read_bytes()
    original_builder = logged_online_stage4_module._build_current_state_route_batch

    def fail_if_scored(*_args, **_kwargs):
        raise AssertionError("manifest identity mismatch must fail before route scoring")

    monkeypatch.setattr(
        logged_online_stage4_module,
        "_build_current_state_route_batch",
        fail_if_scored,
    )

    with pytest.raises(ValueError, match="route record manifest identity mismatch"):
        _evaluate_with_route_records(
            tmp_path,
            [_route_record_eval_row(
                trajectory_id="traj-b",
                next_skill_id="skill/c",
                state_text="state b",
            )],
            route_record_model_digest="model-b",
        )
    monkeypatch.setattr(
        logged_online_stage4_module,
        "_build_current_state_route_batch",
        original_builder,
    )
    cached_copy = _route_record_eval_row(
        trajectory_id="traj-a",
        next_skill_id="skill/b",
        state_text="state a",
    )
    cached_copy["_state_embedding"] = torch.zeros(3)
    with pytest.raises(ValueError, match="duplicate route record row_digest"):
        _evaluate_with_route_records(tmp_path, [cached_copy])

    assert records_path.read_bytes() == original_bytes


def test_logged_online_route_records_are_threaded_through_by_benchmark_evaluation(tmp_path):
    first = _route_record_eval_row(
        trajectory_id="bench-a/traj-a",
        next_skill_id="skill/b",
        state_text="state a",
    )
    first["source_benchmark"] = "bench_a"
    second = _route_record_eval_row(
        trajectory_id="bench-b/traj-b",
        next_skill_id="skill/c",
        state_text="state b",
    )
    second["source_benchmark"] = "bench_b"
    records_path = tmp_path / "by_benchmark_routes.jsonl"

    reports = evaluate_logged_online_stage4_rows_by_benchmark(
        _TinyUnifiedLoggedStage4Model(),
        [first, second],
        batch_size=1,
        device=torch.device("cpu"),
        route_scorer="unified_memory",
        candidate_recall_mode="static_plus_dynamic_extra",
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        static_k=2,
        dynamic_extra_k=1,
        final_k=1,
        route_records_path=records_path,
        route_record_pool_protocol="known_global",
        route_record_model_digest="model-a",
        route_record_sequential_benchmarks={"bench_a", "bench_b"},
    )

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert sorted(reports) == ["bench_a", "bench_b"]
    assert len(records) == 2
    assert {record["benchmark"] for record in records} == {"bench_a", "bench_b"}


def test_logged_online_candidate_recall_rejects_missing_raw_counts(monkeypatch):
    def incomplete_current_state_loss(model, batch, device, **kwargs):
        del model, batch, device, kwargs
        return torch.tensor(0.0), {
            "stage4_act_count": 1.0,
            "candidate_recall_all_union_recall": 1.0,
            "route_scorer": "unified_memory",
        }

    monkeypatch.setattr(
        logged_online_stage4_module,
        "_compute_current_state_route_loss",
        incomplete_current_state_loss,
    )

    with pytest.raises(ValueError, match="raw count metrics"):
        evaluate_logged_online_stage4_rows(
            _TinyUnifiedLoggedStage4Model(),
            [
                {
                    "trajectory_id": "traj-1",
                    "step_index": 0,
                    "state_text": "state 0",
                    "skill_id": "skill/a",
                    "next_skill_id": "skill/b",
                    "candidate_next_skill_indices": [0, 1],
                }
            ],
            batch_size=1,
            device=torch.device("cpu"),
            candidate_recall_mode="static_plus_dynamic_extra",
            skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
            static_k=2,
            dynamic_extra_k=1,
            final_k=1,
        )


def test_logged_online_eval_progress_identity_includes_dynamic_extra_budget(tmp_path, monkeypatch):
    calls = []

    def recording_current_state_loss(model, batch, device, **kwargs):
        del model, device
        calls.append(dict(kwargs))
        return torch.tensor(0.0), {
            "stage4_act_count": float(len(batch)),
            "candidate_recall_all_source_rows": float(len(batch)),
            "candidate_recall_all_union_hit_rows": float(len(batch)),
            "candidate_recall_all_union_recall": 1.0,
            "candidate_recall_memory_active_source_rows": 0.0,
            "candidate_recall_memory_active_union_hit_rows": 0.0,
            "route_scorer": "unified_memory",
            "transition_skill_head_type": "unified_memory_retriever_candidate_union",
            "uses_stage0_prior_at_inference": False,
        }

    monkeypatch.setattr(
        logged_online_stage4_module,
        "_compute_current_state_route_loss",
        recording_current_state_loss,
    )
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]
    common = {
        "model": _TinyUnifiedLoggedStage4Model(),
        "rows": rows,
        "batch_size": 1,
        "device": torch.device("cpu"),
        "route_scorer": "unified_memory",
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "skill_id_to_idx": {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        "static_k": 2,
        "final_k": 1,
        "progress_path": tmp_path / "candidate_union_progress.json",
    }

    evaluate_logged_online_stage4_rows(
        dynamic_extra_k=1,
        reliability_mode="fixed_alpha",
        fixed_alpha=0.25,
        **common,
    )
    evaluate_logged_online_stage4_rows(
        dynamic_extra_k=2,
        reliability_mode="fixed_alpha",
        fixed_alpha=0.25,
        **common,
    )
    evaluate_logged_online_stage4_rows(
        dynamic_extra_k=2,
        reliability_mode="fixed_alpha",
        fixed_alpha=0.75,
        **common,
    )

    assert [(call["dynamic_extra_k"], call["fixed_alpha"]) for call in calls] == [
        (1, 0.25),
        (2, 0.25),
        (2, 0.75),
    ]


def test_logged_online_candidate_recall_aggregates_memory_active_denominator_strictly(monkeypatch):
    def strict_batch_metrics(model, batch, device, **kwargs):
        del model, device, kwargs
        active = float(bool(batch[0]["memory_active"]))
        hit = float(bool(batch[0]["union_hit"]))
        active_hit = hit if active else 0.0
        return torch.tensor(0.0), {
            "stage4_act_count": 1.0,
            "candidate_recall_all_source_rows": 1.0,
            "candidate_recall_all_union_hit_rows": hit,
            "candidate_recall_all_union_recall": hit,
            "candidate_recall_memory_active_source_rows": active,
            "candidate_recall_memory_active_union_hit_rows": active_hit,
            "candidate_recall_memory_active_union_recall": active_hit if active else 0.0,
            "route_scorer": "unified_memory",
            "transition_skill_head_type": "unified_memory_retriever_candidate_union",
            "uses_stage0_prior_at_inference": False,
        }

    monkeypatch.setattr(
        logged_online_stage4_module,
        "_compute_current_state_route_loss",
        strict_batch_metrics,
    )
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "state_text": "state 0",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "candidate_next_skill_indices": [0, 1],
            "memory_active": True,
            "union_hit": True,
        },
        {
            "trajectory_id": "traj-2",
            "step_index": 0,
            "state_text": "state 0",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "candidate_next_skill_indices": [0, 1],
            "memory_active": False,
            "union_hit": False,
        },
    ]

    metrics = evaluate_logged_online_stage4_rows(
        _TinyUnifiedLoggedStage4Model(),
        rows,
        batch_size=1,
        device=torch.device("cpu"),
        candidate_recall_mode="static_plus_dynamic_extra",
        skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2},
        static_k=2,
        dynamic_extra_k=1,
        final_k=1,
    )

    assert metrics["candidate_recall_all_union_recall"] == pytest.approx(0.5)
    assert metrics["candidate_recall_memory_active_union_recall"] == pytest.approx(1.0)
    assert metrics["candidate_recall_memory_active_coverage"] == pytest.approx(0.5)


def test_logged_online_unified_eval_never_calls_causal_stage4_loss(monkeypatch):
    def fail_causal_stage4_loss(*_args, **_kwargs):
        raise AssertionError("current-state route evaluation must not call causal Stage4 loss")

    monkeypatch.setattr(logged_online_stage4_module, "_compute_stage4_act_loss", fail_causal_stage4_loss)
    metrics = evaluate_logged_online_stage4_rows(
        _TinyUnifiedLoggedStage4Model(),
        [
            {
                "trajectory_id": "traj-1",
                "step_index": 0,
                "state_text": "state 0",
                "action_text": "current action must be ignored",
                "next_observation_text": "current observation must be ignored",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1, 2],
            }
        ],
        batch_size=1,
        device=torch.device("cpu"),
        route_scorer="unified_memory",
    )

    assert metrics["current_state_route_count"] == 1.0
    assert metrics["stage4_act_count"] == 1.0


def test_logged_online_adaptation_threads_unified_route_scorer(tmp_path):
    model = _TinyUnifiedLoggedStage4Model()
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "source_benchmark": "traject_bench",
            "state_text": "state 0",
            "action_text": "action 0",
            "next_observation_text": "state 1",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "positive_next_skill_position": 1,
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [1.0, 0.0, 0.0],
        }
    ]

    report = run_logged_online_stage4_adaptation_on_rows(
        model=model,
        rows=rows,
        data_report={"stage4_rows": 1},
        output_dir=tmp_path / "unified_route",
        max_updates=0,
        batch_size=1,
        eval_rows=1,
        route_scorer="unified_memory",
    )

    assert report["status"] == "ok"
    assert report["route_scorer"] == "unified_memory"
    assert report["initial_eval"]["route_scorer"] == "unified_memory"
    assert report["prior_eval"]["route_scorer"] == "unified_memory"
    assert report["final_eval"]["route_scorer"] == "unified_memory"


def test_build_logged_online_stage4_rows_converts_schema_to_act_rows():
    rows, report = build_logged_online_stage4_rows(
        [_logged_step(0)],
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
    )

    assert report["source_steps"] == 1
    assert report["stage4_rows"] == 1
    assert report["candidate_source"] == "logged_candidates"
    assert rows[0]["source_benchmark"] == "toolbench_g3"
    assert rows[0]["skill_idx"] == 0
    assert rows[0]["positive_next_skill_idx"] == 1
    assert rows[0]["positive_next_skill_position"] == 1
    assert rows[0]["candidate_next_skill_ids"] == ["skill/a", "skill/b", "skill/c"]
    assert rows[0]["candidate_next_skill_indices"] == [0, 1, 2]
    assert rows[0]["state_text"].startswith("goal: Use the right tool.")
    assert rows[0]["positive_injected"] is False


def test_build_logged_online_stage4_rows_preserves_multiple_valid_next_skills():
    step = _logged_step(0)
    step["gt_next_skill_ids"] = ["skill/b", "skill/c"]
    step["candidate_skill_ids"] = ["skill/a", "skill/c", "skill/b"]

    rows, report = build_logged_online_stage4_rows(
        [step],
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
    )

    assert report["stage4_rows"] == 1
    assert rows[0]["next_skill_id"] == "skill/b"
    assert rows[0]["positive_next_skill_ids"] == ["skill/c", "skill/b"]
    assert rows[0]["positive_next_skill_indices"] == [2, 1]
    assert rows[0]["positive_next_skill_positions"] == [1, 2]
    assert rows[0]["positive_next_skill_idx"] == 2


def test_logged_online_stage4_adaptation_updates_and_evaluates_heldout(tmp_path):
    steps_path = tmp_path / "logged_steps.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "logged_online"
    _write_jsonl(steps_path, [_logged_step(idx, trajectory_id=f"traj-{idx}") for idx in range(6)])
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    model = _TinyLoggedStage4Model()

    report = run_logged_online_stage4_adaptation_with_model(
        model=model,
        logged_steps_path=steps_path,
        skills_path=skills_path,
        output_dir=output_dir,
        max_updates=12,
        batch_size=1,
        eval_interval=4,
        eval_rows=2,
        learning_rate=0.5,
    )

    assert report["status"] == "ok"
    assert report["training_regime"] == "executor_free_logged_online_adaptation_simulation"
    assert report["on_policy_rollout_used"] is False
    assert report["online_update_count"] == 12
    assert report["eval_row_count"] == 2
    assert "prior_eval" in report
    assert "delta_final_vs_prior" in report
    assert set(report["prior_eval_by_benchmark"]) == {"toolbench_g3"}
    assert set(report["final_eval_by_benchmark"]) == {"toolbench_g3"}
    assert set(report["delta_final_vs_prior_by_benchmark"]) == {"toolbench_g3"}
    assert report["final_eval"]["stage4_act_loss"] < report["initial_eval"]["stage4_act_loss"]
    assert report["final_eval"]["stage4_next_skill_recall@1"] >= report["initial_eval"]["stage4_next_skill_recall@1"]
    assert (output_dir / "online_metrics.jsonl").exists()
    assert (output_dir / "checkpoints" / "latest.pt").exists()


def test_logged_online_stage4_trajectory_prefix_split_uses_same_trajectory_suffix_for_eval():
    rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "row_id": "a0",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "positive_next_skill_position": 0,
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "row_id": "a2",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "positive_next_skill_position": 6,
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "row_id": "a1",
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
            "positive_next_skill_position": 3,
        },
        {
            "trajectory_id": "traj-b",
            "step_index": 0,
            "row_id": "b0",
            "skill_id": "skill/d",
            "next_skill_id": "skill/f",
            "positive_next_skill_position": 20,
        },
        {
            "trajectory_id": "traj-b",
            "step_index": 1,
            "row_id": "b1",
            "skill_id": "skill/d",
            "next_skill_id": "skill/e",
            "positive_next_skill_position": 104,
        },
        {
            "trajectory_id": "traj-c",
            "step_index": 0,
            "row_id": "c0",
            "skill_id": "skill/g",
            "next_skill_id": "skill/h",
            "positive_next_skill_position": 1,
        },
    ]

    stream_rows, eval_rows, split_report = split_logged_online_stage4_rows(
        rows,
        eval_rows=2,
        eval_split_mode="trajectory_prefix",
        trajectory_eval_steps=1,
    )

    assert [row["row_id"] for row in eval_rows] == ["a2", "b1"]
    assert [row["row_id"] for row in stream_rows] == ["a0", "a1", "b0", "c0"]
    assert split_report["eval_split_mode"] == "trajectory_prefix"
    assert split_report["selected_trajectory_count"] == 2
    assert split_report["single_step_trajectory_count"] == 1
    assert split_report["stream_positive_rank_bucket_counts"] == {"1": 1, "2-5": 2, "21-100": 1}
    assert split_report["eval_positive_rank_bucket_counts"] == {"6-20": 1, ">100": 1}
    assert split_report["eval_same_trajectory_prefix_rows"] == 2
    assert split_report["eval_next_skill_seen_in_prefix_count"] == 1
    assert split_report["eval_current_to_next_transition_seen_in_prefix_count"] == 1


def test_logged_online_stage4_trajectory_prefix_split_applies_eval_benchmark_caps():
    rows = []
    for benchmark in ["alfworld", "webshop", "traject_bench", "toolbench_g3"]:
        for traj_idx in range(2):
            for step_idx in range(2):
                rows.append(
                    {
                        "source_benchmark": benchmark,
                        "trajectory_id": f"{benchmark}-{traj_idx}",
                        "step_index": step_idx,
                        "row_id": f"{benchmark}-{traj_idx}-{step_idx}",
                        "skill_id": "skill/current",
                        "next_skill_id": "skill/next",
                        "positive_next_skill_position": 0,
                    }
                )

    stream_rows, eval_rows, split_report = split_logged_online_stage4_rows(
        rows,
        eval_rows=8,
        eval_split_mode="trajectory_prefix",
        trajectory_eval_steps=1,
        eval_benchmark_caps={"alfworld": 1, "webshop": 1, "traject_bench": 1, "toolbench_g3": 1},
    )

    assert [row["source_benchmark"] for row in eval_rows] == [
        "alfworld",
        "webshop",
        "traject_bench",
        "toolbench_g3",
    ]
    assert all(row["step_index"] == 1 for row in eval_rows)
    assert len(stream_rows) == len(rows) - 4
    assert split_report["eval_benchmark_caps"]["enabled"] is True
    assert split_report["eval_benchmark_counts"] == {
        "alfworld": 1,
        "toolbench_g3": 1,
        "traject_bench": 1,
        "webshop": 1,
    }


def test_logged_online_stage4_builder_applies_benchmark_caps_before_max_rows():
    steps = []
    for benchmark in ["alfworld", "webshop", "traject_bench", "toolbench_g3"]:
        for idx in range(3):
            step = _logged_step(idx, trajectory_id=f"{benchmark}-{idx}")
            step["benchmark"] = benchmark
            steps.append(step)

    rows, report = build_logged_online_stage4_rows(
        steps,
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        max_rows=4,
        benchmark_caps={"alfworld": 1, "webshop": 2, "traject_bench": 1, "toolbench_g3": 2},
    )

    assert report["benchmark_caps"]["enabled"] is True
    assert report["benchmark_caps"]["retained_by_benchmark"] == {
        "alfworld": 1,
        "toolbench_g3": 2,
        "traject_bench": 1,
        "webshop": 2,
    }
    assert len(rows) == 4
    assert {row["source_benchmark"] for row in rows} == {"alfworld", "webshop", "traject_bench"}


def test_attach_trajectory_prefix_online_memory_scores_uses_only_past_same_trajectory_feedback():
    feedback_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "skill_id": "skill/current",
            "next_skill_id": "skill/gold",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 3,
            "skill_id": "skill/current",
            "next_skill_id": "skill/future",
        },
        {
            "trajectory_id": "traj-b",
            "step_index": 0,
            "skill_id": "skill/current",
            "next_skill_id": "skill/cross",
        },
    ]
    target_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "skill_id": "skill/current",
            "next_skill_id": "skill/gold",
            "candidate_next_skill_ids": ["skill/other", "skill/gold", "skill/future", "skill/cross"],
        }
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        target_rows,
        feedback_rows=feedback_rows,
        next_skill_bonus=2.0,
        exact_transition_bonus=3.0,
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [0.0, 5.0, 0.0, 0.0]
    assert report["online_memory_rows"] == 1
    assert report["online_memory_positive_boosted_rows"] == 1
    assert report["online_memory_exact_transition_rows"] == 1


def test_attach_trajectory_prefix_online_memory_scores_defaults_to_latest_exact_transition():
    feedback_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "skill_id": "skill/current",
            "next_skill_id": "skill/old",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "skill_id": "skill/current",
            "next_skill_id": "skill/latest",
        },
    ]
    target_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "skill_id": "skill/current",
            "next_skill_id": "skill/latest",
            "candidate_next_skill_ids": ["skill/old", "skill/latest"],
        }
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        target_rows,
        feedback_rows=feedback_rows,
        next_skill_bonus=0.0,
        exact_transition_bonus=3.0,
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [0.0, 3.0]
    assert report["online_memory_mode"] == "latest_exact"


def test_attach_trajectory_prefix_online_memory_scores_state_conditioned_uses_similar_prefix_state():
    feedback_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "state_text": "goal clean kitchen observation open fridge cold food",
            "skill_id": "skill/current",
            "next_skill_id": "skill/fridge",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "state_text": "goal clean kitchen observation wooden table dirty plate",
            "skill_id": "skill/current",
            "next_skill_id": "skill/table",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 4,
            "state_text": "goal clean kitchen observation open fridge cold food",
            "skill_id": "skill/current",
            "next_skill_id": "skill/future",
        },
    ]
    target_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "state_text": "observation open fridge cold food in kitchen",
            "skill_id": "skill/current",
            "next_skill_id": "skill/fridge",
            "candidate_next_skill_ids": ["skill/fridge", "skill/table", "skill/future"],
        }
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        target_rows,
        feedback_rows=feedback_rows,
        next_skill_bonus=0.0,
        exact_transition_bonus=4.0,
        memory_mode="state_conditioned",
        state_similarity_threshold=0.25,
    )

    scores = scored_rows[0]["candidate_next_online_memory_scores"]
    assert scores[0] > scores[1]
    assert scores[2] == 0.0
    assert report["online_memory_mode"] == "state_conditioned"
    assert report["online_memory_state_conditioned_rows"] == 1


def test_attach_trajectory_prefix_online_memory_scores_inventory_remaining_ignores_gt_tool_inventory():
    target_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "tool_inventory_skill_ids": ["skill/a", "skill/b"],
            "candidate_next_skill_ids": ["skill/a", "skill/b"],
        }
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        target_rows,
        feedback_rows=[],
        next_skill_bonus=0.25,
        exact_transition_bonus=3.0,
        memory_mode="inventory_remaining",
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [0.0, 0.0]
    assert report["online_memory_rows"] == 0
    assert report["online_memory_positive_boosted_rows"] == 0


def test_attach_trajectory_prefix_online_memory_scores_inventory_remaining_uses_visible_unfinished_inventory():
    feedback_rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
        },
    ]
    target_rows = [
        {
            "source_benchmark": "any_source",
            "trajectory_id": "traj-a",
            "step_index": 2,
            "skill_id": "skill/c",
            "next_skill_id": "skill/d",
            "visible_inventory_skill_ids": ["skill/a", "skill/b", "skill/c", "skill/d"],
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c", "skill/d", "skill/outside"],
        }
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        target_rows,
        feedback_rows=feedback_rows,
        next_skill_bonus=0.25,
        exact_transition_bonus=3.0,
        memory_mode="inventory_remaining",
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [0.25, 0.25, 0.25, 3.25, 0.0]
    assert report["online_memory_mode"] == "inventory_remaining"
    assert report["online_memory_rows"] == 1
    assert report["online_memory_positive_boosted_rows"] == 1


def test_inventory_remaining_memory_uses_past_prefix_without_future_same_trajectory_rows():
    rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "visible_inventory_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "skill_id": "skill/b",
            "next_skill_id": "skill/c",
            "visible_inventory_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "skill_id": "skill/c",
            "next_skill_id": "skill/a",
            "visible_inventory_skill_ids": ["skill/a", "skill/b", "skill/c"],
            "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
        },
    ]

    scored_rows, report = attach_trajectory_prefix_online_memory_scores(
        rows,
        feedback_rows=rows,
        next_skill_bonus=0.25,
        exact_transition_bonus=3.0,
        memory_mode="inventory_remaining",
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [0.25, 3.25, 3.25]
    assert scored_rows[1]["candidate_next_online_memory_scores"] == [0.25, 0.25, 3.25]
    assert scored_rows[2]["candidate_next_online_memory_scores"] == [0.25, 0.25, 0.25]
    assert report["online_memory_inventory_remaining_rows"] == 3


def test_decide_online_memory_weight_from_calibration_disables_memory_on_negative_delta():
    decision = decide_online_memory_weight_from_calibration(
        requested_weight=1.0,
        prior_eval={
            "stage4_next_skill_mrr": 0.55,
            "stage4_next_skill_recall@5": 0.9,
        },
        memory_eval={
            "stage4_next_skill_mrr": 0.50,
            "stage4_next_skill_recall@5": 0.9,
        },
    )

    assert decision["enabled"] is False
    assert decision["effective_online_memory_weight"] == 0.0


def test_decide_online_memory_weight_from_calibration_keeps_memory_on_positive_delta():
    decision = decide_online_memory_weight_from_calibration(
        requested_weight=0.5,
        prior_eval={
            "stage4_next_skill_mrr": 0.20,
            "stage4_next_skill_recall@5": 0.3,
        },
        memory_eval={
            "stage4_next_skill_mrr": 0.25,
            "stage4_next_skill_recall@5": 0.35,
        },
    )

    assert decision["enabled"] is True
    assert decision["effective_online_memory_weight"] == 0.5


def test_online_memory_source_benchmark_gate_can_keep_helpful_source_only():
    decisions = decide_online_memory_weight_by_benchmark_from_calibration(
        requested_weight=1.0,
        prior_eval_by_benchmark={
            "toolbench_g3": {"stage4_next_skill_mrr": 0.2, "stage4_next_skill_recall@5": 0.3},
            "alfworld": {"stage4_next_skill_mrr": 0.9, "stage4_next_skill_recall@5": 1.0},
        },
        memory_eval_by_benchmark={
            "toolbench_g3": {"stage4_next_skill_mrr": 0.3, "stage4_next_skill_recall@5": 0.4},
            "alfworld": {"stage4_next_skill_mrr": 0.95, "stage4_next_skill_recall@5": 0.99},
        },
    )

    assert decisions["enabled_benchmarks"] == ["toolbench_g3"]
    assert decisions["disabled_benchmarks"] == ["alfworld"]
    assert decisions["effective_online_memory_weight"] == 1.0
    assert decisions["by_benchmark"]["toolbench_g3"]["enabled"] is True
    assert decisions["by_benchmark"]["alfworld"]["enabled"] is False
    assert decisions["effective_online_memory_weight"] == 1.0


def test_online_memory_variant_gate_selects_best_safe_variant_per_source():
    decisions = decide_online_memory_variant_by_benchmark_from_calibration(
        prior_eval_by_benchmark={
            "toolbench_g3": {"stage4_next_skill_mrr": 0.20, "stage4_next_skill_recall@5": 0.30},
            "alfworld": {"stage4_next_skill_mrr": 0.90, "stage4_next_skill_recall@5": 1.00},
        },
        variant_eval_by_benchmark={
            "toolbench_g3": {
                "latest_exact": {"stage4_next_skill_mrr": 0.27, "stage4_next_skill_recall@5": 0.36},
                "state_conditioned": {"stage4_next_skill_mrr": 0.25, "stage4_next_skill_recall@5": 0.35},
                "inventory_remaining": {"stage4_next_skill_mrr": 0.30, "stage4_next_skill_recall@5": 0.38},
            },
            "alfworld": {
                "latest_exact": {"stage4_next_skill_mrr": 0.91, "stage4_next_skill_recall@5": 0.99},
                "state_conditioned": {"stage4_next_skill_mrr": 0.95, "stage4_next_skill_recall@5": 0.99},
                "inventory_remaining": {"stage4_next_skill_mrr": 0.93, "stage4_next_skill_recall@5": 0.99},
            },
        },
    )

    assert decisions["selected_mode_by_benchmark"] == {
        "alfworld": "none",
        "toolbench_g3": "inventory_remaining",
    }
    assert decisions["enabled_benchmarks"] == ["toolbench_g3"]
    assert decisions["disabled_benchmarks"] == ["alfworld"]


def test_attach_selected_variant_online_memory_scores_uses_source_specific_modes():
    feedback_rows = [
        {
            "source_benchmark": "toolbench_g3",
            "trajectory_id": "tool-traj",
            "step_index": 0,
            "state_text": "tool previous state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/tool",
        },
        {
            "source_benchmark": "alfworld",
            "trajectory_id": "alf-traj",
            "step_index": 0,
            "state_text": "alf previous state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/alf",
        },
    ]
    target_rows = [
        {
            "source_benchmark": "toolbench_g3",
            "trajectory_id": "tool-traj",
            "step_index": 1,
            "state_text": "tool previous state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/tool",
            "candidate_next_skill_ids": ["skill/tool", "skill/other"],
        },
        {
            "source_benchmark": "alfworld",
            "trajectory_id": "alf-traj",
            "step_index": 1,
            "state_text": "alf previous state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/alf",
            "candidate_next_skill_ids": ["skill/alf", "skill/other"],
        },
    ]

    scored_rows, report = _attach_selected_variant_online_memory_scores(
        target_rows,
        feedback_rows=feedback_rows,
        selected_mode_by_benchmark={"toolbench_g3": "latest_exact", "alfworld": "none"},
        next_skill_bonus=0.0,
        exact_transition_bonus=3.0,
    )

    assert scored_rows[0]["candidate_next_online_memory_scores"] == [3.0, 0.0]
    assert scored_rows[1]["candidate_next_online_memory_scores"] == [0.0, 0.0]
    assert report["selected_mode_by_benchmark"] == {"alfworld": "none", "toolbench_g3": "latest_exact"}


def test_zero_online_memory_scores_for_disabled_benchmarks_preserves_enabled_sources():
    rows = [
        {
            "source_benchmark": "toolbench_g3",
            "candidate_next_online_memory_scores": [0.0, 5.0],
        },
        {
            "source_benchmark": "alfworld",
            "candidate_next_online_memory_scores": [0.0, 5.0],
        },
    ]

    filtered = _zero_online_memory_scores_for_disabled_benchmarks(
        rows,
        {"toolbench_g3": True, "alfworld": False},
    )

    assert filtered[0]["candidate_next_online_memory_scores"] == [0.0, 5.0]
    assert filtered[1]["candidate_next_online_memory_scores"] == [0.0, 0.0]
    assert rows[1]["candidate_next_online_memory_scores"] == [0.0, 5.0]


def test_source_benchmark_gate_uses_eval_benchmark_caps_for_calibration(tmp_path):
    rows = []
    for benchmark in ["alfworld", "toolbench_g3"]:
        for traj_idx in range(2):
            for step_idx in range(3):
                rows.append(
                    {
                        "source_benchmark": benchmark,
                        "trajectory_id": f"{benchmark}-{traj_idx}",
                        "step_index": step_idx,
                        "state_text": f"{benchmark} state",
                        "action_text": "act",
                        "next_observation_text": "next",
                        "skill_id": "skill/a",
                        "next_skill_id": "skill/b",
                        "skill_idx": 0,
                        "positive_next_skill_idx": 1,
                        "positive_next_skill_position": 1,
                        "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
                        "candidate_next_skill_indices": [0, 1, 2],
                        "candidate_next_prior_scores": [0.0, -1.0, -2.0],
                    }
                )

    report = run_logged_online_stage4_adaptation_on_rows(
        model=_TinyLoggedStage4Model(),
        rows=rows,
        data_report={"stage4_rows": len(rows)},
        output_dir=tmp_path / "source_gate",
        max_updates=0,
        batch_size=2,
        eval_rows=2,
        eval_split_mode="trajectory_prefix",
        trajectory_eval_steps=1,
        eval_benchmark_caps={"alfworld": 1, "toolbench_g3": 1},
        online_memory_weight=1.0,
        online_memory_auto_gate=True,
        online_memory_gate_scope="source_benchmark",
        online_memory_gate_eval_rows=2,
    )

    assert set(report["online_memory_gate"]["by_benchmark"]) == {"alfworld", "toolbench_g3"}
    assert set(report["prior_eval_by_benchmark"]) == {"alfworld", "toolbench_g3"}
    assert set(report["final_eval_by_benchmark"]) == {"alfworld", "toolbench_g3"}
    assert set(report["delta_final_vs_prior_by_benchmark"]) == {"alfworld", "toolbench_g3"}
    assert report["online_memory_gate"]["gate_split"]["eval_benchmark_counts"] == {
        "alfworld": 1,
        "toolbench_g3": 1,
    }


def test_logged_online_stage4_can_train_score_calibrator_only(tmp_path):
    rows = []
    for idx in range(4):
        rows.append(
            {
                "source_benchmark": "toolbench_g3",
                "trajectory_id": f"traj-{idx}",
                "step_index": 0,
                "state_text": "toolbench state",
                "action_text": "act",
                "next_observation_text": "next",
                "skill_id": "skill/a",
                "next_skill_id": "skill/c",
                "skill_idx": 0,
                "positive_next_skill_idx": 2,
                "positive_next_skill_position": 2,
                "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
                "candidate_next_skill_indices": [0, 1, 2],
                "candidate_next_prior_scores": [0.0, -1.0, -5.0],
                "candidate_next_online_memory_scores": [0.0, 0.0, 1.0],
            }
        )

    report = run_logged_online_stage4_adaptation_on_rows(
        model=_TinyLoggedStage4Model(),
        rows=rows,
        data_report={"stage4_rows": len(rows)},
        output_dir=tmp_path / "score_calibrator",
        max_updates=2,
        batch_size=2,
        eval_rows=2,
        learning_rate=0.5,
        train_score_calibrator=True,
        online_memory_weight=1.0,
    )

    assert report["status"] == "ok"
    assert report["train_score_calibrator"] is True
    assert report["freeze_report"]["trainable_modules"] == ["stage4_score_calibrator"]
    assert report["best_update"] >= 0
    assert report["best_eval"]["stage4_next_skill_mrr"] >= report["initial_eval"]["stage4_next_skill_mrr"]
    assert report["final_eval"]["stage4_next_skill_mrr"] == report["best_eval"]["stage4_next_skill_mrr"]
    assert report["prior_eval"]["stage4_score_calibrator_enabled"] == 0.0
    assert report["final_eval"]["stage4_score_calibrator_enabled"] == 1.0
    assert Path(report["best_checkpoint"]).exists()
    assert report["last_metrics"]["stage4_score_calibrator_enabled"] is True
    payload = torch.load(report["latest_checkpoint"], map_location="cpu")
    assert "stage4_score_calibrator.weight" in payload["model_state_dict"]
    assert sorted(payload["model_state_dict"]) == ["stage4_score_calibrator.weight"]
    best_payload = torch.load(report["best_checkpoint"], map_location="cpu")
    assert "stage4_score_calibrator.weight" in best_payload["stage4_score_calibrator_state_dict"]


def test_logged_online_stage4_calibrator_only_does_not_update_frozen_base_buffers(tmp_path):
    rows = []
    for idx in range(4):
        rows.append(
            {
                "source_benchmark": "toolbench_g3",
                "trajectory_id": f"traj-{idx}",
                "step_index": 0,
                "state_text": f"state {idx}",
                "action_text": "act",
                "next_observation_text": f"next {idx}",
                "skill_id": "skill/a",
                "next_skill_id": "skill/c",
                "skill_idx": 0,
                "positive_next_skill_idx": 2,
                "positive_next_skill_position": 2,
                "candidate_next_skill_ids": ["skill/a", "skill/b", "skill/c"],
                "candidate_next_skill_indices": [0, 1, 2],
                "candidate_next_prior_scores": [0.0, -1.0, -5.0],
                "candidate_next_online_memory_scores": [0.0, 0.0, 1.0],
            }
        )
    model = _TinyBatchNormLoggedStage4Model()
    initial_running_mean = model.encoder_bn.running_mean.detach().clone()

    report = run_logged_online_stage4_adaptation_on_rows(
        model=model,
        rows=rows,
        data_report={"stage4_rows": len(rows)},
        output_dir=tmp_path / "score_calibrator_bn",
        max_updates=2,
        batch_size=2,
        eval_rows=2,
        learning_rate=0.5,
        train_score_calibrator=True,
        online_memory_weight=1.0,
    )

    assert report["status"] == "ok"
    assert torch.equal(model.encoder_bn.running_mean, initial_running_mean)
    assert report["final_eval"]["stage4_next_skill_mrr"] == report["best_eval"]["stage4_next_skill_mrr"]


def test_build_logged_online_stage4_rows_with_stage0_handoff_uses_topm_candidates(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage0_handoff"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "task_id": "traj-1::0",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state-a",
                "action_text": "call a",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "visible_inventory_skill_ids": ["skill/a", "skill/b"],
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "task_id": "traj-1::1",
                "step_index": 1,
                "state_text": "next-b",
                "skill_id": "skill/b",
                "done": True,
                "provenance": {"split": "train"},
            },
        ],
    )
    _write_jsonl(skills, [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}])

    rows, report = build_logged_online_stage4_rows_with_stage0_handoff(
        model=_TinyStage0HandoffModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=1,
    )

    assert report["stage0_candidate_handoff"]["enabled"] is True
    assert report["stage0_candidate_handoff"]["next_positive_coverage@M"] == 1.0
    assert report["causal_next_state"]["attached_rows"] == 1
    assert rows[0]["next_skill_id"] == "skill/b"
    assert rows[0]["candidate_next_skill_ids"] == ["skill/b", "skill/a"]
    assert rows[0]["visible_inventory_skill_ids"] == ["skill/a", "skill/b"]
    assert "tool_inventory_skill_ids" not in rows[0]
    assert rows[0]["positive_next_skill_position"] == 0
    assert rows[0]["candidate_next_prior_scores"][0] > rows[0]["candidate_next_prior_scores"][1]


def test_build_logged_online_stage4_rows_full_pool_retains_static_miss_without_injection(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "task_id": "traj-1::0",
                "step_index": 0,
                "state_text": "state-a",
                "action_text": "call a",
                "next_observation_text": "next-c",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "task_id": "traj-1::1",
                "step_index": 1,
                "state_text": "next-c",
                "skill_id": "skill/b",
                "done": True,
                "provenance": {"split": "train"},
            },
        ],
    )
    _write_jsonl(skills, [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}])

    rows, report = build_logged_online_stage4_rows_with_stage0_handoff(
        model=_TinyStage0HandoffModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=tmp_path / "full_pool_handoff",
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=1,
        next_skill_pool_mode="full_pool",
    )

    assert len(rows) == 1
    assert rows[0]["next_skill_id"] == "skill/b"
    assert rows[0]["positive_next_skill_position"] is None
    assert rows[0]["positive_injected"] is False
    assert report["candidate_source"] == "declared_legal_full_skill_pool"
    assert report["stage0_next_static_miss_retained_rows"] == 1
    assert report["positive_injected_rows"] == 0


def test_logged_online_stage4_adaptation_runs_on_prebuilt_handoff_rows(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage0_handoff_train"
    _write_jsonl(
        trajectories,
        [
            row
            for idx in range(4)
            for row in (
                {
                    "benchmark": "toolbench_g3",
                    "trajectory_id": f"traj-{idx}",
                    "task_id": f"traj-{idx}::0",
                    "step_index": 0,
                    "goal_text": "goal",
                    "state_text": "state-a",
                    "action_text": "call a",
                    "next_observation_text": "next-b",
                    "skill_id": "skill/a",
                    "next_skill_id": "skill/b",
                    "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                    "provenance": {"split": "train"},
                },
                {
                    "benchmark": "toolbench_g3",
                    "trajectory_id": f"traj-{idx}",
                    "task_id": f"traj-{idx}::1",
                    "step_index": 1,
                    "state_text": "next-b",
                    "skill_id": "skill/b",
                    "done": True,
                    "provenance": {"split": "train"},
                },
            )
        ],
    )
    _write_jsonl(skills, [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}])
    model = _TinyStage0HandoffModel()
    rows, data_report = build_logged_online_stage4_rows_with_stage0_handoff(
        model=model,
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=tmp_path / "stage0_handoff_rows",
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=1,
    )

    report = run_logged_online_stage4_adaptation_on_rows(
        model=model,
        rows=rows,
        data_report=data_report,
        output_dir=output_dir,
        max_updates=4,
        batch_size=1,
        eval_interval=2,
        eval_rows=1,
        learning_rate=0.5,
    )

    assert report["status"] == "ok"
    assert data_report["causal_next_state"]["attached_rows"] == 4
    assert report["data_report"]["candidate_source"] == "stage0_topm_online"
    assert report["online_update_count"] == 4
    assert (output_dir / "online_metrics.jsonl").exists()
    assert (output_dir / "checkpoints" / "latest.pt").exists()


def test_materialize_logged_steps_from_raw_trajectories_for_cli(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output = tmp_path / "logged_steps.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "candidate_next_skill_ids": ["skill/a", "skill/b"],
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_jsonl(skills, [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])

    report = materialize_logged_steps(
        trajectories_path=trajectories,
        skills_path=skills,
        output_path=output,
        max_rows=None,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["materialized_steps"] == 1
    assert rows[0]["gt_next_skill_ids"] == ["skill/b"]


def test_logged_online_stage4_cli_accepts_stage0_handoff_arguments():
    args = build_parser().parse_args(
        [
            "--use_stage0_handoff",
            "--stage0_top_m",
            "128",
            "--stage0_positive_missing_policy",
            "skip",
            "--stage0_handoff_query_mode",
            "raw_state",
            "--stage0_candidate_encode_batch_size",
            "3",
            "--eval_split_mode",
            "trajectory_prefix",
            "--trajectory_eval_steps",
            "2",
            "--online_memory_weight",
            "1.5",
            "--online_memory_next_skill_bonus",
            "2.5",
            "--online_memory_exact_transition_bonus",
            "4.5",
            "--online_memory_mode",
            "inventory_remaining",
            "--online_memory_state_similarity_threshold",
            "0.2",
            "--online_memory_state_similarity_temperature",
            "0.5",
            "--online_memory_auto_gate",
            "--online_memory_gate_scope",
            "source_benchmark",
            "--online_memory_gate_eval_rows",
            "17",
            "--online_memory_gate_min_delta_mrr",
            "0.01",
            "--online_memory_gate_min_delta_recall5",
            "-0.02",
            "--train_score_calibrator",
            "--benchmark_caps",
            "toolbench_g3=-1:traject_bench=1024,alfworld=512",
            "--eval_benchmark_caps",
            "toolbench_g3=64:traject_bench=64,alfworld=32",
            "--save_stage4_rows_path",
            "cache/stage4_rows.jsonl",
            "--prebuilt_stage4_rows_path",
            "cache/prebuilt_rows.jsonl",
            "--replay_prefix_source_rows_path",
            "cache/full_rows.jsonl",
            "--replay_prefix_max_steps",
            "5",
        ]
    )

    assert args.use_stage0_handoff is True
    assert args.stage0_top_m == 128
    assert args.stage0_positive_missing_policy == "skip"
    assert args.stage0_handoff_query_mode == "raw_state"
    assert args.stage0_candidate_encode_batch_size == 3
    assert args.eval_split_mode == "trajectory_prefix"
    assert args.trajectory_eval_steps == 2
    assert args.transition_scoring_mode == "stage0_rank_prior_plus_transition_residual"
    assert args.online_memory_weight == 1.5
    assert args.online_memory_next_skill_bonus == 2.5
    assert args.online_memory_exact_transition_bonus == 4.5
    assert args.online_memory_mode == "inventory_remaining"
    assert args.online_memory_state_similarity_threshold == 0.2
    assert args.online_memory_state_similarity_temperature == 0.5
    assert args.online_memory_auto_gate is True
    assert args.online_memory_gate_scope == "source_benchmark"
    assert args.online_memory_gate_eval_rows == 17
    assert args.online_memory_gate_min_delta_mrr == 0.01
    assert args.online_memory_gate_min_delta_recall5 == -0.02
    assert args.train_score_calibrator is True
    assert args.benchmark_caps == "toolbench_g3=-1:traject_bench=1024,alfworld=512"
    assert args.eval_benchmark_caps == "toolbench_g3=64:traject_bench=64,alfworld=32"
    assert args.save_stage4_rows_path == "cache/stage4_rows.jsonl"
    assert args.prebuilt_stage4_rows_path == "cache/prebuilt_rows.jsonl"
    assert args.replay_prefix_source_rows_path == "cache/full_rows.jsonl"
    assert args.replay_prefix_max_steps == 5


def test_logged_online_stage4_cli_prebuilt_rows_do_not_materialize_missing_trajectories(tmp_path, monkeypatch):
    prebuilt_rows = tmp_path / "prebuilt_rows.jsonl"
    prebuilt_rows.write_text(
        json.dumps(
            {
                "trajectory_id": "traj-1",
                "step_index": 0,
                "state_text": "state",
                "action_text": "action",
                "next_observation_text": "next",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1],
                "candidate_next_prior_scores": [0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_logged_online_stage4_train_cli,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **kwargs: (_TinyLoggedStage4Model(), {"model": "tiny"}, {"routing": "ok"}),
    )
    monkeypatch.setattr(
        run_logged_online_stage4_train_cli,
        "load_routing_and_head_checkpoints",
        lambda *args, **kwargs: {"loaded": True},
    )

    def fake_run_logged_online_stage4_adaptation_on_rows(**kwargs):
        captured["rows"] = kwargs["rows"]
        captured["input_report"] = kwargs["input_report"]
        return {"status": "ok", "eval_row_count": len(kwargs["rows"])}

    monkeypatch.setattr(
        run_logged_online_stage4_train_cli,
        "run_logged_online_stage4_adaptation_on_rows",
        fake_run_logged_online_stage4_adaptation_on_rows,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_logged_online_stage4_train.py",
            "--trajectories_path",
            str(tmp_path / "missing_trajectories.jsonl"),
            "--skills_path",
            str(tmp_path / "missing_skills.jsonl"),
            "--routing_checkpoint_path",
            str(tmp_path / "routing.pt"),
            "--head_checkpoint_path",
            str(tmp_path / "head.pt"),
            "--output_dir",
            str(output_dir),
            "--prebuilt_stage4_rows_path",
            str(prebuilt_rows),
            "--max_updates",
            "0",
        ],
    )

    assert run_logged_online_stage4_train_cli.main() == 0
    assert captured["rows"][0]["next_skill_id"] == "skill/b"
