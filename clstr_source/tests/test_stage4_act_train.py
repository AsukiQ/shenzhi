from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import clstr.stage4_act_train as stage4_module
from clstr.candidate_admission_residual import (
    CANDIDATE_ADMISSION_RESIDUAL_V1,
    CandidateAdmissionResidualHead,
)
from clstr.counterfactual_ranking import full_pool_causal_route_objective
from clstr.counterfactual_memory_calibration import RouteMemoryResidualAdapter
from clstr.memory_utility_gate import MemoryUtilityGate
from clstr.model import RouteMemoryUtilityGate
from clstr.stage4_act_train import (
    COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
    _build_stage4_candidate_admission_route_batch,
    _build_stage4_cmc_route_batch,
    _build_stage4_safe_route_batch,
    _compute_stage4_act_loss,
    _ensure_stage4_score_calibrator,
    _freeze_for_stage4_act,
    _build_stage4_next_skill_rows_from_source_rows,
    _stage4_lr_multiplier,
    _stable_negative_fill,
    _validate_candidate_admission_training_support,
    build_stage4_next_skill_rows,
    train_stage4_act_with_model,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stage4_causal_source_row() -> dict:
    return {
        "benchmark": "toolbench_g3",
        "task_id": "causal-0",
        "trajectory_id": "causal-traj",
        "step_index": 0,
        "state_text": "current state",
        "action_text": "call current",
        "next_observation_text": "observation after action",
        "next_state_text": "trusted successor state",
        "skill_id": "skill/current",
        "next_skill_id": "skill/next",
        "candidate_next_skill_ids": ["skill/current", "skill/next"],
        "provenance": {"source_id": "causal-source", "split": "train"},
    }


def _stage4_raw_causal_pair(benchmark: str, trajectory_id: str, next_skill_id: str) -> list[dict]:
    source_id = f"{benchmark}-source"
    return [
        {
            "benchmark": benchmark,
            "task_id": f"{trajectory_id}-0",
            "trajectory_id": trajectory_id,
            "step_index": 0,
            "state_text": f"{benchmark} current",
            "action_text": f"{benchmark} action",
            "next_observation_text": f"{benchmark} observation",
            "skill_id": "skill/current",
            "next_skill_id": next_skill_id,
            "provenance": {"source_id": source_id, "split": "train"},
        },
        {
            "benchmark": benchmark,
            "task_id": f"{trajectory_id}-1",
            "trajectory_id": trajectory_id,
            "step_index": 1,
            "state_text": f"{benchmark} trusted successor",
            "skill_id": next_skill_id,
            "provenance": {"source_id": source_id, "split": "train"},
        },
    ]


def _stage4_strict_causal_pairs(source_rows: list[dict]) -> list[dict]:
    paired: list[dict] = []
    for row_idx, row in enumerate(source_rows):
        current = dict(row)
        provenance = dict(current.get("provenance") or {})
        provenance.setdefault("source_id", "stage4-test-source")
        provenance.setdefault("split", "train")
        trajectory_base = str(current.get("trajectory_id") or current.get("task_id") or "stage4-test")
        trajectory_id = f"{trajectory_base}:{row_idx}"
        current["trajectory_id"] = trajectory_id
        current["step_index"] = 0
        current["provenance"] = provenance
        successor = {
            "benchmark": current.get("benchmark"),
            "task_id": f"{current.get('task_id')}:successor",
            "trajectory_id": trajectory_id,
            "step_index": 1,
            "state_text": str(current.get("next_observation_text") or "trusted successor state"),
            "skill_id": current.get("next_skill_id"),
            "done": True,
            "provenance": dict(provenance),
        }
        paired.extend([current, successor])
    return paired


SAFE_STAGE4_BENCHMARKS = (
    "toolbench_g3",
    "traject_bench",
    "alfworld",
    "webshop",
)


def _safe_stage4_causal_rows(
    trajectories_per_benchmark: int = 100,
) -> list[dict]:
    source_rows = []
    for benchmark in SAFE_STAGE4_BENCHMARKS:
        for trajectory_index in range(trajectories_per_benchmark):
            source_rows.append(
                {
                    "benchmark": benchmark,
                    "task_id": f"{benchmark}-{trajectory_index:03d}",
                    "trajectory_id": f"{benchmark}-trajectory-{trajectory_index:03d}",
                    "state_text": f"{benchmark} current {trajectory_index}",
                    "action_text": "call current",
                    "next_observation_text": (
                        f"{benchmark} next observation {trajectory_index}"
                    ),
                    "skill_id": "skill/current",
                    "next_skill_id": "skill/next",
                    "candidate_next_skill_ids": [
                        "skill/current",
                        "skill/next",
                        "skill/other",
                    ],
                    "provenance": {
                        "source_id": f"{benchmark}-safe-stage4-test",
                        "split": "train",
                    },
                }
            )
    return _stage4_strict_causal_pairs(source_rows)


class _Stage4SkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))
        self.W = torch.nn.Linear(3, 3, bias=False)
        self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
        self.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
        self.logit_scale_belief = torch.nn.Parameter(torch.ones(1) * 2.0)
        self.skill_bias_belief = torch.nn.Parameter(torch.ones(3) * 3.0)
        with torch.no_grad():
            self.W.weight.copy_(torch.eye(3))

    def retrieval_logits(self, h):
        return h @ self.E.t() + self.skill_bias_retr.unsqueeze(0)

    def belief_logits(self, h):
        return h @ self.E.t() + self.skill_bias_belief.unsqueeze(0)


class _Stage4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _Stage4SkillTable()
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        self.transition = torch.nn.Linear(6, 3, bias=False)
        self.trans_head = _Stage4TransHead()
        with torch.no_grad():
            self.action_proj.weight.copy_(torch.eye(3))
            self.transition.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    ]
                )
            )

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            vec = torch.zeros(3, dtype=torch.float32, device=self.device)
            if "current" in lowered:
                vec[0] = 1.0
            elif "obs" in lowered:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            rows.append(vec + self.anchor * 0.0)
        return torch.stack(rows)

    def action_embeddings(self, labels):
        labels = labels.to(dtype=torch.long, device=self.skill_table.E.device)
        return self.action_proj(self.skill_table.E.index_select(0, labels.reshape(-1))).view(*labels.shape, -1)

    def initial_belief(self, h_t, top_k=None):
        del top_k
        logits = self.skill_table.belief_logits(h_t)
        return torch.softmax(logits, dim=-1) @ self.skill_table.E.to(device=h_t.device, dtype=h_t.dtype)

    def transition_logits(self, m_hat, candidate_rows):
        idx = torch.tensor(candidate_rows, dtype=torch.long, device=self.device)
        cand = self.action_embeddings(idx)
        m_exp = m_hat.unsqueeze(1).expand(-1, cand.size(1), -1)
        return self.trans_head(torch.cat([m_exp, cand], dim=-1)).squeeze(-1)


class _TrainableRecordingStage4Transition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, m_t, action_input, observation_embedding):
        self.calls.append(
            (
                m_t.detach().clone(),
                action_input.detach().clone(),
                observation_embedding.detach().clone(),
            )
        )
        return m_t + self.scale * (action_input + observation_embedding)


class _TrainableRecordingStage4Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, predicted, observation_memory, observation_embedding):
        self.calls.append(
            (
                predicted.detach().clone(),
                observation_memory.detach().clone(),
                observation_embedding.detach().clone(),
            )
        )
        return torch.sigmoid(self.logit).expand_as(predicted)


class _ZeroStage4Gate(torch.nn.Module):
    def forward(self, predicted, observation_memory, observation_embedding):
        del observation_memory, observation_embedding
        return torch.zeros_like(predicted)


class _Stage4UnifiedMemoryModel(_Stage4Model):
    def __init__(self):
        super().__init__()
        self.transition = _TrainableRecordingStage4Transition()
        self.gate = _TrainableRecordingStage4Gate()
        self.initial_belief_head = torch.nn.Linear(3, 3, bias=False)
        self.unified_retriever = torch.nn.Linear(3, 3, bias=False)
        self.route_memory_utility_gate = RouteMemoryUtilityGate(3, initial_alpha=0.01)
        self.unified_route_calls = []
        self.unified_route_outputs = []
        with torch.no_grad():
            self.initial_belief_head.weight.copy_(torch.eye(3))
            self.unified_retriever.weight.copy_(torch.eye(3))

    def initial_belief(self, h_t, top_k=None):
        del top_k
        return self.initial_belief_head(h_t)

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        routed = self.unified_retriever(h_t + m_t)
        full_logits = routed @ self.skill_table.E.t()
        full_logits = full_logits + torch.tensor([[0.0, -1.0, 8.0]], dtype=h_t.dtype, device=h_t.device)
        if candidate_rows is None:
            self.unified_route_outputs.append(full_logits.detach().clone())
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        self.unified_route_outputs.append(output.detach().clone())
        return output

    def unified_route_full_logits(self, h_t, m_t):
        return self.unified_route_logits(h_t, m_t)

    def route_memory_alpha(self, h_t, static_memory, dynamic_memory, causal_update_count):
        raw = self.route_memory_utility_gate(h_t, static_memory, dynamic_memory)
        counts = causal_update_count.to(device=raw.device, dtype=raw.dtype)
        return torch.where(counts > 0, raw, torch.zeros_like(raw))


class _RoleAwareStage4UnifiedMemoryModel(_Stage4UnifiedMemoryModel):
    def __init__(self):
        super().__init__()
        self.encoded_states = []
        self.encoded_transition_texts = []

    def encode_states(self, texts):
        self.encoded_states.extend(str(text) for text in texts)
        raw = super().encode_observations(texts)
        offset = torch.tensor([4.0, 0.0, 0.0], dtype=raw.dtype, device=raw.device)
        return raw + offset.unsqueeze(0)

    def encode_observations(self, texts):
        self.encoded_transition_texts.extend(str(text) for text in texts)
        return super().encode_observations(texts)


class _CounterfactualStage4UnifiedMemoryModel(_Stage4UnifiedMemoryModel):
    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "action alpha" in lowered:
                row = [0.8, 0.1, 0.0]
            elif "action beta" in lowered:
                row = [0.1, 0.0, 0.9]
            elif "observation alpha" in lowered:
                row = [0.0, 0.7, 0.2]
            elif "observation beta" in lowered:
                row = [0.2, 0.1, 0.8]
            elif "future alpha" in lowered:
                row = [0.2, 0.8, 0.1]
            elif "future beta" in lowered:
                row = [0.7, 0.1, 0.4]
            else:
                row = [1.0, 0.0, 0.0]
            rows.append(torch.tensor(row, dtype=torch.float32, device=self.device))
        return torch.stack(rows) + self.anchor * 0.0


class _ReturnObservationTransition(torch.nn.Module):
    def forward(self, m_obs, action_input, obs_emb):
        del m_obs, action_input
        return obs_emb


class _Stage4MemorySensitiveUnifiedModel(_Stage4Model):
    def __init__(self):
        super().__init__()
        self.transition = _ReturnObservationTransition()
        self.gate = _ZeroStage4Gate()
        self.unified_route_calls = []
        self.unified_route_outputs = []

    def initial_belief(self, h_t, top_k=None):
        del top_k
        return h_t

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = m_t @ self.skill_table.E.to(device=m_t.device, dtype=m_t.dtype).t()
        if candidate_rows is None:
            self.unified_route_outputs.append(full_logits.detach().clone())
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        self.unified_route_outputs.append(output.detach().clone())
        return output

    def unified_route_full_logits(self, h_t, m_t):
        return self.unified_route_logits(h_t, m_t)


class _RecordingTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, m_t, a_t, o_t_emb):
        self.calls.append((m_t.detach().clone(), a_t.detach().clone(), o_t_emb.detach().clone()))
        return o_t_emb


class _Stage4TransHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, -1.0]))

    def forward(self, pred, candidate_embs):
        pred_expanded = pred.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)
        features = torch.cat([pred_expanded, candidate_embs], dim=-1)
        return torch.einsum("bcd,d->bc", features.float(), self.weight.float())


class _Stage4ObservationConditionModel(_Stage4Model):
    def __init__(self):
        super().__init__()
        self.transition = _RecordingTransition()


class _Stage4HandoffModel(_Stage4UnifiedMemoryModel):
    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            vec = torch.zeros(3, dtype=torch.float32, device=self.device)
            if "next-b" in lowered:
                vec[1] = 9.0
            elif "next-c" in lowered:
                vec[2] = 9.0
            elif "state-a" in lowered:
                vec[0] = 9.0
            else:
                vec[2] = 1.0
            rows.append(vec + self.anchor * 0.0)
        return torch.stack(rows)


def test_stage4_builder_uses_train_rows_and_injects_positive_next_skill(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "train-1",
                "trajectory_id": "train-traj",
                "step_index": 0,
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs after current",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current"],
                "provenance": {"source_id": "train-source", "split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "train-2",
                "trajectory_id": "train-traj",
                "step_index": 1,
                "state_text": "trusted next state",
                "skill_id": "skill/next",
                "provenance": {"source_id": "train-source", "split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "dev-1",
                "state_text": "current dev",
                "action_text": "call current",
                "next_observation_text": "obs dev",
                "skill_id": "skill/current",
                "next_skill_id": "skill/other",
                "provenance": {"split": "dev"},
            },
            {
                "benchmark": "appworld",
                "task_id": "done-1",
                "state_text": "terminal",
                "action_text": "stop",
                "skill_id": "skill/current",
                "done": True,
                "provenance": {"split": "train"},
            },
        ],
    )
    rows, report = build_stage4_next_skill_rows(
        trajectories,
        {"skill/current": 0, "skill/next": 1, "skill/other": 2},
    )

    assert report["source_rows"] == 4
    assert report["stage4_rows"] == 1
    assert report["skipped_reasons"]["split_not_train"] == 1
    assert report["skipped_reasons"]["missing_next_skill"] == 2
    assert rows[0]["positive_next_skill_idx"] == 1
    assert rows[0]["candidate_next_skill_ids"] == ["skill/current", "skill/next"]
    assert rows[0]["positive_injected"] is True
    assert rows[0]["source_benchmark"] == "traject_bench"
    assert rows[0]["next_state_text"] == "trusted next state"


def test_stage4_builder_attaches_next_state_before_max_rows_cap(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    current = _stage4_causal_source_row()
    current.pop("next_state_text")
    successor = {
        "benchmark": "toolbench_g3",
        "task_id": "causal-1",
        "trajectory_id": "causal-traj",
        "step_index": 1,
        "state_text": "trusted successor state",
        "skill_id": "skill/next",
        "provenance": {"source_id": "causal-source", "split": "train"},
    }
    _write_jsonl(trajectories, [current, successor])

    rows, report = build_stage4_next_skill_rows(
        trajectories,
        {"skill/current": 0, "skill/next": 1},
        max_rows=1,
    )

    assert len(rows) == 1
    assert rows[0].get("next_state_text") == "trusted successor state"
    assert report["causal_next_state"]["attached_rows"] == 1


@pytest.mark.parametrize(
    ("missing_key", "fallback", "reason"),
    [
        ("action_text", {"expert_action": "must not fallback"}, "missing_action_text"),
        ("next_observation_text", {"observation_text": "must not fallback"}, "missing_next_observation_text"),
        ("next_state_text", {}, "missing_next_state_text"),
        ("skill_id", {}, "missing_current_skill"),
        ("next_skill_id", {}, "missing_next_skill"),
    ],
)
def test_stage4_builder_skips_rows_missing_required_causal_fields(missing_key, fallback, reason):
    source_row = _stage4_causal_source_row()
    source_row.pop(missing_key)
    source_row.update(fallback)

    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [source_row],
        {"skill/current": 0, "skill/next": 1},
    )

    assert rows == []
    assert report["skipped_reasons"][reason] == 1


def test_stage4_builder_copies_trusted_next_state_without_fallback():
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [_stage4_causal_source_row()],
        {"skill/current": 0, "skill/next": 1},
    )

    assert report["stage4_rows"] == 1
    assert rows[0].get("next_state_text") == "trusted successor state"


def test_stage4_builder_current_state_eval_allows_missing_next_state_without_fallback():
    source_row = _stage4_causal_source_row()
    source_row.pop("next_state_text")

    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [source_row],
        {"skill/current": 0, "skill/next": 1},
        require_next_state_text=False,
    )

    assert len(rows) == 1
    assert rows[0]["next_state_text"] == ""
    assert report["require_next_state_text"] is False


def test_stage4_builder_filters_allowed_benchmarks_and_reports_skips(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    source_rows = []
    source_rows.extend(_stage4_raw_causal_pair("traject_bench", "traject-traj", "skill/next"))
    source_rows.extend(_stage4_raw_causal_pair("alfworld", "alf-traj", "skill/other"))
    source_rows.extend(_stage4_raw_causal_pair("toolbench_g3", "tool-traj", "skill/other"))
    _write_jsonl(trajectories, source_rows)

    rows, report = build_stage4_next_skill_rows(
        trajectories,
        {"skill/current": 0, "skill/next": 1, "skill/other": 2},
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
    )

    assert [row["source_benchmark"] for row in rows] == ["traject_bench", "toolbench_g3"]
    assert report["allowed_benchmarks"] == ["toolbench_g3", "traject_bench"]
    assert report["benchmark_filter_enabled"] is True
    assert report["benchmark_counts"] == {"toolbench_g3": 1, "traject_bench": 1}
    assert report["skipped_reasons"]["benchmark_not_allowed"] == 2
    assert report["skipped_benchmarks"]["alfworld"] == 2


def test_stage4_builder_reports_positive_missing_skip_distribution_by_benchmark():
    skill_id_to_idx = {
        "skill/current": 0,
        "skill/next": 1,
        "skill/other": 2,
    }
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-miss",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs",
                "next_state_text": "next state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_ids": ["skill/current", "skill/other"],
            },
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-hit",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs",
                "next_state_text": "next state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_ids": ["skill/current", "skill/next", "skill/other"],
            },
        ],
        skill_id_to_idx,
        candidate_count=3,
        stage0_candidate_handoff_report={"enabled": True},
    )

    assert len(rows) == 1
    assert report["stage4_rows"] == 1
    assert report["stage4_retained_fraction"] == 0.5
    assert report["positive_missing_skip_rows"] == 1
    assert report["positive_missing_skip_by_benchmark"] == {"toolbench_g3": 1}
    assert report["skip_distribution_by_benchmark"] == {
        "toolbench_g3": {
            "source_rows": 2,
            "stage4_rows": 1,
            "skipped_rows": 1,
            "positive_missing_skip_rows": 1,
            "retained_fraction": 0.5,
        }
    }


@pytest.mark.parametrize(
    ("natural_ids", "natural_indices", "natural_scores"),
    [(["skill/a"], [0], [4.0]), ([], [], [])],
)
def test_stage4_builder_retains_static_miss_for_full_pool_training(
    natural_ids,
    natural_indices,
    natural_scores,
):
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "static-miss",
                "state_text": "state",
                "action_text": "act",
                "next_observation_text": "next observation",
                "next_state_text": "next state",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "stage0_next_candidate_skill_ids": natural_ids,
                "stage0_next_candidate_skill_indices": natural_indices,
                "stage0_next_candidate_skill_scores": natural_scores,
                "stage0_current_positive_hit": True,
                "stage0_next_positive_hit": False,
                "provenance": {"split": "train"},
            }
        ],
        {"skill/a": 0, "skill/b": 1, "skill/c": 2},
        stage0_candidate_handoff_report={"enabled": True},
        next_skill_pool_mode="full_pool",
    )

    assert len(rows) == 1
    assert rows[0]["candidate_next_skill_ids"] == natural_ids
    assert rows[0]["candidate_next_skill_indices"] == natural_indices
    assert rows[0]["candidate_next_prior_scores"] == natural_scores
    assert rows[0]["positive_next_skill_position"] is None
    assert rows[0]["stage0_next_positive_hit"] is False
    assert rows[0]["positive_injected"] is False
    assert report["stage0_next_static_miss_retained_rows"] == 1
    assert report["positive_missing_skip_rows"] == 0
    assert report["positive_injected_rows"] == 0


def test_stage4_builder_applies_benchmark_caps_before_max_rows(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    source_rows = []
    for benchmark in ["alfworld", "webshop", "traject_bench", "toolbench_g3"]:
        for idx in range(3):
            source_rows.extend(
                _stage4_raw_causal_pair(benchmark, f"{benchmark}-traj-{idx}", "skill/next")
            )
    _write_jsonl(trajectories, source_rows)

    rows, report = build_stage4_next_skill_rows(
        trajectories,
        {"skill/current": 0, "skill/next": 1, "skill/other": 2},
        benchmark_caps={"alfworld": 1, "webshop": 2, "traject_bench": 1, "toolbench_g3": 2},
        max_rows=4,
    )

    assert report["benchmark_caps"]["enabled"] is True
    assert report["benchmark_caps"]["retained_by_benchmark"] == {
        "alfworld": 1,
        "toolbench_g3": 2,
        "traject_bench": 1,
        "webshop": 2,
    }
    assert report["stage4_rows"] == 4
    assert {row["source_benchmark"] for row in rows} == {"alfworld", "webshop", "traject_bench"}


def test_stage4_negative_fill_does_not_iterate_entire_large_skill_pool():
    class _NoIterSkillIds:
        def __init__(self, size: int):
            self.size = size
            self.accessed = []

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            if idx >= self.size:
                raise IndexError(idx)
            self.accessed.append(idx)
            return f"skill/{idx}"

        def __iter__(self):
            raise AssertionError("Stage 4 candidate fill should not iterate the whole skill pool")

    skill_ids = _NoIterSkillIds(100_000)

    filled = _stable_negative_fill(
        seed_text="task-a",
        existing=["skill/1"],
        skill_ids=skill_ids,
        candidate_count=8,
    )

    assert filled[0] == "skill/1"
    assert len(filled) == 8
    assert len(set(filled)) == 8
    assert len(skill_ids.accessed) < 100


def test_stage4_act_loss_uses_transition_logits_and_reports_ranking():
    model = _Stage4Model()
    rows = [
            {
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs after action",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1, 2],
            }
    ]

    loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert loss.item() > 0.0
    assert metrics["stage4_act_count"] == 1.0
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert metrics["stage4_candidate_count"] == 3.0


def test_stage4_act_loss_reconstructs_recurrent_belief_from_replay_prefix():
    model = _Stage4ObservationConditionModel()
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 1,
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_id": "skill/current",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
            "replay_prefix": [
                {
                    "observation_text": "initial state",
                    "action_text": "call prefix",
                    "next_observation_text": "prefix obs",
                    "skill_id": "skill/current",
                    "skill_idx": 0,
                }
            ],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert metrics["stage4_replay_prefix_used_count"] == 1.0
    assert len(model.transition.calls) == 3
    current_prior_memory = model.transition.calls[1][0]
    assert torch.equal(current_prior_memory, torch.tensor([[0.0, 1.0, 0.0]]))


def test_stage4_act_loss_can_anchor_to_stage0_rank_prior():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, 1.0, 6.0],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
    )

    assert metrics["transition_scoring_mode"] == "stage0_rank_prior_plus_transition_residual"
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert metrics["stage4_prior_next_skill_recall@1"] == 1.0


def test_stage4_builder_preserves_stage0_prior_scores_after_candidate_filter():
    skill_id_to_idx = {
        "skill/current": 0,
        "skill/next": 1,
        "skill/other": 2,
    }

    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "score-prior",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs",
                "next_state_text": "next state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_ids": ["skill/current", "skill/next", "skill/other"],
                "stage0_next_candidate_skill_indices": [0, 1, 2],
                "stage0_next_candidate_skill_scores": [2.5, 9.0, -1.0],
            }
        ],
        skill_id_to_idx,
        candidate_count=2,
        stage0_candidate_handoff_report={"enabled": True},
    )

    assert report["stage4_rows"] == 1
    assert rows[0]["candidate_next_skill_ids"] == ["skill/current", "skill/next"]
    assert rows[0]["candidate_next_prior_scores"] == [2.5, 9.0]


def test_stage4_builder_does_not_rebuild_skill_index_mapping_per_row():
    class _CountingSkillMap(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.items_calls = 0

        def items(self):
            self.items_calls += 1
            return super().items()

    skill_id_to_idx = _CountingSkillMap(
        {
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        }
    )
    source_rows = []
    for idx in range(4):
        source_rows.append(
            {
                "benchmark": "toolbench_g3",
                "task_id": f"row-{idx}",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs",
                "next_state_text": "next state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_ids": ["skill/current", "skill/next", "skill/other"],
            }
        )

    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        source_rows,
        skill_id_to_idx,
        candidate_count=2,
        stage0_candidate_handoff_report={"enabled": True},
    )

    assert report["stage4_rows"] == 4
    assert len(rows) == 4
    assert skill_id_to_idx.items_calls == 1


def test_stage4_act_loss_can_add_online_memory_residual():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, -1.0, -5.0],
            "candidate_next_online_memory_scores": [0.0, 0.0, 6.0],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )

    assert metrics["stage4_prior_next_skill_recall@1"] == 0.0
    assert metrics["stage4_online_memory_next_skill_recall@1"] == 1.0
    assert metrics["stage4_next_skill_recall@1"] == 1.0


def test_stage4_unified_memory_routes_from_post_action_memory_and_next_state():
    model = _Stage4UnifiedMemoryModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/other",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [2, 1, 0],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.05,
    )

    assert len(model.transition.calls) == 1
    assert len(model.gate.calls) == 1
    assert len(model.unified_route_calls) == 2
    dynamic_h, dynamic_m, dynamic_candidates = model.unified_route_calls[0]
    static_h, static_m, static_candidates = model.unified_route_calls[1]
    assert torch.equal(dynamic_h, model.encode_observations(["future state"]))
    assert torch.equal(static_h, dynamic_h)
    assert dynamic_candidates is None
    assert static_candidates is None
    assert not torch.equal(dynamic_m, static_m)
    _m_t, action_input, observation_input = model.transition.calls[0]
    assert torch.equal(action_input, model.action_proj(model.encode_observations(["call current"])))
    assert torch.equal(observation_input, model.encode_observations(["obs after action"]))
    assert metrics["route_scorer"] == "unified_memory"
    assert metrics["transition_scoring_mode"] == "unified_memory"
    assert metrics["transition_residual_lambda"] == 0.0
    assert metrics["transition_skill_head_type"] == "unified_memory_retriever_full_pool"
    assert metrics["next_skill_pool_mode"] == "full_pool"
    assert metrics["stage4_post_action_update_rows"] == 1.0
    assert metrics["stage4_next_state_rows"] == 1.0
    assert metrics["stage4_next_skill_recall@1"] == 1.0
    assert metrics["stage4_unified_dynamic_next_skill_recall@1"] == 1.0
    assert metrics["stage4_unified_static_next_skill_recall@1"] == 1.0
    assert not any(key.startswith("stage4_prior_") for key in metrics)
    assert not any(key.startswith("stage4_residual_") for key in metrics)
    assert not any(key.startswith("stage4_online_memory_") for key in metrics)


def test_stage4_uses_state_encoder_only_for_current_and_next_state():
    model = _RoleAwareStage4UnifiedMemoryModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/other",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [2, 1, 0],
        }
    ]

    _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.05,
    )

    assert model.encoded_states == ["current state", "future state"]
    assert set(model.encoded_transition_texts) == {"call current", "obs after action"}
    dynamic_h, _dynamic_m, _dynamic_candidates = model.unified_route_calls[0]
    assert torch.equal(dynamic_h, torch.tensor([[4.0, 0.0, 1.0]]))


def test_stage4_unified_memory_reports_dynamic_static_rank_flips_and_counterfactual_utility():
    model = _Stage4MemorySensitiveUnifiedModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
            "replay_prefix": [
                {
                    "observation_text": "current state",
                    "action_text": "call current",
                    "next_observation_text": "obs after action",
                    "skill_idx": 0,
                }
            ],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        trainable_replay_prefix=True,
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.5,
        counterfactual_gain_margin=1.5,
        counterfactual_safety_tolerance=0.01,
        counterfactual_gain_weight=1.0,
        counterfactual_safety_weight=1.0,
        counterfactual_scale=1.0,
    )

    assert metrics["stage4_unified_static_next_skill_recall@1"] == 0.0
    assert metrics["stage4_unified_dynamic_next_skill_recall@1"] == 1.0
    assert metrics["stage4_unified_dynamic_vs_static_delta_mrr"] > 0.0
    assert metrics["stage4_unified_positive_rank_improved_rows"] == 1.0
    assert metrics["stage4_unified_positive_rank_worsened_rows"] == 0.0
    assert metrics["stage4_unified_argmax_changed_rows"] == 1.0
    assert metrics["stage4_counterfactual_eligible_rows"] == 1.0
    assert metrics["stage4_counterfactual_utility_loss"] > 0.0
    assert metrics["stage4_weighted_counterfactual_utility_loss"] > 0.0
    expected = full_pool_causal_route_objective(
        dynamic_logits=model.unified_route_outputs[-2],
        static_logits=model.unified_route_outputs[-1],
        positive_mask=torch.tensor([[False, True, False]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
        row_weights=torch.ones(1),
        gain_margin=1.5,
        safety_tolerance=0.01,
        gain_weight=1.0,
        safety_weight=1.0,
    )
    assert metrics["stage4_act_loss"] == pytest.approx(float(expected.main_loss.item()))
    assert metrics["stage4_counterfactual_utility_loss"] == pytest.approx(
        float(expected.counterfactual.loss.item())
    )


def _stage4_unified_counterfactual_outputs(
    model: _CounterfactualStage4UnifiedMemoryModel | None = None,
    **row_overrides,
):
    model = model or _CounterfactualStage4UnifiedMemoryModel()
    row = {
        "state_text": "current state",
        "action_text": "action alpha",
        "next_observation_text": "observation alpha",
        "next_state_text": "future alpha",
        "skill_id": "skill/current",
        "next_skill_id": "skill/next",
        "skill_idx": 0,
        "positive_next_skill_idx": 1,
        "candidate_next_skill_indices": [0, 1, 2],
    }
    row.update(row_overrides)
    _compute_stage4_act_loss(
        model,
        [row],
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.0,
    )
    return {
        "dynamic_h": model.unified_route_calls[-2][0],
        "dynamic_m": model.unified_route_calls[-2][1],
        "static_h": model.unified_route_calls[-1][0],
        "static_m": model.unified_route_calls[-1][1],
        "dynamic_logits": model.unified_route_outputs[-2],
        "static_logits": model.unified_route_outputs[-1],
    }


def test_stage4_unified_memory_action_counterfactual_changes_dynamic_not_static_logits():
    baseline = _stage4_unified_counterfactual_outputs(action_text="action alpha")
    changed = _stage4_unified_counterfactual_outputs(action_text="action beta")

    assert not torch.equal(baseline["dynamic_m"], changed["dynamic_m"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert torch.equal(baseline["static_h"], changed["static_h"])
    assert torch.equal(baseline["static_m"], changed["static_m"])
    assert torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage4_unified_memory_observation_counterfactual_changes_dynamic_not_static_logits():
    baseline = _stage4_unified_counterfactual_outputs(next_observation_text="observation alpha")
    changed = _stage4_unified_counterfactual_outputs(next_observation_text="observation beta")

    assert not torch.equal(baseline["dynamic_m"], changed["dynamic_m"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert torch.equal(baseline["static_h"], changed["static_h"])
    assert torch.equal(baseline["static_m"], changed["static_m"])
    assert torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage4_unified_memory_next_state_counterfactual_changes_both_route_branches():
    baseline = _stage4_unified_counterfactual_outputs(next_state_text="future alpha")
    changed = _stage4_unified_counterfactual_outputs(next_state_text="future beta")

    assert not torch.equal(baseline["dynamic_h"], changed["dynamic_h"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert not torch.equal(baseline["static_h"], changed["static_h"])
    assert not torch.equal(baseline["static_m"], changed["static_m"])
    assert not torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage4_safe_memory_objective_backpropagates_through_causal_modules_and_utility_gate():
    model = _CounterfactualStage4UnifiedMemoryModel()
    _freeze_for_stage4_act(model, route_scorer="unified_memory")
    loss, _metrics = _compute_stage4_act_loss(
        model,
        [
            {
                "state_text": "current state",
                "action_text": "action alpha",
                "next_observation_text": "observation alpha",
                "next_state_text": "future alpha",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1, 2],
            }
        ],
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.05,
    )
    loss.backward()

    for module in (
        model.transition,
        model.gate,
        model.action_proj,
        model.route_memory_utility_gate,
    ):
        grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert grads
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0
    for module in (model.initial_belief_head, model.unified_retriever):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in model.trans_head.parameters())


def test_stage4_safe_optimizer_step_preserves_static_logits() -> None:
    model = _CounterfactualStage4UnifiedMemoryModel()
    _freeze_for_stage4_act(model, route_scorer="unified_memory")
    rows = [
        {
            "state_text": "current state",
            "action_text": "action alpha",
            "next_observation_text": "observation alpha",
            "next_state_text": "future alpha",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]
    before = _stage4_unified_counterfactual_outputs(model=model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss, _metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        },
        counterfactual_utility_weight=0.05,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = _stage4_unified_counterfactual_outputs(model=model)

    assert torch.equal(before["static_logits"], after["static_logits"])


def test_stage4_safe_route_batch_exposes_teacher_and_dynamic_endpoints() -> None:
    model = _CounterfactualStage4UnifiedMemoryModel()
    model.route_logits_from_candidates = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("safe Stage4 must use full memory-conditioned logits"))
    _freeze_for_stage4_act(model, route_scorer="unified_memory")
    batch = _build_stage4_safe_route_batch(
        model,
        [
            {
                "state_text": "current state",
                "action_text": "action alpha",
                "next_observation_text": "observation alpha",
                "next_state_text": "future alpha",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1, 2],
            }
        ],
        torch.device("cpu"),
        skill_id_to_idx={
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        },
        equivalent_skill_ids_by_skill_id=None,
        trainable_replay_prefix=True,
        counterfactual_utility_weight=0.05,
        counterfactual_gain_margin=0.10,
        counterfactual_safety_tolerance=0.01,
        counterfactual_gain_weight=1.0,
        counterfactual_safety_weight=1.0,
        counterfactual_scale=1.0,
    )

    assert batch.dynamic_full_logits.shape == batch.static_full_logits.shape == (
        1,
        3,
    )
    assert batch.dynamic_full_logits.requires_grad
    assert not batch.static_full_logits.requires_grad
    assert batch.causal_update_count.tolist() == [1.0]


def test_stage4_cmc_route_batch_trains_only_residual_and_candidate_gate() -> None:
    model = _CounterfactualStage4UnifiedMemoryModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_utility_gate = MemoryUtilityGate()
    audit = _freeze_for_stage4_act(
        model,
        route_scorer="unified_memory",
        stage4_method=COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
    )

    batch = _build_stage4_cmc_route_batch(
        model,
        [
            {
                "trajectory_id": "cmc-traj",
                "state_text": "current state",
                "action_text": "action alpha",
                "next_observation_text": "observation alpha",
                "next_state_text": "future alpha",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "skill_idx": 0,
                "positive_next_skill_idx": 1,
                "candidate_next_skill_indices": [0, 1, 2],
            }
        ],
        torch.device("cpu"),
        skill_id_to_idx={
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        },
        equivalent_skill_ids_by_skill_id=None,
    )
    batch.total_loss.backward()

    assert audit["trainable_modules"] == [
        "route_memory_residual_adapter",
        "route_memory_candidate_utility_gate",
    ]
    assert batch.dynamic_full_logits.requires_grad
    assert not batch.static_full_logits.requires_grad
    assert batch.metrics["stage4_method"] == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
    assert batch.metrics["stage4_cmc_eligible_rows"] == 1.0
    assert any(
        parameter.grad is not None
        for parameter in model.route_memory_residual_adapter.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.route_memory_candidate_utility_gate.parameters()
    )
    for module in (
        model.transition,
        model.gate,
        model.action_proj,
        model.initial_belief_head,
        model.unified_retriever,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_stage4_candidate_admission_builds_natural_union_and_trains_only_new_head() -> None:
    model = _Stage4MemorySensitiveUnifiedModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_admission_residual = (
        CandidateAdmissionResidualHead(3)
    )
    rows = [
        {
            "trajectory_id": f"admission-traj-{row_idx}",
            "state_text": "current state",
            "action_text": "action alpha",
            "next_observation_text": "obs after action",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": next_skill_id,
            "skill_idx": 0,
            "positive_next_skill_idx": positive_idx,
            "candidate_next_skill_indices": [0, 1, 2],
        }
        for row_idx, (next_skill_id, positive_idx) in enumerate(
            (("skill/next", 1), ("skill/other", 2))
        )
    ]
    audit = _freeze_for_stage4_act(
        model,
        route_scorer="unified_memory",
        stage4_method=CANDIDATE_ADMISSION_RESIDUAL_V1,
    )

    batch = _build_stage4_candidate_admission_route_batch(
        model,
        rows,
        torch.device("cpu"),
        skill_id_to_idx={
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        },
        equivalent_skill_ids_by_skill_id=None,
        static_k=2,
        dynamic_extra_k=1,
    )
    batch.total_loss.backward()

    assert CANDIDATE_ADMISSION_RESIDUAL_V1 in stage4_module.STAGE4_METHODS
    assert audit["trainable_modules"] == [
        "route_memory_candidate_admission_residual"
    ]
    assert batch.candidate_union.candidate_rows == [[2, 0, 1], [2, 0, 1]]
    assert batch.dynamic_extra_mask.tolist() == [
        [False, False, True],
        [False, False, True],
    ]
    assert batch.candidate_positive_mask.tolist() == [
        [False, False, True],
        [True, False, False],
    ]
    assert batch.metrics["stage4_method"] == CANDIDATE_ADMISSION_RESIDUAL_V1
    assert batch.metrics["stage4_candidate_admission_positive_count"] == 1.0
    assert batch.metrics["stage4_candidate_admission_negative_count"] == 1.0
    assert batch.metrics["stage4_candidate_admission_dynamic_rescue_rows"] == 1.0
    assert any(
        parameter.grad is not None
        for parameter in model.route_memory_candidate_admission_residual.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.route_memory_residual_adapter.parameters()
    )


def test_stage4_candidate_admission_dispatches_through_full_pool_act_loss() -> None:
    model = _Stage4MemorySensitiveUnifiedModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_admission_residual = (
        CandidateAdmissionResidualHead(3)
    )
    rows = [
        {
            "trajectory_id": "admission-dispatch",
            "state_text": "current state",
            "action_text": "action alpha",
            "next_observation_text": "obs after action",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]
    _freeze_for_stage4_act(
        model,
        route_scorer="unified_memory",
        stage4_method=CANDIDATE_ADMISSION_RESIDUAL_V1,
    )

    loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={
            "skill/current": 0,
            "skill/next": 1,
            "skill/other": 2,
        },
        stage4_method=CANDIDATE_ADMISSION_RESIDUAL_V1,
        candidate_admission_static_k=2,
        candidate_admission_dynamic_extra_k=1,
    )

    assert loss.requires_grad
    assert metrics["stage4_method"] == CANDIDATE_ADMISSION_RESIDUAL_V1
    assert metrics["stage4_candidate_count"] == 3.0


def test_candidate_admission_training_requires_base_cmc_identity() -> None:
    parameters = inspect.signature(train_stage4_act_with_model).parameters

    assert parameters["base_cmc_checkpoint_sha256"].default is None
    script = Path("scripts/run_clstr_stage4_act_train.py").read_text(
        encoding="utf-8"
    )
    launcher = Path(
        "scripts/sbatch/run_clstr_unified_stage4_act_train.sh"
    ).read_text(encoding="utf-8")
    assert "--base_cmc_checkpoint_path" in script
    assert "validate_stage4_cmc_parent_router" in script
    assert "base_cmc_checkpoint_sha256" in script
    assert "BASE_CMC_CHECKPOINT_PATH" in launcher
    assert "--base_cmc_checkpoint_path" in launcher


def test_candidate_admission_training_support_fails_closed_without_extra_positive() -> None:
    with pytest.raises(ValueError, match="no dynamic-extra positive"):
        _validate_candidate_admission_training_support(
            [{"stage4_candidate_admission_positive_count": 0.0}]
        )

    report = _validate_candidate_admission_training_support(
        [
            {"stage4_candidate_admission_positive_count": 0.0},
            {"stage4_candidate_admission_positive_count": 2.0},
        ]
    )
    assert report["dynamic_extra_positive_count"] == 2.0


def test_stage4_unified_memory_freeze_trains_causal_modules_by_default():
    model = _Stage4UnifiedMemoryModel()
    model.skill_head = torch.nn.Linear(3, 3)

    audit = _freeze_for_stage4_act(model, route_scorer="unified_memory")

    assert audit["trainable_modules"] == [
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ]
    for module_name in (
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ):
        module = getattr(model, module_name)
        assert any(param.requires_grad for param in module.parameters())
    assert "trans_head" not in audit["trainable_modules"]
    assert "skill_head" not in audit["trainable_modules"]
    assert all(
        not parameter.requires_grad
        for parameter in model.initial_belief_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.unified_retriever.parameters()
    )
    assert all(not param.requires_grad for param in model.trans_head.parameters())
    assert all(not param.requires_grad for param in model.skill_head.parameters())
    assert audit["frozen_belief_gate"] is False
    assert audit["train_transition"] is True


def test_stage4_score_calibrator_zero_init_preserves_existing_logits():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, -1.0, -5.0],
            "candidate_next_online_memory_scores": [0.0, 0.0, 6.0],
        }
    ]

    baseline_loss, baseline_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )
    _ensure_stage4_score_calibrator(model)
    calibrated_loss, calibrated_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )

    assert calibrated_loss.item() == pytest.approx(baseline_loss.item())
    assert calibrated_metrics["stage4_next_skill_recall@1"] == baseline_metrics["stage4_next_skill_recall@1"]
    assert calibrated_metrics["stage4_score_calibrator_enabled"] is True
    assert calibrated_metrics["stage4_score_calibrator_feature_count"] == 6


def test_stage4_score_calibrator_receives_gradients_and_can_improve_toy_rank():
    model = _Stage4Model()
    calibrator = _ensure_stage4_score_calibrator(model)
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, -1.0, -5.0],
            "candidate_next_online_memory_scores": [0.0, 0.0, 1.0],
        }
    ]
    optimizer = torch.optim.SGD(calibrator.parameters(), lr=0.5)

    before_loss, before_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )
    optimizer.zero_grad(set_to_none=True)
    before_loss.backward()
    assert calibrator.weight.grad is not None
    assert calibrator.weight.grad.abs().sum().item() > 0.0
    optimizer.step()
    after_loss, after_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )

    assert before_metrics["stage4_next_skill_recall@1"] == 0.0
    assert after_loss.item() < before_loss.item()
    assert after_metrics["stage4_score_calibrator_enabled"] is True


def test_stage4_score_calibrator_is_rowwise_noop_without_online_memory_evidence():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 2,
            "candidate_next_skill_indices": [0, 1, 2],
            "candidate_next_prior_scores": [0.0, -1.0, -5.0],
            "candidate_next_online_memory_scores": [0.0, 0.0, 0.0],
        }
    ]
    baseline_loss, baseline_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )
    calibrator = _ensure_stage4_score_calibrator(model)
    with torch.no_grad():
        calibrator.weight.copy_(torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]))
    calibrated_loss, calibrated_metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        online_memory_weight=1.0,
    )

    assert calibrated_loss.item() == pytest.approx(baseline_loss.item())
    assert calibrated_metrics["stage4_next_skill_mrr"] == baseline_metrics["stage4_next_skill_mrr"]
    assert calibrated_metrics["stage4_score_calibrator_active_row_fraction"] == 0.0


def test_stage4_act_loss_rejects_candidate_rows_missing_positive():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 2],
        }
    ]

    with pytest.raises(ValueError, match="missing positive next skill"):
        _compute_stage4_act_loss(model, rows, torch.device("cpu"))


def test_stage4_act_loss_rejects_stale_positive_position():
    model = _Stage4Model()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call current",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "positive_next_skill_position": 0,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]

    with pytest.raises(ValueError, match="stale positive next skill position"):
        _compute_stage4_act_loss(model, rows, torch.device("cpu"))


def test_stage4_act_loss_conditions_transition_on_next_observation_embedding():
    model = _Stage4ObservationConditionModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call tool action",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]

    _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert len(model.transition.calls) == 2
    prior_action_arg = model.transition.calls[0][1]
    prior_obs_arg = model.transition.calls[0][2]
    residual_action_arg = model.transition.calls[1][1]
    residual_obs_arg = model.transition.calls[1][2]
    assert torch.equal(prior_action_arg, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(prior_obs_arg, torch.zeros(1, 3))
    assert torch.equal(residual_action_arg, torch.tensor([[0.0, 0.0, 1.0]]))
    assert not torch.equal(residual_action_arg, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(residual_obs_arg, torch.tensor([[0.0, 1.0, 0.0]]))
    assert not torch.equal(residual_obs_arg, torch.tensor([[1.0, 0.0, 0.0]]))


def test_stage4_act_loss_can_use_v4_1b_action_observation_scoring():
    model = _Stage4ObservationConditionModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "call tool action",
            "next_observation_text": "obs after action",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]

    _loss, metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        transition_residual_lambda=0.0,
        transition_scoring_mode="v4_1b_action_observation",
    )

    assert len(model.transition.calls) == 1
    legacy_action_arg = model.transition.calls[0][1]
    legacy_obs_arg = model.transition.calls[0][2]
    assert torch.equal(legacy_action_arg, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(legacy_obs_arg, torch.tensor([[0.0, 0.0, 1.0]]))
    assert metrics["transition_scoring_mode"] == "v4_1b_action_observation"
    assert metrics["transition_residual_lambda"] == 0.0


def test_stage4_mainline_exposes_no_preference_hrpo_surface():
    for symbol in (
        "_compute_stage4_preference_loss",
        "_compute_joint_stage4_act_loss",
        "build_unified_hrpo_rows",
        "make_hrpo_reference_modules",
    ):
        assert not hasattr(stage4_module, symbol)

    parameters = inspect.signature(train_stage4_act_with_model).parameters
    for parameter in (
        "preference_max_rows",
        "preference_batch_size",
        "lambda_pref",
        "beta_kl",
    ):
        assert parameter not in parameters
    assert parameters["route_scorer"].default == "unified_memory"
    assert parameters["next_skill_pool_mode"].default == "full_pool"
    assert parameters["counterfactual_utility_weight"].default == 0.05
    assert parameters["counterfactual_gain_margin"].default == 0.1
    assert parameters["counterfactual_safety_tolerance"].default == 0.01
    assert parameters["counterfactual_gain_weight"].default == 1.0
    assert parameters["counterfactual_safety_weight"].default == 1.0
    assert parameters["counterfactual_warmup_fraction"].default == 0.05
    assert parameters["safe_memory_residual_bound"].default == 2.0
    assert parameters["safe_local_candidate_sizes"].default == (2, 3, 4, 5, 8, 10)
    assert parameters["learning_rate"].default == 3.0e-5
    assert parameters["minimum_learning_rate"].default == 3.0e-6
    assert parameters["learning_rate_warmup_fraction"].default == 0.05
    assert parameters["validation_interval_steps"].default == 400
    assert parameters["validation_rows_per_benchmark"].default == 256
    assert parameters["minimum_validation_rows_per_benchmark"].default == 128
    assert parameters["gate_rows_per_benchmark"].default == 512
    assert parameters["resume_checkpoint_path"].default is None


def test_stage4_warmup_cosine_multiplier_has_declared_endpoints() -> None:
    assert _stage4_lr_multiplier(
        0,
        total_steps=100,
        warmup_fraction=0.05,
        minimum_ratio=0.1,
    ) == 0.0
    assert _stage4_lr_multiplier(
        5,
        total_steps=100,
        warmup_fraction=0.05,
        minimum_ratio=0.1,
    ) == pytest.approx(1.0)
    assert _stage4_lr_multiplier(
        100,
        total_steps=100,
        warmup_fraction=0.05,
        minimum_ratio=0.1,
    ) == pytest.approx(0.1)


def test_train_stage4_act_with_model_writes_checkpoint_and_manifest(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([
            {
                "benchmark": "traject_bench",
                "task_id": "train-1",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs after current",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/other"],
                "provenance": {"split": "train"},
            }
        ]),
    )

    report = train_stage4_act_with_model(
        model=_Stage4Model(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=2,
        batch_size=1,
        learning_rate=1.0e-3,
        route_scorer="legacy_prior_residual",
        next_skill_pool_mode="stage0_candidates",
    )

    assert report["status"] == "ok"
    assert report["training_objective"] == "legacy_transition_conditioned_next_skill_ce"
    assert report["training_regime"] == "offline_train_split_transition_conditioned_next_skill"
    assert report["on_policy_rollout_used"] is False
    for removed_key in ("lambda_pref", "beta_kl", "preference_aux_enabled", "preference_data_report"):
        assert removed_key not in report
    assert report["data_report"]["stage4_rows"] == 1
    assert report["data_report"]["benchmark_counts"] == {"traject_bench": 1}
    assert report["row_order"] == {"strategy": "seeded_shuffle", "seed": 17, "stage4_rows": 1}
    assert Path(report["checkpoint"]).is_file()
    assert Path(report["latest_checkpoint"]).is_file()
    assert Path(report["training_metrics_path"]).is_file()
    assert Path(report["loss_curve_path"]).is_file()
    assert Path(report["setup_status_path"]).is_file()
    assert (output_dir / "train_stdout.json").is_file()
    train_report_path = output_dir / "train_report.json"
    assert train_report_path.is_file()
    assert json.loads(train_report_path.read_text(encoding="utf-8"))["status"] == "ok"
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for phase in [
        "skills_loaded",
        "stage4_rows_built",
        "model_moved_to_device",
        "stage4_rows_shuffled",
        "stage4_freeze_applied",
        "training_started",
    ]:
        assert phase in setup_phases
    metric_lines = [
        json.loads(line)
        for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step"] for row in metric_lines] == [1, 2]
    latest_payload = torch.load(report["latest_checkpoint"], map_location="cpu")
    assert latest_payload["step"] == 2
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["stage"] == "clstr_stage4_transition_conditioned_next_skill"
    assert payload["train_report"]["training_objective"] == "legacy_transition_conditioned_next_skill_ce"
    assert payload["train_report"]["training_regime"] == "offline_train_split_transition_conditioned_next_skill"
    assert payload["train_report"]["on_policy_rollout_used"] is False
    for removed_key in ("lambda_pref", "beta_kl", "preference_aux_enabled", "preference_data_report"):
        assert removed_key not in payload["train_report"]
    assert payload["train_report"]["data_report"]["stage4_rows"] == 1
    assert payload["train_report"]["row_order"]["strategy"] == "seeded_shuffle"
    assert payload["checkpoint_excludes_frozen_routing_foundation"] is True
    assert sorted(key for key in payload["model_state_dict"] if key.startswith("skill_table.")) == [
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    ]


def test_train_stage4_act_with_model_threads_unified_route_scorer(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_unified"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([
            {
                "benchmark": "traject_bench",
                "task_id": "train-1",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "obs after current",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next", "skill/other"],
                "provenance": {"split": "train"},
            }
        ]),
    )

    report = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=2,
        batch_size=1,
        learning_rate=1.0e-3,
        route_scorer="unified_memory",
        validation_rows_per_benchmark=0,
        counterfactual_warmup_fraction=1.0,
    )

    assert report["status"] == "ok"
    assert report["training_objective"] == "full_pool_causal_next_skill_with_counterfactual_utility"
    assert report["training_regime"] == "offline_train_split_causal_next_skill"
    assert report["route_scorer"] == "unified_memory"
    assert report["next_skill_pool_mode"] == "full_pool"
    assert report["counterfactual_utility_weight"] == 0.05
    assert report["counterfactual_warmup_fraction"] == 1.0
    assert report["uses_stage0_prior_at_inference"] is False
    assert report["transition_scoring_mode"] == "unified_memory"
    assert report["transition_scoring_mode_requested"] == "stage0_rank_prior_plus_transition_residual"
    assert report["transition_residual_lambda"] == 0.0
    assert report["transition_residual_lambda_requested"] == 0.25
    assert report["freeze_report"]["route_scorer"] == "unified_memory"
    assert report["freeze_report"]["frozen_belief_gate"] is False
    assert report["freeze_report"]["trainable_modules"] == [
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ]
    assert not any(
        name.startswith(("initial_belief_head.", "unified_retriever."))
        for name in report["freeze_report"]["optimizer_parameter_names"]
    )
    assert report["last_metrics"]["route_scorer"] == "unified_memory"
    assert report["last_metrics"]["uses_stage0_prior_at_inference"] is False
    assert report["last_metrics"]["transition_skill_head_type"] == "unified_memory_retriever_full_pool"
    assert report["last_metrics"]["stage4_full_pool_eligible_rows"] == 1.0
    assert report["last_metrics"]["stage4_post_action_update_rows"] == 1.0
    assert report["last_metrics"]["stage4_next_state_rows"] == 1.0
    assert [row["stage4_counterfactual_scale"] for row in report["history"]] == pytest.approx([0.5, 1.0])
    assert report["counterfactual_warmup_step_source"] == "fresh_stage4_local_step"
    assert report["stage4_resume_supported"] is False
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["train_report"]["route_scorer"] == "unified_memory"
    assert payload["train_report"]["uses_stage0_prior_at_inference"] is False


def test_train_stage4_cmc_writes_parent_bound_delta_only_checkpoint(tmp_path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_cmc"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    causal_source = _stage4_causal_source_row()
    causal_source.pop("next_state_text")
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([causal_source]),
    )
    model = _Stage4UnifiedMemoryModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_utility_gate = MemoryUtilityGate()

    report = train_stage4_act_with_model(
        model=model,
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        stage4_method=COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
        validation_rows_per_benchmark=0,
        stage0_top_m=None,
        trainable_replay_prefix=False,
    )

    assert report["status"] == "ok", json.dumps(report, sort_keys=True)
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert report["stage4_method"] == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
    assert report["freeze_report"]["trainable_modules"] == [
        "route_memory_residual_adapter",
        "route_memory_candidate_utility_gate",
    ]
    assert payload["parent_stage2_full_router_digest"]
    assert payload["parent_stage2_fast_router_digest"]
    assert payload["model_state_dict"]
    assert all(
        key.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_utility_gate.",
            )
        )
        for key in payload["model_state_dict"]
    )


def test_train_stage4_safe_memory_selects_fixed_validation_checkpoint(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "safe-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_safe"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(trajectories, _safe_stage4_causal_rows())

    report = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=2,
        batch_size=4,
        seed=17,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        allowed_benchmarks=set(SAFE_STAGE4_BENCHMARKS),
        benchmark_caps={benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS},
        validation_fraction=0.10,
        validation_rows_per_benchmark=2,
        minimum_validation_rows_per_benchmark=1,
        gate_rows_per_benchmark=2,
        validation_interval_steps=1,
        stage0_top_m=3,
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=4,
    )

    assert report["status"] == "ok"
    assert report["safe_memory_protocol_version"] == "stage4_safe_memory_v1"
    assert report["valid_or_test_used_for_training"] is False
    assert report["validation_used_for_checkpoint_selection"] is True
    assert report["data_protocol"]["validation_rows_by_benchmark"] == {
        benchmark: 2 for benchmark in SAFE_STAGE4_BENCHMARKS
    }
    assert report["data_protocol"]["capped_train_rows_by_benchmark"] == {
        benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS
    }
    assert report["batch_protocol"] == {
        "batch_size": 4,
        "benchmarks": list(SAFE_STAGE4_BENCHMARKS),
        "rows_per_benchmark": 1,
        "seed": 17,
    }
    assert report["optimizer"]["peak_learning_rate"] == pytest.approx(3.0e-5)
    assert report["optimizer"]["weight_decay"] == pytest.approx(0.01)
    assert report["scheduler"]["type"] == "linear_warmup_cosine_v1"
    assert report["scheduler"]["minimum_learning_rate"] == pytest.approx(3.0e-6)
    assert report["dynamic_selection"]["selected_checkpoint_path"]
    assert Path(report["dynamic_selection_path"]).is_file()
    assert Path(report["stage4_data_protocol_path"]).is_file()
    assert [item["step"] for item in report["validation_history"]] == [0, 1, 2]

    compact_payload = torch.load(
        report["dynamic_selection"]["selected_checkpoint_path"],
        map_location="cpu",
    )
    assert compact_payload["model_state_dict"]
    assert all(
        key.startswith(
            ("transition.", "gate.", "action_proj.", "route_memory_utility_gate.")
        )
        for key in compact_payload["model_state_dict"]
    )
    assert "optimizer_state_dict" not in compact_payload
    assert "scheduler_state_dict" not in compact_payload
    resume_payload = torch.load(report["latest_checkpoint"], map_location="cpu")
    assert resume_payload["optimizer_state_dict"]
    assert resume_payload["scheduler_state_dict"]
    assert resume_payload["data_protocol_manifest_sha256"] == report[
        "data_protocol"
    ]["manifest_sha256"]

    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert setup_phases.index("stage4_data_protocol_created") < setup_phases.index(
        "stage0_candidate_handoff_prepared"
    )


def test_train_stage4_cmc_validation_matches_deployed_fusion(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "cmc-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_cmc"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(
        trajectories,
        _safe_stage4_causal_rows(),
    )
    model = _Stage4UnifiedMemoryModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_utility_gate = MemoryUtilityGate()

    report = train_stage4_act_with_model(
        model=model,
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=4,
        seed=17,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        stage4_method=COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
        allowed_benchmarks=set(SAFE_STAGE4_BENCHMARKS),
        benchmark_caps={benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS},
        validation_fraction=0.10,
        validation_rows_per_benchmark=1,
        minimum_validation_rows_per_benchmark=1,
        gate_rows_per_benchmark=1,
        validation_interval_steps=1,
        stage0_top_m=3,
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=4,
        trainable_replay_prefix=False,
    )

    assert report["status"] == "ok"
    assert report["stage4_method"] == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
    assert report["dynamic_selection"]["promotion_status"] in {
        "promoted",
        "stage4_not_promoted",
    }
    assert "cmc_fused_selection" in report["dynamic_selection"]
    assert [item["step"] for item in report["validation_history"]] == [0, 1]
    step0 = json.loads(
        (output_dir / "validation" / "step0.json").read_text(encoding="utf-8")
    )
    step1 = json.loads(
        (output_dir / "validation" / "step1.json").read_text(encoding="utf-8")
    )
    assert step0["cmc_fused"]["alpha_zero_exact"] is True
    assert step0["cmc_fused"]["alpha_one_exact"] is True
    assert step0["cmc_fused"]["zero_history_exact"] is True
    assert step0["balanced_macro"]["fused_mrr"] == pytest.approx(
        step0["stage2_baseline"]["static_macro_mrr"]
    )
    assert step1["gradient_health"]["finite"] is True
    assert step1["gradient_health"]["nonzero"] is True
    compact = torch.load(
        report["dynamic_selection"]["selected_checkpoint_path"],
        map_location="cpu",
    )
    assert all(
        key.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_utility_gate.",
            )
        )
        for key in compact["model_state_dict"]
    )


def test_train_candidate_admission_writes_parent_bound_compact_validation(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "candidate-admission-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_candidate_admission"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    rows = _safe_stage4_causal_rows()
    for row in rows:
        if int(row.get("step_index", 0)) == 1:
            row["state_text"] = "future state"
    _write_jsonl(trajectories, rows)
    model = _Stage4MemorySensitiveUnifiedModel()
    model.route_memory_residual_adapter = RouteMemoryResidualAdapter(3)
    model.route_memory_candidate_admission_residual = (
        CandidateAdmissionResidualHead(3)
    )

    report = train_stage4_act_with_model(
        model=model,
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=4,
        seed=17,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        stage4_method=CANDIDATE_ADMISSION_RESIDUAL_V1,
        base_cmc_checkpoint_sha256="a" * 64,
        candidate_admission_static_k=2,
        candidate_admission_dynamic_extra_k=1,
        allowed_benchmarks=set(SAFE_STAGE4_BENCHMARKS),
        benchmark_caps={benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS},
        validation_fraction=0.10,
        validation_rows_per_benchmark=1,
        minimum_validation_rows_per_benchmark=1,
        gate_rows_per_benchmark=1,
        validation_interval_steps=1,
        stage0_top_m=None,
        trainable_replay_prefix=False,
    )

    assert report["status"] == "ok"
    assert report["training_objective"] == (
        "candidate_admission_constrained_residual"
    )
    assert report["base_cmc_checkpoint_sha256"] == "a" * 64
    assert report["freeze_report"]["trainable_modules"] == [
        "route_memory_candidate_admission_residual"
    ]
    assert "candidate_admission_selection" in report["dynamic_selection"]
    compact = torch.load(
        report["dynamic_selection"]["selected_checkpoint_path"],
        map_location="cpu",
    )
    assert compact["base_cmc_checkpoint_sha256"] == "a" * 64
    assert all(
        key.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_admission_residual.",
            )
        )
        for key in compact["model_state_dict"]
    )
    step0 = json.loads(
        (output_dir / "validation" / "step0.json").read_text(encoding="utf-8")
    )
    step1 = json.loads(
        (output_dir / "validation" / "step1.json").read_text(encoding="utf-8")
    )
    assert step0["candidate_admission"]["step0_exact_static"] is True
    assert step1["gradient_health"]["finite"] is True
    assert step1["gradient_health"]["nonzero"] is True
    assert step1["gradient_health"]["gradient_tensor_count"] > 0


def test_stage4_safe_validation_report_pins_resume_checkpoint(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "safe-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_safe"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(trajectories, _safe_stage4_causal_rows())

    report = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=4,
        seed=17,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        allowed_benchmarks=set(SAFE_STAGE4_BENCHMARKS),
        benchmark_caps={benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS},
        validation_fraction=0.10,
        validation_rows_per_benchmark=2,
        minimum_validation_rows_per_benchmark=1,
        gate_rows_per_benchmark=2,
        validation_interval_steps=1,
        stage0_top_m=3,
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=4,
    )

    validation_report = json.loads(
        (output_dir / "validation" / "step1.json").read_text(encoding="utf-8")
    )
    assert validation_report["safe_fused"]["zero_history_exact"] is True
    assert validation_report["balanced_macro"]["safe_fused_mrr"] >= 0.0
    assert all(
        "safe_fused_mrr" in metrics
        for metrics in validation_report["by_benchmark"].values()
    )
    dynamic_selection = json.loads(
        (output_dir / "stage4_dynamic_selection.json").read_text(encoding="utf-8")
    )
    assert dynamic_selection["safe_fused_selection"] == validation_report["safe_fused"]
    resume_identity = validation_report["resume_checkpoint"]
    assert resume_identity["path"] == str(
        (output_dir / "resume" / "clstr_stage4_safe-step1.pt").resolve()
    )
    assert len(resume_identity["sha256"]) == 64
    assert report["latest_resume_checkpoint"] == resume_identity


def test_stage4_safe_materializes_frozen_embedding_cache(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "safe-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_safe"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(trajectories, _safe_stage4_causal_rows())

    report = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=4,
        seed=17,
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        allowed_benchmarks=set(SAFE_STAGE4_BENCHMARKS),
        benchmark_caps={benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS},
        validation_fraction=0.10,
        validation_rows_per_benchmark=2,
        minimum_validation_rows_per_benchmark=1,
        gate_rows_per_benchmark=2,
        validation_interval_steps=1,
        stage0_top_m=3,
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=4,
    )

    cache = report["frozen_embedding_cache"]
    assert cache["mode"] == "scheduled_rows_v1"
    assert cache["backbone_frozen"] is True
    assert len(cache["cache_identity_sha256"]) == 64
    assert cache["train"]["used"] is True
    assert cache["validation"]["used"] is True
    assert cache["gate"]["used"] is True
    assert cache["train"]["row_count"] == 4
    assert cache["train"]["source_row_count"] == 16
    assert cache["train"]["scheduled_reference_count"] == 4
    assert cache["train"]["scheduled_unique_row_count"] == 4
    assert cache["validation"]["row_count"] == 8
    assert cache["gate"]["row_count"] == 8


def test_stage4_safe_resume_is_exact_and_rejects_protocol_drift(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "safe-trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_safe"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
            {"skill_id": "skill/other", "name": "other"},
        ],
    )
    _write_jsonl(trajectories, _safe_stage4_causal_rows())
    common = {
        "trajectories_path": trajectories,
        "skills_path": skills,
        "output_dir": output_dir,
        "max_steps": 2,
        "batch_size": 4,
        "seed": 17,
        "route_scorer": "unified_memory",
        "next_skill_pool_mode": "full_pool",
        "allowed_benchmarks": set(SAFE_STAGE4_BENCHMARKS),
        "benchmark_caps": {
            benchmark: 4 for benchmark in SAFE_STAGE4_BENCHMARKS
        },
        "validation_fraction": 0.10,
        "validation_rows_per_benchmark": 2,
        "minimum_validation_rows_per_benchmark": 1,
        "gate_rows_per_benchmark": 2,
        "validation_interval_steps": 1,
        "stage0_top_m": 3,
        "stage0_handoff_query_mode": "raw_state",
        "stage0_candidate_encode_batch_size": 4,
    }

    uninterrupted = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        **common,
    )
    expected_payload = torch.load(
        output_dir / "checkpoints" / "clstr_stage4_safe-step2.pt",
        map_location="cpu",
    )
    expected_delta = {
        key: value.clone()
        for key, value in expected_payload["model_state_dict"].items()
    }
    expected_history = list(uninterrupted["history"])
    step1_resume = output_dir / "resume" / "clstr_stage4_safe-step1.pt"
    for future_artifact in (output_dir / "validation").glob("step2*"):
        future_artifact.unlink()

    resumed = train_stage4_act_with_model(
        model=_Stage4UnifiedMemoryModel(),
        resume_checkpoint_path=step1_resume,
        **common,
    )
    resumed_payload = torch.load(
        output_dir / "checkpoints" / "clstr_stage4_safe-step2.pt",
        map_location="cpu",
    )
    assert resumed["history"] == expected_history
    assert [item["step"] for item in resumed["validation_history"]] == [0, 1, 2]
    assert expected_delta.keys() == resumed_payload["model_state_dict"].keys()
    assert all(
        torch.equal(expected_delta[key], resumed_payload["model_state_dict"][key])
        for key in expected_delta
    )

    tampered_payload = torch.load(step1_resume, map_location="cpu")
    tampered_payload["data_protocol_manifest_sha256"] = "wrong-split"
    tampered_resume = tmp_path / "tampered-resume.pt"
    torch.save(tampered_payload, tampered_resume)
    with pytest.raises(ValueError, match="resume data protocol mismatch"):
        train_stage4_act_with_model(
            model=_Stage4UnifiedMemoryModel(),
            resume_checkpoint_path=tampered_resume,
            **common,
        )


def test_train_stage4_act_with_model_uses_stage0_topm_handoff_without_gold_injection(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_handoff"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([
            {
                "benchmark": "traject_bench",
                "task_id": "kept",
                "state_text": "state-a",
                "action_text": "do a",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "skipped",
                "state_text": "state-a",
                "action_text": "do a",
                "next_observation_text": "next-c",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            },
        ]),
    )

    report = train_stage4_act_with_model(
        model=_Stage4HandoffModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        validation_rows_per_benchmark=0,
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
    )

    assert report["status"] == "ok"
    data_report = report["data_report"]
    handoff = data_report["stage0_candidate_handoff"]
    assert handoff["enabled"] is True
    assert handoff["candidate_source"] == "stage0_topm_online"
    assert handoff["top_m"] == 1
    assert handoff["positive_missing_policy"] == "skip"
    assert handoff["query_mode"] == "raw_state"
    assert handoff["injected_positive_rows"] == 0
    assert data_report["candidate_source"] == "declared_legal_full_skill_pool"
    assert data_report["static_candidate_source"] == "stage0_topm_online"
    assert data_report["stage4_rows"] == 2
    assert data_report["positive_injected_rows"] == 0
    assert data_report["stage0_next_static_hit_rows"] == 1
    assert data_report["stage0_next_static_miss_retained_rows"] == 1
    assert data_report["positive_missing_skip_rows"] == 0
    assert report["last_metrics"]["stage4_candidate_count"] == 3.0
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "stage0_candidate_handoff_started" in setup_phases


def test_stage4_handoff_does_not_drop_act_row_when_current_positive_misses_topm(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_current_miss"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([
            {
                "benchmark": "traject_bench",
                "task_id": "current-miss-next-hit",
                "state_text": "state-current-miss",
                "action_text": "do a",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        ]),
    )

    report = train_stage4_act_with_model(
        model=_Stage4HandoffModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        validation_rows_per_benchmark=0,
        stage0_top_m=1,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
    )

    assert report["status"] == "ok"
    assert report["data_report"]["stage4_rows"] == 1
    handoff = report["data_report"]["stage0_candidate_handoff"]
    assert handoff["current_positive_required_rows"] == 1
    assert handoff["current_positive_covered_rows"] == 0
    assert handoff["next_positive_required_rows"] == 1
    assert handoff["skipped_rows"] == 0


def test_stage4_handoff_passes_inventory_min_candidates_to_stage0_topm(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "stage4_inventory_min"
    _write_jsonl(
        skills,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    _write_jsonl(
        trajectories,
        _stage4_strict_causal_pairs([
            {
                "benchmark": "traject_bench",
                "task_id": "inventory-min",
                "state_text": "state-a",
                "action_text": "do a",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "visible_inventory_skill_ids": ["skill/b"],
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        ]),
    )

    report = train_stage4_act_with_model(
        model=_Stage4HandoffModel(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        validation_rows_per_benchmark=0,
        stage0_top_m=3,
        stage0_inventory_min_candidates=2,
        stage0_positive_missing_policy="skip",
        stage0_handoff_query_mode="raw_state",
    )

    handoff = report["data_report"]["stage0_candidate_handoff"]
    assert handoff["inventory_candidate_rows"] > 0
    assert handoff["inventory_candidate_topk_backfilled_rows"] > 0
    assert handoff["inventory_candidate_topk_backfilled_candidates"] > 0
    assert report["data_report"]["stage4_rows"] == 1


def test_stage4_builder_filters_stage0_topm_candidates_to_requested_count_without_dropping_positive():
    skills = ["skill/a", "skill/b", "skill/c"]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skills)}
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "topm-filter",
                "state_text": "state",
                "action_text": "act",
                "next_observation_text": "next",
                "next_state_text": "next state",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "stage0_next_candidate_skill_ids": ["skill/a", "skill/c", "skill/b"],
                "provenance": {"split": "train"},
            }
        ],
        skill_id_to_idx,
        candidate_count=2,
        max_rows=None,
        allowed_benchmarks={"toolbench_g3"},
        stage0_candidate_handoff_report={"enabled": True},
    )

    assert report["stage4_rows"] == 1
    assert rows[0]["candidate_next_skill_ids"] == ["skill/a", "skill/b"]
    assert rows[0]["positive_next_skill_position"] == 1
    assert report["stage0_candidate_filter"]["inventory_mask_positive_missing_rows"] == 1
