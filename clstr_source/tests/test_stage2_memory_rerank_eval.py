from __future__ import annotations

import torch
import pytest

from clstr.stage2_memory_rerank_eval import (
    add_candidate_memory_scores,
    evaluate_stage2_memory_rerank_batch,
    strict_metrics_from_retained,
)


def test_add_candidate_memory_scores_aligns_by_skill_id_after_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    candidate_rows = [[11, 22, 33]]
    memory_by_skill = [{"33": 5.0, "11": -1.0}]

    updated = add_candidate_memory_scores(
        logits,
        candidate_rows=candidate_rows,
        memory_scores_by_skill_id=memory_by_skill,
        skill_ids_by_idx={11: "11", 22: "22", 33: "33"},
        weight=0.5,
    )

    assert updated.tolist() == [[0.5, 2.0, 5.5]]


def test_strict_metrics_from_retained_scales_ranking_metrics_to_source_denominator():
    metrics = {"transition_skill_recall@1": 0.5, "transition_skill_recall@5": 0.75, "transition_skill_mrr": 0.6}

    strict = strict_metrics_from_retained(metrics, retained_rows=8, source_rows=10)

    assert strict["strict_transition_skill_recall@1"] == 0.4
    assert strict["strict_transition_skill_recall@5"] == pytest.approx(0.6)
    assert strict["strict_transition_skill_mrr"] == 0.48


def test_memory_rerank_evaluation_initializes_with_model_initial_belief(monkeypatch):
    import clstr.stage2_memory_rerank_eval as module

    captured = {}

    class _SkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.eye(2), requires_grad=False)

        def logits(self, h):
            return h

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.skill_table = _SkillTable()
            self.initial_belief_calls = 0

        @property
        def device(self):
            return self.anchor.device

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                rows.append(
                    torch.tensor([0.0, 1.0], dtype=torch.float32)
                    if "next" in str(text)
                    else torch.tensor([1.0, 0.0], dtype=torch.float32)
                )
            return torch.stack(rows) + self.anchor * 0.0

        def initial_belief(self, h, top_k=None):
            del top_k
            self.initial_belief_calls += 1
            return torch.tensor([[7.0, 9.0]], dtype=h.dtype, device=h.device).expand(h.size(0), -1)

    def fake_transition_logits(model, h, m_obs, *args, **kwargs):
        del model, h, args
        captured["m_obs"] = m_obs.detach().clone()
        candidate_ids = kwargs["candidate_ids"]
        logits = torch.tensor([[0.0, 1.0]], dtype=m_obs.dtype, device=m_obs.device).expand(
            candidate_ids.size(0), -1
        )
        return logits, "recording_transition", logits, torch.zeros_like(logits)

    monkeypatch.setattr(module, "_transition_candidate_logits_for_mode", fake_transition_logits)
    model = _Model()
    metrics = evaluate_stage2_memory_rerank_batch(
        model,
        [
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "stage0_next_candidate_skill_indices": [0, 1],
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/a": 0, "skill/b": 1},
        skill_ids_by_idx={0: "skill/a", 1: "skill/b"},
        skills=[{"skill_id": "skill/a"}, {"skill_id": "skill/b"}],
        device=torch.device("cpu"),
        transition_inventory_mask_mode="off",
    )

    assert metrics["stage2_memory_count"] == 1.0
    assert model.initial_belief_calls == 1
    assert torch.equal(captured["m_obs"], torch.tensor([[7.0, 9.0]]))
