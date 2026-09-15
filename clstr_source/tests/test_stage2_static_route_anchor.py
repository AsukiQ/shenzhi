from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from clstr.belief import BeliefGate, InitialBelief, TransitionPredictor, subspace_obs
from clstr.model import UnifiedMemoryRetriever
from clstr.stage2_static_route_anchor import (
    Stage2StaticRouteTeacher,
    static_route_anchor_kl,
)


class _TinySkillTable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Linear(2, 2, bias=False)
        self.E = nn.Parameter(torch.eye(2), requires_grad=False)
        self.logit_scale_belief = nn.Parameter(torch.tensor([math.log(0.2)]))
        self.skill_bias_belief = nn.Parameter(torch.zeros(2))
        self.belief_top_k = 2
        with torch.no_grad():
            self.W.weight.copy_(torch.eye(2))

    def _cosine_scores(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.W(h), p=2, dim=-1) @ F.normalize(
            self.E, p=2, dim=-1
        ).t()

    def belief_logits(self, h: torch.Tensor) -> torch.Tensor:
        return (
            self.logit_scale_belief.exp().clamp(max=100.0)
            * self._cosine_scores(h)
            + self.skill_bias_belief.unsqueeze(0)
        )


class _TinyUnifiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(initial_belief_top_k=2)
        self.skill_table = _TinySkillTable()
        self.initial_belief_head = InitialBelief(2)
        self.unified_retriever = UnifiedMemoryRetriever(2)
        self.transition = TransitionPredictor(2, 2, None)
        self.gate = BeliefGate(2)
        self.action_proj = nn.Linear(2, 2, bias=False)

    def initial_belief(self, h: torch.Tensor) -> torch.Tensor:
        belief = subspace_obs(self.skill_table, h, top_k=2)
        return self.initial_belief_head(h, belief)

    def unified_route_full_logits(
        self, h: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        route = self.unified_retriever(h, memory)
        return route @ self.skill_table.E.to(route).t()


def test_static_route_anchor_is_zero_for_identical_student() -> None:
    model = _TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.ones(2, 2, dtype=torch.bool)

    loss = static_route_anchor_kl(model, teacher, h, valid)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-7)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not hasattr(teacher, "skill_embeddings")


def test_static_route_anchor_detects_router_drift_without_teacher_gradients() -> None:
    model = _TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)
    with torch.no_grad():
        model.unified_retriever.fuse[0].weight[0, 0].add_(0.25)

    loss = static_route_anchor_kl(
        model,
        teacher,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.ones(2, 2, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() > 0.0
    assert model.unified_retriever.fuse[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_static_route_anchor_reuses_precomputed_full_pool_logits() -> None:
    class _NoRouteCall:
        def initial_belief(self, _h):
            raise AssertionError("student static route was recomputed")

        def unified_route_full_logits(self, _h, _memory):
            raise AssertionError("student static logits were recomputed")

    class _NoTeacherCall:
        def full_logits(self, _model, _h):
            raise AssertionError("teacher static logits were recomputed")

    student_logits = torch.tensor([[2.0, 1.0]], requires_grad=True)
    loss = static_route_anchor_kl(
        _NoRouteCall(),
        _NoTeacherCall(),
        torch.tensor([[1.0, 0.0]]),
        torch.ones(1, 2, dtype=torch.bool),
        student_logits=student_logits,
        teacher_logits=torch.tensor([[1.5, 1.0]]),
    )
    loss.backward()

    assert loss.item() > 0.0
    assert student_logits.grad is not None


def test_static_route_anchor_rejects_rows_without_two_legal_candidates() -> None:
    model = _TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)

    with pytest.raises(ValueError, match="at least two legal candidates"):
        static_route_anchor_kl(
            model,
            teacher,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, False]]),
        )


def test_step_zero_teacher_dynamic_route_is_immutable_after_student_drift() -> None:
    model = _TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)
    kwargs = {
        "h_current": torch.tensor([[1.0, 0.0]]),
        "current_skill_labels": torch.tensor([0]),
        "action_embeddings": torch.tensor([[0.25, 0.75]]),
        "observation_embeddings": torch.tensor([[0.0, 1.0]]),
        "h_next": torch.tensor([[0.0, 1.0]]),
    }

    before = teacher.post_action_full_logits(model, **kwargs)
    with torch.no_grad():
        model.transition.cell.weight_hh.add_(1.0)
        model.gate.linear.weight.add_(1.0)
        model.action_proj.weight.add_(1.0)
    after = teacher.post_action_full_logits(model, **kwargs)

    assert torch.equal(before, after)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
