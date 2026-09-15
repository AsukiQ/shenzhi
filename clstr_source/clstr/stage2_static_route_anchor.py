from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F


def sparse_subspace_from_logits(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    if logits.ndim != 2 or embeddings.ndim != 2:
        raise ValueError("belief logits and skill embeddings must be rank-2")
    if int(logits.size(-1)) != int(embeddings.size(0)):
        raise ValueError("belief logits must align with skill embeddings")
    k = min(max(1, int(top_k)), int(logits.size(-1)))
    values, indices = torch.topk(logits, k=k, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    selected = embeddings.index_select(0, indices.reshape(-1)).view(
        logits.size(0), k, -1
    )
    return torch.bmm(probabilities.unsqueeze(1), selected).squeeze(1)


class Stage2StaticRouteTeacher(torch.nn.Module):
    """Frozen step-zero static router without duplicating Qwen or skill embeddings."""

    def __init__(
        self,
        *,
        initial_belief_head: torch.nn.Module,
        unified_retriever: torch.nn.Module,
        belief_logit_scale: torch.Tensor,
        belief_bias: torch.Tensor,
        initial_belief_top_k: int,
        transition: torch.nn.Module | None = None,
        gate: torch.nn.Module | None = None,
        action_proj: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.initial_belief_head = initial_belief_head
        self.unified_retriever = unified_retriever
        self.transition = transition
        self.gate = gate
        self.action_proj = action_proj
        self.register_buffer("belief_logit_scale", belief_logit_scale.detach().clone())
        self.register_buffer("belief_bias", belief_bias.detach().clone())
        self.initial_belief_top_k = max(1, int(initial_belief_top_k))
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_model(cls, model: Any) -> "Stage2StaticRouteTeacher":
        skill_table = getattr(model, "skill_table", None)
        if skill_table is None:
            raise ValueError("static route teacher requires model.skill_table")
        for name in ("logit_scale_belief", "skill_bias_belief", "E"):
            if not isinstance(getattr(skill_table, name, None), torch.Tensor):
                raise ValueError(f"static route teacher requires skill_table.{name}")
        initial_belief_head = getattr(model, "initial_belief_head", None)
        unified_retriever = getattr(model, "unified_retriever", None)
        if not isinstance(initial_belief_head, torch.nn.Module):
            raise ValueError("static route teacher requires model.initial_belief_head")
        if not isinstance(unified_retriever, torch.nn.Module):
            raise ValueError("static route teacher requires model.unified_retriever")
        top_k = getattr(getattr(model, "config", None), "initial_belief_top_k", 64)
        return cls(
            initial_belief_head=copy.deepcopy(initial_belief_head),
            unified_retriever=copy.deepcopy(unified_retriever),
            belief_logit_scale=skill_table.logit_scale_belief,
            belief_bias=skill_table.skill_bias_belief,
            initial_belief_top_k=int(top_k),
            transition=(
                copy.deepcopy(model.transition)
                if isinstance(getattr(model, "transition", None), torch.nn.Module)
                else None
            ),
            gate=(
                copy.deepcopy(model.gate)
                if isinstance(getattr(model, "gate", None), torch.nn.Module)
                else None
            ),
            action_proj=(
                copy.deepcopy(model.action_proj)
                if isinstance(getattr(model, "action_proj", None), torch.nn.Module)
                else None
            ),
        )

    def _belief_memory(self, model: Any, h: torch.Tensor) -> torch.Tensor:
        skill_table = getattr(model, "skill_table", None)
        cosine_scores = getattr(skill_table, "_cosine_scores", None)
        if not callable(cosine_scores):
            raise ValueError("static route teacher requires skill_table._cosine_scores")
        skill_embeddings = getattr(skill_table, "E", None)
        if not isinstance(skill_embeddings, torch.Tensor):
            raise ValueError("static route teacher requires skill_table.E")
        with torch.no_grad():
            cosine = cosine_scores(h).detach()
            scale = self.belief_logit_scale.to(
                device=cosine.device, dtype=cosine.dtype
            ).exp().clamp(max=100.0)
            bias = self.belief_bias.to(device=cosine.device, dtype=cosine.dtype)
            belief_logits = scale * cosine + bias.unsqueeze(0)
            embeddings = skill_embeddings.detach().to(
                device=cosine.device, dtype=cosine.dtype
            )
            belief = sparse_subspace_from_logits(
                belief_logits,
                embeddings,
                top_k=self.initial_belief_top_k,
            )
            return belief

    def full_logits(self, model: Any, h: torch.Tensor) -> torch.Tensor:
        skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
        if not isinstance(skill_embeddings, torch.Tensor):
            raise ValueError("static route teacher requires skill_table.E")
        with torch.no_grad():
            belief = self._belief_memory(model, h)
            memory = self.initial_belief_head(h, belief)
            route = self.unified_retriever(h, memory)
            embeddings = skill_embeddings.detach().to(device=route.device, dtype=route.dtype)
            return route @ embeddings.t()

    def post_action_full_logits(
        self,
        model: Any,
        *,
        h_current: torch.Tensor,
        current_skill_labels: torch.Tensor,
        action_embeddings: torch.Tensor | None,
        observation_embeddings: torch.Tensor,
        h_next: torch.Tensor,
    ) -> torch.Tensor:
        if self.transition is None or self.gate is None:
            raise ValueError("step-zero dynamic teacher requires transition and gate")
        skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
        if not isinstance(skill_embeddings, torch.Tensor):
            raise ValueError("step-zero dynamic teacher requires skill_table.E")
        with torch.no_grad():
            current_belief = self._belief_memory(model, h_current)
            current_memory = self.initial_belief_head(h_current, current_belief)
            if action_embeddings is None:
                labels = current_skill_labels.to(
                    device=skill_embeddings.device,
                    dtype=torch.long,
                )
                action_input = skill_embeddings.detach().index_select(0, labels)
            else:
                action_input = action_embeddings
                if self.action_proj is not None:
                    action_input = self.action_proj(action_input)
            action_input = action_input.to(
                device=current_memory.device,
                dtype=current_memory.dtype,
            )
            observation_embeddings = observation_embeddings.to(
                device=current_memory.device,
                dtype=current_memory.dtype,
            )
            predicted_memory = self.transition(
                current_memory,
                action_input,
                observation_embeddings,
            )
            observation_memory = self._belief_memory(model, h_next)
            gamma = self.gate(
                predicted_memory,
                observation_memory,
                observation_embeddings,
            )
            next_memory = gamma * observation_memory + (1.0 - gamma) * predicted_memory
            route = self.unified_retriever(h_next, next_memory)
            embeddings = skill_embeddings.detach().to(device=route.device, dtype=route.dtype)
            return route @ embeddings.t()


def static_route_anchor_kl(
    model: Any,
    teacher: Stage2StaticRouteTeacher,
    h: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    student_logits: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return KL(teacher || student) on each row's legal static candidates."""

    if h.ndim != 2 or not h.is_floating_point():
        raise ValueError("static route anchor states must be floating rank-2 tensors")
    if student_logits is None:
        student_logits_fn = getattr(model, "unified_route_full_logits", None)
        initial_belief_fn = getattr(model, "initial_belief", None)
        if not callable(student_logits_fn) or not callable(initial_belief_fn):
            raise ValueError("static route anchor requires unified static routing methods")
        student_memory = initial_belief_fn(h)
        student_logits = student_logits_fn(h, student_memory)
    if teacher_logits is None:
        teacher_logits = teacher.full_logits(model, h)
    if (
        student_logits.ndim != 2
        or teacher_logits.ndim != 2
        or not student_logits.is_floating_point()
        or not teacher_logits.is_floating_point()
        or int(student_logits.size(0)) != int(h.size(0))
    ):
        raise ValueError("static route anchor logits must be floating rank-2 tensors aligned to states")
    if (
        valid_mask.ndim != 2
        or valid_mask.shape != student_logits.shape
        or teacher_logits.shape != student_logits.shape
    ):
        raise ValueError("static route anchor masks and logits must have matching shapes")
    valid = valid_mask.to(device=student_logits.device, dtype=torch.bool)
    if bool((valid.sum(dim=-1) < 2).any()):
        raise ValueError("static route anchor requires at least two legal candidates per row")
    floor = torch.finfo(student_logits.dtype).min
    student_log_prob = torch.log_softmax(student_logits.masked_fill(~valid, floor), dim=-1)
    teacher_prob = torch.softmax(
        teacher_logits.to(student_logits).masked_fill(~valid, floor), dim=-1
    ).detach()
    row_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return row_kl.mean().clamp_min(0.0)
