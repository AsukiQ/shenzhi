from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from clstr.belief import subspace_obs
from clstr.data import RetrievalPositive, Trajectory, VerifiedPair
from clstr.encoders import serialize_state_text
from clstr.transition_utils import model_action_embeddings, transition_action_input


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _encode_states(model, states) -> torch.Tensor:
    if hasattr(model, "encode_states"):
        return model.encode_states(states)
    return model.encoder([serialize_state_text(state) for state in states])


def _encode_observations(model, observations: list[str]) -> torch.Tensor:
    if hasattr(model, "encode_observations"):
        return model.encode_observations(observations)
    return model.encoder(observations)


def policy_loss(trajectories: list[Trajectory]) -> torch.Tensor:
    grouped: dict[str | int, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        grouped[traj.task_id].append(traj)

    losses: list[torch.Tensor] = []
    device = None
    for group in grouped.values():
        rewards = torch.tensor([traj.reward for traj in group], dtype=torch.float32)
        centered = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)
        for traj, adv in zip(group, centered):
            log_probs = [step.log_prob for step in traj.steps if step.log_prob is not None]
            if not log_probs:
                continue
            device = log_probs[0].device
            log_prob_sum = torch.stack(log_probs).sum() / max(len(traj.steps), 1)
            losses.append(-(adv.to(log_prob_sum.device) * log_prob_sum))

    if not losses:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.stack(losses).mean()


def reference_kl_loss(trajectories: list[Trajectory]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    device = None
    for traj in trajectories:
        for step in traj.steps:
            if step.policy_logits is None or step.ref_logits is None:
                continue
            policy_logits = step.policy_logits
            ref_logits = step.ref_logits.to(policy_logits.device)
            device = policy_logits.device
            log_p = F.log_softmax(policy_logits, dim=-1)
            log_q = F.log_softmax(ref_logits, dim=-1)
            p = log_p.exp()
            losses.append((p * (log_p - log_q)).sum())
    if not losses:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.stack(losses).mean()


def transition_loss(trajectories: list[Trajectory]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    device = None
    for traj in trajectories:
        for step in traj.steps:
            if step.m_hat is None or step.m_tilde_next is None:
                continue
            device = step.m_hat.device
            cosine = F.cosine_similarity(step.m_hat, step.m_tilde_next.detach(), dim=-1)
            losses.append(1.0 - cosine.mean())
    if not losses:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.stack(losses).mean()


def _full_replay_m_t(model, prefix, target_state) -> torch.Tensor:
    h_0 = _encode_states(model, [prefix[0].x if prefix else target_state])
    m = subspace_obs(model.skill_table, h_0)
    if not prefix:
        return m.squeeze(0)

    for idx, replay_step in enumerate(prefix):
        x_next = prefix[idx + 1].x if idx + 1 < len(prefix) else target_state
        obs_emb = _encode_observations(model, [replay_step.obs])
        action = torch.tensor([replay_step.skill_idx], device=model.device, dtype=torch.long)
        if hasattr(model, "step_update"):
            m, _, _ = model.step_update(
                m_t=m,
                a_t=action,
                o_t_emb=obs_emb,
                x_next_text=[serialize_state_text(x_next)],
            )
        else:
            action_input = transition_action_input(model, action, like=obs_emb)
            m = model.transition(m, action_input, obs_emb)
    return m.squeeze(0)


def _resolve_m_t(
    model,
    vp: VerifiedPair,
    allow_approximate_m_t: bool = False,
) -> torch.Tensor:
    if vp.m_t_exact is not None:
        return vp.m_t_exact.to(model.device)
    if vp.replay_prefix is not None:
        return _full_replay_m_t(model, vp.replay_prefix, vp.state_before)
    if not allow_approximate_m_t:
        raise ValueError(
            "action_loss requires m_t_exact or replay_prefix; approximate subspace_obs fallback is disabled"
        )
    h_t = _encode_states(model, [vp.state_before])
    return subspace_obs(model.skill_table, h_t).squeeze(0)


def action_loss(
    model,
    verified_pairs: list[VerifiedPair],
    stats: dict[str, Any] | None = None,
    allow_approximate_m_t: bool = False,
) -> torch.Tensor:
    device = _model_device(model)
    if not verified_pairs:
        if stats is not None:
            stats["action_loss_used"] = 0
            stats["retrieval_recall_raw"] = 0.0
            stats["retrieval_miss_raw"] = 0
        return torch.zeros((), device=device)

    losses: list[torch.Tensor] = []
    used = 0
    for vp in verified_pairs:
        if vp.a_next_plus not in vp.candidates_next:
            continue
        m_t = _resolve_m_t(model, vp, allow_approximate_m_t=allow_approximate_m_t).unsqueeze(0)
        obs_emb = _encode_observations(model, [vp.obs_at_t])
        action = torch.tensor([vp.action_at_t], device=device, dtype=torch.long)
        action_input = transition_action_input(model, action, like=obs_emb)
        m_hat = model.transition(m_t, action_input, obs_emb)
        if hasattr(model, "transition_logits"):
            logits = model.transition_logits(m_hat, [vp.candidates_next]).squeeze(0)
        else:
            cand_idx = torch.tensor([vp.candidates_next], device=device, dtype=torch.long)
            cand_emb = model_action_embeddings(model, cand_idx.squeeze(0))
            if cand_emb is None:
                cand_emb = model.action_emb(cand_idx.squeeze(0))
            logits = model.trans_head(m_hat, cand_emb.unsqueeze(0)).squeeze(0)
        target = torch.tensor(
            [vp.candidates_next.index(vp.a_next_plus)],
            device=device,
            dtype=torch.long,
        )
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        used += 1

    if stats is not None:
        raw_hits = sum(1 for vp in verified_pairs if vp.was_in_raw_topk)
        stats["action_loss_used"] = used
        stats["retrieval_recall_raw"] = raw_hits / max(len(verified_pairs), 1)
        stats["retrieval_miss_raw"] = len(verified_pairs) - raw_hits

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def retrieval_loss(
    model,
    positives: list[RetrievalPositive],
    num_negatives: int = 32,
    hard_ratio: float = 0.5,
) -> torch.Tensor:
    device = _model_device(model)
    if not positives:
        return torch.zeros((), device=device)

    losses: list[torch.Tensor] = []
    num_skills = model.skill_table.E.size(0)
    for positive in positives:
        h_t = _encode_states(model, [positive.state])
        logits = model.skill_table.retrieval_logits(h_t).squeeze(0)
        pos_idx = positive.positive_skill_idx
        masked = logits.clone()
        masked[pos_idx] = float("-inf")

        neg_budget = max(min(num_negatives, num_skills - 1), 0)
        hard_count = min(int(round(neg_budget * hard_ratio)), neg_budget)
        rand_count = max(neg_budget - hard_count, 0)

        hard_vals = torch.empty(0, device=device)
        hard_idx = torch.empty(0, device=device, dtype=torch.long)
        if hard_count > 0:
            hard_vals, hard_idx = torch.topk(masked, k=hard_count)

        rand_idx: list[int] = []
        if rand_count > 0:
            forbidden = {pos_idx, *hard_idx.tolist()}
            available = [idx for idx in range(num_skills) if idx not in forbidden]
            if available:
                perm = torch.randperm(len(available), device=device)
                rand_idx = [available[idx] for idx in perm[:rand_count].tolist()]

        rand_vals = logits[torch.tensor(rand_idx, device=device, dtype=torch.long)] if rand_idx else torch.empty(0, device=device)
        candidates = torch.cat([logits[pos_idx].unsqueeze(0), hard_vals, rand_vals])
        target = torch.tensor([0], device=device, dtype=torch.long)
        losses.append(F.cross_entropy(candidates.unsqueeze(0), target))

    return torch.stack(losses).mean()


def multi_positive_routing_loss(
    model,
    tasks,
    skill_id_to_idx: dict[str, int],
    stats: dict[str, Any] | None = None,
) -> torch.Tensor:
    device = _model_device(model)
    losses: list[torch.Tensor] = []
    top1_hits = 0
    top5_hits = 0
    used = 0
    positive_labels = 0

    for task in tasks:
        meta = getattr(task, "meta", {}) or {}
        positive_skill_ids = [str(skill_id) for skill_id in meta.get("positive_skill_ids", [])]
        positive_indices = [skill_id_to_idx[skill_id] for skill_id in positive_skill_ids if skill_id in skill_id_to_idx]
        if not positive_indices:
            continue

        h_t = _encode_states(model, [task.query])
        logits = model.skill_table.retrieval_logits(h_t).squeeze(0)
        positive_logits = logits[torch.tensor(positive_indices, device=logits.device, dtype=torch.long)]
        losses.append(torch.logsumexp(logits, dim=0) - torch.logsumexp(positive_logits, dim=0))

        ranked = torch.topk(logits, k=min(5, logits.numel())).indices.detach().cpu().tolist()
        top1_hits += int(any(idx in positive_indices for idx in ranked[:1]))
        top5_hits += int(any(idx in positive_indices for idx in ranked[:5]))
        used += 1
        positive_labels += len(positive_indices)

    if stats is not None:
        stats["routing_supervision_tasks"] = used
        stats["routing_supervision_positive_labels"] = positive_labels
        stats["routing_supervision_recall@1"] = top1_hits / max(used, 1)
        stats["routing_supervision_recall@5"] = top5_hits / max(used, 1)

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def make_weak_positives(trajectories: list[Trajectory]) -> list[RetrievalPositive]:
    positives: list[RetrievalPositive] = []
    for traj in trajectories:
        if traj.reward < 1:
            continue
        for step in traj.steps:
            if step.skill_idx is None:
                continue
            positives.append(
                RetrievalPositive(
                    state=step.x,
                    positive_skill_idx=step.skill_idx,
                )
            )
    return positives


def make_strong_positives(verified_pairs: list[VerifiedPair]) -> list[RetrievalPositive]:
    return [
        RetrievalPositive(state=vp.state_before, positive_skill_idx=vp.a_next_plus)
        for vp in verified_pairs
    ]


def total_loss(
    model,
    trajectories: list[Trajectory],
    verified_pairs: list[VerifiedPair],
    lambda_1: float = 0.1,
    lambda_2: float = 0.5,
    lambda_3: float = 0.2,
    beta: float = 0.01,
    allow_approximate_m_t: bool = False,
    variant: str = "act",
    stats: dict[str, Any] | None = None,
) -> torch.Tensor:
    loss = policy_loss(trajectories)
    loss = loss + beta * reference_kl_loss(trajectories)
    loss = loss + lambda_1 * transition_loss(trajectories)

    if variant == "base":
        positives = make_weak_positives(trajectories)
        loss = loss + lambda_3 * retrieval_loss(model, positives)
        return loss
    if variant == "act":
        positives = make_strong_positives(verified_pairs)
        loss = loss + lambda_3 * retrieval_loss(model, positives)
        loss = loss + lambda_2 * action_loss(
            model,
            verified_pairs,
            stats=stats,
            allow_approximate_m_t=allow_approximate_m_t,
        )
        return loss
    raise ValueError(variant)
