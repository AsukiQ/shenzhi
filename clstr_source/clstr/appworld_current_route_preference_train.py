from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.belief import subspace_obs
from clstr.full_base_train import (
    V4_1B_TRANSITION_SCORING_MODE,
    _transition_candidate_logits_for_mode,
)


def _set_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(trainable)


def configure_current_route_preference_trainable(
    model: torch.nn.Module,
    *,
    train_transition: bool = False,
) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_modules: list[str] = []
    for name in ("skill_head",):
        module = getattr(model, name, None)
        _set_trainable(module, True)
        if module is not None:
            trainable_modules.append(name)
    if train_transition:
        for name in ("transition", "trans_head", "action_proj"):
            module = getattr(model, name, None)
            _set_trainable(module, True)
            if module is not None:
                trainable_modules.append(name)
    return {
        "frozen_routing_foundation": True,
        "train_transition": bool(train_transition),
        "trainable_modules": trainable_modules,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _candidate_indices(sample: dict[str, Any], skill_id_to_idx: dict[str, int]) -> list[int]:
    indices: list[int] = []
    for skill_id in _as_list(sample.get("candidate_skill_ids")):
        key = str(skill_id)
        if key not in skill_id_to_idx:
            raise KeyError(f"candidate skill_id not found in model skill table: {key}")
        indices.append(int(skill_id_to_idx[key]))
    return indices


def _target_indices(sample: dict[str, Any], candidate_count: int) -> list[int]:
    targets: list[int] = []
    for item in _as_list(sample.get("target_local_indices")):
        idx = int(item)
        if 0 <= idx < int(candidate_count):
            targets.append(idx)
    return targets


def _avoid_indices(sample: dict[str, Any], candidate_count: int) -> list[int]:
    avoids: list[int] = []
    for item in _as_list(sample.get("avoid_local_indices")):
        idx = int(item)
        if 0 <= idx < int(candidate_count):
            avoids.append(idx)
    return avoids


def _rejected_indices(sample: dict[str, Any], candidate_count: int) -> list[int]:
    rejected: list[int] = []
    for item in _as_list(sample.get("rejected_local_indices")):
        idx = int(item)
        if 0 <= idx < int(candidate_count):
            rejected.append(idx)
    return rejected


def _belief_from_state(model: Any, h_t: torch.Tensor) -> torch.Tensor:
    try:
        return subspace_obs(model.skill_table, h_t)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return h_t


def _candidate_embeddings(model: Any, state_text: str, candidates: list[int]) -> torch.Tensor:
    if callable(getattr(model, "batch_cross_encode", None)):
        try:
            return model.batch_cross_encode([state_text], [candidates])
        except RuntimeError:
            pass
    skill_embs = model.skill_table.E.index_select(
        0,
        torch.tensor(candidates, device=model.device, dtype=torch.long),
    )
    return skill_embs.unsqueeze(0)


def _routing_logits(model: Any, h_t: torch.Tensor, candidates: list[int]) -> torch.Tensor:
    try:
        all_logits = model.skill_table.retrieval_logits(h_t).squeeze(0)
        candidate_idx = torch.tensor(candidates, device=all_logits.device, dtype=torch.long)
        return all_logits.index_select(0, candidate_idx).detach()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return torch.zeros(len(candidates), device=model.device)


def _standardize_scores(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float()
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1.0e-8)


def _score_range(scores: torch.Tensor | None) -> float | None:
    if scores is None:
        return None
    values = scores.float()
    if int(values.numel()) <= 1:
        return 0.0
    return float((values.max() - values.min()).detach().cpu().item())


def _reference_log_probs(sample: dict[str, Any], candidate_count: int, device: torch.device) -> torch.Tensor | None:
    values = _as_list(sample.get("policy_log_probs"))
    if not values:
        logits = _as_list(sample.get("candidate_policy_logits"))
        if logits:
            try:
                tensor = torch.tensor([float(item) for item in logits], device=device, dtype=torch.float32)
            except (TypeError, ValueError):
                return None
            if int(tensor.numel()) != int(candidate_count):
                return None
            return F.log_softmax(tensor, dim=-1)
        return None
    try:
        tensor = torch.tensor([float(item) for item in values], device=device, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if int(tensor.numel()) < int(candidate_count):
        return None
    if int(tensor.numel()) > int(candidate_count):
        tensor = tensor[: int(candidate_count)]
    return tensor - torch.logsumexp(tensor, dim=-1)


def _gated_standardize_scores(
    scores: torch.Tensor | None,
    *,
    min_range: float,
) -> tuple[torch.Tensor | None, bool, float | None]:
    if scores is None:
        return None, False, None
    raw_range = _score_range(scores)
    threshold = max(float(min_range), 0.0)
    enabled = bool(raw_range is not None and raw_range > 1.0e-8 and raw_range >= threshold)
    if not enabled:
        return torch.zeros_like(scores.float()), False, raw_range
    return _standardize_scores(scores), True, raw_range


def _previous_action_index(sample: dict[str, Any], skill_id_to_idx: dict[str, int]) -> int | None:
    for skill_id in _as_list(sample.get("previous_selected_skill_ids")):
        key = str(skill_id)
        if key in skill_id_to_idx:
            return int(skill_id_to_idx[key])
    return None


def _transition_candidate_logits(
    model: Any,
    sample: dict[str, Any],
    *,
    h_t: torch.Tensor,
    candidates: list[int],
    skill_id_to_idx: dict[str, int],
    transition_residual_lambda: float = 0.0,
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE,
) -> torch.Tensor | None:
    previous_idx = _previous_action_index(sample, skill_id_to_idx)
    if previous_idx is None:
        return None
    skill_table = getattr(model, "skill_table", None)
    skill_embs = getattr(skill_table, "E", None)
    if not isinstance(skill_embs, torch.Tensor) or not candidates:
        return None
    device = h_t.device
    with torch.no_grad():
        m_obs = _belief_from_state(model, h_t).detach()
        obs_text = str(sample.get("previous_execute_output") or sample.get("state_text") or "")
        try:
            obs_emb = model.encode_observations([obs_text]).detach()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            obs_emb = h_t.detach()
        current_labels = torch.tensor([previous_idx], device=device, dtype=torch.long)
        action_emb = skill_embs.index_select(0, current_labels.to(skill_embs.device)).to(
            device=device,
            dtype=h_t.dtype,
        )
    candidate_ids = torch.tensor([candidates], device=device, dtype=torch.long)
    logits, _head_type, _prior_logits, _residual_logits = _transition_candidate_logits_for_mode(
        model,
        h_t.detach(),
        m_obs,
        current_labels,
        obs_emb,
        action_emb,
        int(skill_embs.size(0)),
        candidate_ids=candidate_ids,
        residual_lambda=float(transition_residual_lambda),
        scoring_mode=str(transition_scoring_mode),
    )
    return logits.squeeze(0)


def _blend_with_routing(
    *,
    candidate_routing_logits: torch.Tensor,
    learned_component: torch.Tensor,
    policy_blend_alpha: float,
) -> torch.Tensor:
    alpha = min(max(float(policy_blend_alpha), 0.0), 1.0)
    return (1.0 - alpha) * _standardize_scores(candidate_routing_logits.detach()) + alpha * learned_component


def compute_current_route_preference_batch_loss(
    model: Any,
    samples: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    loss_score_mode: str = "policy_head",
    enable_suppress_loss: bool = False,
    enable_pairwise_correction_loss: bool = False,
    pairwise_correction_margin: float = 0.5,
    pairwise_correction_weight: float = 1.0,
    policy_blend_alpha: float = 0.5,
    transition_residual_lambda: float = 0.0,
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE,
    learned_component_min_range: float = 0.1,
    learned_component_trust_top_k: int | None = 80,
    stability_kl_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not samples:
        device = torch.device(getattr(model, "device", "cpu"))
        return torch.zeros((), device=device), {
            "sample_count": 0,
            "decision_count": 0,
            "loss": 0.0,
        }
    losses: list[torch.Tensor] = []
    target_counts: list[int] = []
    avoid_counts: list[int] = []
    candidate_counts: list[int] = []
    sample_weights: list[float] = []
    positive_sample_count = 0
    suppress_sample_count = 0
    pairwise_correction_sample_count = 0
    pairwise_correction_losses: list[float] = []
    transition_blend_count = 0
    policy_transition_blend_count = 0
    policy_transition_blend_transition_count = 0
    policy_transition_blend_policy_fallback_count = 0
    policy_component_enabled_count = 0
    policy_component_disabled_count = 0
    transition_component_enabled_count = 0
    transition_component_disabled_count = 0
    learned_component_enabled_count = 0
    learned_component_disabled_count = 0
    low_confidence_warmup_count = 0
    trust_region_active_count = 0
    trust_region_target_kept_count = 0
    skipped_learned_component_unavailable = 0
    skipped_target_outside_trust_region = 0
    skipped_avoid_outside_trust_region = 0
    skipped_transition_unavailable = 0
    skipped_no_supervision = 0
    skipped_suppress_disabled = 0
    base_preference_losses: list[torch.Tensor] = []
    stability_kl_losses: list[torch.Tensor] = []
    device = torch.device(getattr(model, "device", "cpu"))
    for sample in samples:
        state_text = str(sample.get("state_text") or sample.get("user_goal") or "")
        candidates = _candidate_indices(sample, skill_id_to_idx)
        targets = _target_indices(sample, len(candidates))
        avoids = _avoid_indices(sample, len(candidates))
        rejected = _rejected_indices(sample, len(candidates))
        if not targets and not avoids:
            skipped_no_supervision += 1
            continue
        candidate_counts.append(len(candidates))
        target_counts.append(len(targets))
        avoid_counts.append(len(avoids))
        try:
            sample_weight = float(sample.get("weight", 1.0))
        except (TypeError, ValueError):
            sample_weight = 1.0
        sample_weights.append(sample_weight)
        with torch.no_grad():
            h_t = model.encode_states([state_text])
            m_t = _belief_from_state(model, h_t)
            candidate_embs = _candidate_embeddings(model, state_text, candidates)
            candidate_routing_logits = _routing_logits(model, h_t, candidates)
        mode = str(loss_score_mode)
        if mode == "transition_blend":
            transition_logits = _transition_candidate_logits(
                model,
                sample,
                h_t=h_t.detach(),
                candidates=candidates,
                skill_id_to_idx=skill_id_to_idx,
                transition_residual_lambda=transition_residual_lambda,
                transition_scoring_mode=transition_scoring_mode,
            )
            if transition_logits is None:
                skipped_transition_unavailable += 1
                continue
            transition_component, transition_enabled, _transition_range = _gated_standardize_scores(
                transition_logits,
                min_range=float(learned_component_min_range),
            )
            if transition_enabled:
                transition_component_enabled_count += 1
            else:
                transition_component_disabled_count += 1
            if not transition_enabled or transition_component is None:
                learned_component_disabled_count += 1
                low_confidence_warmup_count += 1
                score_logits = transition_logits.float()
            else:
                learned_component_enabled_count += 1
                score_logits = _blend_with_routing(
                    candidate_routing_logits=candidate_routing_logits,
                    learned_component=transition_component,
                    policy_blend_alpha=policy_blend_alpha,
                )
            transition_blend_count += 1
        elif mode == "policy_head":
            policy_logits = model.policy_forward(
                h_t.detach(),
                m_t.detach(),
                candidate_embs.detach(),
                routing_logits=candidate_routing_logits,
            ).squeeze(0)[: len(candidates)]
            score_logits = policy_logits
        elif mode == "policy_transition_blend":
            policy_transition_blend_count += 1
            policy_logits = model.policy_forward(
                h_t.detach(),
                m_t.detach(),
                candidate_embs.detach(),
                routing_logits=candidate_routing_logits,
            ).squeeze(0)[: len(candidates)]
            transition_logits = _transition_candidate_logits(
                model,
                sample,
                h_t=h_t.detach(),
                candidates=candidates,
                skill_id_to_idx=skill_id_to_idx,
                transition_residual_lambda=transition_residual_lambda,
                transition_scoring_mode=transition_scoring_mode,
            )
            policy_component, policy_enabled, _policy_range = _gated_standardize_scores(
                policy_logits,
                min_range=float(learned_component_min_range),
            )
            if policy_enabled:
                policy_component_enabled_count += 1
            else:
                policy_component_disabled_count += 1
            transition_component, transition_enabled, _transition_range = _gated_standardize_scores(
                transition_logits,
                min_range=float(learned_component_min_range),
            )
            if transition_logits is not None:
                if transition_enabled:
                    transition_component_enabled_count += 1
                else:
                    transition_component_disabled_count += 1
            learned_parts = []
            if policy_enabled and policy_component is not None:
                learned_parts.append(policy_component)
            if transition_enabled and transition_component is not None:
                learned_parts.append(transition_component)
            if transition_component is None or not transition_enabled:
                policy_transition_blend_policy_fallback_count += 1
            else:
                policy_transition_blend_transition_count += 1
            if not learned_parts:
                learned_component_disabled_count += 1
                warmup_parts = [policy_logits.float()]
                if transition_logits is not None:
                    warmup_parts.append(transition_logits.float())
                if not warmup_parts:
                    skipped_learned_component_unavailable += 1
                    continue
                low_confidence_warmup_count += 1
                score_logits = torch.stack(warmup_parts, dim=0).mean(dim=0)
            else:
                learned_component_enabled_count += 1
                learned_component = torch.stack(learned_parts, dim=0).mean(dim=0)
                score_logits = _blend_with_routing(
                    candidate_routing_logits=candidate_routing_logits,
                    learned_component=learned_component,
                    policy_blend_alpha=policy_blend_alpha,
                )
        else:
            raise ValueError(f"unsupported current-route preference loss_score_mode: {mode}")

        trust_top_k = learned_component_trust_top_k
        if mode in {"transition_blend", "policy_transition_blend"} and trust_top_k is not None and int(trust_top_k) > 0:
            trusted_candidate_count = min(len(candidates), int(trust_top_k))
            if trusted_candidate_count < len(candidates):
                trust_region_active_count += 1
                trusted_targets = [idx for idx in targets if idx < trusted_candidate_count]
                trusted_avoids = [idx for idx in avoids if idx < trusted_candidate_count]
                trusted_rejected = [idx for idx in rejected if idx < trusted_candidate_count]
                if targets and not trusted_targets:
                    skipped_target_outside_trust_region += 1
                    continue
                if avoids and not trusted_avoids:
                    skipped_avoid_outside_trust_region += 1
                    continue
                trust_region_target_kept_count += int(bool(trusted_targets))
                score_logits = score_logits[:trusted_candidate_count]
                targets = trusted_targets
                avoids = trusted_avoids
                rejected = trusted_rejected

        log_probs = F.log_softmax(score_logits.float(), dim=-1)
        if targets:
            positive_sample_count += 1
            target_tensor = torch.tensor(targets, device=log_probs.device, dtype=torch.long)
            sample_loss = -torch.logsumexp(log_probs.index_select(0, target_tensor), dim=0)
            if bool(enable_pairwise_correction_loss) and rejected:
                rejected_tensor = torch.tensor(rejected, device=score_logits.device, dtype=torch.long)
                target_scores = score_logits.float().index_select(0, target_tensor)
                rejected_scores = score_logits.float().index_select(0, rejected_tensor)
                pairwise = F.relu(
                    float(pairwise_correction_margin)
                    - (target_scores.unsqueeze(1) - rejected_scores.unsqueeze(0))
                ).mean()
                sample_loss = sample_loss + float(pairwise_correction_weight) * pairwise
                pairwise_correction_sample_count += 1
                pairwise_correction_losses.append(float(pairwise.detach().cpu().item()))
        else:
            if not bool(enable_suppress_loss):
                skipped_suppress_disabled += 1
                continue
            suppress_sample_count += 1
            avoid_tensor = torch.tensor(avoids, device=log_probs.device, dtype=torch.long)
            avoid_prob = torch.softmax(score_logits.float(), dim=-1).index_select(0, avoid_tensor).sum()
            sample_loss = -torch.log1p(-avoid_prob.clamp(max=1.0 - 1.0e-6))
        base_preference_losses.append(sample_loss.detach())
        if float(stability_kl_weight) > 0.0:
            ref_log_probs = _reference_log_probs(sample, int(score_logits.numel()), log_probs.device)
            if ref_log_probs is not None:
                ref_probs = ref_log_probs.exp()
                stability_kl = (ref_probs * (ref_log_probs - log_probs.float())).sum()
                stability_kl_losses.append(stability_kl)
                sample_loss = sample_loss + float(stability_kl_weight) * stability_kl
        losses.append(float(sample_weight) * sample_loss)
    loss = torch.stack(losses).mean() if losses else torch.zeros((), device=device, requires_grad=True)
    preference_loss = (
        torch.stack(base_preference_losses).mean()
        if base_preference_losses
        else torch.zeros((), device=device)
    )
    stability_kl_loss = (
        torch.stack(stability_kl_losses).mean()
        if stability_kl_losses
        else torch.zeros((), device=device)
    )
    metrics = {
        "sample_count": len(samples),
        "decision_count": len(losses),
        "positive_sample_count": int(positive_sample_count),
        "suppress_sample_count": int(suppress_sample_count),
        "pairwise_correction_sample_count": int(pairwise_correction_sample_count),
        "pairwise_correction_loss": float(
            sum(pairwise_correction_losses) / max(1, len(pairwise_correction_losses))
        ),
        "loss": float(loss.detach().cpu().item()),
        "preference_loss": float(preference_loss.detach().cpu().item()),
        "stability_kl_weight": float(stability_kl_weight),
        "stability_kl_loss": float(stability_kl_loss.detach().cpu().item()),
        "stability_kl_sample_count": int(len(stability_kl_losses)),
        "mean_candidate_count": float(sum(candidate_counts) / max(1, len(candidate_counts))),
        "mean_target_count": float(sum(target_counts) / max(1, len(target_counts))),
        "mean_avoid_count": float(sum(avoid_counts) / max(1, len(avoid_counts))),
        "mean_sample_weight": float(sum(sample_weights) / max(1, len(sample_weights))),
        "loss_score_mode": str(loss_score_mode),
        "enable_suppress_loss": bool(enable_suppress_loss),
        "enable_pairwise_correction_loss": bool(enable_pairwise_correction_loss),
        "pairwise_correction_margin": float(pairwise_correction_margin),
        "pairwise_correction_weight": float(pairwise_correction_weight),
        "transition_blend_sample_count": int(transition_blend_count),
        "policy_transition_blend_sample_count": int(policy_transition_blend_count),
        "policy_transition_blend_transition_count": int(policy_transition_blend_transition_count),
        "policy_transition_blend_policy_fallback_count": int(policy_transition_blend_policy_fallback_count),
        "policy_component_enabled_count": int(policy_component_enabled_count),
        "policy_component_disabled_count": int(policy_component_disabled_count),
        "transition_component_enabled_count": int(transition_component_enabled_count),
        "transition_component_disabled_count": int(transition_component_disabled_count),
        "learned_component_enabled_count": int(learned_component_enabled_count),
        "learned_component_disabled_count": int(learned_component_disabled_count),
        "low_confidence_warmup_count": int(low_confidence_warmup_count),
        "learned_component_min_range": float(learned_component_min_range),
        "learned_component_trust_top_k": (
            int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
        ),
        "trust_region_active_count": int(trust_region_active_count),
        "trust_region_target_kept_count": int(trust_region_target_kept_count),
        "skipped_learned_component_unavailable_count": int(skipped_learned_component_unavailable),
        "skipped_target_outside_trust_region_count": int(skipped_target_outside_trust_region),
        "skipped_avoid_outside_trust_region_count": int(skipped_avoid_outside_trust_region),
        "skipped_transition_unavailable_count": int(skipped_transition_unavailable),
        "skipped_no_supervision_count": int(skipped_no_supervision),
        "skipped_suppress_disabled_count": int(skipped_suppress_disabled),
    }
    return loss.to(device), metrics


def _batch_for_step(samples: list[dict[str, Any]], step_idx: int, batch_size: int) -> list[dict[str, Any]]:
    start = ((int(step_idx) - 1) * max(1, int(batch_size))) % len(samples)
    return [samples[(start + offset) % len(samples)] for offset in range(max(1, int(batch_size)))]


def _model_config_payload(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }


def train_current_route_preference_with_model(
    *,
    model: Any,
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5.0e-5,
    train_transition: bool = False,
    checkpoint_metadata: dict[str, Any] | None = None,
    loss_score_mode: str = "policy_head",
    enable_suppress_loss: bool = False,
    enable_pairwise_correction_loss: bool = False,
    pairwise_correction_margin: float = 0.5,
    pairwise_correction_weight: float = 1.0,
    policy_blend_alpha: float = 0.5,
    transition_residual_lambda: float = 0.0,
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE,
    learned_component_min_range: float = 0.1,
    learned_component_trust_top_k: int | None = 80,
) -> dict[str, Any]:
    if not samples:
        return {"status": "blocked", "blocker": "no_current_route_preference_samples"}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    freeze_report = configure_current_route_preference_trainable(model, train_transition=train_transition)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return {"status": "blocked", "blocker": "no_trainable_parameters", "freeze_report": freeze_report}
    skill_ids = [
        str(skill.get("skill_id") if isinstance(skill, dict) else getattr(skill, "skill_id", idx))
        for idx, skill in enumerate(getattr(model, "skills", []))
    ]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate))
    metrics_path = output_path / "training_metrics.jsonl"
    last_metrics: dict[str, Any] = {}
    for step_idx in range(1, max(1, int(max_steps)) + 1):
        batch = _batch_for_step(samples, step_idx, batch_size)
        loss, metrics = compute_current_route_preference_batch_loss(
            model,
            batch,
            skill_id_to_idx=skill_id_to_idx,
            loss_score_mode=loss_score_mode,
            enable_suppress_loss=enable_suppress_loss,
            enable_pairwise_correction_loss=enable_pairwise_correction_loss,
            pairwise_correction_margin=pairwise_correction_margin,
            pairwise_correction_weight=pairwise_correction_weight,
            policy_blend_alpha=policy_blend_alpha,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            learned_component_min_range=learned_component_min_range,
            learned_component_trust_top_k=learned_component_trust_top_k,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_metrics = {"step": step_idx, **metrics}
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(last_metrics, ensure_ascii=False, sort_keys=True) + "\n")
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    torch.save(
        {
            "stage": "appworld_current_route_preference",
            "config": _model_config_payload(model),
            "model_state_dict": model.state_dict(),
            "train_report": {
                "sample_count": len(samples),
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "freeze_report": freeze_report,
                "checkpoint_metadata": dict(checkpoint_metadata or {}),
                "loss_score_mode": str(loss_score_mode),
                "enable_suppress_loss": bool(enable_suppress_loss),
                "enable_pairwise_correction_loss": bool(enable_pairwise_correction_loss),
                "pairwise_correction_margin": float(pairwise_correction_margin),
                "pairwise_correction_weight": float(pairwise_correction_weight),
                "policy_blend_alpha": float(policy_blend_alpha),
                "transition_residual_lambda": float(transition_residual_lambda),
                "transition_scoring_mode": str(transition_scoring_mode),
                "learned_component_min_range": float(learned_component_min_range),
                "learned_component_trust_top_k": (
                    int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
                ),
            },
        },
        checkpoint_path,
    )
    report = {
        "status": "ok",
        "sample_count": len(samples),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "freeze_report": freeze_report,
        "loss_score_mode": str(loss_score_mode),
        "enable_suppress_loss": bool(enable_suppress_loss),
        "enable_pairwise_correction_loss": bool(enable_pairwise_correction_loss),
        "pairwise_correction_margin": float(pairwise_correction_margin),
        "pairwise_correction_weight": float(pairwise_correction_weight),
        "policy_blend_alpha": float(policy_blend_alpha),
        "transition_residual_lambda": float(transition_residual_lambda),
        "transition_scoring_mode": str(transition_scoring_mode),
        "learned_component_min_range": float(learned_component_min_range),
        "learned_component_trust_top_k": (
            int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
        ),
        "last_metrics": last_metrics,
        "training_metrics_path": str(metrics_path),
        "latest_checkpoint": str(checkpoint_path),
    }
    (output_path / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
