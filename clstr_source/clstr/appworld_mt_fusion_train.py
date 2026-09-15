from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clstr.appworld_clstr_eval import _load_checkpoint_model
from clstr.appworld_routing import read_jsonl, write_json
from clstr.belief import subspace_obs


def _skill_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("skill_id") or row.get("id") or idx) for idx, row in enumerate(rows)]


def iter_trajectory_batches(trajectories: list[dict[str, Any]], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(trajectories), size):
        yield trajectories[start : start + size]


def _candidate_indices(
    *,
    routing_logits: torch.Tensor,
    positive_indices: list[int],
    candidate_top_k: int,
) -> list[int]:
    k = min(max(1, int(candidate_top_k)), int(routing_logits.numel()))
    candidates = [int(idx) for idx in torch.topk(routing_logits, k=k).indices.detach().cpu().tolist()]
    for positive in positive_indices:
        if positive in candidates:
            continue
        if len(candidates) < k:
            candidates.append(int(positive))
        else:
            candidates[-1] = int(positive)
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


def _candidate_embeddings(model: Any, state_text: str, candidates: list[int]) -> torch.Tensor:
    if callable(getattr(model, "batch_cross_encode", None)):
        try:
            return model.batch_cross_encode([state_text], [candidates])
        except RuntimeError:
            pass
    idx = torch.tensor(candidates, dtype=torch.long, device=model.device)
    return model.skill_table.E.index_select(0, idx).unsqueeze(0)


def _first_skill_index(step: dict[str, Any], skill_to_idx: dict[str, int], fallback: int) -> int:
    for key in ("selected_skill_ids", "positive_skill_ids"):
        for skill_id in step.get(key, []) or []:
            idx = skill_to_idx.get(str(skill_id))
            if idx is not None:
                return int(idx)
    return int(fallback)


def multistep_oracle_policy_loss(
    *,
    model: Any,
    trajectories: list[dict[str, Any]],
    skill_ids: list[str],
    candidate_top_k: int = 16,
    detach_belief_between_steps: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    skill_to_idx = {str(skill_id): idx for idx, skill_id in enumerate(skill_ids)}
    losses: list[torch.Tensor] = []
    supervised_steps = 0
    positive_in_candidates = 0
    skipped_without_positive = 0
    skipped_without_candidates = 0

    for trajectory in trajectories:
        m_t: torch.Tensor | None = None
        for step in trajectory.get("steps", []) or []:
            positive_indices = [
                skill_to_idx[str(skill_id)]
                for skill_id in step.get("positive_skill_ids", []) or []
                if str(skill_id) in skill_to_idx
            ]
            if not positive_indices:
                skipped_without_positive += 1
                continue

            state_text = str(step.get("state_text") or "")
            h_t = model.encode_states([state_text])
            if m_t is None:
                m_t = subspace_obs(model.skill_table, h_t)
            routing_logits = model.skill_table.retrieval_logits(h_t).squeeze(0)
            candidates = _candidate_indices(
                routing_logits=routing_logits,
                positive_indices=positive_indices,
                candidate_top_k=candidate_top_k,
            )
            if not candidates:
                skipped_without_candidates += 1
                continue
            local_positive = [idx for idx, candidate in enumerate(candidates) if candidate in set(positive_indices)]
            if not local_positive:
                skipped_without_candidates += 1
                continue
            positive_in_candidates += 1
            candidate_embs = _candidate_embeddings(model, state_text, candidates)
            candidate_idx = torch.tensor(candidates, dtype=torch.long, device=routing_logits.device)
            candidate_routing_logits = routing_logits.index_select(0, candidate_idx)
            logits = model.policy_forward(
                h_t,
                m_t,
                candidate_embs,
                routing_logits=candidate_routing_logits,
            ).squeeze(0)[: len(candidates)]
            pos = torch.tensor(local_positive, dtype=torch.long, device=logits.device)
            losses.append(torch.logsumexp(logits, dim=0) - torch.logsumexp(logits.index_select(0, pos), dim=0))
            supervised_steps += 1

            action_idx = _first_skill_index(step, skill_to_idx, fallback=len(skill_ids))
            obs_text = str(step.get("execute_output") or "")
            a_t = torch.tensor([action_idx], dtype=torch.long, device=model.device)
            o_t = model.encode_observations([obs_text])
            m_t, _m_hat, _m_obs = model.step_update(m_t, a_t, o_t, [obs_text])
            if detach_belief_between_steps:
                m_t = m_t.detach()

    if losses:
        loss = torch.stack(losses).mean()
    else:
        device = getattr(model, "device", torch.device("cpu"))
        loss = torch.zeros((), device=device, requires_grad=True)
    stats = {
        "trajectory_count": len(trajectories),
        "supervised_steps": supervised_steps,
        "positive_in_candidates": positive_in_candidates,
        "skipped_without_positive": skipped_without_positive,
        "skipped_without_candidates": skipped_without_candidates,
        "candidate_top_k": int(candidate_top_k),
        "loss": float(loss.detach().cpu().item()),
    }
    return loss, stats


def train_appworld_mt_fusion_from_trajectories(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    skill_pool_path: str | Path,
    trajectories_path: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    learning_rate: float = 5.0e-5,
    weight_decay: float = 0.0,
    candidate_top_k: int = 16,
    max_trajectories: int | None = None,
    trajectory_batch_size: int = 8,
    detach_belief_between_steps: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    model, load_report = _load_checkpoint_model(
        model_config_path=model_config_path,
        skill_pool_path=skill_pool_path,
        checkpoint_path=checkpoint_path,
    )
    skill_rows = read_jsonl(skill_pool_path)
    skill_ids = _skill_ids_from_rows(skill_rows)
    trajectories = read_jsonl(trajectories_path)
    if max_trajectories is not None:
        trajectories = trajectories[: int(max_trajectories)]

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    epoch_stats: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.train()
        epoch_summary = {
            "epoch": epoch + 1,
            "trajectory_count": len(trajectories),
            "batch_count": 0,
            "supervised_steps": 0,
            "positive_in_candidates": 0,
            "skipped_without_positive": 0,
            "skipped_without_candidates": 0,
            "candidate_top_k": int(candidate_top_k),
            "trajectory_batch_size": int(trajectory_batch_size),
            "loss": 0.0,
        }
        weighted_loss_sum = 0.0
        weighted_loss_count = 0
        for batch in iter_trajectory_batches(trajectories, trajectory_batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss, stats = multistep_oracle_policy_loss(
                model=model,
                trajectories=batch,
                skill_ids=skill_ids,
                candidate_top_k=candidate_top_k,
                detach_belief_between_steps=detach_belief_between_steps,
            )
            if int(stats["supervised_steps"]) > 0:
                loss.backward()
                optimizer.step()
            epoch_summary["batch_count"] += 1
            for key in ("supervised_steps", "positive_in_candidates", "skipped_without_positive", "skipped_without_candidates"):
                epoch_summary[key] += int(stats[key])
            weight = int(stats["supervised_steps"])
            if weight > 0:
                weighted_loss_sum += float(stats["loss"]) * weight
                weighted_loss_count += weight
        if weighted_loss_count:
            epoch_summary["loss"] = weighted_loss_sum / weighted_loss_count
        epoch_stats.append(epoch_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = output_dir / "checkpoints" / "appworld_mt_fusion.pt"
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config_path": str(model_config_path),
            "base_checkpoint_path": str(checkpoint_path),
            "trajectories_path": str(trajectories_path),
            "epoch_stats": epoch_stats,
        },
        checkpoint_out,
    )
    report = {
        "status": "ok" if epoch_stats and epoch_stats[-1]["supervised_steps"] else "empty",
        "method": "appworld_mt_fusion_oracle_supervised",
        "model_config_path": str(model_config_path),
        "base_checkpoint_path": str(checkpoint_path),
        "skill_pool_path": str(skill_pool_path),
        "trajectories_path": str(trajectories_path),
        "output_checkpoint": str(checkpoint_out),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "candidate_top_k": int(candidate_top_k),
        "trajectory_batch_size": int(trajectory_batch_size),
        "trajectory_count": len(trajectories),
        "epoch_stats": epoch_stats,
        "checkpoint_load": load_report,
        "caveat": "Offline train-only oracle API trace supervision for mt_fusion; not final AppWorld closed-loop evidence.",
    }
    write_json(output_dir / "report.json", report)
    return report
