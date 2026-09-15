from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from clstr.full_base_train import (
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    STATE_QUERY_ROLE,
    TRANSITION_SCORING_MODE,
    TRANSITION_TEXT_ROLE,
    _batch_action_text_embedding_or_none,
    _batch_cached_or_encode,
    _equivalent_skill_ids_by_skill_id,
    _filter_transition_candidates_by_inventory,
    _ranking_metrics_from_logits,
    _rank_delta_metrics_vs_prior,
    _row_candidate_indices,
    _stage0_candidate_prior_scores_tensor,
    _transition_candidate_logits_for_mode,
    _transition_positive_mask,
)


def add_candidate_memory_scores(
    logits: torch.Tensor,
    *,
    candidate_rows: list[list[int]],
    memory_scores_by_skill_id: list[dict[str, float]],
    skill_ids_by_idx: dict[int, str],
    weight: float,
) -> torch.Tensor:
    if float(weight) == 0.0 or not candidate_rows:
        return logits
    memory = torch.zeros_like(logits)
    for row_idx, candidate_indices in enumerate(candidate_rows):
        row_scores = memory_scores_by_skill_id[row_idx] if row_idx < len(memory_scores_by_skill_id) else {}
        for local_idx, skill_idx in enumerate(candidate_indices):
            if local_idx >= memory.size(1):
                break
            skill_id = str(skill_ids_by_idx.get(int(skill_idx), ""))
            memory[row_idx, local_idx] = float(row_scores.get(skill_id, 0.0))
    return logits + float(weight) * memory


def strict_metrics_from_retained(
    metrics: dict[str, Any],
    *,
    retained_rows: int,
    source_rows: int,
) -> dict[str, float]:
    source_rows = max(0, int(source_rows))
    retained_rows = max(0, int(retained_rows))
    scale = retained_rows / source_rows if source_rows > 0 else 0.0
    return {
        "strict_transition_skill_recall@1": float(metrics.get("transition_skill_recall@1") or 0.0) * scale,
        "strict_transition_skill_recall@5": float(metrics.get("transition_skill_recall@5") or 0.0) * scale,
        "strict_transition_skill_mrr": float(metrics.get("transition_skill_mrr") or 0.0) * scale,
        "retained_row_fraction": scale,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
        "retained_transition_candidate_count": float(metrics.get("transition_skill_ce_candidate_count") or 0.0),
    }


def _pad_candidate_rows(candidate_rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_width = max(len(row) for row in candidate_rows)
    candidate_ids = torch.zeros(len(candidate_rows), max_width, dtype=torch.long, device=device)
    valid_mask = torch.zeros(len(candidate_rows), max_width, dtype=torch.bool, device=device)
    for row_idx, row in enumerate(candidate_rows):
        width = len(row)
        candidate_ids[row_idx, :width] = torch.tensor(row, dtype=torch.long, device=device)
        valid_mask[row_idx, :width] = True
    return candidate_ids, valid_mask


def _memory_scores_by_skill_id(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for row in rows:
        candidate_ids = [str(item) for item in row.get("candidate_next_skill_ids") or row.get("stage0_next_candidate_skill_ids") or []]
        scores = [float(item) for item in row.get("candidate_next_online_memory_scores") or []]
        output.append({skill_id: scores[idx] for idx, skill_id in enumerate(candidate_ids) if idx < len(scores)})
    return output


def evaluate_stage2_memory_rerank_batch(
    model: Any,
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    skill_ids_by_idx: dict[int, str],
    skills: list[dict[str, Any]],
    device: torch.device,
    *,
    transition_inventory_mask_mode: str = "auto",
    transition_inventory_min_candidates: int = 64,
    transition_positive_mode: str = "gold_plus_equivalent",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    online_memory_weight: float = 1.0,
) -> dict[str, Any]:
    if not rows:
        return {"stage2_memory_count": 0.0}
    eval_rows = [
        row
        for row in rows
        if (not row.get("loss_mask") or (row.get("loss_mask") or {}).get("L_trans_skill_ce"))
        and str(row.get("next_skill_id") or "") in skill_id_to_idx
    ]
    if not eval_rows:
        return {"stage2_memory_count": 0.0}
    h = _batch_cached_or_encode(
        model,
        eval_rows,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    initial_belief = getattr(model, "initial_belief", None)
    if not callable(initial_belief):
        raise ValueError("stage2 memory-rerank evaluation requires model.initial_belief")
    m_obs = initial_belief(h)
    obs_emb = _batch_cached_or_encode(
        model,
        eval_rows,
        "_next_observation_embedding",
        "next_observation_text",
        device,
        text_role=TRANSITION_TEXT_ROLE,
    )
    action_h = _batch_action_text_embedding_or_none(model, eval_rows, device)
    current_labels = torch.tensor(
        [skill_id_to_idx.get(str(row.get("skill_id")), 0) for row in eval_rows],
        dtype=torch.long,
        device=device,
    )
    next_labels_global = torch.tensor(
        [skill_id_to_idx[str(row.get("next_skill_id"))] for row in eval_rows],
        dtype=torch.long,
        device=device,
    )
    next_label_values = [int(label) for label in next_labels_global.detach().cpu().tolist()]
    candidate_rows = [
        _row_candidate_indices(row, "stage0_next_candidate_skill_indices", len(skill_id_to_idx))
        or _row_candidate_indices(row, "candidate_next_skill_indices", len(skill_id_to_idx))
        or _row_candidate_indices(row, "stage0_candidate_skill_indices", len(skill_id_to_idx))
        for row in eval_rows
    ]
    use_candidates = bool(candidate_rows) and all(candidate_rows) and all(
        label in row for label, row in zip(next_label_values, candidate_rows)
    )
    if not use_candidates:
        return {"stage2_memory_count": 0.0, "stage2_memory_blocker": "missing_stage0_candidate_rows"}

    candidate_rows, inventory_audit = _filter_transition_candidates_by_inventory(
        rows=eval_rows,
        candidate_rows=candidate_rows,
        labels=next_labels_global,
        skill_ids_by_idx=skill_ids_by_idx,
        mode=transition_inventory_mask_mode,
        min_candidates=transition_inventory_min_candidates,
    )
    use_candidates = all(row and label in row for label, row in zip(next_label_values, candidate_rows))
    if not use_candidates:
        return {"stage2_memory_count": 0.0, "stage2_memory_blocker": "positive_removed_by_inventory_mask"}

    candidate_ids, candidate_valid_mask = _pad_candidate_rows(candidate_rows, device)
    next_labels = torch.tensor(
        [row.index(label) for row, label in zip(candidate_rows, next_label_values)],
        dtype=torch.long,
        device=device,
    )
    logits, head_type, prior_logits, residual_logits = _transition_candidate_logits_for_mode(
        model,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_h,
        len(skill_id_to_idx),
        candidate_ids=candidate_ids,
        candidate_valid_mask=candidate_valid_mask,
        candidate_stage0_prior_scores=_stage0_candidate_prior_scores_tensor(
            eval_rows,
            candidate_rows,
            width=int(candidate_ids.size(1)),
            device=device,
            dtype=h.dtype,
        ),
        residual_lambda=transition_residual_lambda,
        scoring_mode=transition_scoring_mode,
    )
    equivalent = _equivalent_skill_ids_by_skill_id(skills)
    positive_mask, positive_counts = _transition_positive_mask(
        rows=eval_rows,
        candidate_rows=candidate_rows,
        labels=next_labels,
        skill_ids_by_idx=skill_ids_by_idx,
        positive_mode=transition_positive_mode,
        device=logits.device,
        equivalent_skill_ids_by_skill_id=equivalent,
    )
    memory_logits = add_candidate_memory_scores(
        logits,
        candidate_rows=candidate_rows,
        memory_scores_by_skill_id=_memory_scores_by_skill_id(eval_rows),
        skill_ids_by_idx=skill_ids_by_idx,
        weight=online_memory_weight,
    )
    stage2_metrics = _ranking_metrics_from_logits(logits, next_labels, rows=eval_rows, positive_mask=positive_mask)
    memory_metrics = _ranking_metrics_from_logits(memory_logits, next_labels, rows=eval_rows, positive_mask=positive_mask)
    prior_metrics = _ranking_metrics_from_logits(prior_logits, next_labels, rows=eval_rows, positive_mask=positive_mask)
    delta_vs_stage2 = _rank_delta_metrics_vs_prior(memory_logits, logits, next_labels, positive_mask=positive_mask)
    memory_hit_rows = sum(
        1 for item in _memory_scores_by_skill_id(eval_rows) if any(float(score) != 0.0 for score in item.values())
    )
    return {
        "stage2_memory_count": float(len(eval_rows)),
        "transition_skill_head_type": head_type,
        "transition_inventory_mask_mode": transition_inventory_mask_mode,
        "transition_inventory_mask_applied_rows": float(inventory_audit["inventory_mask_applied_rows"]),
        "transition_inventory_mask_removed_candidates": float(inventory_audit["inventory_mask_removed_candidates"]),
        "transition_inventory_mask_missing_rows": float(inventory_audit["inventory_mask_missing_rows"]),
        "transition_inventory_mask_positive_missing_rows": float(inventory_audit["inventory_mask_positive_missing_rows"]),
        "transition_inventory_mask_backfilled_rows": float(inventory_audit["inventory_mask_backfilled_rows"]),
        "transition_inventory_mask_backfilled_candidates": float(inventory_audit["inventory_mask_backfilled_candidates"]),
        "stage2_memory_online_memory_hit_rows": float(memory_hit_rows),
        **positive_counts,
        **{f"stage2_{key}": value for key, value in stage2_metrics.items()},
        **{f"stage2_plus_memory_{key}": value for key, value in memory_metrics.items()},
        **{f"stage0_prior_{key}": value for key, value in prior_metrics.items()},
        **{f"stage2_plus_memory_delta_vs_stage2_{key.removeprefix('transition_delta_vs_stage0_prior_')}": value for key, value in delta_vs_stage2.items()},
    }


def aggregate_stage2_memory_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for row in metric_rows:
        weight = int(row.get("batch_size") or row.get("stage2_memory_count") or 0)
        for key, value in (row.get("metrics") or row).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key == "stage2_memory_count":
                continue
            sums[str(key)] += float(value) * max(1, weight)
            weights[str(key)] += max(1, weight)
    return {
        key: float(sums[key] / weights[key])
        for key in sorted(sums)
        if weights[key] > 0
    }
