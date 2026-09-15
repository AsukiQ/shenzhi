from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from clstr.candidate_admission_residual import score_candidate_admission_residual
from clstr.counterfactual_memory_calibration import score_cmc_candidates
from clstr.full_base_train import (
    STATE_QUERY_ROLE,
    UNIFIED_MEMORY_ROUTE_SCORER,
    _apply_replay_prefix_beliefs,
    _batch_cached_or_encode,
    _dynamic_static_rank_metrics,
    _multi_positive_listwise_nll,
)
from clstr.memory_candidate_recall import (
    CANDIDATE_RECALL_MODES,
    CandidateUnion,
    build_static_dynamic_union,
    candidate_provenance_mask,
    candidate_recall_report,
    full_pool_positive_mask,
    legal_skill_pool_mask,
    stable_masked_topk_rows,
)
from clstr.memory_utility_gate import (
    HEURISTIC_ALPHA_VERSION,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    RELIABILITY_MODES,
    effective_memory_alpha,
    fuse_route_scores,
    heuristic_memory_alpha,
    memory_utility_features,
)
from clstr.stage4_act_train import (
    _pad_candidate_indices,
    _stage4_skill_id_to_idx,
)
from clstr.safe_memory_ranking import (
    bounded_memory_fusion,
    candidate_provenance_positive_residual_fusion,
)


@dataclass
class CurrentStateRouteBatchOutput:
    loss: torch.Tensor
    metrics: dict[str, Any]
    static_full_logits: torch.Tensor
    raw_dynamic_full_logits: torch.Tensor
    dynamic_full_logits: torch.Tensor
    legal_mask: torch.Tensor
    positive_mask: torch.Tensor
    candidate_union: CandidateUnion
    static_candidate_logits: torch.Tensor
    dynamic_candidate_logits: torch.Tensor
    fused_candidate_logits: torch.Tensor
    candidate_valid_mask: torch.Tensor
    candidate_positive_mask: torch.Tensor
    features: torch.Tensor
    raw_alpha: torch.Tensor
    effective_alpha: torch.Tensor
    static_memory: torch.Tensor
    dynamic_memory: torch.Tensor
    causal_update_count: torch.Tensor


def _validate_full_skill_mapping(
    skill_id_to_idx: dict[str, int] | None,
    *,
    skill_count: int,
) -> dict[str, int]:
    if not isinstance(skill_id_to_idx, dict) or not skill_id_to_idx:
        raise ValueError("candidate-union current-state routing requires the complete skill_id_to_idx mapping")
    normalized = {str(skill_id): int(idx) for skill_id, idx in skill_id_to_idx.items()}
    if len(normalized) != int(skill_count):
        raise ValueError("skill_id_to_idx size must match model.skill_table.E")
    if sorted(normalized.values()) != list(range(int(skill_count))):
        raise ValueError("skill_id_to_idx indices must be contiguous in declared skill-table order")
    return normalized


def _causal_update_counts(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    values: list[float] = []
    for row in rows:
        prefix = row.get("replay_prefix")
        if not isinstance(prefix, list) or not prefix:
            values.append(0.0)
            continue
        first = prefix[0] if isinstance(prefix[0], dict) else {}
        if not str(first.get("observation_text") or "").strip():
            values.append(0.0)
            continue
        values.append(
            float(
                sum(
                    1
                    for step in prefix
                    if isinstance(step, dict) and str(step.get("action_text") or "").strip()
                )
            )
        )
    return torch.tensor(values, dtype=torch.float32, device=device)


def _candidate_masks(
    candidate_rows: list[list[int]],
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in candidate_rows), default=0)
    valid = torch.zeros(
        (len(candidate_rows), width),
        dtype=torch.bool,
        device=positive_mask.device,
    )
    local_positive = torch.zeros_like(valid)
    skill_count = int(positive_mask.size(1))
    for row_idx, row in enumerate(candidate_rows):
        normalized = [int(idx) for idx in row]
        if any(idx < 0 or idx >= skill_count for idx in normalized):
            raise ValueError("candidate union contains an index outside the declared skill table")
        if normalized:
            ids = torch.tensor(normalized, dtype=torch.long, device=positive_mask.device)
            valid[row_idx, : len(normalized)] = True
            local_positive[row_idx, : len(normalized)] = positive_mask[row_idx].index_select(0, ids)
    return valid, local_positive


def _candidate_embeddings(
    skill_embeddings: torch.Tensor,
    candidate_rows: list[list[int]],
    *,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.zeros(
        (len(candidate_rows), int(width)),
        dtype=torch.long,
        device=device,
    )
    for row_idx, row in enumerate(candidate_rows):
        if row:
            ids[row_idx, : len(row)] = torch.tensor(
                row,
                dtype=torch.long,
                device=device,
            )
    embeddings = skill_embeddings.detach().to(device=device, dtype=dtype)
    return embeddings.index_select(0, ids.reshape(-1)).reshape(
        len(candidate_rows),
        int(width),
        -1,
    )


def _strict_ranking_metrics(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    row_count = int(logits.size(0))
    if row_count == 0:
        return {
            "stage4_next_skill_recall@1": 0.0,
            "stage4_next_skill_recall@5": 0.0,
            "stage4_next_skill_mrr": 0.0,
        }
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~valid, floor)
    best_positive = masked.masked_fill(~positive, floor).max(dim=-1).values
    has_positive = positive.any(dim=-1)
    ranks = ((masked > best_positive.unsqueeze(1)) & valid).sum(dim=-1) + 1
    reciprocal = torch.where(
        has_positive,
        1.0 / ranks.to(torch.float32),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    return {
        "stage4_next_skill_recall@1": float(
            (has_positive & (ranks <= 1)).float().mean().detach().cpu().item()
        ),
        "stage4_next_skill_recall@5": float(
            (has_positive & (ranks <= 5)).float().mean().detach().cpu().item()
        ),
        "stage4_next_skill_mrr": float(reciprocal.mean().detach().cpu().item()),
    }


def _flatten_candidate_recall_metrics(report: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        "candidate_recall_memory_active_coverage": report["memory_active_coverage"],
        "candidate_recall_positive_not_in_declared_pool_rows": report[
            "positive_not_in_declared_pool_rows"
        ],
        "candidate_recall_positive_outside_legal_pool_rows": report[
            "positive_outside_legal_pool_rows"
        ],
        "candidate_union_version": report["candidate_union_version"],
        "candidate_selection_version": report["candidate_selection_version"],
        "candidate_tie_break_policy": report["tie_break_policy"],
    }
    for report_key, prefix in (
        ("all_eligible_source_strict", "candidate_recall_all_"),
        ("memory_active_strict", "candidate_recall_memory_active_"),
    ):
        for key, value in report[report_key].items():
            flattened[f"{prefix}{key}"] = value
    return flattened


def _build_current_state_route_batch(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    skill_id_to_idx: dict[str, int] | None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    static_k: int,
    dynamic_extra_k: int,
    final_k: int,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate: torch.nn.Module | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    safe_memory_residual_bound: float = 2.0,
) -> CurrentStateRouteBatchOutput:
    if not rows:
        raise ValueError("candidate-union current-state routing requires at least one row")
    if int(static_k) < 0 or int(dynamic_extra_k) < 0 or int(final_k) <= 0:
        raise ValueError("candidate budgets must be nonnegative and final_k must be positive")
    if int(final_k) > int(static_k):
        raise ValueError("final_k must not exceed static_k for exact static fallback")
    reliability_mode = str(reliability_mode or "dynamic")
    if reliability_mode not in RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    full_logits_fn = getattr(model, "unified_route_full_logits", None)
    gather_logits_fn = getattr(model, "gather_unified_route_logits", None)
    if not callable(full_logits_fn) or not callable(gather_logits_fn):
        raise ValueError(
            "candidate-union current-state routing requires full unified logits and gather APIs"
        )
    skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
    if not isinstance(skill_embeddings, torch.Tensor) or skill_embeddings.ndim != 2:
        raise ValueError("candidate-union current-state routing requires model.skill_table.E")
    skill_count = int(skill_embeddings.size(0))
    mapping = _validate_full_skill_mapping(skill_id_to_idx, skill_count=skill_count)

    h_t = _batch_cached_or_encode(
        model,
        rows,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    static_memory = model.initial_belief(h_t)
    dynamic_memory, replay_used = _apply_replay_prefix_beliefs(
        model,
        rows,
        static_memory,
        mapping,
        skill_count,
        device,
        trainable=False,
    )
    causal_update_count = _causal_update_counts(rows, device)
    static_full = full_logits_fn(h_t, static_memory)
    raw_dynamic_full = full_logits_fn(h_t, dynamic_memory)
    legal_pool = legal_skill_pool_mask(
        rows,
        mapping,
        skill_count=skill_count,
        device=device,
    )
    dynamic_full = raw_dynamic_full
    cmc_scoring = None
    if reliability_mode == "candidate_admission_residual":
        adapter = getattr(model, "route_memory_residual_adapter", None)
        head = getattr(model, "route_memory_candidate_admission_residual", None)
        if not callable(adapter) or not callable(head):
            raise ValueError(
                "candidate-admission routing requires frozen CMC adapter and candidate head"
            )
        route_residual = adapter(h_t, dynamic_memory - static_memory)
        dynamic_full = raw_dynamic_full + route_residual @ skill_embeddings.detach().to(
            device=raw_dynamic_full.device,
            dtype=raw_dynamic_full.dtype,
        ).t()
    elif reliability_mode in {"cmc", "cmc_candidate_provenance"}:
        cmc_scoring = score_cmc_candidates(
            model,
            h=h_t,
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
            static_logits=static_full,
            raw_dynamic_logits=raw_dynamic_full,
            candidate_embeddings=skill_embeddings,
            valid_mask=legal_pool.mask,
            causal_update_count=causal_update_count,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
        )
        dynamic_full = cmc_scoring.dynamic_logits
    expected_shape = (len(rows), skill_count)
    if tuple(static_full.shape) != expected_shape or tuple(dynamic_full.shape) != expected_shape:
        raise ValueError("full unified logits must match rows and declared skill table")

    positive = full_pool_positive_mask(
        rows,
        mapping,
        equivalent_skill_ids_by_skill_id,
        skill_count=skill_count,
        device=device,
    )
    retrieval_union = build_static_dynamic_union(
        static_full,
        dynamic_full,
        legal_pool.mask,
        static_k=int(static_k),
        dynamic_extra_k=int(dynamic_extra_k),
    )
    static_candidates = gather_logits_fn(static_full, retrieval_union.candidate_rows)
    dynamic_candidates = gather_logits_fn(dynamic_full, retrieval_union.candidate_rows)
    candidate_valid, candidate_positive = _candidate_masks(
        retrieval_union.candidate_rows,
        positive & legal_pool.mask,
    )
    floor = torch.finfo(dynamic_candidates.dtype).min
    dynamic_masked = dynamic_candidates.masked_fill(~candidate_valid, floor)
    static_masked = static_candidates.masked_fill(~candidate_valid, floor)
    features = (
        cmc_scoring.features
        if cmc_scoring is not None
        else memory_utility_features(
            static_masked,
            dynamic_masked,
            candidate_valid,
            static_memory,
            dynamic_memory,
            causal_update_count,
            update_count_cap=feature_update_count_cap,
            candidate_count_cap=feature_candidate_count_cap,
        )
    )
    candidate_scoring = None
    if reliability_mode == "candidate_admission_residual":
        candidate_scoring = score_candidate_admission_residual(
            model.route_memory_candidate_admission_residual,
            h=h_t,
            memory_delta=dynamic_memory - static_memory,
            candidate_embeddings=_candidate_embeddings(
                skill_embeddings,
                retrieval_union.candidate_rows,
                width=int(static_masked.size(1)),
                dtype=static_masked.dtype,
                device=static_masked.device,
            ),
            static_logits=static_masked,
            dynamic_logits=dynamic_masked,
            dynamic_extra_mask=candidate_provenance_mask(
                retrieval_union.candidate_rows,
                retrieval_union.dynamic_extra_rows,
                candidate_valid,
            ),
            valid_mask=candidate_valid,
            causal_update_count=causal_update_count,
        )
        raw_alpha = (
            candidate_scoring.admission_probability
            .masked_fill(~candidate_valid, 0.0)
            .sum(dim=-1)
            / candidate_valid.sum(dim=-1).clamp_min(1)
        )
    elif reliability_mode == "static":
        raw_alpha = dynamic_masked.new_zeros(len(rows))
    elif reliability_mode == "dynamic":
        raw_alpha = dynamic_masked.new_ones(len(rows))
    elif reliability_mode == "fixed_alpha":
        raw_alpha = dynamic_masked.new_full((len(rows),), float(fixed_alpha))
    elif reliability_mode == "heuristic":
        raw_alpha = heuristic_memory_alpha(features).to(
            device=dynamic_masked.device,
            dtype=dynamic_masked.dtype,
        )
    elif reliability_mode == "learned":
        if memory_utility_gate is None or not callable(memory_utility_gate):
            raise ValueError("learned reliability mode requires a supplied trained memory utility gate")
        raw_alpha = memory_utility_gate(features)
        if not isinstance(raw_alpha, torch.Tensor):
            raise ValueError("learned memory utility gate must return a tensor")
        raw_alpha = raw_alpha.to(device=dynamic_masked.device, dtype=dynamic_masked.dtype)
    elif reliability_mode == "cmc":
        if cmc_scoring is None:
            raise AssertionError("CMC scoring output is missing")
        raw_alpha = (
            memory_utility_gate(features)
            if memory_utility_gate is not None
            else cmc_scoring.raw_alpha
        )
        if not isinstance(raw_alpha, torch.Tensor):
            raise ValueError("CMC memory utility gate must return a tensor")
        raw_alpha = raw_alpha.to(
            device=dynamic_masked.device,
            dtype=dynamic_masked.dtype,
        )
    elif reliability_mode == "cmc_candidate_provenance":
        if cmc_scoring is None:
            raise AssertionError("CMC scoring output is missing")
        raw_alpha = dynamic_masked.new_ones(len(rows))
    else:
        route_alpha_fn = getattr(model, "route_memory_alpha", None)
        if not callable(route_alpha_fn):
            raise ValueError("causal_gate reliability mode requires model.route_memory_alpha")
        raw_alpha = route_alpha_fn(
            h_t,
            static_memory,
            dynamic_memory,
            causal_update_count,
        ).to(device=dynamic_masked.device, dtype=dynamic_masked.dtype)
    effective_alpha = effective_memory_alpha(raw_alpha, causal_update_count)
    if reliability_mode == "candidate_admission_residual":
        if candidate_scoring is None:
            raise AssertionError("candidate-admission scoring output is missing")
        fused_masked = candidate_scoring.final_logits
    elif reliability_mode == "cmc_candidate_provenance":
        fused_masked = candidate_provenance_positive_residual_fusion(
            static_masked,
            dynamic_masked,
            candidate_provenance_mask(
                retrieval_union.candidate_rows,
                retrieval_union.dynamic_extra_rows,
                candidate_valid,
            ),
            candidate_valid,
            causal_update_count,
            residual_bound=safe_memory_residual_bound,
        )
    elif reliability_mode == "causal_gate":
        fused_masked = bounded_memory_fusion(
            static_masked,
            dynamic_masked,
            effective_alpha,
            candidate_valid,
            residual_bound=safe_memory_residual_bound,
        )
    else:
        fused_masked = fuse_route_scores(
            static_masked,
            dynamic_masked,
            effective_alpha,
            candidate_valid,
        )
    selected_positions = stable_masked_topk_rows(
        fused_masked,
        candidate_valid,
        k=int(final_k),
    )
    final_candidate_rows = [
        [retrieval_union.candidate_rows[row_idx][position] for position in positions]
        for row_idx, positions in enumerate(selected_positions)
    ]
    union = CandidateUnion(
        candidate_rows=final_candidate_rows,
        static_rows=retrieval_union.static_rows,
        dynamic_top_rows=retrieval_union.dynamic_top_rows,
        dynamic_extra_rows=retrieval_union.dynamic_extra_rows,
        static_equal_budget_rows=retrieval_union.static_equal_budget_rows,
    )
    static_candidates = gather_logits_fn(static_full, final_candidate_rows)
    dynamic_candidates = gather_logits_fn(dynamic_full, final_candidate_rows)
    candidate_valid, candidate_positive = _candidate_masks(
        final_candidate_rows,
        positive & legal_pool.mask,
    )
    floor = torch.finfo(dynamic_candidates.dtype).min
    dynamic_masked = dynamic_candidates.masked_fill(~candidate_valid, floor)
    static_masked = static_candidates.masked_fill(~candidate_valid, floor)
    if reliability_mode == "candidate_admission_residual":
        final_scoring = score_candidate_admission_residual(
            model.route_memory_candidate_admission_residual,
            h=h_t,
            memory_delta=dynamic_memory - static_memory,
            candidate_embeddings=_candidate_embeddings(
                skill_embeddings,
                final_candidate_rows,
                width=int(static_masked.size(1)),
                dtype=static_masked.dtype,
                device=static_masked.device,
            ),
            static_logits=static_masked,
            dynamic_logits=dynamic_masked,
            dynamic_extra_mask=candidate_provenance_mask(
                final_candidate_rows,
                retrieval_union.dynamic_extra_rows,
                candidate_valid,
            ),
            valid_mask=candidate_valid,
            causal_update_count=causal_update_count,
        )
        fused_masked = final_scoring.final_logits
    elif reliability_mode == "cmc_candidate_provenance":
        fused_masked = candidate_provenance_positive_residual_fusion(
            static_masked,
            dynamic_masked,
            candidate_provenance_mask(
                final_candidate_rows,
                retrieval_union.dynamic_extra_rows,
                candidate_valid,
            ),
            candidate_valid,
            causal_update_count,
            residual_bound=safe_memory_residual_bound,
        )
    elif reliability_mode == "causal_gate":
        fused_masked = bounded_memory_fusion(
            static_masked,
            dynamic_masked,
            effective_alpha,
            candidate_valid,
            residual_bound=safe_memory_residual_bound,
        )
    else:
        fused_masked = fuse_route_scores(
            static_masked,
            dynamic_masked,
            effective_alpha,
            candidate_valid,
        )
    eligible = candidate_positive.any(dim=-1) & (candidate_valid & ~candidate_positive).any(dim=-1)
    loss = (
        _multi_positive_listwise_nll(
            fused_masked[eligible],
            candidate_positive[eligible],
        )
        if bool(eligible.any())
        else dynamic_full.new_zeros(())
    )

    fused_metrics = _strict_ranking_metrics(fused_masked, candidate_positive, candidate_valid)
    dynamic_metrics = _strict_ranking_metrics(dynamic_masked, candidate_positive, candidate_valid)
    static_metrics = _strict_ranking_metrics(static_masked, candidate_positive, candidate_valid)
    recall_report = candidate_recall_report(
        retrieval_union,
        positive,
        legal_pool.mask,
        causal_update_count,
    )
    metrics: dict[str, Any] = {
        **fused_metrics,
        **_flatten_candidate_recall_metrics(recall_report),
        "stage4_act_loss": float(loss.detach().cpu().item()),
        "stage4_total_loss": float(loss.detach().cpu().item()),
        "stage4_act_count": float(len(rows)),
        "stage4_candidate_count": float(
            candidate_valid.sum(dim=-1).float().mean().detach().cpu().item()
        ),
        "current_state_route_loss": float(loss.detach().cpu().item()),
        "current_state_route_count": float(len(rows)),
        "current_state_replay_prefix_used_count": float(replay_used),
        "current_state_post_action_update_count": 0.0,
        "stage4_post_action_update_rows": 0.0,
        "stage4_next_state_rows": 0.0,
        "stage4_score_calibrator_enabled": False,
        "stage4_score_calibrator_active_row_fraction": 0.0,
        "stage4_recurrent_belief_memory_enabled": True,
        "stage4_replay_prefix_used_count": float(replay_used),
        "stage4_replay_prefix_trainable_used_count": 0.0,
        "stage4_replay_prefix_trainable_enabled": False,
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_scoring_mode": UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_residual_lambda": 0.0,
        "online_memory_weight": 0.0,
        "transition_skill_head_type": "unified_memory_retriever_candidate_union",
        "uses_stage0_prior_at_inference": False,
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "candidate_recall_static_k": float(static_k),
        "candidate_recall_dynamic_extra_k": float(dynamic_extra_k),
        "candidate_recall_final_k": float(final_k),
        "memory_utility_reliability_mode": reliability_mode,
        "memory_utility_fixed_alpha": float(fixed_alpha),
        "memory_utility_raw_alpha_mean": float(raw_alpha.mean().detach().cpu().item()),
        "memory_utility_alpha_mean": float(effective_alpha.mean().detach().cpu().item()),
        "memory_utility_alpha_min": float(effective_alpha.min().detach().cpu().item()),
        "memory_utility_alpha_max": float(effective_alpha.max().detach().cpu().item()),
        "memory_utility_zero_history_rows": float(
            (causal_update_count <= 0).sum().detach().cpu().item()
        ),
        "zero_history_fallback": "exact_static",
        "reliability_feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
        "memory_utility_heuristic_version": HEURISTIC_ALPHA_VERSION,
        "reliability_changes_memory_state": False,
        "cmc_adapter_enabled": reliability_mode
        in {
            "cmc",
            "cmc_candidate_provenance",
            "candidate_admission_residual",
        },
        "memory_utility_external_gate_loaded": memory_utility_gate is not None,
        "safe_memory_residual_bound": float(safe_memory_residual_bound),
    }
    metrics.update(
        {
            f"stage4_unified_fused_{key.removeprefix('stage4_')}": value
            for key, value in fused_metrics.items()
        }
    )
    metrics.update(
        {
            f"stage4_unified_dynamic_{key.removeprefix('stage4_')}": value
            for key, value in dynamic_metrics.items()
        }
    )
    metrics.update(
        {
            f"stage4_unified_static_{key.removeprefix('stage4_')}": value
            for key, value in static_metrics.items()
        }
    )
    if bool(eligible.any()):
        eligible_labels = (
            candidate_positive[eligible]
            .to(torch.float32)
            .argmax(dim=-1)
            .to(torch.long)
        )
        metrics.update(
            _dynamic_static_rank_metrics(
                dynamic_masked[eligible],
                static_masked[eligible],
                eligible_labels,
                positive_mask=candidate_positive[eligible],
                prefix="stage4_unified",
            )
        )
    else:
        metrics.update(
            _dynamic_static_rank_metrics(
                dynamic_masked[:0],
                static_masked[:0],
                torch.zeros(0, dtype=torch.long, device=device),
                positive_mask=candidate_positive[:0],
                prefix="stage4_unified",
            )
        )
    return CurrentStateRouteBatchOutput(
        loss=loss,
        metrics=metrics,
        static_full_logits=static_full,
        raw_dynamic_full_logits=raw_dynamic_full,
        dynamic_full_logits=dynamic_full,
        legal_mask=legal_pool.mask,
        positive_mask=positive,
        candidate_union=union,
        static_candidate_logits=static_masked,
        dynamic_candidate_logits=dynamic_masked,
        fused_candidate_logits=fused_masked,
        candidate_valid_mask=candidate_valid,
        candidate_positive_mask=candidate_positive,
        features=features,
        raw_alpha=raw_alpha,
        effective_alpha=effective_alpha,
        static_memory=static_memory,
        dynamic_memory=dynamic_memory,
        causal_update_count=causal_update_count,
    )


def _compute_current_state_route_loss(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    candidate_recall_mode: str = "stage0_candidates",
    skill_id_to_idx: dict[str, int] | None = None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    final_k: int = 64,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate: torch.nn.Module | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    safe_memory_residual_bound: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not callable(getattr(model, "initial_belief", None)) or not callable(
        getattr(model, "unified_route_logits", None)
    ):
        raise ValueError(
            "current-state unified routing requires model.initial_belief and model.unified_route_logits"
        )
    skill_table = getattr(model, "skill_table", None)
    skill_embeddings = getattr(skill_table, "E", None)
    if skill_embeddings is None:
        raise ValueError("current-state unified routing requires model.skill_table.E")
    candidate_recall_mode = str(candidate_recall_mode or "stage0_candidates")
    if candidate_recall_mode not in CANDIDATE_RECALL_MODES:
        raise ValueError(f"unsupported candidate_recall_mode: {candidate_recall_mode}")
    if not rows:
        reference = next(model.parameters())
        zero = reference.sum() * 0.0
        return zero, {
            "stage4_act_count": 0.0,
            "current_state_route_count": 0.0,
            "current_state_replay_prefix_used_count": 0.0,
            "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
            "candidate_recall_mode": candidate_recall_mode,
            "memory_utility_reliability_mode": str(reliability_mode or "dynamic"),
        }

    if candidate_recall_mode == "static_plus_dynamic_extra":
        output = _build_current_state_route_batch(
            model,
            rows,
            device,
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            static_k=static_k,
            dynamic_extra_k=dynamic_extra_k,
            final_k=final_k,
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
            safe_memory_residual_bound=safe_memory_residual_bound,
        )
        return output.loss, output.metrics

    if str(reliability_mode or "dynamic") != "dynamic" or memory_utility_gate is not None:
        raise ValueError("non-default reliability modes require static_plus_dynamic_extra candidates")

    h_t = _batch_cached_or_encode(
        model,
        rows,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    static_memory = model.initial_belief(h_t)
    dynamic_memory, replay_used = _apply_replay_prefix_beliefs(
        model,
        rows,
        static_memory,
        _stage4_skill_id_to_idx(rows),
        int(skill_embeddings.size(0)),
        device,
        trainable=False,
    )
    candidate_rows, mask, _targets = _pad_candidate_indices(rows, device)
    dynamic_logits = model.unified_route_logits(h_t, dynamic_memory, candidate_rows=candidate_rows)
    static_logits = model.unified_route_logits(h_t, static_memory, candidate_rows=candidate_rows)
    minimum = torch.finfo(dynamic_logits.dtype).min
    dynamic_masked = dynamic_logits.masked_fill(~mask, minimum)
    static_masked = static_logits.masked_fill(~mask, torch.finfo(static_logits.dtype).min)
    mapping = _stage4_skill_id_to_idx(rows)
    positive_full = full_pool_positive_mask(
        rows,
        mapping,
        equivalent_skill_ids_by_skill_id,
        skill_count=int(skill_embeddings.size(0)),
        device=device,
    )
    padded_indices = torch.tensor(candidate_rows, dtype=torch.long, device=device)
    candidate_positive = positive_full.gather(1, padded_indices) & mask
    if not bool(candidate_positive.any(dim=-1).all()):
        raise ValueError("current-state candidate row lacks every declared positive")
    loss = _multi_positive_listwise_nll(dynamic_masked, candidate_positive)

    dynamic_metrics = _strict_ranking_metrics(dynamic_masked, candidate_positive, mask)
    static_metrics = _strict_ranking_metrics(static_masked, candidate_positive, mask)
    candidate_count = float(mask.float().sum(dim=-1).mean().detach().cpu().item())
    dynamic_metrics["stage4_candidate_count"] = candidate_count
    static_metrics["stage4_candidate_count"] = candidate_count
    metrics: dict[str, Any] = {
        **dynamic_metrics,
        "stage4_act_loss": float(loss.detach().cpu().item()),
        "stage4_total_loss": float(loss.detach().cpu().item()),
        "stage4_act_count": float(len(rows)),
        "current_state_route_loss": float(loss.detach().cpu().item()),
        "current_state_route_count": float(len(rows)),
        "current_state_replay_prefix_used_count": float(replay_used),
        "current_state_post_action_update_count": 0.0,
        "stage4_post_action_update_rows": 0.0,
        "stage4_next_state_rows": 0.0,
        "stage4_score_calibrator_enabled": False,
        "stage4_score_calibrator_active_row_fraction": 0.0,
        "stage4_recurrent_belief_memory_enabled": True,
        "stage4_replay_prefix_used_count": float(replay_used),
        "stage4_replay_prefix_trainable_used_count": 0.0,
        "stage4_replay_prefix_trainable_enabled": False,
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_scoring_mode": UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_residual_lambda": 0.0,
        "online_memory_weight": 0.0,
        "transition_skill_head_type": "unified_memory_retriever",
        "uses_stage0_prior_at_inference": False,
        "memory_utility_reliability_mode": "dynamic",
        "zero_history_fallback": "exact_static",
        "reliability_changes_memory_state": False,
    }
    metrics.update(
        {
            f"stage4_unified_dynamic_{key.removeprefix('stage4_')}": value
            for key, value in dynamic_metrics.items()
        }
    )
    metrics.update(
        {
            f"stage4_unified_static_{key.removeprefix('stage4_')}": value
            for key, value in static_metrics.items()
        }
    )
    metrics.update(
        _dynamic_static_rank_metrics(
            dynamic_masked,
            static_masked,
            candidate_positive.to(torch.float32).argmax(dim=-1).to(torch.long),
            positive_mask=candidate_positive,
            prefix="stage4_unified",
        )
    )
    return loss, metrics
