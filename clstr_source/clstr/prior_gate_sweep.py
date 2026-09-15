from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_FORMULAS = (
    "fixed_0_00",
    "fixed_0_10",
    "fixed_0_15",
    "fixed_0_25",
    "fixed_0_50",
    "fixed_1_00",
    "margin_gate",
    "entropy_gate",
    "margin_entropy_gate",
)


def _parse_fixed_lambda(formula: str) -> float | None:
    if not formula.startswith("fixed_"):
        return None
    raw = formula.removeprefix("fixed_")
    if "_" in raw:
        left, right = raw.split("_", 1)
        return float(f"{int(left)}.{right}")
    return float(raw)


def _valid_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [rows, candidates]")
    return logits.float()


def _prior_margin(prior_logits: torch.Tensor) -> torch.Tensor:
    prior_logits = _valid_logits(prior_logits)
    if prior_logits.size(1) <= 1:
        return torch.full((prior_logits.size(0),), float("inf"), device=prior_logits.device)
    top2 = torch.topk(prior_logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def _normalized_entropy(prior_logits: torch.Tensor) -> torch.Tensor:
    prior_logits = _valid_logits(prior_logits)
    width = int(prior_logits.size(1))
    if width <= 1:
        return torch.zeros(prior_logits.size(0), dtype=torch.float32, device=prior_logits.device)
    probs = torch.softmax(prior_logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return entropy / math.log(float(width))


def gate_lambdas(
    formula: str,
    prior_logits: torch.Tensor,
    *,
    lambda_min: float = 0.05,
    lambda_max: float = 0.5,
    margin_threshold: float = 1.0,
    entropy_low: float = 0.2,
    entropy_high: float = 0.8,
) -> torch.Tensor:
    prior_logits = _valid_logits(prior_logits)
    fixed = _parse_fixed_lambda(str(formula))
    if fixed is not None:
        return torch.full((prior_logits.size(0),), float(fixed), dtype=torch.float32, device=prior_logits.device)
    lambda_min = float(lambda_min)
    lambda_max = float(lambda_max)
    if formula == "margin_gate":
        confident = _prior_margin(prior_logits) >= float(margin_threshold)
        return torch.where(
            confident,
            torch.full((prior_logits.size(0),), lambda_min, dtype=torch.float32, device=prior_logits.device),
            torch.full((prior_logits.size(0),), lambda_max, dtype=torch.float32, device=prior_logits.device),
        )
    if formula == "entropy_gate":
        entropy = _normalized_entropy(prior_logits)
        scale = ((entropy - float(entropy_low)) / max(float(entropy_high) - float(entropy_low), 1e-6)).clamp(0.0, 1.0)
        return lambda_min + (lambda_max - lambda_min) * scale
    if formula == "margin_entropy_gate":
        margin_lambdas = gate_lambdas(
            "margin_gate",
            prior_logits,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            margin_threshold=margin_threshold,
            entropy_low=entropy_low,
            entropy_high=entropy_high,
        )
        entropy_lambdas = gate_lambdas(
            "entropy_gate",
            prior_logits,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            margin_threshold=margin_threshold,
            entropy_low=entropy_low,
            entropy_high=entropy_high,
        )
        return torch.minimum(margin_lambdas, entropy_lambdas)
    raise ValueError(f"unsupported gate formula: {formula}")


def _ranks_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> list[int]:
    logits = _valid_logits(logits)
    if labels.ndim != 1:
        raise ValueError("expected labels with shape [rows]")
    order = torch.argsort(logits.detach(), dim=-1, descending=True)
    ranks: list[int] = []
    for row_idx, label in enumerate(labels.detach().cpu().tolist()):
        if positive_mask is None:
            positives = [int(label)]
        else:
            positives = [int(item) for item in positive_mask[row_idx].detach().cpu().nonzero(as_tuple=False).view(-1).tolist()]
            if not positives:
                positives = [int(label)]
        row_order = order[row_idx].detach().cpu()
        positive_ranks: list[int] = []
        for positive in positives:
            positions = (row_order == int(positive)).nonzero(as_tuple=False)
            if positions.numel():
                positive_ranks.append(int(positions[0].item()) + 1)
        ranks.append(min(positive_ranks) if positive_ranks else int(logits.size(1)) + 1)
    return ranks


def _rank_metrics(
    final_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    final_ranks = _ranks_from_logits(final_logits, labels, positive_mask=positive_mask)
    prior_ranks = _ranks_from_logits(prior_logits, labels, positive_mask=positive_mask)
    denom = max(1, len(final_ranks))

    def recall(ranks: list[int], k: int) -> float:
        return float(sum(1 for rank in ranks if rank <= k) / denom)

    final_mrr = float(sum(1.0 / rank for rank in final_ranks) / denom)
    prior_mrr = float(sum(1.0 / rank for rank in prior_ranks) / denom)
    return {
        "row_count": int(len(final_ranks)),
        "recall@1": recall(final_ranks, 1),
        "recall@5": recall(final_ranks, 5),
        "recall@10": recall(final_ranks, 10),
        "recall@20": recall(final_ranks, 20),
        "mrr": final_mrr,
        "prior_recall@1": recall(prior_ranks, 1),
        "prior_recall@5": recall(prior_ranks, 5),
        "prior_recall@10": recall(prior_ranks, 10),
        "prior_recall@20": recall(prior_ranks, 20),
        "prior_mrr": prior_mrr,
        "delta_vs_prior_recall@1": recall(final_ranks, 1) - recall(prior_ranks, 1),
        "delta_vs_prior_recall@5": recall(final_ranks, 5) - recall(prior_ranks, 5),
        "delta_vs_prior_mrr": final_mrr - prior_mrr,
        "improved_vs_prior_fraction": float(sum(1 for rank, prior in zip(final_ranks, prior_ranks) if rank < prior) / denom),
        "worse_than_prior_fraction": float(sum(1 for rank, prior in zip(final_ranks, prior_ranks) if rank > prior) / denom),
        "rank_drop_count": int(sum(1 for rank, prior in zip(final_ranks, prior_ranks) if prior <= 5 and rank > 5)),
    }


def sweep_prior_residual_logits(
    prior_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
    formulas: Iterable[str] = DEFAULT_FORMULAS,
    lambda_min: float = 0.05,
    lambda_max: float = 0.5,
    margin_threshold: float = 1.0,
    entropy_low: float = 0.2,
    entropy_high: float = 0.8,
) -> dict[str, Any]:
    prior_logits = _valid_logits(prior_logits)
    residual_logits = _valid_logits(residual_logits)
    if prior_logits.shape != residual_logits.shape:
        raise ValueError("prior_logits and residual_logits must have the same shape")
    if labels.size(0) != prior_logits.size(0):
        raise ValueError("labels row count must match logits")

    formula_reports: dict[str, Any] = {}
    for formula in formulas:
        lambdas = gate_lambdas(
            str(formula),
            prior_logits,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            margin_threshold=margin_threshold,
            entropy_low=entropy_low,
            entropy_high=entropy_high,
        ).to(device=prior_logits.device, dtype=prior_logits.dtype)
        final_logits = prior_logits + lambdas.unsqueeze(-1) * residual_logits
        metrics = _rank_metrics(final_logits, prior_logits, labels, positive_mask=positive_mask)
        metrics.update(
            {
                "lambda_mean": float(lambdas.detach().cpu().mean().item()) if lambdas.numel() else 0.0,
                "lambda_min": float(lambdas.detach().cpu().min().item()) if lambdas.numel() else 0.0,
                "lambda_max": float(lambdas.detach().cpu().max().item()) if lambdas.numel() else 0.0,
            }
        )
        formula_reports[str(formula)] = metrics
    best_by_mrr = max(formula_reports, key=lambda name: formula_reports[name]["mrr"]) if formula_reports else ""
    best_by_damage = min(
        formula_reports,
        key=lambda name: (
            formula_reports[name]["worse_than_prior_fraction"],
            -formula_reports[name]["mrr"],
        ),
    ) if formula_reports else ""
    return {
        "row_count": int(prior_logits.size(0)),
        "candidate_count": int(prior_logits.size(1)),
        "formulas": formula_reports,
        "best_by_mrr": best_by_mrr,
        "best_by_damage": best_by_damage,
    }


def _merge_tensor_chunks(chunks: list[torch.Tensor], *, name: str) -> torch.Tensor:
    if not chunks:
        raise RuntimeError(f"no {name} chunks were collected")
    widths = {int(chunk.size(1)) for chunk in chunks if chunk.ndim == 2}
    if len(widths) != 1:
        raise RuntimeError(f"{name} chunks have inconsistent candidate widths: {sorted(widths)}")
    return torch.cat(chunks, dim=0)


def write_gate_sweep_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_prior_gate_replay_sweep(
    *,
    stage0_checkpoint_path: str | Path,
    stage_checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    top_m: int = 500,
    batch_size: int = 8,
    max_rows: int | None = None,
    max_diagnostic_rows: int | None = None,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    skill_text_format: str | None = None,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 100,
    transition_scoring_mode: str = "stage0_rank_prior_plus_transition_residual",
    transition_inventory_mask_mode: str = "stage0_topk_trajectory_prior",
    transition_inventory_min_candidates: int = 50,
    transition_positive_mode: str = "gold_plus_equivalent",
    formulas: Iterable[str] = DEFAULT_FORMULAS,
    lambda_min: float = 0.05,
    lambda_max: float = 0.5,
    margin_threshold: float = 1.0,
    entropy_low: float = 0.2,
    entropy_high: float = 0.8,
) -> dict[str, Any]:
    # Heavy CLSTR imports stay local so pure gate tests remain lightweight.
    from clstr.full_base_train import (
        STATE_QUERY_ROLE,
        TRANSITION_TEXT_ROLE,
        _apply_replay_prefix_beliefs,
        _apply_skill_text_format,
        _attach_stage0_topm_candidates,
        _batch_action_text_embedding_or_none,
        _batch_cached_or_encode,
        _cap_rows_by_benchmark,
        _equivalent_skill_ids_by_skill_id,
        _filter_rows_by_allowed_benchmarks,
        _filter_rows_by_train_split,
        _filter_transition_candidates_by_inventory,
        _limit_rows_for_smoke,
        _normalize_loss_weights,
        _read_jsonl,
        _row_candidate_indices,
        _skill_id,
        _transition_candidate_logits_for_mode,
        _transition_positive_mask,
    )
    from clstr.stage2_real_topm_eval import DEFAULT_STAGE2_V2_LOSS_WEIGHTS
    from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage0_checkpoint_path = Path(stage0_checkpoint_path)
    stage_checkpoint_path = Path(stage_checkpoint_path)
    train_path = Path(train_path)
    skills_path = Path(skills_path)

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=stage0_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    head_load_report = load_head_checkpoint_into_model(
        model,
        stage_checkpoint_path,
        partial_load_mode="stage_checkpoint_compatible_state",
    )
    rows = _read_jsonl(train_path)
    raw_row_count = len(rows)
    rows, train_split_filter_report = _filter_rows_by_train_split(rows)
    rows, benchmark_filter_report = _filter_rows_by_allowed_benchmarks(rows, allowed_benchmarks)
    rows, benchmark_caps_report = _cap_rows_by_benchmark(rows, benchmark_caps)
    rows, max_rows_filter_report = _limit_rows_for_smoke(rows, max_rows)

    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    skill_ids_by_idx = {idx: str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)}
    equivalent_skill_ids_by_skill_id = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    initial_belief = getattr(model, "initial_belief", None)
    if not callable(initial_belief):
        raise ValueError("prior-gate replay sweep requires model.initial_belief")
    skill_text_format_report = _apply_skill_text_format(model, model_config, skill_text_format)

    setup_status_path = output_dir / "setup_status.jsonl"
    if setup_status_path.exists():
        setup_status_path.unlink()
    rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=stage0_checkpoint_path,
        manifest_path=output_dir / "stage0_candidate_handoff_no_inject.json",
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        setup_status_path=setup_status_path,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
    )
    if max_diagnostic_rows is not None:
        rows = rows[: max(0, int(max_diagnostic_rows))]

    loss_weights = _normalize_loss_weights(DEFAULT_STAGE2_V2_LOSS_WEIGHTS)
    del loss_weights
    skill_count = max(1, len(skill_id_to_idx))
    batch_size = max(1, int(batch_size))
    prior_chunks: list[torch.Tensor] = []
    residual_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    positive_mask_chunks: list[torch.Tensor] = []
    skipped = {
        "no_transition_loss": 0,
        "missing_next_skill_id": 0,
        "missing_stage0_candidates": 0,
        "positive_missing_after_inventory": 0,
    }

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if not batch:
                continue
            h = _batch_cached_or_encode(
                model,
                batch,
                "_state_embedding",
                "state_text",
                device,
                text_role=STATE_QUERY_ROLE,
            )
            m_obs = initial_belief(h)
            m_obs, _prefix_count = _apply_replay_prefix_beliefs(
                model,
                batch,
                m_obs,
                skill_id_to_idx,
                skill_count,
                device,
            )
            item_indices: list[int] = []
            for idx, row in enumerate(batch):
                if not (row.get("loss_mask") or {}).get("L_trans_skill_ce"):
                    skipped["no_transition_loss"] += 1
                    continue
                if str(row.get("next_skill_id") or "") not in skill_id_to_idx:
                    skipped["missing_next_skill_id"] += 1
                    continue
                item_indices.append(idx)
            if not item_indices:
                continue
            selected_rows = [batch[idx] for idx in item_indices]
            selected_indices = torch.tensor(item_indices, dtype=torch.long, device=device)
            h_selected = h.index_select(0, selected_indices)
            m_selected = m_obs.index_select(0, selected_indices)
            obs_emb = _batch_cached_or_encode(
                model,
                selected_rows,
                "_next_observation_embedding",
                "next_observation_text",
                device,
                text_role=TRANSITION_TEXT_ROLE,
            )
            action_emb = _batch_action_text_embedding_or_none(model, selected_rows, device)
            current_labels = torch.tensor(
                [skill_id_to_idx.get(str(row.get("skill_id")), 0) for row in selected_rows],
                dtype=torch.long,
                device=device,
            )
            next_labels_global = torch.tensor(
                [skill_id_to_idx[str(row.get("next_skill_id"))] for row in selected_rows],
                dtype=torch.long,
                device=device,
            )
            candidate_rows = [
                _row_candidate_indices(row, "stage0_next_candidate_skill_indices", len(skill_id_to_idx))
                or _row_candidate_indices(row, "stage0_candidate_skill_indices", len(skill_id_to_idx))
                for row in selected_rows
            ]
            label_values = [int(item) for item in next_labels_global.detach().cpu().tolist()]
            keep_positions = [
                pos for pos, (label, candidates) in enumerate(zip(label_values, candidate_rows)) if candidates and label in candidates
            ]
            skipped["missing_stage0_candidates"] += len(selected_rows) - len(keep_positions)
            if not keep_positions:
                continue
            selected_rows = [selected_rows[pos] for pos in keep_positions]
            candidate_rows = [candidate_rows[pos] for pos in keep_positions]
            next_labels_global = next_labels_global.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))
            current_labels = current_labels.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))
            h_selected = h_selected.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))
            m_selected = m_selected.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))
            obs_emb = obs_emb.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))
            if action_emb is not None:
                action_emb = action_emb.index_select(0, torch.tensor(keep_positions, dtype=torch.long, device=device))

            candidate_rows, _inventory_audit = _filter_transition_candidates_by_inventory(
                rows=selected_rows,
                candidate_rows=candidate_rows,
                labels=next_labels_global,
                skill_ids_by_idx=skill_ids_by_idx,
                mode=transition_inventory_mask_mode,
                min_candidates=transition_inventory_min_candidates,
            )
            label_values = [int(item) for item in next_labels_global.detach().cpu().tolist()]
            keep_positions = [
                pos for pos, (label, candidates) in enumerate(zip(label_values, candidate_rows)) if candidates and label in candidates
            ]
            skipped["positive_missing_after_inventory"] += len(selected_rows) - len(keep_positions)
            if not keep_positions:
                continue
            selected_rows = [selected_rows[pos] for pos in keep_positions]
            candidate_rows = [candidate_rows[pos] for pos in keep_positions]
            selector = torch.tensor(keep_positions, dtype=torch.long, device=device)
            next_labels_global = next_labels_global.index_select(0, selector)
            current_labels = current_labels.index_select(0, selector)
            h_selected = h_selected.index_select(0, selector)
            m_selected = m_selected.index_select(0, selector)
            obs_emb = obs_emb.index_select(0, selector)
            if action_emb is not None:
                action_emb = action_emb.index_select(0, selector)

            max_width = max(len(row) for row in candidate_rows)
            candidate_ids = torch.zeros(len(candidate_rows), max_width, dtype=torch.long, device=device)
            candidate_valid_mask = torch.zeros(len(candidate_rows), max_width, dtype=torch.bool, device=device)
            local_labels: list[int] = []
            for row_idx, (candidates, label) in enumerate(zip(candidate_rows, next_labels_global.detach().cpu().tolist())):
                width = len(candidates)
                candidate_ids[row_idx, :width] = torch.tensor(candidates, dtype=torch.long, device=device)
                candidate_valid_mask[row_idx, :width] = True
                local_labels.append(candidates.index(int(label)))
            labels = torch.tensor(local_labels, dtype=torch.long, device=device)
            _combined, _head_type, prior_logits, residual_logits = _transition_candidate_logits_for_mode(
                model,
                h_selected,
                m_selected,
                current_labels,
                obs_emb,
                action_emb,
                skill_count,
                candidate_ids=candidate_ids,
                candidate_valid_mask=candidate_valid_mask,
                residual_lambda=0.0,
                scoring_mode=transition_scoring_mode,
            )
            positive_mask, _positive_counts = _transition_positive_mask(
                rows=selected_rows,
                candidate_rows=candidate_rows,
                labels=labels,
                skill_ids_by_idx=skill_ids_by_idx,
                positive_mode=transition_positive_mode,
                device=device,
                equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            )
            prior_chunks.append(prior_logits.detach().cpu())
            residual_chunks.append(residual_logits.detach().cpu())
            label_chunks.append(labels.detach().cpu())
            positive_mask_chunks.append(positive_mask.detach().cpu())

    prior_logits = _merge_tensor_chunks(prior_chunks, name="prior_logits")
    residual_logits = _merge_tensor_chunks(residual_chunks, name="residual_logits")
    labels = torch.cat(label_chunks, dim=0)
    positive_mask = _merge_tensor_chunks(positive_mask_chunks, name="positive_mask").to(torch.bool)
    sweep = sweep_prior_residual_logits(
        prior_logits,
        residual_logits,
        labels,
        positive_mask=positive_mask,
        formulas=formulas,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        margin_threshold=margin_threshold,
        entropy_low=entropy_low,
        entropy_high=entropy_high,
    )
    report = {
        "status": "ok" if sweep["row_count"] else "action_required",
        "blockers": [] if sweep["row_count"] else ["no_rows_swept"],
        **sweep,
        "output_dir": str(output_dir),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage_checkpoint_path": str(stage_checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "config": {
            "top_m": int(top_m),
            "batch_size": int(batch_size),
            "max_rows": max_rows,
            "max_diagnostic_rows": max_diagnostic_rows,
            "transition_scoring_mode": str(transition_scoring_mode),
            "transition_inventory_mask_mode": str(transition_inventory_mask_mode),
            "transition_inventory_min_candidates": int(transition_inventory_min_candidates),
            "transition_positive_mode": str(transition_positive_mode),
            "formulas": [str(item) for item in formulas],
            "lambda_min": float(lambda_min),
            "lambda_max": float(lambda_max),
            "margin_threshold": float(margin_threshold),
            "entropy_low": float(entropy_low),
            "entropy_high": float(entropy_high),
        },
        "data_filters": {
            "raw_row_count": raw_row_count,
            "train_split_filter": train_split_filter_report,
            "benchmark_filter": benchmark_filter_report,
            "benchmark_caps": benchmark_caps_report,
            "max_rows_filter": max_rows_filter_report,
        },
        "stage0_candidate_handoff": handoff_report,
        "model_load": {
            "routing_init": routing_report,
            "head_load": head_load_report,
            "skill_text_format": skill_text_format_report,
        },
        "skipped": skipped,
    }
    write_gate_sweep_report(output_dir / "gate_sweep_report.json", report)
    return report
