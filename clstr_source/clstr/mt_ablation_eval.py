from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

MT_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "static_no_replay": {
        "auto_replay_prefix_max_steps": 0,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
    },
    "dynamic_replay_no_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
    },
    "dynamic_replay_with_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 1.0,
        "score_calibrator_enabled": True,
    },
    "mismatch_replay_no_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
    },
    "order_shuffled_replay_no_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
    },
    "shuffled_replay_no_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
        "alias_of": "mismatch_replay_no_online",
    },
    "masked_replay_no_online": {
        "auto_replay_prefix_max_steps": 3,
        "online_memory_weight": 0.0,
        "score_calibrator_enabled": False,
    },
}


def _numeric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_pairwise_effect_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(records)

    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]

    def count_changed(lhs: str, rhs: str) -> int:
        return sum(1 for record in records if record.get(lhs) != record.get(rhs))

    def count_improved(static_key: str, dynamic_key: str) -> int:
        return sum(
            1
            for record in records
            if isinstance(record.get(static_key), (int, float))
            and isinstance(record.get(dynamic_key), (int, float))
            and int(record[dynamic_key]) < int(record[static_key])
        )

    def count_worsened(static_key: str, dynamic_key: str) -> int:
        return sum(
            1
            for record in records
            if isinstance(record.get(static_key), (int, float))
            and isinstance(record.get(dynamic_key), (int, float))
            and int(record[dynamic_key]) > int(record[static_key])
        )

    return {
        "row_count": int(row_count),
        "m_l2_mean": _mean(values("m_l2")),
        "m_cosine_mean": _mean(values("m_cosine")),
        "final_max_abs_diff_mean": _mean(values("final_max_abs_diff")),
        "final_max_abs_diff_max": max(values("final_max_abs_diff"), default=0.0),
        "residual_max_abs_diff_mean": _mean(values("residual_max_abs_diff")),
        "residual_max_abs_diff_max": max(values("residual_max_abs_diff"), default=0.0),
        "final_positive_logit_delta_mean": _mean(values("final_positive_logit_delta")),
        "residual_positive_logit_delta_mean": _mean(values("residual_positive_logit_delta")),
        "final_argmax_changed_rows": count_changed("static_final_argmax", "dynamic_final_argmax"),
        "residual_argmax_changed_rows": count_changed("static_residual_argmax", "dynamic_residual_argmax"),
        "final_positive_rank_changed_rows": count_changed("static_final_rank", "dynamic_final_rank"),
        "final_positive_rank_improved_rows": count_improved("static_final_rank", "dynamic_final_rank"),
        "final_positive_rank_worsened_rows": count_worsened("static_final_rank", "dynamic_final_rank"),
        "residual_positive_rank_changed_rows": count_changed("static_residual_rank", "dynamic_residual_rank"),
        "residual_positive_rank_improved_rows": count_improved("static_residual_rank", "dynamic_residual_rank"),
        "residual_positive_rank_worsened_rows": count_worsened("static_residual_rank", "dynamic_residual_rank"),
    }


def _strict_stage4_metrics(
    metrics: dict[str, Any],
    *,
    retained_rows: int,
    source_rows: int,
) -> dict[str, float]:
    retained_rows = max(0, int(retained_rows))
    source_rows = max(0, int(source_rows))
    scale = retained_rows / source_rows if source_rows > 0 else 0.0
    return {
        "strict_stage4_next_skill_recall@1": float(metrics.get("stage4_next_skill_recall@1") or 0.0) * scale,
        "strict_stage4_next_skill_recall@5": float(metrics.get("stage4_next_skill_recall@5") or 0.0) * scale,
        "strict_stage4_next_skill_mrr": float(metrics.get("stage4_next_skill_mrr") or 0.0) * scale,
        "retained_row_fraction": scale,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
        "retained_stage4_candidate_count": float(metrics.get("stage4_candidate_count") or 0.0),
    }


def build_mt_ablation_report(
    *,
    evaluations: dict[str, dict[str, Any]],
    source_rows: int,
    retained_rows: int,
    online_memory_weight: float = 1.0,
    auto_replay_prefix_max_steps: int = 3,
    pairwise_effect_diagnostics: dict[str, Any] | None = None,
    replay_preparation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_config = {
        name: dict(config)
        for name, config in MT_ABLATION_VARIANTS.items()
    }
    variant_config["dynamic_replay_no_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["dynamic_replay_with_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["mismatch_replay_no_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["order_shuffled_replay_no_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["shuffled_replay_no_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["masked_replay_no_online"]["auto_replay_prefix_max_steps"] = int(auto_replay_prefix_max_steps)
    variant_config["dynamic_replay_with_online"]["online_memory_weight"] = float(online_memory_weight)

    strict = {
        name: _strict_stage4_metrics(
            evaluation,
            retained_rows=retained_rows,
            source_rows=source_rows,
        )
        for name, evaluation in evaluations.items()
    }
    static_eval = evaluations.get("static_no_replay", {})
    dynamic_no_online = evaluations.get("dynamic_replay_no_online", {})
    dynamic_with_online = evaluations.get("dynamic_replay_with_online", {})
    strict_static = strict.get("static_no_replay", {})
    strict_dynamic_no_online = strict.get("dynamic_replay_no_online", {})
    strict_dynamic_with_online = strict.get("dynamic_replay_with_online", {})
    return {
        "purpose": "diagnostic_only_true_m_t_ablation",
        "source_rows": int(source_rows),
        "retained_rows": int(retained_rows),
        "variant_config": variant_config,
        "evaluations": evaluations,
        "strict": strict,
        "delta_dynamic_replay_no_online_vs_static_no_replay": _numeric_delta(dynamic_no_online, static_eval),
        "delta_dynamic_replay_with_online_vs_dynamic_replay_no_online": _numeric_delta(
            dynamic_with_online,
            dynamic_no_online,
        ),
        "delta_dynamic_replay_with_online_vs_static_no_replay": _numeric_delta(dynamic_with_online, static_eval),
        "delta_true_replay_vs_mismatch_replay": _numeric_delta(
            dynamic_no_online,
            evaluations.get("mismatch_replay_no_online", evaluations.get("shuffled_replay_no_online", {})),
        ),
        "delta_true_replay_vs_order_shuffled_replay": _numeric_delta(
            dynamic_no_online,
            evaluations.get("order_shuffled_replay_no_online", {}),
        ),
        "delta_mismatch_replay_vs_static_no_replay": _numeric_delta(
            evaluations.get("mismatch_replay_no_online", evaluations.get("shuffled_replay_no_online", {})),
            static_eval,
        ),
        "delta_true_replay_vs_shuffled_replay": _numeric_delta(
            dynamic_no_online,
            evaluations.get("mismatch_replay_no_online", evaluations.get("shuffled_replay_no_online", {})),
        ),
        "delta_shuffled_replay_vs_static_no_replay": _numeric_delta(
            evaluations.get("mismatch_replay_no_online", evaluations.get("shuffled_replay_no_online", {})),
            static_eval,
        ),
        "delta_masked_replay_vs_static_no_replay": _numeric_delta(
            evaluations.get("masked_replay_no_online", {}),
            static_eval,
        ),
        "strict_delta_dynamic_replay_no_online_vs_static_no_replay": _numeric_delta(
            strict_dynamic_no_online,
            strict_static,
        ),
        "strict_delta_dynamic_replay_with_online_vs_dynamic_replay_no_online": _numeric_delta(
            strict_dynamic_with_online,
            strict_dynamic_no_online,
        ),
        "strict_delta_dynamic_replay_with_online_vs_static_no_replay": _numeric_delta(
            strict_dynamic_with_online,
            strict_static,
        ),
        "pairwise_effect_diagnostics": pairwise_effect_diagnostics or {},
        "replay_preparation": replay_preparation or {},
        "compatibility_aliases": {
            "shuffled_replay_no_online": "mismatch_replay_no_online",
            "delta_true_replay_vs_shuffled_replay": "delta_true_replay_vs_mismatch_replay",
            "delta_shuffled_replay_vs_static_no_replay": "delta_mismatch_replay_vs_static_no_replay",
        },
        "interpretation_note": (
            "static_no_replay is the true replay/m_t-off condition because "
            "auto_replay_prefix_max_steps=0. dynamic_replay_no_online isolates learned replay/belief m_t "
            "without exact online-memory bonuses. dynamic_replay_with_online matches the usual Stage4 "
            "memory-enabled diagnostic path. mismatch_replay_no_online uses a different-trajectory "
            "donor, while order_shuffled_replay_no_online permutes events from the same trajectory."
        ),
    }


def _materialize_true_replay_rows(
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from clstr.full_base_train import _attach_auto_replay_prefixes

    copied = copy.deepcopy(rows)
    return _attach_auto_replay_prefixes(copied, max_steps=max_steps)


def _mismatch_replay_prefixes(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mismatched = copy.deepcopy(rows)
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row_idx, row in enumerate(mismatched):
        prefix = row.get("replay_prefix")
        if isinstance(prefix, list) and prefix:
            benchmark = str(row.get("source_benchmark") or row.get("benchmark") or "<missing>")
            current_skill = str(row.get("skill_id") or "")
            groups[(benchmark, len(prefix), current_skill)].append(row_idx)

    def candidate_set(row: dict[str, Any]) -> set[str]:
        for key in (
            "candidate_next_skill_ids",
            "candidate_next_skill_indices",
            "stage0_next_candidate_skill_indices",
            "visible_inventory_skill_ids",
            "tool_inventory_skill_ids",
        ):
            values = row.get(key)
            if isinstance(values, list) and values:
                return {str(value) for value in values if str(value)}
        return set()

    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for position, row_idx in enumerate(indices):
            target_trajectory = str(mismatched[row_idx].get("trajectory_id") or "")
            target_candidates = candidate_set(mismatched[row_idx])
            donor_options: list[tuple[float, int, int]] = []
            for offset in range(1, len(indices)):
                candidate_idx = indices[(position + offset) % len(indices)]
                if (
                    str(mismatched[candidate_idx].get("trajectory_id") or "")
                    == target_trajectory
                ):
                    continue
                target_next_skill = str(
                    mismatched[row_idx].get("next_skill_id") or ""
                )
                donor_next_skill = str(
                    mismatched[candidate_idx].get("next_skill_id") or ""
                )
                if (
                    target_next_skill
                    and donor_next_skill
                    and donor_next_skill == target_next_skill
                ):
                    continue
                donor_candidates = candidate_set(mismatched[candidate_idx])
                union = target_candidates | donor_candidates
                overlap = (
                    len(target_candidates & donor_candidates) / len(union)
                    if union
                    else 0.0
                )
                donor_options.append((overlap, -offset, candidate_idx))
            donor_idx = max(donor_options)[-1] if donor_options else None
            if donor_idx is None:
                continue
            mismatched[row_idx]["replay_prefix"] = copy.deepcopy(
                mismatched[donor_idx]["replay_prefix"]
            )
            changed += 1
    eligible_rows = int(sum(len(indices) for indices in groups.values()))
    return mismatched, {
        "mismatched_prefix_rows": int(changed),
        "eligible_prefix_rows": eligible_rows,
        "donor_coverage": float(changed / eligible_rows) if eligible_rows else 0.0,
        "mismatch_group_count": int(len(groups)),
        "donor_scope": (
            "same_benchmark_prefix_length_current_skill_different_trajectory_and_next_skill"
        ),
        "accidental_same_target_donors": 0,
    }


def _order_shuffle_replay_prefixes(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    shuffled = copy.deepcopy(rows)
    eligible = 0
    changed = 0
    for row in shuffled:
        prefix = row.get("replay_prefix")
        if not isinstance(prefix, list) or len(prefix) < 2:
            continue
        eligible += 1
        reordered = list(reversed(prefix))
        if reordered != prefix:
            row["replay_prefix"] = reordered
            changed += 1
    return shuffled, {
        "order_shuffled_prefix_rows": int(changed),
        "eligible_prefix_rows": int(eligible),
        "permutation": "reverse_same_trajectory_prefix",
    }


def _mask_replay_prefixes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    masked = copy.deepcopy(rows)
    count = sum(
        len(prefix)
        for row in masked
        for prefix in [row.get("replay_prefix")]
        if isinstance(prefix, list)
    )
    return _strip_replay_prefixes(masked), int(count)


def evaluate_mt_ablation_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    source_rows: int,
    retained_rows: int | None = None,
    batch_size: int = 8,
    device: Any = None,
    transition_residual_lambda: float,
    transition_scoring_mode: str,
    online_memory_weight: float = 1.0,
    auto_replay_prefix_max_steps: int = 3,
    route_scorer: str = "legacy_prior_residual",
    evaluator: Any = None,
    include_pairwise_effect_diagnostics: bool = False,
) -> dict[str, Any]:
    if evaluator is None:
        from clstr.logged_online_stage4_train import evaluate_logged_online_stage4_rows as evaluator

    retained = len(rows) if retained_rows is None else int(retained_rows)
    evaluations: dict[str, dict[str, Any]] = {}
    variant_config = build_mt_ablation_report(
        evaluations={},
        source_rows=source_rows,
        retained_rows=retained,
        online_memory_weight=online_memory_weight,
        auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
    )["variant_config"]
    true_rows, true_replay_report = _materialize_true_replay_rows(
        rows,
        max_steps=int(auto_replay_prefix_max_steps),
    )
    static_rows = _strip_replay_prefixes(true_rows)
    mismatch_rows, mismatch_report = _mismatch_replay_prefixes(true_rows)
    order_shuffled_rows, order_shuffle_report = _order_shuffle_replay_prefixes(true_rows)
    masked_rows, masked_prefix_steps = _mask_replay_prefixes(true_rows)
    variant_rows = {
        "static_no_replay": static_rows,
        "dynamic_replay_no_online": true_rows,
        "mismatch_replay_no_online": mismatch_rows,
        "order_shuffled_replay_no_online": order_shuffled_rows,
        "masked_replay_no_online": masked_rows,
        "dynamic_replay_with_online": true_rows,
    }
    for name in (
        "static_no_replay",
        "dynamic_replay_no_online",
        "mismatch_replay_no_online",
        "order_shuffled_replay_no_online",
        "masked_replay_no_online",
        "dynamic_replay_with_online",
    ):
        config = variant_config[name]
        evaluations[name] = evaluator(
            model,
            variant_rows[name],
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=float(config["online_memory_weight"]),
            score_calibrator_enabled=bool(config["score_calibrator_enabled"]),
            auto_replay_prefix_max_steps=0,
            route_scorer=route_scorer,
        )
    evaluations["shuffled_replay_no_online"] = copy.deepcopy(
        evaluations["mismatch_replay_no_online"]
    )
    pairwise_effect_diagnostics = None
    if include_pairwise_effect_diagnostics:
        pairwise_effect_diagnostics = compute_mt_pairwise_effect_diagnostics(
            model,
            rows,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
            route_scorer=route_scorer,
        )
    return build_mt_ablation_report(
        evaluations=evaluations,
        source_rows=source_rows,
        retained_rows=retained,
        online_memory_weight=online_memory_weight,
        auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
        pairwise_effect_diagnostics=pairwise_effect_diagnostics,
        replay_preparation={
            "true_replay": true_replay_report,
            "mismatch": mismatch_report,
            "order_shuffle": order_shuffle_report,
            "shuffled_prefix_rows": int(mismatch_report["mismatched_prefix_rows"]),
            "masked_prefix_steps": int(masked_prefix_steps),
        },
    )


def _strip_replay_prefixes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for row in rows:
        updated = copy.deepcopy(row)
        updated.pop("replay_prefix", None)
        stripped.append(updated)
    return stripped


def compute_mt_pairwise_effect_diagnostics(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    device: Any = None,
    transition_residual_lambda: float,
    transition_scoring_mode: str,
    auto_replay_prefix_max_steps: int = 3,
    route_scorer: str = "legacy_prior_residual",
) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "auto_replay_prefix": {}}

    import torch
    import torch.nn.functional as F

    from clstr.full_base_train import (
        STATE_QUERY_ROLE,
        TRANSITION_TEXT_ROLE,
        TRANSITION_SCORING_MODE,
        UNIFIED_MEMORY_ROUTE_SCORER,
        _apply_replay_prefix_beliefs,
        _attach_auto_replay_prefixes,
        _batch_action_text_embedding_or_none,
        _batch_cached_or_encode,
        _skill_logits_and_memory,
        _transition_candidate_logits_for_mode,
    )
    from clstr.stage4_act_train import (
        STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
        _pad_candidate_indices,
        _pad_candidate_prior_scores,
        _stage4_skill_id_to_idx,
    )

    device = torch.device(device or getattr(model, "device", "cpu"))
    stripped_rows = _strip_replay_prefixes(rows)
    dynamic_rows, auto_replay_prefix_report = _attach_auto_replay_prefixes(
        stripped_rows,
        max_steps=auto_replay_prefix_max_steps,
    )
    skill_count = int(getattr(getattr(model, "skill_table", None), "E").size(0))
    skill_id_to_idx = _stage4_skill_id_to_idx(dynamic_rows)
    helper_scoring_mode = (
        TRANSITION_SCORING_MODE
        if str(transition_scoring_mode) == STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
        else str(transition_scoring_mode)
    )
    route_scorer = str(route_scorer or "legacy_prior_residual")
    records: list[dict[str, Any]] = []
    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "eval", None)):
        model.eval()

    def positive_ranks(logits: Any, target_positions: Any, mask: Any) -> Any:
        target_logits = logits.gather(1, target_positions.unsqueeze(1)).squeeze(1)
        return ((logits > target_logits.unsqueeze(1)) & mask).sum(dim=1) + 1

    with torch.no_grad():
        for start in range(0, len(stripped_rows), max(1, int(batch_size))):
            static_batch = stripped_rows[start : start + max(1, int(batch_size))]
            dynamic_batch = dynamic_rows[start : start + max(1, int(batch_size))]
            h = _batch_cached_or_encode(
                model,
                static_batch,
                "_state_embedding",
                "state_text",
                device,
                text_role=STATE_QUERY_ROLE,
            )
            _logits, static_m = _skill_logits_and_memory(model, h, skill_count)
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
                if not callable(getattr(model, "initial_belief", None)) or not callable(
                    getattr(model, "unified_route_logits", None)
                ):
                    raise ValueError("route_scorer=unified_memory requires model.initial_belief and model.unified_route_logits")
                static_m = model.initial_belief(h)
            dynamic_m, replay_used_count = _apply_replay_prefix_beliefs(
                model,
                dynamic_batch,
                static_m,
                skill_id_to_idx,
                skill_count,
                device,
                trainable=False,
            )
            obs_emb = _batch_cached_or_encode(
                model,
                static_batch,
                "_next_observation_embedding",
                "next_observation_text",
                device,
                text_role=TRANSITION_TEXT_ROLE,
            )
            action_emb = _batch_action_text_embedding_or_none(model, static_batch, device)
            labels = torch.tensor([int(row["skill_idx"]) for row in static_batch], dtype=torch.long, device=device)
            candidate_rows, mask, target_positions = _pad_candidate_indices(static_batch, device)
            candidate_ids = torch.tensor(candidate_rows, dtype=torch.long, device=device)
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
                static_logits = model.unified_route_logits(h, static_m, candidate_rows=candidate_rows)
                dynamic_logits = model.unified_route_logits(h, dynamic_m, candidate_rows=candidate_rows)
                static_prior = static_logits
                dynamic_prior = static_logits
                static_residual = torch.zeros_like(static_logits)
                dynamic_residual = dynamic_logits - static_logits
            else:
                static_logits, _static_head, static_prior, static_residual = _transition_candidate_logits_for_mode(
                    model,
                    h,
                    static_m,
                    labels,
                    obs_emb,
                    action_emb,
                    skill_count,
                    candidate_ids=candidate_ids,
                    candidate_valid_mask=mask,
                    residual_lambda=float(transition_residual_lambda),
                    scoring_mode=helper_scoring_mode,
                )
                dynamic_logits, _dynamic_head, dynamic_prior, dynamic_residual = _transition_candidate_logits_for_mode(
                    model,
                    h,
                    dynamic_m,
                    labels,
                    obs_emb,
                    action_emb,
                    skill_count,
                    candidate_ids=candidate_ids,
                    candidate_valid_mask=mask,
                    residual_lambda=float(transition_residual_lambda),
                    scoring_mode=helper_scoring_mode,
                )
                if str(transition_scoring_mode) == STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE:
                    prior_logits = _pad_candidate_prior_scores(static_batch, candidate_ids.size(1), device).to(
                        dtype=static_residual.dtype
                    )
                    static_prior = prior_logits
                    dynamic_prior = prior_logits
                    static_logits = prior_logits + float(transition_residual_lambda) * static_residual
                    dynamic_logits = prior_logits + float(transition_residual_lambda) * dynamic_residual
            static_logits = static_logits.masked_fill(~mask, torch.finfo(static_logits.dtype).min)
            dynamic_logits = dynamic_logits.masked_fill(~mask, torch.finfo(dynamic_logits.dtype).min)
            static_residual = static_residual.masked_fill(~mask, torch.finfo(static_residual.dtype).min)
            dynamic_residual = dynamic_residual.masked_fill(~mask, torch.finfo(dynamic_residual.dtype).min)
            _ = static_prior, dynamic_prior, replay_used_count

            static_final_ranks = positive_ranks(static_logits, target_positions, mask)
            dynamic_final_ranks = positive_ranks(dynamic_logits, target_positions, mask)
            static_residual_ranks = positive_ranks(static_residual, target_positions, mask)
            dynamic_residual_ranks = positive_ranks(dynamic_residual, target_positions, mask)
            static_final_argmax = static_logits.argmax(dim=1)
            dynamic_final_argmax = dynamic_logits.argmax(dim=1)
            static_residual_argmax = static_residual.argmax(dim=1)
            dynamic_residual_argmax = dynamic_residual.argmax(dim=1)
            m_cos = F.cosine_similarity(static_m.float(), dynamic_m.float(), dim=-1)
            m_l2 = torch.linalg.vector_norm((dynamic_m - static_m).float(), ord=2, dim=-1)
            final_max_abs_diff = (dynamic_logits - static_logits).abs().masked_fill(~mask, 0.0).amax(dim=1)
            residual_max_abs_diff = (dynamic_residual - static_residual).abs().masked_fill(~mask, 0.0).amax(dim=1)
            final_positive_delta = (
                dynamic_logits.gather(1, target_positions.unsqueeze(1))
                - static_logits.gather(1, target_positions.unsqueeze(1))
            ).squeeze(1)
            residual_positive_delta = (
                dynamic_residual.gather(1, target_positions.unsqueeze(1))
                - static_residual.gather(1, target_positions.unsqueeze(1))
            ).squeeze(1)
            for idx in range(len(static_batch)):
                records.append(
                    {
                        "m_l2": float(m_l2[idx].detach().cpu().item()),
                        "m_cosine": float(m_cos[idx].detach().cpu().item()),
                        "final_max_abs_diff": float(final_max_abs_diff[idx].detach().cpu().item()),
                        "residual_max_abs_diff": float(residual_max_abs_diff[idx].detach().cpu().item()),
                        "final_positive_logit_delta": float(final_positive_delta[idx].detach().cpu().item()),
                        "residual_positive_logit_delta": float(residual_positive_delta[idx].detach().cpu().item()),
                        "static_final_rank": int(static_final_ranks[idx].detach().cpu().item()),
                        "dynamic_final_rank": int(dynamic_final_ranks[idx].detach().cpu().item()),
                        "static_residual_rank": int(static_residual_ranks[idx].detach().cpu().item()),
                        "dynamic_residual_rank": int(dynamic_residual_ranks[idx].detach().cpu().item()),
                        "static_final_argmax": int(static_final_argmax[idx].detach().cpu().item()),
                        "dynamic_final_argmax": int(dynamic_final_argmax[idx].detach().cpu().item()),
                        "static_residual_argmax": int(static_residual_argmax[idx].detach().cpu().item()),
                        "dynamic_residual_argmax": int(dynamic_residual_argmax[idx].detach().cpu().item()),
                    }
                )
    if was_training and callable(getattr(model, "train", None)):
        model.train()
    summary = summarize_pairwise_effect_records(records)
    summary["auto_replay_prefix"] = auto_replay_prefix_report
    summary["transition_residual_lambda"] = float(transition_residual_lambda)
    summary["transition_scoring_mode"] = str(transition_scoring_mode)
    summary["route_scorer"] = route_scorer
    return summary
