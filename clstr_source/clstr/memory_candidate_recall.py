from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


CANDIDATE_SELECTION_VERSION = "stable_declared_pool_v1"
CANDIDATE_UNION_VERSION = "memory_union_v1"
CANDIDATE_RECALL_MODES = {"stage0_candidates", "static_plus_dynamic_extra"}
TIE_BREAK_POLICY = "declared_pool_index_ascending"
EXPLICIT_INVENTORY_SKILL_ID_KEYS = (
    "visible_inventory_skill_ids",
    "available_skill_ids",
    "available_skills",
    "admissible_skill_ids",
    "skill_inventory_ids",
    "stage0_allowed_skill_ids",
)


def _flatten_skill_id_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("skill_id", "canonical_skill_id", "id", "name"):
            if value.get(key):
                return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_skill_id_values(item))
        return output
    text = str(value).strip()
    return [text] if text else []


def row_positive_skill_ids(
    row: dict[str, Any],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return ordered row-local positives plus optional catalog aliases.

    Row-local alternatives are decision-specific (for example, multiple valid
    Tau2 actions after one task prefix), so they cannot be represented safely by
    a global skill-alias map alone.
    """

    target = str(row.get("next_skill_id") or "").strip()
    ordered = _flatten_skill_id_values(
        [
            target,
            row.get("positive_next_skill_ids"),
            row.get("equivalent_next_skill_ids"),
        ]
    )
    equivalents = equivalent_skill_ids_by_skill_id or {}
    expanded: list[str] = []
    seen: set[str] = set()
    for skill_id in ordered:
        for item in [skill_id, *_flatten_skill_id_values(equivalents.get(skill_id))]:
            if item and item not in seen:
                seen.add(item)
                expanded.append(item)
    return expanded


def explicit_inventory_skill_ids_ordered(row: dict[str, Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for key in EXPLICIT_INVENTORY_SKILL_ID_KEYS:
        for item in _flatten_skill_id_values(row.get(key)):
            if item and item not in seen:
                seen.add(item)
                output.append(item)
    return output


def legal_skill_ids_ordered(row: dict[str, Any]) -> list[str]:
    explicit = explicit_inventory_skill_ids_ordered(row)
    if explicit:
        return explicit
    if str(row.get("candidate_pool_protocol") or "").strip().lower() != "benchmark_local":
        return []
    return list(
        dict.fromkeys(
            item
            for item in _flatten_skill_id_values(row.get("candidate_next_skill_ids"))
            if item
        )
    )


def source_rows_have_causal_sequence(rows: list[dict[str, Any]]) -> bool:
    seen_trajectories: set[str] = set()
    for row in rows:
        prefix = row.get("replay_prefix")
        if isinstance(prefix, list) and any(
            isinstance(step, dict) and str(step.get("action_text") or "").strip()
            for step in prefix
        ):
            return True
        try:
            if int(row.get("step_index") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if trajectory_id:
            if trajectory_id in seen_trajectories:
                return True
            seen_trajectories.add(trajectory_id)
    return False


def declared_candidate_pool_size(rows: list[dict[str, Any]]) -> int:
    largest = 0
    for row in rows:
        declared = _flatten_skill_id_values(row.get("candidate_next_skill_ids"))
        if not declared:
            declared = explicit_inventory_skill_ids_ordered(row)
        largest = max(largest, len(dict.fromkeys(item for item in declared if item)))
    return largest


@dataclass
class LegalSkillPoolMaskOutput:
    mask: torch.Tensor
    explicit_inventory_no_known_skill_mask: torch.Tensor


@dataclass(frozen=True)
class CandidateUnion:
    candidate_rows: list[list[int]]
    static_rows: list[list[int]]
    dynamic_top_rows: list[list[int]]
    dynamic_extra_rows: list[list[int]]
    static_equal_budget_rows: list[list[int]]


def candidate_provenance_mask(
    candidate_rows: list[list[int]],
    dynamic_extra_rows: list[list[int]],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark candidate positions introduced only by dynamic-memory retrieval."""

    if valid_mask.ndim != 2 or int(valid_mask.size(0)) != len(candidate_rows):
        raise ValueError("candidate provenance valid mask must match candidate rows")
    if len(dynamic_extra_rows) != len(candidate_rows):
        raise ValueError("dynamic extra rows must align with candidate rows")
    output = torch.zeros_like(valid_mask, dtype=torch.bool)
    width = int(valid_mask.size(1))
    valid = valid_mask.to(dtype=torch.bool)
    for row_idx, (candidates, extras) in enumerate(
        zip(candidate_rows, dynamic_extra_rows)
    ):
        extra_ids = {int(candidate) for candidate in extras}
        for position, candidate in enumerate(candidates[:width]):
            if bool(valid[row_idx, position]) and int(candidate) in extra_ids:
                output[row_idx, position] = True
    return output


def full_pool_positive_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    *,
    skill_count: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(len(rows), skill_count, dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        positive_ids = row_positive_skill_ids(row, equivalent_skill_ids_by_skill_id)
        indices = [skill_id_to_idx[item] for item in positive_ids if item in skill_id_to_idx]
        if indices:
            mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def legal_skill_pool_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    skill_count: int,
    device: torch.device,
) -> LegalSkillPoolMaskOutput:
    mask = torch.ones(len(rows), skill_count, dtype=torch.bool, device=device)
    no_known = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        legal_ids = legal_skill_ids_ordered(row)
        if not legal_ids:
            continue
        mask[row_idx].zero_()
        indices = [skill_id_to_idx[item] for item in legal_ids if item in skill_id_to_idx]
        if indices:
            mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
        else:
            no_known[row_idx] = True
    return LegalSkillPoolMaskOutput(mask=mask, explicit_inventory_no_known_skill_mask=no_known)


def stable_masked_topk_rows(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
) -> list[list[int]]:
    if logits.ndim != 2 or logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must be matching rank-2 tensors")
    if not logits.is_floating_point():
        raise ValueError("candidate logits must use a floating dtype")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("valid candidate logits must be finite")
    limit = max(0, int(k))
    if limit == 0 or int(logits.size(1)) == 0:
        return [[] for _ in range(int(logits.size(0)))]

    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probe_k = min(limit, int(logits.size(1)))
    thresholds = torch.topk(masked, k=probe_k, dim=-1, largest=True, sorted=False).values.min(dim=-1).values
    selected: list[list[int]] = []
    for row_idx in range(int(logits.size(0))):
        row_valid = valid[row_idx]
        effective_k = min(limit, int(row_valid.sum().detach().cpu().item()))
        if effective_k == 0:
            selected.append([])
            continue
        threshold = thresholds[row_idx]
        strict_ids = (row_valid & (logits[row_idx] > threshold)).nonzero(as_tuple=False).view(-1)
        boundary_ids = (row_valid & (logits[row_idx] == threshold)).nonzero(as_tuple=False).view(-1)
        fill = max(0, effective_k - int(strict_ids.numel()))
        chosen = torch.cat([strict_ids, boundary_ids[:fill]], dim=0)
        chosen_scores = logits[row_idx].gather(0, chosen)
        chosen_order = torch.argsort(chosen_scores, descending=True, stable=True)
        chosen = chosen.gather(0, chosen_order)
        selected.append([int(item) for item in chosen.detach().cpu().tolist()])
    return selected


def build_static_dynamic_union(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    static_k: int,
    dynamic_extra_k: int,
) -> CandidateUnion:
    if static_logits.ndim != 2 or static_logits.shape != dynamic_logits.shape:
        raise ValueError("static and dynamic logits must have matching rank-2 shapes")
    if static_logits.shape != valid_mask.shape:
        raise ValueError("valid_mask must match branch logits")
    if static_logits.dtype != dynamic_logits.dtype or static_logits.device != dynamic_logits.device:
        raise ValueError("static and dynamic logits must share dtype and device")
    if not static_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError("candidate logits must use floating dtypes")
    if int(static_k) < 0 or int(dynamic_extra_k) < 0:
        raise ValueError("candidate budgets must be nonnegative")

    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    static_rows = stable_masked_topk_rows(static_logits, valid, k=int(static_k))
    dynamic_top_rows = stable_masked_topk_rows(dynamic_logits, valid, k=int(dynamic_extra_k))
    static_equal_budget_rows = stable_masked_topk_rows(
        static_logits,
        valid,
        k=int(static_k) + int(dynamic_extra_k),
    )
    extra_valid = valid.clone()
    for row_idx, static_ids in enumerate(static_rows):
        if static_ids:
            ids = torch.tensor(static_ids, dtype=torch.long, device=extra_valid.device)
            extra_valid[row_idx, ids] = False
    dynamic_extra_rows = stable_masked_topk_rows(
        dynamic_logits,
        extra_valid,
        k=int(dynamic_extra_k),
    )
    candidate_rows = [
        static_ids + extra_ids
        for static_ids, extra_ids in zip(static_rows, dynamic_extra_rows)
    ]
    return CandidateUnion(
        candidate_rows=candidate_rows,
        static_rows=static_rows,
        dynamic_top_rows=dynamic_top_rows,
        dynamic_extra_rows=dynamic_extra_rows,
        static_equal_budget_rows=static_equal_budget_rows,
    )


def _candidate_hit_rows(
    candidate_rows: list[list[int]],
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if positive_mask.ndim != 2 or int(positive_mask.size(0)) != len(candidate_rows):
        raise ValueError("positive_mask must have one row per candidate list")
    hits = torch.zeros(len(candidate_rows), dtype=torch.bool, device=positive_mask.device)
    skill_count = int(positive_mask.size(1))
    for row_idx, indices in enumerate(candidate_rows):
        normalized = [int(index) for index in indices]
        if any(index < 0 or index >= skill_count for index in normalized):
            raise ValueError("candidate index is outside positive_mask")
        if normalized:
            ids = torch.tensor(normalized, dtype=torch.long, device=positive_mask.device)
            hits[row_idx] = positive_mask[row_idx].index_select(0, ids).any()
    return hits


def candidate_recall_report(
    union: CandidateUnion,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> dict[str, Any]:
    row_count = len(union.candidate_rows)
    union_fields = (
        union.static_rows,
        union.dynamic_top_rows,
        union.dynamic_extra_rows,
        union.static_equal_budget_rows,
    )
    if any(len(rows) != row_count for rows in union_fields):
        raise ValueError("candidate union fields must have matching row counts")
    if positive_mask.ndim != 2 or int(positive_mask.size(0)) != row_count:
        raise ValueError("positive_mask must have one row per candidate union")
    if legal_mask.shape != positive_mask.shape:
        raise ValueError("legal_mask must match positive_mask")
    if causal_update_count.ndim != 1 or int(causal_update_count.numel()) != row_count:
        raise ValueError("causal_update_count must have one value per row")

    known_positive = positive_mask.to(dtype=torch.bool)
    legal = legal_mask.to(device=known_positive.device, dtype=torch.bool)
    legal_positive = known_positive & legal
    counts = causal_update_count.to(device=known_positive.device, dtype=torch.float32)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")

    static_hit = _candidate_hit_rows(union.static_rows, legal_positive)
    dynamic_top_hit = _candidate_hit_rows(union.dynamic_top_rows, legal_positive)
    dynamic_extra_hit = _candidate_hit_rows(union.dynamic_extra_rows, legal_positive)
    union_hit = _candidate_hit_rows(union.candidate_rows, legal_positive)
    equal_budget_hit = _candidate_hit_rows(union.static_equal_budget_rows, legal_positive)
    overlap = torch.tensor(
        [
            len(set(static_ids).intersection(dynamic_ids))
            / max(1, min(len(static_ids), len(dynamic_ids)))
            for static_ids, dynamic_ids in zip(union.static_rows, union.dynamic_top_rows)
        ],
        dtype=torch.float32,
        device=known_positive.device,
    )
    dynamic_only_count = torch.tensor(
        [len(indices) for indices in union.dynamic_extra_rows],
        dtype=torch.float32,
        device=known_positive.device,
    )

    def summarize(row_mask: torch.Tensor) -> dict[str, float]:
        selected = row_mask.to(device=known_positive.device, dtype=torch.bool)
        denominator = int(selected.sum().detach().cpu().item())
        static_miss = selected & ~static_hit
        rescued = selected & ~static_hit & union_hit

        def count(values: torch.Tensor) -> int:
            return int((selected & values).sum().detach().cpu().item())

        def rate(values: torch.Tensor) -> float:
            if denominator <= 0:
                return 0.0
            return float(values[selected].to(torch.float32).mean().detach().cpu().item())

        miss_count = int(static_miss.sum().detach().cpu().item())
        rescue_count = int(rescued.sum().detach().cpu().item())
        return {
            "source_rows": float(denominator),
            "static_recall": rate(static_hit),
            "dynamic_top_recall": rate(dynamic_top_hit),
            "dynamic_extra_recall": rate(dynamic_extra_hit),
            "union_recall": rate(union_hit),
            "static_equal_budget_recall": rate(equal_budget_hit),
            "static_hit_rows": float(count(static_hit)),
            "dynamic_top_hit_rows": float(count(dynamic_top_hit)),
            "dynamic_extra_hit_rows": float(count(dynamic_extra_hit)),
            "union_hit_rows": float(count(union_hit)),
            "static_equal_budget_hit_rows": float(count(equal_budget_hit)),
            "static_miss_rows": float(miss_count),
            "dynamic_rescue_rows": float(rescue_count),
            "static_miss_recovery_rate": float(rescue_count) / miss_count if miss_count else 0.0,
            "static_dynamic_overlap": rate(overlap),
            "dynamic_only_candidate_count": rate(dynamic_only_count),
            "static_dynamic_overlap_sum": float(overlap[selected].sum().detach().cpu().item()),
            "dynamic_only_candidate_total": float(
                dynamic_only_count[selected].sum().detach().cpu().item()
            ),
        }

    all_rows = torch.ones(row_count, dtype=torch.bool, device=known_positive.device)
    memory_active = counts > 0
    return {
        "all_eligible_source_strict": summarize(all_rows),
        "memory_active_strict": summarize(memory_active),
        "memory_active_coverage": (
            float(memory_active.to(torch.float32).mean().detach().cpu().item())
            if row_count
            else 0.0
        ),
        "positive_not_in_declared_pool_rows": float(
            (~known_positive.any(dim=-1)).sum().detach().cpu().item()
        ),
        "positive_outside_legal_pool_rows": float(
            (known_positive.any(dim=-1) & ~legal_positive.any(dim=-1)).sum().detach().cpu().item()
        ),
        "candidate_union_version": CANDIDATE_UNION_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "tie_break_policy": TIE_BREAK_POLICY,
    }


def candidate_recall_applicability(
    *,
    pool_protocol: str,
    legal_pool_size: int,
    static_k: int,
    dynamic_extra_k: int,
    causal_sequential: bool,
) -> dict[str, Any]:
    protocol = str(pool_protocol or "").strip().lower()
    allowed = {
        "known_global",
        "benchmark_local",
        "environment_candidates",
        "appended_untrained",
    }
    if protocol not in allowed:
        raise ValueError(f"unsupported pool_protocol: {pool_protocol}")
    if int(legal_pool_size) < 0 or int(static_k) < 0 or int(dynamic_extra_k) < 0:
        raise ValueError("pool size and candidate budgets must be nonnegative")

    if protocol == "environment_candidates":
        return {
            "candidate_recall_applicable": False,
            "candidate_recall_applicability_reason": "environment_admissible_actions",
            "candidate_recall_saturated": False,
        }
    if protocol == "appended_untrained":
        return {
            "candidate_recall_applicable": False,
            "candidate_recall_applicability_reason": "appended_untrained_skill_transfer",
            "candidate_recall_saturated": False,
        }
    if protocol == "benchmark_local" and int(static_k) + int(dynamic_extra_k) >= int(legal_pool_size):
        return {
            "candidate_recall_applicable": False,
            "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
            "candidate_recall_saturated": True,
        }
    if not bool(causal_sequential):
        return {
            "candidate_recall_applicable": False,
            "candidate_recall_applicability_reason": "nonsequential_without_causal_update",
            "candidate_recall_saturated": False,
        }
    if protocol == "known_global":
        return {
            "candidate_recall_applicable": True,
            "candidate_recall_applicability_reason": "known_global_sequential",
            "candidate_recall_saturated": False,
        }
    return {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "benchmark_local_diagnostic_only",
        "candidate_recall_saturated": False,
    }


def candidate_recall_protocol_metadata(
    *,
    pool_protocol: str,
    candidate_source: str,
    legal_pool_size: int,
    static_k: int,
    dynamic_extra_k: int,
    final_k: int,
    causal_sequential: bool,
) -> dict[str, Any]:
    if int(final_k) <= 0:
        raise ValueError("final_k must be positive")
    return {
        **candidate_recall_applicability(
            pool_protocol=pool_protocol,
            legal_pool_size=legal_pool_size,
            static_k=static_k,
            dynamic_extra_k=dynamic_extra_k,
            causal_sequential=causal_sequential,
        ),
        "pool_protocol": str(pool_protocol),
        "candidate_source": str(candidate_source),
        "declared_legal_pool_size": int(legal_pool_size),
        "requested_static_m": int(static_k),
        "requested_dynamic_extra_d": int(dynamic_extra_k),
        "final_k": int(final_k),
        "equal_budget_comparator_mode": "same_final_scorer_static_top_m_plus_d",
    }


def environment_candidate_recall_metadata() -> dict[str, Any]:
    return {
        **candidate_recall_applicability(
            pool_protocol="environment_candidates",
            legal_pool_size=0,
            static_k=0,
            dynamic_extra_k=0,
            causal_sequential=True,
        ),
        "pool_protocol": "environment_candidates",
        "candidate_source": "environment_admissible_actions",
    }


def candidate_recall_protocol_report(
    metrics: dict[str, Any],
    *,
    pool_protocol: str,
    candidate_source: str,
    legal_pool_size: int,
    source_rows: int,
    static_k: int,
    dynamic_extra_k: int,
    final_k: int,
    causal_sequential: bool,
    target_outside_declared_legal_pool_rows: int = 0,
) -> dict[str, Any]:
    source_count = int(source_rows)
    outside_count = int(target_outside_declared_legal_pool_rows)
    if source_count < 0 or outside_count < 0 or outside_count > source_count:
        raise ValueError("source and target-outside-pool counts must define a valid denominator")
    if int(final_k) <= 0:
        raise ValueError("final_k must be positive")
    protocol_metadata = candidate_recall_protocol_metadata(
        pool_protocol=pool_protocol,
        candidate_source=candidate_source,
        legal_pool_size=legal_pool_size,
        static_k=static_k,
        dynamic_extra_k=dynamic_extra_k,
        final_k=final_k,
        causal_sequential=causal_sequential,
    )
    report: dict[str, Any] = {
        **protocol_metadata,
        "candidate_recall_mode": str(metrics.get("candidate_recall_mode") or "stage0_candidates"),
        "candidate_union_version": str(metrics.get("candidate_union_version") or "disabled"),
        "candidate_selection_version": str(
            metrics.get("candidate_selection_version") or CANDIDATE_SELECTION_VERSION
        ),
        "candidate_tie_break_policy": str(
            metrics.get("candidate_tie_break_policy") or TIE_BREAK_POLICY
        ),
        "reliability_changes_candidate_set": False,
        "protocol_blockers": (
            ["target_outside_declared_legal_pool"] if outside_count > 0 else []
        ),
    }
    if (
        report["candidate_recall_mode"] != "static_plus_dynamic_extra"
        and report["candidate_recall_applicable"]
    ):
        report.update(
            {
                "candidate_recall_applicable": False,
                "candidate_recall_applicability_reason": "candidate_recall_mode_disabled",
                "candidate_recall_saturated": False,
            }
        )

    has_raw_metrics = "candidate_recall_all_source_rows" in metrics
    if not has_raw_metrics:
        if (
            report["candidate_recall_applicable"]
            and report["candidate_recall_mode"] == "static_plus_dynamic_extra"
        ):
            raise ValueError("applicable candidate recall requires strict raw count metrics")
        report.update(
            {
                "positive_not_in_declared_pool_rows": float(outside_count),
                "positive_outside_legal_pool_rows": float(outside_count),
            }
        )
        return report

    count_pairs = (
        ("static_recall", "static_hit_rows"),
        ("dynamic_top_recall", "dynamic_top_hit_rows"),
        ("dynamic_extra_recall", "dynamic_extra_hit_rows"),
        ("union_recall", "union_hit_rows"),
        ("static_equal_budget_recall", "static_equal_budget_hit_rows"),
    )

    def summarize(prefix: str, denominator: float) -> dict[str, float]:
        required = [f"{prefix}source_rows"]
        required.extend(f"{prefix}{count_name}" for _metric_name, count_name in count_pairs)
        required.extend(
            (
                f"{prefix}dynamic_rescue_rows",
                f"{prefix}static_dynamic_overlap_sum",
                f"{prefix}dynamic_only_candidate_total",
            )
        )
        missing = [key for key in required if key not in metrics]
        if missing:
            raise ValueError(
                "candidate recall protocol report requires raw count metrics: "
                + ", ".join(sorted(missing))
            )
        retained_rows = float(metrics[f"{prefix}source_rows"])
        if retained_rows < 0 or retained_rows > denominator:
            raise ValueError("retained candidate-recall rows exceed the strict source denominator")
        result = {
            "source_rows": float(denominator),
            "retained_evaluator_rows": retained_rows,
        }
        static_hits = float(metrics[f"{prefix}static_hit_rows"])
        for metric_name, count_name in count_pairs:
            count = float(metrics[f"{prefix}{count_name}"])
            result[count_name] = count
            result[metric_name] = count / denominator if denominator > 0 else 0.0
        static_miss_rows = max(0.0, float(denominator) - static_hits)
        rescue_rows = float(metrics[f"{prefix}dynamic_rescue_rows"])
        overlap_sum = float(metrics[f"{prefix}static_dynamic_overlap_sum"])
        dynamic_only_total = float(metrics[f"{prefix}dynamic_only_candidate_total"])
        result.update(
            {
                "static_miss_rows": static_miss_rows,
                "dynamic_rescue_rows": rescue_rows,
                "static_miss_recovery_rate": (
                    rescue_rows / static_miss_rows if static_miss_rows > 0 else 0.0
                ),
                "static_dynamic_overlap_sum": overlap_sum,
                "static_dynamic_overlap": overlap_sum / denominator if denominator > 0 else 0.0,
                "dynamic_only_candidate_total": dynamic_only_total,
                "dynamic_only_candidate_count": (
                    dynamic_only_total / denominator if denominator > 0 else 0.0
                ),
            }
        )
        return result

    all_strict = summarize("candidate_recall_all_", float(source_count))
    active_source_rows = float(metrics.get("candidate_recall_memory_active_source_rows", 0.0))
    active_strict = summarize("candidate_recall_memory_active_", active_source_rows)
    not_declared = max(
        float(metrics.get("candidate_recall_positive_not_in_declared_pool_rows", 0.0)),
        float(outside_count),
    )
    outside_legal = max(
        float(metrics.get("candidate_recall_positive_outside_legal_pool_rows", 0.0)),
        float(outside_count),
    )
    report.update(
        {
            "all_eligible_source_strict": all_strict,
            "memory_active_strict": active_strict,
            "memory_active_coverage": (
                active_source_rows / source_count if source_count > 0 else 0.0
            ),
            "positive_not_in_declared_pool_rows": not_declared,
            "positive_outside_legal_pool_rows": outside_legal,
        }
    )
    return report
