from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


def _skill_id(skill: Mapping[str, Any], fallback: str) -> str:
    return str(
        skill.get("skill_id")
        or skill.get("canonical_skill_id")
        or skill.get("id")
        or fallback
    )


@dataclass
class DynamicSkillRegistry:
    rows: list[dict[str, Any]]
    id_to_index: dict[str, int]

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "DynamicSkillRegistry":
        normalized: list[dict[str, Any]] = []
        id_to_index: dict[str, int] = {}
        for idx, row in enumerate(rows):
            copied = dict(row)
            skill_id = _skill_id(copied, str(idx))
            if skill_id in id_to_index:
                continue
            copied["skill_id"] = skill_id
            copied.setdefault("canonical_skill_id", skill_id)
            id_to_index[skill_id] = len(normalized)
            normalized.append(copied)
        return cls(rows=normalized, id_to_index=id_to_index)

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        old_count = len(self.rows)
        appended_ids: list[str] = []
        skipped_duplicates: list[str] = []
        for offset, row in enumerate(rows):
            copied = dict(row)
            skill_id = _skill_id(copied, str(old_count + offset))
            if skill_id in self.id_to_index:
                skipped_duplicates.append(skill_id)
                continue
            copied["skill_id"] = skill_id
            copied.setdefault("canonical_skill_id", skill_id)
            copied.setdefault("is_appended_after_checkpoint", True)
            copied.setdefault("retrieval_seen_count", 0)
            copied.setdefault("transition_seen_count", 0)
            copied.setdefault("act_seen_count", 0)
            self.id_to_index[skill_id] = len(self.rows)
            self.rows.append(copied)
            appended_ids.append(skill_id)
        return {
            "old_count": old_count,
            "new_count": len(self.rows),
            "appended_count": len(appended_ids),
            "appended_skill_ids": appended_ids,
            "skipped_duplicate_skill_ids": skipped_duplicates,
        }

    def available_skill_ids(self, *, executor_domain: str | None = None) -> list[str]:
        if executor_domain is None:
            return [_skill_id(row, str(idx)) for idx, row in enumerate(self.rows)]
        domain = str(executor_domain).strip().lower()
        compatible_key = f"{domain}_executor_compatible"
        output: list[str] = []
        for idx, row in enumerate(self.rows):
            row_domain = str(row.get("executor_domain") or "").strip().lower()
            if bool(row.get(compatible_key)) or row_domain == domain:
                output.append(_skill_id(row, str(idx)))
        return output


_HEAD_COUNT_KEYS = {
    "retrieval": ("retrieval_seen_count",),
    "policy": ("policy_seen_count", "retrieval_seen_count"),
    "transition": ("transition_seen_count", "act_seen_count"),
    "belief": ("belief_seen_count", "transition_seen_count", "act_seen_count"),
    "act": ("act_seen_count", "transition_seen_count"),
}


def _is_appended_or_unseen(skill: Mapping[str, Any]) -> bool:
    return bool(
        skill.get("is_appended_after_checkpoint")
        or skill.get("unseen_skill")
        or skill.get("unseen_skill_mask")
    )


def _coverage_count(skill: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    values: list[float] = []
    for key in keys:
        if key not in skill or skill.get(key) is None:
            continue
        try:
            values.append(float(skill[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return max(values)


def skill_head_coverage_weights(
    skill_rows: Sequence[Mapping[str, Any]],
    *,
    head: str,
    min_count: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return per-skill CLSTR-head trust weights from supervision coverage.

    Existing skills without explicit coverage metadata keep full weight for
    backward compatibility. Newly appended/unseen skills default to zero head
    weight until trajectory/ACT coverage is observed.
    """
    head_key = str(head).strip().lower()
    if head_key not in _HEAD_COUNT_KEYS:
        raise ValueError(f"unsupported head for coverage weights: {head}")
    denom = max(1, int(min_count))
    weights: list[float] = []
    for skill in skill_rows:
        count = _coverage_count(skill, _HEAD_COUNT_KEYS[head_key])
        if count is None:
            weights.append(0.0 if _is_appended_or_unseen(skill) else 1.0)
            continue
        weights.append(max(0.0, min(1.0, float(count) / float(denom))))
    return torch.tensor(weights, device=device, dtype=dtype)


def blend_skill_scores_with_coverage(
    *,
    stage0_scores: torch.Tensor,
    skill_rows: Sequence[Mapping[str, Any]],
    policy_scores: torch.Tensor | None = None,
    transition_scores: torch.Tensor | None = None,
    belief_scores: torch.Tensor | None = None,
    policy_weight: float = 1.0,
    transition_weight: float = 1.0,
    belief_weight: float = 1.0,
    min_count: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Blend Stage0 text scores with CLSTR heads using per-skill coverage.

    This keeps dynamic skills usable immediately through text retrieval while
    preventing unsupported transition/belief logits from dominating them.
    """
    if stage0_scores.ndim != 2:
        raise ValueError("stage0_scores must be rank-2 [batch, skills]")
    if stage0_scores.shape[1] != len(skill_rows):
        raise ValueError(
            "stage0_scores skill dimension must match skill_rows: "
            f"{stage0_scores.shape[1]} vs {len(skill_rows)}"
        )
    output = stage0_scores.clone()
    report: dict[str, Any] = {}

    def add_head(name: str, scores: torch.Tensor | None, weight: float) -> None:
        nonlocal output
        if scores is None or float(weight) == 0.0:
            return
        if tuple(scores.shape) != tuple(stage0_scores.shape):
            raise ValueError(
                f"{name}_scores shape must match stage0_scores: "
                f"{tuple(scores.shape)} vs {tuple(stage0_scores.shape)}"
            )
        head_weights = skill_head_coverage_weights(
            skill_rows,
            head=name,
            min_count=min_count,
            device=stage0_scores.device,
            dtype=stage0_scores.dtype,
        )
        output = output + float(weight) * scores.to(device=stage0_scores.device, dtype=stage0_scores.dtype) * head_weights.unsqueeze(0)
        report[f"{name}_head_weights"] = [float(item) for item in head_weights.detach().cpu().tolist()]

    add_head("policy", policy_scores, policy_weight)
    add_head("transition", transition_scores, transition_weight)
    add_head("belief", belief_scores, belief_weight)
    return output, report
