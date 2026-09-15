from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NaturalCompressedSupport:
    coarse_candidate_ids: torch.Tensor
    coarse_valid_mask: torch.Tensor
    candidate_ids: torch.Tensor
    valid_mask: torch.Tensor
    compression_logits: torch.Tensor


@dataclass(frozen=True)
class NaturalSupportLabels:
    positive_mask: torch.Tensor
    eligible_mask: torch.Tensor


@dataclass(frozen=True)
class NaturalCandidatePath:
    query: torch.Tensor
    recall_logits: torch.Tensor
    coarse_candidate_ids: torch.Tensor
    coarse_valid_mask: torch.Tensor
    coarse_base_logits: torch.Tensor
    compression_logits: torch.Tensor
    support: NaturalCompressedSupport


@dataclass(frozen=True)
class NaturalCandidateUnionPath:
    static_query: torch.Tensor
    dynamic_query: torch.Tensor
    static_recall_logits: torch.Tensor
    recall_logits: torch.Tensor
    coarse_candidate_ids: torch.Tensor
    coarse_valid_mask: torch.Tensor
    coarse_base_logits: torch.Tensor
    compression_logits: torch.Tensor
    dynamic_extra_candidate_ids: torch.Tensor
    dynamic_extra_valid_mask: torch.Tensor
    support: NaturalCompressedSupport


@dataclass(frozen=True)
class TrainingTeacherSupport:
    candidate_ids: torch.Tensor
    valid_mask: torch.Tensor
    retained_mask: torch.Tensor
    natural_positive_hit: torch.Tensor


def candidate_membership_mask(
    candidate_ids: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    member_ids: torch.Tensor,
    member_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return label-free set membership for batched candidate supports."""

    if candidate_ids.ndim != 2 or candidate_valid_mask.shape != candidate_ids.shape:
        raise ValueError("candidate ids and validity must match [batch, candidates]")
    if member_ids.ndim != 2 or member_valid_mask.shape != member_ids.shape:
        raise ValueError("member ids and validity must match [batch, members]")
    if int(candidate_ids.size(0)) != int(member_ids.size(0)):
        raise ValueError("candidate and member supports must share a batch")
    candidate_valid = candidate_valid_mask.to(
        device=candidate_ids.device,
        dtype=torch.bool,
    )
    member_valid = member_valid_mask.to(device=candidate_ids.device, dtype=torch.bool)
    membership = (
        candidate_ids.unsqueeze(-1)
        == member_ids.to(device=candidate_ids.device).unsqueeze(1)
    ) & member_valid.unsqueeze(1)
    return candidate_valid & membership.any(dim=-1)


def direct_natural_support(
    candidate_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    skill_count: int,
) -> NaturalCompressedSupport:
    """Return an unordered natural support without learned compression."""

    if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
        raise ValueError("direct candidate ids and validity must match [batch, k]")
    sentinel = int(skill_count)
    if sentinel <= 0:
        raise ValueError("direct candidate support requires a nonempty skill table")
    ordered = torch.where(
        valid_mask.to(device=candidate_ids.device, dtype=torch.bool),
        candidate_ids,
        torch.full_like(candidate_ids, sentinel),
    ).sort(dim=-1).values
    valid = ordered.lt(sentinel)
    canonical_ids = ordered.clamp_max(sentinel - 1)
    scores = torch.zeros(
        canonical_ids.shape,
        device=canonical_ids.device,
        dtype=torch.float32,
    ).masked_fill(~valid, torch.finfo(torch.float32).min)
    return NaturalCompressedSupport(
        coarse_candidate_ids=canonical_ids,
        coarse_valid_mask=valid,
        candidate_ids=canonical_ids,
        valid_mask=valid,
        compression_logits=scores,
    )


def _validate_scores(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or legal_mask.shape != logits.shape:
        raise ValueError("candidate logits and legal mask must match [batch, skills]")
    legal = legal_mask.to(device=logits.device, dtype=torch.bool)
    if bool((~legal.any(dim=-1)).any().item()):
        raise ValueError("every candidate row requires at least one legal skill")
    return legal


def masked_topk_tensor(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized legal Top-K without per-row CPU list materialization."""

    legal = _validate_scores(logits, legal_mask)
    width = min(max(1, int(k)), int(logits.size(1)))
    floor = torch.finfo(logits.dtype).min
    candidate_ids = torch.topk(
        logits.masked_fill(~legal, floor),
        k=width,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    valid = legal.gather(1, candidate_ids)
    return candidate_ids, valid


def static_plus_dynamic_extra_support(
    static_candidate_ids: torch.Tensor,
    static_valid_mask: torch.Tensor,
    dynamic_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    admit_extras_mask: torch.Tensor,
    *,
    dynamic_extra_k: int,
) -> tuple[NaturalCompressedSupport, torch.Tensor, torch.Tensor]:
    """Union a frozen static support with label-free memory-only proposals.

    Dynamic extras are the memory-conditioned Top-K set difference relative to
    the anchored static support.  Merely being ranked below the static cutoff
    is insufficient: a skill must enter the dynamic Top-K before it can be
    admitted. Rows whose proposal query is unchanged receive no extras.
    """

    if static_candidate_ids.ndim != 2 or static_valid_mask.shape != static_candidate_ids.shape:
        raise ValueError("static support ids and validity must match [batch, candidates]")
    legal = _validate_scores(dynamic_logits, legal_mask)
    if int(static_candidate_ids.size(0)) != int(dynamic_logits.size(0)):
        raise ValueError("static support and dynamic logits must share a batch")
    if admit_extras_mask.ndim != 1 or int(admit_extras_mask.numel()) != int(
        dynamic_logits.size(0)
    ):
        raise ValueError("extra-admission mask must have one value per candidate row")
    if int(dynamic_extra_k) <= 0:
        raise ValueError("dynamic_extra_k must be positive")
    if bool(
        (
            (static_candidate_ids < 0)
            | (static_candidate_ids >= int(dynamic_logits.size(1)))
        ).any().item()
    ):
        raise ValueError("static support index is outside the skill table")

    static_valid = static_valid_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    occupied_count = torch.zeros_like(legal, dtype=torch.int32)
    occupied_count.scatter_add_(
        1,
        static_candidate_ids.to(device=dynamic_logits.device),
        static_valid.to(dtype=torch.int32),
    )
    occupied = occupied_count.gt(0)
    dynamic_width = min(int(static_candidate_ids.size(1)), int(dynamic_logits.size(1)))
    floor = torch.finfo(dynamic_logits.dtype).min
    dynamic_top_ids = torch.topk(
        dynamic_logits.masked_fill(~legal, floor),
        k=dynamic_width,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    dynamic_top_valid = legal.gather(1, dynamic_top_ids)
    dynamic_top_outside_static = (
        dynamic_top_valid
        & ~occupied.gather(1, dynamic_top_ids)
        & admit_extras_mask.to(
            device=dynamic_logits.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
    )
    extra_width = min(max(1, int(dynamic_extra_k)), dynamic_width)
    dynamic_top_scores = dynamic_logits.gather(1, dynamic_top_ids)
    extra_positions = torch.topk(
        dynamic_top_scores.masked_fill(~dynamic_top_outside_static, floor),
        k=extra_width,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    extra_ids = dynamic_top_ids.gather(1, extra_positions)
    extra_valid = dynamic_top_outside_static.gather(1, extra_positions)
    union_ids = torch.cat(
        (static_candidate_ids.to(device=dynamic_logits.device), extra_ids),
        dim=-1,
    )
    union_valid = torch.cat((static_valid, extra_valid), dim=-1)
    union_logits = dynamic_logits.gather(1, union_ids).masked_fill(
        ~union_valid,
        torch.finfo(dynamic_logits.dtype).min,
    )
    support = NaturalCompressedSupport(
        coarse_candidate_ids=static_candidate_ids.to(device=dynamic_logits.device),
        coarse_valid_mask=static_valid,
        candidate_ids=union_ids,
        valid_mask=union_valid,
        compression_logits=union_logits,
    )
    return support, extra_ids, extra_valid


def natural_compressed_support(
    coarse_candidate_ids: torch.Tensor,
    coarse_valid_mask: torch.Tensor,
    compression_logits: torch.Tensor,
    *,
    compressed_m: int,
) -> NaturalCompressedSupport:
    """Compress natural candidates without accepting labels or positives."""

    if coarse_candidate_ids.ndim != 2:
        raise ValueError("coarse candidate ids must be rank-2")
    if coarse_valid_mask.shape != coarse_candidate_ids.shape:
        raise ValueError("coarse candidate validity must match ids")
    if compression_logits.shape != coarse_candidate_ids.shape:
        raise ValueError("compression logits must match coarse candidate ids")
    coarse_valid = coarse_valid_mask.to(
        device=compression_logits.device,
        dtype=torch.bool,
    )
    if bool((~coarse_valid.any(dim=-1)).any().item()):
        raise ValueError("every compressed row requires a natural coarse candidate")
    positions, compressed_valid = masked_topk_tensor(
        compression_logits,
        coarse_valid,
        k=compressed_m,
    )
    candidate_ids = coarse_candidate_ids.to(device=positions.device).gather(1, positions)
    return NaturalCompressedSupport(
        coarse_candidate_ids=coarse_candidate_ids,
        coarse_valid_mask=coarse_valid,
        candidate_ids=candidate_ids,
        valid_mask=compressed_valid,
        compression_logits=compression_logits.gather(1, positions).masked_fill(
            ~compressed_valid,
            torch.finfo(compression_logits.dtype).min,
        ),
    )


def label_natural_support(
    candidate_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    full_positive_mask: torch.Tensor,
) -> NaturalSupportLabels:
    """Label an existing support without adding or replacing candidates."""

    if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
        raise ValueError("natural support ids and validity must match [batch, candidates]")
    if full_positive_mask.ndim != 2 or int(full_positive_mask.size(0)) != int(
        candidate_ids.size(0)
    ):
        raise ValueError("full positive mask must match the natural support batch")
    if bool(
        (
            (candidate_ids < 0)
            | (candidate_ids >= int(full_positive_mask.size(1)))
        ).any().item()
    ):
        raise ValueError("natural support candidate index is outside the skill table")
    valid = valid_mask.to(device=candidate_ids.device, dtype=torch.bool)
    positive = full_positive_mask.to(
        device=candidate_ids.device,
        dtype=torch.bool,
    ).gather(1, candidate_ids) & valid
    negative = valid & ~positive
    return NaturalSupportLabels(
        positive_mask=positive,
        eligible_mask=positive.any(dim=-1) & negative.any(dim=-1),
    )


def training_only_teacher_retained_support(
    natural_support: NaturalCompressedSupport,
    full_positive_mask: torch.Tensor,
    full_scores: torch.Tensor,
    retain_rows: torch.Tensor,
) -> TrainingTeacherSupport:
    """Retain one positive only for training-time route teacher forcing.

    The input ``natural_support`` remains untouched and must still be used for
    recall/compression supervision and every held-out/evaluation metric.  This
    helper is intentionally label-aware and therefore must never be called by
    a candidate generator or evaluator.
    """

    candidate_ids = natural_support.candidate_ids
    valid_mask = natural_support.valid_mask
    if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
        raise ValueError("teacher support requires rank-2 natural candidates")
    if (
        full_positive_mask.ndim != 2
        or full_scores.shape != full_positive_mask.shape
        or int(full_positive_mask.size(0)) != int(candidate_ids.size(0))
    ):
        raise ValueError("teacher full-pool labels and scores must match the batch")
    if retain_rows.ndim != 1 or int(retain_rows.numel()) != int(candidate_ids.size(0)):
        raise ValueError("teacher retention mask must have one value per row")
    if bool(
        (
            (candidate_ids < 0)
            | (candidate_ids >= int(full_positive_mask.size(1)))
        ).any().item()
    ):
        raise ValueError("teacher natural candidate index is outside the skill table")

    valid = valid_mask.to(device=candidate_ids.device, dtype=torch.bool)
    positive = full_positive_mask.to(device=candidate_ids.device, dtype=torch.bool)
    natural_positive = positive.gather(1, candidate_ids) & valid
    natural_hit = natural_positive.any(dim=-1)
    requested = retain_rows.to(device=candidate_ids.device, dtype=torch.bool)
    retained = requested & ~natural_hit & positive.any(dim=-1)

    teacher_ids = candidate_ids.clone()
    teacher_valid = valid.clone()
    if bool(retained.any().item()):
        scores = full_scores.to(device=candidate_ids.device)
        floor = torch.finfo(scores.dtype).min
        best_positive = scores.masked_fill(~positive, floor).argmax(dim=-1)
        width = int(candidate_ids.size(1))
        positions = torch.arange(width, device=candidate_ids.device).unsqueeze(0)
        invalid_slot = positions.masked_fill(valid, width).amin(dim=-1)
        last_valid_slot = positions.masked_fill(~valid, -1).amax(dim=-1)
        replacement_slot = torch.where(
            invalid_slot.lt(width),
            invalid_slot,
            last_valid_slot,
        )
        if bool((replacement_slot[retained] < 0).any().item()):
            raise ValueError("teacher retention requires at least one candidate slot")
        row_ids = retained.nonzero(as_tuple=False).view(-1)
        slots = replacement_slot.index_select(0, row_ids)
        teacher_ids[row_ids, slots] = best_positive.index_select(0, row_ids)
        teacher_valid[row_ids, slots] = True

    return TrainingTeacherSupport(
        candidate_ids=teacher_ids,
        valid_mask=teacher_valid,
        retained_mask=retained,
        natural_positive_hit=natural_hit,
    )
