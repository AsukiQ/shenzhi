from __future__ import annotations

import pytest
import torch

from clstr.safe_memory_ranking import (
    bounded_memory_fusion,
    build_local_candidate_masks,
    candidate_provenance_positive_residual_fusion,
    safe_local_route_objective,
)


def test_bounded_memory_fusion_alpha_zero_is_exact_static() -> None:
    static = torch.tensor([[4.0, 2.0, -1.0]])
    dynamic = torch.tensor([[-3.0, 9.0, 5.0]])
    valid = torch.tensor([[True, True, True]])

    fused = bounded_memory_fusion(
        static,
        dynamic,
        torch.zeros(1),
        valid,
        residual_bound=1.5,
    )

    assert torch.equal(fused, static)


def test_bounded_memory_fusion_is_shift_invariant_and_bounded() -> None:
    static = torch.tensor([[0.0, 1.0, 2.0, 100.0]])
    dynamic = torch.tensor([[9.0, -4.0, 7.0, -100.0]])
    valid = torch.tensor([[True, True, True, False]])

    fused = bounded_memory_fusion(
        static,
        dynamic,
        torch.ones(1),
        valid,
        residual_bound=0.75,
    )
    shifted = bounded_memory_fusion(
        static,
        dynamic + 37.0,
        torch.ones(1),
        valid,
        residual_bound=0.75,
    )

    assert torch.allclose(fused[valid], shifted[valid])
    assert torch.all((fused[valid] - static[valid]).abs() <= 0.75 + 1.0e-6)
    assert fused[0, 3] == torch.finfo(static.dtype).min


def test_bounded_memory_fusion_backpropagates_to_dynamic_and_alpha() -> None:
    static = torch.tensor([[2.0, 1.0, 0.0]])
    dynamic = torch.tensor([[0.0, 3.0, 1.0]], requires_grad=True)
    alpha = torch.tensor([0.4], requires_grad=True)
    valid = torch.tensor([[True, True, True]])

    fused = bounded_memory_fusion(
        static,
        dynamic,
        alpha,
        valid,
        residual_bound=2.0,
    )
    loss = -torch.log_softmax(fused, dim=-1)[0, 1]
    loss.backward()

    assert dynamic.grad is not None
    assert float(dynamic.grad.abs().sum()) > 0.0
    assert alpha.grad is not None
    assert float(alpha.grad.abs().sum()) > 0.0


def test_candidate_provenance_fusion_changes_only_dynamic_extras_with_positive_bonus() -> None:
    static = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    dynamic = torch.tensor([[25.0, -20.0, 8.0, -6.0]])
    valid = torch.ones_like(static, dtype=torch.bool)
    extra = torch.tensor([[False, False, True, True]])

    fused = candidate_provenance_positive_residual_fusion(
        static,
        dynamic,
        extra,
        valid,
        torch.tensor([2.0]),
        residual_bound=2.0,
    )

    assert torch.equal(fused[0, :2], static[0, :2])
    assert fused[0, 2] > static[0, 2]
    assert fused[0, 2] - static[0, 2] <= 2.0 + 1.0e-6
    assert fused[0, 3] == static[0, 3]


def test_candidate_provenance_fusion_is_exact_static_without_extras_or_history() -> None:
    static = torch.tensor([[3.0, 2.0], [5.0, 1.0], [4.0, 0.0]])
    dynamic = torch.tensor([[0.0, 9.0], [0.0, 9.0], [0.0, 9.0]])
    valid = torch.tensor([[True, True], [True, True], [True, False]])
    extra = torch.tensor([[False, False], [False, True], [False, True]])

    fused = candidate_provenance_positive_residual_fusion(
        static,
        dynamic,
        extra,
        valid,
        torch.tensor([3.0, 0.0, 2.0]),
        residual_bound=2.0,
    )

    assert torch.equal(fused[0], static[0])
    assert torch.equal(fused[1], static[1])
    assert fused[2, 0] == static[2, 0]
    assert fused[2, 1] == torch.finfo(static.dtype).min


def test_candidate_provenance_fusion_rejects_a_tunable_residual_bound() -> None:
    with pytest.raises(ValueError, match="must equal 2.0"):
        candidate_provenance_positive_residual_fusion(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.tensor([[False, True]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1),
            residual_bound=2.1,
        )


@pytest.mark.parametrize(
    ("alpha", "valid", "residual_bound", "message"),
    [
        (torch.tensor([[0.5]]), torch.ones(1, 3, dtype=torch.bool), 1.0, "alpha"),
        (torch.tensor([0.5]), torch.ones(1, 2, dtype=torch.bool), 1.0, "matching"),
        (torch.tensor([1.5]), torch.ones(1, 3, dtype=torch.bool), 1.0, r"\[0, 1\]"),
        (torch.tensor([0.5]), torch.ones(1, 3, dtype=torch.bool), 0.0, "positive"),
    ],
)
def test_bounded_memory_fusion_rejects_invalid_contracts(
    alpha: torch.Tensor,
    valid: torch.Tensor,
    residual_bound: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bounded_memory_fusion(
            torch.zeros(1, 3),
            torch.ones(1, 3),
            alpha,
            valid,
            residual_bound=residual_bound,
        )


def test_explicit_local_candidate_mask_preserves_the_declared_inventory() -> None:
    static = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0]])
    dynamic = torch.tensor([[0.0, 9.0, 3.0, 8.0, 2.0]])
    positive = torch.tensor([[False, False, False, True, False]])
    valid = torch.tensor([[False, True, False, True, True]])

    masks = build_local_candidate_masks(
        static,
        dynamic,
        positive,
        valid,
        torch.tensor([True]),
        candidate_sizes=(2, 3, 4),
    )

    assert masks.source_row_indices.tolist() == [0]
    assert masks.requested_sizes.tolist() == [3]
    assert torch.equal(masks.masks[0], valid[0])


def test_synthetic_local_masks_include_positive_static_and_disagreement_negatives() -> None:
    static = torch.tensor([[10.0, 9.0, 8.0, 1.0, 0.0, -1.0]])
    dynamic = torch.tensor([[0.0, 1.0, 2.0, 12.0, 3.0, 4.0]])
    positive = torch.tensor([[False, False, True, False, False, False]])
    valid = torch.ones_like(positive)

    masks = build_local_candidate_masks(
        static,
        dynamic,
        positive,
        valid,
        torch.tensor([False]),
        candidate_sizes=(2, 3, 4, 5, 8, 10),
    )

    assert masks.requested_sizes.tolist() == [2, 3, 4, 5, 6]
    for mask in masks.masks:
        assert bool(mask[2])
        assert bool((mask & ~positive[0]).any())
    assert bool(masks.masks[1, 0])  # strongest frozen-static negative
    assert bool(masks.masks[1, 3])  # strongest dynamic-minus-static disagreement


def test_synthetic_local_mask_ties_are_deterministic_by_skill_index() -> None:
    static = torch.zeros(1, 5)
    dynamic = torch.zeros(1, 5)
    positive = torch.tensor([[False, False, False, False, True]])
    valid = torch.ones_like(positive)

    first = build_local_candidate_masks(
        static,
        dynamic,
        positive,
        valid,
        torch.tensor([False]),
        candidate_sizes=(3,),
    )
    second = build_local_candidate_masks(
        static,
        dynamic,
        positive,
        valid,
        torch.tensor([False]),
        candidate_sizes=(3,),
    )

    assert torch.equal(first.masks, second.masks)
    assert first.masks[0].nonzero(as_tuple=False).view(-1).tolist() == [0, 1, 4]


def test_synthetic_local_mask_does_not_full_sort_large_skill_pool(monkeypatch) -> None:
    skill_count = 1000
    static = torch.linspace(1.0, 0.0, skill_count).unsqueeze(0)
    dynamic = torch.linspace(0.0, 1.0, skill_count).unsqueeze(0)
    positive = torch.zeros(1, skill_count, dtype=torch.bool)
    positive[0, 500] = True
    valid = torch.ones_like(positive)
    original_argsort = torch.argsort

    def guarded_argsort(values, *args, **kwargs):
        if int(values.numel()) > 10:
            raise AssertionError("local mask construction must not full-sort the skill pool")
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(torch, "argsort", guarded_argsort)

    masks = build_local_candidate_masks(
        static,
        dynamic,
        positive,
        valid,
        torch.tensor([False]),
        candidate_sizes=(2, 3, 4, 5, 8, 10),
    )

    assert masks.masks.shape == (6, skill_count)


def test_safe_local_objective_penalizes_static_correct_rank_damage() -> None:
    static = torch.tensor([[4.0, 2.0, 0.0]])
    fused = torch.tensor([[1.0, 3.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[True, False, False]])
    masks = torch.tensor([[True, True, True]])

    output = safe_local_route_objective(
        fused_logits=fused,
        static_logits=static,
        positive_mask=positive,
        candidate_masks=masks,
        source_row_indices=torch.tensor([0]),
        row_weights=torch.ones(1),
        gain_margin=0.5,
        safety_tolerance=0.1,
    )

    assert output.strong_count == 1
    assert output.weak_count == 0
    assert float(output.safety_loss.detach()) > 0.0
    assert float(output.gain_loss.detach()) == 0.0
    output.loss.backward()
    assert fused.grad is not None
    assert float(fused.grad.abs().sum()) > 0.0


def test_safe_local_objective_rewards_gain_when_static_is_wrong() -> None:
    static = torch.tensor([[1.0, 4.0, 0.0]])
    fused = torch.tensor([[5.0, 2.0, 0.0]])
    positive = torch.tensor([[True, False, False]])

    output = safe_local_route_objective(
        fused_logits=fused,
        static_logits=static,
        positive_mask=positive,
        candidate_masks=torch.tensor([[True, True, True]]),
        source_row_indices=torch.tensor([0]),
        row_weights=torch.tensor([2.0]),
        gain_margin=0.5,
        safety_tolerance=0.1,
    )

    assert output.weak_count == 1
    assert output.strong_count == 0
    assert float(output.gain_loss) == 0.0
    assert float(output.nll_loss) > 0.0


def test_safe_local_objective_returns_zero_for_no_eligible_masks() -> None:
    fused = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = safe_local_route_objective(
        fused_logits=fused,
        static_logits=fused.detach(),
        positive_mask=torch.tensor([[True, False]]),
        candidate_masks=torch.tensor([[True, False]]),
        source_row_indices=torch.tensor([0]),
        row_weights=torch.ones(1),
        gain_margin=0.5,
        safety_tolerance=0.1,
    )

    assert output.eligible_count == 0
    assert float(output.loss.detach()) == 0.0
