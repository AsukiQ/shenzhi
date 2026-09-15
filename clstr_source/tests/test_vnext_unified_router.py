from __future__ import annotations

import pytest
import torch

from clstr.vnext_unified_router import (
    UNIFIED_ROUTER_FEATURE_NAMES,
    UNIFIED_ROUTING_MODE_SPARSE,
    UnifiedThreeExpertRouter,
    route_with_support_aware_anchor,
    route_with_unified_three_experts,
    unified_router_features,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    foundation = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    static = torch.tensor([[3.0, 2.0, -100.0], [0.0, 1.0, 3.0]])
    recurrent = torch.tensor([[1.0, 4.0, 0.0], [0.0, 4.0, 1.0]])
    valid = torch.ones_like(foundation, dtype=torch.bool)
    static_valid = torch.tensor([[True, True, False], [True, True, True]])
    depth = torch.tensor([0.0, 3.0])
    corrections = torch.tensor([0.0, 2.0])
    pool = torch.tensor([3.0, 1000.0])
    return foundation, static, recurrent, valid, static_valid, depth, corrections, pool


def test_unified_router_features_are_finite_and_source_free() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    features = unified_router_features(
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=static_valid,
        history_depth=depth,
        observation_correction_count=corrections,
        legal_pool_size=pool,
    )
    assert features.shape == (2, len(UNIFIED_ROUTER_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    assert not any("benchmark" in name or "source" in name for name in UNIFIED_ROUTER_FEATURE_NAMES)
    correction_index = UNIFIED_ROUTER_FEATURE_NAMES.index(
        "observation_correction_fraction"
    )
    assert torch.allclose(features[:, correction_index], torch.tensor([0.0, 2.0 / 3.0]))


def test_unified_router_residual_ignores_static_floor_on_memory_only_extras() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    static = static.clone()
    static[0, 2] = torch.finfo(static.dtype).min
    features = unified_router_features(
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=static_valid,
        history_depth=depth,
        observation_correction_count=corrections,
        legal_pool_size=pool,
    )
    residual_index = UNIFIED_ROUTER_FEATURE_NAMES.index(
        "recurrent_static_residual_rms"
    )
    expected = torch.tensor(((1.0 - 3.0) ** 2 + (4.0 - 2.0) ** 2) / 2.0).sqrt()
    recurrent_scale = torch.tensor((1.0**2 + 4.0**2) / 2.0).sqrt()
    assert torch.isfinite(features).all()
    assert torch.allclose(
        features[0, residual_index],
        expected / recurrent_scale,
    )


def test_unified_router_assigns_zero_memory_weight_without_history() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    router = UnifiedThreeExpertRouter()
    scores = route_with_unified_three_experts(
        router,
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=static_valid,
        history_depth=depth,
        observation_correction_count=corrections,
        legal_pool_size=pool,
    )
    assert scores.expert_weights.shape == (2, 3)
    assert int(scores.expert_weights[0].argmax().item()) == 0
    assert scores.expert_weights[0, 2].item() == 0.0
    assert scores.expert_weights[1, 2].item() > 0.0
    assert torch.allclose(scores.expert_weights.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(scores.logits).all()


def test_unified_router_rejects_more_corrections_than_history_events() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    corrections = corrections.clone()
    corrections[1] = depth[1] + 1
    with pytest.raises(
        ValueError,
        match="cannot exceed replay history depth",
    ):
        unified_router_features(
            foundation,
            static,
            recurrent,
            valid,
            static_support_mask=static_valid,
            history_depth=depth,
            observation_correction_count=corrections,
            legal_pool_size=pool,
        )


def test_unified_router_backpropagates_only_through_router_when_logits_detached() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    router = UnifiedThreeExpertRouter()
    scores = route_with_unified_three_experts(
        router,
        foundation.detach(),
        static.detach(),
        recurrent.detach(),
        valid,
        static_support_mask=static_valid,
        history_depth=depth,
        observation_correction_count=corrections,
        legal_pool_size=pool,
    )
    loss = -scores.logits[1, 1]
    loss.backward()
    assert all(parameter.grad is not None for parameter in router.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in router.parameters())


def test_unified_sparse_router_deploys_one_expert_per_row() -> None:
    foundation, static, recurrent, valid, static_valid, depth, corrections, pool = _inputs()
    router = UnifiedThreeExpertRouter()
    scores = route_with_unified_three_experts(
        router,
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=static_valid,
        history_depth=depth,
        observation_correction_count=corrections,
        legal_pool_size=pool,
        routing_mode=UNIFIED_ROUTING_MODE_SPARSE,
    )
    assert torch.equal(
        scores.expert_weights,
        torch.nn.functional.one_hot(
            scores.router_probabilities.argmax(dim=-1),
            num_classes=3,
        ).to(dtype=scores.expert_weights.dtype),
    )
    assert torch.allclose(scores.expert_weights.sum(dim=-1), torch.ones(2))
    assert scores.expert_weights[0, 2].item() == 0.0
    expected_logits = torch.stack(
        (
            foundation,
            static.masked_fill(~static_valid, torch.finfo(torch.float32).min),
            recurrent,
        ),
        dim=1,
    ).gather(
        1,
        scores.router_probabilities.argmax(dim=-1)
        .view(-1, 1, 1)
        .expand(-1, 1, 3),
    ).squeeze(1)
    assert torch.equal(scores.logits, expected_logits)
    assert torch.isfinite(scores.logits).all()


def test_support_aware_anchor_preserves_complete_support_foundation_top4() -> None:
    foundation = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    static = torch.tensor([[0.0, 0.0, 0.0, 0.0, 6.0, 5.0]])
    recurrent = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0, 6.0]])
    valid = torch.ones_like(foundation, dtype=torch.bool)
    scores = route_with_support_aware_anchor(
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=valid,
        history_depth=torch.tensor([2.0]),
        observation_correction_count=torch.tensor([1.0]),
        legal_pool_size=torch.tensor([6.0]),
    )
    order = torch.argsort(scores.logits[0], descending=True, stable=True)
    assert order[:4].tolist() == [0, 1, 2, 3]
    assert set(order[4:].tolist()) == {4, 5}
    assert torch.allclose(
        scores.expert_weights,
        torch.full((1, 3), 1.0 / 3.0),
    )


def test_support_aware_anchor_masks_recurrent_evidence_without_history() -> None:
    foundation = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    static = torch.tensor([[1.0, 2.0, 3.0, 5.0, 4.0]])
    recurrent_a = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])
    recurrent_b = torch.tensor([[0.0, 0.0, 0.0, 0.0, 100.0]])
    valid = torch.ones_like(foundation, dtype=torch.bool)

    def score(recurrent: torch.Tensor) -> torch.Tensor:
        routed = route_with_support_aware_anchor(
            foundation,
            static,
            recurrent,
            valid,
            static_support_mask=valid,
            history_depth=torch.tensor([0.0]),
            observation_correction_count=torch.tensor([0.0]),
            legal_pool_size=torch.tensor([5.0]),
        )
        assert torch.allclose(
            routed.expert_weights,
            torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 0.0]]),
        )
        return routed.logits

    assert torch.equal(score(recurrent_a), score(recurrent_b))


def test_support_aware_anchor_uses_history_only_for_incomplete_support() -> None:
    foundation = torch.tensor([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]])
    static = torch.tensor([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
    recurrent = torch.tensor([[2.0, 1.0, 4.0], [2.0, 1.0, 4.0]])
    valid = torch.ones_like(foundation, dtype=torch.bool)
    scores = route_with_support_aware_anchor(
        foundation,
        static,
        recurrent,
        valid,
        static_support_mask=valid,
        history_depth=torch.tensor([0.0, 2.0]),
        observation_correction_count=torch.tensor([0.0, 1.0]),
        legal_pool_size=torch.tensor([10.0, 10.0]),
    )
    assert torch.equal(scores.logits[0], static[0])
    assert torch.equal(scores.logits[1], recurrent[1])
    assert torch.equal(
        scores.expert_weights,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_support_aware_anchor_rejects_support_larger_than_legal_pool() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    with pytest.raises(
        ValueError,
        match="cannot exceed the legal pool",
    ):
        route_with_support_aware_anchor(
            logits,
            logits,
            logits,
            valid,
            static_support_mask=valid,
            history_depth=torch.tensor([1.0]),
            observation_correction_count=torch.tensor([0.0]),
            legal_pool_size=torch.tensor([2.0]),
        )
