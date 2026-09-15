from __future__ import annotations

import pytest
import torch

from clstr.counterfactual_memory_calibration import (
    RouteMemoryResidualAdapter,
    counterfactual_memory_calibration_loss,
    score_cmc_candidates,
)


def test_zero_initialized_adapter_preserves_dynamic_route() -> None:
    adapter = RouteMemoryResidualAdapter(4, hidden_dim=64)

    delta = adapter(torch.ones(2, 4), torch.ones(2, 4))

    assert torch.equal(delta, torch.zeros_like(delta))


def test_cmc_exact_endpoints_and_harmful_memory_pushes_alpha_down() -> None:
    alpha = torch.tensor([0.5], requires_grad=True)
    dynamic_logits = torch.tensor([[1.0, 4.0]], requires_grad=True)

    result = counterfactual_memory_calibration_loss(
        static_logits=torch.tensor([[4.0, 1.0]]),
        dynamic_logits=dynamic_logits,
        alpha=alpha,
        positive_mask=torch.tensor([[True, False]]),
        valid_mask=torch.tensor([[True, True]]),
    )

    assert torch.equal(result.alpha_zero_logits, result.static_logits)
    assert torch.equal(result.alpha_one_logits, result.dynamic_logits)
    result.loss.backward()
    assert alpha.grad is not None and alpha.grad.item() > 0.0
    assert dynamic_logits.grad is not None


def test_cmc_helpful_memory_pushes_alpha_up() -> None:
    alpha = torch.tensor([0.5], requires_grad=True)

    result = counterfactual_memory_calibration_loss(
        static_logits=torch.tensor([[1.0, 4.0]]),
        dynamic_logits=torch.tensor([[4.0, 1.0]], requires_grad=True),
        alpha=alpha,
        positive_mask=torch.tensor([[True, False]]),
        valid_mask=torch.tensor([[True, True]]),
    )
    result.loss.backward()

    assert alpha.grad is not None and alpha.grad.item() < 0.0


def test_cmc_rejects_rows_without_a_valid_positive() -> None:
    with pytest.raises(ValueError, match="valid positive"):
        counterfactual_memory_calibration_loss(
            static_logits=torch.tensor([[2.0, 1.0]]),
            dynamic_logits=torch.tensor([[1.0, 2.0]]),
            alpha=torch.tensor([0.5]),
            positive_mask=torch.tensor([[False, True]]),
            valid_mask=torch.tensor([[True, False]]),
        )


def test_shared_cmc_candidate_scoring_applies_adapter_gate_and_exact_zero_history() -> None:
    class _Adapter(torch.nn.Module):
        def forward(self, h, memory_delta):
            del memory_delta
            return torch.tensor([[0.0, 4.0], [0.0, 4.0]], dtype=h.dtype)

    class _Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = None

        def forward(self, features):
            self.features = features.detach().clone()
            return torch.ones(features.size(0), dtype=features.dtype)

    gate = _Gate()
    model = type(
        "_Model",
        (),
        {
            "route_memory_residual_adapter": _Adapter(),
            "route_memory_candidate_utility_gate": gate,
        },
    )()
    static = torch.tensor([[3.0, 1.0], [3.0, 1.0]])
    raw_dynamic = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    candidate_embeddings = torch.eye(2).unsqueeze(0).expand(2, -1, -1)

    result = score_cmc_candidates(
        model,
        h=torch.ones(2, 2),
        static_memory=torch.zeros(2, 2),
        dynamic_memory=torch.ones(2, 2),
        static_logits=static,
        raw_dynamic_logits=raw_dynamic,
        candidate_embeddings=candidate_embeddings,
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
        causal_update_count=torch.tensor([1.0, 0.0]),
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )

    assert torch.equal(result.dynamic_logits, torch.tensor([[1.0, 6.0], [1.0, 6.0]]))
    assert torch.equal(result.raw_alpha, torch.ones(2))
    assert torch.equal(result.effective_alpha, torch.tensor([1.0, 0.0]))
    assert torch.equal(result.fused_logits[0], result.dynamic_logits[0])
    assert torch.equal(result.fused_logits[1], result.static_logits[1])
    assert gate.features is not None
    assert gate.features[0, -2].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item() / torch.log(torch.tensor(17.0)).item()
    )
    assert gate.features[0, -1].item() == pytest.approx(
        torch.log(torch.tensor(3.0)).item() / torch.log(torch.tensor(257.0)).item()
    )
