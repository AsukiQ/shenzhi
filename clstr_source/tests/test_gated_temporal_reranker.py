import pytest
import torch

from clstr.gated_temporal_reranker import (
    GatedTemporalConfig,
    GatedTemporalReranker,
    prior_preserving_loss,
)


def test_gated_temporal_lambda_is_lower_for_confident_stage0_prior():
    reranker = GatedTemporalReranker(GatedTemporalConfig(dim=4, lambda_max=0.5))
    prior_logits = torch.tensor(
        [
            [6.0, 0.0, -1.0, -2.0],
            [0.0, -0.01, -0.02, -0.03],
        ],
        dtype=torch.float32,
    )

    lambdas, features = reranker.gate_lambda(prior_logits)

    assert lambdas.shape == (2, 1)
    assert features["stage0_margin"].tolist()[0] > features["stage0_margin"].tolist()[1]
    assert lambdas[0].item() < lambdas[1].item()
    assert 0.0 <= lambdas.min().item() <= lambdas.max().item() <= 0.5


def test_candidate_context_pooling_ignores_invalid_candidates():
    reranker = GatedTemporalReranker(GatedTemporalConfig(dim=3, lambda_max=0.5, context_top_k=2))
    query = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidate_embs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],
            ]
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[True, True, False]])

    context = reranker.candidate_context_pool(query, candidate_embs, valid_mask)

    assert context.shape == (1, 3)
    assert torch.isfinite(context).all()
    assert context[0, 2].item() == pytest.approx(0.0, abs=1.0e-6)


def test_gated_temporal_forward_returns_prior_plus_lambda_times_residual():
    reranker = GatedTemporalReranker(GatedTemporalConfig(dim=3, lambda_max=0.5, context_top_k=3))
    query = torch.tensor([[0.5, 0.25, 0.0]], dtype=torch.float32)
    candidate_embs = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    prior_logits = torch.tensor([[0.0, -0.7, -1.1]], dtype=torch.float32)

    output = reranker(query, candidate_embs, prior_logits)

    assert output.final_logits.shape == prior_logits.shape
    assert output.residual_logits.shape == prior_logits.shape
    assert output.lambda_t.shape == (1, 1)
    assert torch.allclose(output.final_logits, prior_logits + output.lambda_t * output.residual_logits)


def test_gated_temporal_forward_projects_query_when_dims_differ():
    reranker = GatedTemporalReranker(GatedTemporalConfig(dim=3, query_dim=5, lambda_max=0.5, context_top_k=3))
    query = torch.randn(2, 5)
    candidate_embs = torch.randn(2, 4, 3)
    prior_logits = torch.zeros(2, 4)

    output = reranker(query, candidate_embs, prior_logits)

    assert output.final_logits.shape == prior_logits.shape
    assert output.candidate_context.shape == (2, 3)


def test_prior_preserving_loss_penalizes_dropping_prior_top5_positive():
    prior_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    dropped_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    preserved_logits = prior_logits.clone()
    positive_mask = torch.tensor([[True, False, False, False, False, False]])

    dropped = prior_preserving_loss(
        final_logits=dropped_logits,
        prior_logits=prior_logits,
        positive_mask=positive_mask,
        rank_drop_k=5,
    )
    preserved = prior_preserving_loss(
        final_logits=preserved_logits,
        prior_logits=prior_logits,
        positive_mask=positive_mask,
        rank_drop_k=5,
    )

    assert dropped["rank_drop_loss"].item() > 0.0
    assert dropped["kl_loss"].item() > 0.0
    assert preserved["rank_drop_loss"].item() == pytest.approx(0.0, abs=1.0e-6)
