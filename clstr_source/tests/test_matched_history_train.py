from __future__ import annotations

import pytest
import torch

from clstr.matched_history_train import (
    MatchedHistoryRouteScorer,
    best_positive_support_rank,
    multi_positive_support_loss,
)


def test_shared_scorer_has_exact_zero_history_fallback() -> None:
    torch.manual_seed(3)
    scorer = MatchedHistoryRouteScorer(8, hidden_dim=4)
    base = torch.randn(2, 8)
    current = torch.randn(2, 8)
    memory = torch.randn(2, 8)
    initial = torch.randn(2, 8)
    output = scorer(
        base,
        current,
        memory,
        initial,
        torch.tensor([False, True]),
    )
    assert torch.equal(output[0], base[0])
    assert not torch.equal(output[1], base[1])


def test_multi_positive_loss_uses_only_natural_support_hits() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.0], [3.0, 2.0, 1.0]])
    positive = torch.tensor([[False, True, False], [False, False, False]])
    valid = torch.ones_like(positive)
    loss, eligible = multi_positive_support_loss(logits, positive, valid)
    assert eligible.tolist() == [True, False]
    expected = torch.logsumexp(logits[0], dim=0) - logits[0, 1]
    assert torch.allclose(loss, expected)


def test_best_positive_rank_is_global_id_deterministic_and_zero_on_miss() -> None:
    logits = torch.tensor([[2.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    candidate_ids = torch.tensor([[9, 4, 7], [1, 2, 3]])
    positive = torch.tensor([[True, False, False], [False, False, False]])
    valid = torch.ones_like(positive)
    ranks = best_positive_support_rank(logits, candidate_ids, positive, valid)
    assert ranks.tolist() == [2, 0]


def test_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share"):
        multi_positive_support_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 2, dtype=torch.bool),
            torch.ones(2, 3, dtype=torch.bool),
        )
