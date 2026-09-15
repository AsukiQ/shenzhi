import torch

from clstr.bge_reranker import bge_reranker_scores_from_logits


def test_bge_reranker_uses_single_classifier_logit() -> None:
    logits = torch.tensor([[1.25], [-0.5], [0.0]])

    assert bge_reranker_scores_from_logits(logits) == [1.25, -0.5, 0.0]


def test_bge_reranker_uses_logit_diff_for_two_class_classifier() -> None:
    logits = torch.tensor([[0.25, 1.25], [1.5, -0.5]])

    assert bge_reranker_scores_from_logits(logits) == [1.0, -2.0]
