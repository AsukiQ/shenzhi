import torch

from clstr.success_value import QSuccessHead, compute_q_success_loss, q_success_scores


def test_q_success_head_scores_candidate_actions_and_computes_weighted_bce():
    head = QSuccessHead(3)
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        head.net[-1].bias.fill_(0.0)

    state = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    memory = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])

    logits = q_success_scores(head, state, memory, candidates, mask)
    loss, metrics = compute_q_success_loss(logits, labels, weights, mask)

    assert logits.shape == (1, 2)
    assert loss.item() > 0.0
    assert metrics["q_success_bce_loss"] > 0.0
    assert metrics["q_success_sample_count"] == 2


def test_q_success_loss_ignores_masked_candidates():
    logits = torch.tensor([[0.0, 100.0]], dtype=torch.float32)
    labels = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    loss, metrics = compute_q_success_loss(logits, labels, weights, mask)

    assert abs(loss.item() - torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([1.0])).item()) < 1e-6
    assert metrics["q_success_sample_count"] == 1
