from __future__ import annotations

import torch
import torch.nn.functional as F

from clstr.vnext_losses import (
    chunked_full_pool_hard_negative_margin,
    chunked_full_pool_multi_positive_nll,
    dense_full_pool_nll_and_hard_negative,
    head_local_counterfactual_loss,
    natural_candidate_multi_positive_nll,
    natural_candidate_topk_coverage_loss,
    no_regret_loss,
    query_expert_mixture_target_loss,
)


def test_natural_candidate_loss_skips_misses_and_never_injects() -> None:
    logits = torch.tensor([[3.0, 1.0], [2.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[True, False], [False, False]])
    valid = torch.ones_like(positive)
    output = natural_candidate_multi_positive_nll(logits, positive, valid)
    assert output.eligible_mask.tolist() == [True, False]
    assert output.report["natural_miss_rows"] == 1
    assert output.report["positive_injection_count"] == 0
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_candidate_topk_coverage_loss_targets_recoverable_boundary() -> None:
    base = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]
    )
    logits = base.clone().requires_grad_(True)
    positive = torch.tensor(
        [[True, False, False, False], [False, False, True, False]]
    )
    valid = torch.ones_like(positive)
    output = natural_candidate_topk_coverage_loss(
        logits,
        base,
        positive,
        valid,
        k=2,
        recoverable_weight=4.0,
        listwise_weight=0.05,
    )
    assert output.report["direct_topk_hit_rows"] == 1
    assert output.report["recoverable_rows"] == 1
    assert output.report["positive_injection_count"] == 0
    output.loss.backward()
    assert logits.grad is not None
    assert logits.grad[1, 2] < 0
    assert torch.isfinite(logits.grad).all()


def test_chunked_full_pool_loss_matches_dense_reference() -> None:
    torch.manual_seed(11)
    query = torch.randn(3, 5, requires_grad=True)
    skills = torch.randn(13, 5)
    legal = torch.ones(3, 13, dtype=torch.bool)
    legal[1, :2] = False
    positive = torch.zeros_like(legal)
    positive[0, [2, 4]] = True
    positive[1, [5]] = True
    positive[2, [1, 12]] = True
    temperature = 7.0
    output = chunked_full_pool_multi_positive_nll(
        query,
        skills,
        positive,
        legal,
        temperature=temperature,
        chunk_size=4,
    )
    logits = temperature * F.normalize(query, dim=-1) @ F.normalize(skills, dim=-1).t()
    floor = torch.finfo(logits.dtype).min
    dense = (
        torch.logsumexp(logits.masked_fill(~legal, floor), dim=-1)
        - torch.logsumexp(logits.masked_fill(~(positive & legal), floor), dim=-1)
    ).mean()
    assert torch.allclose(output.loss, dense, atol=1.0e-6)
    output.loss.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_full_pool_loss_excludes_rows_without_legal_positive() -> None:
    query = torch.randn(2, 4)
    skills = torch.randn(6, 4)
    legal = torch.ones(2, 6, dtype=torch.bool)
    positive = torch.zeros_like(legal)
    positive[0, 2] = True
    output = chunked_full_pool_multi_positive_nll(
        query,
        skills,
        positive,
        legal,
        temperature=3.0,
        chunk_size=2,
    )
    assert output.report["eligible_rows"] == 1
    assert output.report["no_positive_rows"] == 1


def test_full_pool_loss_pre_normalized_skill_cache_matches_reference() -> None:
    torch.manual_seed(19)
    query = torch.randn(2, 4)
    skills = torch.randn(9, 4)
    legal = torch.ones(2, 9, dtype=torch.bool)
    positive = torch.zeros_like(legal)
    positive[0, 2] = True
    positive[1, 7] = True
    reference = chunked_full_pool_multi_positive_nll(
        query,
        skills,
        positive,
        legal,
        temperature=5.0,
        chunk_size=3,
    )
    cached = F.normalize(skills.float(), dim=-1).to(query.dtype)
    accelerated = chunked_full_pool_multi_positive_nll(
        query,
        cached,
        positive,
        legal,
        temperature=5.0,
        chunk_size=3,
        skill_embeddings_are_normalized=True,
    )
    assert torch.allclose(accelerated.loss, reference.loss, atol=1.0e-6)


def test_raw_unified_full_pool_and_hard_negative_match_dense_reference() -> None:
    torch.manual_seed(23)
    query = torch.randn(2, 4, requires_grad=True)
    skills = F.normalize(torch.randn(11, 4), dim=-1)
    legal = torch.ones(2, 11, dtype=torch.bool)
    legal[1, 8:] = False
    positive = torch.zeros_like(legal)
    positive[0, [1, 3]] = True
    positive[1, 2] = True
    nll = chunked_full_pool_multi_positive_nll(
        query,
        skills,
        positive,
        legal,
        temperature=1.0,
        chunk_size=3,
        skill_embeddings_are_normalized=True,
        normalize_query=False,
    )
    hard = chunked_full_pool_hard_negative_margin(
        query,
        skills,
        positive,
        legal,
        temperature=1.0,
        top_k=3,
        margin=0.1,
        chunk_size=3,
        skill_embeddings_are_normalized=True,
        normalize_query=False,
    )
    logits = query @ skills.t()
    floor = torch.finfo(logits.dtype).min
    dense_nll = (
        torch.logsumexp(logits.masked_fill(~legal, floor), dim=-1)
        - torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    ).mean()
    positive_score = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    weakest_positive_score = logits.masked_fill(~positive, torch.inf).amin(dim=-1)
    negative = logits.masked_fill(~(legal & ~positive), floor)
    mined = torch.topk(negative, k=3, dim=-1).values
    dense_hard = F.relu(0.1 + mined - positive_score.unsqueeze(-1)).mean()
    dense_coverage_hard = F.relu(
        0.1 + mined - weakest_positive_score.unsqueeze(-1)
    ).mean()
    dense_objective = dense_full_pool_nll_and_hard_negative(
        logits,
        positive,
        legal,
        hard_negative_top_k=3,
        hard_negative_margin=0.1,
        hard_negative_positive_anchor="weakest_legal_positive_v1",
    )
    assert torch.allclose(nll.loss, dense_nll, atol=1.0e-6)
    assert torch.allclose(hard.loss, dense_hard, atol=1.0e-6)
    assert torch.allclose(dense_objective.nll_loss, dense_nll, atol=1.0e-6)
    assert torch.allclose(
        dense_objective.hard_negative_loss,
        dense_coverage_hard,
        atol=1.0e-6,
    )
    assert dense_objective.report["hard_negative_positive_anchor"] == (
        "weakest_legal_positive_v1"
    )
    (nll.loss + 0.2 * hard.loss).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_stage0_hard_negative_cannot_hide_a_weak_required_positive() -> None:
    logits = torch.tensor([[3.0, 0.0, 2.0]], requires_grad=True)
    positive = torch.tensor([[True, True, False]])
    legal = torch.ones_like(positive)

    output = dense_full_pool_nll_and_hard_negative(
        logits,
        positive,
        legal,
        hard_negative_top_k=1,
        hard_negative_margin=0.1,
        hard_negative_positive_anchor="weakest_legal_positive_v1",
    )

    assert torch.allclose(output.hard_negative_loss, torch.tensor(2.1))
    output.hard_negative_loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 2] > 0


def test_stage0_static_set_anchor_preserves_alternative_positive_semantics() -> None:
    logits = torch.tensor([[3.0, 0.0, 2.0]])
    positive = torch.tensor([[True, True, False]])
    legal = torch.ones_like(positive)

    output = dense_full_pool_nll_and_hard_negative(
        logits,
        positive,
        legal,
        hard_negative_top_k=1,
        hard_negative_margin=0.1,
        hard_negative_positive_anchor="set_logsumexp_v1",
    )

    expected = F.relu(
        torch.tensor(2.1) - torch.logsumexp(torch.tensor([3.0, 0.0]), dim=0)
    )
    assert torch.allclose(output.hard_negative_loss, expected)
    assert output.report["hard_negative_positive_anchor"] == "set_logsumexp_v1"


def test_head_local_counterfactual_loss_rewards_both_correct_branches() -> None:
    good = head_local_counterfactual_loss(
        torch.tensor([3.0, 2.0]),
        torch.tensor([0.0, -1.0]),
        torch.tensor([2.5, 2.0]),
        torch.tensor([-0.5, 0.0]),
        margin=0.5,
    )
    bad = head_local_counterfactual_loss(
        torch.tensor([0.0, -1.0]),
        torch.tensor([3.0, 2.0]),
        torch.tensor([-0.5, 0.0]),
        torch.tensor([2.5, 2.0]),
        margin=0.5,
    )
    assert good < bad


def test_no_regret_is_one_sided() -> None:
    static = torch.tensor([2.0, 2.0])
    better = no_regret_loss(static, torch.tensor([3.0, 2.1]), tolerance=0.1)
    worse = no_regret_loss(static, torch.tensor([1.0, 1.5]), tolerance=0.1)
    assert better == 0
    assert worse > 0


def test_query_expert_mixture_target_rewards_natural_dynamic_benefit() -> None:
    probability = torch.tensor([0.2, 0.2, 0.2, 0.2], requires_grad=True)
    output = query_expert_mixture_target_loss(
        probability,
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.5, 1.0, -5.0]),
        static_supported_mask=torch.tensor([True, True, False, False]),
        eligible_mask=torch.ones(4, dtype=torch.bool),
        utility_scale=0.5,
    )
    assert output.target[0] > 0.5
    assert output.target[1] == 0.0
    assert output.target[2] > 0.5
    assert output.target[3] == 0.0
    assert output.report["positive_benefit_rows"] == 2
    assert output.report["static_preferred_rows"] == 2
    assert output.report["static_support_miss_rows"] == 2
    output.loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()
    assert probability.grad[0] < 0
    assert probability.grad[1] > 0
    assert probability.grad[2] < 0
    assert probability.grad[3] > 0


def test_query_expert_mixture_target_ignores_teacher_only_rows() -> None:
    probability = torch.tensor([0.1, 0.1], requires_grad=True)
    output = query_expert_mixture_target_loss(
        probability,
        torch.zeros(2),
        torch.ones(2),
        static_supported_mask=torch.tensor([False, False]),
        eligible_mask=torch.tensor([False, False]),
    )
    assert output.loss == 0
    output.loss.backward()
    assert probability.grad is not None
    assert torch.equal(probability.grad, torch.zeros_like(probability))
