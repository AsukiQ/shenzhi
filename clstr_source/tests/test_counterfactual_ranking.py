import pytest
import torch

from clstr.counterfactual_ranking import (
    counterfactual_utility_loss,
    full_pool_causal_route_objective,
    multi_positive_log_utility,
    shuffled_history_utility_loss,
)


def test_counterfactual_utility_detects_negative_growth_that_fools_positive_margin():
    static = torch.tensor([[2.0, 2.5, 0.0]], requires_grad=True)
    dynamic = torch.tensor([[2.2, 3.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    valid = torch.ones_like(positives)

    output = counterfactual_utility_loss(
        dynamic_logits=dynamic,
        static_logits=static,
        positive_mask=positives,
        valid_mask=valid,
        gain_margin=0.1,
        safety_tolerance=0.0,
    )

    assert output.gain_loss > 0
    assert output.weak_count == 1
    output.loss.backward()
    assert dynamic.grad is not None
    assert static.grad is None


def test_counterfactual_utility_protects_static_correct_rows():
    static = torch.tensor([[4.0, 1.0, 0.0]], requires_grad=True)
    dynamic = torch.tensor([[1.0, 4.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    valid = torch.ones_like(positives)

    output = counterfactual_utility_loss(
        dynamic_logits=dynamic,
        static_logits=static,
        positive_mask=positives,
        valid_mask=valid,
        gain_margin=0.1,
        safety_tolerance=0.01,
    )

    assert output.safety_loss > 0
    assert output.strong_count == 1
    assert output.safety_violation_count == 1


def test_multi_positive_log_utility_is_alias_permutation_invariant():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    first = multi_positive_log_utility(
        logits,
        torch.tensor([[True, True, False]]),
        valid,
    )
    permuted = multi_positive_log_utility(
        logits[:, [1, 0, 2]],
        torch.tensor([[True, True, False]]),
        valid,
    )

    assert torch.allclose(first, permuted)


def test_shuffled_history_utility_loss_rewards_true_history_over_shuffled_history():
    positives = torch.tensor([[True, False, False]])
    valid = torch.ones_like(positives)
    true_better = shuffled_history_utility_loss(
        true_logits=torch.tensor([[4.0, 1.0, 0.0]]),
        shuffled_logits=torch.tensor([[1.0, 4.0, 0.0]]),
        positive_mask=positives,
        valid_mask=valid,
        margin=0.1,
    )
    shuffled_better = shuffled_history_utility_loss(
        true_logits=torch.tensor([[1.0, 4.0, 0.0]]),
        shuffled_logits=torch.tensor([[4.0, 1.0, 0.0]]),
        positive_mask=positives,
        valid_mask=valid,
        margin=0.1,
    )

    assert true_better.item() == 0.0
    assert shuffled_better.item() > 0.0


def test_counterfactual_utility_empty_eligible_rows_return_finite_zero():
    dynamic = torch.tensor([[1.0, torch.finfo(torch.float32).min]], requires_grad=True)
    static = dynamic.detach().clone().requires_grad_(True)
    positives = torch.tensor([[True, False]])
    valid = torch.tensor([[True, False]])

    output = counterfactual_utility_loss(
        dynamic_logits=dynamic,
        static_logits=static,
        positive_mask=positives,
        valid_mask=valid,
        gain_margin=0.1,
        safety_tolerance=0.0,
    )

    assert torch.isfinite(output.loss)
    assert output.loss.item() == 0.0
    assert output.weak_count == 0
    assert output.strong_count == 0


def test_full_pool_causal_route_objective_uses_row_weights_and_named_exclusions():
    dynamic = torch.tensor([[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]], requires_grad=True)
    static = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False], [False, False, False]])
    valid = torch.ones_like(positives)

    output = full_pool_causal_route_objective(
        dynamic_logits=dynamic,
        static_logits=static,
        positive_mask=positives,
        valid_mask=valid,
        row_weights=torch.tensor([2.0, 1.0]),
        gain_margin=0.1,
        safety_tolerance=0.01,
    )

    expected = -multi_positive_log_utility(dynamic[:1], positives[:1], valid[:1]).mean()
    assert torch.allclose(output.main_loss, expected)
    assert output.eligible_mask.tolist() == [True, False]
    assert output.exclusion_counts["eligible_rows"] == 1
    assert output.exclusion_counts["no_valid_positive_rows"] == 1


@pytest.mark.parametrize("argument", ["gain_margin", "safety_tolerance", "gain_weight", "safety_weight"])
def test_counterfactual_utility_rejects_negative_hyperparameters(argument):
    kwargs = {
        "gain_margin": 0.1,
        "safety_tolerance": 0.0,
        "gain_weight": 1.0,
        "safety_weight": 1.0,
    }
    kwargs[argument] = -0.1

    with pytest.raises(ValueError, match="nonnegative"):
        counterfactual_utility_loss(
            dynamic_logits=torch.tensor([[1.0, 0.0]]),
            static_logits=torch.tensor([[1.0, 0.0]]),
            positive_mask=torch.tensor([[True, False]]),
            valid_mask=torch.tensor([[True, True]]),
            **kwargs,
        )
