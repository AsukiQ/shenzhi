from __future__ import annotations

import pytest
import torch

from clstr.candidate_admission_residual import (
    CandidateAdmissionResidualHead,
    CandidateAdmissionScoringOutput,
    candidate_admission_residual_loss,
    candidate_admission_scalar_features,
    score_candidate_admission_residual,
)


class _FixedHead(torch.nn.Module):
    def __init__(
        self,
        admission_logits: torch.Tensor,
        raw_residual: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("fixed_admission_logits", admission_logits)
        self.register_buffer("fixed_raw_residual", raw_residual)

    def forward(
        self,
        h: torch.Tensor,
        memory_delta: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del h, memory_delta, candidate_embeddings, scalar_features
        return self.fixed_admission_logits, self.fixed_raw_residual


def _score(
    head: torch.nn.Module,
    *,
    causal_update_count: torch.Tensor | None = None,
) -> CandidateAdmissionScoringOutput:
    return score_candidate_admission_residual(
        head,
        h=torch.zeros(1, 4),
        memory_delta=torch.zeros(1, 4),
        candidate_embeddings=torch.zeros(1, 4, 4),
        static_logits=torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
        dynamic_logits=torch.tensor([[4.0, 3.0, 2.0, 9.0]]),
        dynamic_extra_mask=torch.tensor([[False, True, True, False]]),
        valid_mask=torch.tensor([[True, True, True, False]]),
        causal_update_count=(
            torch.ones(1)
            if causal_update_count is None
            else causal_update_count
        ),
        residual_bound=2.0,
    )


def test_candidate_admission_scores_shared_and_extra_candidates() -> None:
    output = _score(
        _FixedHead(
            torch.tensor([[0.0, 4.0, -4.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0, 10.0]]),
        )
    )

    shared_delta = output.final_logits[0, 0] - output.static_logits[0, 0]
    admitted_delta = output.final_logits[0, 1] - output.static_logits[0, 1]
    rejected_delta = output.final_logits[0, 2] - output.static_logits[0, 2]
    assert shared_delta > 0
    assert admitted_delta > rejected_delta > 0
    assert torch.all(
        (output.final_logits[0, :3] - output.static_logits[0, :3]).abs()
        <= 2.0
    )
    assert output.final_logits[0, 3] == torch.finfo(torch.float32).min


def test_candidate_admission_zero_history_is_exact_static() -> None:
    output = _score(
        _FixedHead(
            torch.full((1, 4), 10.0),
            torch.full((1, 4), 10.0),
        ),
        causal_update_count=torch.zeros(1),
    )
    expected = output.static_logits.masked_fill(
        torch.tensor([[False, False, False, True]]),
        torch.finfo(torch.float32).min,
    )

    assert torch.equal(output.final_logits, expected)


def test_candidate_admission_zero_initialized_head_is_exact_static() -> None:
    output = _score(CandidateAdmissionResidualHead(model_dim=4))
    expected = output.static_logits.masked_fill(
        torch.tensor([[False, False, False, True]]),
        torch.finfo(torch.float32).min,
    )

    assert torch.equal(output.final_logits, expected)


def test_candidate_admission_scalar_features_are_finite_and_label_free() -> None:
    features = candidate_admission_scalar_features(
        torch.tensor([[3.0, 2.0, 1.0, 99.0]]),
        torch.tensor([[2.0, 4.0, 1.0, -99.0]]),
        torch.tensor([[False, True, True, False]]),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([2.0]),
    )

    assert features.shape == (1, 4, 8)
    assert torch.isfinite(features).all()
    assert torch.equal(features[0, 3], torch.zeros(8))
    assert features[0, 1, 5] == 1.0
    assert features[0, 0, 5] == 0.0


def test_candidate_admission_objective_trains_admission_and_no_regret() -> None:
    static = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    final = torch.tensor([[1.0, 3.0, 1.0], [3.0, 2.0, 1.0]])
    scoring = CandidateAdmissionScoringOutput(
        final_logits=final,
        static_logits=static,
        dynamic_logits=torch.tensor([[1.0, 3.0, 1.0], [3.0, 2.0, 4.0]]),
        admission_logits=torch.zeros(2, 3, requires_grad=True),
        admission_probability=torch.full((2, 3), 0.5),
        bounded_residual=final - static,
    )
    positive = torch.tensor([[True, False, False], [False, False, True]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    extras = torch.tensor([[False, False, True], [False, False, True]])

    objective = candidate_admission_residual_loss(
        scoring=scoring,
        positive_mask=positive,
        valid_mask=valid,
        dynamic_extra_mask=extras,
        admission_weight=1.0,
        counterfactual_weight=1.0,
        gain_margin=0.1,
    )

    assert objective.admission_positive_count == 1
    assert objective.admission_negative_count == 1
    assert objective.admission_loss.item() > 0
    assert objective.safety_loss.item() > 0
    assert objective.gain_loss.item() > 0
    assert objective.no_regret_loss.item() == pytest.approx(
        objective.safety_loss.item() + objective.gain_loss.item()
    )
    assert torch.isfinite(objective.loss)


def test_candidate_admission_rejects_a_tunable_residual_bound() -> None:
    with pytest.raises(ValueError, match="must equal 2.0"):
        score_candidate_admission_residual(
            _FixedHead(torch.zeros(1, 1), torch.zeros(1, 1)),
            h=torch.zeros(1, 4),
            memory_delta=torch.zeros(1, 4),
            candidate_embeddings=torch.zeros(1, 1, 4),
            static_logits=torch.zeros(1, 1),
            dynamic_logits=torch.zeros(1, 1),
            dynamic_extra_mask=torch.ones(1, 1, dtype=torch.bool),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
            causal_update_count=torch.ones(1),
            residual_bound=1.0,
        )
