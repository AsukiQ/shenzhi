from __future__ import annotations

import torch
from torch import nn

from clstr.model import CLSTRConfig, CLSTRModel


def _bare_model() -> CLSTRModel:
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(top_k=3, candidate_widen_factor=2, candidate_sample_temperature=1.0)
    model.K = 3
    return model


def test_sample_candidates_eval_is_deterministic_top_k():
    model = _bare_model()
    logits = torch.tensor([[0.0, 9.0, 3.0, 8.0, 7.0, 1.0]])

    rows, values = CLSTRModel.sample_candidates(model, logits, train=False)

    assert rows == [[1, 3, 4]]
    assert torch.allclose(values, torch.tensor([[9.0, 8.0, 7.0]]))


def test_sample_candidates_train_samples_without_replacement_from_widened_top_m():
    model = _bare_model()
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, -100.0, -101.0]])
    generator = torch.Generator().manual_seed(7)

    rows, values = CLSTRModel.sample_candidates(model, logits, train=True, generator=generator)

    assert len(rows) == 1
    assert len(rows[0]) == 3
    assert len(set(rows[0])) == 3
    assert set(rows[0]).issubset({0, 1, 2, 3, 4, 5})
    assert values.shape == (1, 3)


def test_sample_candidates_low_temperature_degenerates_to_top_k():
    model = _bare_model()
    logits = torch.tensor([[0.0, 9.0, 3.0, 8.0, 7.0, 1.0]])

    rows, _ = CLSTRModel.sample_candidates(model, logits, train=True, temperature=1.0e-9)

    assert rows == [[1, 3, 4]]
