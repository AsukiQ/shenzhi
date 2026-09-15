from __future__ import annotations

import torch
from torch import nn

from clstr.model import CLSTRConfig, CLSTRModel


class _CountingCrossEncoder:
    def __init__(self):
        self.calls: list[tuple[list[str], list[list[str]]]] = []

    def batch_forward(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        self.calls.append((list(state_texts), [list(row) for row in candidate_texts]))
        rows = []
        for state_idx, row in enumerate(candidate_texts):
            embs = []
            for cand_idx, _ in enumerate(row):
                embs.append(torch.tensor([float(state_idx + 1), float(cand_idx + 1)]))
            rows.append(torch.stack(embs))
        return torch.stack(rows)


def _model() -> CLSTRModel:
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=2)
    model.cross_encoder = _CountingCrossEncoder()
    model.skills = [
        {"name": "a", "description": "A"},
        {"name": "b", "description": "B"},
        {"name": "c", "description": "C"},
    ]
    return model


def test_cross_encoder_cache_reuses_task_local_state_skill_pairs():
    model = _model()
    CLSTRModel.reset_task_cache(model)

    first = CLSTRModel.batch_cross_encode(model, ["state"], [[0, 1]])
    second = CLSTRModel.batch_cross_encode(model, ["state"], [[1, 0]])

    assert first.shape == (1, 2, 2)
    assert second.shape == (1, 2, 2)
    assert len(model.cross_encoder.calls) == 1
    assert model.cross_encoder.calls[0][0] == ["state", "state"]


def test_cross_encoder_cache_miss_encodes_only_new_pairs():
    model = _model()
    CLSTRModel.reset_task_cache(model)

    CLSTRModel.batch_cross_encode(model, ["state"], [[0, 1]])
    CLSTRModel.batch_cross_encode(model, ["state"], [[1, 2]])

    assert len(model.cross_encoder.calls) == 2
    assert model.cross_encoder.calls[1][1] == [[model.candidate_texts([[2]])[0][0]]]


def test_reset_task_cache_clears_previous_entries():
    model = _model()
    CLSTRModel.reset_task_cache(model)
    CLSTRModel.batch_cross_encode(model, ["state"], [[0]])
    CLSTRModel.reset_task_cache(model)
    CLSTRModel.batch_cross_encode(model, ["state"], [[0]])

    assert len(model.cross_encoder.calls) == 2
