from __future__ import annotations

import pytest
import torch

from clstr.matched_history_encoders import (
    CausalTransformerHistoryEncoder,
    GRUHistoryEncoder,
    LSTRHistoryEncoder,
    SerializedHistoryEncoder,
    build_matched_history_encoder,
    trainable_parameter_count,
)
from clstr.vnext_core import CLSTRVNextCore


def _inputs(*, batch: int = 3, time: int = 4, d: int = 8):
    torch.manual_seed(17)
    initial = torch.randn(batch, d)
    values = [torch.randn(batch, time, d) for _ in range(4)]
    event_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [False, False, False, False]],
        dtype=torch.bool,
    )
    result_mask = torch.tensor(
        [[True, False, True, True], [False, True, False, False], [False, False, False, False]],
        dtype=torch.bool,
    )
    return initial, *values, event_mask, result_mask


@pytest.mark.parametrize(
    "encoder",
    [
        GRUHistoryEncoder(8, input_dim=4),
        CausalTransformerHistoryEncoder(
            8,
            model_dim=8,
            num_heads=2,
            num_layers=2,
            feedforward_dim=16,
            max_horizon=4,
        ),
        LSTRHistoryEncoder(8, hidden_dim=4),
    ],
)
def test_sequence_encoders_preserve_exact_zero_history(encoder: torch.nn.Module) -> None:
    inputs = _inputs()
    output = encoder(*inputs)
    assert output.shape == inputs[0].shape
    assert torch.equal(output[2], inputs[0][2])


def test_serialized_history_requires_embedding_only_for_nonempty_rows() -> None:
    inputs = _inputs()
    encoder = SerializedHistoryEncoder(8, hidden_dim=8)
    with pytest.raises(ValueError, match="requires factual-history embeddings"):
        encoder(*inputs)
    serialized = torch.randn(3, 8)
    output = encoder(*inputs, serialized_history_embedding=serialized)
    assert output.shape == inputs[0].shape
    assert torch.equal(output[2], inputs[0][2])


@pytest.mark.parametrize(
    "encoder",
    [
        GRUHistoryEncoder(8, input_dim=4),
        CausalTransformerHistoryEncoder(
            8,
            model_dim=8,
            num_heads=2,
            num_layers=2,
            feedforward_dim=16,
            max_horizon=4,
        ),
        LSTRHistoryEncoder(8, hidden_dim=4),
    ],
)
def test_padded_events_do_not_change_sequence_representation(encoder: torch.nn.Module) -> None:
    inputs = list(_inputs())
    baseline = encoder(*inputs)
    for tensor_index in range(1, 5):
        changed = inputs[tensor_index].clone()
        changed[1, 2:] = changed[1, 2:] + 1000.0
        changed[2] = changed[2] - 1000.0
        candidate = list(inputs)
        candidate[tensor_index] = changed
        output = encoder(*candidate)
        assert torch.allclose(output[1:], baseline[1:], atol=1.0e-6, rtol=1.0e-6)


def test_lstr_encoder_matches_production_transition_and_correction() -> None:
    torch.manual_seed(23)
    d = 8
    encoder = LSTRHistoryEncoder(d, hidden_dim=4)
    core = CLSTRVNextCore(
        d,
        hidden_dim=4,
        scale_initial=0.05,
        scale_maximum=1.0,
    )
    for name in (
        "action_adapter",
        "result_adapter",
        "transition_delta",
        "correction_delta",
        "correction_gate",
        "transition_scale",
        "result_scale",
    ):
        getattr(core, name).load_state_dict(getattr(encoder, name).state_dict())
    initial, states, skills, actions, results, event_mask, result_mask = _inputs(
        batch=3,
        time=4,
        d=d,
    )
    expected = initial
    for index in range(4):
        update = core.update_memory(
            expected,
            states[:, index],
            skills[:, index],
            actions[:, index],
            result_embedding=results[:, index] if bool(result_mask[:, index].all()) else None,
        )
        candidate = update.effective_memory
        mixed = candidate
        partial = event_mask[:, index] & result_mask[:, index]
        if bool(partial.any().item()) and not bool(result_mask[:, index].all()):
            predicted, _delta, adapted_action = core.predict_memory(
                expected,
                states[:, index],
                skills[:, index],
                actions[:, index],
            )
            selected = partial.nonzero(as_tuple=False).view(-1)
            corrected, _correction, _beta = core.correct_memory(
                predicted.index_select(0, selected),
                states[:, index].index_select(0, selected),
                skills[:, index].index_select(0, selected),
                adapted_action.index_select(0, selected),
                results[:, index].index_select(0, selected),
                action_is_adapted=True,
            )
            mixed = predicted.index_copy(0, selected, corrected)
        expected = torch.where(event_mask[:, index].unsqueeze(-1), mixed, expected)
    observed = encoder(
        initial,
        states,
        skills,
        actions,
        results,
        event_mask,
        result_mask,
    )
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_lstr_encoder_aligns_autocast_vectors_like_production_wrapper() -> None:
    encoder = LSTRHistoryEncoder(8, hidden_dim=4)
    inputs = _inputs(batch=3, time=4, d=8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(*inputs)
    assert output.shape == inputs[0].shape
    assert torch.isfinite(output.float()).all()


def test_factory_rejects_unknown_encoder_and_reports_parameters() -> None:
    encoder = build_matched_history_encoder("gru", 8)
    assert trainable_parameter_count(encoder) > 0
    with pytest.raises(ValueError, match="unsupported matched history encoder"):
        build_matched_history_encoder("unknown", 8)  # type: ignore[arg-type]
