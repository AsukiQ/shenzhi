from types import SimpleNamespace

import pytest
import torch

from clstr.vnext_online_selector import (
    _selected_route_logits,
    _selected_route_support_mask,
)


def _route():
    return SimpleNamespace(
        static_logits=torch.tensor([[1.0, 0.0]]),
        raw_dynamic_logits=torch.tensor([[0.0, 2.0]]),
        mixed_logits=torch.tensor([[0.0, 2.0]]),
        mixture_probability=torch.tensor([1.0]),
    )


def test_route_mode_preserves_adaptive_and_supports_static_dynamic_controls():
    route = _route()

    adaptive, adaptive_expert, adaptive_probability = _selected_route_logits(
        route, route_mode="adaptive", has_history=True
    )
    static, static_expert, static_probability = _selected_route_logits(
        route, route_mode="static", has_history=True
    )
    dynamic, dynamic_expert, dynamic_probability = _selected_route_logits(
        route, route_mode="dynamic", has_history=True
    )

    assert adaptive is route.mixed_logits
    assert (adaptive_expert, adaptive_probability) == ("dynamic", 1.0)
    assert static is route.static_logits
    assert (static_expert, static_probability) == ("static", 0.0)
    assert dynamic is route.raw_dynamic_logits
    assert (dynamic_expert, dynamic_probability) == ("dynamic", 1.0)


def test_forced_dynamic_is_exact_static_without_history():
    route = _route()

    selected, expert, probability = _selected_route_logits(
        route, route_mode="dynamic", has_history=False
    )

    assert selected is route.static_logits
    assert (expert, probability) == ("static", 0.0)


def test_static_output_excludes_dynamic_only_candidate_support():
    route = SimpleNamespace(
        static_support_mask=torch.tensor([[True, False, True]]),
    )
    support_valid = torch.tensor([[True, True, True]])

    static = _selected_route_support_mask(
        route,
        support_valid=support_valid,
        selected_expert="static",
    )
    dynamic = _selected_route_support_mask(
        route,
        support_valid=support_valid,
        selected_expert="dynamic",
    )

    assert torch.equal(static, torch.tensor([[True, False, True]]))
    assert torch.equal(dynamic, support_valid)


def test_route_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported vNext route mode"):
        _selected_route_logits(_route(), route_mode="benchmark_specific", has_history=True)
