from __future__ import annotations

import torch

from clstr.candidate_admission_residual import (
    CANDIDATE_ADMISSION_RESIDUAL_V1,
    CandidateAdmissionResidualHead,
)
from clstr.counterfactual_memory_calibration import RouteMemoryResidualAdapter
from clstr.memory_utility_gate import MemoryUtilityGate
from clstr.stage4_safe_memory import (
    freeze_stage4_candidate_admission,
    freeze_stage4_cmc,
    freeze_stage4_safe_memory,
    router_state_digest,
    set_stage4_candidate_admission_training_mode,
    set_stage4_safe_training_mode,
    stage4_candidate_admission_delta_state_dict,
    stage4_cmc_delta_state_dict,
    stage4_delta_state_dict,
    validate_stage4_candidate_admission_delta_state_dict,
    validate_stage4_candidate_admission_parent_cmc,
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_checkpoint_payload,
    validate_stage4_cmc_parent_router,
    validate_stage4_delta_state_dict,
)


class _SkillTable(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(dim), requires_grad=False)
        self.logit_scale_belief = torch.nn.Parameter(
            torch.tensor(0.0),
            requires_grad=False,
        )
        self.skill_bias_belief = torch.nn.Parameter(
            torch.zeros(dim),
            requires_grad=False,
        )


class _TinySafeMemoryModel(torch.nn.Module):
    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.proj = torch.nn.Linear(dim, dim, bias=False)
        self.skill_table = _SkillTable(dim)
        self.initial_belief_head = torch.nn.Linear(dim, dim)
        self.unified_retriever = torch.nn.Linear(dim * 2, dim)
        self.route_memory_utility_gate = torch.nn.Linear(dim * 3, 1)
        self.route_memory_residual_adapter = RouteMemoryResidualAdapter(dim)
        self.route_memory_candidate_utility_gate = MemoryUtilityGate()
        self.route_memory_candidate_admission_residual = (
            CandidateAdmissionResidualHead(dim)
        )
        self.transition = torch.nn.Linear(dim, dim)
        self.gate = torch.nn.Linear(dim, dim)
        self.action_proj = torch.nn.Linear(dim, dim, bias=False)
        self.trans_head = torch.nn.Linear(dim, dim)
        self.skill_head = torch.nn.Linear(dim, dim)
        self.stop_head = torch.nn.Linear(dim, 1)


def test_safe_memory_freeze_and_delta_state_are_exact() -> None:
    model = _TinySafeMemoryModel()
    before = router_state_digest(model, scope="full")

    report = freeze_stage4_safe_memory(model)
    set_stage4_safe_training_mode(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = sum(
        parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()

    assert report["trainable_modules"] == [
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ]
    assert router_state_digest(model, scope="full") == before
    assert all(
        parameter.grad is None
        for parameter in model.initial_belief_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.unified_retriever.parameters()
    )
    assert model.encoder.training is False
    assert model.initial_belief_head.training is False
    assert model.unified_retriever.training is False
    assert model.transition.training is True
    assert model.gate.training is True
    assert model.action_proj.training is True
    assert model.route_memory_utility_gate.training is True

    state = stage4_delta_state_dict(model)
    assert state
    assert all(
        key.startswith(
            ("transition.", "gate.", "action_proj.", "route_memory_utility_gate.")
        )
        for key in state
    )
    assert validate_stage4_delta_state_dict(state)["status"] == "ok"


def test_safe_memory_delta_rejects_router_keys() -> None:
    bad_state = {
        "transition.weight": torch.ones(1, 1),
        "initial_belief_head.weight": torch.ones(1, 1),
    }

    try:
        validate_stage4_delta_state_dict(bad_state)
    except ValueError as exc:
        assert "forbidden Stage4 delta key" in str(exc)
    else:
        raise AssertionError("router key must be rejected")


def test_cmc_stage4_optimizer_and_delta_contain_only_two_modules() -> None:
    model = _TinySafeMemoryModel()
    before = router_state_digest(model, scope="full")

    report = freeze_stage4_cmc(model)
    state = stage4_cmc_delta_state_dict(model)

    assert report["trainable_modules"] == [
        "route_memory_residual_adapter",
        "route_memory_candidate_utility_gate",
    ]
    assert all(
        name.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_utility_gate.",
            )
        )
        for name in report["optimizer_parameter_names"]
    )
    assert all(
        key.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_utility_gate.",
            )
        )
        for key in state
    )
    assert router_state_digest(model, scope="full") == before
    assert all(not parameter.requires_grad for parameter in model.transition.parameters())
    assert all(not parameter.requires_grad for parameter in model.gate.parameters())
    assert all(not parameter.requires_grad for parameter in model.action_proj.parameters())


def test_cmc_delta_rejects_stage2_keys() -> None:
    try:
        validate_stage4_cmc_delta_state_dict({"transition.weight": torch.ones(1)})
    except ValueError as exc:
        assert "forbidden CMC delta key" in str(exc)
    else:
        raise AssertionError("Stage2 key must be rejected from a CMC delta")


def test_cmc_delta_rejects_wrong_parent_stage2_router() -> None:
    model = _TinySafeMemoryModel()
    payload = {
        "parent_stage2_full_router_digest": "wrong-full",
        "parent_stage2_fast_router_digest": "wrong-fast",
    }

    try:
        validate_stage4_cmc_parent_router(payload, model)
    except ValueError as exc:
        assert "parent Stage2 router digest mismatch" in str(exc)
    else:
        raise AssertionError("CMC delta must reject a different Stage2 parent")


def test_checkpoint_payload_dispatches_to_cmc_delta_validation() -> None:
    payload = {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "model_state_dict": {
            "route_memory_residual_adapter.net.0.weight": torch.ones(1, 1),
            "route_memory_candidate_utility_gate.feature_mean": torch.zeros(8),
        },
    }

    report = validate_stage4_checkpoint_payload(payload)

    assert report["stage4_method"] == "counterfactual_memory_calibration_v1"


def test_checkpoint_payload_keeps_legacy_delta_validation() -> None:
    payload = {
        "stage4_method": "stage4_safe_memory_v1",
        "model_state_dict": {
            "transition.weight": torch.ones(1, 1),
            "gate.weight": torch.ones(1, 1),
            "action_proj.weight": torch.ones(1, 1),
        },
    }

    report = validate_stage4_checkpoint_payload(payload)

    assert report["status"] == "ok"
    assert report.get("stage4_method") is None


def test_candidate_admission_freeze_trains_only_new_module_and_keeps_cmc_delta() -> None:
    model = _TinySafeMemoryModel()
    before_full = router_state_digest(model, scope="full")
    before_fast = router_state_digest(model, scope="fast")

    report = freeze_stage4_candidate_admission(model)
    set_stage4_candidate_admission_training_mode(model)
    state = stage4_candidate_admission_delta_state_dict(model)

    assert report["stage4_method"] == CANDIDATE_ADMISSION_RESIDUAL_V1
    assert report["trainable_modules"] == [
        "route_memory_candidate_admission_residual"
    ]
    assert report["parent_stage2_full_router_digest"] == before_full
    assert report["parent_stage2_fast_router_digest"] == before_fast
    assert report["optimizer_parameter_names"]
    assert all(
        name.startswith("route_memory_candidate_admission_residual.")
        for name in report["optimizer_parameter_names"]
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.route_memory_residual_adapter.parameters()
    )
    assert model.route_memory_residual_adapter.training is False
    assert model.route_memory_candidate_admission_residual.training is True
    assert state
    assert all(
        key.startswith(
            (
                "route_memory_residual_adapter.",
                "route_memory_candidate_admission_residual.",
            )
        )
        for key in state
    )
    assert not any(
        key.startswith("route_memory_candidate_utility_gate.") for key in state
    )


def test_candidate_admission_delta_rejects_old_gate_and_stage2_keys() -> None:
    valid_state = {
        "route_memory_residual_adapter.net.0.weight": torch.ones(1, 1),
        "route_memory_candidate_admission_residual.trunk.1.weight": torch.ones(1, 1),
    }
    assert validate_stage4_candidate_admission_delta_state_dict(valid_state)[
        "status"
    ] == "ok"

    for forbidden_key in (
        "route_memory_candidate_utility_gate.feature_mean",
        "transition.weight",
    ):
        try:
            validate_stage4_candidate_admission_delta_state_dict(
                {**valid_state, forbidden_key: torch.ones(1)}
            )
        except ValueError as exc:
            assert "forbidden candidate-admission delta key" in str(exc)
        else:
            raise AssertionError(f"candidate admission delta accepted {forbidden_key}")


def test_candidate_admission_checkpoint_requires_parent_identity_metadata() -> None:
    state = {
        "route_memory_residual_adapter.net.0.weight": torch.ones(1, 1),
        "route_memory_candidate_admission_residual.trunk.1.weight": torch.ones(1, 1),
    }
    payload = {
        "stage4_method": CANDIDATE_ADMISSION_RESIDUAL_V1,
        "base_cmc_checkpoint_sha256": "a" * 64,
        "parent_stage2_full_router_digest": "b" * 64,
        "parent_stage2_fast_router_digest": "c" * 64,
        "model_state_dict": state,
    }

    report = validate_stage4_checkpoint_payload(payload)

    assert report["stage4_method"] == CANDIDATE_ADMISSION_RESIDUAL_V1
    for field in (
        "base_cmc_checkpoint_sha256",
        "parent_stage2_full_router_digest",
        "parent_stage2_fast_router_digest",
    ):
        incomplete = dict(payload)
        incomplete.pop(field)
        try:
            validate_stage4_checkpoint_payload(incomplete)
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"candidate admission checkpoint accepted no {field}")


def test_candidate_admission_parent_cmc_rejects_sha_drift() -> None:
    payload = {"base_cmc_checkpoint_sha256": "a" * 64}

    report = validate_stage4_candidate_admission_parent_cmc(payload, "a" * 64)
    assert report["base_cmc_checkpoint_sha256"] == "a" * 64

    try:
        validate_stage4_candidate_admission_parent_cmc(payload, "b" * 64)
    except ValueError as exc:
        assert "base CMC identity mismatch" in str(exc)
    else:
        raise AssertionError("candidate admission checkpoint accepted CMC SHA drift")
