import json
import math
from pathlib import Path

import torch

from clstr import stage_checkpoint_init
from clstr.gated_temporal_reranker import GatedTemporalConfig, GatedTemporalReranker
from clstr.model import RouteMemoryUtilityGate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_model_from_stage0_checkpoint_uses_stage0_config_and_weights(tmp_path, monkeypatch):
    skills_path = tmp_path / "skills.jsonl"
    checkpoint_path = tmp_path / "stage0.pt"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "A"}])
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "config": {
                "base_model_name": "models/stage0-encoder",
                "d": 2,
                "d_a": 2,
                "top_k": 100,
                "encoder_pooling": "last_token",
                "cross_encoder_pooling": "last_token",
                "tokenizer_padding_side": "left",
                "torch_dtype": "bfloat16",
                "freeze_backbone": True,
                "max_length": 2048,
                "projection_init": "identity",
                "normalize_embeddings": True,
                "skill_text_format": "skillret_official",
                "skill_table_batch_size": 16,
                "skill_table_adapter_init": "identity",
                "use_cross_encoder": False,
                "state_query_prompt_version": "clstr_causal_state_v1",
                "state_query_instruction": (
                    "Given an agent task or current execution state and interaction history, "
                    "retrieve the skill or tool document most useful for the next action."
                ),
                "state_query_max_chars": 2000,
                "state_query_truncation": "head_tail_v1",
            },
            "model_state_dict": {
                "linear.weight": torch.eye(2),
                "wrong_shape.weight": torch.ones(3, 3),
            },
        },
        checkpoint_path,
    )

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModel(torch.nn.Module):
        def __init__(self, config, skills):
            super().__init__()
            self.config = config
            self.skills = list(skills)
            self.linear = torch.nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(stage_checkpoint_init, "CLSTRConfig", FakeConfig)
    monkeypatch.setattr(stage_checkpoint_init, "CLSTRModel", FakeModel)

    model, config, report = stage_checkpoint_init.build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=tmp_path / "cache",
        base_model_name_override="models/deployment-encoder",
    )

    assert config["base_model_name"] == "models/deployment-encoder"
    assert config["local_files_only"] is True
    assert config["encoder_pooling"] == "last_token"
    assert config["tokenizer_padding_side"] == "left"
    assert config["defer_skill_table_init"] is True
    assert config["use_cross_encoder"] is False
    assert config["state_query_prompt_version"] == "clstr_causal_state_v1"
    assert model.config.kwargs["base_model_name"] == "models/deployment-encoder"
    assert model.config.kwargs["d"] == 2
    assert model.config.kwargs["state_query_prompt_version"] == "clstr_causal_state_v1"
    assert model.config.kwargs["state_query_max_chars"] == 2000
    assert model.config.kwargs["state_query_truncation"] == "head_tail_v1"
    assert model.skills == [{"skill_id": "skill/a", "name": "A"}]
    assert torch.allclose(model.linear.weight, torch.eye(2))
    assert report["stage0_checkpoint_stage"] == "clstr_unified_retrieval_v2"
    assert report["stage0_checkpoint_base_model_name"] == "models/stage0-encoder"
    assert report["stage0_effective_base_model_name"] == "models/deployment-encoder"
    assert report["stage0_base_model_name_overridden"] is True
    assert report["stage0_loaded"] is True
    assert report["stage0_loaded_keys"] == ["linear.weight"]
    assert "wrong_shape.weight" in report["stage0_skipped_keys"]


def test_load_routing_and_head_checkpoints_merges_head_over_routing(tmp_path):
    routing_path = tmp_path / "routing.pt"
    head_path = tmp_path / "head.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "weight": torch.ones(2, 2),
            },
        },
        routing_path,
    )
    torch.save(
        {
            "stage": "clstr_full_base",
            "model_state_dict": {
                "weight": torch.eye(2) * 3.0,
            },
        },
        head_path,
    )
    model = torch.nn.Linear(2, 2, bias=False)

    report = stage_checkpoint_init.load_routing_and_head_checkpoints(
        model,
        routing_checkpoint_path=routing_path,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_merge",
    )

    assert torch.allclose(model.weight, torch.eye(2) * 3.0)
    assert report["routing_checkpoint_stage"] == "clstr_unified_retrieval_v2"
    assert report["head_checkpoint_stage"] == "clstr_full_base"
    assert report["head_overrides_routing_keys"] == ["weight"]
    assert report["loaded_keys"] == ["weight"]


def test_load_compatible_state_dict_seeds_default_belief_calibration_from_retrieval():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skill_table = torch.nn.Module()
            self.skill_table.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))

    model = _Model()
    state = {
        "skill_table.logit_scale_retr": torch.tensor([1.25]),
        "skill_table.skill_bias_retr": torch.tensor([0.1, -0.2, 0.3]),
        "skill_table.logit_scale_belief": torch.tensor([math.log(0.2)]),
        "skill_table.skill_bias_belief": torch.zeros(3),
    }

    report = stage_checkpoint_init.load_compatible_state_dict(
        model,
        state,
        partial_load_mode="unit_test_seed_belief_from_retrieval",
    )

    assert torch.allclose(model.skill_table.logit_scale_belief, state["skill_table.logit_scale_retr"])
    assert torch.allclose(model.skill_table.skill_bias_belief, state["skill_table.skill_bias_retr"])
    assert report["belief_calibration_seeded_from_retrieval"]["applied"] is True
    assert report["belief_calibration_seeded_from_retrieval"]["reason"] == "belief_was_default"


def test_load_compatible_state_dict_keeps_trained_belief_calibration():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skill_table = torch.nn.Module()
            self.skill_table.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))

    model = _Model()
    trained_belief_scale = torch.tensor([0.75])
    trained_belief_bias = torch.tensor([0.4, -0.1, 0.2])
    state = {
        "skill_table.logit_scale_retr": torch.tensor([1.25]),
        "skill_table.skill_bias_retr": torch.tensor([0.1, -0.2, 0.3]),
        "skill_table.logit_scale_belief": trained_belief_scale,
        "skill_table.skill_bias_belief": trained_belief_bias,
    }

    report = stage_checkpoint_init.load_compatible_state_dict(
        model,
        state,
        partial_load_mode="unit_test_keep_trained_belief",
    )

    assert torch.allclose(model.skill_table.logit_scale_belief, trained_belief_scale)
    assert torch.allclose(model.skill_table.skill_bias_belief, trained_belief_bias)
    assert report["belief_calibration_seeded_from_retrieval"]["applied"] is False
    assert report["belief_calibration_seeded_from_retrieval"]["reason"] == "belief_not_default"


def test_load_routing_and_head_checkpoints_can_protect_stage0_routing_foundation(tmp_path):
    routing_path = tmp_path / "routing.pt"
    head_path = tmp_path / "head.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "encoder.weight": torch.ones(2, 2),
                "skill_table.E": torch.ones(2, 2),
                "skill_table.logit_scale_retr": torch.ones(1),
                "skill_table.logit_scale_belief": torch.ones(1) * 0.5,
                "skill_table.skill_bias_belief": torch.ones(2) * 0.25,
                "transition.weight": torch.ones(2, 2),
            },
        },
        routing_path,
    )
    torch.save(
        {
            "stage": "clstr_full_base",
            "model_state_dict": {
                "encoder.weight": torch.eye(2) * 3.0,
                "skill_table.E": torch.eye(2) * 4.0,
                "skill_table.logit_scale_retr": torch.ones(1) * 4.0,
                "skill_table.logit_scale_belief": torch.ones(1) * 5.0,
                "skill_table.skill_bias_belief": torch.ones(2) * 6.0,
                "transition.weight": torch.eye(2) * 5.0,
            },
        },
        head_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.zeros(2, 2))
            self.skill_table.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(2))
            self.transition = torch.nn.Linear(2, 2, bias=False)

    model = _Model()

    report = stage_checkpoint_init.load_routing_and_head_checkpoints(
        model,
        routing_checkpoint_path=routing_path,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_protect_routing_foundation",
        protect_routing_foundation=True,
    )

    assert torch.allclose(model.encoder.weight, torch.ones(2, 2))
    assert torch.allclose(model.skill_table.E, torch.ones(2, 2))
    assert torch.allclose(model.skill_table.logit_scale_retr, torch.ones(1))
    assert torch.allclose(model.skill_table.logit_scale_belief, torch.ones(1) * 5.0)
    assert torch.allclose(model.skill_table.skill_bias_belief, torch.ones(2) * 6.0)
    assert torch.allclose(model.transition.weight, torch.eye(2) * 5.0)
    assert report["protect_routing_foundation"] is True
    assert report["head_overrides_routing_keys"] == [
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
        "transition.weight",
    ]
    assert report["skipped_head_routing_foundation_keys"] == [
        "encoder.weight",
        "skill_table.E",
        "skill_table.logit_scale_retr",
    ]


def test_load_head_checkpoint_into_model_protects_stage0_routing_foundation_by_default(tmp_path):
    head_path = tmp_path / "stage1.pt"
    torch.save(
        {
            "stage": "clstr_stage1_heads_init",
            "model_state_dict": {
                "encoder.weight": torch.eye(2) * 3.0,
                "skill_table.E": torch.eye(2) * 4.0,
                "skill_table.logit_scale_belief": torch.ones(1) * 5.0,
                "skill_table.skill_bias_belief": torch.ones(2) * 6.0,
                "transition.weight": torch.eye(2) * 5.0,
            },
        },
        head_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.ones(2, 2))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(2))
            self.transition = torch.nn.Linear(2, 2, bias=False)

    model = _Model()
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.transition.weight.zero_()

    report = stage_checkpoint_init.load_head_checkpoint_into_model(
        model,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_stage1_head_load",
    )

    assert torch.allclose(model.encoder.weight, torch.ones(2, 2))
    assert torch.allclose(model.skill_table.E, torch.ones(2, 2))
    assert torch.allclose(model.skill_table.logit_scale_belief, torch.ones(1) * 5.0)
    assert torch.allclose(model.skill_table.skill_bias_belief, torch.ones(2) * 6.0)
    assert torch.allclose(model.transition.weight, torch.eye(2) * 5.0)
    assert report["protect_routing_foundation"] is True
    assert report["skipped_head_routing_foundation_keys"] == ["encoder.weight", "skill_table.E"]


def test_load_head_checkpoint_into_model_allows_skill_table_prefix_expansion(tmp_path):
    head_path = tmp_path / "stage4.pt"
    torch.save(
        {
            "stage": "clstr_stage4_act",
            "model_state_dict": {
                "skill_table.E": torch.ones(2, 2),
                "skill_table.skill_bias_retr": torch.tensor([0.1, 0.2]),
                "skill_table.skill_bias_belief": torch.tensor([0.3, 0.4]),
                "head.weight": torch.eye(2),
            },
        },
        head_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.zeros(3, 2))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))
            self.head = torch.nn.Linear(2, 2, bias=False)

    model = _Model()

    report = stage_checkpoint_init.load_head_checkpoint_into_model(
        model,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_prefix_expand",
        protect_routing_foundation=False,
        allow_skill_table_prefix_expansion=True,
    )

    assert torch.allclose(model.skill_table.E[:2], torch.ones(2, 2))
    assert torch.allclose(model.skill_table.E[2], torch.zeros(2))
    assert torch.allclose(model.skill_table.skill_bias_retr[:2], torch.tensor([0.1, 0.2]))
    assert torch.allclose(model.skill_table.skill_bias_retr[2:], torch.zeros(1))
    assert torch.allclose(model.skill_table.skill_bias_belief[:2], torch.tensor([0.3, 0.4]))
    assert torch.allclose(model.skill_table.skill_bias_belief[2:], torch.zeros(1))
    assert torch.allclose(model.head.weight, torch.eye(2))
    assert set(report["skill_table_prefix_expanded_keys"]) == {
        "skill_table.E",
        "skill_table.skill_bias_retr",
        "skill_table.skill_bias_belief",
    }


def test_load_head_checkpoint_reconfigures_gated_temporal_reranker_from_checkpoint_shapes(tmp_path):
    head_path = tmp_path / "stage2_gated.pt"
    gated_state = {
        "gated_temporal_reranker.query_proj.weight": torch.full((2, 3), 0.25),
        "gated_temporal_reranker.query_proj.bias": torch.full((2,), 0.5),
        "gated_temporal_reranker.gate.weight": torch.full((1, 4), 0.75),
        "gated_temporal_reranker.gate.bias": torch.full((1,), -0.25),
        "gated_temporal_reranker.residual_adapter.0.weight": torch.full((5, 7), 0.125),
        "gated_temporal_reranker.residual_adapter.0.bias": torch.full((5,), 0.375),
        "gated_temporal_reranker.residual_adapter.2.weight": torch.full((1, 5), -0.5),
        "gated_temporal_reranker.residual_adapter.2.bias": torch.full((1,), 0.625),
    }
    torch.save(
        {
            "stage": "clstr_full_base",
            "step": 3000,
            "model_state_dict": gated_state,
        },
        head_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gated_temporal_reranker = GatedTemporalReranker(
                GatedTemporalConfig(dim=4, query_dim=4, hidden_dim=4)
            )

    model = _Model()

    report = stage_checkpoint_init.load_head_checkpoint_into_model(
        model,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_stage2_gated_head_load",
    )

    reranker = model.gated_temporal_reranker
    assert reranker.config.dim == 2
    assert reranker.config.query_dim == 3
    assert reranker.config.hidden_dim == 5
    assert "gated_temporal_reranker_reconfigured" in report
    assert report["gated_temporal_reranker_reconfigured"]["reconfigured"] is True
    assert not any(key.startswith("gated_temporal_reranker.") for key in report["skipped_keys"])
    assert report["shape_mismatched"] == {}
    assert torch.allclose(reranker.query_proj.weight, gated_state["gated_temporal_reranker.query_proj.weight"])
    assert torch.allclose(
        reranker.residual_adapter[0].weight,
        gated_state["gated_temporal_reranker.residual_adapter.0.weight"],
    )


def test_build_stage0_model_rebuilds_local_skill_embeddings_when_checkpoint_e_shape_mismatches(
    tmp_path, monkeypatch
):
    skills_path = tmp_path / "skills.jsonl"
    skills_path.write_text(
        "\n".join(
            [
                json.dumps({"skill_id": "local/a", "name": "a"}),
                json.dumps({"skill_id": "local/b", "name": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "stage0.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "config": {"d": 2, "d_a": 2, "defer_skill_table_init": True},
            "model_state_dict": {
                "skill_table.E": torch.ones(3, 2),
                "skill_table.logit_scale_retr": torch.zeros(1),
            },
        },
        checkpoint_path,
    )

    class _FakeSkillTable(torch.nn.Module):
        def __init__(self, skill_count: int):
            super().__init__()
            self.E = torch.nn.Parameter(torch.zeros(skill_count, 2))
            self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))

    class _FakeModel(torch.nn.Module):
        def __init__(self, _config, skills):
            super().__init__()
            self.skill_table = _FakeSkillTable(len(skills))
            self.rebuild_called = False

        def rebuild_skill_table(self):
            self.rebuild_called = True
            with torch.no_grad():
                self.skill_table.E.copy_(torch.arange(1, self.skill_table.E.numel() + 1).view_as(self.skill_table.E))
            return self.skill_table.E

    monkeypatch.setattr(stage_checkpoint_init, "CLSTRModel", _FakeModel)

    model, _config, report = stage_checkpoint_init.build_clstr_model_from_stage0_checkpoint(
        checkpoint_path,
        skills_path,
    )

    assert model.rebuild_called is True
    assert torch.count_nonzero(model.skill_table.E).item() == model.skill_table.E.numel()
    assert report["local_skill_table_rebuilt"] is True
    assert report["local_skill_table_rebuild_reason"] == "skill_table.E_not_loaded"


def test_load_head_checkpoint_attaches_stage4_score_calibrator_from_checkpoint(tmp_path):
    head_path = tmp_path / "stage4_calibrator.pt"
    weight = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]])
    torch.save(
        {
            "stage": "executor_free_logged_online_stage4_score_calibrator",
            "model_state_dict": {
                "stage4_score_calibrator.weight": weight,
            },
        },
        head_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    model = _Model()

    report = stage_checkpoint_init.load_head_checkpoint_into_model(
        model,
        head_checkpoint_path=head_path,
        partial_load_mode="unit_test_stage4_calibrator_load",
    )

    assert isinstance(model.stage4_score_calibrator, torch.nn.Linear)
    assert model.stage4_score_calibrator.bias is None
    assert model.stage4_score_calibrator.in_features == 6
    assert torch.allclose(model.stage4_score_calibrator.weight, weight)
    assert "stage4_score_calibrator.weight" in report["loaded_keys"]
    assert "stage4_score_calibrator.weight" not in report["skipped_keys"]
    assert report["stage4_score_calibrator_reconfigured"]["reconfigured"] is True


def test_pre_safe_memory_checkpoint_initializes_route_gate_from_default() -> None:
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_memory_utility_gate = RouteMemoryUtilityGate(2, initial_alpha=0.01)

    model = _Model()
    report = stage_checkpoint_init.load_compatible_state_dict(
        model,
        {},
        partial_load_mode="unit_test_pre_safe_memory_checkpoint",
    )

    compatibility = report["route_memory_utility_gate_compatibility"]
    assert compatibility["status"] == "initialized_default"
    assert compatibility["loaded_keys"] == []
    assert compatibility["missing_keys"]
    assert all(key.startswith("route_memory_utility_gate.") for key in compatibility["missing_keys"])
