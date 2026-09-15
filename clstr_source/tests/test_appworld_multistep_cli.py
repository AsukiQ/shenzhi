import json
from pathlib import Path

import torch

from clstr.appworld_multistep import CLSTRMultiStepController, HybridCLSTRSkillRouterController
from scripts import run_appworld_multistep_executor_eval as cli


def test_cli_accepts_schema_plan_skill_context_mode():
    assert "schema_plan" in cli.SKILL_CONTEXT_MODE_CHOICES


def test_cli_accepts_api_evidence_skill_context_mode():
    assert "api_evidence" in cli.SKILL_CONTEXT_MODE_CHOICES


def test_cli_accepts_verified_hints_skill_context_mode():
    assert "verified_hints" in cli.SKILL_CONTEXT_MODE_CHOICES


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class _CliFakeModel:
    def eval(self):
        return self


class _DynamicCliFakeModel(_CliFakeModel):
    def __init__(self):
        self.skills = [{"skill_id": "base/a"}]
        self.appended_rows = []

    def append_skills(self, rows):
        self.appended_rows = list(rows)
        self.skills.extend(self.appended_rows)
        return {
            "appended_count": len(self.appended_rows),
            "appended_skill_ids": [row["skill_id"] for row in self.appended_rows],
        }


def test_build_controller_can_load_clstr_multistep_checkpoint(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool_path,
        [
            {"skill_id": "skill-a", "name": "A"},
            {"skill_id": "skill-b", "name": "B"},
        ],
    )
    captured = {}

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        captured["model_config_path"] = str(model_config_path)
        captured["skill_pool_path"] = str(skill_pool_path)
        captured["checkpoint_path"] = str(checkpoint_path)
        return _CliFakeModel(), {"status": "ok"}

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)

    controller = cli._build_controller(
        method="clstr_mt_fusion",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_model_config_path="configs/model/appworld_skillrouter_init_mt_fusion.yaml",
        clstr_checkpoint_path="outputs/checkpoints/model.pt",
        ranking_mode="policy_head",
        candidate_top_k=8,
        policy_blend_alpha=0.75,
        allow_legacy_policy_skill_router=True,
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.ranking_mode == "policy_head"
    assert controller.candidate_top_k == 8
    assert controller.policy_blend_alpha == 0.75
    assert controller.allow_legacy_policy_skill_router is True
    assert captured == {
        "model_config_path": "configs/model/appworld_skillrouter_init_mt_fusion.yaml",
        "skill_pool_path": str(skill_pool_path),
        "checkpoint_path": "outputs/checkpoints/model.pt",
    }


def test_build_controller_auto_ranking_uses_skill_table_for_routing_only_base(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(skill_pool_path, [{"skill_id": "skill-a", "name": "A"}])

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        return _CliFakeModel(), {
            "status": "ok",
            "stage": "stage2_base",
            "metrics": {
                "routing_supervision_tasks": 90.0,
                "routing_supervision_recall@1": 1.0,
            },
        }

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)

    controller = cli._build_controller(
        method="clstr_multistep",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_checkpoint_path="outputs/checkpoints/stage2_base-step500.pt",
        ranking_mode="auto",
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.ranking_mode == "skill_table"


def test_build_controller_default_ranking_is_auto_for_routing_only_base(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(skill_pool_path, [{"skill_id": "skill-a", "name": "A"}])

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        return _CliFakeModel(), {"stage": "stage2_base", "metrics": {"routing_supervision_tasks": 1.0}}

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)

    controller = cli._build_controller(
        method="clstr_multistep",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_checkpoint_path="outputs/checkpoints/stage2_base-step500.pt",
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.ranking_mode == "skill_table"


def test_build_controller_auto_ranking_uses_transition_blend_for_stage4_checkpoint(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(skill_pool_path, [{"skill_id": "skill-a", "name": "A"}])

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        return _CliFakeModel(), {
            "status": "ok",
            "stage": "clstr_stage4_transition_conditioned_next_skill",
        }

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)

    controller = cli._build_controller(
        method="clstr_multistep",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_checkpoint_path="outputs/checkpoints/clstr_stage4_act-step2000.pt",
        ranking_mode="auto",
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.ranking_mode == "transition_blend"


def test_resolve_auto_ranking_uses_policy_transition_blend_for_current_route_stage4():
    assert (
        cli._resolve_auto_ranking_mode(
            "clstr_multistep",
            "auto",
            {"stage": "appworld_current_route_preference"},
        )
        == "policy_transition_blend"
    )
    assert (
        cli._resolve_auto_ranking_mode(
            "clstr_multistep",
            "auto",
            {"stage": "appworld_current_route_online_stage4"},
        )
        == "policy_transition_blend"
    )


def test_build_controller_can_set_candidate_source_for_clstr_multistep(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(skill_pool_path, [{"skill_id": "skill-a", "name": "A"}])

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        return _CliFakeModel(), {"stage": "clstr_stage4_transition_conditioned_next_skill"}

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)

    controller = cli._build_controller(
        method="clstr_multistep",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_checkpoint_path="outputs/checkpoints/clstr_stage4_act-step2000.pt",
        ranking_mode="auto",
        candidate_source="routing_belief_union",
        appworld_executor_compatible_only=True,
        allow_legacy_policy_skill_router=True,
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.candidate_source == "routing_belief_union"
    assert controller.appworld_executor_compatible_only is True


def test_build_controller_can_create_hybrid_clstr_skillrouter_live(monkeypatch, tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool_path,
        [
            {"skill_id": "skill-a", "name": "A"},
            {"skill_id": "skill-b", "name": "B"},
        ],
    )

    def fake_load_checkpoint_model(*, model_config_path, skill_pool_path, checkpoint_path):
        return _CliFakeModel(), {"status": "ok"}

    class FakeEncoder:
        def __call__(self, texts):
            import torch

            return torch.eye(2)[: len(texts)]

    monkeypatch.setattr(cli, "_load_checkpoint_model", fake_load_checkpoint_model)
    monkeypatch.setattr(cli, "_HFTextEncoder", lambda **kwargs: FakeEncoder())
    monkeypatch.setattr(cli, "load_skillrouter_base_checkpoint", lambda path: (None, {}))

    controller = cli._build_controller(
        method="clstr_skillrouter_hybrid_live",
        skill_pool_path=str(skill_pool_path),
        predictions_path=None,
        clstr_model_config_path="configs/model/appworld_skillrouter_init.yaml",
        clstr_checkpoint_path="outputs/checkpoints/model.pt",
        clstr_alpha=0.2,
    )

    assert isinstance(controller, HybridCLSTRSkillRouterController)
    assert controller.clstr_alpha == 0.2


def test_build_controller_can_load_dynamic_appworld_pool_from_checkpoint_prefix(monkeypatch, tmp_path):
    base_pool_path = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool_path = tmp_path / "dynamic_skill_pool.jsonl"
    _write_jsonl(base_pool_path, [{"skill_id": "base/a", "name": "base\u2028skill", "appworld_executor_compatible": False}])
    _write_jsonl(
        dynamic_pool_path,
        [
            {"skill_id": "base/a", "name": "base\u2028skill", "appworld_executor_compatible": False},
            {
                "skill_id": "skillx/appworld/new",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
            },
        ],
    )
    fake_model = _DynamicCliFakeModel()
    captured = {}

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        captured["checkpoint_path"] = str(checkpoint_path)
        captured["skills_path"] = str(skills_path)
        captured["model_cache_dir"] = model_cache_dir
        return fake_model, {"query_text_format": "skillrouter"}, {"stage0_loaded": True, "skill_count": 1}

    monkeypatch.setattr(cli, "build_clstr_model_from_stage0_checkpoint", fake_build, raising=False)
    monkeypatch.setattr(
        cli,
        "checkpoint_payload",
        lambda path, label="checkpoint": {"stage": "clstr_unified_retrieval_v2", "model_state_dict": {}},
        raising=False,
    )

    controller = cli._build_controller(
        method="clstr_multistep",
        skill_pool_path=str(dynamic_pool_path),
        base_skill_pool_path=str(base_pool_path),
        predictions_path=None,
        clstr_checkpoint_path="outputs/checkpoints/stage0.pt",
        ranking_mode="skill_table",
        candidate_top_k=8,
        appworld_executor_compatible_only=True,
    )

    assert isinstance(controller, CLSTRMultiStepController)
    assert controller.skill_ids == ["base/a", "skillx/appworld/new"]
    assert fake_model.appended_rows == [
        {
            "skill_id": "skillx/appworld/new",
            "skill_pool_role": "appworld_appended_after_checkpoint",
            "appworld_executor_compatible": True,
        }
    ]
    assert captured == {
        "checkpoint_path": "outputs/checkpoints/stage0.pt",
        "skills_path": str(base_pool_path),
        "model_cache_dir": None,
    }
    assert controller.appworld_executor_compatible_only is True


def test_dynamic_checkpoint_loader_restores_stage4_delta_from_recorded_routing_and_head(monkeypatch, tmp_path):
    base_pool_path = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool_path = tmp_path / "dynamic_skill_pool.jsonl"
    routing_checkpoint = tmp_path / "stage0.pt"
    head_checkpoint = tmp_path / "stage2.pt"
    stage4_checkpoint = tmp_path / "stage4_delta.pt"
    _write_jsonl(base_pool_path, [{"skill_id": "base/a", "name": "base"}])
    _write_jsonl(
        dynamic_pool_path,
        [
            {"skill_id": "base/a", "name": "base"},
            {
                "skill_id": "skillx/appworld/new",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
            },
        ],
    )
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {"transition.cell.weight_ih": torch.ones(1, 1)},
            "train_report": {
                "checkpoint_init_report": {
                    "routing_checkpoint_path": str(routing_checkpoint),
                    "head_checkpoint_path": str(head_checkpoint),
                }
            },
        },
        stage4_checkpoint,
    )

    fake_model = _DynamicCliFakeModel()
    captured = {}

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        captured["build_checkpoint_path"] = str(checkpoint_path)
        captured["build_skills_path"] = str(skills_path)
        return fake_model, {"query_text_format": "skillrouter"}, {"stage0_checkpoint_stage": "clstr_unified_retrieval_v2"}

    def fake_load_routing_and_head_checkpoints(model, *, routing_checkpoint_path, head_checkpoint_path, partial_load_mode, protect_routing_foundation):
        captured["routing_checkpoint_path"] = str(routing_checkpoint_path)
        captured["head_checkpoint_path"] = str(head_checkpoint_path)
        captured["partial_load_mode"] = partial_load_mode
        captured["protect_routing_foundation"] = protect_routing_foundation
        return {"loaded": True, "loaded_keys": ["skill_table.E", "transition.cell.weight_ih"]}

    def fake_load_compatible_state_dict(model, state, *, partial_load_mode, allow_skill_table_prefix_expansion=False):
        captured["stage4_delta_state_keys"] = sorted(state)
        captured["stage4_delta_partial_load_mode"] = partial_load_mode
        captured["stage4_delta_prefix_expansion"] = allow_skill_table_prefix_expansion
        return {"loaded": True, "loaded_keys": sorted(state), "skipped_keys": []}

    monkeypatch.setattr(cli, "build_clstr_model_from_stage0_checkpoint", fake_build, raising=False)
    monkeypatch.setattr(cli, "load_routing_and_head_checkpoints", fake_load_routing_and_head_checkpoints, raising=False)
    monkeypatch.setattr(cli, "load_compatible_state_dict", fake_load_compatible_state_dict, raising=False)

    model, report = cli._load_dynamic_checkpoint_model(
        checkpoint_path=str(stage4_checkpoint),
        base_skill_pool_path=str(base_pool_path),
        dynamic_skill_pool_path=str(dynamic_pool_path),
    )

    assert model is fake_model
    assert captured["build_checkpoint_path"] == str(routing_checkpoint)
    assert captured["build_skills_path"] == str(base_pool_path)
    assert captured["routing_checkpoint_path"] == str(routing_checkpoint)
    assert captured["head_checkpoint_path"] == str(head_checkpoint)
    assert captured["partial_load_mode"] == "stage4_recorded_routing_plus_head_init"
    assert captured["protect_routing_foundation"] is True
    assert captured["stage4_delta_state_keys"] == ["transition.cell.weight_ih"]
    assert captured["stage4_delta_partial_load_mode"] == "stage4_delta_after_recorded_routing_head_init"
    assert fake_model.appended_rows == [
        {
            "skill_id": "skillx/appworld/new",
            "skill_pool_role": "appworld_appended_after_checkpoint",
            "appworld_executor_compatible": True,
        }
    ]
    assert report["stage"] == "clstr_stage4_transition_conditioned_next_skill"
    assert report["dynamic_checkpoint_load_mode"] == "stage4_delta_with_recorded_routing_head"


def test_dynamic_checkpoint_loader_loads_current_route_preference_as_full_dynamic_checkpoint(monkeypatch, tmp_path):
    base_pool_path = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool_path = tmp_path / "dynamic_skill_pool.jsonl"
    preference_checkpoint = tmp_path / "current_route_preference.pt"
    _write_jsonl(base_pool_path, [{"skill_id": "base/a", "name": "base"}])
    _write_jsonl(
        dynamic_pool_path,
        [
            {"skill_id": "base/a", "name": "base"},
            {
                "skill_id": "skillx/appworld/new",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
            },
        ],
    )
    torch.save(
        {
            "stage": "appworld_current_route_preference",
            "config": {"d": 8, "base_model_name": "fake"},
            "model_state_dict": {"skill_head.net.0.weight": torch.ones(1, 1)},
            "train_report": {
                "checkpoint_metadata": {
                    "dynamic_skill_pool_path": str(dynamic_pool_path),
                    "source_clstr_checkpoint_path": "stage4.pt",
                }
            },
        },
        preference_checkpoint,
    )

    fake_model = _DynamicCliFakeModel()
    captured = {}

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        captured["build_checkpoint_path"] = str(checkpoint_path)
        captured["build_skills_path"] = str(skills_path)
        captured["model_cache_dir"] = model_cache_dir
        fake_model.skills = [{"skill_id": "base/a"}, {"skill_id": "skillx/appworld/new"}]
        return fake_model, {"query_text_format": "skillrouter"}, {"stage0_checkpoint_stage": "appworld_current_route_preference"}

    monkeypatch.setattr(cli, "build_clstr_model_from_stage0_checkpoint", fake_build, raising=False)

    model, report = cli._load_dynamic_checkpoint_model(
        checkpoint_path=str(preference_checkpoint),
        base_skill_pool_path=str(base_pool_path),
        dynamic_skill_pool_path=str(dynamic_pool_path),
    )

    assert model is fake_model
    assert captured["build_checkpoint_path"] == str(preference_checkpoint)
    assert captured["build_skills_path"] == str(dynamic_pool_path)
    assert fake_model.appended_rows == []
    assert report["stage"] == "appworld_current_route_preference"
    assert report["dynamic_checkpoint_load_mode"] == "current_route_preference_full_dynamic_checkpoint"
    assert report["append_report"]["appended_count"] == 0


def test_dynamic_checkpoint_loader_loads_current_route_online_stage4_as_full_dynamic_checkpoint(monkeypatch, tmp_path):
    base_pool_path = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool_path = tmp_path / "dynamic_skill_pool.jsonl"
    online_checkpoint = tmp_path / "current_route_online.pt"
    _write_jsonl(base_pool_path, [{"skill_id": "base/a", "name": "base"}])
    _write_jsonl(
        dynamic_pool_path,
        [
            {"skill_id": "base/a", "name": "base"},
            {
                "skill_id": "skillx/appworld/new",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
            },
        ],
    )
    torch.save(
        {
            "stage": "appworld_current_route_online_stage4",
            "config": {"d": 8, "base_model_name": "fake"},
            "model_state_dict": {"skill_head.net.0.weight": torch.ones(1, 1)},
            "train_report": {
                "online_update_count": 2,
                "checkpoint_metadata": {
                    "dynamic_skill_pool_path": str(dynamic_pool_path),
                    "source_clstr_checkpoint_path": "stage4.pt",
                },
            },
        },
        online_checkpoint,
    )

    fake_model = _DynamicCliFakeModel()
    captured = {}

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        captured["build_checkpoint_path"] = str(checkpoint_path)
        captured["build_skills_path"] = str(skills_path)
        captured["model_cache_dir"] = model_cache_dir
        fake_model.skills = [{"skill_id": "base/a"}, {"skill_id": "skillx/appworld/new"}]
        return fake_model, {"query_text_format": "skillrouter"}, {"stage0_checkpoint_stage": "appworld_current_route_online_stage4"}

    monkeypatch.setattr(cli, "build_clstr_model_from_stage0_checkpoint", fake_build, raising=False)

    model, report = cli._load_dynamic_checkpoint_model(
        checkpoint_path=str(online_checkpoint),
        base_skill_pool_path=str(base_pool_path),
        dynamic_skill_pool_path=str(dynamic_pool_path),
    )

    assert model is fake_model
    assert captured["build_checkpoint_path"] == str(online_checkpoint)
    assert captured["build_skills_path"] == str(dynamic_pool_path)
    assert fake_model.appended_rows == []
    assert report["stage"] == "appworld_current_route_online_stage4"
    assert report["dynamic_checkpoint_load_mode"] == "current_route_online_stage4_full_dynamic_checkpoint"
    assert report["append_report"]["appended_count"] == 0
