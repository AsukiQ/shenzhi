from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_restore_native_benchmark_chain_preserves_training_prefix_then_appends_text_skills(
    tmp_path,
    monkeypatch,
):
    import clstr.native_benchmark_checkpoint_adapter as adapter

    training_skills = [
        {"skill_id": "training/a", "description": "base a"},
        {"skill_id": "training/b", "description": "base b"},
    ]
    benchmark_skills = [
        {"skill_id": "training/b", "description": "known b"},
        {"skill_id": "tau2/tool_x", "description": "new x"},
    ]
    training_skills_path = tmp_path / "training_skills.jsonl"
    _write_jsonl(training_skills_path, training_skills)
    calls: list[tuple] = []

    class _FakeModel:
        def __init__(self):
            self.skills = list(training_skills)
            self.encoder = type("Encoder", (), {"backbone": torch.nn.Linear(2, 2)})()
            for parameter in self.encoder.backbone.parameters():
                parameter.requires_grad_(False)

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def append_skills(self, rows):
            skill_ids = [str(row["skill_id"]) for row in rows]
            calls.append(("append", skill_ids))
            self.skills.extend(dict(row) for row in rows)
            return {
                "appended_count": len(rows),
                "appended_skill_ids": skill_ids,
            }

        def eval(self):
            calls.append(("eval",))
            return self

    model = _FakeModel()

    def fake_build(**kwargs):
        calls.append(
            (
                "stage0",
                Path(kwargs["skills_path"]),
                kwargs["allow_skill_table_prefix_expansion"],
            )
        )
        return model, {"freeze_backbone": True}, {
            "stage0_loaded": True,
            "stage0_loaded_keys": ["skill_table.E"],
            "local_skill_table_rebuilt": False,
        }

    def fake_overlay(_model, path, **kwargs):
        calls.append(("overlay", Path(path).name, kwargs["protect_routing_foundation"]))
        return {"loaded": True, "shape_mismatched": {}}

    monkeypatch.setattr(adapter, "build_clstr_model_from_stage0_checkpoint", fake_build)
    monkeypatch.setattr(adapter, "load_head_checkpoint_into_model", fake_overlay)

    restored_model, model_config, skill_id_to_idx, report = (
        adapter.restore_native_benchmark_checkpoint_chain(
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            stage4_checkpoint_path=tmp_path / "stage4.pt",
            training_skills_path=training_skills_path,
            benchmark_skills=benchmark_skills,
            model_cache_dir=tmp_path / "model_cache",
            device=torch.device("cpu"),
        )
    )

    assert restored_model is model
    assert model_config == {"freeze_backbone": True}
    assert calls == [
        ("stage0", training_skills_path, False),
        ("overlay", "stage2.pt", True),
        ("overlay", "stage4.pt", True),
        ("to", "cpu"),
        ("append", ["tau2/tool_x"]),
        ("eval",),
    ]
    assert skill_id_to_idx == {
        "training/a": 0,
        "training/b": 1,
        "tau2/tool_x": 2,
    }
    assert report["skill_merge"] == {
        "training_skill_count": 2,
        "benchmark_skill_count": 2,
        "known_benchmark_skill_count": 1,
        "appended_benchmark_skill_count": 1,
        "final_skill_count": 3,
    }
    assert report["skill_append"]["appended_skill_ids"] == ["tau2/tool_x"]
    assert report["trainable_backbone_parameters"] == 0


def test_restore_native_safe_memory_delta_preserves_router_and_rejects_router_keys(
    tmp_path,
    monkeypatch,
):
    import clstr.native_benchmark_checkpoint_adapter as adapter

    training_skills = [{"skill_id": "training/a", "description": "base"}]
    training_skills_path = tmp_path / "training_skills.jsonl"
    _write_jsonl(training_skills_path, training_skills)

    class _FakeModel:
        def __init__(self):
            self.skills = list(training_skills)
            self.encoder = type("Encoder", (), {"backbone": torch.nn.Linear(2, 2)})()
            for parameter in self.encoder.backbone.parameters():
                parameter.requires_grad_(False)

        def to(self, _device):
            return self

        def append_skills(self, rows):
            self.skills.extend(rows)
            return {
                "appended_count": len(rows),
                "appended_skill_ids": [row["skill_id"] for row in rows],
            }

        def eval(self):
            return self

    model = _FakeModel()
    monkeypatch.setattr(
        adapter,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (
            model,
            {"freeze_backbone": True},
            {
                "stage0_loaded": True,
                "stage0_loaded_keys": ["skill_table.E"],
                "local_skill_table_rebuilt": False,
            },
        ),
    )
    monkeypatch.setattr(
        adapter,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {"loaded": True, "shape_mismatched": {}},
    )
    digests = iter(["router-a", "router-a"])
    monkeypatch.setattr(
        adapter,
        "router_state_digest",
        lambda _model, *, scope: next(digests),
    )
    stage4 = tmp_path / "stage4.pt"
    torch.save(
        {
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
            }
        },
        stage4,
    )

    *_rest, report = adapter.restore_native_benchmark_checkpoint_chain(
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        stage2_checkpoint_path=tmp_path / "stage2.pt",
        stage4_checkpoint_path=stage4,
        training_skills_path=training_skills_path,
        benchmark_skills=training_skills,
        model_cache_dir=tmp_path / "cache",
        device=torch.device("cpu"),
        require_safe_memory_delta=True,
    )
    assert report["router_digest_unchanged"] is True

    payload = torch.load(stage4, map_location="cpu")
    payload["model_state_dict"]["initial_belief_head.weight"] = torch.zeros(1, 1)
    torch.save(payload, stage4)
    with pytest.raises(ValueError, match="forbidden Stage4 delta key"):
        adapter.restore_native_benchmark_checkpoint_chain(
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            stage4_checkpoint_path=stage4,
            training_skills_path=training_skills_path,
            benchmark_skills=training_skills,
            model_cache_dir=tmp_path / "cache",
            device=torch.device("cpu"),
            require_safe_memory_delta=True,
        )


def test_restore_native_cmc_delta_rejects_wrong_stage2_parent_digest(
    tmp_path,
    monkeypatch,
):
    import clstr.native_benchmark_checkpoint_adapter as adapter

    training_skills = [{"skill_id": "training/a", "description": "base"}]
    training_skills_path = tmp_path / "training_skills.jsonl"
    _write_jsonl(training_skills_path, training_skills)

    class _FakeModel:
        def __init__(self):
            self.skills = list(training_skills)
            self.encoder = type("Encoder", (), {"backbone": torch.nn.Linear(2, 2)})()
            for parameter in self.encoder.backbone.parameters():
                parameter.requires_grad_(False)

        def to(self, _device):
            return self

        def append_skills(self, rows):
            self.skills.extend(rows)
            return {
                "appended_count": len(rows),
                "appended_skill_ids": [row["skill_id"] for row in rows],
            }

        def eval(self):
            return self

    model = _FakeModel()
    monkeypatch.setattr(
        adapter,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (
            model,
            {"freeze_backbone": True},
            {
                "stage0_loaded": True,
                "stage0_loaded_keys": ["skill_table.E"],
                "local_skill_table_rebuilt": False,
            },
        ),
    )
    monkeypatch.setattr(
        adapter,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {"loaded": True, "shape_mismatched": {}},
    )
    monkeypatch.setattr(
        adapter,
        "router_state_digest",
        lambda _model, *, scope: f"router-{scope}",
    )
    stage4 = tmp_path / "stage4-cmc.pt"
    payload = {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "parent_stage2_full_router_digest": "wrong-full",
        "parent_stage2_fast_router_digest": "router-fast",
        "model_state_dict": {
            "route_memory_residual_adapter.net.3.weight": torch.zeros(1, 1),
            "route_memory_candidate_utility_gate.net.0.weight": torch.zeros(1, 1),
        },
    }
    torch.save(payload, stage4)

    with pytest.raises(ValueError, match="parent Stage2 router digest mismatch"):
        adapter.restore_native_benchmark_checkpoint_chain(
            stage0_checkpoint_path=tmp_path / "stage0.pt",
            stage2_checkpoint_path=tmp_path / "stage2.pt",
            stage4_checkpoint_path=stage4,
            training_skills_path=training_skills_path,
            benchmark_skills=training_skills,
            model_cache_dir=tmp_path / "cache",
            device=torch.device("cpu"),
            require_safe_memory_delta=True,
        )

    payload["parent_stage2_full_router_digest"] = "router-full"
    torch.save(payload, stage4)
    *_rest, report = adapter.restore_native_benchmark_checkpoint_chain(
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        stage2_checkpoint_path=tmp_path / "stage2.pt",
        stage4_checkpoint_path=stage4,
        training_skills_path=training_skills_path,
        benchmark_skills=training_skills,
        model_cache_dir=tmp_path / "cache",
        device=torch.device("cpu"),
        require_safe_memory_delta=True,
    )

    assert report["stage4_delta_validation"]["stage4_method"] == (
        "counterfactual_memory_calibration_v1"
    )
    assert report["router_digest_unchanged"] is True


def test_restore_native_candidate_admission_delta_uses_exact_contract(
    tmp_path,
    monkeypatch,
):
    import clstr.native_benchmark_checkpoint_adapter as adapter

    training_skills = [{"skill_id": "training/a", "description": "base"}]
    training_skills_path = tmp_path / "training-skills.jsonl"
    _write_jsonl(training_skills_path, training_skills)

    class _FakeModel:
        def __init__(self):
            self.skills = list(training_skills)
            self.encoder = type(
                "Encoder",
                (),
                {"backbone": torch.nn.Linear(2, 2)},
            )()
            for parameter in self.encoder.backbone.parameters():
                parameter.requires_grad_(False)

        def to(self, _device):
            return self

        def append_skills(self, rows):
            self.skills.extend(rows)
            return {
                "appended_count": len(rows),
                "appended_skill_ids": [row["skill_id"] for row in rows],
            }

        def eval(self):
            return self

    model = _FakeModel()
    monkeypatch.setattr(
        adapter,
        "build_clstr_model_from_stage0_checkpoint",
        lambda **_kwargs: (
            model,
            {"freeze_backbone": True},
            {
                "stage0_loaded": True,
                "stage0_loaded_keys": ["skill_table.E"],
                "local_skill_table_rebuilt": False,
            },
        ),
    )
    monkeypatch.setattr(
        adapter,
        "load_head_checkpoint_into_model",
        lambda *_args, **_kwargs: {"loaded": True, "shape_mismatched": {}},
    )
    monkeypatch.setattr(
        adapter,
        "router_state_digest",
        lambda _model, *, scope: f"router-{scope}",
    )
    stage4 = tmp_path / "stage4-candidate-admission.pt"
    torch.save(
        {
            "stage4_method": "candidate_admission_residual_v1",
            "base_cmc_checkpoint_sha256": "a" * 64,
            "parent_stage2_full_router_digest": "router-full",
            "parent_stage2_fast_router_digest": "router-fast",
            "model_state_dict": {
                "route_memory_residual_adapter.net.3.weight": torch.zeros(1, 1),
                "route_memory_candidate_admission_residual.trunk.1.weight": torch.zeros(1, 1),
            },
        },
        stage4,
    )

    *_rest, report = adapter.restore_native_benchmark_checkpoint_chain(
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        stage2_checkpoint_path=tmp_path / "stage2.pt",
        stage4_checkpoint_path=stage4,
        training_skills_path=training_skills_path,
        benchmark_skills=training_skills,
        model_cache_dir=tmp_path / "cache",
        device=torch.device("cpu"),
        require_safe_memory_delta=True,
    )

    assert report["stage4_delta_validation"]["stage4_method"] == (
        "candidate_admission_residual_v1"
    )
    assert report["router_digest_unchanged"] is True
