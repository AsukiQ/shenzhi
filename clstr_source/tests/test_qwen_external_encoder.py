import json
from pathlib import Path

import torch
from clstr.external_data import write_jsonl

from clstr.qwen_external_encoder import (
    QwenExternalEncoderConfig,
    FrozenQwenExternalEncoder,
    build_qwen_external_clstr_config,
    build_qwen_external_clstr_model,
    qwen_external_metadata,
)
from clstr.qwen_full_base_train import run_clstr_qwen3_full_base_train
from clstr.qwen_stage4_act_train import run_clstr_qwen3_stage4_act_train


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]] * batch, dtype=torch.long),
        }


class _FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4, "max_position_embeddings": 16})()
        self.weight = torch.nn.Parameter(torch.ones(1))

    @property
    def device(self):
        return self.weight.device

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = torch.arange(input_ids.numel() * 4, dtype=torch.float32).view(input_ids.size(0), input_ids.size(1), 4)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_frozen_qwen_external_encoder_left_pads_freezes_backbone_and_projects(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_backbone = _FakeBackbone()

    monkeypatch.setattr(
        "clstr.qwen_external_encoder.AutoTokenizer",
        type("TokLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: fake_tokenizer)}),
    )
    monkeypatch.setattr(
        "clstr.qwen_external_encoder.AutoModel",
        type("ModelLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: fake_backbone)}),
    )

    encoder = FrozenQwenExternalEncoder(
        QwenExternalEncoderConfig(
            model_name_or_path="fake-qwen",
            output_dim=2,
            pooling="last_token",
            freeze_backbone=True,
            projection_trainable=True,
        )
    )
    encoded = encoder(["alpha", "beta"])

    assert encoded.shape == (2, 2)
    assert fake_tokenizer.padding_side == "left"
    assert fake_tokenizer.pad_token == "<eos>"
    assert all(not param.requires_grad for param in fake_backbone.parameters())
    assert any(param.requires_grad for param in encoder.proj.parameters())


def test_build_qwen_external_clstr_config_uses_identity_projection_when_dim_matches(monkeypatch):
    monkeypatch.setattr(
        "clstr.qwen_external_encoder.AutoConfig",
        type("ConfigLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: type("Cfg", (), {"hidden_size": 4096})())}),
    )

    config = build_qwen_external_clstr_config(
        model_name_or_path="Qwen/Qwen3-8B",
        max_length=2048,
    )

    assert config.base_model_name == "Qwen/Qwen3-8B"
    assert config.d == 4096
    assert config.d_a == 4096
    assert config.encoder_pooling == "last_token"
    assert config.tokenizer_padding_side == "left"
    assert config.freeze_backbone is True
    assert config.projection_init == "identity"
    assert config.use_cross_encoder is False
    assert config.defer_skill_table_init is True


def test_build_qwen_external_clstr_model_passes_cache_and_offline_to_state_encoder(monkeypatch):
    captured = {}

    class _FakeModel:
        def __init__(self, config, skills):
            captured["config"] = config
            captured["skills"] = skills
            self.qwen_external_metadata = {}

    monkeypatch.setattr(
        "clstr.qwen_external_encoder.AutoConfig",
        type("ConfigLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: type("Cfg", (), {"hidden_size": 4})())}),
    )
    monkeypatch.setattr("clstr.qwen_external_encoder.CLSTRModel", _FakeModel)

    _model, config, report = build_qwen_external_clstr_model(
        [{"skill_id": "skill/a", "name": "a", "description": "a"}],
        model_name_or_path="models/Qwen3-8B",
        cache_dir=".cache/huggingface",
        local_files_only=True,
        model_dim=4,
    )

    assert captured["config"].hf_cache_dir == ".cache/huggingface"
    assert captured["config"].local_files_only is True
    assert config["hf_cache_dir"] == ".cache/huggingface"
    assert config["local_files_only"] is True
    assert report["qwen_external_encoder"] is True


def test_qwen_full_base_train_forwards_stage1_routing_checkpoint(monkeypatch, tmp_path):
    captured = {}
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    write_jsonl(train_path, [{"state_text": "state", "skill_id": "skill/a", "loss_mask": {}}])
    write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    checkpoint_path.write_bytes(b"placeholder")

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    def fake_build(skills, **kwargs):
        captured["build_skills"] = skills
        captured["build_kwargs"] = kwargs
        return _FakeModel(), {"d": 4}, {"qwen_external_encoder": True}

    def fake_train(**kwargs):
        captured["train_kwargs"] = kwargs
        return {
            "status": "ok",
            "routing_checkpoint": {"path": str(kwargs["routing_checkpoint_path"])},
            "setup_status_path": str(kwargs["setup_status_path"]),
        }

    monkeypatch.setattr("clstr.qwen_full_base_train.build_qwen_external_clstr_model", fake_build)
    monkeypatch.setattr("clstr.qwen_full_base_train.train_clstr_full_base_with_model", fake_train)

    report = run_clstr_qwen3_full_base_train(
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        qwen_model_path="models/Qwen3-8B",
        routing_checkpoint_path=checkpoint_path,
        embedding_cache_mode="auto",
        embedding_cache_max_rows=123,
        max_steps=1,
        batch_size=1,
        local_files_only=True,
    )

    assert report["status"] == "ok"
    assert captured["train_kwargs"]["routing_checkpoint_path"] == checkpoint_path
    assert captured["train_kwargs"]["embedding_cache_mode"] == "auto"
    assert captured["train_kwargs"]["embedding_cache_max_rows"] == 123
    assert captured["train_kwargs"]["setup_status_path"] == tmp_path / "out" / "setup_status.jsonl"
    assert report["routing_checkpoint"]["path"] == str(checkpoint_path)
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for phase in [
        "qwen_stage2_started",
        "qwen_stage2_checkpoint_preflight_ok",
        "qwen_model_load_started",
        "qwen_model_loaded",
    ]:
        assert phase in setup_phases


def test_qwen_full_base_train_blocks_before_loading_qwen_when_stage1_checkpoint_missing(monkeypatch, tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    missing_checkpoint = tmp_path / "missing-stage1.pt"
    write_jsonl(train_path, [{"state_text": "state", "skill_id": "skill/a", "loss_mask": {}}])
    write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    called = {"build": False}

    def fake_build(*args, **kwargs):
        called["build"] = True
        raise AssertionError("Qwen should not be loaded when routing checkpoint is absent")

    monkeypatch.setattr("clstr.qwen_full_base_train.build_qwen_external_clstr_model", fake_build)

    report = run_clstr_qwen3_full_base_train(
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        qwen_model_path="models/Qwen3-8B",
        routing_checkpoint_path=missing_checkpoint,
        max_steps=1,
        batch_size=1,
        local_files_only=True,
    )

    assert called["build"] is False
    assert report["status"] == "blocked"
    assert report["error_type"] == "FileNotFoundError"
    assert "routing checkpoint not found" in report["error"]
    assert Path(report["setup_status_path"]).is_file()
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert setup_phases == ["qwen_stage2_started", "qwen_stage2_blocked"]


def test_qwen_stage4_act_train_blocks_before_loading_qwen_when_checkpoint_missing(monkeypatch, tmp_path):
    trajectories_path = tmp_path / "trajectories.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    missing_stage1 = tmp_path / "missing-stage1.pt"
    missing_stage23 = tmp_path / "missing-stage23.pt"
    write_jsonl(trajectories_path, [{"state_text": "state", "provenance": {"split": "train"}}])
    write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    called = {"build": False}

    def fake_build(*args, **kwargs):
        called["build"] = True
        raise AssertionError("Qwen should not be loaded when required checkpoints are absent")

    monkeypatch.setattr("clstr.qwen_stage4_act_train.build_qwen_external_clstr_model", fake_build)

    report = run_clstr_qwen3_stage4_act_train(
        trajectories_path=trajectories_path,
        skills_path=skills_path,
        output_dir=tmp_path / "stage4",
        qwen_model_path="models/Qwen3-8B",
        routing_checkpoint_path=missing_stage1,
        head_checkpoint_path=missing_stage23,
        max_steps=1,
        batch_size=1,
        local_files_only=True,
    )

    assert called["build"] is False
    assert report["status"] == "blocked"
    assert report["error_type"] == "FileNotFoundError"
    assert "routing checkpoint not found" in report["error"]
    assert Path(report["setup_status_path"]).is_file()
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert setup_phases == ["qwen_stage4_started", "qwen_stage4_blocked"]


def test_qwen_stage4_act_train_loads_stage1_and_head_checkpoints(monkeypatch, tmp_path):
    captured = {}
    trajectories_path = tmp_path / "trajectories.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    routing_checkpoint = tmp_path / "stage1.pt"
    head_checkpoint = tmp_path / "stage2.pt"
    write_jsonl(trajectories_path, [{"state_text": "state", "provenance": {"split": "train"}}])
    write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    torch.save({"model_state_dict": {"skill_table.E": torch.zeros(1, 4)}}, routing_checkpoint)
    torch.save(
        {
            "model_state_dict": {
                "skill_table.E": torch.ones(1, 4) * 9.0,
                "trans_head.weight": torch.ones(1, 4),
            }
        },
        head_checkpoint,
    )

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.full((1, 4), -1.0))
            self.trans_head = torch.nn.Linear(4, 1, bias=False)

        def load_state_dict(self, state, strict=False):
            captured["loaded_state"] = state
            captured["strict"] = strict
            return super().load_state_dict(state, strict=strict)

    def fake_build(skills, **kwargs):
        captured["build_skills"] = skills
        captured["build_kwargs"] = kwargs
        return _FakeModel(), {"d": 4}, {"qwen_external_encoder": True}

    def fake_stage4_train(**kwargs):
        captured["train_kwargs"] = kwargs
        return {
            "status": "ok",
            "checkpoint_init": kwargs["checkpoint_init_report"],
            "setup_status_path": str(kwargs["setup_status_path"]),
        }

    monkeypatch.setattr("clstr.qwen_stage4_act_train.build_qwen_external_clstr_model", fake_build)
    monkeypatch.setattr("clstr.qwen_stage4_act_train.train_stage4_act_with_model", fake_stage4_train)

    report = run_clstr_qwen3_stage4_act_train(
        trajectories_path=trajectories_path,
        skills_path=skills_path,
        output_dir=tmp_path / "stage4",
        qwen_model_path="models/Qwen3-8B",
        routing_checkpoint_path=routing_checkpoint,
        head_checkpoint_path=head_checkpoint,
        max_steps=1,
        batch_size=1,
        local_files_only=True,
    )

    assert report["status"] == "ok"
    assert captured["build_kwargs"]["local_files_only"] is True
    assert captured["strict"] is False
    assert captured["train_kwargs"]["checkpoint_init_report"]["routing_checkpoint_path"] == str(routing_checkpoint)
    assert captured["train_kwargs"]["checkpoint_init_report"]["head_checkpoint_path"] == str(head_checkpoint)
    assert captured["train_kwargs"]["setup_status_path"] == tmp_path / "stage4" / "setup_status.jsonl"
    assert "skill_table.E" in captured["loaded_state"]
    assert "trans_head.weight" in captured["loaded_state"]
    assert captured["train_kwargs"]["checkpoint_init_report"]["protect_routing_foundation"] is True
    assert captured["train_kwargs"]["checkpoint_init_report"]["skipped_head_routing_foundation_keys"] == ["skill_table.E"]
    assert captured["train_kwargs"]["route_scorer"] == "unified_memory"
    assert torch.allclose(captured["train_kwargs"]["model"].skill_table.E, torch.zeros(1, 4))
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for phase in [
        "qwen_stage4_started",
        "qwen_stage4_checkpoint_preflight_ok",
        "qwen_model_load_started",
        "qwen_model_loaded",
        "stage4_checkpoint_init_loaded",
    ]:
        assert phase in setup_phases


def test_qwen_full_base_training_rebuilds_deferred_skill_table_after_device_move(monkeypatch, tmp_path):
    from clstr.full_base_train import train_clstr_full_base_with_model
    from clstr.model import CLSTRConfig
    from clstr.external_data import write_jsonl

    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "task/1",
                "state_text": "goal: inspect room\nobservation: room",
                "action_text": "look",
                "admissible_actions": ["look"],
                "expert_action": "look",
                "next_observation_text": "room",
                "done": False,
                "skill_id": "skill/look",
                "loss_mask": {"L_policy": False, "routing": False, "L_trans": False, "belief": False, "STOP": False},
                "provenance": {"split": "train"},
            }
        ],
    )
    write_jsonl(skills_path, [{"skill_id": "skill/look", "name": "look", "description": "look"}])

    class _DeferredModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = CLSTRConfig(defer_skill_table_init=True, d=4)
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.rebuild_seen_device = None
            self.skills = [{"skill_id": "skill/look"}]

        def rebuild_skill_table(self):
            self.rebuild_seen_device = self.weight.device

        def encode_observations(self, texts):
            return torch.zeros(len(texts), 4, device=self.weight.device)

    model = _DeferredModel()
    train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 4},
        routing_report={},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        loss_weights={"policy": 0.0, "transition": 0.0, "transition_skill_ce": 0.0, "belief": 0.0, "stop": 0.0, "retrieval": 0.0},
        allow_full_pool_stage2_debug=True,
    )

    assert model.rebuild_seen_device == model.weight.device


def test_qwen_external_metadata_marks_frozen_encoder_not_direct_generator():
    metadata = qwen_external_metadata(
        model_name_or_path="/root/autodl-tmp/clstr/models/Qwen3-8B",
        quantization_mode="none",
        output_dim=4096,
    )

    assert metadata["qwen_external_encoder"] is True
    assert metadata["qwen_frozen"] is True
    assert metadata["qwen_direct_generator"] is False
    assert metadata["native_clstr_heads_used"] is True
