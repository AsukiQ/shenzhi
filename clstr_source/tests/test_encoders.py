import subprocess
import sys
from pathlib import Path

from clstr.model import CLSTRConfig, CLSTRModel
from clstr.encoders import SkillTable, StateEncoder


def test_state_encoder_truncates_to_backbone_position_limit(tmp_path):
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    encoder = StateEncoder(str(model_dir), d=16)

    tokenized = encoder.tokenize(["word " * 1000])

    assert tokenized["input_ids"].shape[1] <= encoder.backbone.config.max_position_embeddings


def test_state_encoder_exposes_skillrouter_warm_start_knobs(tmp_path):
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    encoder = StateEncoder(
        str(model_dir),
        d=16,
        pooling="last_token",
        tokenizer_padding_side="left",
        torch_dtype="float32",
        freeze_backbone=True,
    )

    assert encoder.pooling == "last_token"
    assert encoder.tokenizer.padding_side == "left"
    assert all(not param.requires_grad for param in encoder.backbone.parameters())


def test_state_encoder_can_preserve_skillrouter_geometry_with_identity_projection(tmp_path):
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    encoder = StateEncoder(
        str(model_dir),
        d=32,
        projection_init="identity",
        normalize_embeddings=True,
    )

    import torch

    assert torch.allclose(encoder.proj.weight, torch.eye(32))
    assert torch.allclose(encoder.proj.bias, torch.zeros(32))
    encoded = encoder(["query"])
    assert torch.allclose(encoded.norm(dim=1), torch.ones(1), atol=1e-5)


def test_clstr_config_passes_skillrouter_encoder_options_to_backbones(tmp_path):
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(model_dir),
            d=16,
            d_a=4,
            encoder_pooling="last_token",
            cross_encoder_pooling="last_token",
            tokenizer_padding_side="left",
            torch_dtype="float32",
            freeze_backbone=True,
        ),
        [
            {
                "skill_id": "skill-a",
                "name": "skill a",
                "description": "query helper",
            }
        ],
    )

    assert model.encoder.pooling == "last_token"
    assert model.cross_encoder.pooling == "last_token"
    assert model.encoder.tokenizer.padding_side == "left"
    assert model.cross_encoder.tokenizer.padding_side == "left"
    assert all(not param.requires_grad for param in model.encoder.backbone.parameters())
    assert any(param.requires_grad for param in model.encoder.proj.parameters())


def test_clstr_config_can_identity_initialize_skillrouter_projection_and_adapter(tmp_path):
    model_dir = tmp_path / "tiny-model"
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(model_dir),
            d=32,
            d_a=8,
            projection_init="identity",
            skill_table_adapter_init="identity",
            normalize_embeddings=True,
        ),
        [{"skill_id": "skill-a", "name": "skill a", "description": "query helper"}],
    )

    import torch

    assert torch.allclose(model.encoder.proj.weight, torch.eye(32))
    assert torch.allclose(model.skill_table.W.weight, torch.eye(32))


def test_skill_table_rebuilds_embeddings_in_batches():
    batch_sizes = []

    def encoder(texts):
        batch_sizes.append(len(texts))
        import torch

        return torch.ones(len(texts), 2)

    table = SkillTable(
        skills=[
            {"name": "a", "description": "A"},
            {"name": "b", "description": "B"},
            {"name": "c", "description": "C"},
        ],
        encoder_fn=encoder,
        d=2,
        trainable=False,
        initialize_embeddings=False,
        embedding_batch_size=2,
    )

    table.rebuild_embeddings()

    assert batch_sizes == [2, 1]
