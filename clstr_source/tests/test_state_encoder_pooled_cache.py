from types import SimpleNamespace

import torch
import torch.nn as nn

from clstr.encoders import StateEncoder


class _FakeBackbone(nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = input_ids.unsqueeze(-1).to(torch.float32).repeat(1, 1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


def _build_encoder(*, normalize_embeddings: bool = True) -> StateEncoder:
    encoder = StateEncoder.__new__(StateEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = _FakeBackbone()
    encoder.proj = nn.Linear(3, 2)
    encoder.pooling = "last_token"
    encoder.normalize_embeddings = normalize_embeddings
    encoder.tokenize = lambda _texts: {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    return encoder


def test_state_encoder_forward_composes_backbone_pool_and_projection():
    encoder = _build_encoder()

    pooled = encoder.encode_backbone_pooled(["a", "b"])
    expected = encoder.project_pooled(pooled)
    actual = encoder(["a", "b"])

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.norm(dim=-1),
        torch.ones(2),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_project_pooled_keeps_projection_in_the_trainable_graph():
    encoder = _build_encoder(normalize_embeddings=False)
    pooled = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    encoder.project_pooled(pooled).sum().backward()

    assert encoder.proj.weight.grad is not None
    assert encoder.proj.bias.grad is not None


def test_encode_tokenized_uses_the_same_projection_boundary():
    encoder = _build_encoder()
    tokenized = encoder.tokenize(["a", "b"])
    backbone_output = encoder.backbone(**tokenized)
    pooled = backbone_output.last_hidden_state[:, -1]

    torch.testing.assert_close(
        encoder.encode_tokenized(tokenized),
        encoder.project_pooled(pooled),
        rtol=0.0,
        atol=0.0,
    )
