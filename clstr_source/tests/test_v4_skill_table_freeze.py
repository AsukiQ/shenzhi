from __future__ import annotations

import torch

from clstr.encoders import SkillTable


def _encoder_fn(d: int):
    def encode(texts: list[str]) -> torch.Tensor:
        embs = torch.zeros(len(texts), d)
        for i in range(len(texts)):
            embs[i, i % d] = 1.0
        return embs

    return encode


def _tiny_skills(n: int) -> list[dict]:
    return [
        {
            "name": f"skill_{i}",
            "description": f"d{i}",
            "input_schema": {},
            "output_schema": {},
            "executor_desc": "noop",
            "failure_modes": [],
        }
        for i in range(n)
    ]


def test_set_retrieval_trainable_freezes_only_retrieval_path():
    st = SkillTable(_tiny_skills(4), _encoder_fn(8), d=8, adapter_init="identity", trainable=True)

    assert hasattr(st, "set_retrieval_trainable")
    assert st.set_retrieval_trainable(False) is st

    assert st.W.weight.requires_grad is False
    assert st.E.requires_grad is False
    assert st.logit_scale_retr.requires_grad is False
    assert st.skill_bias_retr.requires_grad is False
    assert st.logit_scale_belief.requires_grad is True
    assert st.skill_bias_belief.requires_grad is True


def test_set_belief_scale_trainable_controls_belief_scale_and_bias():
    st = SkillTable(_tiny_skills(4), _encoder_fn(8), d=8, adapter_init="identity", trainable=True)

    assert hasattr(st, "set_belief_scale_trainable")
    assert st.set_belief_scale_trainable(False) is st

    assert st.logit_scale_belief.requires_grad is False
    assert st.skill_bias_belief.requires_grad is False
    assert st.W.weight.requires_grad is True
    assert st.E.requires_grad is True
    assert st.logit_scale_retr.requires_grad is True
    assert st.skill_bias_retr.requires_grad is True


def test_set_trainable_still_exists_for_legacy_full_freeze():
    st = SkillTable(_tiny_skills(4), _encoder_fn(8), d=8, adapter_init="identity", trainable=True)

    assert st.set_trainable(False) is st
    assert all(param.requires_grad is False for _, param in st.named_parameters())

    st.set_trainable(True)
    assert all(param.requires_grad is True for _, param in st.named_parameters())


def test_skill_table_reports_embedding_build_progress_batches():
    events = []

    st = SkillTable(
        _tiny_skills(5),
        _encoder_fn(8),
        d=8,
        adapter_init="identity",
        embedding_batch_size=2,
        embedding_progress_callback=events.append,
    )

    assert st.E.shape == (5, 8)
    assert events == [
        {"batch_index": 1, "batch_count": 3, "start": 0, "end": 2, "total": 5},
        {"batch_index": 2, "batch_count": 3, "start": 2, "end": 4, "total": 5},
        {"batch_index": 3, "batch_count": 3, "start": 4, "end": 5, "total": 5},
    ]


def test_retrieval_freeze_blocks_retrieval_param_grads_but_not_belief_grads():
    st = SkillTable(_tiny_skills(4), _encoder_fn(8), d=8, adapter_init="identity", trainable=True)
    st.set_retrieval_trainable(False)

    h = torch.randn(2, 8, requires_grad=True)
    loss = st.retrieval_logits(h).sum() + st.belief_logits(h).sum()
    loss.backward()

    assert st.W.weight.grad is None
    assert st.E.grad is None
    assert st.logit_scale_retr.grad is None
    assert st.skill_bias_retr.grad is None
    assert st.logit_scale_belief.grad is not None
    assert st.skill_bias_belief.grad is not None
    assert h.grad is not None
