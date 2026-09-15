from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from clstr.belief import subspace_obs
from clstr.encoders import SkillTable


def _make_encoder_fn(d: int):
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
            "description": f"desc {i}",
            "input_schema": {},
            "output_schema": {},
            "executor_desc": "noop",
            "failure_modes": [],
        }
        for i in range(n)
    ]


def test_skill_table_exposes_separate_retrieval_and_belief_scale_params():
    st = SkillTable(_tiny_skills(4), _make_encoder_fn(8), d=8, adapter_init="identity")

    assert isinstance(st.logit_scale_retr, torch.nn.Parameter)
    assert isinstance(st.skill_bias_retr, torch.nn.Parameter)
    assert isinstance(st.logit_scale_belief, torch.nn.Parameter)
    assert isinstance(st.skill_bias_belief, torch.nn.Parameter)
    assert st.skill_bias_retr.shape == (4,)
    assert st.skill_bias_belief.shape == (4,)
    assert torch.allclose(st.logit_scale_retr, torch.tensor([0.0]))
    assert float(st.logit_scale_belief.exp().detach().item()) < 1.0


def test_retrieval_logits_are_l2_normalized_scaled_and_biased():
    st = SkillTable(_tiny_skills(4), _make_encoder_fn(8), d=8, adapter_init="identity")
    h = torch.randn(2, 8) * 3.0

    out = st.retrieval_logits(h)
    q = F.normalize(st.W(h), p=2, dim=-1)
    e = F.normalize(st.E, p=2, dim=-1)
    expected = st.logit_scale_retr.exp().clamp(max=100.0) * (q @ e.t()) + st.skill_bias_retr.unsqueeze(0)

    assert out.shape == (2, 4)
    assert torch.allclose(out, expected, atol=1.0e-5)


def test_belief_logits_use_independent_scale_and_bias():
    st = SkillTable(_tiny_skills(4), _make_encoder_fn(8), d=8, adapter_init="identity")
    h = torch.randn(1, 8)

    with torch.no_grad():
        st.logit_scale_retr.fill_(math.log(50.0))
        st.logit_scale_belief.fill_(math.log(0.25))
        st.skill_bias_retr[1] = 5.0
        st.skill_bias_belief[2] = 7.0

    retr = st.retrieval_logits(h)
    belief = st.belief_logits(h)

    assert not torch.allclose(retr, belief)
    assert torch.isclose(retr[0, 1] - st.logit_scale_retr.exp().clamp(max=100.0) * (F.normalize(st.W(h), p=2, dim=-1) @ F.normalize(st.E, p=2, dim=-1).t())[0, 1], torch.tensor(5.0))
    assert torch.isclose(belief[0, 2] - st.logit_scale_belief.exp().clamp(max=100.0) * (F.normalize(st.W(h), p=2, dim=-1) @ F.normalize(st.E, p=2, dim=-1).t())[0, 2], torch.tensor(7.0))


def test_subspace_obs_uses_belief_logits_not_retrieval_logits():
    st = SkillTable(_tiny_skills(3), _make_encoder_fn(3), d=3, adapter_init="identity")
    h = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        st.logit_scale_retr.fill_(math.log(100.0))
        st.logit_scale_belief.fill_(math.log(0.2))
        st.skill_bias_retr.zero_()
        st.skill_bias_belief.zero_()

    obs = subspace_obs(st, h)
    expected = torch.softmax(st.belief_logits(h), dim=-1) @ st.E
    wrong = torch.softmax(st.retrieval_logits(h), dim=-1) @ st.E

    assert torch.allclose(obs, expected, atol=1.0e-6)
    assert not torch.allclose(obs, wrong, atol=1.0e-3)


def test_retrieval_scale_clamped_at_100():
    st = SkillTable(_tiny_skills(4), _make_encoder_fn(8), d=8, adapter_init="identity")
    with torch.no_grad():
        st.logit_scale_retr.fill_(math.log(200.0))

    h = torch.randn(1, 8)
    q = F.normalize(st.W(h), p=2, dim=-1)
    e = F.normalize(st.E, p=2, dim=-1)
    expected = 100.0 * (q @ e.t())

    assert torch.allclose(st.retrieval_logits(h), expected, atol=1.0e-4)


def test_logits_grad_flows_to_retrieval_and_belief_params():
    st = SkillTable(_tiny_skills(4), _make_encoder_fn(8), d=8, adapter_init="identity", trainable=True)
    h = torch.randn(2, 8)

    loss = st.retrieval_logits(h).sum() + subspace_obs(st, h).sum()
    loss.backward()

    assert st.logit_scale_retr.grad is not None
    assert st.skill_bias_retr.grad is not None
    assert st.logit_scale_belief.grad is not None
    assert st.skill_bias_belief.grad is not None
    assert st.W.weight.grad is not None
    assert st.E.grad is not None
