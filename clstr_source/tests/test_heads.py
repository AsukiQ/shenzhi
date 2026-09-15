import torch

from clstr.heads import SkillHead, StopHead, TransHead


def test_skill_head_output_shape():
    head = SkillHead(d=4)
    out = head(torch.ones(2, 4), torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_skill_head_supports_batched_candidates():
    head = SkillHead(d=4)
    out = head(torch.ones(3, 5, 4), torch.zeros(3, 4))
    assert out.shape == (3, 5)


def test_stop_head_output_shape():
    head = StopHead(d=4)
    out = head(torch.ones(2, 4), torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_trans_head_output_shape():
    head = TransHead(d=4, use_sn=True)
    out = head(torch.ones(2, 4), torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_trans_head_supports_candidate_batches():
    head = TransHead(d=4, d_a=2, use_sn=True)
    out = head(torch.ones(2, 4), torch.zeros(2, 3, 2))
    assert out.shape == (2, 3)
