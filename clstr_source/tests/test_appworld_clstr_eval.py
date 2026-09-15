import torch
import pytest

from clstr.appworld_clstr_eval import rank_clstr_skills_for_query


class _PolicyEvalSkillTable:
    def __init__(self):
        self.E = torch.eye(3)

    def logits(self, h):
        return torch.tensor([[3.0, 2.0, 1.0]])

    def retrieval_logits(self, h):
        return self.logits(h)

    def belief_logits(self, h):
        return self.logits(h)


class _PolicyEvalModel:
    def __init__(self):
        self.skill_table = _PolicyEvalSkillTable()

    def encode_states(self, states):
        return torch.tensor([[1.0, 0.0, 0.0]])

    def batch_cross_encode(self, states, candidate_rows):
        return torch.eye(3).view(1, 3, 3)[:, : len(candidate_rows[0]), :]

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        # Candidate order follows skill_table top-k: skill-a, skill-b, skill-c.
        # The ACT policy should be able to override that base routing order.
        return torch.tensor([[0.0, 5.0, 1.0, -100.0]])


def test_policy_head_ranking_requires_explicit_legacy_opt_in():
    with pytest.raises(ValueError, match="legacy_policy_skill_router"):
        rank_clstr_skills_for_query(
            model=_PolicyEvalModel(),
            query="spotify task",
            skill_ids=["skill-a", "skill-b", "skill-c"],
            top_k=2,
            ranking_mode="policy_head",
            candidate_top_k=3,
        )


def test_policy_head_ranking_uses_act_policy_not_raw_skill_table_order_when_legacy_opted_in():
    ranked, scores, diagnostics = rank_clstr_skills_for_query(
        model=_PolicyEvalModel(),
        query="spotify task",
        skill_ids=["skill-a", "skill-b", "skill-c"],
        top_k=2,
        ranking_mode="policy_head",
        candidate_top_k=3,
        allow_legacy_policy_skill_router=True,
    )

    assert ranked == ["skill-b", "skill-c"]
    assert scores[0] > scores[1]
    assert diagnostics["base_candidate_skill_ids"] == ["skill-a", "skill-b", "skill-c"]
    assert diagnostics["legacy_policy_skill_router"] is True


def test_policy_head_candidate_top_k_limits_policy_rerank_pool():
    ranked, _scores, diagnostics = rank_clstr_skills_for_query(
        model=_PolicyEvalModel(),
        query="spotify task",
        skill_ids=["skill-a", "skill-b", "skill-c"],
        top_k=3,
        ranking_mode="policy_head",
        candidate_top_k=2,
        allow_legacy_policy_skill_router=True,
    )

    assert ranked == ["skill-b", "skill-a"]
    assert diagnostics["candidate_top_k"] == 2
    assert diagnostics["base_candidate_skill_ids"] == ["skill-a", "skill-b"]


def test_policy_blend_keeps_base_prior_but_allows_policy_adjustment():
    conservative, _scores, conservative_diag = rank_clstr_skills_for_query(
        model=_PolicyEvalModel(),
        query="spotify task",
        skill_ids=["skill-a", "skill-b", "skill-c"],
        top_k=3,
        ranking_mode="policy_blend",
        candidate_top_k=3,
        policy_blend_alpha=0.25,
        allow_legacy_policy_skill_router=True,
    )
    assert conservative == ["skill-a", "skill-b", "skill-c"]
    assert conservative_diag["policy_blend_alpha"] == 0.25

    policy_weighted, _scores, policy_diag = rank_clstr_skills_for_query(
        model=_PolicyEvalModel(),
        query="spotify task",
        skill_ids=["skill-a", "skill-b", "skill-c"],
        top_k=3,
        ranking_mode="policy_blend",
        candidate_top_k=3,
        policy_blend_alpha=0.75,
        allow_legacy_policy_skill_router=True,
    )
    assert policy_weighted[0] == "skill-b"
    assert policy_diag["ranking_mode"] == "policy_blend"
