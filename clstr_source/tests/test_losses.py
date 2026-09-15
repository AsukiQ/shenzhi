from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from clstr.data import ExecutionState, ReplayStep, RetrievalPositive, VerifiedPair, serialize_execution_state
from clstr.losses import (
    action_loss,
    make_strong_positives,
    multi_positive_routing_loss,
    make_weak_positives,
    policy_loss,
    reference_kl_loss,
    retrieval_loss,
    total_loss,
    transition_loss,
)


@dataclass
class DummyStep:
    x: ExecutionState | None = None
    m_hat: torch.Tensor | None = None
    m_tilde_next: torch.Tensor | None = None
    log_prob: torch.Tensor | None = None
    skill_idx: int | None = None
    policy_logits: torch.Tensor | None = None
    ref_logits: torch.Tensor | None = None


@dataclass
class DummyTraj:
    steps: list
    reward: float
    task_id: str


class FakeEncoder:
    def __call__(self, texts):
        rows = []
        for idx, text in enumerate(texts):
            rows.append(torch.tensor([float(len(text) % 5 + 1), float(idx + 1)]))
        return torch.stack(rows)


class FakeSkillTable:
    def __init__(self):
        self.E = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 1.5],
            ]
        )

    def logits(self, h):
        return h @ self.E.t()


class FakeTransition:
    def __call__(self, m_t, a_t, o_t_emb):
        return m_t + o_t_emb + a_t.unsqueeze(-1).float()


class FakeTransHead:
    def __call__(self, m_hat, cand_emb):
        return (cand_emb.sum(dim=-1) + m_hat.sum(dim=-1, keepdim=True)).float()


class FakeActionEmbedding:
    def __call__(self, indices):
        stacked = []
        for idx in indices.tolist():
            stacked.append(torch.tensor([float(idx), float(idx + 1)]))
        return torch.stack(stacked)


class FakeModel:
    def __init__(self):
        self.encoder = FakeEncoder()
        self.skill_table = FakeSkillTable()
        self.transition = FakeTransition()
        self.trans_head = FakeTransHead()
        self.action_emb = FakeActionEmbedding()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    @property
    def device(self):
        return self._param.device


class FakeTask:
    def __init__(self, task_id, query, positive_skill_ids):
        self.task_id = task_id
        self.query = query
        self.meta = {"positive_skill_ids": list(positive_skill_ids)}


def _state(query="q", history=None, observation="obs"):
    return ExecutionState(
        query=query,
        history=history or [],
        observation=observation,
        artifact={},
        error=None,
    )


def test_transition_loss_skips_stop_steps_and_returns_scalar():
    traj = DummyTraj(
        steps=[
            DummyStep(m_hat=torch.tensor([1.0, 0.0]), m_tilde_next=torch.tensor([0.0, 1.0])),
            DummyStep(m_hat=None, m_tilde_next=None),
        ],
        reward=1.0,
        task_id="t1",
    )
    loss = transition_loss([traj])
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_policy_loss_normalizes_rewards_per_task():
    trajs = [
        DummyTraj(
            steps=[DummyStep(log_prob=torch.tensor(-0.2))],
            reward=1.0,
            task_id="t1",
        ),
        DummyTraj(
            steps=[DummyStep(log_prob=torch.tensor(-0.4))],
            reward=3.0,
            task_id="t1",
        ),
    ]
    loss = policy_loss(trajs)
    assert loss.ndim == 0


def test_make_weak_and_strong_positives_follow_variant_contract():
    success_traj = DummyTraj(
        steps=[
            DummyStep(x=_state("q1"), skill_idx=2),
            DummyStep(x=_state("q2"), skill_idx=None),
        ],
        reward=1.0,
        task_id="ok",
    )
    failure_traj = DummyTraj(
        steps=[DummyStep(x=_state("q3"), skill_idx=1)],
        reward=0.0,
        task_id="bad",
    )
    weak = make_weak_positives([success_traj, failure_traj])
    assert len(weak) == 1
    assert weak[0].positive_skill_idx == 2

    verified = [
        VerifiedPair(
            state_before=_state("vq"),
            action_at_t=0,
            obs_at_t="obs",
            candidates_next=[1, 2, 3],
            a_next_plus=2,
        )
    ]
    strong = make_strong_positives(verified)
    assert len(strong) == 1
    assert strong[0].positive_skill_idx == 2


def test_retrieval_loss_returns_scalar_with_explicit_positive_exclusion():
    model = FakeModel()
    positives = [RetrievalPositive(state=_state("abc"), positive_skill_idx=1)]
    loss = retrieval_loss(model, positives, num_negatives=2, hard_ratio=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_action_loss_prefers_exact_snapshot_and_reports_raw_recall_stats():
    model = FakeModel()
    stats = {}
    verified = [
        VerifiedPair(
            state_before=_state("vp"),
            action_at_t=1,
            obs_at_t="obs",
            candidates_next=[1, 2, 3],
            a_next_plus=2,
            was_in_raw_topk=False,
            m_t_exact=torch.tensor([1.0, 2.0]),
        )
    ]
    loss = action_loss(model, verified, stats=stats)
    assert loss.ndim == 0
    assert stats["action_loss_used"] == 1
    assert stats["retrieval_recall_raw"] == 0.0
    assert stats["retrieval_miss_raw"] == 1


def test_action_loss_replays_prefix_when_exact_snapshot_missing():
    model = FakeModel()
    verified = [
        VerifiedPair(
            state_before=_state("vp", history=[("a", "b")]),
            action_at_t=1,
            obs_at_t="obs",
            candidates_next=[1, 2, 3],
            a_next_plus=3,
            replay_prefix=[
                ReplayStep(
                    x=_state("root"),
                    skill_idx=0,
                    obs="first",
                )
            ],
        )
    ]
    loss = action_loss(model, verified)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_action_loss_rejects_approximate_m_t_by_default():
    model = FakeModel()
    verified = [
        VerifiedPair(
            state_before=_state("vp"),
            action_at_t=1,
            obs_at_t="obs",
            candidates_next=[1, 2, 3],
            a_next_plus=3,
        )
    ]

    try:
        action_loss(model, verified)
    except ValueError as exc:
        assert "m_t_exact" in str(exc)
    else:
        raise AssertionError("expected strict action_loss to reject approximate m_t")


def test_total_loss_separates_base_and_act_variants():
    model = FakeModel()
    trajs = [
        DummyTraj(
            steps=[
                DummyStep(
                    x=_state("q"),
                    m_hat=torch.tensor([1.0, 0.0]),
                    m_tilde_next=torch.tensor([0.0, 1.0]),
                    log_prob=torch.tensor(-0.3),
                    skill_idx=1,
                )
            ],
            reward=1.0,
            task_id="t1",
        ),
        DummyTraj(
            steps=[
                DummyStep(
                    x=_state("q"),
                    m_hat=torch.tensor([0.5, 1.0]),
                    m_tilde_next=torch.tensor([0.25, 0.75]),
                    log_prob=torch.tensor(-0.1),
                    skill_idx=2,
                )
            ],
            reward=0.0,
            task_id="t1",
        ),
    ]
    verified = [
        VerifiedPair(
            state_before=_state("vp"),
            action_at_t=1,
            obs_at_t="obs",
            candidates_next=[1, 2, 3],
            a_next_plus=2,
            m_t_exact=torch.tensor([1.0, 2.0]),
        )
    ]

    base = total_loss(model, trajs, [], variant="base")
    act = total_loss(model, trajs, verified, variant="act")

    assert base.ndim == 0
    assert act.ndim == 0
    assert act.item() != base.item()


def test_total_loss_rejects_unknown_variant():
    model = FakeModel()
    try:
        total_loss(model, [], [], variant="unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown variant")


def test_total_loss_adds_reference_policy_kl_when_ref_logits_are_available():
    model = FakeModel()
    trajs = [
        DummyTraj(
            steps=[
                DummyStep(
                    x=_state("q"),
                    log_prob=torch.tensor(-0.2),
                    skill_idx=1,
                    policy_logits=torch.tensor([2.0, 0.0]),
                    ref_logits=torch.tensor([0.0, 2.0]),
                )
            ],
            reward=1.0,
            task_id="t1",
        )
    ]

    without_kl = total_loss(model, trajs, [], variant="base", beta=0.0)
    with_kl = total_loss(model, trajs, [], variant="base", beta=0.5)

    assert with_kl.item() > without_kl.item()


def test_reference_kl_loss_uses_policy_to_reference_direction():
    trajs = [
        DummyTraj(
            steps=[
                DummyStep(
                    policy_logits=torch.log(torch.tensor([0.8, 0.2])),
                    ref_logits=torch.log(torch.tensor([0.5, 0.5])),
                )
            ],
            reward=1.0,
            task_id="t-kl",
        )
    ]

    expected = torch.sum(torch.tensor([0.8, 0.2]) * torch.log(torch.tensor([0.8 / 0.5, 0.2 / 0.5])))

    assert reference_kl_loss(trajs).item() == pytest.approx(expected.item())


def test_retrieval_loss_random_negatives_do_not_follow_fixed_index_order(monkeypatch):
    model = FakeModel()
    positives = [RetrievalPositive(state=_state("abc"), positive_skill_idx=1)]
    captured: list[tuple[int, ...]] = []

    original_cross_entropy = torch.nn.functional.cross_entropy

    def capture_candidates(candidates, target):
        captured.append(tuple(candidates.squeeze(0).tolist()))
        return original_cross_entropy(candidates, target)

    monkeypatch.setattr("clstr.losses.F.cross_entropy", capture_candidates)
    monkeypatch.setattr("clstr.losses.torch.randperm", lambda n, device=None: torch.tensor([2, 0, 1], device=device))

    retrieval_loss(model, positives, num_negatives=2, hard_ratio=0.0)

    h_t = model.encoder([serialize_execution_state(_state("abc"))])
    logits = model.skill_table.retrieval_logits(h_t).squeeze(0)
    fixed_order = (logits[1].item(), logits[0].item(), logits[2].item())
    random_order = (logits[1].item(), logits[3].item(), logits[0].item())
    assert captured[-1] == random_order
    assert captured[-1] != fixed_order


def test_multi_positive_routing_loss_uses_task_positive_skill_ids():
    model = FakeModel()
    skill_id_to_idx = {
        "skill/a": 0,
        "skill/b": 1,
        "skill/c": 2,
        "skill/d": 3,
    }
    tasks = [
        FakeTask(
            task_id="appworld-task-1",
            query="Instruction: find the spotify song\nRequired apps: spotify",
            positive_skill_ids=["skill/b", "skill/d"],
        )
    ]
    stats = {}

    loss = multi_positive_routing_loss(model, tasks, skill_id_to_idx, stats=stats)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert stats["routing_supervision_tasks"] == 1
    assert stats["routing_supervision_positive_labels"] == 2
    assert 0.0 <= stats["routing_supervision_recall@1"] <= 1.0
