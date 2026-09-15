import torch
from torch import nn

from clstr.appworld_mt_fusion_train import iter_trajectory_batches, multistep_oracle_policy_loss


class _TinySkillTable:
    def __init__(self):
        self.E = torch.eye(2)

    def logits(self, h):
        return torch.tensor([[2.0, 1.0]])


class _TinyPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.skill_table = _TinySkillTable()
        self.K = 2
        self.bias = nn.Parameter(torch.zeros(2))

    @property
    def device(self):
        return torch.device("cpu")

    def encode_states(self, states):
        return torch.tensor([[1.0, 0.0]])

    def batch_cross_encode(self, states, candidate_rows):
        embs = self.skill_table.E.index_select(0, torch.tensor(candidate_rows[0], dtype=torch.long))
        return embs.unsqueeze(0)

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        skill_logits = candidate_embs @ self.bias
        stop = torch.full((1, 1), -100.0)
        return torch.cat([skill_logits.squeeze(0).unsqueeze(0), stop], dim=-1)

    def encode_observations(self, observations):
        return torch.tensor([[0.0, 1.0]])

    def step_update(self, m_t, a_t, o_t_emb, x_next_text):
        return m_t, m_t, m_t


def test_multistep_oracle_policy_loss_uses_step_positive_skill_ids():
    model = _TinyPolicyModel()
    trajectories = [
        {
            "task_id": "task_1",
            "steps": [
                {
                    "state_text": "state",
                    "positive_skill_ids": ["skill-b"],
                    "selected_skill_ids": ["skill-b"],
                    "execute_output": "Execution successful.",
                }
            ],
        }
    ]

    loss, stats = multistep_oracle_policy_loss(
        model=model,
        trajectories=trajectories,
        skill_ids=["skill-a", "skill-b"],
        candidate_top_k=2,
    )

    assert loss.item() > 0
    assert stats["trajectory_count"] == 1
    assert stats["supervised_steps"] == 1
    assert stats["positive_in_candidates"] == 1
    assert stats["skipped_without_positive"] == 0


def test_iter_trajectory_batches_chunks_without_dropping_rows():
    trajectories = [{"task_id": str(idx)} for idx in range(5)]

    batches = list(iter_trajectory_batches(trajectories, batch_size=2))

    assert [[row["task_id"] for row in batch] for batch in batches] == [["0", "1"], ["2", "3"], ["4"]]
