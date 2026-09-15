import torch

from clstr.transition_lambda_sweep import evaluate_transition_lambda_sweep


class _ToySkillTable(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _ToyTransition(torch.nn.Module):
    def forward(self, m_obs, action_input, obs_emb):
        del m_obs, action_input
        is_prior = torch.all(obs_emb == 0, dim=-1, keepdim=True).to(dtype=obs_emb.dtype)
        prior_wrong = torch.tensor([[6.0, 0.0, 0.0]], device=obs_emb.device, dtype=obs_emb.dtype)
        residual_right = torch.tensor([[0.0, 7.0, 0.0]], device=obs_emb.device, dtype=obs_emb.dtype)
        return is_prior * prior_wrong + (1.0 - is_prior) * residual_right


class _ToyPriorResidualTransitionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _ToySkillTable()
        self.transition = _ToyTransition()
        self.gate = None
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "action marker" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            elif "next observation marker" in lowered:
                rows.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.zeros(3, dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

def test_transition_lambda_sweep_reweights_same_checkpoint_without_training():
    rows = [
        {
            "benchmark": "toy",
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]
    skills = [
        {"skill_id": "skill/current"},
        {"skill_id": "skill/next"},
        {"skill_id": "skill/other"},
    ]

    report = evaluate_transition_lambda_sweep(
        model=_ToyPriorResidualTransitionModel(),
        rows=rows,
        skills=skills,
        lambdas=[0.0, 1.0],
        batch_size=1,
        max_eval_batches=1,
        sampling_strategy="balanced_deterministic",
        transition_inventory_mask_mode="off",
        transition_loss_type="cross_entropy",
        transition_positive_mode="single",
        device=torch.device("cpu"),
    )

    by_lambda = {row["lambda"]: row for row in report["lambda_reports"]}
    assert by_lambda[0.0]["transition_skill_recall@1"] == 0.0
    assert by_lambda[1.0]["transition_skill_recall@1"] == 1.0
    assert by_lambda[0.0]["transition_prior_skill_recall@1"] == by_lambda[1.0]["transition_prior_skill_recall@1"] == 0.0
    assert by_lambda[0.0]["transition_residual_skill_recall@1"] == by_lambda[1.0]["transition_residual_skill_recall@1"] == 1.0
    assert report["best_lambda_by_transition_recall@1"] == 1.0
