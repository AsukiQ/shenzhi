from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from clstr.prior_gate_sweep import (
    gate_lambdas,
    run_prior_gate_replay_sweep,
    sweep_prior_residual_logits,
)


def _load_cli_module():
    path = Path("scripts/sweep_clstr_prior_gate.py")
    spec = importlib.util.spec_from_file_location("sweep_clstr_prior_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_lambda_zero_preserves_prior_ranking():
    prior = torch.tensor([[0.0, 2.0, 1.0]])
    residual = torch.tensor([[5.0, 0.0, 0.0]])
    labels = torch.tensor([1])

    report = sweep_prior_residual_logits(prior, residual, labels, formulas=["fixed_0_00", "fixed_1_00"])

    assert report["formulas"]["fixed_0_00"]["recall@1"] == 1.0
    assert report["formulas"]["fixed_0_00"]["delta_vs_prior_mrr"] == 0.0
    assert report["formulas"]["fixed_1_00"]["recall@1"] == 0.0
    assert report["formulas"]["fixed_1_00"]["worse_than_prior_fraction"] == 1.0


def test_fixed_lambda_can_improve_prior_when_residual_is_strong_enough():
    prior = torch.tensor([[2.0, 1.0, 0.0]])
    residual = torch.tensor([[0.0, 3.0, 0.0]])
    labels = torch.tensor([1])

    report = sweep_prior_residual_logits(prior, residual, labels, formulas=["fixed_0_25", "fixed_1_00"])

    assert report["formulas"]["fixed_0_25"]["recall@1"] == 0.0
    assert report["formulas"]["fixed_1_00"]["recall@1"] == 1.0
    assert report["formulas"]["fixed_1_00"]["improved_vs_prior_fraction"] == 1.0


def test_margin_gate_lowers_lambda_for_confident_prior():
    prior = torch.tensor([[5.0, 1.0, 0.0], [1.0, 0.9, 0.0]])

    lambdas = gate_lambdas(
        "margin_gate",
        prior,
        lambda_min=0.05,
        lambda_max=0.5,
        margin_threshold=1.0,
    )

    assert torch.allclose(lambdas, torch.tensor([0.05, 0.5]))


def test_entropy_gate_lowers_lambda_for_low_entropy_prior():
    prior = torch.tensor([[8.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    lambdas = gate_lambdas(
        "entropy_gate",
        prior,
        lambda_min=0.05,
        lambda_max=0.5,
        entropy_low=0.1,
        entropy_high=0.9,
    )

    assert lambdas[0].item() < lambdas[1].item()
    assert torch.all(lambdas >= 0.05)
    assert torch.all(lambdas <= 0.5)


def test_cli_forwards_replay_sweep_arguments(monkeypatch, tmp_path):
    module = _load_cli_module()
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "formulas": {}}

    monkeypatch.setattr(module, "run_prior_gate_replay_sweep", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_clstr_prior_gate.py",
            "--stage0_checkpoint_path",
            str(tmp_path / "stage0.pt"),
            "--stage_checkpoint_path",
            str(tmp_path / "stage1.pt"),
            "--train_path",
            str(tmp_path / "train.jsonl"),
            "--skills_path",
            str(tmp_path / "skills.jsonl"),
            "--output_dir",
            str(tmp_path / "out"),
            "--top_m",
            "500",
            "--formulas",
            "fixed_0_00,fixed_0_25,margin_gate",
            "--max_rows",
            "123",
        ],
    )

    assert module.main() == 0

    assert captured["top_m"] == 500
    assert captured["max_rows"] == 123
    assert captured["formulas"] == ["fixed_0_00", "fixed_0_25", "margin_gate"]
    report = json.loads((tmp_path / "out" / "gate_sweep_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"


def test_prior_gate_replay_sweep_initializes_with_model_initial_belief(monkeypatch, tmp_path):
    import clstr.full_base_train as full_base_train
    import clstr.stage_checkpoint_init as stage_checkpoint_init

    captured = {}

    class _SkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.eye(2), requires_grad=False)

        def logits(self, h):
            return h

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.skill_table = _SkillTable()
            self.initial_belief_calls = 0

        @property
        def device(self):
            return self.anchor.device

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                rows.append(
                    torch.tensor([0.0, 1.0], dtype=torch.float32)
                    if "next" in str(text)
                    else torch.tensor([1.0, 0.0], dtype=torch.float32)
                )
            return torch.stack(rows) + self.anchor * 0.0

        def initial_belief(self, h, top_k=None):
            del top_k
            self.initial_belief_calls += 1
            return torch.tensor([[7.0, 9.0]], dtype=h.dtype, device=h.device).expand(h.size(0), -1)

    model = _Model()

    def fake_build_model(**kwargs):
        del kwargs
        return model, {}, {"status": "ok"}

    def fake_attach_candidates(model_arg, rows, *args, **kwargs):
        del model_arg, args, kwargs
        return rows, {"status": "ok"}

    def fake_transition_logits(model_arg, h, m_obs, *args, **kwargs):
        del model_arg, h, args
        captured["m_obs"] = m_obs.detach().clone()
        candidate_ids = kwargs["candidate_ids"]
        logits = torch.tensor([[0.0, 1.0]], dtype=m_obs.dtype, device=m_obs.device).expand(
            candidate_ids.size(0), -1
        )
        return logits, "recording_transition", logits, torch.zeros_like(logits)

    monkeypatch.setattr(stage_checkpoint_init, "build_clstr_model_from_stage0_checkpoint", fake_build_model)
    monkeypatch.setattr(
        stage_checkpoint_init,
        "load_head_checkpoint_into_model",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(full_base_train, "_attach_stage0_topm_candidates", fake_attach_candidates)
    monkeypatch.setattr(full_base_train, "_apply_skill_text_format", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(full_base_train, "_transition_candidate_logits_for_mode", fake_transition_logits)

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "split": "train",
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "stage0_next_candidate_skill_indices": [0, 1],
                "loss_mask": {"L_trans_skill_ce": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    skills_path = tmp_path / "skills.jsonl"
    skills_path.write_text(
        "".join(json.dumps({"skill_id": skill_id}) + "\n" for skill_id in ("skill/a", "skill/b")),
        encoding="utf-8",
    )

    report = run_prior_gate_replay_sweep(
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        stage_checkpoint_path=tmp_path / "stage1.pt",
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "output",
        batch_size=1,
        transition_inventory_mask_mode="off",
        formulas=["fixed_0_00"],
    )

    assert report["status"] == "ok"
    assert model.initial_belief_calls == 1
    assert torch.equal(captured["m_obs"], torch.tensor([[7.0, 9.0]]))
