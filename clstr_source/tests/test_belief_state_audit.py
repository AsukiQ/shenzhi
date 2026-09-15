from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from clstr.belief_state_audit import audit_belief_tensors, write_belief_audit_report


class TinySkillTable:
    def __init__(self, *, scale: float, skill_count: int = 8, bias: torch.Tensor | None = None):
        self.E = torch.eye(skill_count)
        self.belief_top_k = None
        self.logit_scale_belief = torch.nn.Parameter(torch.tensor([math.log(scale)]))
        self.logit_scale_retr = torch.nn.Parameter(torch.tensor([math.log(12.0)]))
        self.skill_bias_belief = torch.nn.Parameter(torch.zeros(skill_count) if bias is None else bias.clone().float())
        retrieval_bias = torch.zeros(skill_count)
        retrieval_bias[0] = 4.0
        self.skill_bias_retr = torch.nn.Parameter(retrieval_bias)

    def _scores(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.E.t()

    def belief_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.logit_scale_belief.exp() * self._scores(h) + self.skill_bias_belief.unsqueeze(0)

    def retrieval_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.logit_scale_retr.exp() * self._scores(h) + self.skill_bias_retr.unsqueeze(0)


def test_audit_flags_low_temperature_belief_as_near_constant():
    h = torch.eye(8)
    report = audit_belief_tensors(
        skill_table=TinySkillTable(scale=0.2),
        state_embeddings=h,
        benchmark_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
        max_effective_support_fraction=0.75,
        min_top5_mass=0.8,
        max_pairwise_cosine=0.98,
    )

    assert report["status"] == "action_required"
    assert "belief_effective_support_too_large" in report["blockers"]
    assert "belief_top5_mass_too_low" in report["blockers"]
    assert report["belief"]["effective_support_fraction_mean"] > 0.75
    assert report["belief"]["top5_mass_mean"] < 0.8
    assert report["metadata"]["skill_count"] == 8
    assert report["by_benchmark"]["a"]["row_count"] == 4


def test_audit_accepts_sharp_belief_with_nonconstant_memory():
    h = torch.eye(8)
    report = audit_belief_tensors(
        skill_table=TinySkillTable(scale=20.0, bias=torch.arange(8).float()),
        state_embeddings=h,
        max_effective_support_fraction=0.75,
        min_top5_mass=0.8,
        max_pairwise_cosine=0.98,
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["belief"]["effective_support_fraction_mean"] < 0.75
    assert report["belief"]["pairwise_m_cosine_mean"] < 0.98
    assert report["retrieval"]["top1_mass_mean"] > 0.9


def test_audit_uses_sparse_belief_topk_when_available():
    table = TinySkillTable(scale=0.2, skill_count=128)
    table.belief_top_k = 4
    h = torch.eye(128)[:8]

    report = audit_belief_tensors(
        skill_table=table,
        state_embeddings=h,
        max_effective_support_fraction=0.10,
        min_top5_mass=0.8,
        max_pairwise_cosine=1.0,
    )

    assert report["belief"]["probability_support"] == "topk"
    assert report["belief"]["probability_support_size"] == 4
    assert report["belief"]["top5_mass_mean"] > 0.999
    assert report["belief"]["effective_support_fraction_mean"] < 0.10


def test_write_belief_audit_report_roundtrips(tmp_path):
    report = audit_belief_tensors(
        skill_table=TinySkillTable(scale=0.2),
        state_embeddings=torch.eye(8),
    )

    output_path = write_belief_audit_report(tmp_path / "belief" / "report.json", report)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["status"] == report["status"]
    assert loaded["metadata"]["row_count"] == 8


def test_cli_help_loads():
    path = Path("scripts/audit_clstr_belief_state.py")
    spec = importlib.util.spec_from_file_location("audit_clstr_belief_state", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    old_argv = sys.argv
    try:
        sys.argv = ["audit_clstr_belief_state.py", "--help"]
        try:
            module.main()
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = old_argv
