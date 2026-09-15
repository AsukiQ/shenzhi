from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.stage0_handoff_audit import audit_stage0_handoff_coverage


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class _AuditSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _AuditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _AuditSkillTable()
        self.state_inputs = []

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if lowered.startswith("instruct:") and "previous_action:" in lowered and "next-b" in lowered:
                rows.append(torch.tensor([0.0, 8.0, 1.0]))
            elif lowered.startswith("instruct:") and "state-a" in lowered:
                rows.append(torch.tensor([9.0, 1.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 1.0, 9.0]))
        return torch.stack(rows).to(self.device) + self.anchor * 0.0

    def encode_states(self, texts):
        self.state_inputs.extend(str(text) for text in texts)
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "previous_action:" in lowered and "next-b" in lowered:
                rows.append(torch.tensor([0.0, 8.0, 1.0]))
            elif "state-a" in lowered:
                rows.append(torch.tensor([9.0, 1.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 1.0, 9.0]))
        return torch.stack(rows).to(self.device) + self.anchor * 0.0

    def initial_belief(self, h, top_k=None):
        del top_k
        return torch.zeros_like(h)

    def unified_route_full_logits(self, h, m):
        del m
        return h


class _HOnlyForbiddenAuditSkillTable(_AuditSkillTable):
    def logits(self, h):
        del h
        raise AssertionError("handoff audit must use the unified static scorer")


class _UnifiedAuditModel(_AuditModel):
    def __init__(self):
        super().__init__()
        self.skill_table = _HOnlyForbiddenAuditSkillTable()
        self.initial_belief_calls = 0
        self.unified_route_calls = 0

    def initial_belief(self, h, top_k=None):
        del top_k
        self.initial_belief_calls += 1
        return torch.zeros_like(h)

    def unified_route_full_logits(self, h, m):
        del m
        self.unified_route_calls += 1
        return h


def test_stage0_handoff_audit_reports_current_and_next_coverage_by_query_mode(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    output_path = tmp_path / "audit.json"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "covered",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        ],
    )

    model = _AuditModel()
    report = audit_stage0_handoff_coverage(
        model=model,
        train_path=train_path,
        skills_path=skills_path,
        output_path=output_path,
        top_k_values=(1, 2),
        query_modes=("raw_state", "skillrouter_state", "checkpoint_state_query"),
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert report["status"] == "ok"
    assert report["row_count"] == 1
    raw = report["query_modes"]["raw_state"]["global"]
    wrapped = report["query_modes"]["skillrouter_state"]["global"]
    checkpoint = report["query_modes"]["checkpoint_state_query"]["global"]
    assert raw["current_recall@1"] == 0.0
    assert raw["next_recall@1"] == 0.0
    assert wrapped["current_recall@1"] == 1.0
    assert wrapped["next_recall@1"] == 1.0
    assert checkpoint["current_recall@1"] == 1.0
    assert checkpoint["next_recall@1"] == 1.0
    assert model.state_inputs == [
        "state-a",
        "state-a\nprevious_action: do\nnext_observation: next-b",
    ]
    assert report["query_modes"]["skillrouter_state"]["benchmarks"]["traject_bench"]["current_recall@1"] == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_stage0_handoff_audit_uses_same_unified_static_scorer_as_candidate_handoff(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "unified-static",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        ],
    )
    model = _UnifiedAuditModel()

    report = audit_stage0_handoff_coverage(
        model=model,
        train_path=train_path,
        skills_path=skills_path,
        top_k_values=(1,),
        query_modes=("checkpoint_state_query",),
        batch_size=1,
        device=torch.device("cpu"),
    )

    metrics = report["query_modes"]["checkpoint_state_query"]["global"]
    assert metrics["current_recall@1"] == 1.0
    assert metrics["next_recall@1"] == 1.0
    assert model.initial_belief_calls == 2
    assert model.unified_route_calls == 2


def test_stage0_handoff_audit_max_rows_samples_across_benchmarks(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
            {"skill_id": "skill/c", "name": "c"},
        ],
    )
    rows = []
    for idx in range(6):
        rows.append(
            {
                "benchmark": "alfworld",
                "task_id": f"alf-{idx}",
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        )
    for benchmark in ("toolbench_g3", "traject_bench"):
        rows.append(
            {
                "benchmark": benchmark,
                "task_id": benchmark,
                "state_text": "state-a",
                "action_text": "do",
                "next_observation_text": "next-b",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                "provenance": {"split": "train"},
            }
        )
    write_jsonl(train_path, rows)

    report = audit_stage0_handoff_coverage(
        model=_AuditModel(),
        train_path=train_path,
        skills_path=skills_path,
        top_k_values=(1,),
        query_modes=("skillrouter_state",),
        max_rows=3,
        batch_size=1,
        device=torch.device("cpu"),
    )

    benchmarks = report["query_modes"]["skillrouter_state"]["benchmarks"]
    assert report["row_count"] == 3
    assert set(benchmarks) == {"alfworld", "toolbench_g3", "traject_bench"}


def test_stage0_handoff_audit_cli_exposes_query_modes():
    proc = subprocess.run(
        [sys.executable, "scripts/audit_clstr_stage0_handoff_coverage.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--checkpoint_path" in proc.stdout
    assert "--query_modes" in proc.stdout
    assert "--top_k_values" in proc.stdout
