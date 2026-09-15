from __future__ import annotations

import json
from pathlib import Path

import torch

from clstr.logged_online_stage4_diagnostics import run_logged_online_stage4_candidate_diagnostics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _RoutingSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3), requires_grad=False)

    def retrieval_logits(self, h):
        return h @ self.E.t()


class _TinyStage0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skill_table = _RoutingSkillTable()

    @property
    def device(self):
        return torch.device("cpu")

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "next-b" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0]))
            elif "next-c" in lowered:
                rows.append(torch.tensor([0.0, 0.0, 1.0]))
            elif "state-a" in lowered:
                rows.append(torch.tensor([1.0, 0.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 0.0, 1.0]))
        return torch.stack(rows)


def _trajectory_row(task_id: str, next_observation_text: str) -> dict:
    return {
        "benchmark": "toolbench_g3",
        "trajectory_id": task_id,
        "task_id": task_id,
        "step_index": 0,
        "goal_text": "Use the matching tool.",
        "state_text": "state-a",
        "action_text": "call a",
        "next_observation_text": next_observation_text,
        "skill_id": "tool/a",
        "next_skill_id": "tool/b",
        "loss_mask": {"routing": True, "L_trans_skill_ce": True},
        "provenance": {"split": "train"},
    }


def _trajectory_pair(task_id: str, next_observation_text: str) -> list[dict]:
    return [
        _trajectory_row(task_id, next_observation_text),
        {
            "benchmark": "toolbench_g3",
            "trajectory_id": task_id,
            "task_id": f"{task_id}::1",
            "step_index": 1,
            "state_text": next_observation_text,
            "skill_id": "tool/b",
            "done": True,
            "provenance": {"split": "train"},
        },
    ]


def test_logged_online_stage4_candidate_diagnostics_reports_missing_and_rank(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skills = tmp_path / "skills.jsonl"
    output_dir = tmp_path / "diag"
    _write_jsonl(
        trajectories,
        [
            row
            for task_id, next_observation in (("hit", "next-b"), ("miss", "next-c"))
            for row in _trajectory_pair(task_id, next_observation)
        ],
    )
    _write_jsonl(
        skills,
        [
            {"skill_id": "tool/a", "name": "a"},
            {"skill_id": "tool/b", "name": "b", "alias_skill_ids": ["tool/b_alias"]},
            {"skill_id": "tool/c", "name": "c", "canonical_skill_id": "tool/b"},
        ],
    )

    report = run_logged_online_stage4_candidate_diagnostics(
        model=_TinyStage0Model(),
        trajectories_path=trajectories,
        skills_path=skills,
        output_dir=output_dir,
        stage0_top_m=1,
        allowed_benchmarks={"toolbench_g3"},
        stage0_handoff_query_mode="raw_state",
        stage0_candidate_encode_batch_size=1,
        routing_checkpoint_path="dummy.pt",
    )

    assert report["status"] == "ok"
    assert report["causal_next_state"]["attached_rows"] == 2
    assert report["summary"]["overall"]["row_count"] == 2
    assert report["summary"]["overall"]["candidate_positive_coverage"] == 0.5
    assert report["summary"]["overall"]["candidate_recall@1"] == 0.5
    assert report["summary"]["missing_positive_count"] == 1
    assert report["summary"]["equivalent_candidate_hit_count"] == 1
    records = [json.loads(line) for line in (output_dir / "stage4_candidate_diagnostics.jsonl").read_text().splitlines()]
    assert records[0]["candidate_positive_rank"] == 1
    assert records[0]["stage0_topm_next_positive_hit"] is True
    assert records[1]["candidate_positive_rank"] is None
    assert records[1]["stage0_next_positive_missing"] is True
    assert records[1]["equivalent_candidate_skill_ids"] == ["tool/c"]
    assert "next-c" in records[1]["next_observation_text"]


def test_logged_online_stage4_candidate_diagnostics_cli_exposes_toolbench_defaults():
    script = Path("scripts/audit_logged_online_stage4_candidates.py")
    text = script.read_text(encoding="utf-8")

    assert "run_logged_online_stage4_candidate_diagnostics_from_checkpoint" in text
    assert "data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl" in text
    assert "--allowed_benchmarks" in text
    assert "toolbench_g3" in text
