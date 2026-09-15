from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.trajectbench_skillrouter_eval import (
    run_trajectbench_skillrouter_eval_on_rows,
    run_trajectbench_skillrouter_finetuned_eval,
)


def test_trajectbench_skillrouter_eval_applies_adapter_on_same_candidate_rows(tmp_path, monkeypatch):
    skills = [
        {"skill_id": "skill/a", "name": "Skill A", "description": "first option"},
        {"skill_id": "skill/b", "name": "Skill B", "description": "second option"},
    ]
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "benchmark": "traject_bench",
            "state_text": "flip query toward b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "candidate_next_skill_ids": ["skill/a", "skill/b"],
        }
    ]

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        encoded = []
        for text in texts:
            if "Skill B" in text:
                encoded.append(torch.tensor([0.0, 1.0]))
            elif "Skill A" in text:
                encoded.append(torch.tensor([1.0, 0.0]))
            else:
                encoded.append(torch.tensor([1.0, 0.0]))
        return torch.stack(encoded)

    monkeypatch.setattr("clstr.trajectbench_skillrouter_eval._encode_skillrouter_texts", fake_encode)
    adapter_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter_state_dict": {
                "q_proj.weight": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "d_proj.weight": torch.eye(2),
            },
            "config": {"train_doc_projection": False},
            "uses_clstr_heads": False,
            "uses_clstr_transition": False,
        },
        adapter_path,
    )

    report = run_trajectbench_skillrouter_eval_on_rows(
        rows=rows,
        skills=skills,
        output_dir=tmp_path / "out",
        model_name_or_path="fake-skillrouter",
        adapter_checkpoint_path=adapter_path,
        batch_size=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_finetuned_biencoder_adapter"
    assert report["adapter_checkpoint_path"] == str(adapter_path)
    assert report["metrics"]["next_skill_recall@1"] == pytest.approx(1.0)
    ranked = [
        json.loads(line)
        for line in (tmp_path / "out" / "trajectbench_skillrouter_ranked_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ranked[0]["candidate_next_skill_ids"] == ["skill/b", "skill/a"]


def test_trajectbench_skillrouter_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_trajectbench_skillrouter_finetuned_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "trajectbench" in result.stdout.lower()
    assert "--adapter_checkpoint_path" in result.stdout


def test_trajectbench_skillrouter_eval_can_reuse_prebuilt_stage4_rows(tmp_path, monkeypatch):
    skills_path = tmp_path / "skills.jsonl"
    rows_path = tmp_path / "stage4_rows.jsonl"
    skills = [
        {"skill_id": "skill/a", "name": "Skill A"},
        {"skill_id": "skill/b", "name": "Skill B"},
    ]
    rows = [
        {
            "trajectory_id": "traj-1",
            "step_index": 0,
            "benchmark": "traject_bench",
            "state_text": "choose b",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "candidate_next_skill_ids": ["skill/a", "skill/b"],
        }
    ]
    skills_path.write_text("".join(json.dumps(row) + "\n" for row in skills), encoding="utf-8")
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def fail_if_handoff_called(**kwargs):
        raise AssertionError("handoff builder should not run when prebuilt rows are provided")

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        encoded = []
        for text in texts:
            encoded.append(torch.tensor([0.0, 1.0]) if "Skill B" in text or "choose b" in text else torch.tensor([1.0, 0.0]))
        return torch.stack(encoded)

    monkeypatch.setattr("clstr.trajectbench_skillrouter_eval.build_logged_online_stage4_rows_with_stage0_handoff", fail_if_handoff_called)
    monkeypatch.setattr("clstr.trajectbench_skillrouter_eval._encode_skillrouter_texts", fake_encode)

    report = run_trajectbench_skillrouter_finetuned_eval(
        trajectories_path="unused.jsonl",
        skills_path=skills_path,
        output_dir=tmp_path / "out_prebuilt",
        routing_checkpoint_path="unused.pt",
        model_name_or_path="fake-skillrouter",
        adapter_checkpoint_path=None,
        prebuilt_stage4_rows_path=rows_path,
        eval_rows=10,
        eval_split_mode="sequential_tail",
    )

    assert report["status"] == "ok"
    assert report["data_report"]["candidate_source"] == "prebuilt_stage4_rows"
    assert report["metrics"]["next_skill_recall@1"] == pytest.approx(1.0)
