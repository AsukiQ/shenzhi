from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import torch

from clstr.stage2_transition_row_diagnostics import (
    _markdown_summary,
    build_transition_row_records,
    compute_transition_row_diagnostics_batch,
    rank_label,
    summarize_transition_row_records,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_cli_module():
    path = Path("scripts/run_clstr_stage2_transition_row_diagnostics.py")
    spec = importlib.util.spec_from_file_location("run_clstr_stage2_transition_row_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rank_label_is_one_based_with_descending_logits():
    logits = torch.tensor([0.2, 0.9, -1.0, 0.4])

    assert rank_label(logits, 1) == 1
    assert rank_label(logits, 3) == 2
    assert rank_label(logits, 2) == 4


def test_build_transition_row_records_keeps_stage0_and_stage2_rank():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "source_quality": "official_replay",
            "task_id": "task-a",
            "trajectory_id": "traj-a",
            "step_index": 3,
            "skill_id": "skill/current",
            "next_skill_id": "skill/gold",
            "history_text": "a\nb\nc",
            "stage0_next_candidate_skill_ids": ["skill/wrong", "skill/gold", "skill/other"],
        }
    ]
    candidate_indices = [[5, 7, 9]]
    skill_ids = {5: "skill/wrong", 7: "skill/gold", 9: "skill/other"}
    logits = torch.tensor([[0.7, 0.2, 0.9]])
    labels = torch.tensor([1])

    records = build_transition_row_records(
        logits=logits,
        labels=labels,
        rows=rows,
        candidate_indices=candidate_indices,
        skill_ids=skill_ids,
        top_k=2,
        row_offset=10,
    )

    assert records[0]["row_index"] == 10
    assert records[0]["stage0_gold_rank"] == 2
    assert records[0]["stage2_gold_rank"] == 3
    assert records[0]["stage2_hit@5"] is True
    assert records[0]["top_skill_ids"] == ["skill/other", "skill/wrong"]
    assert records[0]["failure_type"] == "hit_top5_not_top1"
    assert records[0]["history_line_count"] == 3


def test_summarize_transition_row_records_breaks_down_benchmark_and_failures(tmp_path):
    rows = [
        {
            "benchmark": "a",
            "source_quality": "x",
            "stage0_gold_rank": 1,
            "stage2_gold_rank": 1,
            "stage2_hit@1": True,
            "stage2_hit@5": True,
            "stage2_mrr": 1.0,
            "stage2_cross_entropy": 0.1,
            "failure_type": "hit_top1",
        },
        {
            "benchmark": "a",
            "source_quality": "x",
            "stage0_gold_rank": 2,
            "stage2_gold_rank": 6,
            "stage2_hit@1": False,
            "stage2_hit@5": False,
            "stage2_mrr": 1 / 6,
            "stage2_cross_entropy": 2.0,
            "failure_type": "miss_top5_stage0_top5",
        },
        {
            "benchmark": "b",
            "source_quality": "y",
            "stage0_gold_rank": 10,
            "stage2_gold_rank": 4,
            "stage2_hit@1": False,
            "stage2_hit@5": True,
            "stage2_mrr": 0.25,
            "stage2_cross_entropy": 1.0,
            "failure_type": "hit_top5_not_top1",
        },
    ]
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, rows)

    summary = summarize_transition_row_records(path)

    assert summary["row_count"] == 3
    assert summary["overall"]["recall@1"] == 1 / 3
    assert summary["overall"]["recall@5"] == 2 / 3
    assert summary["overall"]["stage0_recall@5"] == 2 / 3
    assert summary["overall"]["stage0_recall@20"] == 1.0
    assert summary["overall"]["mean_stage0_gold_rank"] == 13 / 3
    assert summary["failure_type_counts"]["miss_top5_stage0_top5"] == 1
    assert summary["by_benchmark"]["a"]["row_count"] == 2
    assert summary["by_source_quality"]["y"]["recall@5"] == 1.0


def test_cli_benchmark_caps_accepts_existing_equals_sbatch_format():
    module = _load_cli_module()

    caps = module._benchmark_caps("toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000")

    assert caps == {
        "toolbench_g3": -1,
        "traject_bench": -1,
        "alfworld": 10000,
        "scienceworld": 10000,
    }


def test_cli_forwards_transition_scoring_mode(monkeypatch, tmp_path):
    module = _load_cli_module()
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(module, "run_stage2_transition_row_diagnostics", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_clstr_stage2_transition_row_diagnostics.py",
            "--stage0_checkpoint_path",
            str(tmp_path / "stage0.pt"),
            "--stage2_checkpoint_path",
            str(tmp_path / "stage2.pt"),
            "--train_path",
            str(tmp_path / "train.jsonl"),
            "--skills_path",
            str(tmp_path / "skills.jsonl"),
            "--output_dir",
            str(tmp_path / "out"),
            "--transition_scoring_mode",
            "stage0_rank_prior_plus_transition_residual",
            "--transition_residual_lambda",
            "0.125",
        ],
    )

    assert module.main() == 0
    assert captured["transition_scoring_mode"] == "stage0_rank_prior_plus_transition_residual"
    assert captured["transition_residual_lambda"] == 0.125


def test_markdown_summary_includes_stage0_and_benchmark_breakdown():
    report = {
        "status": "ok",
        "summary": {
            "overall": {
                "row_count": 10,
                "recall@1": 0.1,
                "recall@5": 0.2,
                "recall@10": 0.3,
                "recall@20": 0.4,
                "stage0_recall@5": 0.25,
                "stage0_recall@10": 0.35,
                "stage0_recall@20": 0.45,
                "mrr": 0.15,
                "mean_stage0_gold_rank": 12.0,
                "mean_stage2_gold_rank": 9.0,
                "stage2_improved_vs_stage0_fraction": 0.6,
                "stage2_worse_than_stage0_fraction": 0.2,
            },
            "failure_type_counts": {"hit_top1": 1},
            "by_benchmark": {
                "toolbench_g3": {
                    "row_count": 4,
                    "stage0_recall@5": 0.5,
                    "stage0_recall@20": 0.75,
                    "recall@5": 0.25,
                    "recall@20": 0.5,
                    "mean_stage0_gold_rank": 20.0,
                    "mean_stage2_gold_rank": 30.0,
                    "stage2_worse_than_stage0_fraction": 0.8,
                }
            },
        },
    }

    markdown = _markdown_summary(report)

    assert "Stage0 recall@5" in markdown
    assert "Stage0 recall@20" in markdown
    assert "## By Benchmark" in markdown
    assert "toolbench_g3" in markdown
    assert "0.8" in markdown


class _DiagSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _DiagTransitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _DiagSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None
        self.initial_belief_calls = 0
        self.transition_memory_inputs = []

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            if "next observation" in str(text):
                rows.append(torch.tensor([8.0, 0.0, 0.0], dtype=torch.float32))
            else:
                rows.append(torch.zeros(3, dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def initial_belief(self, h, top_k=None):
        del top_k
        self.initial_belief_calls += 1
        return torch.tensor([[7.0, 9.0, 11.0]], dtype=h.dtype, device=h.device).expand(h.size(0), -1)

    def __call__(self, m_obs, action_input, obs_emb):
        self.transition_memory_inputs.append(m_obs.detach().clone())
        del action_input
        return obs_emb


def test_transition_row_diagnostics_use_stage0_rank_prior_scoring_without_native_trans_head():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "state_text": "current state",
            "action_text": "call tool",
            "next_observation_text": "next observation",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [2, 1, 0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    model = _DiagTransitionModel()
    records = compute_transition_row_diagnostics_batch(
        model=model,
        batch=rows,
        skill_id_to_idx={"skill/current": 0, "skill/wrong": 1, "skill/next": 2},
        skill_ids_by_idx={0: "skill/current", 1: "skill/wrong", 2: "skill/next"},
        equivalent_skill_ids_by_skill_id={},
        device=torch.device("cpu"),
        top_k=3,
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
    )

    assert records[0]["stage0_gold_rank"] == 1
    assert records[0]["stage2_gold_rank"] == 1
    assert records[0]["transition_skill_head_type"].startswith("stage0_rank_prior+")
    assert model.initial_belief_calls == 1
    assert any(torch.equal(item, torch.tensor([[7.0, 9.0, 11.0]])) for item in model.transition_memory_inputs)
