from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clstr.skillrouter_style import (
    _query_text_for_mode,
    _resolve_hf_pooling,
    _resolve_query_text_mode,
    _resolve_tokenizer_padding_side,
)
from clstr.unified_skillrouter_finetune import (
    _write_training_progress,
    load_unified_skillrouter_training_corpus,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_unified_skillrouter_loader_preserves_required_positive_when_skill_pool_is_capped(tmp_path):
    data_root = tmp_path / "unified"
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "skill/a", "name": "A", "description": "first", "body": "alpha"},
            {"skill_id": "skill/b", "name": "B", "description": "second", "body": "beta"},
            {"skill_id": "skill/c", "name": "C", "description": "third", "body": "gamma"},
        ],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "task_id": "t0",
                "trajectory_id": "traj0",
                "state_text": "need c",
                "next_skill_id": "skill/c",
                "benchmark": "toy",
                "loss_mask": {"routing": True},
            },
            {
                "task_id": "t1",
                "trajectory_id": "traj1",
                "state_text": "missing positive",
                "next_skill_id": "skill/missing",
                "benchmark": "toy",
                "loss_mask": {"routing": True},
            },
        ],
    )

    corpus = load_unified_skillrouter_training_corpus(
        data_root,
        max_rows=2,
        max_skills=2,
        eval_rows=0,
        seed=7,
    )

    assert corpus.report["usable_query_count"] == 1
    assert corpus.report["skipped_reasons"]["positive_missing_from_skill_pool"] == 1
    assert "skill/c" in {row["skill_id"] for row in corpus.skills}
    assert corpus.train_queries[0].positive_indices


def test_unified_skillrouter_loader_streams_trajectories_when_max_rows_is_set(tmp_path):
    data_root = tmp_path / "unified"
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [{"skill_id": "skill/a", "name": "A", "description": "first", "body": "alpha"}],
    )
    (data_root / "trajectories.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t0",
                "state_text": "need a",
                "next_skill_id": "skill/a",
                "benchmark": "toy",
                "loss_mask": {"routing": True},
            }
        )
        + "\n"
        + "{this line is intentionally invalid and should not be read}\n",
        encoding="utf-8",
    )

    corpus = load_unified_skillrouter_training_corpus(
        data_root,
        max_rows=1,
        eval_rows=0,
        seed=7,
    )

    assert corpus.report["source_rows_seen"] == 1
    assert corpus.report["usable_query_count"] == 1


def test_unified_skillrouter_finetune_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_unified_skillrouter_finetune.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--data_root" in result.stdout
    assert "--max_steps" in result.stdout
    assert "--train_doc_projection" in result.stdout
    assert "--pooling" in result.stdout
    assert "--query_text_mode" in result.stdout
    assert "--checkpoint_every" in result.stdout


def test_bge_auto_skillrouter_training_protocol_uses_cls_raw_and_right_padding(tmp_path):
    model_dir = tmp_path / "bge-m3"
    pooling_dir = model_dir / "1_Pooling"
    pooling_dir.mkdir(parents=True)
    (pooling_dir / "config.json").write_text(
        '{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}\n',
        encoding="utf-8",
    )

    pooling = _resolve_hf_pooling(str(model_dir), "auto")

    assert pooling == "cls"
    assert _resolve_tokenizer_padding_side(str(model_dir), "auto", pooling) == "right"
    assert _resolve_query_text_mode(str(model_dir), "auto", pooling) == "raw"
    assert _query_text_for_mode({"query": "raw task state"}, "raw") == "raw task state"


def test_non_bge_auto_skillrouter_training_protocol_keeps_skillrouter_prompt(tmp_path):
    model_dir = tmp_path / "SkillRouter-Embedding-0.6B"
    model_dir.mkdir()

    pooling = _resolve_hf_pooling(str(model_dir), "auto")
    query_text = _query_text_for_mode({"query": "pick next tool"}, "skillrouter")

    assert pooling == "last_token"
    assert _resolve_tokenizer_padding_side(str(model_dir), "auto", pooling) == "left"
    assert _resolve_query_text_mode(str(model_dir), "auto", pooling) == "skillrouter"
    assert query_text.startswith("Instruct: Given a task description")
    assert "pick next tool" in query_text


def test_write_training_progress_overwrites_progress_json(tmp_path):
    progress = _write_training_progress(
        tmp_path,
        stage="training",
        step=3,
        max_steps=10,
        extra={"loss": 1.25},
    )

    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "training"
    assert saved["step"] == 3
    assert saved["max_steps"] == 10
    assert saved["loss"] == 1.25
    assert progress == saved
