import json
from pathlib import Path

import pytest
import torch

from clstr.stage_preflight import validate_stage2_routing_checkpoint


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage2_preflight_accepts_matching_stage1_routing_checkpoint(tmp_path):
    skills_path = tmp_path / "skill_pool.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "A"},
            {"skill_id": "skill/b", "name": "B"},
        ],
    )
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "train_safety": {
                "unified_v2_train_safe_retrieval": True,
                "excluded_retrieval_splits": ["public_train_or_eval_unlabeled"],
            },
            "model_state_dict": {
                "skill_table.E": torch.zeros(2, 8),
                "skill_table.W.weight": torch.eye(8),
                "skill_table.logit_scale_retr": torch.zeros(1),
                "skill_table.skill_bias_retr": torch.zeros(2),
            },
        },
        checkpoint_path,
    )

    report = validate_stage2_routing_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_dim=8,
    )

    assert report["status"] == "ok"
    assert report["checkpoint_stage"] == "clstr_unified_retrieval_v2"
    assert report["train_safety"]["unified_v2_train_safe_retrieval"] is True
    assert report["skill_count"] == 2
    assert report["embedding_shape"] == [2, 8]
    assert report["model_dim"] == 8
    assert report["has_retrieval_adapter"] is True
    assert report["has_retrieval_scale_and_bias"] is True


def test_stage2_preflight_rejects_skill_pool_checkpoint_mismatch(tmp_path):
    skills_path = tmp_path / "skill_pool.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "A"},
            {"skill_id": "skill/b", "name": "B"},
        ],
    )
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {"skill_table.E": torch.zeros(1, 8)},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="skill count mismatch"):
        validate_stage2_routing_checkpoint(
            checkpoint_path=checkpoint_path,
            skills_path=skills_path,
            model_dim=8,
        )


def test_stage2_preflight_rejects_legacy_unified_checkpoint_without_train_safety(tmp_path):
    skills_path = tmp_path / "skill_pool.jsonl"
    checkpoint_path = tmp_path / "legacy_stage1.pt"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "A"},
            {"skill_id": "skill/b", "name": "B"},
        ],
    )
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "skill_table.E": torch.zeros(2, 8),
                "skill_table.W.weight": torch.eye(8),
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="train-safe retrieval metadata"):
        validate_stage2_routing_checkpoint(
            checkpoint_path=checkpoint_path,
            skills_path=skills_path,
            model_dim=8,
        )


def test_stage2_preflight_rejects_skill_pool_placeholder_records(tmp_path):
    skills_path = tmp_path / "skill_pool.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "A", "description": "real skill"},
            {
                "skill_id": "traject/query-only",
                "name": "traject/query-only",
                "description": "traject/query-only",
                "provenance": {"missing_source_record": True},
            },
        ],
    )
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "train_safety": {
                "unified_v2_train_safe_retrieval": True,
                "excluded_retrieval_splits": ["public_train_or_eval_unlabeled"],
            },
            "model_state_dict": {
                "skill_table.E": torch.zeros(2, 8),
                "skill_table.W.weight": torch.eye(8),
                "skill_table.logit_scale_retr": torch.zeros(1),
                "skill_table.skill_bias_retr": torch.zeros(2),
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="skill_pool has placeholder records"):
        validate_stage2_routing_checkpoint(
            checkpoint_path=checkpoint_path,
            skills_path=skills_path,
            model_dim=8,
        )
