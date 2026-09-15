from __future__ import annotations

import pytest
import torch

from clstr.retrieval_warmup import _load_stage0_resume_state
from clstr.stage0_skill_pool_identity import (
    ordered_skill_pool_identity,
    validate_verified_resume_skill_table,
)


def _serialize_skill(row):
    return f"{row['name']}\n{row['description']}"


def _skill_rows():
    return [
        {"skill_id": "skill/a", "name": "a", "description": "first"},
        {"skill_id": "skill/b", "name": "b", "description": "second"},
    ]


def _validate(
    *,
    checkpoint_state=None,
    checkpoint_config=None,
    checkpoint_identity=None,
    resume_skill_rows=None,
    current_skill_rows=None,
    expected_shape=(2, 3),
    expected_config=None,
):
    return validate_verified_resume_skill_table(
        checkpoint_state=(
            {"skill_table.E": torch.zeros(2, 3)}
            if checkpoint_state is None
            else checkpoint_state
        ),
        checkpoint_config=(
            {"d": 3, "skill_text_format": "test"}
            if checkpoint_config is None
            else checkpoint_config
        ),
        checkpoint_identity=checkpoint_identity,
        resume_skill_rows=resume_skill_rows,
        current_skill_rows=(
            _skill_rows() if current_skill_rows is None else current_skill_rows
        ),
        expected_shape=expected_shape,
        expected_config=(
            {"d": 3, "skill_text_format": "test"}
            if expected_config is None
            else expected_config
        ),
        skill_text_fn=_serialize_skill,
    )


def test_ordered_identity_changes_when_pool_order_changes():
    rows = _skill_rows()

    original = ordered_skill_pool_identity(rows, skill_text_fn=_serialize_skill)
    reordered = ordered_skill_pool_identity(
        list(reversed(rows)),
        skill_text_fn=_serialize_skill,
    )

    assert original["count"] == 2
    assert len(original["digest"]) == 64
    assert reordered != original


def test_ordered_identity_changes_when_serialized_text_changes():
    rows = _skill_rows()
    changed = [
        rows[0],
        {**rows[1], "description": "changed"},
    ]

    assert ordered_skill_pool_identity(
        rows,
        skill_text_fn=_serialize_skill,
    ) != ordered_skill_pool_identity(
        changed,
        skill_text_fn=_serialize_skill,
    )


def test_verified_resume_accepts_matching_sidecar_identity():
    report = _validate(resume_skill_rows=_skill_rows())

    assert report["verified"] is True
    assert report["identity_source"] == "resume_selected_skills"
    assert report["skill_embedding_shape"] == [2, 3]


def test_verified_resume_accepts_matching_embedded_identity_without_sidecar():
    identity = ordered_skill_pool_identity(
        _skill_rows(),
        skill_text_fn=_serialize_skill,
    )

    report = _validate(checkpoint_identity=identity)

    assert report["verified"] is True
    assert report["identity_source"] == "checkpoint"


def test_verified_resume_rejects_missing_identity_and_sidecar():
    with pytest.raises(ValueError, match="verified resume skill table"):
        _validate()


def test_verified_resume_rejects_same_shape_reordered_pool():
    with pytest.raises(ValueError, match="ordered skill-pool identity mismatch"):
        _validate(resume_skill_rows=list(reversed(_skill_rows())))


def test_verified_resume_rejects_changed_serialized_skill_text():
    changed = _skill_rows()
    changed[1] = {**changed[1], "description": "changed"}

    with pytest.raises(ValueError, match="ordered skill-pool identity mismatch"):
        _validate(resume_skill_rows=changed)


def test_verified_resume_rejects_skill_embedding_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate(
            checkpoint_state={"skill_table.E": torch.zeros(3, 3)},
            resume_skill_rows=_skill_rows(),
        )


def test_verified_resume_rejects_checkpoint_config_mismatch():
    with pytest.raises(ValueError, match="config mismatch"):
        _validate(
            checkpoint_config={"d": 4, "skill_text_format": "test"},
            resume_skill_rows=_skill_rows(),
        )


def test_verified_resume_rejects_missing_skill_table_tensor():
    with pytest.raises(ValueError, match="missing skill_table.E"):
        _validate(
            checkpoint_state={},
            resume_skill_rows=_skill_rows(),
        )


def test_resume_loader_captures_selected_skills_sidecar_before_training_rewrites_it(
    tmp_path,
):
    output_dir = tmp_path / "stage0"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "clstr_unified_retrieval_v2-step1.pt"
    torch.save(
        {
            "step": 1,
            "model_state_dict": {"skill_table.E": torch.zeros(2, 3)},
        },
        checkpoint_path,
    )
    selected_skills_path = output_dir / "selected_skills.jsonl"
    selected_skills_path.write_text(
        "".join(
            [
                '{"skill_id": "skill/a", "name": "a", "description": "first"}\n',
                '{"skill_id": "skill/b", "name": "b", "description": "second"}\n',
            ]
        ),
        encoding="utf-8",
    )

    loaded = _load_stage0_resume_state(checkpoint_path)

    assert loaded["resume_selected_skills_path"] == str(selected_skills_path)
    assert loaded["resume_selected_skill_rows"] == _skill_rows()
