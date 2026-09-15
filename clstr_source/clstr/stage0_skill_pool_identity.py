from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from typing import Any

import torch

from clstr.encoders import SkillTable


SKILL_POOL_IDENTITY_ALGORITHM = (
    "sha256_ordered_skill_id_and_serialized_text_v1"
)


def _update_field(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def ordered_skill_pool_identity(
    skills: Sequence[Mapping[str, Any] | Any],
    *,
    skill_text_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for index, skill in enumerate(skills):
        payload = SkillTable._skill_payload(skill)
        skill_id = str(
            payload.get("skill_id")
            or payload.get("canonical_skill_id")
            or payload.get("id")
            or index
        )
        _update_field(digest, skill_id)
        _update_field(digest, str(skill_text_fn(payload)))
    return {
        "algorithm": SKILL_POOL_IDENTITY_ALGORITHM,
        "count": len(skills),
        "digest": digest.hexdigest(),
    }


def _resume_error(message: str) -> ValueError:
    return ValueError(f"verified resume skill table {message}")


def validate_verified_resume_skill_table(
    *,
    checkpoint_state: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any] | None,
    resume_skill_rows: Sequence[Mapping[str, Any] | Any] | None,
    current_skill_rows: Sequence[Mapping[str, Any] | Any],
    expected_shape: tuple[int, int],
    expected_config: Mapping[str, Any],
    skill_text_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    embeddings = checkpoint_state.get("skill_table.E")
    if not isinstance(embeddings, torch.Tensor):
        raise _resume_error("is missing skill_table.E")
    if embeddings.ndim != 2 or tuple(embeddings.shape) != tuple(expected_shape):
        raise _resume_error(
            "shape mismatch: "
            f"checkpoint={list(embeddings.shape)} expected={list(expected_shape)}"
        )

    mismatched_config = {
        key: {
            "checkpoint": checkpoint_config.get(key),
            "expected": expected_value,
        }
        for key, expected_value in expected_config.items()
        if checkpoint_config.get(key) != expected_value
    }
    if mismatched_config:
        raise _resume_error(f"config mismatch: {mismatched_config}")

    current_identity = ordered_skill_pool_identity(
        current_skill_rows,
        skill_text_fn=skill_text_fn,
    )
    if checkpoint_identity is not None:
        candidate_identity = dict(checkpoint_identity)
        identity_source = "checkpoint"
    elif resume_skill_rows is not None:
        candidate_identity = ordered_skill_pool_identity(
            resume_skill_rows,
            skill_text_fn=skill_text_fn,
        )
        identity_source = "resume_selected_skills"
    else:
        raise _resume_error(
            "cannot verify ordered skill-pool identity without checkpoint "
            "identity or selected-skills sidecar"
        )
    if candidate_identity != current_identity:
        raise _resume_error(
            "ordered skill-pool identity mismatch: "
            f"checkpoint={candidate_identity} current={current_identity}"
        )

    return {
        "verified": True,
        "identity_source": identity_source,
        "skill_pool_identity": current_identity,
        "skill_embedding_shape": list(embeddings.shape),
        "verified_config": {
            key: checkpoint_config.get(key)
            for key in expected_config
        },
    }
