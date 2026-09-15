from __future__ import annotations

from typing import Any

import torch


_ACTION_EMBEDDING_ERRORS = (AttributeError, IndexError, RuntimeError, TypeError, ValueError)


def model_action_embeddings(model: Any, labels: torch.Tensor) -> torch.Tensor | None:
    action_embeddings = getattr(model, "action_embeddings", None)
    if not callable(action_embeddings):
        return None
    try:
        embeddings = action_embeddings(labels)
    except _ACTION_EMBEDDING_ERRORS:
        return None
    if not isinstance(embeddings, torch.Tensor):
        return None
    return embeddings


def transition_action_input(
    model: Any,
    labels: torch.Tensor,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    embeddings = model_action_embeddings(model, labels)
    if embeddings is None:
        return labels
    if like is None:
        return embeddings
    return embeddings.to(device=like.device, dtype=like.dtype)
