from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    torch = None


def map_backbone_state_dict(
    state_dict: dict[str, Any],
    prefixes: Iterable[str],
) -> dict[str, Any]:
    """Strip the first matching prefix from checkpoint keys and ignore non-matches."""
    mapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                stripped_key = key[len(prefix) :]
                if stripped_key in mapped:
                    raise ValueError(
                        f"multiple checkpoint keys map to stripped key {stripped_key!r}"
                    )
                mapped[stripped_key] = value
                break
    return mapped


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to load checkpoint files")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(
            f"checkpoint payload must resolve to a dict, got {type(obj).__name__}"
        )
    return obj
