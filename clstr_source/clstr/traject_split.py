from __future__ import annotations

import hashlib
from typing import Literal

TrajectSplit = Literal["train", "dev", "test"]


def assign_traject_split(
    trajectory_id: str,
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    salt: str = "clstr_v4_traject_split_v1",
) -> TrajectSplit:
    """Assign TRAJECT public_data rows to a stable trajectory-level split.

    TRAJECT public_data does not ship explicit train/dev/test labels. CLSTR
    therefore uses a deterministic hash split at trajectory granularity so
    all steps from one trajectory stay in exactly one partition.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError("dev_ratio must be in [0, 1)")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError("train_ratio + dev_ratio must be < 1")
    digest = hashlib.sha256(f"{salt}:{trajectory_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + dev_ratio:
        return "dev"
    return "test"
