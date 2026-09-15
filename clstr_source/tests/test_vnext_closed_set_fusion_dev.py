from __future__ import annotations

from pathlib import Path

import torch

from scripts.run_clstr_vnext_closed_set_fusion_dev import _row_ranks, _summarize


def test_equal_rrf_uses_both_independent_rankings() -> None:
    static_rank, fused_rank = _row_ranks(
        torch.tensor([4.0, 3.0, 2.0]),
        torch.tensor([1.0, 3.0, 4.0]),
        torch.tensor([10, 11, 12]),
        {12},
    )
    assert static_rank == 3
    assert fused_rank == 2


def test_pool_regime_keeps_tiny_static_and_fuses_larger_pool() -> None:
    report = _summarize(
        [
            {
                "family": "tiny",
                "pool_size": 4,
                "static_rank": 1,
                "rrf_rank": 3,
            },
            {
                "family": "moderate",
                "pool_size": 14,
                "static_rank": 6,
                "rrf_rank": 4,
            },
        ],
        "static_le_8_else_rrf",
    )
    assert report["fused_row_count"] == 1
    assert report["recall@1"] == 0.5
    assert report["recall@5"] == 1.0


def test_fusion_dev_launcher_exposes_all_immutable_inputs() -> None:
    launcher = Path(__file__).parents[1].joinpath(
        "scripts/sbatch/run_clstr_vnext_closed_set_fusion_dev.sh"
    ).read_text(encoding="utf-8")
    for name in (
        "CHECKPOINT_PATH",
        "TRAINING_SKILLS_PATH",
        "STATIC_ROUTE_DEV_ROWS",
        "INVENTORY_CATALOGS",
        "EVAL_CACHE_ROOT",
        "OUTPUT_PATH",
    ):
        assert f"{name}=${{{name}:?" in launcher
