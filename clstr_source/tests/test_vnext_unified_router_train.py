from __future__ import annotations

from pathlib import Path

import torch

from clstr.vnext_unified_router import (
    UNIFIED_ROUTER_FEATURE_NAMES,
    UNIFIED_ROUTING_MODE_SPARSE,
)
from scripts.train_clstr_vnext_unified_router import (
    ACTION_ONLY_MEMORY_EVIDENCE_MODE,
    FACTUAL_MEMORY_EVIDENCE_MODE,
    STATIC_MEMORY_EVIDENCE_MODE,
    _calibration_blockers,
    _evaluate,
    _minimum_regret_expert_targets,
    _pad_records,
    _paired_dynamic_anchor_views,
    _split_records,
)


def _record(cluster: str, width: int, *, history_depth: float) -> dict:
    evidence_mode = (
        STATIC_MEMORY_EVIDENCE_MODE
        if history_depth <= 0
        else FACTUAL_MEMORY_EVIDENCE_MODE
    )
    return {
        "family": "fixture",
        "source": "fixture-source",
        "cluster_id": cluster,
        "features": torch.zeros(len(UNIFIED_ROUTER_FEATURE_NAMES)),
        "expert_logits": torch.stack(
            [
                torch.arange(width, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32).flip(0),
                torch.zeros(width),
            ]
        ),
        "valid": torch.ones(width, dtype=torch.bool),
        "positive": torch.tensor([True] + [False] * (width - 1)),
        "history_depth": history_depth,
        "observation_correction_count": 0,
        "legal_pool_size": width,
        "memory_evidence_mode": evidence_mode,
        "calibration_pair_id": cluster if history_depth > 0 else "",
    }


def test_unified_router_builds_same_prefix_factual_and_action_only_views() -> None:
    rows = [
        {
            "actual_result_text": "real result",
            "actual_result_executed": True,
            "capabilities": {"actual_execution_result": True},
        },
        {
            "actual_result_text": "",
            "actual_result_executed": False,
            "capabilities": {"actual_execution_result": False},
        },
        {
            "actual_result_text": "future result",
            "actual_result_executed": True,
            "capabilities": {"actual_execution_result": True},
        },
    ]
    views = _paired_dynamic_anchor_views(
        [
            {
                "identity": "pair-a",
                "rows": rows,
                "target_index": 2,
            }
        ]
    )
    assert [view["memory_evidence_mode"] for view in views] == [
        FACTUAL_MEMORY_EVIDENCE_MODE,
        ACTION_ONLY_MEMORY_EVIDENCE_MODE,
    ]
    assert [view["observation_correction_count"] for view in views] == [1, 0]
    assert views[0]["rows"] is rows
    assert views[1]["rows"] is not rows
    assert views[1]["rows"][0]["actual_result_text"] == ""
    assert views[1]["rows"][0]["actual_result_executed"] is False
    assert views[1]["rows"][2]["actual_result_text"] == "future result"
    assert rows[0]["actual_result_text"] == "real result"


def test_unified_router_split_keeps_multiple_views_of_one_trajectory_together() -> None:
    clusters = ("a", "b", "c", "i")
    records = [
        _record(
            f"fixture-source:trajectory-{cluster}",
            width,
            history_depth=depth,
        )
        for cluster in clusters
        for width, depth in ((3, 0.0), (4, 2.0))
    ]
    train, dev = _split_records(records, seed=29)
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert (left in train) == (right in train)
        assert (left in dev) == (right in dev)


def test_unified_router_record_padding_preserves_variable_natural_support() -> None:
    records = [
        _record("fixture-source:trajectory-a", 3, history_depth=0.0),
        _record("fixture-source:trajectory-b", 5, history_depth=2.0),
    ]
    tensors = _pad_records(records)
    assert tensors["expert_logits"].shape == (2, 3, 5)
    assert tensors["valid"].sum(dim=-1).tolist() == [3, 5]
    assert tensors["positive"].sum(dim=-1).tolist() == [1, 1]
    assert tensors["history_depth"].tolist() == [0.0, 2.0]


def test_unified_router_training_exposes_global_utility_weight() -> None:
    root = Path(__file__).parents[1]
    trainer = root.joinpath("scripts/train_clstr_vnext_unified_router.py").read_text(
        encoding="utf-8"
    )
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_unified_router_train.sh"
    ).read_text(encoding="utf-8")
    assert '--expert_utility_loss_weight", type=float, default=1.0' in trainer
    assert "float(expert_utility_loss_weight) * calibration_loss" in trainer
    assert "target = _minimum_regret_expert_targets" in trainer
    assert "weights.gather(1, target.unsqueeze(-1))" in trainer
    assert "torch.softmax(utility / 0.15" not in trainer
    assert "dev_oracle_expert_recall_below_floor" in trainer
    assert "MIN_ORACLE_EXPERT_RECALL = 0.20" in trainer
    assert "GLOBAL_R5_NO_REGRET_TOLERANCE = 0.01" in trainer
    assert "FAMILY_R5_NO_REGRET_TOLERANCE = 0.03" in trainer
    assert "MEMORY_EVIDENCE_R5_NO_REGRET_TOLERANCE = 0.01" in trainer
    assert "EXPERT_UTILITY_LOSS_WEIGHT:-1.0" in launcher
    assert "ROUTING_MODE:-sparse_top1_expert" in launcher
    assert "_preserved_foundation_candidate_logits" in trainer
    assert "def _foundation_candidate_logits" not in trainer


def _no_regret_gate_report(
    *,
    global_r5_gap: float,
    family_r5_gap: float,
    evidence_r5_gap: float,
) -> dict:
    def group(row_count: int, r5_gap: float) -> dict:
        unified = {"mrr": 0.90, "recall@5": 0.90}
        return {
            "row_count": row_count,
            "unified": unified,
            "experts": {
                "foundation": {
                    "mrr": unified["mrr"],
                    "recall@5": unified["recall@5"] + r5_gap,
                },
                "adapted_static": unified,
                "recurrent_memory": unified,
            },
        }

    report = group(100, global_r5_gap)
    report.update(
        {
            "winner_counts": {"foundation": 50, "adapted_static": 50},
            "oracle_winner_counts": {
                "foundation": 34,
                "adapted_static": 33,
                "recurrent_memory": 33,
            },
            "oracle_expert_recall": {
                "foundation": 0.50,
                "adapted_static": 0.50,
                "recurrent_memory": 0.50,
            },
            "per_family": {"fixture": group(100, family_r5_gap)},
            "per_memory_evidence_mode": {
                FACTUAL_MEMORY_EVIDENCE_MODE: group(50, 0.0),
                ACTION_ONLY_MEMORY_EVIDENCE_MODE: group(
                    50,
                    evidence_r5_gap,
                ),
            },
        }
    )
    return report


def test_unified_router_no_regret_r5_tolerances_are_bounded() -> None:
    assert _calibration_blockers(
        _no_regret_gate_report(
            global_r5_gap=0.01,
            family_r5_gap=0.03,
            evidence_r5_gap=0.01,
        )
    ) == []

    blockers = _calibration_blockers(
        _no_regret_gate_report(
            global_r5_gap=0.0101,
            family_r5_gap=0.0301,
            evidence_r5_gap=0.0101,
        )
    )
    assert "dev_r5_regresses_best_constant_expert" in blockers
    assert "dev_family_r5_regression:fixture" in blockers
    assert (
        "dev_memory_evidence_r5_regression:action_only_counterfactual"
        in blockers
    )


def test_unified_router_sparse_targets_prefer_simpler_expert_on_ties() -> None:
    targets = _minimum_regret_expert_targets(
        torch.tensor(
            [
                [1.0, 1.0, 2.0],
                [2.0, 1.0, 1.0],
                [2.0, 3.0, 1.0],
            ]
        ),
        torch.tensor([2.0, 0.0, 2.0]),
    )
    assert targets.tolist() == [0, 1, 2]


class _FixedRouter(torch.nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        *,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.zeros((features.size(0), 3), dtype=features.dtype)
        weights[:, 1] = 1.0
        weights[history_mask, 1] = 0.0
        weights[history_mask, 2] = 1.0
        return weights


def test_unified_router_evaluation_reports_soft_hard_and_oracle_routes() -> None:
    records = [
        _record("fixture-source:trajectory-a", 3, history_depth=0.0),
        _record("fixture-source:trajectory-b", 4, history_depth=2.0),
    ]
    report = _evaluate(
        _FixedRouter(),
        _pad_records(records),
        [0, 1],
        records,
        device=torch.device("cpu"),
        batch_size=2,
        routing_mode=UNIFIED_ROUTING_MODE_SPARSE,
    )
    assert report["routing_mode"] == UNIFIED_ROUTING_MODE_SPARSE
    assert report["unified"] == report["hard_routed"]
    assert report["hard_routed"]["mrr"] > 0.0
    assert report["oracle"]["mrr"] >= report["hard_routed"]["mrr"]
    assert set(report["oracle_expert_recall"]) == {
        "foundation",
        "adapted_static",
        "recurrent_memory",
    }
    assert report["winner_counts"] == {
        "adapted_static": 1,
        "recurrent_memory": 1,
    }
    assert report["per_family"]["fixture"]["mean_expert_weights"] == {
        "foundation": 0.0,
        "adapted_static": 0.5,
        "recurrent_memory": 0.5,
    }
