from __future__ import annotations

import json
import sys
from pathlib import Path

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_eval_batch_parity import compare_qwen_clstr_eval_batches


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assert_self_hash(report: dict) -> None:
    payload = dict(report)
    digest = payload.pop("report_sha256")
    assert canonical_digest(payload) == digest


def _write_frozen_fixture(root: Path, *, identity: str, reverse_second: bool = False) -> None:
    rankings = [["tau2/a", "tau2/b"], ["tau2/b", "tau2/a"]]
    if reverse_second:
        rankings[1] = list(reversed(rankings[1]))
    rows = [
        {
            "row_id": f"row-{index}",
            "positive_skill_id": ranking[0],
            "positive_rank": 1,
            "declared_candidate_count": 2,
            "ranked_skill_ids": ranking,
            "ranked_scores": [1.0 + index, 0.5 - index],
            "evaluation_identity_sha256": identity,
            "prediction_sha256": f"prediction-{identity}-{index}",
        }
        for index, ranking in enumerate(rankings)
    ]
    _write_jsonl(root / "frozen_route_predictions.jsonl", rows)
    _write_json(
        root / "frozen_route_eval_report.json",
        {
            "status": "ok",
            "prediction_rows": 2,
            "evaluation_identity_sha256": identity,
            "report_sha256": f"report-{identity}",
            "routing_metrics": {
                "recall@1": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
                "prediction_rows": 2,
            },
        },
    )


def _write_native_fixture(
    root: Path,
    *,
    metric_delta: float = 0.0,
    reverse_second: bool = False,
) -> None:
    rankings = [["tau2/a", "tau2/b"], ["tau2/b", "tau2/a"]]
    if reverse_second:
        rankings[1] = list(reversed(rankings[1]))
    _write_jsonl(
        root / "tau2_ranked_rows.jsonl",
        [
            {
                "row_id": f"row-{index}",
                "candidate_next_skill_ids": ranking,
                "candidate_next_prior_scores": [1.0 + index, 0.5 - index],
            }
            for index, ranking in enumerate(rankings)
        ],
    )
    metric = 0.75 + metric_delta
    _write_json(
        root / "tau2_full_clstr_route_eval_report.json",
        {
            "status": "ok",
            "source_eval_rows": 2,
            "retained_eval_rows": 2,
            "stage0_prior_report": {"positive_stage0_domain_mrr": metric},
            "stage0_prior_eval": {"stage4_next_skill_mrr": metric},
            "base_eval": {"stage4_next_skill_mrr": metric},
            "stage4_eval": {
                "stage4_next_skill_recall@1": 0.5,
                "stage4_next_skill_recall@5": 1.0,
                "stage4_next_skill_mrr": metric,
            },
            "strict": {
                "stage4": {
                    "strict_source_rows": 2.0,
                    "strict_stage4_next_skill_mrr": metric,
                }
            },
        },
    )


def _write_alfworld_fixture(root: Path, *, changed_action: bool = False) -> None:
    rows = []
    for index in range(2):
        action = "look" if changed_action and index == 1 else f"open fridge {index}"
        rows.append(
            {
                "episode_index": index,
                "split": "valid_seen",
                "method": "qwen06_clstr_valid_seen",
                "gamefile": f"task/game-{index}",
                "success": True,
                "points": 1.0,
                "goal_condition_points": 1.0,
                "steps": 1.0,
                "action_trace": [action],
                "chosen_action_trace": [action],
            }
        )
    _write_jsonl(root / "run.jsonl", rows)
    _write_json(
        root / "metrics.json",
        {
            "status": "ok",
            "success_rate": 1.0,
            "average_reward": 1.0,
            "average_goal_condition_points": 1.0,
            "average_episode_steps": 1.0,
            "episode_count": 2,
        },
    )


def test_frozen_tau2_parity_ignores_execution_identity_and_raw_scores(tmp_path):
    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    _write_frozen_fixture(fallback, identity="fallback")
    _write_frozen_fixture(accelerated, identity="accelerated")

    report = compare_qwen_clstr_eval_batches(
        kind="frozen_tau2",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["effective_profile"] == "accelerated"
    assert report["semantic_rows"] == {"fallback": 2, "accelerated": 2}
    _assert_self_hash(report)


def test_frozen_tau2_parity_falls_back_on_ranked_skill_order_drift(tmp_path):
    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    _write_frozen_fixture(fallback, identity="fallback")
    _write_frozen_fixture(accelerated, identity="accelerated", reverse_second=True)

    report = compare_qwen_clstr_eval_batches(
        kind="frozen_tau2",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )

    assert report["status"] == "action_required"
    assert report["effective_profile"] == "fallback"
    assert "ranked_skill_ids_mismatch" in report["blockers"]


def test_native_tau2_parity_accepts_tiny_metric_roundoff_and_exact_candidate_order(tmp_path):
    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    _write_native_fixture(fallback)
    _write_native_fixture(accelerated, metric_delta=1.0e-10)

    report = compare_qwen_clstr_eval_batches(
        kind="native_tau2",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )

    assert report["status"] == "ok"
    assert report["effective_profile"] == "accelerated"
    assert report["numeric_tolerance"] == 1.0e-8


def test_native_tau2_parity_falls_back_on_stage0_candidate_order_drift(tmp_path):
    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    _write_native_fixture(fallback)
    _write_native_fixture(accelerated, reverse_second=True)

    report = compare_qwen_clstr_eval_batches(
        kind="native_tau2",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )

    assert report["status"] == "action_required"
    assert "candidate_order_mismatch" in report["blockers"]


def test_alfworld_parity_compares_episode_identity_actions_and_metrics(tmp_path):
    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    _write_alfworld_fixture(fallback)
    _write_alfworld_fixture(accelerated)

    report = compare_qwen_clstr_eval_batches(
        kind="alfworld",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )

    assert report["status"] == "ok"
    assert report["effective_profile"] == "accelerated"

    _write_alfworld_fixture(accelerated, changed_action=True)
    drifted = compare_qwen_clstr_eval_batches(
        kind="alfworld",
        fallback_dir=fallback,
        accelerated_dir=accelerated,
    )
    assert drifted["status"] == "action_required"
    assert "episode_action_trace_mismatch" in drifted["blockers"]


def test_missing_accelerated_artifacts_selects_fallback(tmp_path):
    fallback = tmp_path / "fallback"
    _write_frozen_fixture(fallback, identity="fallback")

    report = compare_qwen_clstr_eval_batches(
        kind="frozen_tau2",
        fallback_dir=fallback,
        accelerated_dir=tmp_path / "missing",
    )

    assert report["status"] == "action_required"
    assert report["effective_profile"] == "fallback"
    assert "missing_accelerated_artifacts" in report["blockers"]


def test_parity_cli_writes_report_and_returns_success(tmp_path, monkeypatch):
    from scripts import compare_qwen06_clstr_eval_batches as cli

    fallback = tmp_path / "fallback"
    accelerated = tmp_path / "accelerated"
    output_path = tmp_path / "parity.json"
    _write_frozen_fixture(fallback, identity="fallback")
    _write_frozen_fixture(accelerated, identity="accelerated")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_qwen06_clstr_eval_batches.py",
            "--kind",
            "frozen_tau2",
            "--fallback_dir",
            str(fallback),
            "--accelerated_dir",
            str(accelerated),
            "--output_path",
            str(output_path),
        ],
    )

    assert cli.main() == 0
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["effective_profile"] == "accelerated"
    _assert_self_hash(saved)
