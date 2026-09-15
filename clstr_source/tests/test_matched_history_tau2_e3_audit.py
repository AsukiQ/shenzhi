from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_clstr_matched_history_tau2_e3 import (
    audit_e3,
    paired_binary_statistics,
)
from clstr.matched_history_online import (
    E3_FOUNDATION_SHA256,
    E3_SELECTED_CHECKPOINTS,
    E3_SKILLS_SHA256,
)


def test_paired_binary_statistics_uses_candidate_minus_reference() -> None:
    report = paired_binary_statistics(
        [False, False, True, True],
        [True, False, True, False],
        bootstrap_draws=100,
        seed=7,
    )
    assert report["mean_success_difference"] == 0.0
    assert report["mcnemar_exact"]["candidate_only_success"] == 1
    assert report["mcnemar_exact"]["reference_only_success"] == 1
    assert report["mcnemar_exact"]["two_sided_p"] == 1.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_method(tmp_path: Path, method: str) -> Path:
    root = tmp_path / method
    root.mkdir()
    outcomes = []
    selections = []
    counts = {"airline": 20, "retail": 40, "telecom": 40}
    for domain, count in counts.items():
        for index in range(count):
            task_id = str(index)
            success = (index + len(method)) % 3 == 0
            outcomes.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "trial": 0,
                    "simulation_id": f"{method}-{domain}-{index}",
                    "reward": float(success),
                    "success": success,
                    "termination_reason": "user_stop",
                    "infra_error": False,
                }
            )
            candidate_ids = [f"tau2/{domain}/tool{value}" for value in range(14)]
            items = [
                {
                    "rank": value + 1,
                    "skill_id": candidate_ids[value],
                    "score": float(8 - value),
                    "static_score": float(8 - value),
                    "dynamic_score": float(8 - value),
                    "selector_probability": 0.0,
                    "mixture_probability": 0.0,
                    "adaptive_mixture_probability": 0.0,
                    "selected_expert": "static",
                    "encoder_kind": method,
                    "candidate_count": 14,
                    "natural_support_count": 14,
                    "static_support_count": 14,
                    "static_support_sha256": f"support-{domain}-{index}",
                    "history_window_depth": 0,
                }
                for value in range(8)
            ]
            selections.append(
                {
                    "record_type": "selection",
                    "method": f"matched_history_e3_{method}",
                    "route_mode": "static" if method == "static" else "dynamic",
                    "task_id": task_id,
                    "domain": domain,
                    "history_depth": 0,
                    "candidate_skill_ids": candidate_ids,
                    "selected": items,
                }
            )
    if method != "static":
        history_items = [dict(item) for item in selections[0]["selected"]]
        for item in history_items:
            item["selected_expert"] = "dynamic"
            item["mixture_probability"] = 1.0
            item["adaptive_mixture_probability"] = 1.0
            item["history_window_depth"] = 1
        selections.append(
            {
                **selections[0],
                "history_depth": 1,
                "selected": history_items,
            }
        )
    selections.append(
        {
            "record_type": "memory_update",
            "method": f"matched_history_e3_{method}",
            "event_action_text": "tool: tool0 arguments: {}",
            "prefix_reconstructed_at_next_selection": True,
        }
    )
    outcomes_path = root / "task_outcomes.jsonl"
    selections_path = root / "retrieval_selections.jsonl"
    _write_jsonl(outcomes_path, outcomes)
    _write_jsonl(selections_path, selections)
    import hashlib

    outcomes_sha = hashlib.sha256(outcomes_path.read_bytes()).hexdigest()
    success_count = sum(bool(row["success"]) for row in outcomes)
    binding = {
        "foundation_checkpoint_sha256": E3_FOUNDATION_SHA256,
        "skills_sha256": E3_SKILLS_SHA256,
        "e1_checkpoint_sha256": (
            None if method == "static" else E3_SELECTED_CHECKPOINTS[method]["sha256"]
        ),
    }
    report = {
        "schema_version": "clstr_matched_history_tau2_e3_result_v1",
        "status": "ok",
        "evaluation_scope": "full",
        "controlled_method": method,
        "simulation_count": 100,
        "reward_count": 100,
        "missing_reward_count": 0,
        "infra_error_count": 0,
        "success_count": success_count,
        "top_k": 8,
        "max_steps": 100,
        "num_trials": 1,
        "executor_model": "qwen3-14b",
        "executor_temperature": 0.0,
        "user_model": "gpt-4.1",
        "matched_union_manifest_sha256": "manifest",
        "seed": 300,
        "domains": {
            domain: {
                "reward_count": count,
                "official_test_identity_verified": True,
            }
            for domain, count in counts.items()
        },
        "checkpoint_binding": binding,
        "task_outcomes_path": str(outcomes_path),
        "task_outcomes_sha256": outcomes_sha,
        "selection_log_path": str(selections_path),
    }
    report_path = root / "clstr_task_accuracy.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_full_synthetic_e3_audit(tmp_path: Path) -> None:
    paths = {method: _fake_method(tmp_path, method) for method in ("static", "transformer", "lstr")}
    report = audit_e3(paths, bootstrap_draws=100, seed=11)
    assert report["status"] == "ok"
    assert report["task_identity_count"] == 100
    assert report["checks"]["paired_outcomes_complete"] is True
    assert set(report["pairwise"]) == {
        "transformer_minus_static",
        "lstr_minus_static",
        "lstr_minus_transformer",
    }
