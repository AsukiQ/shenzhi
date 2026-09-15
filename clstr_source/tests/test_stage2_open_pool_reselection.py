from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clstr.vnext_eval import (
    _stage2_release_selection_contract,
    _unified_router_release_wrapper,
)
from scripts.reselect_clstr_vnext_stage2 import (
    _refresh_coarse_union_gates,
    _select_open_pool_validation,
    select_open_pool_release,
)


REQUIRED_GATES = {
    "causal_route_gate": True,
    "cluster_sufficiency_gate": True,
    "causal_safety_gate": True,
    "coarse_recall_gate": True,
    "exact_fallback_gate": True,
    "gradient_health_gate": True,
    "robust_prefix_dev_gate": True,
}


def _legacy_union_record(*, union_delta_mean: float = 0.0) -> dict:
    metric = {
        "count": 20,
        "cluster_count": 20,
        "mean": 0.9,
        "ci_low": 0.8,
        "ci_high": 1.0,
    }
    return {
        "step": 1000,
        "ordinary_dev": {
            "overall": {
                "candidate_recall_at_500": dict(metric),
                "candidate_union_recall": {
                    **metric,
                    "mean": 0.9 + union_delta_mean,
                },
                "candidate_support_change_rate": {
                    **metric,
                    "mean": 0.25,
                },
                "candidate_recall": {
                    "100": {
                        "deployment_full_pool": {
                            "fixed_causal_hit_rate": dict(metric),
                            "dynamic_hit_rate": dict(metric),
                            "dynamic_minus_static": {
                                **metric,
                                "mean": 0.0,
                                "ci_low": 0.0,
                            },
                        }
                    },
                    "500": {
                        "deployment_full_pool": {
                            "fixed_causal_hit_rate": dict(metric),
                            "dynamic_hit_rate": {
                                **metric,
                                "mean": 0.75,
                                "ci_low": 0.6,
                            },
                            "dynamic_minus_static": {
                                **metric,
                                "mean": -0.15,
                                "ci_low": -0.2,
                            },
                        }
                    },
                },
            }
        },
        "gates": {
            **REQUIRED_GATES,
            "ordinary_safety_gate": True,
            "coarse_recall_gate": False,
            "ordinary_safety": {"sentinel": "unchanged"},
            "pass": False,
        },
    }


def test_reselection_refreshes_only_mismatched_coarse_union_gate() -> None:
    original = _legacy_union_record()
    refreshed, audit = _refresh_coarse_union_gates(
        [original],
        minimum_full_pool_clusters=20,
    )
    gates = refreshed[0]["gates"]
    assert gates["coarse_recall_gate"] is True
    assert gates["pass"] is True
    assert gates["ordinary_safety"] == {"sentinel": "unchanged"}
    assert original["gates"]["coarse_recall_gate"] is False
    assert audit[0]["previous_coarse_recall_gate"] is False
    assert audit[0]["refreshed_coarse_recall_gate"] is True
    assert gates["coarse_recall"][
        "dynamic_only_deployment_noninferiority_pass"
    ] is False


def test_reselection_coarse_refresh_still_rejects_union_regression() -> None:
    record = _legacy_union_record(union_delta_mean=-0.01)
    refreshed, _audit = _refresh_coarse_union_gates(
        [record],
        minimum_full_pool_clusters=20,
    )
    assert refreshed[0]["gates"]["coarse_recall_gate"] is False
    assert refreshed[0]["gates"]["pass"] is False


def _record(
    step: int,
    *,
    heldout_low: float,
    heldout_mean: float,
    overall_low: float,
    heldout_r5_low: float | None = None,
    heldout_r5_mean: float | None = None,
    failed_gate: str | None = None,
) -> dict:
    gates = dict(REQUIRED_GATES)
    if failed_gate is not None:
        gates[failed_gate] = False
    resolved_r5_low = heldout_low if heldout_r5_low is None else heldout_r5_low
    resolved_r5_mean = heldout_mean if heldout_r5_mean is None else heldout_r5_mean
    return {
        "step": step,
        "gates": gates,
        "ordinary_dev": {
            "overall": {
                "raw_route_mrr_delta": {
                    "mean": overall_low + 0.02,
                    "ci_low": overall_low,
                    "ci_high": overall_low + 0.04,
                    "count": 64,
                    "cluster_count": 32,
                },
                "candidate_recall": {
                    "500": {
                        "deployment_full_pool": {
                            "row_count": 24,
                            "cluster_count": 12,
                        }
                    }
                },
            },
            "per_family": {
                "toolbench": {
                    "row_count": 24,
                    "cluster_count": 12,
                    "raw_route_mrr_delta": {
                        "mean": heldout_mean,
                        "ci_low": heldout_low,
                        "ci_high": heldout_low + 0.08,
                        "count": 24,
                        "cluster_count": 12,
                    },
                    "raw_route_recall_at_5_delta": {
                        "mean": resolved_r5_mean,
                        "ci_low": resolved_r5_low,
                        "ci_high": resolved_r5_low + 0.08,
                        "count": 24,
                        "cluster_count": 12,
                    },
                    "route_recall_at_5_delta": {
                        "mean": resolved_r5_mean,
                        "ci_low": resolved_r5_low,
                        "ci_high": resolved_r5_low + 0.08,
                        "count": 24,
                        "cluster_count": 12,
                    },
                }
            },
            "per_source": {
                "toolbench_g3": {
                    "row_count": 24,
                    "cluster_count": 12,
                    "raw_route_mrr_delta": {
                        "mean": heldout_mean,
                        "ci_low": heldout_low - 0.001,
                        "ci_high": heldout_low + 0.079,
                        "count": 24,
                        "cluster_count": 12,
                    },
                    "raw_route_recall_at_5_delta": {
                        "mean": resolved_r5_mean,
                        "ci_low": resolved_r5_low - 0.001,
                        "ci_high": resolved_r5_low + 0.079,
                        "count": 24,
                        "cluster_count": 12,
                    },
                    "route_recall_at_5_delta": {
                        "mean": resolved_r5_mean,
                        "ci_low": resolved_r5_low - 0.001,
                        "ci_high": resolved_r5_low + 0.079,
                        "count": 24,
                        "cluster_count": 12,
                    },
                }
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_open_pool_selector_prefers_best_heldout_ci_not_latest_or_aggregate() -> None:
    selected, summaries = _select_open_pool_validation(
        [
            _record(1000, heldout_low=-0.03, heldout_mean=0.02, overall_low=0.01),
            _record(5500, heldout_low=-0.01, heldout_mean=0.04, overall_low=0.05),
            _record(10000, heldout_low=-0.06, heldout_mean=0.01, overall_low=0.07),
        ],
        minimum_full_pool_clusters=10,
    )
    assert selected["step"] == 5500
    assert all(summary["eligible"] for summary in summaries)


def test_open_pool_selector_requires_overall_safety_and_causal_gates() -> None:
    selected, summaries = _select_open_pool_validation(
        [
            _record(5000, heldout_low=0.10, heldout_mean=0.12, overall_low=-0.01),
            _record(
                6000,
                heldout_low=0.20,
                heldout_mean=0.22,
                overall_low=0.03,
                failed_gate="gradient_health_gate",
            ),
            _record(5500, heldout_low=-0.01, heldout_mean=0.04, overall_low=0.05),
        ],
        minimum_full_pool_clusters=10,
    )
    assert selected["step"] == 5500
    by_step = {summary["step"]: summary for summary in summaries}
    assert "overall_raw_route_ci_not_positive" in by_step[5000][
        "ineligibility_reasons"
    ]
    assert "failed_gate:gradient_health_gate" in by_step[6000][
        "ineligibility_reasons"
    ]


def test_open_pool_selector_can_select_heldout_r5_without_test_metrics() -> None:
    selected, summaries = _select_open_pool_validation(
        [
            _record(
                250,
                heldout_low=0.03,
                heldout_mean=0.05,
                heldout_r5_low=-0.02,
                heldout_r5_mean=0.01,
                overall_low=0.02,
            ),
            _record(
                500,
                heldout_low=0.02,
                heldout_mean=0.04,
                heldout_r5_low=0.01,
                heldout_r5_mean=0.06,
                overall_low=0.02,
            ),
        ],
        minimum_full_pool_clusters=10,
        heldout_metric="route_recall_at_5_delta",
    )
    assert selected["step"] == 500
    assert selected["heldout_metric"] == "route_recall_at_5_delta"
    assert all(summary["eligible"] for summary in summaries)


def test_open_pool_release_is_additive_and_evaluator_binds_its_digest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage2"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    for step in (1000, 5500, 10000):
        (checkpoints / f"clstr_vnext_stage2-step{step}.pt").write_bytes(
            f"checkpoint-{step}".encode()
        )
    records = [
        _record(1000, heldout_low=-0.03, heldout_mean=0.02, overall_low=0.01),
        _record(5500, heldout_low=-0.01, heldout_mean=0.04, overall_low=0.05),
        _record(10000, heldout_low=-0.06, heldout_mean=0.01, overall_low=0.07),
    ]
    original = {
        "status": "ok",
        "selected_step": 1000,
        "robust_prefix_exposure_gate": True,
        "positive_injection_count": 0,
        "minimum_full_pool_clusters": 10,
        "validation_records": records,
    }
    selection_path = output / "stage2_selection.json"
    _write_json(selection_path, original)
    _write_json(output / "stage2_quality_gate.json", original)
    _write_json(output / "train_report.json", {"status": "ok"})
    _write_json(output / "source_manifest.json", {"source": "unit"})
    original_bytes = selection_path.read_bytes()

    release_dir = tmp_path / "release"
    result = select_open_pool_release(
        output,
        release_dir,
        source_commit="unit-source-commit",
    )
    assert result["selected_step"] == 5500
    assert selection_path.read_bytes() == original_bytes

    release_path = Path(result["selection_path"])
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    assert payload["original_selected_step"] == 1000
    assert payload["uses_benchmark_eval_rows"] is False
    assert payload["uses_test_metrics"] is False
    assert payload["closed_set_dispatch_required"] is True
    assert payload["selected_checkpoint_sha256"] == hashlib.sha256(
        Path(result["selected_checkpoint_path"]).read_bytes()
    ).hexdigest()

    contract = _stage2_release_selection_contract(
        release_path,
        stage2_checkpoint_path=result["selected_checkpoint_path"],
    )
    assert contract["selected_step"] == 5500
    assert contract["selection_sha256"] == hashlib.sha256(
        release_path.read_bytes()
    ).hexdigest()
    foundation = tmp_path / "foundation.pt"
    skills = tmp_path / "skills.jsonl"
    foundation.write_bytes(b"foundation")
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")
    wrapper = _unified_router_release_wrapper(
        contract,
        router_source_commit="router-source-commit",
        foundation_checkpoint_path=foundation,
        foundation_checkpoint_sha256=hashlib.sha256(
            foundation.read_bytes()
        ).hexdigest(),
        training_skills_path=skills,
        training_skills_sha256=hashlib.sha256(skills.read_bytes()).hexdigest(),
    )
    assert wrapper["router_source_commit"] == "router-source-commit"
    assert wrapper["stage2_release_source_commit"] == "unit-source-commit"
    assert wrapper["stage2_release_selection_sha256"] == contract["selection_sha256"]
    assert wrapper["stage2_checkpoint_sha256"] == contract[
        "selected_checkpoint_sha256"
    ]
    assert wrapper["uses_benchmark_eval_rows"] is False
    assert wrapper["uses_test_metrics"] is False

    test_bound_contract = {**contract, "uses_test_metrics": True}
    with pytest.raises(ValueError, match="benchmark test evidence"):
        _unified_router_release_wrapper(
            test_bound_contract,
            router_source_commit="router-source-commit",
            foundation_checkpoint_path=foundation,
            foundation_checkpoint_sha256=wrapper[
                "foundation_checkpoint_sha256"
            ],
            training_skills_path=skills,
            training_skills_sha256=wrapper["training_skills_sha256"],
        )

    with pytest.raises(ValueError, match="differs from the release selection"):
        _stage2_release_selection_contract(
            release_path,
            stage2_checkpoint_path=(
                checkpoints / "clstr_vnext_stage2-step1000.pt"
            ),
        )


def test_open_pool_release_accepts_verified_interface_scoped_refinement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage2_refinement"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "clstr_vnext_stage2-step500.pt"
    checkpoint.write_bytes(b"refined-checkpoint")
    records = [
        _record(
            500,
            heldout_low=0.01,
            heldout_mean=0.06,
            heldout_r5_low=0.02,
            heldout_r5_mean=0.08,
            overall_low=0.04,
        )
    ]
    selection = {
        "status": "action_required",
        "selected_step": 500,
        "robust_prefix_exposure_gate": True,
        "positive_injection_count": 0,
        "minimum_full_pool_clusters": 10,
        "validation_records": records,
    }
    warm_start = {
        "enabled": True,
        "model_state_exact": True,
        "optimizer_restored": False,
        "trainer_progress_restored": False,
    }
    topk = {
        "enabled": True,
        "k": 5,
        "lambda": 1.0,
        "eligible_rows": 128,
        "positive_injection_count": 0,
        "candidate_membership_protocol": "immutable_natural_support",
    }
    report = {
        "status": "action_required",
        "finite_loss": True,
        "selection": selection,
        "objective_warm_start": warm_start,
        "route_topk": topk,
        "positive_injection_count": 0,
        "training_teacher_retained_rows": 0,
    }
    _write_json(output / "stage2_selection.json", selection)
    _write_json(output / "stage2_quality_gate.json", selection)
    _write_json(output / "train_report.json", report)
    _write_json(output / "source_manifest.json", {"source": "unit"})

    result = select_open_pool_release(
        output,
        tmp_path / "release_refinement",
        source_commit="unit-source-commit",
        heldout_metric="route_recall_at_5_delta",
    )
    payload = json.loads(Path(result["selection_path"]).read_text())
    assert payload["original_training_status"] == "action_required"
    assert payload["interface_scoped_training_approval"] is True
    contract = _stage2_release_selection_contract(
        result["selection_path"],
        stage2_checkpoint_path=checkpoint,
    )
    assert contract["interface_scoped_training_approval"] is True
    assert contract["heldout_metric"] == "route_recall_at_5_delta"

    report["training_teacher_retained_rows"] = 1
    _write_json(output / "train_report.json", report)
    with pytest.raises(ValueError, match="verified natural-support refinement"):
        select_open_pool_release(
            output,
            tmp_path / "release_refinement_teacher_tamper",
            source_commit="unit-source-commit",
            heldout_metric="route_recall_at_5_delta",
        )
