from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.memory_utility_records import canonical_digest
from clstr.memory_utility_gate import (
    MEMORY_UTILITY_FEATURE_NAMES,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    MemoryUtilityGate,
)
from clstr.qwen_clstr_lineage import sha256_path
from clstr.stage4_validation import (
    _build_route_record,
    choose_fixed_alpha,
    finalize_stage4_selection,
    select_stage4_cmc_checkpoint,
    select_stage4_candidate_admission_checkpoint,
    select_stage4_dynamic_checkpoint,
    validate_static_baseline_batches,
    write_stage4_dynamic_selection,
)


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")


def test_stage4_route_record_persists_deployed_gate_alpha() -> None:
    record = _build_route_record(
        row={"trajectory_id": "t0", "benchmark": "toolbench_g3"},
        source_row_index=0,
        candidate_indices=[0, 1],
        ordered_skill_ids=["skill/a", "skill/b"],
        static_logits=torch.tensor([2.0, 1.0]),
        dynamic_logits=torch.tensor([1.0, 2.0]),
        valid_mask=torch.tensor([True, True]),
        positive_mask=torch.tensor([False, True]),
        features=torch.zeros(len(MEMORY_UTILITY_FEATURE_NAMES)),
        causal_update_count=torch.tensor(1.0),
        raw_alpha=torch.tensor(0.75),
        effective_alpha=torch.tensor(0.75),
    )

    assert record["raw_alpha"] == pytest.approx(0.75)
    assert record["effective_alpha"] == pytest.approx(0.75)


def _alpha_metrics(mrr: float) -> dict:
    return {
        "balanced_macro_mrr": mrr,
        "balanced_macro_recall@5": mrr,
        "zero_history_exact": True,
        "residual_bound": 2.0,
        "by_benchmark": {
            benchmark: {"mrr": mrr} for benchmark in BENCHMARKS
        },
    }


def _report(
    step: int,
    dynamic_mrr: float,
    delta_mrr: float,
    recall5: float,
) -> dict:
    safe_mrr = dynamic_mrr + 0.01
    return {
        "schema_version": "stage4_validation_report_v1",
        "status": "ok",
        "release_status": "ok",
        "step": step,
        "mechanically_eligible": True,
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "report_sha256": "stage2-baseline-sha256",
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.59,
            "dynamic_macro_mrr": 0.60,
            "by_benchmark": {
                benchmark: {
                    "static_mrr": 0.59,
                    "dynamic_mrr": 0.60,
                }
                for benchmark in BENCHMARKS
            },
        },
        "candidate_union": {
            "static_k": 500,
            "dynamic_extra_k": 64,
            "final_k": 64,
        },
        "nonfinite_counts": {
            "loss": 0,
            "logits": 0,
            "metrics": 0,
            "gradients": 0,
        },
        "exclusion_counts": {
            "missing_positive": 0,
            "no_legal_candidate": 0,
        },
        "delta_state_validation": {"status": "ok", "state_key_count": 6},
        "balanced_macro": {
            "dynamic_mrr": dynamic_mrr,
            "dynamic_minus_static_mrr": delta_mrr,
            "dynamic_recall@5": recall5,
            "safe_fused_mrr": safe_mrr,
            "safe_fused_minus_static_mrr": safe_mrr - (dynamic_mrr - delta_mrr),
            "safe_fused_recall@5": recall5,
        },
        "by_benchmark": {
            benchmark: {
                "dynamic_mrr": dynamic_mrr,
                "static_mrr": dynamic_mrr - delta_mrr,
                "dynamic_minus_static_mrr": delta_mrr,
                "safe_fused_mrr": safe_mrr,
                "safe_fused_minus_static_mrr": safe_mrr - (dynamic_mrr - delta_mrr),
            }
            for benchmark in BENCHMARKS
        },
        "safe_fused": _alpha_metrics(safe_mrr),
        "fixed_alpha": {
            "0.0": _alpha_metrics(dynamic_mrr - delta_mrr),
            "0.25": _alpha_metrics(dynamic_mrr - 0.01),
            "0.5": _alpha_metrics(dynamic_mrr + 0.01),
            "0.75": _alpha_metrics(dynamic_mrr + 0.005),
            "1.0": _alpha_metrics(dynamic_mrr),
        },
    }


def _persist_report(tmp_path: Path, report: dict) -> tuple[Path, Path]:
    step = int(report["step"])
    checkpoint = tmp_path / f"persisted-stage4-step{step}.pt"
    checkpoint.write_bytes(str(step).encode("utf-8"))
    validation_records = tmp_path / f"persisted-validation-step{step}.jsonl"
    validation_records.write_text('{"row_digest":"row-a"}\n', encoding="utf-8")
    gate_records = tmp_path / f"persisted-gate-step{step}.jsonl"
    gate_records.write_text(
        '{"row_digest":"gate-row-a"}\n',
        encoding="utf-8",
    )
    gate_manifest = tmp_path / f"persisted-gate-step{step}.manifest.json"
    gate_manifest.write_text("{}", encoding="utf-8")
    report["validation_route_records"] = {
        **sha256_path(validation_records),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["row-a"]),
    }
    report["gate_route_records"] = {
        **sha256_path(gate_records),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["gate-row-a"]),
    }
    report["gate_route_manifest"] = sha256_path(gate_manifest)
    report["manifest_sha256"] = canonical_digest(report)
    report_path = tmp_path / f"persisted-validation-step{step}.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint, report_path


def _cmc_report(
    step: int,
    *,
    raw_dynamic_mrr: float,
    fused_by_benchmark: dict[str, float],
    fused_regret: float,
) -> dict:
    report = _report(
        step,
        raw_dynamic_mrr,
        raw_dynamic_mrr - 0.60,
        0.80,
    )
    fused_macro = sum(fused_by_benchmark.values()) / len(fused_by_benchmark)
    report["stage4_method"] = "counterfactual_memory_calibration_v1"
    report["stage2_baseline"].update(
        {
            "static_macro_mrr": 0.60,
            "raw_dynamic_macro_mrr": 0.62,
            "raw_dynamic_regret": 0.04,
            "by_benchmark": {
                benchmark: {
                    "static_mrr": 0.60,
                    "raw_dynamic_mrr": 0.62,
                }
                for benchmark in BENCHMARKS
            },
        }
    )
    report["balanced_macro"].update(
        {
            "raw_dynamic_mrr": raw_dynamic_mrr,
            "fused_mrr": fused_macro,
            "fused_minus_static_mrr": fused_macro - 0.60,
            "fused_regret": fused_regret,
        }
    )
    report["by_benchmark"] = {
        benchmark: {
            "static_mrr": 0.60,
            "raw_dynamic_mrr": raw_dynamic_mrr,
            "fused_mrr": fused_by_benchmark[benchmark],
            "fused_minus_static_mrr": fused_by_benchmark[benchmark] - 0.60,
        }
        for benchmark in BENCHMARKS
    }
    report["cmc_fused"] = {
        "balanced_macro_mrr": fused_macro,
        "regret": fused_regret,
        "alpha_zero_exact": True,
        "alpha_one_exact": True,
        "zero_history_exact": True,
        "by_benchmark": {
            benchmark: {"mrr": fused_by_benchmark[benchmark]}
            for benchmark in BENCHMARKS
        },
    }
    report["gradient_health"] = {
        "finite": True,
        "nonzero": step > 0,
    }
    report["router_integrity"].update(
        {
            "full_router_digest_unchanged": True,
            "fast_router_digest_unchanged": True,
        }
    )
    return report


def _candidate_admission_report(
    step: int,
    *,
    final_mrr: float,
    admission_auprc: float,
    admission_prevalence: float = 0.25,
    no_regret_violation_rate: float = 0.0,
) -> dict:
    report = _report(step, 0.62, 0.02, 0.80)
    report["stage4_method"] = "candidate_admission_residual_v1"
    report["base_cmc_checkpoint_sha256"] = "a" * 64
    report["stage2_baseline"].update(
        {
            "static_macro_mrr": 0.60,
            "candidate_admission_step0_exact_static": True,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60}
                for benchmark in BENCHMARKS
            },
        }
    )
    report["balanced_macro"].update(
        {
            "static_mrr": 0.60,
            "candidate_admission_final_mrr": final_mrr,
            "candidate_admission_final_minus_static_mrr": final_mrr - 0.60,
        }
    )
    report["by_benchmark"] = {
        benchmark: {
            "static_mrr": 0.60,
            "candidate_admission_final_mrr": final_mrr,
            "candidate_admission_final_minus_static_mrr": final_mrr - 0.60,
        }
        for benchmark in BENCHMARKS
    }
    report["candidate_admission"] = {
        "balanced_macro_mrr": final_mrr,
        "static_balanced_macro_mrr": 0.60,
        "admission_auprc": admission_auprc,
        "admission_prevalence": admission_prevalence,
        "dynamic_extra_positive_count": 4,
        "no_regret_margin_violation_rate": no_regret_violation_rate,
        "no_regret_tolerance": 0.01,
        "step0_exact_static": step == 0,
    }
    report["gradient_health"] = {
        "finite": True,
        "nonzero": step > 0,
    }
    report["router_integrity"].update(
        {
            "full_router_digest_unchanged": True,
            "fast_router_digest_unchanged": True,
        }
    )
    return report


def test_candidate_admission_selection_applies_signal_and_no_regret_constraints(
    tmp_path: Path,
) -> None:
    high_mrr_bad_signal = _candidate_admission_report(
        300,
        final_mrr=0.66,
        admission_auprc=0.20,
    )
    lower_mrr_valid = _candidate_admission_report(
        600,
        final_mrr=0.64,
        admission_auprc=0.50,
    )
    persisted = [
        _persist_report(tmp_path, report)
        for report in (high_mrr_bad_signal, lower_mrr_valid)
    ]

    selected = select_stage4_candidate_admission_checkpoint(
        [high_mrr_bad_signal, lower_mrr_valid],
        checkpoint_paths={300: persisted[0][0], 600: persisted[1][0]},
        validation_report_paths={300: persisted[0][1], 600: persisted[1][1]},
        terminal_step=600,
    )

    assert selected["selected_step"] == 600
    assert selected["promotion_status"] == "promoted"
    assert selected["candidate_admission_selection"]["admission_auprc"] == 0.50

    dynamic_path = tmp_path / "candidate-admission-dynamic.json"
    final_path = tmp_path / "candidate-admission-selection.json"
    write_stage4_dynamic_selection(dynamic_path, selection=selected)
    finalized = finalize_stage4_selection(
        dynamic_selection_path=dynamic_path,
        output_path=final_path,
    )
    assert finalized["stage4_method"] == "candidate_admission_residual_v1"
    assert finalized["reliability"] == {
        "mode": "candidate_admission_residual",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "safe_memory_residual_bound": 2.0,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
        "base_cmc_checkpoint_sha256": finalized["reliability"][
            "base_cmc_checkpoint_sha256"
        ],
        "deployed_reliability_source": (
            "candidate_admission_constrained_residual"
        ),
        "reliability_sha256": finalized["reliability"]["reliability_sha256"],
    }


def test_cmc_selection_uses_deployed_fused_ranking_not_raw_dynamic(
    tmp_path: Path,
) -> None:
    step0 = _cmc_report(
        0,
        raw_dynamic_mrr=0.62,
        fused_by_benchmark={benchmark: 0.60 for benchmark in BENCHMARKS},
        fused_regret=0.0,
    )
    raw_dynamic_winner = _cmc_report(
        400,
        raw_dynamic_mrr=0.70,
        fused_by_benchmark={
            "toolbench_g3": 0.58,
            "traject_bench": 0.60,
            "alfworld": 0.60,
            "webshop": 0.60,
        },
        fused_regret=0.02,
    )
    deployed_fused_winner = _cmc_report(
        800,
        raw_dynamic_mrr=0.64,
        fused_by_benchmark={
            "toolbench_g3": 0.591,
            "traject_bench": 0.611,
            "alfworld": 0.611,
            "webshop": 0.611,
        },
        fused_regret=0.01,
    )
    reports = [step0, raw_dynamic_winner, deployed_fused_winner]
    persisted = [_persist_report(tmp_path, report) for report in reports]

    selected = select_stage4_cmc_checkpoint(
        reports,
        checkpoint_paths={
            report["step"]: artifact[0]
            for report, artifact in zip(reports, persisted)
        },
        validation_report_paths={
            report["step"]: artifact[1]
            for report, artifact in zip(reports, persisted)
        },
    )

    assert selected["selected_step"] == 800
    assert selected["release_status"] == "ok"
    assert selected["promotion_status"] == "promoted"
    assert selected["cmc_fused_selection"]["balanced_macro_mrr"] == pytest.approx(
        0.606
    )
    assert all(selected["release_checks"].values())


def test_cmc_step_zero_is_exact_static_control_and_not_promoted(
    tmp_path: Path,
) -> None:
    step0 = _cmc_report(
        0,
        raw_dynamic_mrr=0.62,
        fused_by_benchmark={benchmark: 0.60 for benchmark in BENCHMARKS},
        fused_regret=0.0,
    )
    checkpoint, report_path = _persist_report(tmp_path, step0)

    selected = select_stage4_cmc_checkpoint(
        [step0],
        checkpoint_paths={0: checkpoint},
        validation_report_paths={0: report_path},
    )

    assert selected["selected_step"] == 0
    assert selected["release_status"] == "action_required"
    assert selected["promotion_status"] == "stage4_not_promoted"
    assert selected["release_checks"]["fused_macro_improvement"] is False


def test_cmc_selection_breaks_fused_ties_by_regret_then_earlier_step(
    tmp_path: Path,
) -> None:
    reports = [
        _cmc_report(
            800,
            raw_dynamic_mrr=0.64,
            fused_by_benchmark={benchmark: 0.606 for benchmark in BENCHMARKS},
            fused_regret=0.012,
        ),
        _cmc_report(
            1200,
            raw_dynamic_mrr=0.63,
            fused_by_benchmark={benchmark: 0.606 for benchmark in BENCHMARKS},
            fused_regret=0.008,
        ),
        _cmc_report(
            1600,
            raw_dynamic_mrr=0.65,
            fused_by_benchmark={benchmark: 0.606 for benchmark in BENCHMARKS},
            fused_regret=0.008,
        ),
    ]
    persisted = [_persist_report(tmp_path, report) for report in reports]

    selected = select_stage4_cmc_checkpoint(
        reports,
        checkpoint_paths={
            report["step"]: artifact[0]
            for report, artifact in zip(reports, persisted)
        },
        validation_report_paths={
            report["step"]: artifact[1]
            for report, artifact in zip(reports, persisted)
        },
    )

    assert selected["selected_step"] == 1200


def test_cmc_selection_accepts_an_explicit_off_interval_terminal_step(
    tmp_path: Path,
) -> None:
    reports = [
        _cmc_report(
            0,
            raw_dynamic_mrr=0.62,
            fused_by_benchmark={benchmark: 0.60 for benchmark in BENCHMARKS},
            fused_regret=0.0,
        ),
        _cmc_report(
            2800,
            raw_dynamic_mrr=0.64,
            fused_by_benchmark={benchmark: 0.606 for benchmark in BENCHMARKS},
            fused_regret=0.01,
        ),
        _cmc_report(
            3000,
            raw_dynamic_mrr=0.65,
            fused_by_benchmark={benchmark: 0.608 for benchmark in BENCHMARKS},
            fused_regret=0.009,
        ),
    ]
    persisted = [_persist_report(tmp_path, report) for report in reports]

    selected = select_stage4_cmc_checkpoint(
        reports,
        checkpoint_paths={
            report["step"]: artifact[0]
            for report, artifact in zip(reports, persisted)
        },
        validation_report_paths={
            report["step"]: artifact[1]
            for report, artifact in zip(reports, persisted)
        },
        terminal_step=3000,
    )

    assert selected["selected_step"] == 3000
    assert selected["release_status"] == "ok"


def test_finalize_cmc_selection_keeps_checkpoint_native_candidate_gate(
    tmp_path: Path,
) -> None:
    report = _cmc_report(
        800,
        raw_dynamic_mrr=0.64,
        fused_by_benchmark={
            "toolbench_g3": 0.591,
            "traject_bench": 0.611,
            "alfworld": 0.611,
            "webshop": 0.611,
        },
        fused_regret=0.01,
    )
    checkpoint, report_path = _persist_report(tmp_path, report)
    dynamic = select_stage4_cmc_checkpoint(
        [report],
        checkpoint_paths={800: checkpoint},
        validation_report_paths={800: report_path},
    )
    dynamic_path = tmp_path / "stage4_dynamic_selection.json"
    write_stage4_dynamic_selection(dynamic_path, selection=dynamic)

    finalized = finalize_stage4_selection(
        dynamic_selection_path=dynamic_path,
        output_path=tmp_path / "stage4_selection.json",
    )

    assert finalized["release_status"] == "ok"
    assert finalized["stage4_method"] == "counterfactual_memory_calibration_v1"
    assert finalized["promotion_status"] == "promoted"
    assert finalized["reliability"]["mode"] == "cmc_candidate_gate"
    assert finalized["reliability"]["gate_checkpoint"] is None
    assert finalized["reliability_validation"] == dynamic["cmc_fused_selection"]


def test_dynamic_selection_prefers_best_validation_not_final_step(
    tmp_path: Path,
) -> None:
    reports = [
        _report(400, 0.61, 0.04, 0.72),
        _report(800, 0.68, 0.07, 0.79),
        _report(1200, 0.66, 0.08, 0.81),
    ]
    checkpoint_paths = {}
    validation_report_paths = {}
    for report in reports:
        checkpoint = tmp_path / f"stage4-step{report['step']}.pt"
        checkpoint.write_bytes(str(report["step"]).encode("utf-8"))
        checkpoint_paths[report["step"]] = checkpoint
        route_records_path = (
            tmp_path / f"validation-step{report['step']}.route_records.jsonl"
        )
        route_records_path.write_text(
            '{"row_digest":"row-a"}\n',
            encoding="utf-8",
        )
        report["validation_route_records"] = {
            **sha256_path(route_records_path),
            "record_count": 1,
            "row_digest_sha256": canonical_digest(["row-a"]),
        }
        gate_records_path = (
            tmp_path / f"validation-step{report['step']}.gate_records.jsonl"
        )
        gate_records_path.write_text(
            '{"row_digest":"gate-row-a"}\n',
            encoding="utf-8",
        )
        report["gate_route_records"] = {
            **sha256_path(gate_records_path),
            "record_count": 1,
            "row_digest_sha256": canonical_digest(["gate-row-a"]),
        }
        gate_manifest_path = (
            tmp_path / f"validation-step{report['step']}.gate_manifest.json"
        )
        gate_manifest_path.write_text("{}", encoding="utf-8")
        report["gate_route_manifest"] = sha256_path(gate_manifest_path)
        report["manifest_sha256"] = canonical_digest(report)
        report_path = tmp_path / f"validation-step{report['step']}.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True),
            encoding="utf-8",
        )
        validation_report_paths[report["step"]] = report_path

    selected = select_stage4_dynamic_checkpoint(
        reports,
        checkpoint_paths=checkpoint_paths,
        validation_report_paths=validation_report_paths,
    )

    assert selected["selected_step"] == 800
    assert selected["selected_checkpoint_path"] == str(
        checkpoint_paths[800].resolve()
    )
    assert selected["selected_validation_report_path"] == str(
        validation_report_paths[800].resolve()
    )
    assert selected["release_status"] == "ok"
    assert choose_fixed_alpha(reports[1])["fixed_alpha"] == pytest.approx(0.5)


def test_stage4_selection_prefers_jointly_trained_safe_fused_route(
    tmp_path: Path,
) -> None:
    raw_dynamic_winner = _report(800, 0.70, 0.08, 0.80)
    safe_fused_winner = _report(1200, 0.68, 0.07, 0.79)
    raw_dynamic_winner["safe_fused"] = _alpha_metrics(0.69)
    raw_dynamic_winner["balanced_macro"]["safe_fused_mrr"] = 0.69
    safe_fused_winner["safe_fused"] = _alpha_metrics(0.72)
    safe_fused_winner["balanced_macro"]["safe_fused_mrr"] = 0.72
    persisted = [
        _persist_report(tmp_path, raw_dynamic_winner),
        _persist_report(tmp_path, safe_fused_winner),
    ]

    selected = select_stage4_dynamic_checkpoint(
        [raw_dynamic_winner, safe_fused_winner],
        checkpoint_paths={800: persisted[0][0], 1200: persisted[1][0]},
        validation_report_paths={800: persisted[0][1], 1200: persisted[1][1]},
    )

    assert selected["selected_step"] == 1200
    assert selected["safe_fused_selection"]["balanced_macro_mrr"] == pytest.approx(0.72)


def test_stage4_selection_prefers_release_safe_checkpoint_before_mrr(
    tmp_path: Path,
) -> None:
    unsafe = _report(800, 0.72, 0.08, 0.80)
    safe = _report(1200, 0.69, 0.07, 0.79)
    unsafe["safe_fused"] = _alpha_metrics(0.74)
    unsafe["balanced_macro"]["safe_fused_mrr"] = 0.74
    unsafe["release_status"] = "action_required"
    safe["safe_fused"] = _alpha_metrics(0.71)
    safe["balanced_macro"]["safe_fused_mrr"] = 0.71
    persisted = [_persist_report(tmp_path, unsafe), _persist_report(tmp_path, safe)]

    selected = select_stage4_dynamic_checkpoint(
        [unsafe, safe],
        checkpoint_paths={800: persisted[0][0], 1200: persisted[1][0]},
        validation_report_paths={800: persisted[0][1], 1200: persisted[1][1]},
    )

    assert selected["selected_step"] == 1200
    assert selected["release_status"] == "ok"


def test_static_baseline_batches_require_bitwise_equality() -> None:
    baseline = [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]
    assert validate_static_baseline_batches(
        baseline,
        [baseline[0].clone()],
    )["status"] == "ok"
    with pytest.raises(ValueError, match="static validation logits drifted"):
        validate_static_baseline_batches(
            baseline,
            [torch.tensor([[1.0, 2.001]])],
        )


def test_dynamic_selection_manifest_is_self_hashed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage4-step800.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = _report(800, 0.68, 0.07, 0.79)
    route_records_path = tmp_path / "validation-step800.route_records.jsonl"
    route_records_path.write_text(
        '{"row_digest":"row-a"}\n',
        encoding="utf-8",
    )
    report["validation_route_records"] = {
        **sha256_path(route_records_path),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["row-a"]),
    }
    gate_records_path = tmp_path / "validation-step800.gate_records.jsonl"
    gate_records_path.write_text(
        '{"row_digest":"gate-row-a"}\n',
        encoding="utf-8",
    )
    report["gate_route_records"] = {
        **sha256_path(gate_records_path),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["gate-row-a"]),
    }
    gate_manifest_path = tmp_path / "validation-step800.gate_manifest.json"
    gate_manifest_path.write_text("{}", encoding="utf-8")
    report["gate_route_manifest"] = sha256_path(gate_manifest_path)
    report["manifest_sha256"] = canonical_digest(report)
    report_path = tmp_path / "validation-step800.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    selection = select_stage4_dynamic_checkpoint(
        [report],
        checkpoint_paths={800: checkpoint},
        validation_report_paths={800: report_path},
    )
    output = tmp_path / "stage4_dynamic_selection.json"
    written = write_stage4_dynamic_selection(output, selection=selection)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == written
    assert loaded["manifest_sha256"]
    assert loaded["selected_checkpoint_sha256"]
    assert loaded["selected_validation_report_sha256"]
    assert loaded["router_integrity"]["static_logits_exact"] is True
    assert loaded["fixed_alpha_selection"]["fixed_alpha"] == pytest.approx(0.5)
    assert loaded["validation_route_records"]["sha256"] == sha256_path(
        route_records_path
    )["sha256"]
    assert loaded["gate_route_records"]["sha256"] == sha256_path(
        gate_records_path
    )["sha256"]
    assert loaded["gate_route_manifest"]["sha256"] == sha256_path(
        gate_manifest_path
    )["sha256"]


def test_dynamic_selection_uses_earliest_step_as_final_tie_break(
    tmp_path: Path,
) -> None:
    reports = [
        _report(400, 0.68, 0.07, 0.79),
        _report(800, 0.68, 0.07, 0.79),
    ]
    persisted = [_persist_report(tmp_path, report) for report in reports]
    selected = select_stage4_dynamic_checkpoint(
        reports,
        checkpoint_paths={
            step: pair[0]
            for step, pair in zip((400, 800), persisted)
        },
        validation_report_paths={
            step: pair[1]
            for step, pair in zip((400, 800), persisted)
        },
    )
    assert selected["selected_step"] == 400


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("static_drift", "exact static parity"),
        ("missing_benchmark", "no mechanically eligible"),
        ("nonfinite", "must be finite"),
        ("oversized_final_k", "final_k must not exceed static_k"),
    ],
)
def test_dynamic_selection_rejects_invalid_validation_contract(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    if case == "static_drift":
        report["router_integrity"]["max_abs_static_logit_difference"] = 0.001
    elif case == "missing_benchmark":
        report["by_benchmark"].pop("webshop")
    elif case == "nonfinite":
        report["balanced_macro"]["dynamic_mrr"] = float("nan")
    elif case == "oversized_final_k":
        report["candidate_union"]["final_k"] = 501
    checkpoint, report_path = _persist_report(tmp_path, report)
    with pytest.raises(ValueError, match=message):
        select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        )


def test_dynamic_selection_writer_rejects_checkpoint_digest_drift(
    tmp_path: Path,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    selection = select_stage4_dynamic_checkpoint(
        [report],
        checkpoint_paths={800: checkpoint},
        validation_report_paths={800: report_path},
    )
    checkpoint.write_bytes(b"mutated-after-selection")
    with pytest.raises(
        ValueError,
        match="selected Stage4 checkpoint SHA-256 mismatch",
    ):
        write_stage4_dynamic_selection(
            tmp_path / "stage4_dynamic_selection.json",
            selection=selection,
        )


def test_finalize_stage4_selection_releases_joint_causal_gate(
    tmp_path: Path,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    dynamic_selection_path = tmp_path / "stage4_dynamic_selection.json"
    write_stage4_dynamic_selection(
        dynamic_selection_path,
        selection=select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        ),
    )

    output_path = tmp_path / "stage4_selection.json"
    finalized = finalize_stage4_selection(
        dynamic_selection_path=dynamic_selection_path,
        output_path=output_path,
    )

    assert finalized["schema_version"] == "stage4_selection_v1"
    assert finalized["status"] == "ok"
    assert finalized["release_status"] == "ok"
    assert all(finalized["release_checks"].values())
    assert finalized["selected_checkpoint_path"] == str(checkpoint.resolve())
    assert finalized["reliability"]["mode"] == "causal_gate"
    assert finalized["reliability"]["fixed_alpha"] is None
    assert finalized["reliability"]["gate_checkpoint"] is None
    assert finalized["reliability_validation"] == finalized["safe_fused_selection"]
    reliability_without_hash = dict(finalized["reliability"])
    recorded_reliability_hash = reliability_without_hash.pop(
        "reliability_sha256"
    )
    assert recorded_reliability_hash == canonical_digest(reliability_without_hash)
    payload_without_hash = dict(finalized)
    recorded_manifest_hash = payload_without_hash.pop("manifest_sha256")
    assert recorded_manifest_hash == canonical_digest(payload_without_hash)
    assert json.loads(output_path.read_text(encoding="utf-8")) == finalized


def test_finalize_stage4_selection_rejects_obsolete_posthoc_gate_report(
    tmp_path: Path,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    dynamic_selection_path = tmp_path / "stage4_dynamic_selection.json"
    write_stage4_dynamic_selection(
        dynamic_selection_path,
        selection=select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        ),
    )
    gate_report = tmp_path / "gate_report.json"
    gate_payload = {
        "schema_version": "memory_utility_gate_report_v1",
        "status": "not_recommended",
        "promoted": False,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
    }
    gate_payload["manifest_sha256"] = canonical_digest(gate_payload)
    gate_report.write_text(json.dumps(gate_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot override"):
        finalize_stage4_selection(
            dynamic_selection_path=dynamic_selection_path,
            gate_report_path=gate_report,
            output_path=tmp_path / "stage4_selection.json",
        )


def test_finalize_stage4_selection_cli_writes_manifest(
    tmp_path: Path,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    dynamic_selection_path = tmp_path / "stage4_dynamic_selection.json"
    write_stage4_dynamic_selection(
        dynamic_selection_path,
        selection=select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        ),
    )
    output_path = tmp_path / "stage4_selection.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/finalize_clstr_stage4_selection.py",
            "--dynamic_selection_path",
            str(dynamic_selection_path),
            "--output_path",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "release_status"
    ] == "ok"


def test_finalize_stage4_selection_does_not_promote_audit_bound_posthoc_gate(
    tmp_path: Path,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    dynamic_selection_path = tmp_path / "stage4_dynamic_selection.json"
    dynamic = write_stage4_dynamic_selection(
        dynamic_selection_path,
        selection=select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        ),
    )
    audit_sha = "audit-manifest-sha256"
    gate = MemoryUtilityGate()
    gate_checkpoint = tmp_path / "memory_utility_gate.pt"
    learned_summary = {
        "balanced_macro_mrr": 0.70,
        "zero_history_exact": True,
        "by_benchmark": {
            benchmark: {"mrr": 0.70} for benchmark in BENCHMARKS
        },
    }
    torch.save(
        {
            "stage": "clstr_memory_utility_gate",
            "schema_version": "memory_utility_gate_checkpoint_v1",
            "feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
            "feature_names": list(MEMORY_UTILITY_FEATURE_NAMES),
            "audit_manifest_sha256": audit_sha,
            "zero_history_fallback": "exact_static",
            "reliability_changes_memory_state": False,
            "selected_stage4_checkpoint_sha256": dynamic[
                "selected_checkpoint_sha256"
            ],
            "training_seed": 17,
            "feature_update_count_cap": 4.0,
            "feature_candidate_count_cap": 64.0,
            "memory_utility_gate_state_dict": gate.state_dict(),
            "validation": {
                "learned": learned_summary,
                "fixed_alpha": dynamic["fixed_alpha_selection"],
                "balanced_macro_mrr_improvement": 0.01,
            },
        },
        gate_checkpoint,
    )
    gate_report = {
        "schema_version": "memory_utility_gate_report_v1",
        "status": "ok",
        "promoted": True,
        "checkpoint_path": str(gate_checkpoint.resolve()),
        "checkpoint_sha256": sha256_path(gate_checkpoint)["sha256"],
        "audit_manifest_sha256": audit_sha,
        "selected_stage4_checkpoint_sha256": dynamic[
            "selected_checkpoint_sha256"
        ],
        "validation": {
            "learned": learned_summary,
            "fixed_alpha": dynamic["fixed_alpha_selection"],
            "balanced_macro_mrr_improvement": 0.01,
            "per_benchmark_nonregression": True,
        },
    }
    gate_report["manifest_sha256"] = canonical_digest(gate_report)
    gate_report_path = tmp_path / "gate_report.json"
    gate_report_path.write_text(
        json.dumps(gate_report, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot override"):
        finalize_stage4_selection(
            dynamic_selection_path=dynamic_selection_path,
            gate_report_path=gate_report_path,
            output_path=tmp_path / "stage4_selection.json",
        )
