from __future__ import annotations

import math
from typing import Any


BENCHMARKS = ("global", "toolbench_g3", "traject_bench")
FLOAT_TOLERANCE = 1.0e-12
RELEASE_FLOORS = {
    ("global", "next_recall@500"): 0.90,
    ("toolbench_g3", "next_recall@500"): 0.84,
    ("traject_bench", "next_recall@500"): 0.70,
    ("traject_bench", "next_recall@200"): 0.52,
}


def handoff_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    mode = dict((report.get("query_modes") or {}).get("checkpoint_state_query") or {})
    benchmarks = dict(mode.get("benchmarks") or {})
    return {
        "global": dict(mode.get("global") or {}),
        "toolbench_g3": dict(benchmarks.get("toolbench_g3") or {}),
        "traject_bench": dict(benchmarks.get("traject_bench") or {}),
    }


def _metric(metrics: dict[str, dict[str, float]], benchmark: str, key: str) -> float:
    try:
        value = float(metrics[benchmark][key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing handoff metric: {benchmark}.{key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite handoff metric: {benchmark}.{key}")
    return value


def primary_composite(metrics: dict[str, dict[str, float]]) -> float:
    return sum(_metric(metrics, name, "next_recall@100") for name in BENCHMARKS) / 3.0


def _check_report(checks: dict[str, bool], *, ok_status: str = "ok") -> dict[str, Any]:
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    return {
        "status": ok_status if not failed else "action_required",
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": failed,
    }


def audit_stage0_smoke(
    train_report: dict[str, Any],
    lineage_report: dict[str, Any],
) -> dict[str, Any]:
    metrics = dict(train_report.get("metrics") or {})
    protocol = dict(train_report.get("stage0_protocol") or train_report.get("protocol_metadata") or {})
    trainable = dict(train_report.get("trainable_parameter_policy") or {})
    optimizer_names = [str(name) for name in trainable.get("optimizer_parameter_names") or []]
    try:
        loss = float(metrics.get("loss"))
    except (TypeError, ValueError):
        loss = float("nan")
    lineage_ok = bool(
        lineage_report.get("status") == "ok"
        or (
            lineage_report.get("schema_version") == "qwen06_clstr_lineage_v1"
            and lineage_report.get("stage") == "stage0"
        )
    )
    checks = {
        "train_status_ok": train_report.get("status") == "ok",
        "lineage_status_ok": lineage_ok,
        "finite_loss": math.isfinite(loss),
        "mined_hard_negative_rows_positive": float(metrics.get("mined_hard_negative_rows") or 0.0) > 0.0,
        "mined_hard_negative_pairs_positive": float(
            metrics.get("mined_hard_negative_pair_count") or 0.0
        )
        > 0.0,
        "unified_memory_route": protocol.get("route_scorer") == "unified_memory",
        "causal_prompt_version": protocol.get("state_query_prompt_version") == "clstr_causal_state_v1",
        "causal_prompt_max_chars": protocol.get("state_query_max_chars") == 2000,
        "causal_prompt_truncation": protocol.get("state_query_truncation") == "head_tail_v1",
        "encoder_backbone_frozen": trainable.get("trains_encoder_backbone") is False,
        "initial_belief_optimized": any("initial_belief_head" in name for name in optimizer_names),
        "unified_retriever_optimized": any("unified_retriever" in name for name in optimizer_names),
        "encoder_backbone_not_optimized": not any("encoder.backbone" in name for name in optimizer_names),
    }
    return {
        **_check_report(checks),
        "gate": "stage0_smoke",
        "loss": loss,
        "optimizer_parameter_names": optimizer_names,
    }


def _delta(
    current: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    benchmark: str,
    key: str,
) -> float:
    return _metric(current, benchmark, key) - _metric(reference, benchmark, key)


def audit_stage0_promotion(
    *,
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    previous_report: dict[str, Any],
    target_step: int,
) -> dict[str, Any]:
    target_step = int(target_step)
    baseline = handoff_metrics(baseline_report)
    current = handoff_metrics(current_report)
    previous = handoff_metrics(previous_report)
    if target_step == 1200:
        checks = {
            "global_next100_gain": _delta(current, baseline, "global", "next_recall@100") > 0.0,
            "global_next500_gain": _delta(current, baseline, "global", "next_recall@500") > 0.0,
            "toolbench_next500_safe": _delta(
                current, baseline, "toolbench_g3", "next_recall@500"
            )
            >= -0.01 - FLOAT_TOLERANCE,
            "traject_next500_safe": _delta(
                current, baseline, "traject_bench", "next_recall@500"
            )
            >= -0.01 - FLOAT_TOLERANCE,
            "hard_domain_joint_gain": (
                _delta(current, baseline, "toolbench_g3", "next_recall@100") > 0.0
                and _delta(current, baseline, "toolbench_g3", "next_recall@500") > 0.0
            )
            or (
                _delta(current, baseline, "traject_bench", "next_recall@100") > 0.0
                and _delta(current, baseline, "traject_bench", "next_recall@500") > 0.0
            ),
        }
        reference_name = "step0_baseline"
        reference = baseline
    elif target_step > 1200:
        composite_gain = primary_composite(current) - primary_composite(previous)
        hard_domain_gains = {
            benchmark: _delta(current, previous, benchmark, "next_recall@100")
            for benchmark in ("toolbench_g3", "traject_bench")
        }
        checks = {
            "composite_or_hard_domain_gain": composite_gain >= 0.005 - FLOAT_TOLERANCE
            or max(hard_domain_gains.values()) >= 0.01 - FLOAT_TOLERANCE,
            **{
                f"{benchmark}_next500_safe": _delta(
                    current, previous, benchmark, "next_recall@500"
                )
                >= -0.01 - FLOAT_TOLERANCE
                for benchmark in BENCHMARKS
            },
        }
        reference_name = "previous_segment"
        reference = previous
    else:
        raise ValueError("target_step must be 1200 or a later segment")
    deltas = {
        benchmark: {
            key: _delta(current, reference, benchmark, key)
            for key in ("next_recall@100", "next_recall@500")
        }
        for benchmark in BENCHMARKS
    }
    return {
        **_check_report(checks),
        "gate": "stage0_promotion",
        "target_step": target_step,
        "reference": reference_name,
        "primary_composite": primary_composite(current),
        "reference_primary_composite": primary_composite(reference),
        "deltas": deltas,
    }


def audit_stage0_release(
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    checkpoint_path: str,
) -> dict[str, Any]:
    baseline = handoff_metrics(baseline_report)
    current = handoff_metrics(current_report)
    floors: dict[str, Any] = {}
    misses: list[float] = []
    for (benchmark, key), floor in RELEASE_FLOORS.items():
        value = _metric(current, benchmark, key)
        miss = max(0.0, float(floor) - value)
        misses.append(miss)
        floors[f"{benchmark}.{key}"] = {
            "value": value,
            "floor": float(floor),
            "miss": miss,
            "passed": miss == 0.0,
        }
    max_miss = max(misses, default=0.0)
    if max_miss == 0.0:
        status = "ok"
    elif max_miss < 0.02:
        status = "borderline"
    else:
        status = "action_required"
    return {
        "status": status,
        "gate": "stage0_release",
        "checkpoint_path": str(checkpoint_path),
        "release_safe": status == "ok",
        "max_floor_miss": max_miss,
        "floors": floors,
        "primary_composite": primary_composite(current),
        "baseline_primary_composite": primary_composite(baseline),
        "primary_composite_gain": primary_composite(current) - primary_composite(baseline),
    }


def select_stage0_checkpoint(
    *,
    baseline_report: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    audited: list[dict[str, Any]] = []
    for candidate in candidates:
        release = audit_stage0_release(
            baseline_report,
            candidate["report"],
            str(candidate["checkpoint_path"]),
        )
        audited.append(
            {
                **candidate,
                "step": int(candidate["step"]),
                "release": release,
                "primary_composite": float(release["primary_composite"]),
            }
        )
    safe = [candidate for candidate in audited if candidate["release"]["status"] == "ok"]
    if not safe:
        return {
            "status": "action_required",
            "gate": "stage0_selection",
            "reason": "no_release_safe_checkpoint",
            "candidates": [
                {
                    "step": item["step"],
                    "checkpoint_path": str(item["checkpoint_path"]),
                    "release_status": item["release"]["status"],
                    "primary_composite": item["primary_composite"],
                }
                for item in audited
            ],
        }
    selected = sorted(safe, key=lambda item: (-item["primary_composite"], item["step"]))[0]
    return {
        "status": "ok",
        "gate": "stage0_selection",
        "selected_step": selected["step"],
        "selected_checkpoint_path": str(selected["checkpoint_path"]),
        "selected_report_path": (
            None if selected.get("report_path") is None else str(selected["report_path"])
        ),
        "selected_lineage_path": str(selected["lineage_manifest_path"]),
        "selected_primary_composite": selected["primary_composite"],
        "release_status": selected["release"]["status"],
        "candidate_count": len(audited),
        "release_safe_candidate_count": len(safe),
    }
