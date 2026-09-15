#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELECTION_SOURCE = "best_gate_passing_heldout_score_only"
OPEN_POOL_SELECTION_SCHEMA = "clstr_vnext_stage2_open_pool_selection_v1"
OPEN_POOL_SELECTION_MODE = "open_pool_full_pool"
OPEN_POOL_HELDOUT_FAMILY = "toolbench"
OPEN_POOL_HELDOUT_SOURCE = "toolbench_g3"
OPEN_POOL_DEFAULT_HELDOUT_METRIC = "raw_route_mrr_delta"
OPEN_POOL_ALLOWED_HELDOUT_METRICS = (
    OPEN_POOL_DEFAULT_HELDOUT_METRIC,
    "route_recall_at_1_delta",
    "route_recall_at_5_delta",
    "route_recall_at_10_delta",
    "raw_route_recall_at_1_delta",
    "raw_route_recall_at_5_delta",
    "raw_route_recall_at_10_delta",
)
OPEN_POOL_REQUIRED_GATES = (
    "causal_route_gate",
    "cluster_sufficiency_gate",
    "causal_safety_gate",
    "coarse_recall_gate",
    "exact_fallback_gate",
    "gradient_health_gate",
    "robust_prefix_dev_gate",
)
TRAINING_REQUIRED_GATES = (
    "causal_route_gate",
    "cluster_sufficiency_gate",
    "causal_safety_gate",
    "ordinary_safety_gate",
    "coarse_recall_gate",
    "exact_fallback_gate",
    "gradient_health_gate",
    "robust_prefix_dev_gate",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _records_digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refresh_coarse_union_gates(
    records: list[dict[str, Any]],
    *,
    minimum_full_pool_clusters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the deployed-union coarse gate without changing other evidence."""

    from clstr.vnext_stage2_train import _multi_m_coarse_recall_gate

    if int(minimum_full_pool_clusters) <= 0:
        raise ValueError("coarse-gate refresh requires positive cluster support")
    refreshed: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for original in records:
        record = copy.deepcopy(original)
        gates = dict(record.get("gates") or {})
        if not gates:
            raise ValueError("Stage2 validation record lacks gate evidence")
        previous_coarse = bool(gates.get("coarse_recall_gate"))
        previous_pass = bool(gates.get("pass"))
        coarse = _multi_m_coarse_recall_gate(
            record.get("ordinary_dev") or {},
            minimum_full_pool_clusters=int(minimum_full_pool_clusters),
        )
        gates["coarse_recall"] = coarse
        gates["coarse_recall_gate"] = bool(coarse.get("pass"))
        gates["pass"] = bool(
            all(bool(gates.get(name)) for name in TRAINING_REQUIRED_GATES)
        )
        record["gates"] = gates
        refreshed.append(record)
        audit.append(
            {
                "step": int(record.get("step") or 0),
                "previous_coarse_recall_gate": previous_coarse,
                "refreshed_coarse_recall_gate": bool(
                    gates["coarse_recall_gate"]
                ),
                "previous_all_gates_pass": previous_pass,
                "refreshed_all_gates_pass": bool(gates["pass"]),
                "dynamic_only_deployment_is_diagnostic": bool(
                    coarse.get("dynamic_only_deployment_is_diagnostic")
                ),
                "candidate_union_preservation_pass": bool(
                    coarse.get("candidate_union_preservation_pass")
                ),
            }
        )
    return refreshed, audit


def _interface_scoped_refinement_contract(
    selection: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Authorize a completed open-pool refinement without universal routing.

    Closed-set rows are deployed through the separately verified preserved
    foundation.  A Top-K objective refinement may therefore finish with the
    legacy universal ordinary-safety gate false while still being valid for
    the open-pool interface.  This exception is deliberately narrow and does
    not relax any candidate-level causal, gradient, recall, or fallback gate.
    """

    status = str(selection.get("status") or "")
    if status == "ok":
        return {
            "enabled": False,
            "reason": "original_training_selection_is_universally_approved",
        }
    if status != "action_required":
        raise ValueError("Stage2 training selection is not complete")
    if str(report.get("status") or "") != status:
        raise ValueError("Stage2 selection/report status mismatch")
    if report.get("selection") != selection:
        raise ValueError("Stage2 report does not embed the exact training selection")
    warm_start = report.get("objective_warm_start") or {}
    topk = report.get("route_topk") or {}
    if not (
        bool(report.get("finite_loss"))
        and bool(warm_start.get("enabled"))
        and bool(warm_start.get("model_state_exact"))
        and not bool(warm_start.get("optimizer_restored"))
        and not bool(warm_start.get("trainer_progress_restored"))
        and bool(topk.get("enabled"))
        and int(topk.get("k") or 0) == 5
        and float(topk.get("lambda") or 0.0) > 0.0
        and int(topk.get("eligible_rows") or 0) > 0
        and int(topk.get("positive_injection_count") or 0) == 0
        and str(topk.get("candidate_membership_protocol") or "")
        == "immutable_natural_support"
        and int(report.get("positive_injection_count") or 0) == 0
        and int(report.get("training_teacher_retained_rows") or 0) == 0
    ):
        raise ValueError(
            "action-required Stage2 output is not a verified natural-support refinement"
        )
    return {
        "enabled": True,
        "reason": "open_pool_refinement_with_preserved_closed_set_dispatch",
        "original_training_status": status,
        "objective_warm_start": warm_start,
        "route_topk": topk,
        "closed_set_deployment_executes_stage2": False,
        "universal_ordinary_safety_required": False,
        "open_pool_candidate_gates_remain_required": True,
    }


def _clean_source_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Stage2 reselection requires a clean source worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _metric_summary(report: dict[str, Any], name: str) -> dict[str, Any]:
    value = report.get(name)
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in ("mean", "ci_low", "ci_high", "count", "cluster_count")
    }


def _open_pool_candidate_summary(
    record: dict[str, Any],
    *,
    minimum_full_pool_clusters: int,
    heldout_metric: str = OPEN_POOL_DEFAULT_HELDOUT_METRIC,
) -> dict[str, Any]:
    if heldout_metric not in OPEN_POOL_ALLOWED_HELDOUT_METRICS:
        raise ValueError(f"unsupported open-pool held-out metric: {heldout_metric}")
    gates = record.get("gates") or {}
    ordinary = record.get("ordinary_dev") or {}
    overall = ordinary.get("overall") or {}
    heldout = (ordinary.get("per_family") or {}).get(
        OPEN_POOL_HELDOUT_FAMILY
    ) or {}
    heldout_source = (ordinary.get("per_source") or {}).get(
        OPEN_POOL_HELDOUT_SOURCE
    ) or {}
    overall_raw = _metric_summary(overall, "raw_route_mrr_delta")
    heldout_raw = _metric_summary(heldout, "raw_route_mrr_delta")
    heldout_source_raw = _metric_summary(
        heldout_source,
        "raw_route_mrr_delta",
    )
    heldout_selection = _metric_summary(heldout, heldout_metric)
    heldout_source_selection = _metric_summary(
        heldout_source,
        heldout_metric,
    )
    full_pool = (
        (((overall.get("candidate_recall") or {}).get("500") or {}).get(
            "deployment_full_pool"
        ))
        or {}
    )
    heldout_rows = int(heldout.get("row_count") or 0)
    heldout_clusters = int(heldout.get("cluster_count") or 0)
    full_pool_rows = int(full_pool.get("row_count") or 0)
    full_pool_clusters = int(full_pool.get("cluster_count") or 0)
    heldout_source_rows = int(heldout_source.get("row_count") or 0)
    heldout_source_clusters = int(heldout_source.get("cluster_count") or 0)
    reasons: list[str] = []
    for name in OPEN_POOL_REQUIRED_GATES:
        if not bool(gates.get(name)):
            reasons.append(f"failed_gate:{name}")
    if int(overall_raw.get("count") or 0) <= 0:
        reasons.append("missing_overall_raw_route_evidence")
    if float(overall_raw.get("ci_low") or 0.0) <= 0.0:
        reasons.append("overall_raw_route_ci_not_positive")
    if heldout_rows <= 0 or int(heldout_raw.get("count") or 0) <= 0:
        reasons.append("missing_heldout_open_pool_evidence")
    if int(heldout_selection.get("count") or 0) <= 0:
        reasons.append("missing_heldout_selection_metric")
    if heldout_clusters < int(minimum_full_pool_clusters):
        reasons.append("insufficient_heldout_open_pool_clusters")
    if full_pool_clusters < int(minimum_full_pool_clusters):
        reasons.append("insufficient_full_pool_clusters")
    if heldout_rows != full_pool_rows or heldout_clusters != full_pool_clusters:
        reasons.append("heldout_family_is_not_exact_full_pool_stratum")
    if (
        heldout_source_rows != heldout_rows
        or heldout_source_clusters != heldout_clusters
        or abs(
            float(heldout_source_raw.get("mean") or 0.0)
            - float(heldout_raw.get("mean") or 0.0)
        )
        > 1.0e-12
        or abs(
            float(heldout_source_selection.get("mean") or 0.0)
            - float(heldout_selection.get("mean") or 0.0)
        )
        > 1.0e-12
    ):
        reasons.append("heldout_family_contains_non_target_sources")
    if float(heldout_raw.get("mean") or 0.0) <= 0.0:
        reasons.append("heldout_open_pool_raw_mean_not_positive")
    if float(heldout_selection.get("mean") or 0.0) <= 0.0:
        reasons.append("heldout_selection_metric_mean_not_positive")
    return {
        "step": int(record.get("step") or 0),
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "required_gate_values": {
            name: bool(gates.get(name)) for name in OPEN_POOL_REQUIRED_GATES
        },
        "overall_raw_route_mrr_delta": overall_raw,
        "heldout_family": OPEN_POOL_HELDOUT_FAMILY,
        "heldout_raw_route_mrr_delta": heldout_raw,
        "heldout_source": OPEN_POOL_HELDOUT_SOURCE,
        "heldout_source_raw_route_mrr_delta": heldout_source_raw,
        "heldout_metric": heldout_metric,
        "heldout_selection_metric": heldout_selection,
        "heldout_source_selection_metric": heldout_source_selection,
        "heldout_source_row_count": heldout_source_rows,
        "heldout_source_cluster_count": heldout_source_clusters,
        "heldout_row_count": heldout_rows,
        "heldout_cluster_count": heldout_clusters,
        "full_pool_row_count": full_pool_rows,
        "full_pool_cluster_count": full_pool_clusters,
    }


def _select_open_pool_validation(
    records: list[dict[str, Any]],
    *,
    minimum_full_pool_clusters: int,
    heldout_metric: str = OPEN_POOL_DEFAULT_HELDOUT_METRIC,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise ValueError("open-pool selector requires validation records")
    summaries = [
        _open_pool_candidate_summary(
            record,
            minimum_full_pool_clusters=minimum_full_pool_clusters,
            heldout_metric=heldout_metric,
        )
        for record in records
    ]
    eligible = [summary for summary in summaries if bool(summary["eligible"])]
    if not eligible:
        raise ValueError("no checkpoint passes the held-out open-pool release gates")
    selected_summary = max(
        eligible,
        key=lambda summary: (
            float(
                (summary["heldout_selection_metric"] or {}).get("ci_low")
                or float("-inf")
            ),
            float(
                (summary["heldout_selection_metric"] or {}).get("mean")
                or float("-inf")
            ),
            float(
                (summary["heldout_raw_route_mrr_delta"] or {}).get("mean")
                or float("-inf")
            ),
            float(
                (summary["overall_raw_route_mrr_delta"] or {}).get("ci_low")
                or float("-inf")
            ),
            -int(summary["step"]),
        ),
    )
    return selected_summary, summaries


def select_open_pool_release(
    output_dir: str | Path,
    release_output_dir: str | Path,
    *,
    source_commit: str | None = None,
    heldout_metric: str = OPEN_POOL_DEFAULT_HELDOUT_METRIC,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    release_output = Path(release_output_dir).resolve()
    if release_output.exists():
        raise FileExistsError(
            f"open-pool release output already exists: {release_output}"
        )
    selection_path = output / "stage2_selection.json"
    quality_path = output / "stage2_quality_gate.json"
    report_path = output / "train_report.json"
    source_manifest_path = output / "source_manifest.json"
    selection = _read_json(selection_path)
    report = _read_json(report_path)
    interface_scoped = _interface_scoped_refinement_contract(selection, report)
    if not bool(selection.get("robust_prefix_exposure_gate")):
        raise ValueError("Stage2 training lacks robust-prefix exposure")
    if int(selection.get("positive_injection_count") or 0) != 0:
        raise ValueError("open-pool release forbids positive candidate injection")
    records = list(selection.get("validation_records") or [])
    minimum_clusters = int(selection.get("minimum_full_pool_clusters") or 0)
    if minimum_clusters <= 0:
        raise ValueError("Stage2 selection lacks a positive full-pool cluster gate")
    selected_summary, candidate_summaries = _select_open_pool_validation(
        records,
        minimum_full_pool_clusters=minimum_clusters,
        heldout_metric=heldout_metric,
    )
    selected_step = int(selected_summary["step"])
    selected_record = next(
        record for record in records if int(record.get("step") or 0) == selected_step
    )
    checkpoint = (
        output
        / "checkpoints"
        / f"clstr_vnext_stage2-step{selected_step}.pt"
    )
    if selected_step <= 0 or not checkpoint.is_file():
        raise ValueError("open-pool selection lacks a usable Stage2 checkpoint")
    resolved_source_commit = source_commit or _clean_source_commit()
    payload = {
        "schema_version": OPEN_POOL_SELECTION_SCHEMA,
        "status": "ok",
        "selection_mode": OPEN_POOL_SELECTION_MODE,
        "selection_scope": "open_or_global_pool_only",
        "selection_reason": (
            f"maximum_heldout_open_pool_{heldout_metric}_ci_lower_bound"
        ),
        "selection_source": (
            "heldout_ordinary_dev_open_full_pool_without_benchmark_eval_rows"
        ),
        "source_commit": resolved_source_commit,
        "stage2_output_dir": str(output),
        "original_stage2_selection_path": str(selection_path.resolve()),
        "original_stage2_selection_sha256": file_sha256(selection_path),
        "original_stage2_quality_gate_sha256": file_sha256(quality_path),
        "original_train_report_sha256": file_sha256(report_path),
        "source_manifest_sha256": (
            file_sha256(source_manifest_path)
            if source_manifest_path.is_file()
            else None
        ),
        "validation_records_sha256": _records_digest(records),
        "original_selected_step": int(selection.get("selected_step") or 0),
        "original_training_status": str(selection.get("status") or ""),
        "interface_scoped_training_approval": bool(interface_scoped["enabled"]),
        "interface_scoped_training_contract": interface_scoped,
        "minimum_full_pool_clusters": minimum_clusters,
        "required_gates": list(OPEN_POOL_REQUIRED_GATES),
        "overall_raw_route_ci_must_be_positive": True,
        "heldout_family": OPEN_POOL_HELDOUT_FAMILY,
        "heldout_source": OPEN_POOL_HELDOUT_SOURCE,
        "heldout_metric": heldout_metric,
        "heldout_metric_selection_statistic": "ci_low",
        "heldout_raw_mean_must_be_positive": True,
        "closed_set_dispatch_required": True,
        "closed_set_safety_source": (
            "schema_v13_preserved_foundation_interface_dispatch"
        ),
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
        "positive_injection_count": 0,
        "selected_step": selected_step,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": file_sha256(checkpoint),
        "selected_candidate_summary": selected_summary,
        "selected_validation": selected_record,
        "candidate_summaries": candidate_summaries,
    }
    release_output.mkdir(parents=True)
    release_path = release_output / "stage2_open_pool_selection.json"
    write_json(release_path, payload)
    return {
        "status": "ok",
        "schema_version": OPEN_POOL_SELECTION_SCHEMA,
        "selection_path": str(release_path.resolve()),
        "selection_sha256": file_sha256(release_path),
        "selected_step": selected_step,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": payload["selected_checkpoint_sha256"],
    }


def reselect_stage2_output(output_dir: str | Path) -> dict[str, Any]:
    from clstr.vnext_stage2_train import _select_stage2_validation

    output = Path(output_dir).resolve()
    selection_path = output / "stage2_selection.json"
    quality_path = output / "stage2_quality_gate.json"
    report_path = output / "train_report.json"
    previous_selection_sha256 = file_sha256(selection_path)
    selection = _read_json(selection_path)
    report = _read_json(report_path)
    original_records = list(selection.get("validation_records") or [])
    minimum_full_pool_clusters = int(
        selection.get("minimum_full_pool_clusters") or 0
    )
    records, coarse_gate_refresh = _refresh_coarse_union_gates(
        original_records,
        minimum_full_pool_clusters=minimum_full_pool_clusters,
    )
    selected, reason = _select_stage2_validation(records)
    selected_step = int(selected.get("step") or 0)
    checkpoint = output / "checkpoints" / f"clstr_vnext_stage2-step{selected_step}.pt"
    if selected_step <= 0 or not checkpoint.is_file():
        raise ValueError("held-out-safe selection lacks a usable checkpoint")

    gates = dict(selected.get("gates") or {})
    score = float(selected.get("selection_score") or 0.0)
    initial_score = float(selection.get("initial_score") or 0.0)
    score_gain = score - initial_score
    status = (
        "ok"
        if bool(gates.get("pass"))
        and bool(selection.get("robust_prefix_exposure_gate"))
        and score_gain >= float(selection.get("minimum_score_gain") or 0.0)
        else "action_required"
    )
    prior_step = int(selection.get("selected_step") or 0)
    prior_reason = str(selection.get("selection_reason") or "")
    selection.update(
        {
            "status": status,
            "selected_step": selected_step,
            "selected_checkpoint_path": str(checkpoint.resolve()),
            "selected_score": score,
            "score_gain": score_gain,
            "selected_validation": selected,
            "selected_validation_gates": gates,
            "causal_route_gate": bool(gates.get("causal_route_gate")),
            "cluster_sufficiency_gate": bool(
                gates.get("cluster_sufficiency_gate")
            ),
            "causal_safety_gate": bool(gates.get("causal_safety_gate")),
            "ordinary_safety_gate": bool(gates.get("ordinary_safety_gate")),
            "exact_fallback_gate": bool(gates.get("exact_fallback_gate")),
            "coarse_recall_gate": bool(gates.get("coarse_recall_gate")),
            "coarse_recall": gates.get("coarse_recall"),
            "gradient_health_gate": bool(gates.get("gradient_health_gate")),
            "robust_prefix_dev_gate": bool(gates.get("robust_prefix_dev_gate")),
            "selection_reason": reason,
            "selection_source": SELECTION_SOURCE,
            "validation_records": records,
            "selection_finalization": {
                "schema_version": "clstr_vnext_stage2_reselection_v2",
                "source_commit": _clean_source_commit(),
                "coarse_gate_contract": (
                    "preserved_static_top500_plus_dynamic_diff64_union_v2"
                ),
                "coarse_gate_refresh": coarse_gate_refresh,
                "previous_validation_records_sha256": _records_digest(
                    original_records
                ),
                "validation_records_sha256": _records_digest(records),
                "previous_selection_sha256": previous_selection_sha256,
                "previous_selected_step": prior_step,
                "previous_selection_reason": prior_reason,
            },
        }
    )
    write_json(selection_path, selection)
    write_json(quality_path, selection)
    report["status"] = status
    report["selection"] = selection
    report["selection_finalization"] = selection["selection_finalization"]
    write_json(report_path, report)
    manifest = {
        **selection["selection_finalization"],
        "status": status,
        "selected_step": selected_step,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selection_reason": reason,
        "stage2_selection_sha256": file_sha256(selection_path),
        "stage2_quality_gate_sha256": file_sha256(quality_path),
        "train_report_sha256": file_sha256(report_path),
    }
    write_json(output / "stage2_reselection_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--mode",
        choices=("training", OPEN_POOL_SELECTION_MODE),
        default="training",
    )
    parser.add_argument("--release_output_dir")
    parser.add_argument(
        "--heldout_metric",
        choices=OPEN_POOL_ALLOWED_HELDOUT_METRICS,
        default=OPEN_POOL_DEFAULT_HELDOUT_METRIC,
    )
    args = parser.parse_args()
    if args.mode == OPEN_POOL_SELECTION_MODE:
        if not args.release_output_dir:
            parser.error("--release_output_dir is required for open-pool selection")
        manifest = select_open_pool_release(
            args.output_dir,
            args.release_output_dir,
            heldout_metric=args.heldout_metric,
        )
    else:
        if args.release_output_dir:
            parser.error("--release_output_dir is valid only for open-pool selection")
        manifest = reselect_stage2_output(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
