from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable


MATCHED_RELEASE_SELECTION_SCHEMA = (
    "clstr_vnext_stage2_matched_multibench_selection_v1"
)
MATCHED_RELEASE_SELECTION_MODE = "matched_multibench_unified_recurrent"
MATCHED_RELEASE_BOOTSTRAP_PROTOCOL = (
    "trajectory_cluster_percentile_bootstrap_seed29_samples2000_fsum12_v2"
)


def _canonical_metric_value(value: float) -> float:
    """Make release metrics byte-stable across supported Python runtimes."""

    return float(f"{float(value):.12g}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"required JSON artifact is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {resolved}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"required JSONL artifact is missing: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object: {resolved}:{line_number}"
                )
            rows.append(row)
    return rows


def _rank_value(rank: Any, *, k: int | None) -> float:
    if not isinstance(rank, (int, float)) or float(rank) <= 0.0:
        return 0.0
    if k is None:
        return 1.0 / float(rank)
    return float(float(rank) <= float(k))


def _comparison_rows(
    rows: Iterable[dict[str, Any]],
    comparison: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row.get("factual"), dict) or not isinstance(
            row.get(comparison),
            dict,
        ):
            continue
        if comparison == "mismatch" and row.get("mismatch_donor_source_index") is None:
            continue
        if comparison == "order_shuffle" and not bool(row.get("order_shuffle_eligible")):
            continue
        selected.append(row)
    return selected


def _paired_delta(
    rows: Iterable[dict[str, Any]],
    *,
    comparison: str,
    k: int | None,
) -> float:
    values = [
        _rank_value(row["factual"].get("end_to_end_route_rank"), k=k)
        - _rank_value(row[comparison].get("end_to_end_route_rank"), k=k)
        for row in rows
    ]
    return (
        _canonical_metric_value(math.fsum(values) / len(values))
        if values
        else 0.0
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * float(probability)))
    return _canonical_metric_value(
        sorted_values[max(0, min(index, len(sorted_values) - 1))]
    )


def paired_cluster_bootstrap(
    rows: Iterable[dict[str, Any]],
    *,
    comparison: str,
    seed: int = 29,
    samples: int = 2000,
) -> dict[str, Any]:
    selected = _comparison_rows(rows, comparison)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        cluster_id = str(row.get("trajectory_id") or "")
        if not cluster_id:
            raise ValueError("release evidence row lacks trajectory identity")
        clusters[cluster_id].append(row)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {
            "comparison": comparison,
            "eligible": False,
            "row_count": 0,
            "cluster_count": 0,
            "metrics": {},
        }
    random_state = random.Random(int(seed))
    metric_specs = (("mrr", None), ("recall_at_1", 1), ("recall_at_5", 5))
    bootstrap: dict[str, list[float]] = {name: [] for name, _ in metric_specs}
    for _ in range(int(samples)):
        sampled_rows: list[dict[str, Any]] = []
        for _ in cluster_ids:
            sampled_rows.extend(clusters[random_state.choice(cluster_ids)])
        for name, k in metric_specs:
            bootstrap[name].append(
                _paired_delta(sampled_rows, comparison=comparison, k=k)
            )
    metrics: dict[str, dict[str, Any]] = {}
    for name, k in metric_specs:
        values = sorted(bootstrap[name])
        metrics[name] = {
            "count": len(selected),
            "cluster_count": len(cluster_ids),
            "mean": _paired_delta(selected, comparison=comparison, k=k),
            "ci_low": _percentile(values, 0.025),
            "ci_high": _percentile(values, 0.975),
        }
    return {
        "comparison": comparison,
        "eligible": True,
        "row_count": len(selected),
        "cluster_count": len(cluster_ids),
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(samples),
        "bootstrap_protocol": MATCHED_RELEASE_BOOTSTRAP_PROTOCOL,
        "metrics": metrics,
    }


def _metric_summary(container: dict[str, Any], key: str) -> dict[str, Any]:
    summary = container.get(key)
    if not isinstance(summary, dict):
        raise ValueError(f"release evidence lacks metric summary: {key}")
    required = ("count", "cluster_count", "mean", "ci_low", "ci_high")
    if any(field not in summary for field in required):
        raise ValueError(f"release metric summary is incomplete: {key}")
    return {field: summary[field] for field in required}


def _validate_matched_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    manifest = _load_json(resolved)
    if (
        manifest.get("schema_version") != "clstr_matched_multibench_union_v1"
        or manifest.get("status") != "ok"
    ):
        raise ValueError("matched-union manifest is not approved")
    unsigned = dict(manifest)
    embedded = str(unsigned.pop("manifest_sha256", ""))
    if not embedded or embedded != json_digest(unsigned):
        raise ValueError("matched-union embedded digest differs")
    route_files = manifest.get("route_files") or {}
    for benchmark, expected_split in (("tau2", "dev"), ("toolsandbox", "dev")):
        entry = route_files.get(f"{benchmark}_{expected_split}") or {}
        route_path = Path(str(entry.get("path") or "")).resolve()
        if not route_path.is_file() or file_sha256(route_path) != str(
            entry.get("sha256") or ""
        ):
            raise ValueError(f"matched {benchmark}-dev route artifact differs")
    tau2_split = manifest.get("tau2_split_manifest") or {}
    if not bool(tau2_split.get("official_test_preserved")):
        raise ValueError("Tau2 official test split is not preserved")
    toolsandbox_split = manifest.get("toolsandbox_split_manifest") or {}
    if not toolsandbox_split.get("family_to_split"):
        raise ValueError("ToolSandbox grouped-family split is missing")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "embedded_manifest_sha256": embedded,
        "split_seed": str(manifest.get("split_seed") or ""),
        "tau2_split_manifest_sha256": str(tau2_split.get("manifest_sha256") or ""),
        "toolsandbox_split_manifest_sha256": str(
            toolsandbox_split.get("manifest_sha256") or ""
        ),
    }


def _validate_dev_report(
    report_path: str | Path,
    *,
    benchmark: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    training_skills_path: Path,
    training_skills_sha256: str,
    matched_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = Path(report_path).resolve()
    report = _load_json(resolved)
    if (
        report.get("schema_version") != "clstr_vnext_checkpoint_native_eval_v20"
        or report.get("status") != "ok"
        or report.get("blockers")
        or report.get("benchmark") != benchmark
        or report.get("method") != "clstr_vnext_stage2_checkpoint_native"
    ):
        raise ValueError(f"{benchmark} dev report is not an approved native evaluation")
    if report.get("evaluation_scope") != "stage2_complete_method":
        raise ValueError(f"{benchmark} dev report uses the wrong evaluation scope")
    if report.get("stage2_release_selection_contract") is not None:
        raise ValueError(f"{benchmark} dev evidence must precede release selection")
    requested = Path(str(report.get("requested_stage2_checkpoint_path") or "")).resolve()
    checkpoint = report.get("checkpoint") or {}
    if (
        requested != checkpoint_path
        or Path(str(checkpoint.get("checkpoint_path") or "")).resolve()
        != checkpoint_path
        or str(checkpoint.get("checkpoint_sha256") or "") != checkpoint_sha256
        or int(checkpoint.get("checkpoint_step") or 0) <= 0
    ):
        raise ValueError(f"{benchmark} dev report names a different checkpoint")
    if (
        Path(str(checkpoint.get("training_skills_path") or "")).resolve()
        != training_skills_path
        or str(checkpoint.get("training_skills_sha256") or "")
        != training_skills_sha256
    ):
        raise ValueError(f"{benchmark} dev report names different training skills")
    corpus = report.get("corpus") or {}
    corpus_contract = report.get("corpus_source_contract") or {}
    if (
        corpus.get("task_split") != "dev"
        or corpus_contract.get("task_split") != "dev"
        or Path(str(corpus.get("matched_prebuilt_manifest_path") or "")).resolve()
        != Path(matched_manifest["path"])
        or str(corpus_contract.get("matched_union_manifest_sha256") or "")
        != str(matched_manifest["embedded_manifest_sha256"])
    ):
        raise ValueError(f"{benchmark} report is not immutable held-out dev evidence")
    if not bool(report.get("non_static_memory_metrics_release_eligible")):
        raise ValueError(f"{benchmark} recurrent metrics are not release-eligible")
    if list(report.get("release_eligible_metrics") or []) != [
        "factual.end_to_end_route",
        "factual.full_pool_recall",
    ]:
        raise ValueError(f"{benchmark} release-eligible metric contract differs")
    history = ((report.get("rows") or {}).get("history_channel") or {})
    if history.get("status") != "ok" or int(history.get("leaked_row_count") or 0):
        raise ValueError(f"{benchmark} dev history channel is not leakage-safe")
    records_path = Path(str(report.get("route_records_path") or "")).resolve()
    protocol_path = Path(str(report.get("protocol_manifest_path") or "")).resolve()
    records = _load_jsonl(records_path)
    if len(records) != int((report.get("rows") or {}).get("routing_row_count") or 0):
        raise ValueError(f"{benchmark} dev route-record count differs")
    if any(str(row.get("benchmark") or "") != benchmark for row in records):
        raise ValueError(f"{benchmark} dev records contain another benchmark")
    return {
        "report_path": str(resolved),
        "report_sha256": file_sha256(resolved),
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "route_records_path": str(records_path),
        "route_records_sha256": file_sha256(records_path),
        "routing_row_count": len(records),
        "trajectory_count": len({str(row.get("trajectory_id") or "") for row in records}),
        "candidate_union_recall": float(
            ((report.get("metrics") or {}).get("factual") or {}).get(
                "candidate_union_recall",
                0.0,
            )
        ),
        "factual_metrics": dict(
            ((report.get("metrics") or {}).get("factual") or {}).get(
                "end_to_end_route",
                {},
            )
        ),
        "static_metrics": dict(
            ((report.get("metrics") or {}).get("static") or {}).get(
                "end_to_end_route",
                {},
            )
        ),
        "memory": dict(report.get("memory") or {}),
    }, records


def build_matched_release_selection(
    *,
    stage2_output_dir: str | Path,
    selected_step: int,
    checkpoint_path: str | Path,
    training_skills_path: str | Path,
    matched_union_manifest_path: str | Path,
    tau2_dev_report_path: str | Path,
    toolsandbox_dev_report_path: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    if not str(source_commit).strip():
        raise ValueError("matched release requires source-commit provenance")
    output = Path(stage2_output_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    training_skills = Path(training_skills_path).resolve()
    if int(selected_step) <= 0:
        raise ValueError("matched release requires a positive checkpoint step")
    if checkpoint != (output / "checkpoints" / f"clstr_vnext_stage2-step{int(selected_step)}.pt"):
        raise ValueError("matched release checkpoint path/step contract differs")
    for path in (checkpoint, training_skills):
        if not path.is_file():
            raise ValueError(f"matched release input is missing: {path}")
    checkpoint_sha = file_sha256(checkpoint)
    training_skills_sha = file_sha256(training_skills)
    original_selection_path = output / "stage2_selection.json"
    quality_gate_path = output / "stage2_quality_gate.json"
    train_report_path = output / "train_report.json"
    source_manifest_path = output / "source_manifest.json"
    original = _load_json(original_selection_path)
    train_report = _load_json(train_report_path)
    source_manifest = _load_json(source_manifest_path)
    if original.get("status") not in {"ok", "action_required"}:
        raise ValueError("original Stage2 selection is incomplete")
    if (
        train_report.get("status") != original.get("status")
        or train_report.get("selection") != original
        or not bool(train_report.get("finite_loss"))
        or int(train_report.get("positive_injection_count") or 0) != 0
        or int(train_report.get("training_teacher_retained_rows") or 0) != 0
    ):
        raise ValueError("Stage2 train report is not a clean completed refinement")
    warm_start = train_report.get("objective_warm_start") or {}
    if not (
        bool(warm_start.get("enabled"))
        and bool(warm_start.get("model_state_exact"))
        and not bool(warm_start.get("optimizer_restored"))
        and not bool(warm_start.get("trainer_progress_restored"))
    ):
        raise ValueError("Stage2 refinement warm-start lineage is not exact")
    source_contract = source_manifest.get("source_contract") or {}
    checkpoint_source_commit = str(source_contract.get("git_commit") or "")
    if not checkpoint_source_commit or bool(source_contract.get("dirty")):
        raise ValueError("Stage2 checkpoint source is not an immutable commit")
    validation_records = list(original.get("validation_records") or [])
    selected_validation = next(
        (
            record
            for record in validation_records
            if int(record.get("step") or 0) == int(selected_step)
        ),
        None,
    )
    if not isinstance(selected_validation, dict):
        raise ValueError("selected checkpoint lacks its held-out validation record")
    gates = selected_validation.get("gates") or {}
    required_gate_names = (
        "causal_route_gate",
        "causal_safety_gate",
        "cluster_sufficiency_gate",
        "coarse_recall_gate",
        "exact_fallback_gate",
        "gradient_health_gate",
        "robust_prefix_dev_gate",
    )
    required_gates = {name: bool(gates.get(name)) for name in required_gate_names}
    ordinary = selected_validation.get("ordinary_dev") or {}
    overall = ordinary.get("overall") or {}
    toolbench = (ordinary.get("per_family") or {}).get("toolbench") or {}
    toolbench_source = (ordinary.get("per_source") or {}).get("toolbench_g3") or {}
    toolbench_evidence = {
        "row_count": int(toolbench.get("row_count") or 0),
        "cluster_count": int(toolbench.get("cluster_count") or 0),
        "route_mrr_delta": _metric_summary(toolbench, "route_mrr_delta"),
        "route_recall_at_1_delta": _metric_summary(
            toolbench,
            "route_recall_at_1_delta",
        ),
        "route_recall_at_5_delta": _metric_summary(
            toolbench,
            "route_recall_at_5_delta",
        ),
        "raw_route_mrr_delta": _metric_summary(toolbench, "raw_route_mrr_delta"),
        "candidate_union_minus_static_recall": _metric_summary(
            toolbench,
            "candidate_union_minus_static_recall",
        ),
        "source_row_count": int(toolbench_source.get("row_count") or 0),
        "source_cluster_count": int(toolbench_source.get("cluster_count") or 0),
        "source_route_mrr_delta": _metric_summary(
            toolbench_source,
            "route_mrr_delta",
        ),
        "overall_route_mrr_delta": _metric_summary(overall, "route_mrr_delta"),
    }
    matched_manifest = _validate_matched_manifest(matched_union_manifest_path)
    tau2_evidence, tau2_records = _validate_dev_report(
        tau2_dev_report_path,
        benchmark="tau2",
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        training_skills_path=training_skills,
        training_skills_sha256=training_skills_sha,
        matched_manifest=matched_manifest,
    )
    toolsandbox_evidence, toolsandbox_records = _validate_dev_report(
        toolsandbox_dev_report_path,
        benchmark="toolsandbox",
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        training_skills_path=training_skills,
        training_skills_sha256=training_skills_sha,
        matched_manifest=matched_manifest,
    )
    tau2_comparisons = {
        mode: paired_cluster_bootstrap(tau2_records, comparison=mode)
        for mode in ("static", "masked", "mismatch", "order_shuffle")
    }
    toolsandbox_comparisons = {
        mode: paired_cluster_bootstrap(toolsandbox_records, comparison=mode)
        for mode in ("static", "masked", "mismatch", "order_shuffle")
    }
    tau2_evidence["paired_comparisons"] = tau2_comparisons
    toolsandbox_evidence["paired_comparisons"] = toolsandbox_comparisons
    blockers: list[str] = []
    if not all(required_gates.values()):
        blockers.append("stage2_required_gate_failed")
    if int(toolbench_evidence["row_count"]) < 100 or int(
        toolbench_evidence["cluster_count"]
    ) < 50:
        blockers.append("toolbench_dev_evidence_insufficient")
    if (
        toolbench_evidence["row_count"] != toolbench_evidence["source_row_count"]
        or toolbench_evidence["cluster_count"]
        != toolbench_evidence["source_cluster_count"]
    ):
        blockers.append("toolbench_family_source_identity_differs")
    if float(toolbench_evidence["route_mrr_delta"]["ci_low"]) <= 0.0:
        blockers.append("toolbench_dev_mrr_not_positive")
    if float(toolbench_evidence["route_recall_at_5_delta"]["ci_low"]) <= 0.0:
        blockers.append("toolbench_dev_r5_not_positive")
    if (
        float(toolbench_evidence["route_recall_at_1_delta"]["mean"]) <= 0.0
        or float(toolbench_evidence["route_recall_at_1_delta"]["ci_low"]) < -0.01
    ):
        blockers.append("toolbench_dev_r1_not_safe")
    if float(
        toolbench_evidence["candidate_union_minus_static_recall"]["ci_low"]
    ) < 0.0:
        blockers.append("toolbench_candidate_union_regresses")
    for comparison in ("static", "masked", "mismatch"):
        evidence = tau2_comparisons[comparison]
        if not evidence.get("eligible"):
            blockers.append(f"tau2_{comparison}_comparison_missing")
            continue
        for metric in ("mrr", "recall_at_1", "recall_at_5"):
            if float((evidence.get("metrics") or {}).get(metric, {}).get("ci_low") or 0.0) <= 0.0:
                blockers.append(f"tau2_factual_not_above_{comparison}_{metric}")
    if float(tau2_evidence["candidate_union_recall"]) != 1.0:
        blockers.append("tau2_candidate_union_recall_not_complete")
    if int((tau2_evidence.get("memory") or {}).get("uses_recurrent_m_t_count") or 0) <= 0:
        blockers.append("tau2_recurrent_memory_not_executed")
    for comparison in ("static", "masked"):
        evidence = toolsandbox_comparisons[comparison]
        if not evidence.get("eligible"):
            blockers.append(f"toolsandbox_{comparison}_comparison_missing")
            continue
        for metric in ("mrr", "recall_at_1", "recall_at_5"):
            if float((evidence.get("metrics") or {}).get(metric, {}).get("ci_low") or 0.0) < 0.0:
                blockers.append(f"toolsandbox_{comparison}_{metric}_regresses")
    if float(toolsandbox_evidence["candidate_union_recall"]) != 1.0:
        blockers.append("toolsandbox_candidate_union_recall_not_complete")
    if int((toolsandbox_evidence.get("memory") or {}).get("uses_recurrent_m_t_count") or 0) <= 0:
        blockers.append("toolsandbox_recurrent_memory_not_executed")
    return {
        "schema_version": MATCHED_RELEASE_SELECTION_SCHEMA,
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "selection_mode": MATCHED_RELEASE_SELECTION_MODE,
        "selection_scope": "single_checkpoint_train_and_heldout_dev_only",
        "selection_reason": "matched_multibench_dev_causal_no_regret_contract",
        "source_commit": str(source_commit),
        "checkpoint_source_commit": checkpoint_source_commit,
        "stage2_output_dir": str(output),
        "selected_step": int(selected_step),
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
        "training_skills_path": str(training_skills),
        "training_skills_sha256": training_skills_sha,
        "original_stage2_selection_path": str(original_selection_path.resolve()),
        "original_stage2_selection_sha256": file_sha256(original_selection_path),
        "original_stage2_quality_gate_sha256": file_sha256(quality_gate_path),
        "original_train_report_sha256": file_sha256(train_report_path),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "original_training_status": str(original.get("status") or ""),
        "validation_records_sha256": json_digest(validation_records),
        "selected_validation_sha256": json_digest(selected_validation),
        "required_stage2_gates": required_gates,
        "objective_warm_start": dict(warm_start),
        "positive_injection_count": 0,
        "training_teacher_retained_rows": 0,
        "matched_union_manifest": matched_manifest,
        "dev_evidence": {
            "toolbench_g3": toolbench_evidence,
            "tau2": tau2_evidence,
            "toolsandbox": toolsandbox_evidence,
        },
        "order_shuffle_is_diagnostic_only": True,
        "closed_set_dispatch_required": False,
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
        "uses_benchmark_identity_for_dispatch": False,
        "uses_source_identity_for_dispatch": False,
        "candidate_positive_injection": False,
    }


def validate_matched_release_selection(
    selection_path: str | Path,
    *,
    stage2_checkpoint_path: str | Path,
) -> dict[str, Any]:
    path = Path(selection_path).resolve()
    payload = _load_json(path)
    if payload.get("schema_version") != MATCHED_RELEASE_SELECTION_SCHEMA:
        raise ValueError("unsupported matched Stage2 release-selection schema")
    if payload.get("status") != "ok" or payload.get("blockers"):
        raise ValueError("matched Stage2 release selection is not approved")
    if payload.get("selection_mode") != MATCHED_RELEASE_SELECTION_MODE:
        raise ValueError("matched Stage2 release selection has the wrong mode")
    if any(
        bool(payload.get(field))
        for field in (
            "uses_benchmark_eval_rows",
            "uses_test_metrics",
            "uses_benchmark_identity_for_dispatch",
            "uses_source_identity_for_dispatch",
            "candidate_positive_injection",
            "closed_set_dispatch_required",
        )
    ):
        raise ValueError("matched Stage2 release selection violates unified deployment")
    checkpoint = Path(stage2_checkpoint_path).resolve()
    selected = Path(str(payload.get("selected_checkpoint_path") or "")).resolve()
    if checkpoint != selected or not selected.is_file():
        raise ValueError("Stage2 checkpoint differs from matched release selection")
    expected_checkpoint_sha = str(payload.get("selected_checkpoint_sha256") or "")
    if not expected_checkpoint_sha or file_sha256(selected) != expected_checkpoint_sha:
        raise ValueError("matched release-selected checkpoint digest differs")
    output = Path(str(payload.get("stage2_output_dir") or "")).resolve()
    evidence = payload.get("dev_evidence") or {}
    recomputed = build_matched_release_selection(
        stage2_output_dir=output,
        selected_step=int(payload.get("selected_step") or 0),
        checkpoint_path=selected,
        training_skills_path=str(payload.get("training_skills_path") or ""),
        matched_union_manifest_path=str(
            (payload.get("matched_union_manifest") or {}).get("path") or ""
        ),
        tau2_dev_report_path=str(
            (evidence.get("tau2") or {}).get("report_path") or ""
        ),
        toolsandbox_dev_report_path=str(
            (evidence.get("toolsandbox") or {}).get("report_path") or ""
        ),
        source_commit=str(payload.get("source_commit") or ""),
    )
    if recomputed != payload:
        raise ValueError("matched release selection differs from recomputed dev evidence")
    try:
        selected.relative_to((output / "checkpoints").resolve())
    except ValueError as error:
        raise ValueError("matched release checkpoint escapes Stage2 output") from error
    required_artifacts = {
        "original_stage2_selection_path": (
            output / "stage2_selection.json",
            "original_stage2_selection_sha256",
        ),
        "source_manifest_path": (
            output / "source_manifest.json",
            "source_manifest_sha256",
        ),
    }
    for path_field, (expected_path, digest_field) in required_artifacts.items():
        actual_path = Path(str(payload.get(path_field) or "")).resolve()
        if actual_path != expected_path.resolve() or not actual_path.is_file():
            raise ValueError(f"matched release source path differs: {path_field}")
        if file_sha256(actual_path) != str(payload.get(digest_field) or ""):
            raise ValueError(f"matched release source digest differs: {path_field}")
    for filename, digest_field in (
        ("stage2_quality_gate.json", "original_stage2_quality_gate_sha256"),
        ("train_report.json", "original_train_report_sha256"),
    ):
        artifact = output / filename
        if not artifact.is_file() or file_sha256(artifact) != str(
            payload.get(digest_field) or ""
        ):
            raise ValueError(f"matched release source artifact differs: {filename}")
    original = _load_json(output / "stage2_selection.json")
    records = list(original.get("validation_records") or [])
    if json_digest(records) != str(payload.get("validation_records_sha256") or ""):
        raise ValueError("matched release validation-record digest differs")
    selected_step = int(payload.get("selected_step") or 0)
    selected_validation = next(
        (record for record in records if int(record.get("step") or 0) == selected_step),
        None,
    )
    if not isinstance(selected_validation, dict) or json_digest(
        selected_validation
    ) != str(payload.get("selected_validation_sha256") or ""):
        raise ValueError("matched release selected-validation digest differs")
    required_gates = dict(payload.get("required_stage2_gates") or {})
    if not required_gates or not all(bool(value) for value in required_gates.values()):
        raise ValueError("matched release does not pass required Stage2 gates")
    training_skills = Path(str(payload.get("training_skills_path") or "")).resolve()
    if not training_skills.is_file() or file_sha256(training_skills) != str(
        payload.get("training_skills_sha256") or ""
    ):
        raise ValueError("matched release training-skill digest differs")
    manifest = _validate_matched_manifest(
        (payload.get("matched_union_manifest") or {}).get("path") or ""
    )
    if manifest != payload.get("matched_union_manifest"):
        raise ValueError("matched release immutable-union contract differs")
    for benchmark in ("tau2", "toolsandbox"):
        record = evidence.get(benchmark) or {}
        for path_field, digest_field in (
            ("report_path", "report_sha256"),
            ("protocol_manifest_path", "protocol_manifest_sha256"),
            ("route_records_path", "route_records_sha256"),
        ):
            artifact = Path(str(record.get(path_field) or "")).resolve()
            if not artifact.is_file() or file_sha256(artifact) != str(
                record.get(digest_field) or ""
            ):
                raise ValueError(
                    f"matched release {benchmark} dev artifact differs: {path_field}"
                )
    if int(payload.get("positive_injection_count") or 0) != 0 or int(
        payload.get("training_teacher_retained_rows") or 0
    ) != 0:
        raise ValueError("matched release declares training-only candidate support")
    if not str(payload.get("source_commit") or "") or not str(
        payload.get("checkpoint_source_commit") or ""
    ):
        raise ValueError("matched release lacks source provenance")
    return {
        "schema_version": MATCHED_RELEASE_SELECTION_SCHEMA,
        "selection_path": str(path),
        "selection_sha256": file_sha256(path),
        "selection_mode": MATCHED_RELEASE_SELECTION_MODE,
        "selection_scope": str(payload.get("selection_scope") or ""),
        "selection_reason": str(payload.get("selection_reason") or ""),
        "source_commit": str(payload["source_commit"]),
        "checkpoint_source_commit": str(payload["checkpoint_source_commit"]),
        "selected_step": selected_step,
        "selected_checkpoint_path": str(selected),
        "selected_checkpoint_sha256": expected_checkpoint_sha,
        "original_training_status": str(payload.get("original_training_status") or ""),
        "original_stage2_selection_path": str(
            Path(payload["original_stage2_selection_path"]).resolve()
        ),
        "original_stage2_selection_sha256": str(
            payload["original_stage2_selection_sha256"]
        ),
        "validation_records_sha256": str(payload["validation_records_sha256"]),
        "matched_union_manifest": manifest,
        "dev_evidence_sha256": json_digest(evidence),
        "closed_set_dispatch_required": False,
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
        "uses_benchmark_identity_for_dispatch": False,
        "uses_source_identity_for_dispatch": False,
    }
