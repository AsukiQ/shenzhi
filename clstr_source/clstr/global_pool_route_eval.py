from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clstr.full_base_train import UNIFIED_MEMORY_ROUTE_SCORER, _equivalent_skill_ids_by_skill_id
from clstr.memory_candidate_recall import (
    candidate_recall_protocol_report,
    source_rows_have_causal_sequence,
)
from clstr.memory_utility_gate import RELIABILITY_MODES
from clstr.memory_utility_records import checkpoint_chain_digest
from clstr.memory_utility_gate_train import resolve_reliability_gate


SAFE_PUBLIC_RELIABILITY_MODES = frozenset(RELIABILITY_MODES)


@dataclass(frozen=True)
class GlobalPoolCorpus:
    benchmark: str
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or "").strip()


def merge_global_skill_pool(
    base_skills: list[dict[str, Any]],
    benchmark_skills: list[dict[str, Any]],
    *,
    dynamic_source_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    duplicate_base = 0
    for skill in base_skills:
        sid = _skill_id(skill)
        if not sid:
            continue
        if sid in seen:
            duplicate_base += 1
            continue
        seen.add(sid)
        merged.append(dict(skill))

    appended = 0
    existing = 0
    missing_ids: list[str] = []
    for skill in benchmark_skills:
        sid = _skill_id(skill)
        if not sid:
            continue
        if sid in seen:
            existing += 1
            continue
        copied = dict(skill)
        copied["global_pool_dynamic_append"] = True
        copied["global_pool_dynamic_source"] = str(dynamic_source_label)
        seen.add(sid)
        merged.append(copied)
        missing_ids.append(sid)
        appended += 1

    return merged, {
        "base_skill_count": len(base_skills),
        "base_unique_skill_count": len(merged) - appended,
        "base_duplicate_skill_count": int(duplicate_base),
        "benchmark_skill_count": len(benchmark_skills),
        "existing_skill_count": int(existing),
        "appended_skill_count": int(appended),
        "merged_skill_count": len(merged),
        "dynamic_source_label": str(dynamic_source_label),
        "appended_skill_id_samples": missing_ids[:10],
    }


def load_prebuilt_global_pool_corpus(
    *,
    benchmark: str,
    source_rows_path: str | Path,
    skills_path: str | Path,
) -> GlobalPoolCorpus:
    source_rows = _read_jsonl(source_rows_path)
    skills = _read_jsonl(skills_path)
    benchmark_name = str(benchmark).strip().lower()
    report = {
        "status": "ok",
        "benchmark": benchmark_name,
        "source": "prebuilt_jsonl",
        "source_rows_path": str(source_rows_path),
        "skills_path": str(skills_path),
        "source_row_count": len(source_rows),
        "skill_count": len(skills),
        "skipped_reasons": {},
    }
    return GlobalPoolCorpus(
        benchmark=benchmark_name,
        skills=skills,
        source_rows=source_rows,
        report=report,
    )


def _strict_metrics(metrics: dict[str, Any], *, retained_rows: int, source_rows: int) -> dict[str, float]:
    denom = float(source_rows) if source_rows else 0.0

    def value(key: str) -> float:
        raw = float(metrics.get(key, 0.0) or 0.0)
        return raw * float(retained_rows) / denom if denom > 0 else 0.0

    return {
        "strict_stage4_next_skill_recall@1": value("stage4_next_skill_recall@1"),
        "strict_stage4_next_skill_recall@5": value("stage4_next_skill_recall@5"),
        "strict_stage4_next_skill_mrr": value("stage4_next_skill_mrr"),
        "retained_row_fraction": float(retained_rows) / denom if denom > 0 else 0.0,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
        "retained_stage4_candidate_count": float(metrics.get("stage4_candidate_count", 0.0) or 0.0),
    }


def _numeric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def _source_rows(rows: list[dict[str, Any]], max_eval_rows: int | None) -> list[dict[str, Any]]:
    if max_eval_rows is None:
        return list(rows)
    return list(rows[: max(0, int(max_eval_rows))])


def _row_benchmark(row: dict[str, Any]) -> str:
    return str(row.get("source_benchmark") or row.get("benchmark") or "unknown")


def _eval_by_benchmark_with_single_group_reuse(
    rows: list[dict[str, Any]],
    overall_eval: dict[str, Any],
    evaluate_by_benchmark_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    benchmarks = sorted({_row_benchmark(row) for row in rows})
    if len(benchmarks) == 1:
        return {benchmarks[0]: dict(overall_eval)}
    return evaluate_by_benchmark_fn(*args, **kwargs)


def _eval_handoff_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        loss_mask = dict(copied.get("loss_mask") or {})
        loss_mask["routing"] = False
        loss_mask["L_trans_skill_ce"] = True
        copied["loss_mask"] = loss_mask
        output.append(copied)
    return output


def run_global_pool_clstr_route_eval(
    *,
    corpus: GlobalPoolCorpus,
    base_skills_path: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    output_dir: str | Path,
    max_eval_rows: int | None = None,
    stage0_top_m: int = 500,
    dynamic_extra_k: int = 64,
    candidate_count: int | None = 64,
    batch_size: int = 8,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_handoff_query_mode: str = "skillrouter_state",
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.25,
    transition_scoring_mode: str = "stage0_rank_prior_plus_transition_residual",
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    expected_memory_utility_gate_checkpoint_sha256: str | None = None,
    expected_memory_utility_gate_audit_sha256: str | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    route_records_path: str | Path | None = None,
    route_record_manifest_path: str | Path | None = None,
    route_record_model_digest: str | None = None,
) -> dict[str, Any]:
    from clstr.full_base_train import _attach_stage0_topm_candidates
    from clstr.logged_online_stage4_train import (
        attach_trajectory_prefix_online_memory_scores,
        evaluate_logged_online_stage4_rows,
        evaluate_logged_online_stage4_rows_by_benchmark,
    )
    from clstr.stage4_act_train import _build_stage4_next_skill_rows_from_source_rows
    from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
    from clstr.toolbench_full_clstr_route_eval import strict_metric_contract
    import torch

    reliability_mode = str(reliability_mode or "dynamic")
    if reliability_mode not in SAFE_PUBLIC_RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    if not math.isfinite(float(fixed_alpha)) or not 0.0 <= float(fixed_alpha) <= 1.0:
        raise ValueError("fixed_alpha must be finite and in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_utility_gate, reliability_gate_report = resolve_reliability_gate(
        reliability_mode=reliability_mode,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
        expected_gate_sha256=expected_memory_utility_gate_checkpoint_sha256,
        expected_audit_sha256=expected_memory_utility_gate_audit_sha256,
        device=device,
    )
    if reliability_mode == "learned":
        feature_update_count_cap = float(reliability_gate_report["feature_update_count_cap"])
        feature_candidate_count_cap = float(reliability_gate_report["feature_candidate_count_cap"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)

    base_skills = _read_jsonl(base_skills_path)
    merged_skills, pool_report = merge_global_skill_pool(
        base_skills,
        corpus.skills,
        dynamic_source_label=corpus.benchmark,
    )
    dynamic_append = int(pool_report.get("appended_skill_count") or 0) > 0
    skills_path = output_dir / "global_skill_pool.jsonl"
    source_rows_path = output_dir / f"{corpus.benchmark}_source_rows.jsonl"
    _write_jsonl(skills_path, merged_skills)
    source_rows = _source_rows(corpus.source_rows, max_eval_rows)
    _write_jsonl(source_rows_path, source_rows)

    skill_id_to_idx = {_skill_id(skill): idx for idx, skill in enumerate(merged_skills) if _skill_id(skill)}
    equivalent_skill_ids = _equivalent_skill_ids_by_skill_id(merged_skills)
    candidate_recall_mode = (
        "static_plus_dynamic_extra"
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else "stage0_candidates"
    )
    if candidate_recall_mode == "static_plus_dynamic_extra" and int(stage0_top_m) <= 0:
        raise ValueError("stage0_top_m must be positive for memory candidate recall")
    final_k = (
        int(candidate_count)
        if candidate_count is not None
        else max(1, min(int(stage0_top_m), len(merged_skills)))
    )
    next_skill_pool_mode = (
        "full_pool"
        if candidate_recall_mode == "static_plus_dynamic_extra"
        else "stage0_candidates"
    )
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
        allow_skill_table_prefix_expansion=dynamic_append,
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
        allow_skill_table_prefix_expansion=dynamic_append,
    )
    stage4_load_report = None
    if stage4_checkpoint_path:
        stage4_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state",
            allow_skill_table_prefix_expansion=dynamic_append,
        )

    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    handoff_source_rows = _eval_handoff_source_rows(source_rows)
    handoff_rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        handoff_source_rows,
        merged_skills,
        skill_id_to_idx,
        top_m=stage0_top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        routing_checkpoint_path=stage0_checkpoint_path,
        manifest_path=output_dir / "global_stage0_candidate_handoff.json",
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        setup_status_path=output_dir / "setup_status.jsonl",
        progress_interval_batches=25,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    stage4_rows, route_data_report = _build_stage4_next_skill_rows_from_source_rows(
        handoff_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
        stage0_candidate_handoff_report=handoff_report,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    _write_jsonl(output_dir / f"{corpus.benchmark}_stage4_rows.jsonl", stage4_rows)
    scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        stage4_rows,
        feedback_rows=stage4_rows,
        next_skill_bonus=online_memory_next_skill_bonus,
        exact_transition_bonus=online_memory_exact_transition_bonus,
        memory_mode=online_memory_mode,
    )
    candidate_eval_kwargs = {
        "candidate_recall_mode": candidate_recall_mode,
        "skill_id_to_idx": skill_id_to_idx,
        "equivalent_skill_ids_by_skill_id": equivalent_skill_ids,
        "static_k": int(stage0_top_m),
        "dynamic_extra_k": int(dynamic_extra_k),
        "final_k": int(final_k),
    }
    route_record_kwargs: dict[str, Any] = {}
    if route_records_path is not None:
        rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
        for row in source_rows:
            rows_by_benchmark.setdefault(_row_benchmark(row), []).append(row)
        sequential_benchmarks = {
            benchmark
            for benchmark, benchmark_rows in rows_by_benchmark.items()
            if source_rows_have_causal_sequence(benchmark_rows)
        }
        effective_model_digest = str(route_record_model_digest or "").strip() or checkpoint_chain_digest(
            {
                "stage0": stage0_checkpoint_path,
                "stage2": stage2_checkpoint_path,
                "stage4": stage4_checkpoint_path,
            }
        )
        route_record_kwargs = {
            "route_records_path": route_records_path,
            "route_record_manifest_path": route_record_manifest_path,
            "route_record_pool_protocol": (
                "appended_untrained" if dynamic_append else "known_global"
            ),
            "route_record_model_digest": effective_model_digest,
            "route_record_sequential_benchmarks": sequential_benchmarks,
        }
    prior_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
        route_scorer=route_scorer,
        reliability_mode="static",
        fixed_alpha=1.0,
        **candidate_eval_kwargs,
    )
    prior_eval_by_benchmark = _eval_by_benchmark_with_single_group_reuse(
        scored_rows,
        prior_eval,
        evaluate_logged_online_stage4_rows_by_benchmark,
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
        route_scorer=route_scorer,
        reliability_mode="static",
        fixed_alpha=1.0,
        **candidate_eval_kwargs,
    )
    stage4_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
        route_scorer=route_scorer,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        memory_utility_gate=memory_utility_gate,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
        **route_record_kwargs,
        **candidate_eval_kwargs,
    )
    stage4_eval_by_benchmark = _eval_by_benchmark_with_single_group_reuse(
        scored_rows,
        stage4_eval,
        evaluate_logged_online_stage4_rows_by_benchmark,
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
        route_scorer=route_scorer,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        memory_utility_gate=memory_utility_gate,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
        **candidate_eval_kwargs,
    )

    source_eval_rows = len(source_rows)
    retained_eval_rows = len(stage4_rows)
    strict_prior = _strict_metrics(prior_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_stage4 = _strict_metrics(stage4_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    if int(route_data_report.get("positive_injected_rows") or 0) > 0:
        blockers.append("gold_positive_injected")
    if dynamic_append:
        blockers.append("dynamic_untrained_skill_append")
    target_outside_pool_rows = sum(
        1
        for row in source_rows
        if str(row.get("next_skill_id") or "").strip()
        and str(row.get("next_skill_id") or "").strip() not in skill_id_to_idx
    )
    if target_outside_pool_rows > 0:
        blockers.append("target_outside_declared_legal_pool")
    candidate_recall = candidate_recall_protocol_report(
        stage4_eval,
        pool_protocol="appended_untrained" if dynamic_append else "known_global",
        candidate_source=str(
            route_data_report.get("candidate_source") or "declared_legal_full_skill_pool"
        ),
        legal_pool_size=len(merged_skills),
        source_rows=source_eval_rows,
        static_k=int(stage0_top_m),
        dynamic_extra_k=int(dynamic_extra_k),
        final_k=int(final_k),
        causal_sequential=source_rows_have_causal_sequence(source_rows),
        target_outside_declared_legal_pool_rows=target_outside_pool_rows,
    )

    report = {
        "status": "ok" if not blockers or blockers == ["dynamic_untrained_skill_append"] else "action_required",
        "blockers": blockers,
        "benchmark": corpus.benchmark,
        "method": "clstr_global_pool_stage4",
        "stage4_route": "global_pool_stage0_topm_logged_online_memory",
        "route_scorer": str(route_scorer),
        "output_dir": str(output_dir),
        "base_skills_path": str(base_skills_path),
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "source_eval_rows": source_eval_rows,
        "retained_eval_rows": retained_eval_rows,
        "corpus_report": corpus.report,
        "global_pool_report": pool_report,
        "stage0_candidate_handoff": handoff_report,
        "route_data_report": route_data_report,
        "memory_report": memory_report,
        "prior_eval": prior_eval,
        "stage4_eval": stage4_eval,
        "prior_eval_by_benchmark": prior_eval_by_benchmark,
        "stage4_eval_by_benchmark": stage4_eval_by_benchmark,
        "strict": {
            "prior": strict_prior,
            "stage4": strict_stage4,
        },
        "strict_delta_stage4_vs_prior": _numeric_delta(strict_stage4, strict_prior),
        "candidate_recall": candidate_recall,
        "metric_contract": strict_metric_contract(),
        "config": {
            "max_eval_rows": max_eval_rows,
            "stage0_top_m": int(stage0_top_m),
            "dynamic_extra_k": int(dynamic_extra_k),
            "candidate_count": candidate_count,
            "candidate_recall_mode": candidate_recall_mode,
            "next_skill_pool_mode": next_skill_pool_mode,
            "final_k": int(final_k),
            "batch_size": int(batch_size),
            "stage0_candidate_encode_batch_size": int(stage0_candidate_encode_batch_size),
            "stage0_handoff_query_mode": str(stage0_handoff_query_mode),
            "online_memory_mode": str(online_memory_mode),
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
            "route_scorer": str(route_scorer),
            "reliability_mode": reliability_mode,
            "fixed_alpha": float(fixed_alpha),
            "reliability_gate": reliability_gate_report,
            "dynamic_append": bool(dynamic_append),
            "route_records_path": None if route_records_path is None else str(route_records_path),
            "route_record_manifest_path": (
                None if route_record_manifest_path is None else str(route_record_manifest_path)
            ),
        },
        "model_load": {
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
        },
        "paper_scope_note": (
            "Global-pool CLSTR routing. If dynamic_untrained_skill_append is present in blockers, the benchmark "
            "contains newly appended skills that were not in the trained clean skill pool and should be treated as "
            "unseen-skill transfer rather than fully trained in-pool evidence."
        ),
    }
    _write_json(output_dir / f"{corpus.benchmark}_global_pool_clstr_route_eval_report.json", report)
    return report
