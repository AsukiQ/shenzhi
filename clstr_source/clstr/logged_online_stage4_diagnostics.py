from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import torch

from clstr.full_base_train import (
    _attach_adjacent_next_states,
    _attach_stage0_topm_candidates,
    _equivalent_skill_ids_by_skill_id,
    _read_jsonl,
    _skill_id,
)
from clstr.logged_online_stage4_train import _resolve_logged_online_device
from clstr.stage4_act_train import _as_stage4_handoff_row, _eligible_stage4_source_rows, _provenance
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _short(value: Any, limit: int = 700) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _namespace(skill_id: str) -> str:
    text = str(skill_id or "")
    return text.split("/", 1)[0] if "/" in text else text.split(":", 1)[0]


def _skill_summary(skill: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(skill, dict):
        return {}
    return {
        "skill_id": skill.get("skill_id") or skill.get("id"),
        "canonical_skill_id": skill.get("canonical_skill_id"),
        "name": skill.get("name"),
        "source": skill.get("source"),
        "description": _short(skill.get("description"), 260),
        "body": _short(skill.get("body"), 260),
        "alias_skill_ids": skill.get("alias_skill_ids") or [],
    }


def _rank_in(values: list[str], target: str) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return ">100"


def _candidate_window(candidate_ids: list[str], positive_rank: int | None, *, width: int = 2) -> list[dict[str, Any]]:
    if positive_rank is None:
        return []
    center = positive_rank - 1
    start = max(0, center - int(width))
    end = min(len(candidate_ids), center + int(width) + 1)
    return [{"rank": idx + 1, "skill_id": candidate_ids[idx]} for idx in range(start, end)]


def _handoff_flags(row: dict[str, Any]) -> dict[str, Any]:
    provenance = _provenance(row)
    handoff = provenance.get("stage0_candidate_handoff") if isinstance(provenance, dict) else None
    return handoff if isinstance(handoff, dict) else {}


def build_logged_online_stage4_candidate_records(
    rows: list[dict[str, Any]],
    *,
    skills: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    skill_ids = [str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)]
    skill_by_id = {skill_id: skill for skill_id, skill in zip(skill_ids, skills)}
    equivalent_by_id = _equivalent_skill_ids_by_skill_id(skills)
    top_k = max(1, int(top_k))
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        candidate_ids = [str(item) for item in row.get("stage0_next_candidate_skill_ids") or [] if str(item)]
        next_skill_id = str(row.get("next_skill_id") or "")
        current_skill_id = str(row.get("skill_id") or "")
        positive_rank = _rank_in(candidate_ids, next_skill_id)
        flags = _handoff_flags(row)
        next_masked = bool(flags.get("next_skill_ce_masked"))
        current_added = bool(flags.get("current_skill_candidate_added"))
        current_added_as_positive = current_added and str(flags.get("current_skill_candidate_role")) == "positive"
        stage0_topm_hit = positive_rank is not None and not next_masked and not current_added_as_positive
        equivalent_ids = equivalent_by_id.get(next_skill_id, [])
        equivalent_hits = [skill_id for skill_id in candidate_ids if skill_id in set(equivalent_ids)]
        top_candidate_ids = candidate_ids[:top_k]
        record = {
            "row_index": row_idx,
            "benchmark": str(row.get("benchmark") or row.get("source_benchmark") or "unknown"),
            "task_id": row.get("task_id"),
            "trajectory_id": row.get("trajectory_id"),
            "step_index": row.get("step_index"),
            "skill_id": current_skill_id,
            "next_skill_id": next_skill_id,
            "next_skill_namespace": _namespace(next_skill_id),
            "candidate_count": len(candidate_ids),
            "candidate_positive_rank": positive_rank,
            "candidate_rank_bucket": _rank_bucket(positive_rank),
            "candidate_positive_hit": positive_rank is not None,
            "stage0_next_positive_missing": positive_rank is None,
            "stage0_topm_next_positive_hit": bool(stage0_topm_hit),
            "next_skill_ce_masked": next_masked,
            "current_skill_candidate_added": current_added,
            "current_skill_candidate_role": str(flags.get("current_skill_candidate_role") or "none"),
            "equivalent_next_skill_ids": equivalent_ids,
            "equivalent_candidate_skill_ids": equivalent_hits,
            "equivalent_candidate_hit": bool(equivalent_hits),
            "top_candidate_skill_ids": top_candidate_ids,
            "positive_candidate_window": _candidate_window(candidate_ids, positive_rank),
            "top_candidate_skills": [_skill_summary(skill_by_id.get(skill_id)) for skill_id in top_candidate_ids],
            "next_skill": _skill_summary(skill_by_id.get(next_skill_id)),
            "goal_text": _short(row.get("goal_text"), 500),
            "state_text": _short(row.get("state_text"), 900),
            "history_text": _short(row.get("history_text"), 700),
            "action_text": _short(row.get("action_text"), 500),
            "next_observation_text": _short(row.get("next_observation_text"), 700),
            "provenance": _provenance(row),
        }
        records.append(record)
    return records


def _stats(records: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, Any]:
    count = len(records)
    if count <= 0:
        return {"row_count": 0}
    ranks = [int(row["candidate_positive_rank"]) for row in records if isinstance(row.get("candidate_positive_rank"), int)]
    stats: dict[str, Any] = {
        "row_count": count,
        "candidate_positive_coverage": len(ranks) / count,
        "mean_candidate_count": sum(int(row.get("candidate_count") or 0) for row in records) / count,
        "rank_bucket_counts": dict(sorted(Counter(str(row.get("candidate_rank_bucket")) for row in records).items())),
    }
    if ranks:
        stats["mean_positive_rank"] = sum(ranks) / len(ranks)
        stats["median_positive_rank"] = float(median(ranks))
    else:
        stats["mean_positive_rank"] = None
        stats["median_positive_rank"] = None
    for k in k_values:
        stats[f"candidate_recall@{k}"] = sum(1 for rank in ranks if rank <= int(k)) / count
    return stats


def _breakdown(records: list[dict[str, Any]], key: str, k_values: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: _stats(group_rows, k_values) for name, group_rows in sorted(grouped.items())}


def summarize_logged_online_stage4_candidate_records(
    records: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] | list[int] = (1, 5, 10, 20, 50, 100, 350),
) -> dict[str, Any]:
    k_tuple = tuple(sorted({max(1, int(k)) for k in k_values}))
    missing = [row for row in records if row.get("candidate_positive_rank") is None]
    equivalent_hits = [row for row in records if bool(row.get("equivalent_candidate_hit"))]
    return {
        "overall": _stats(records, k_tuple),
        "missing_positive_count": len(missing),
        "equivalent_candidate_hit_count": len(equivalent_hits),
        "top_missing_next_skill_ids": Counter(str(row.get("next_skill_id") or "") for row in missing).most_common(30),
        "top_next_skill_ids": Counter(str(row.get("next_skill_id") or "") for row in records).most_common(30),
        "by_benchmark": _breakdown(records, "benchmark", k_tuple),
        "by_next_skill_namespace": _breakdown(records, "next_skill_namespace", k_tuple),
        "missing_examples": missing[:20],
        "equivalent_hit_examples": equivalent_hits[:20],
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    overall = (report.get("summary") or {}).get("overall") or {}
    lines = [
        "# Logged-Online Stage4 Candidate Diagnostics",
        "",
        f"- status: `{report.get('status')}`",
        f"- rows: `{overall.get('row_count')}`",
        f"- candidate coverage: `{overall.get('candidate_positive_coverage')}`",
        f"- recall@1: `{overall.get('candidate_recall@1')}`",
        f"- recall@5: `{overall.get('candidate_recall@5')}`",
        f"- recall@100: `{overall.get('candidate_recall@100')}`",
        f"- missing positives: `{(report.get('summary') or {}).get('missing_positive_count')}`",
        f"- equivalent candidate hits: `{(report.get('summary') or {}).get('equivalent_candidate_hit_count')}`",
        "",
        "## Rank Buckets",
        "",
    ]
    for bucket, count in (overall.get("rank_bucket_counts") or {}).items():
        lines.append(f"- `{bucket}`: `{count}`")
    return "\n".join(lines) + "\n"


def run_logged_online_stage4_candidate_diagnostics(
    *,
    model: Any,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    stage0_top_m: int = 350,
    allowed_benchmarks: set[str] | None = None,
    max_rows: int | None = None,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    routing_checkpoint_path: str | Path | None = None,
    top_k: int = 10,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    raw_source_rows = _read_jsonl(trajectories_path)
    prepared_source_rows, causal_next_state_report = _attach_adjacent_next_states(raw_source_rows)
    source_rows, source_report = _eligible_stage4_source_rows(
        prepared_source_rows,
        skill_id_to_idx,
        allowed_benchmarks=allowed_benchmarks,
        max_source_rows=max_rows,
    )
    handoff_rows = [_as_stage4_handoff_row(row) for row in source_rows]
    resolved_device = _resolve_logged_online_device(model, device)
    retained_rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        handoff_rows,
        skills,
        skill_id_to_idx,
        top_m=stage0_top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=routing_checkpoint_path,
        manifest_path=output / "stage0_candidate_handoff.json",
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=resolved_device,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
    )
    records = build_logged_online_stage4_candidate_records(retained_rows, skills=skills, top_k=top_k)
    rows_path = output / "stage4_candidate_diagnostics.jsonl"
    _write_jsonl(rows_path, records)
    summary = summarize_logged_online_stage4_candidate_records(records)
    report = {
        "status": "ok" if records else "action_required",
        "blockers": [] if records else ["no_logged_online_stage4_candidate_rows"],
        "output_dir": str(output),
        "row_diagnostics_path": str(rows_path),
        "trajectories_path": str(trajectories_path),
        "skills_path": str(skills_path),
        "config": {
            "stage0_top_m": int(stage0_top_m),
            "allowed_benchmarks": sorted(allowed_benchmarks) if allowed_benchmarks is not None else None,
            "max_rows": max_rows,
            "stage0_handoff_query_mode": str(stage0_handoff_query_mode),
            "stage0_candidate_encode_batch_size": int(stage0_candidate_encode_batch_size),
            "top_k": int(top_k),
        },
        "source_report": source_report,
        "causal_next_state": causal_next_state_report,
        "stage0_candidate_handoff": handoff_report,
        "summary": summary,
    }
    report_path = output / "stage4_candidate_diagnostics_report.json"
    markdown_path = output / "stage4_candidate_diagnostics_report.md"
    _write_json(report_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report


def run_logged_online_stage4_candidate_diagnostics_from_checkpoint(
    *,
    routing_checkpoint_path: str | Path,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    stage0_top_m: int = 350,
    allowed_benchmarks: set[str] | None = None,
    max_rows: int | None = None,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    top_k: int = 10,
    model_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=routing_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=model_cache_dir or output / "model_cache",
    )
    report = run_logged_online_stage4_candidate_diagnostics(
        model=model,
        trajectories_path=trajectories_path,
        skills_path=skills_path,
        output_dir=output,
        stage0_top_m=stage0_top_m,
        allowed_benchmarks=allowed_benchmarks,
        max_rows=max_rows,
        stage0_handoff_query_mode=stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
        routing_checkpoint_path=routing_checkpoint_path,
        top_k=top_k,
    )
    report["model_load"] = {
        "model_config": model_config,
        "routing_init": routing_report,
    }
    _write_json(Path(report["report_path"]), report)
    return report
