from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from clstr.appworld_dynamic_routing_audit import (
    _extract_appended_appworld_rows,
    _format_query_text,
    _positive_pairs,
    _progress_path,
    _read_jsonl,
    _resolve_device,
    _write_json,
    sample_positive_pairs,
)
from clstr.appworld_multistep import build_multistep_state_text
from clstr.appworld_official_executor import build_official_api_docs_context, build_official_clstr_state_text
from clstr.appworld_multistep import CLSTRMultiStepController
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint


def _rank_in(values: Sequence[str], target: str) -> int | None:
    try:
        return list(values).index(str(target)) + 1
    except ValueError:
        return None


def _hit_rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key) is True) / float(len(rows)), 6)


def _task_lookup_key(query_id: str) -> str:
    return str(query_id).split("::", 1)[0]


def _tasks_by_id(tasks_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if tasks_path is None or not str(tasks_path).strip():
        return {}
    tasks: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(tasks_path):
        for key in (row.get("task_id"), row.get("query_id")):
            if key:
                tasks[str(key)] = dict(row)
    return tasks


def _official_state_text(
    *,
    pair: dict[str, str],
    task: dict[str, Any] | None,
    appworld_root: str | Path | None,
    state_text_source: str,
) -> tuple[str, dict[str, Any]]:
    task_id = _task_lookup_key(pair["query_id"])
    resolved_task = dict(task or {})
    if not resolved_task:
        resolved_task = {
            "task_id": task_id,
            "query_id": task_id,
            "instruction_text": pair["query_text"],
            "required_apps": [],
        }
    if state_text_source == "official_state":
        state_text = build_multistep_state_text(
            task=resolved_task,
            steps=[],
            include_selected_skill_ids=False,
        )
        return state_text, {"task_found": bool(task), "state_api_docs_chars": 0}
    if state_text_source == "official_schema_state":
        api_docs_context = ""
        if appworld_root is not None and str(appworld_root).strip():
            api_docs_context = build_official_api_docs_context(
                appworld_root=appworld_root,
                task=resolved_task,
            )
        state_text = build_official_clstr_state_text(
            task=resolved_task,
            history=[],
            api_docs_context=api_docs_context,
        )
        return state_text, {"task_found": bool(task), "state_api_docs_chars": len(api_docs_context)}
    raise ValueError(f"unsupported state_text_source: {state_text_source}")


def audit_appworld_dynamic_controller_candidates(
    *,
    checkpoint_path: str | Path,
    base_skill_pool_path: str | Path,
    dynamic_skill_pool_path: str | Path,
    retrieval_path: str | Path,
    output_path: str | Path | None = None,
    tasks_path: str | Path | None = None,
    appworld_root: str | Path | None = None,
    model_cache_dir: str | Path | None = None,
    candidate_top_k: int = 20,
    top_k: int = 5,
    max_pairs: int | None = None,
    sampling_strategy: str = "head",
    device: str | torch.device | None = "auto",
    query_text_format: str | None = "auto",
    state_text_source: str = "retrieval_query",
    ranking_mode: str = "skill_table",
    candidate_source: str = "routing",
    appworld_executor_compatible_only: bool = True,
) -> dict[str, Any]:
    state_text_source = str(state_text_source or "retrieval_query")
    if state_text_source not in {"retrieval_query", "official_state", "official_schema_state"}:
        raise ValueError(f"unsupported state_text_source: {state_text_source}")
    progress = _progress_path(output_path)
    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "loading_inputs", "processed_pairs": 0, "total_pairs": 0})
    base_rows = _read_jsonl(base_skill_pool_path)
    dynamic_rows = _read_jsonl(dynamic_skill_pool_path)
    appended_rows, extraction_report = _extract_appended_appworld_rows(
        base_rows=base_rows,
        dynamic_rows=dynamic_rows,
    )
    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "loading_checkpoint", "processed_pairs": 0, "total_pairs": 0})
    model, model_config, checkpoint_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(checkpoint_path),
        skills_path=Path(base_skill_pool_path),
        model_cache_dir=model_cache_dir,
    )
    applied_query_text_format = (
        str(model_config.get("query_text_format") or "raw")
        if query_text_format is None or str(query_text_format).strip().lower() == "auto"
        else str(query_text_format)
    )
    run_device = _resolve_device(device)
    model.to(run_device).eval()
    if progress is not None:
        _write_json(
            progress,
            {
                "status": "running",
                "phase": "appending_dynamic_skills",
                "processed_pairs": 0,
                "total_pairs": 0,
                "appended_appworld_candidate_count": len(appended_rows),
            },
        )
    append_report = model.append_skills(appended_rows)
    retrieval_rows = _read_jsonl(retrieval_path)
    pairs, pair_report = _positive_pairs(retrieval_rows)
    pairs, sampling_report = sample_positive_pairs(
        pairs,
        max_pairs=max_pairs,
        sampling_strategy=sampling_strategy,
    )
    pair_report["sampling"] = sampling_report
    task_map = _tasks_by_id(tasks_path) if state_text_source != "retrieval_query" else {}
    if not pairs:
        report = {
            "status": "blocked",
            "blockers": ["no_positive_pairs"],
            "checkpoint_path": str(checkpoint_path),
            "base_skill_pool_path": str(base_skill_pool_path),
            "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
            "retrieval_path": str(retrieval_path),
            "extraction_report": extraction_report,
            "append_report": append_report,
            "pair_report": pair_report,
        }
        if output_path is not None:
            _write_json(output_path, report)
        if progress is not None:
            _write_json(progress, {"status": "blocked", "phase": "no_positive_pairs", "processed_pairs": 0, "total_pairs": 0})
        return report

    controller = CLSTRMultiStepController(
        model=model,
        skills=list(getattr(model, "skills", [])),
        ranking_mode=ranking_mode,
        candidate_top_k=int(candidate_top_k),
        candidate_source=candidate_source,
        appworld_executor_compatible_only=bool(appworld_executor_compatible_only),
    )
    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "selecting", "processed_pairs": 0, "total_pairs": len(pairs)})

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, pair in enumerate(pairs, start=1):
            controller.reset({"task_id": pair["query_id"]})
            raw_state_text = pair["query_text"]
            state_diag: dict[str, Any] = {"task_found": None, "state_api_docs_chars": 0}
            if state_text_source != "retrieval_query":
                task_key = _task_lookup_key(pair["query_id"])
                raw_state_text, state_diag = _official_state_text(
                    pair=pair,
                    task=task_map.get(task_key),
                    appworld_root=appworld_root,
                    state_text_source=state_text_source,
                )
            state_text = _format_query_text(raw_state_text, applied_query_text_format)
            selection = controller.select(
                task={"task_id": pair["query_id"]},
                state_text=state_text,
                steps=[],
                top_k=int(top_k),
            )
            candidate_skill_ids = [str(item) for item in selection.diagnostics.get("candidate_skill_ids", [])]
            selected_skill_ids = selection.selected_skill_ids
            positive_skill_id = str(pair["positive_skill_id"])
            candidate_rank = _rank_in(candidate_skill_ids, positive_skill_id)
            selected_rank = _rank_in(selected_skill_ids, positive_skill_id)
            rows.append(
                {
                    "query_id": pair["query_id"],
                    "positive_skill_id": positive_skill_id,
                    "candidate_positive_hit": candidate_rank is not None,
                    "candidate_positive_rank": candidate_rank,
                    "selected_positive_hit": selected_rank is not None,
                    "selected_positive_rank": selected_rank,
                    "candidate_skill_ids": candidate_skill_ids,
                    "selected_skill_ids": selected_skill_ids,
                    "state_text_source": state_text_source,
                    "state_text_chars": len(state_text),
                    "raw_state_text_chars": len(raw_state_text),
                    "state_task_found": state_diag.get("task_found"),
                    "state_api_docs_chars": state_diag.get("state_api_docs_chars", 0),
                    "diagnostics": selection.diagnostics,
                }
            )
            if progress is not None:
                _write_json(
                    progress,
                    {
                        "status": "running",
                        "phase": "selecting",
                        "processed_pairs": idx,
                        "total_pairs": len(pairs),
                    },
                )

    blockers: list[str] = []
    if int(append_report.get("appended_count", 0)) <= 0:
        blockers.append("no_appworld_skills_appended")
    if _hit_rate(rows, "candidate_positive_hit") <= 0.0:
        blockers.append("no_candidate_positive_hits")
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "checkpoint_path": str(checkpoint_path),
        "base_skill_pool_path": str(base_skill_pool_path),
        "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
        "retrieval_path": str(retrieval_path),
        "model_config": model_config,
        "checkpoint_report": checkpoint_report,
        "extraction_report": extraction_report,
        "append_report": append_report,
        "pair_report": pair_report,
        "device": str(run_device),
        "query_text_format": applied_query_text_format,
        "state_text_source": state_text_source,
        "tasks_path": None if tasks_path is None else str(tasks_path),
        "appworld_root": None if appworld_root is None else str(appworld_root),
        "candidate_top_k": int(candidate_top_k),
        "top_k": int(top_k),
        "ranking_mode": str(ranking_mode),
        "candidate_source": str(candidate_source),
        "appworld_executor_compatible_only": bool(appworld_executor_compatible_only),
        "candidate_positive_hit_rate": _hit_rate(rows, "candidate_positive_hit"),
        "selected_positive_hit_rate": _hit_rate(rows, "selected_positive_hit"),
        "rows": rows,
        "progress_path": None if progress is None else str(progress),
    }
    if output_path is not None:
        _write_json(output_path, report)
        if progress is not None:
            _write_json(progress, {"status": report["status"], "phase": "done", "processed_pairs": len(pairs), "total_pairs": len(pairs)})
    return report
