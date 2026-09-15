from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.appworld_corrective_preference import (
    _api_app,
    _is_write_api_ref,
    _select_api_diverse_matches,
    extract_appworld_api_refs,
)


_AUXILIARY_API_TERMS = ("access_token_from", "login", "authenticate")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_skill_by_id(skill_pool_path: str | Path) -> dict[str, dict[str, Any]]:
    skills: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(_read_jsonl(skill_pool_path)):
        skill_id = str(row.get("skill_id") or row.get("canonical_skill_id") or idx)
        skills[skill_id] = row
    return skills


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        str(skill.get(key) or "")
        for key in ("skill_id", "name", "description", "executor_desc", "body", "failure_modes")
    )


def _precompute_skill_records(skill_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local_idx, skill_id in enumerate(sorted(skill_by_id)):
        refs = {
            ref
            for ref in extract_appworld_api_refs(_skill_text(skill_by_id[skill_id]), include_auth=True)
            if not _is_auxiliary_api_ref(ref)
        }
        records.append(
            {
                "skill_id": str(skill_id),
                "local_index": int(local_idx),
                "api_refs": refs,
                "apps": {_api_app(ref) for ref in refs},
            }
        )
    return records


def _match_solution_api_refs_to_skill_records(
    *,
    solution_api_refs: list[str],
    skill_records: list[dict[str, Any]],
    max_targets: int,
) -> list[dict[str, Any]]:
    target_refs = [str(ref) for ref in solution_api_refs if str(ref)]
    target_set = set(target_refs)
    required_write_refs = {ref for ref in target_refs if _is_write_api_ref(ref)}
    target_apps = {_api_app(ref) for ref in target_refs}
    scored: list[dict[str, Any]] = []
    for record in skill_records:
        skill_refs = set(record.get("api_refs") or set())
        exact = sorted(target_set & skill_refs)
        write_exact = sorted(required_write_refs & skill_refs)
        if required_write_refs and not write_exact:
            continue
        app_overlap = sorted(target_apps & set(record.get("apps") or set()))
        if not exact:
            continue
        score = 10 * len(exact) + len(app_overlap)
        scored.append(
            {
                "skill_id": str(record["skill_id"]),
                "local_index": int(record["local_index"]),
                "score": int(score),
                "exact_api_overlap": exact,
                "write_api_overlap": write_exact,
                "app_overlap": app_overlap,
            }
        )
    scored.sort(
        key=lambda item: (
            item["score"],
            len(item["write_api_overlap"]),
            len(item["exact_api_overlap"]),
            -item["local_index"],
        ),
        reverse=True,
    )
    return _select_api_diverse_matches(scored, required_write_refs=required_write_refs, max_targets=max_targets)


def _is_train_task(row: dict[str, Any]) -> bool:
    split = str(row.get("split") or "").lower()
    if split and split != "train":
        return False
    if row.get("train_allowed") is False:
        return False
    return True


def _normalize_api_ref(ref: Any) -> str:
    text = str(ref or "").strip()
    if not text:
        return ""
    if text.startswith("apis."):
        return text
    parts = text.split(".")
    if len(parts) >= 2:
        return f"apis.{parts[-2]}.{parts[-1]}"
    return text


def _is_auxiliary_api_ref(ref: str) -> bool:
    if _api_app(ref) == "supervisor":
        return True
    api_name = str(ref).rsplit(".", 1)[-1]
    return any(term in api_name for term in _AUXILIARY_API_TERMS)


def _task_solution_refs(task: dict[str, Any]) -> list[str]:
    ignored = {"apis.supervisor.complete_task"}
    refs: list[str] = []
    for ref in task.get("api_refs") or task.get("solution_api_refs") or []:
        normalized = _normalize_api_ref(ref)
        if not normalized or normalized in ignored:
            continue
        if _is_auxiliary_api_ref(normalized):
            continue
        if normalized not in refs:
            refs.append(normalized)
    return refs


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    for key in ("candidate_skill_ids", "ranked_skill_ids", "top_skill_ids", "skill_ids", "candidates"):
        value = row.get(key)
        if isinstance(value, list):
            output: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    skill_id = item.get("skill_id") or item.get("id")
                else:
                    skill_id = item
                if skill_id:
                    output.append(str(skill_id))
            return output
    return []


def _load_candidate_map(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    candidate_map: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("query_id") or row.get("task_id") or row.get("id") or "")
        if not qid:
            continue
        candidates = _candidate_ids(row)
        if candidates:
            candidate_map[qid] = candidates
    return candidate_map


def _query_text(task: dict[str, Any], solution_refs: list[str]) -> str:
    base = str(task.get("query") or task.get("instruction_text") or task.get("user_goal") or "")
    return "\n".join(
        [
            base,
            f"Required API refs: {', '.join(sorted(solution_refs)) if solution_refs else 'unknown'}",
            "Retrieve the reusable skill that can execute the required next tool/API operation.",
        ]
    ).strip()


def _pool_positive_row(
    *,
    task: dict[str, Any],
    match: dict[str, Any],
    solution_refs: list[str],
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or task.get("query_id") or "")
    qid = str(task.get("query_id") or task_id)
    return {
        "source": "appworld_train_stage0_api_pool_positive",
        "query_id": qid,
        "query_text": _query_text(task, solution_refs),
        "positive_skill_id": str(match["skill_id"]),
        "negative_skill_ids": [],
        "split": "train",
        "provenance": {
            "source_dataset": "appworld_train_tasks",
            "task_id": task_id,
            "query_id": qid,
            "solution_api_refs": solution_refs,
            "target_match_detail": match,
            "candidate_positive_missing": None,
            "supervision_source": "train_split_api_refs",
        },
    }


def _correction_row(
    *,
    base_row: dict[str, Any],
    candidates: list[str],
    positive_ids: set[str],
) -> dict[str, Any]:
    row = dict(base_row)
    negatives: list[str] = []
    for skill_id in candidates:
        skill_id = str(skill_id)
        if skill_id and skill_id not in positive_ids and skill_id not in negatives:
            negatives.append(skill_id)
    row["source"] = "appworld_train_stage0_api_correction"
    row["negative_skill_ids"] = negatives[:50]
    provenance = dict(row.get("provenance") or {})
    provenance["candidate_positive_missing"] = True
    provenance["candidate_count"] = len(candidates)
    row["provenance"] = provenance
    return row


def build_train_stage0_correction_audit(
    *,
    train_tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_dir: str | Path,
    candidate_rows_path: str | Path | None = None,
    max_targets: int = 5,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    tasks = [row for row in _read_jsonl(train_tasks_path) if _is_train_task(row)]
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    skill_by_id = _load_skill_by_id(skill_pool_path)
    skill_records = _precompute_skill_records(skill_by_id)
    candidate_map = _load_candidate_map(candidate_rows_path)
    candidate_audit_enabled = candidate_rows_path is not None

    pool_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    no_solution_ref_count = 0
    no_skill_pool_match_count = 0
    pool_match_task_count = 0
    candidate_positive_missing_task_count = 0
    candidate_positive_covered_task_count = 0
    candidate_missing_query_count = 0

    for task in tasks:
        task_id = str(task.get("task_id") or task.get("query_id") or "")
        qid = str(task.get("query_id") or task_id)
        solution_refs = _task_solution_refs(task)
        if not solution_refs:
            no_solution_ref_count += 1
            continue
        matches = _match_solution_api_refs_to_skill_records(
            solution_api_refs=solution_refs,
            skill_records=skill_records,
            max_targets=max_targets,
        )
        if not matches:
            no_skill_pool_match_count += 1
            continue
        pool_match_task_count += 1
        task_pool_rows = [_pool_positive_row(task=task, match=match, solution_refs=solution_refs) for match in matches]
        pool_rows.extend(task_pool_rows)
        if not candidate_audit_enabled:
            continue
        candidates = candidate_map.get(qid) or candidate_map.get(task_id) or []
        if not candidates:
            candidate_missing_query_count += 1
            continue
        candidate_set = {str(skill_id) for skill_id in candidates}
        positive_ids = {str(match["skill_id"]) for match in matches}
        missing_rows = [row for row in task_pool_rows if str(row["positive_skill_id"]) not in candidate_set]
        if missing_rows:
            candidate_positive_missing_task_count += 1
            correction_rows.extend(
                _correction_row(base_row=row, candidates=candidates, positive_ids=positive_ids) for row in missing_rows
            )
        else:
            candidate_positive_covered_task_count += 1

    output_path = Path(output_dir)
    pool_path = output_path / "stage0_pool_positive_rows.jsonl"
    correction_path = output_path / "stage0_api_retrieval_corrections.jsonl"
    report_path = output_path / "report.json"
    _write_jsonl(pool_path, pool_rows)
    _write_jsonl(correction_path, correction_rows)
    report = {
        "schema_version": "appworld_train_stage0_correction_audit.v1",
        "train_tasks_path": str(train_tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "candidate_rows_path": str(candidate_rows_path) if candidate_rows_path is not None else "",
        "candidate_audit_enabled": bool(candidate_audit_enabled),
        "train_task_count": len(tasks),
        "skill_count": len(skill_by_id),
        "no_solution_ref_count": int(no_solution_ref_count),
        "pool_match_task_count": int(pool_match_task_count),
        "no_skill_pool_match_count": int(no_skill_pool_match_count),
        "pool_positive_row_count": len(pool_rows),
        "candidate_missing_query_count": int(candidate_missing_query_count),
        "candidate_positive_missing_task_count": int(candidate_positive_missing_task_count),
        "candidate_positive_covered_task_count": int(candidate_positive_covered_task_count),
        "stage0_retrieval_correction_count": len(correction_rows),
        "pool_positive_rows_path": str(pool_path),
        "stage0_retrieval_corrections_path": str(correction_path),
        "report_path": str(report_path),
        "supervision_source": "AppWorld train split API refs only; dev/test GT is not used for training.",
    }
    _write_json(report_path, report)
    return report


def _row_write_api_overlap(row: dict[str, Any]) -> list[str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    match = provenance.get("target_match_detail") if isinstance(provenance.get("target_match_detail"), dict) else {}
    overlap = match.get("write_api_overlap")
    if isinstance(overlap, list):
        return [str(ref) for ref in overlap if str(ref)]
    return []


def build_high_confidence_stage0_correction_subset(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    policy: str = "write_only",
) -> dict[str, Any]:
    if policy != "write_only":
        raise ValueError(f"unsupported high-confidence policy: {policy}")
    rows = _read_jsonl(input_path)
    kept: list[dict[str, Any]] = []
    skipped_read_only_count = 0
    for row in rows:
        write_overlap = _row_write_api_overlap(row)
        if not write_overlap:
            skipped_read_only_count += 1
            continue
        out = dict(row)
        out["source"] = "appworld_train_stage0_high_confidence_api_correction"
        provenance = dict(out.get("provenance") or {})
        provenance["high_confidence_policy"] = policy
        provenance["high_confidence_reason"] = "positive_skill_overlaps_required_write_api"
        provenance["high_confidence_write_api_overlap"] = write_overlap
        out["provenance"] = provenance
        kept.append(out)

    output_path = Path(output_dir)
    output_rows_path = output_path / "stage0_high_confidence_corrections.jsonl"
    report_path = output_path / "report.json"
    _write_jsonl(output_rows_path, kept)
    report = {
        "schema_version": "appworld_stage0_high_confidence_correction_subset.v1",
        "input_path": str(input_path),
        "policy": policy,
        "input_count": len(rows),
        "kept_count": len(kept),
        "skipped_read_only_count": int(skipped_read_only_count),
        "output_rows_path": str(output_rows_path),
        "report_path": str(report_path),
        "training_note": (
            "Rows are high-confidence Stage0 retrieval corrections because the positive skill overlaps "
            "a required write/action API. Read-only API-overlap rows are left for audit until stricter "
            "semantic filtering is available."
        ),
    }
    _write_json(report_path, report)
    return report
