from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _dedup_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _param_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("schema_type") or value.get("data_type") or "").lower()
    return str(value or "").lower()


def _param_signature(schema: Any) -> str:
    params: list[tuple[str, str, str]] = []

    def add_param(item: Any, required: str) -> None:
        if not isinstance(item, dict):
            return
        name = str(item.get("name") or item.get("param_name") or item.get("key") or "").strip().lower()
        if name:
            params.append((name, _param_type(item), required))

    if isinstance(schema, dict):
        has_grouped_schema = any(
            key in schema
            for key in ("required_parameters", "required", "optional_parameters", "optional")
        )
        if has_grouped_schema:
            required = schema.get("required_parameters") or schema.get("required") or []
            optional = schema.get("optional_parameters") or schema.get("optional") or []
            for item in required if isinstance(required, list) else []:
                add_param(item, "required")
            for item in optional if isinstance(optional, list) else []:
                add_param(item, "optional")
        else:
            for name, spec in schema.items():
                if name in {"method", "http_method"}:
                    continue
                params.append((str(name).strip().lower(), _param_type(spec), "unknown"))
    elif isinstance(schema, list):
        for item in schema:
            add_param(item, "unknown")
    return json.dumps(sorted(params), ensure_ascii=False, separators=(",", ":"))


def _canonical_key(row: dict[str, Any]) -> tuple[str, str, str]:
    provenance = _provenance(row)
    if provenance.get("missing_source_record"):
        return ("missing", str(row.get("skill_id") or ""), "")
    name = _dedup_name(str(row.get("name") or row.get("skill_id") or ""))
    if not name:
        return ("id", str(row.get("skill_id") or ""), "")
    return ("name_param", name, _param_signature(row.get("input_schema") or {}))


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_tool_ids")
    if values is None:
        single = row.get("positive_skill_id") or row.get("skill_id") or row.get("tool_id") or row.get("doc_id")
        values = [single] if single else []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if item]


def _negative_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("negative_skill_ids")
    if values is None:
        values = row.get("negative_tool_ids")
    if values is None:
        values = row.get("negative_doc_ids")
    if values is None:
        values = []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if item]


def _resolve_skill_id(skill_id: str, skill_ids: set[str], alias_to_canonical: dict[str, str]) -> tuple[str | None, bool]:
    if skill_id in skill_ids:
        return skill_id, False
    canonical = alias_to_canonical.get(skill_id)
    if canonical in skill_ids:
        return canonical, canonical != skill_id
    return None, False


def review_skill_dedup_borderline_candidates(
    *,
    candidates_path: str | Path,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    duplicate_name_similarity: float = 0.98,
    duplicate_param_overlap: float = 0.95,
) -> dict[str, Any]:
    candidates = list(_read_jsonl(candidates_path))
    reviewed_rows: list[dict[str, Any]] = []
    keep_separate_count = 0
    unresolved_rows: list[dict[str, Any]] = []

    for row in candidates:
        row = dict(row)
        name_similarity = _as_float(row.get("name_similarity"))
        param_overlap = _as_float(row.get("param_overlap"))
        high_confidence_duplicate = (
            name_similarity >= float(duplicate_name_similarity)
            and param_overlap >= float(duplicate_param_overlap)
        )
        if high_confidence_duplicate:
            row["review_status"] = "unresolved_possible_duplicate"
            row["review_decision"] = "requires_manual_or_llm_merge_review"
            unresolved_rows.append(row)
        else:
            row["review_status"] = "reviewed_keep_separate"
            row["review_decision"] = "keep_separate_low_confidence_borderline"
            keep_separate_count += 1
        reviewed_rows.append(row)

    if output_path is not None:
        _write_jsonl(output_path, reviewed_rows)

    report = {
        "status": "complete" if not unresolved_rows else "action_required",
        "method": "conservative_threshold_review_v1",
        "candidates_path": str(candidates_path),
        "reviewed_candidates_path": str(output_path) if output_path is not None else None,
        "reviewed_candidate_count": len(reviewed_rows),
        "keep_separate_count": keep_separate_count,
        "unresolved_candidate_count": len(unresolved_rows),
        "duplicate_name_similarity_threshold": float(duplicate_name_similarity),
        "duplicate_param_overlap_threshold": float(duplicate_param_overlap),
        "unresolved_samples": unresolved_rows[:20],
        "policy": (
            "This conservative review does not auto-merge borderline candidates. "
            "It only clears low-confidence fuzzy candidates as keep-separate and "
            "continues to block high-confidence duplicate candidates."
        ),
    }
    if report_path is not None:
        _write_json(report_path, report)
    return report


def audit_clstr_skill_pool_quality(
    *,
    data_root: str | Path,
    borderline_review_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    manifest = _read_json(data_root / "manifest.json")
    manifest_skill_pool = manifest.get("skill_pool") or {}
    borderline_review = manifest_skill_pool.get("borderline_review") or {}
    borderline_review_report = _read_json(borderline_review_report_path) if borderline_review_report_path else {}

    skills = list(_read_jsonl(data_root / "skill_pool.jsonl"))
    skill_ids = {str(row.get("skill_id")) for row in skills if row.get("skill_id")}
    alias_to_canonical: dict[str, str] = {}
    placeholder_skill_count = 0
    canonical_key_to_ids: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in skills:
        sid = str(row.get("skill_id") or "")
        if not sid:
            continue
        provenance = _provenance(row)
        description = str(row.get("description") or "").strip()
        if provenance.get("missing_source_record") or not description or description == sid:
            placeholder_skill_count += 1
        canonical = str(row.get("canonical_skill_id") or sid)
        alias_to_canonical[sid] = canonical
        alias_to_canonical[canonical] = canonical
        aliases = row.get("alias_skill_ids") or []
        if isinstance(aliases, list):
            for alias in aliases:
                if alias:
                    alias_to_canonical[str(alias)] = canonical
        canonical_key_to_ids[_canonical_key(row)].append(sid)

    duplicate_canonical_key_groups = [
        {"dedup_key": list(key), "skill_ids": sorted(ids)}
        for key, ids in canonical_key_to_ids.items()
        if key[0] != "missing" and len(set(ids)) > 1
    ]

    retrieval_rows = list(_read_jsonl(data_root / "retrieval.jsonl"))
    missing_positive_refs_counter: Counter[str] = Counter()
    positive_alias_ref_count = 0
    multi_positive_query_count = 0
    positive_negative_conflict_count = 0
    for row in retrieval_rows:
        positives = _positive_skill_ids(row)
        negatives = _negative_skill_ids(row)
        if len(positives) > 1:
            multi_positive_query_count += 1
        resolved_positives: set[str] = set()
        for skill_id in positives:
            resolved, used_alias = _resolve_skill_id(skill_id, skill_ids, alias_to_canonical)
            if resolved is None:
                missing_positive_refs_counter[skill_id] += 1
                continue
            resolved_positives.add(resolved)
            positive_alias_ref_count += int(used_alias)
        resolved_negatives = {
            resolved
            for skill_id in negatives
            if (resolved := _resolve_skill_id(skill_id, skill_ids, alias_to_canonical)[0]) is not None
        }
        if resolved_positives & resolved_negatives:
            positive_negative_conflict_count += 1

    raw_skill_count = int(manifest_skill_pool.get("raw_skill_count") or len(skills))
    canonical_skill_count = int(manifest_skill_pool.get("canonical_skill_count") or len(skill_ids))
    merged_group_count = int(manifest_skill_pool.get("merged_group_count") or 0)
    borderline_review_status = str(borderline_review.get("status") or "unknown")
    borderline_candidate_count = int(borderline_review.get("candidate_count") or 0)
    if (
        borderline_review_status == "pending_manual_or_llm_review"
        and borderline_review_report.get("status") == "complete"
    ):
        borderline_review_status = "complete_by_conservative_review"
        borderline_candidate_count = int(borderline_review_report.get("unresolved_candidate_count") or 0)

    blockers: list[str] = []
    if not skills:
        blockers.append("missing_skill_pool")
    if placeholder_skill_count > 0:
        blockers.append("placeholder_skill_records")
    if duplicate_canonical_key_groups:
        blockers.append("duplicate_canonical_keys")
    if missing_positive_refs_counter:
        blockers.append("missing_positive_refs")
    if positive_negative_conflict_count > 0:
        blockers.append("positive_negative_canonical_conflicts")
    if borderline_review_status == "pending_manual_or_llm_review" and borderline_candidate_count > 0:
        blockers.append("skill_pool_borderline_review_pending")

    report = {
        "status": "ok" if not blockers else "action_required",
        "data_root": str(data_root),
        "blockers": blockers,
        "raw_skill_count": raw_skill_count,
        "canonical_skill_count": canonical_skill_count,
        "merged_group_count": merged_group_count,
        "skill_count": len(skills),
        "unique_skill_id_count": len(skill_ids),
        "borderline_review_status": borderline_review_status,
        "borderline_candidate_count": borderline_candidate_count,
        "borderline_review_method": borderline_review.get("method"),
        "borderline_review_report_path": str(borderline_review_report_path) if borderline_review_report_path else None,
        "borderline_review_report": borderline_review_report,
        "missing_positive_ref_count": sum(missing_positive_refs_counter.values()),
        "missing_positive_refs": sorted(missing_positive_refs_counter),
        "placeholder_skill_count": placeholder_skill_count,
        "duplicate_canonical_key_count": len(duplicate_canonical_key_groups),
        "duplicate_canonical_key_samples": duplicate_canonical_key_groups[:20],
        "retrieval_row_count": len(retrieval_rows),
        "multi_positive_query_count": multi_positive_query_count,
        "positive_alias_ref_count": positive_alias_ref_count,
        "positive_negative_conflict_count": positive_negative_conflict_count,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
