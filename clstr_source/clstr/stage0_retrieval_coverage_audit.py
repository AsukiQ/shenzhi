from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _skill_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or fallback)


def _load_skill_ids(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    return {_skill_id(row, str(idx)) for idx, row in enumerate(_read_jsonl(path))}


def _provenance_dict(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        return provenance
    if isinstance(provenance, str):
        try:
            parsed = json.loads(provenance)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    for key in ("ranked_skill_ids", "candidate_skill_ids", "top_skill_ids", "skill_ids", "candidates"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
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


def _empty_source_stats(k_values: list[int]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0,
        "positive_in_pool_rows": 0,
        "positive_absent_from_pool": 0,
        "missing_at_max_k": 0,
    }
    for k in k_values:
        stats[f"covered@{k}"] = 0
        stats[f"missing@{k}"] = 0
        stats[f"recall@{k}"] = 0.0
        stats[f"recall_in_pool@{k}"] = 0.0
    return stats


def _finalize_stats(stats: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    rows = int(stats.get("rows") or 0)
    in_pool_rows = int(stats.get("positive_in_pool_rows") or 0)
    out = dict(stats)
    for k in k_values:
        covered = int(out.get(f"covered@{k}") or 0)
        out[f"missing@{k}"] = max(0, rows - covered)
        out[f"recall@{k}"] = float(covered / rows) if rows else 0.0
        out[f"recall_in_pool@{k}"] = float(covered / in_pool_rows) if in_pool_rows else 0.0
    return out


def _missing_row(
    *,
    row: dict[str, Any],
    candidates: list[str],
    max_k: int,
    negative_k: int,
) -> dict[str, Any]:
    positive = str(row.get("positive_skill_id") or row.get("skill_id") or "")
    negatives: list[str] = []
    for skill_id in candidates:
        if skill_id and skill_id != positive and skill_id not in negatives:
            negatives.append(skill_id)
        if len(negatives) >= int(negative_k):
            break
    provenance = _provenance_dict(row)
    provenance.update(
        {
            "candidate_positive_missing": True,
            "audit_max_k": int(max_k),
            "candidate_count": len(candidates),
            "audit_source": "stage0_retrieval_coverage_audit",
        }
    )
    return {
        "source": f"{row.get('source') or 'unknown'}_stage0_missing_positive",
        "query_id": str(row.get("query_id") or row.get("id") or ""),
        "query_text": str(row.get("query_text") or row.get("query") or ""),
        "positive_skill_id": positive,
        "negative_skill_ids": negatives,
        "provenance": provenance,
    }


def build_stage0_retrieval_coverage_audit(
    *,
    retrieval_rows_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    skill_pool_path: str | Path | None = None,
    k_values: list[int] | tuple[int, ...] = (20, 50, 100, 350),
    negative_k: int = 50,
    max_missing_rows: int | None = None,
) -> dict[str, Any]:
    ks = sorted({int(k) for k in k_values if int(k) > 0})
    if not ks:
        raise ValueError("k_values must contain at least one positive integer")
    max_k = max(ks)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    missing_path = output / "stage0_missing_positive_rows.jsonl"
    report_path = output / "report.json"

    by_source: dict[str, dict[str, Any]] = defaultdict(lambda: _empty_source_stats(ks))
    overall = _empty_source_stats(ks)
    query_mismatch_examples: list[dict[str, str]] = []
    absent_positive_examples: list[dict[str, str]] = []
    counters = Counter()
    skill_ids = _load_skill_ids(skill_pool_path)

    pred_iter = _read_jsonl(predictions_path)
    with missing_path.open("w", encoding="utf-8") as missing_handle:
        for row_idx, row in enumerate(_read_jsonl(retrieval_rows_path)):
            try:
                pred = next(pred_iter)
            except StopIteration:
                counters["missing_prediction_rows"] += 1
                break
            qid = str(row.get("query_id") or row.get("id") or "")
            pred_qid = str(pred.get("query_id") or pred.get("task_id") or pred.get("id") or "")
            if qid != pred_qid:
                counters["mismatched_query_id"] += 1
                if len(query_mismatch_examples) < 5:
                    query_mismatch_examples.append({"row_query_id": qid, "prediction_query_id": pred_qid})
            positive = str(row.get("positive_skill_id") or row.get("skill_id") or "")
            if not qid or not positive:
                counters["skipped_missing_query_or_positive"] += 1
                continue
            source = str(row.get("source") or "unknown")
            candidates = _candidate_ids(pred)
            candidate_prefix = candidates[:max_k]
            source_stats = by_source[source]
            source_stats["rows"] += 1
            overall["rows"] += 1
            positive_in_pool = skill_ids is None or positive in skill_ids
            if not positive_in_pool:
                source_stats["positive_absent_from_pool"] += 1
                overall["positive_absent_from_pool"] += 1
                if len(absent_positive_examples) < 10:
                    absent_positive_examples.append(
                        {
                            "query_id": qid,
                            "source": source,
                            "positive_skill_id": positive,
                        }
                    )
                counters["positive_absent_from_pool"] += 1
                counters["row_count"] = row_idx + 1
                continue
            source_stats["positive_in_pool_rows"] += 1
            overall["positive_in_pool_rows"] += 1
            for k in ks:
                covered = positive in candidates[: min(k, len(candidates))]
                if covered:
                    source_stats[f"covered@{k}"] += 1
                    overall[f"covered@{k}"] += 1
            if positive not in candidate_prefix:
                source_stats["missing_at_max_k"] += 1
                overall["missing_at_max_k"] += 1
                if max_missing_rows is None or counters["written_missing_rows"] < int(max_missing_rows):
                    missing_handle.write(
                        json.dumps(
                            _missing_row(row=row, candidates=candidate_prefix, max_k=max_k, negative_k=negative_k),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    counters["written_missing_rows"] += 1
            counters["row_count"] = row_idx + 1

    # Detect extra prediction rows without materializing them.
    for _extra in pred_iter:
        counters["extra_prediction_rows"] += 1
        break

    finalized_by_source = {
        source: _finalize_stats(stats, ks)
        for source, stats in sorted(by_source.items(), key=lambda item: (-int(item[1].get("rows") or 0), item[0]))
    }
    report = {
        "schema_version": "stage0_retrieval_coverage_audit.v1",
        "retrieval_rows_path": str(retrieval_rows_path),
        "predictions_path": str(predictions_path),
        "skill_pool_path": "" if skill_pool_path is None else str(skill_pool_path),
        "skill_pool_membership_audit_enabled": skill_ids is not None,
        "skill_pool_size": None if skill_ids is None else len(skill_ids),
        "k_values": ks,
        "max_k": max_k,
        "row_count": int(counters.get("row_count") or 0),
        "positive_absent_from_pool": int(counters.get("positive_absent_from_pool") or 0),
        "mismatched_query_id_count": int(counters.get("mismatched_query_id") or 0),
        "missing_prediction_rows": int(counters.get("missing_prediction_rows") or 0),
        "extra_prediction_rows_detected": int(counters.get("extra_prediction_rows") or 0),
        "skipped_missing_query_or_positive": int(counters.get("skipped_missing_query_or_positive") or 0),
        "written_missing_rows": int(counters.get("written_missing_rows") or 0),
        "query_mismatch_examples": query_mismatch_examples,
        "absent_positive_examples": absent_positive_examples,
        "overall": _finalize_stats(overall, ks),
        "by_source": finalized_by_source,
        "missing_rows_path": str(missing_path),
        "report_path": str(report_path),
        "note": "Missing rows are in-pool positive labels absent from the audited top-k candidate list; positives absent from the skill pool are reported separately and require pool alignment before Stage0 correction.",
    }
    _write_json(report_path, report)
    return report


def _base_missing_source(source: Any) -> str:
    text = str(source or "unknown")
    suffix = "_stage0_missing_positive"
    return text[: -len(suffix)] if text.endswith(suffix) else text


def build_stage0_missing_positive_subset(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    source_allowlist: list[str] | tuple[str, ...] | set[str],
    per_source_cap: int | None = None,
) -> dict[str, Any]:
    allowed = {str(source) for source in source_allowlist if str(source)}
    if not allowed:
        raise ValueError("source_allowlist must not be empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "stage0_missing_positive_subset.jsonl"
    report_path = output / "report.json"

    kept_by_source: Counter[str] = Counter()
    counters = Counter()
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in _read_jsonl(input_path):
            counters["input_count"] += 1
            source = _base_missing_source(row.get("source"))
            if source not in allowed:
                counters["skipped_by_source_filter"] += 1
                continue
            if per_source_cap is not None and kept_by_source[source] >= int(per_source_cap):
                counters["skipped_by_cap"] += 1
                continue
            out = dict(row)
            provenance = _provenance_dict(out)
            provenance["subset_source"] = source
            provenance["subset_policy"] = "source_allowlist_balanced_cap"
            if per_source_cap is not None:
                provenance["subset_per_source_cap"] = int(per_source_cap)
            out["provenance"] = provenance
            out["source"] = f"{source}_stage0_balanced_missing_positive"
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            kept_by_source[source] += 1
            counters["kept_count"] += 1

    report = {
        "schema_version": "stage0_missing_positive_subset.v1",
        "input_path": str(input_path),
        "source_allowlist": sorted(allowed),
        "per_source_cap": per_source_cap,
        "input_count": int(counters["input_count"]),
        "kept_count": int(counters["kept_count"]),
        "skipped_by_source_filter": int(counters["skipped_by_source_filter"]),
        "skipped_by_cap": int(counters["skipped_by_cap"]),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "output_rows_path": str(rows_path),
        "report_path": str(report_path),
        "note": "Subset rows are balanced candidates for extra Stage0 retrieval correction; they still require small-run validation before full training.",
    }
    _write_json(report_path, report)
    return report
