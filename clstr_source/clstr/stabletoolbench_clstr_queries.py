from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


METRIC_SCOPE = (
    "Build StableToolBench executable query JSON from CLSTR routing predictions; "
    "prerequisite for CLSTR-routed SoPR, not itself pass-rate."
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _query_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("query_id") or row.get("id") or row.get("qid") or index)


def _query_text(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("query_text") or row.get("instruction") or row.get("text") or "")


def _normalize_query_id(query_id: str) -> str:
    text = str(query_id)
    for prefix in ("toolbench-g3-", "stabletoolbench-g3-", "stabletoolbench-"):
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix) :]
    return text


def _parse_provenance(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance") or {}
    if isinstance(provenance, dict):
        return provenance
    if isinstance(provenance, str):
        try:
            parsed = json.loads(provenance)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _skill_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or index)


def _skill_to_api(row: dict[str, Any]) -> dict[str, Any]:
    provenance = _parse_provenance(row)
    input_schema = row.get("input_schema") if isinstance(row.get("input_schema"), dict) else {}
    output_schema = row.get("output_schema") if isinstance(row.get("output_schema"), dict) else {}
    tool_name = str(provenance.get("tool_name") or row.get("tool_name") or row.get("tool") or "")
    api_name = str(provenance.get("api_name") or row.get("api_name") or row.get("name") or row.get("skill_id") or "")
    api_record: dict[str, Any] = {
        "category_name": str(row.get("environment") or provenance.get("category_name") or "Unknown"),
        "tool_name": tool_name or "unknown",
        "api_name": api_name or "unknown",
        "api_description": str(row.get("executor_desc") or row.get("description") or api_name or ""),
        "required_parameters": list(input_schema.get("required_parameters") or row.get("required_parameters") or []),
        "optional_parameters": list(input_schema.get("optional_parameters") or row.get("optional_parameters") or []),
        "method": input_schema.get("method") or row.get("method") or "GET",
    }
    if output_schema:
        api_record["template_response"] = output_schema
    return api_record


def _load_skill_api_index(skills_path: str | Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(_read_jsonl(skills_path)):
        sid = _skill_id(row, row_index)
        index[sid] = _skill_to_api(row)
        canonical = row.get("canonical_skill_id")
        if canonical:
            index.setdefault(str(canonical), index[sid])
    return index


def _load_trec_run(run_path: str | Path) -> dict[str, list[tuple[int, str, float]]]:
    by_query: dict[str, list[tuple[int, str, float]]] = {}
    with Path(run_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 6:
                raise ValueError(f"Invalid TREC run line in {run_path}: {line.rstrip()}")
            qid, _q0, skill_id, rank_text, score_text, *_rest = parts
            try:
                rank = int(rank_text)
            except ValueError:
                rank = len(by_query.get(_normalize_query_id(qid), [])) + 1
            try:
                score = float(score_text)
            except ValueError:
                score = 0.0
            by_query.setdefault(_normalize_query_id(qid), []).append((rank, skill_id, score))
    for query_id, rows in by_query.items():
        rows.sort(key=lambda item: (item[0], -item[2]))
        by_query[query_id] = rows
    return by_query


def export_stabletoolbench_retrieval_queries(
    *,
    query_file: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = _read_json(query_file)
    if not isinstance(queries, list):
        raise ValueError(f"Expected StableToolBench query list in {query_file}")
    rows = [
        {
            "query_id": _query_id(query, index),
            "query_text": _query_text(query),
            "source": "stabletoolbench_g3",
            "split": "stabletoolbench_solvable_test",
        }
        for index, query in enumerate(queries)
        if isinstance(query, dict)
    ]
    _write_jsonl(output_path, rows)
    report = {
        "status": "ok",
        "query_file": str(query_file),
        "output_path": str(output_path),
        "query_count": len(rows),
        "metric_scope": "StableToolBench solvable query export for CLSTR retrieval; not pass-rate.",
    }
    if report_path is not None:
        _write_json(report_path, report)
    return report


def build_stabletoolbench_queries_from_clstr_run(
    *,
    original_query_file: str | Path,
    skills_path: str | Path,
    run_path: str | Path,
    output_query_file: str | Path,
    report_path: str | Path | None = None,
    top_k: int = 50,
) -> dict[str, Any]:
    queries = _read_json(original_query_file)
    if not isinstance(queries, list):
        raise ValueError(f"Expected StableToolBench query list in {original_query_file}")
    skill_api_index = _load_skill_api_index(skills_path)
    run_by_query = _load_trec_run(run_path)
    output_rows: list[dict[str, Any]] = []
    missing_run_queries: list[str] = []
    empty_api_list_queries: list[str] = []
    missing_skill_ids: list[str] = []
    api_list_sizes: list[int] = []
    retained_query_count = 0

    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            continue
        qid = _query_id(query, query_index)
        normalized_qid = _normalize_query_id(qid)
        ranked = run_by_query.get(normalized_qid, [])
        if not ranked:
            missing_run_queries.append(str(qid))

        api_list: list[dict[str, Any]] = []
        seen_skill_ids: set[str] = set()
        for _rank, skill_id, _score in ranked[: max(0, int(top_k))]:
            if skill_id in seen_skill_ids:
                continue
            seen_skill_ids.add(skill_id)
            api = skill_api_index.get(skill_id)
            if api is None:
                missing_skill_ids.append(skill_id)
                continue
            api_list.append(dict(api))

        if api_list:
            retained_query_count += 1
        else:
            empty_api_list_queries.append(str(qid))
        api_list_sizes.append(len(api_list))
        row = dict(query)
        row["api_list"] = api_list
        output_rows.append(row)

    _write_json(output_query_file, output_rows)
    average_api_list_size = (sum(api_list_sizes) / len(api_list_sizes)) if api_list_sizes else 0.0
    blockers: list[str] = []
    if missing_run_queries:
        blockers.append("missing_run_queries")
    if empty_api_list_queries:
        blockers.append("empty_api_list_queries")
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "original_query_file": str(original_query_file),
        "skills_path": str(skills_path),
        "run_path": str(run_path),
        "output_query_file": str(output_query_file),
        "top_k": int(top_k),
        "query_count": len(output_rows),
        "retained_query_count": retained_query_count,
        "missing_run_query_count": len(missing_run_queries),
        "missing_run_queries": missing_run_queries[:100],
        "empty_api_list_query_count": len(empty_api_list_queries),
        "empty_api_list_queries": empty_api_list_queries[:100],
        "missing_skill_id_count": len(missing_skill_ids),
        "missing_skill_ids": missing_skill_ids[:100],
        "average_api_list_size": average_api_list_size,
        "metric_scope": METRIC_SCOPE,
        "caveat": (
            "This derived StableToolBench input uses CLSTR top-K predicted ToolBench-G3 APIs. "
            "Only raw generation, conversion, and official SoPR evaluation over this file can be reported as CLSTR-routed SoPR."
        ),
    }
    if report_path is not None:
        _write_json(report_path, report)
    return report
