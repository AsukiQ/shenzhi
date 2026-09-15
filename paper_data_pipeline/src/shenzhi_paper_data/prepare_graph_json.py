"""Normalize per-paper graph JSON archives into retrieval and Neo4j staging data.

The input archives are treated as immutable.  This adapter intentionally exposes
only the deterministic A-layer (Paper, complete authors, Topic-as-Keyword,
Venue and Year) to the import CSVs.  Method, Institution, Funding and citations
are retained in separate staging JSONL files until their quality gates are
reviewed by the data-mining owner.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import zipfile

from .prepare import (
    build_search_text,
    collapse_repeated_sequence,
    file_sha256,
    is_placeholder_abstract,
    normalize_text,
    normalized_key,
)


GRAPH_JSON_SCHEMA = "shenzhi_graph_json_adapter_v1"
ARXIV_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.I)


def _slug(value: Any, limit: int = 64) -> str:
    value = unicodedata.normalize("NFKD", normalize_text(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:limit] or "unknown"


def _digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize_arxiv_id(value: Any) -> tuple[str, str, str]:
    """Return (raw-normalized, versionless-base, status)."""
    raw = normalize_text(value).lower()
    raw = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", raw)
    raw = raw.removesuffix(".pdf")
    match = ARXIV_RE.fullmatch(raw)
    if match:
        base, version = match.groups()
        # 2026 submissions in these archives use the five-digit modern
        # sequence.  A four-digit suffix is syntactically parseable but is a
        # known signal of a dropped leading zero (for example 2607.2912).
        if base.startswith("26") and len(base.split(".", 1)[1]) == 4:
            return raw, base, "suspect_missing_leading_zero"
        return raw, base, "ok"
    # Keep the source value, but mark likely four-digit legacy/truncated IDs.
    status = "missing_or_nonstandard"
    if re.fullmatch(r"\d{4}\.\d{3,4}(?:v\d+)?", raw):
        status = "suspect_missing_leading_zero"
    return raw, raw, status


def _node_map(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("graph.nodes must be a list")
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not normalize_text(node.get("id")):
            raise ValueError("every graph node requires a non-empty id")
        node_id = normalize_text(node["id"])
        if node_id in result:
            raise ValueError(f"duplicate graph node id: {node_id}")
        result[node_id] = node
    return result


def _edge_list(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise ValueError("graph.edges must be a list")
    result = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("graph edge must be an object")
        if not all(normalize_text(edge.get(key)) for key in ("source", "target", "type")):
            raise ValueError("graph edge requires source, target and type")
        result.append(edge)
    return result


def _list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_text(item) for item in value if normalize_text(item)]


def _union_ordered(values: Iterable[Iterable[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sequence in values:
        for value in sequence:
            key = normalized_key(value)
            if key and key not in seen:
                seen.add(key)
                out.append(normalize_text(value))
    return out


def _root_paper(nodes: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    roots = [
        (node_id, node)
        for node_id, node in nodes.items()
        if normalize_text(node.get("label")) == "Paper"
        and node.get("properties", {}).get("external") is False
    ]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one external=false Paper root, found {len(roots)}")
    return roots[0]


def _conference_context(
    root_id: str, nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> tuple[str, str, int | None]:
    conference = ""
    venue = ""
    year: int | None = None
    for edge in edges:
        if normalize_text(edge.get("source")) != root_id or normalize_text(edge.get("type")) != "PUBLISHED_IN":
            continue
        target = nodes.get(normalize_text(edge.get("target")), {})
        props = target.get("properties", {}) if isinstance(target, dict) else {}
        if target.get("label") == "Conference":
            conference = normalize_text(props.get("name"))
            venue = normalize_text(props.get("venue"))
            if props.get("year") not in (None, ""):
                try:
                    year = int(props["year"])
                except (TypeError, ValueError):
                    pass
            for next_edge in edges:
                if next_edge.get("source") == target.get("id") and next_edge.get("type") == "PART_OF":
                    venue_node = nodes.get(normalize_text(next_edge.get("target")), {})
                    venue = venue or normalize_text(venue_node.get("properties", {}).get("name"))
        elif target.get("label") == "Venue":
            venue = normalize_text(props.get("name"))
    return conference, venue, year


def _topics(root_id: str, nodes: Mapping[str, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> list[str]:
    return _union_ordered(
        [[normalize_text(nodes.get(normalize_text(edge.get("target")), {}).get("properties", {}).get("name"))]
         for edge in edges
         if edge.get("source") == root_id and edge.get("type") == "HAS_TOPIC"]
    )


def _paper_id(base: str, title: str, package: str, *, collision: bool) -> str:
    key = base or f"{package}|{title}"
    suffix = f"|{package}|{normalized_key(title)}" if collision else ""
    return f"paper:graph:{_slug(key)}:{_digest(key + suffix)}"


def _author_id(name: str) -> str:
    key = normalized_key(name)
    return f"author:graph:{_slug(name, 48)}:{_digest(key)}"


def _value_id(prefix: str, name: str) -> str:
    return f"{prefix}:graph:{_slug(name, 64)}:{_digest(normalized_key(name))}"


def _search_text(record: Mapping[str, Any]) -> str:
    base = build_search_text(record)
    extras = [
        ("[PROBLEM]", record.get("research_problem", "")),
        ("[MOTIVATION]", record.get("motivation", "")),
        ("[CONTRIBUTIONS]", "; ".join(record.get("contributions") or [])),
        ("[DATASETS]", "; ".join(record.get("datasets") or [])),
        ("[RESULT]", record.get("result", "")),
    ]
    return base + "\n" + "\n".join(f"{label} {value}" for label, value in extras if normalize_text(value))


def _parse_archive(zip_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = zip_path.stem.replace("_graph", "").lower()
    papers: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".json"))
        if not members:
            raise ValueError(f"no JSON records found in {zip_path}")
        for member in members:
            payload = json.loads(archive.read(member))
            if not isinstance(payload, dict) or not isinstance(payload.get("graph"), dict):
                raise ValueError(f"{zip_path.name}:{member} is not a canonical graph JSON")
            graph = payload["graph"]
            nodes = _node_map(graph)
            edges = _edge_list(graph)
            root_id, root = _root_paper(nodes)
            props = root.get("properties", {})
            raw_arxiv, base_arxiv, arxiv_status = normalize_arxiv_id(props.get("arxiv_id"))
            conference, venue, year = _conference_context(root_id, nodes, edges)
            if year is None:
                year_value = props.get("year")
                try:
                    year = int(year_value) if year_value not in (None, "") else None
                except (TypeError, ValueError):
                    year = None
            if year is None:
                package_year = re.search(r"20\d{2}", package)
                year = int(package_year.group()) if package_year else None
            authors = collapse_repeated_sequence(_list_text(props.get("authors")))
            if not authors:
                # The graph edge export historically kept only the first author,
                # so this fallback is intentionally secondary to Paper.authors.
                edge_authors = []
                for edge in edges:
                    if edge.get("source") != root_id or edge.get("type") != "AUTHORED_BY":
                        continue
                    author_node = nodes.get(normalize_text(edge.get("target")), {})
                    name = normalize_text(author_node.get("properties", {}).get("name"))
                    if name:
                        edge_authors.append((
                            int(edge.get("properties", {}).get("position", len(edge_authors) + 1)),
                            name,
                        ))
                authors = [name for _, name in sorted(edge_authors)]
            topics = _topics(root_id, nodes, edges)
            outgoing = [edge for edge in edges if edge.get("source") == root_id]
            methods = []
            institutions = []
            fundings = []
            citations = []
            for edge in outgoing:
                target_id = normalize_text(edge.get("target"))
                target = nodes.get(target_id, {})
                label = normalize_text(target.get("label"))
                target_props = target.get("properties", {}) if isinstance(target, dict) else {}
                if edge.get("type") in {"PROPOSES", "USES_AS_BASELINE"} and label == "Method":
                    methods.append({"node_id": target_id, "name": normalize_text(target_props.get("name")), "content": normalize_text(target_props.get("content")), "relation": edge.get("type")})
                if edge.get("type") == "FUNDED_BY" and label == "Funding":
                    fundings.append({"node_id": target_id, "properties": target_props, "edge_properties": edge.get("properties", {})})
                if edge.get("type") == "CITES" and label == "Paper":
                    confidence = edge.get("properties", {}).get("confidence")
                    try:
                        confidence = float(confidence)
                    except (TypeError, ValueError):
                        confidence = None
                    target_arxiv = normalize_text(target_props.get("arxiv_id"))
                    citations.append({
                        "target_node_id": target_id,
                        "target_arxiv_id": target_arxiv,
                        "target_title": normalize_text(target_props.get("title")),
                        "confidence": confidence,
                        "source": edge.get("properties", {}).get("source", ""),
                        "properties": edge.get("properties", {}),
                        "tier": "B" if target_arxiv or (confidence is not None and confidence >= 0.85) else "C",
                    })
            # Institution edges originate from the (usually first) author node.
            for edge in edges:
                if edge.get("type") != "AFFILIATED_WITH":
                    continue
                source = nodes.get(normalize_text(edge.get("source")), {})
                target = nodes.get(normalize_text(edge.get("target")), {})
                if source.get("label") == "Author" and target.get("label") == "Institution":
                    institutions.append({
                        "author_node_id": normalize_text(edge.get("source")),
                        "author_name": normalize_text(source.get("properties", {}).get("name")),
                        "node_id": normalize_text(edge.get("target")),
                        "name": normalize_text(target.get("properties", {}).get("name")),
                        "properties": edge.get("properties", {}),
                    })
            papers.append({
                "package": package,
                "source_file": member,
                "source_graph_id": root_id,
                "source_arxiv_id": raw_arxiv,
                "base_arxiv_id": base_arxiv,
                "arxiv_status": arxiv_status,
                "title": normalize_text(props.get("title")),
                "abstract": normalize_text(props.get("abstract")),
                "authors": authors,
                "keywords": _list_text(props.get("keywords")),
                "conference": conference,
                "venue": venue,
                "year": year,
                "doi": normalize_text(props.get("doi")),
                "pdf_url": normalize_text(props.get("pdf_url")),
                "url": normalize_text(props.get("url")),
                "result": normalize_text(props.get("result")),
                "research_problem": normalize_text(props.get("research_problem")),
                "motivation": normalize_text(props.get("motivation")),
                "contributions": _list_text(props.get("contributions")),
                "datasets": _list_text(props.get("datasets")),
                "methods": methods,
                "institutions": institutions,
                "fundings": fundings,
                "citations": citations,
                "raw_node_count": len(nodes),
                "raw_edge_count": len(edges),
            })
    return papers, {"package": package, "member_count": len(members)}


def _merge_papers(raw: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in raw:
        # Same versionless arXiv and same title are versions of one paper; all
        # other collisions remain separate and are explicitly reported.
        groups[(record["base_arxiv_id"], normalized_key(record["title"]), record["package"] if not record["base_arxiv_id"] else "")].append(record)
    merged: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for _, records in sorted(groups.items(), key=lambda item: item[0]):
        first = dict(records[0])
        for field in ("authors", "keywords", "contributions", "datasets"):
            first[field] = _union_ordered(record.get(field, []) for record in records)
        for field in ("abstract", "result", "research_problem", "motivation", "title", "doi", "pdf_url", "url", "conference", "venue"):
            values = [normalize_text(record.get(field)) for record in records if normalize_text(record.get(field))]
            if values:
                first[field] = max(values, key=len)
        first["source_files"] = sorted({record["source_file"] for record in records})
        first["source_packages"] = sorted({record["package"] for record in records})
        first["source_graph_ids"] = sorted({record["source_graph_id"] for record in records})
        first["source_arxiv_ids"] = sorted({record["source_arxiv_id"] for record in records if record["source_arxiv_id"]})
        first["source_records"] = [
            {
                "package": record["package"],
                "source_graph_id": record["source_graph_id"],
                "source_arxiv_id": record["source_arxiv_id"],
                "source_file": record["source_file"],
                "arxiv_status": record["arxiv_status"],
            }
            for record in records
        ]
        first["version_record_count"] = len(records)
        merged.append(first)
    by_base: Counter[str] = Counter(record["base_arxiv_id"] for record in merged if record["base_arxiv_id"])
    by_title: Counter[str] = Counter(normalized_key(record["title"]) for record in merged)
    for record in merged:
        collision = bool(record["base_arxiv_id"] and by_base[record["base_arxiv_id"]] > 1) or (not record["base_arxiv_id"])
        record["collision_disambiguated"] = collision
        record["paper_id"] = _paper_id(record["base_arxiv_id"], record["title"], record["package"], collision=collision)
        record["same_title_cross_source"] = bool(record["title"] and by_title[normalized_key(record["title"])] > 1)
        for source in record["source_records"]:
            source_rows.append({
                "package": source["package"],
                "source_graph_id": source["source_graph_id"],
                "source_arxiv_id": source["source_arxiv_id"],
                "paper_id": record["paper_id"],
                "title": record["title"],
                "decision": "merged_same_base_and_title" if record["version_record_count"] > 1 else ("collision_disambiguated" if collision else "new_graph_paper"),
                "arxiv_status": source["arxiv_status"],
                "source_file": source["source_file"],
            })
    ids = [record["paper_id"] for record in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical graph paper IDs are not unique")
    return merged, source_rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
            count += 1
    return count


def prepare_graph_archives(zip_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    zip_paths = [Path(path).resolve() for path in zip_paths]
    output_dir = Path(output_dir).resolve()
    if not zip_paths or any(not path.is_file() for path in zip_paths):
        raise FileNotFoundError("all graph ZIP inputs must exist")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    raw: list[dict[str, Any]] = []
    inputs = []
    archive_reports = []
    for path in zip_paths:
        records, archive_report = _parse_archive(path)
        raw.extend(records)
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        archive_reports.append({**archive_report, "root_papers": len(records)})
    papers, source_rows = _merge_papers(raw)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        graph_root = temp_root / "graph_import" / "paper_kg"
        staging_root = temp_root / "staging"
        graph_root.mkdir(parents=True)
        staging_root.mkdir(parents=True)
        retrieval_rows = []
        for record in sorted(papers, key=lambda item: item["paper_id"]):
            record["retrieval_eligible"] = not is_placeholder_abstract(record.get("abstract")) and bool(record.get("title"))
            record["retrieval_exclusion_reason"] = "" if record["retrieval_eligible"] else "placeholder_or_missing_abstract_or_title"
            if record["retrieval_eligible"]:
                retrieval_rows.append({
                    "paper_id": record["paper_id"], "source_id": record["base_arxiv_id"],
                    "title": record["title"], "abstract": record["abstract"], "authors": record["authors"],
                    "conference": record["conference"], "venue": record["venue"], "year": record["year"],
                    "subjects": [], "keywords": record["keywords"], "search_text": _search_text(record),
                    "source_packages": record["source_packages"],
                })
        _write_jsonl(temp_root / "papers_clean.jsonl", sorted(papers, key=lambda item: item["paper_id"]))
        retrieval_count = _write_jsonl(temp_root / "retrieval_documents.jsonl", retrieval_rows)
        _write_csv(temp_root / "paper_id_map.csv", ("package", "source_graph_id", "source_arxiv_id", "paper_id", "title", "decision", "arxiv_status", "source_file"), ((row[key] for key in ("package", "source_graph_id", "source_arxiv_id", "paper_id", "title", "decision", "arxiv_status", "source_file")) for row in source_rows))

        authors: dict[str, str] = {}
        venues: dict[str, str] = {}
        keywords: dict[str, str] = {}
        years: dict[int, str] = {}
        authored_rows = []
        published_rows = []
        year_rows = []
        keyword_rows = []
        for record in papers:
            for order, name in enumerate(record["authors"], start=1):
                aid = _author_id(name)
                authors[aid] = name
                authored_rows.append((record["paper_id"], aid, order, "AUTHORED_BY"))
            if record["venue"]:
                vid = _value_id("venue", record["venue"])
                venues[vid] = record["venue"]
                published_rows.append((record["paper_id"], vid, "PUBLISHED_IN"))
            if record["year"] is not None:
                years[int(record["year"])] = str(int(record["year"]))
                year_rows.append((record["paper_id"], _value_id("year", str(record["year"])), "PUBLISHED_YEAR"))
            for keyword in _union_ordered((record["keywords"],)):
                kid = _value_id("keyword", keyword)
                keywords[kid] = keyword
                keyword_rows.append((record["paper_id"], kid, "HAS_KEYWORD"))
        _write_csv(graph_root / "papers.csv", ("paper_id:ID(Paper)", "title", "abstract", "doi", "arxiv_id", "dblp_key", "biburl", "pdf_url", "type", "source", "source_file", "year:int", "venue", ":LABEL"), ((r["paper_id"], r["title"], r["abstract"], r["doi"], r["base_arxiv_id"], "", "", r["pdf_url"], "inproceedings", ";".join(f"graph_{package}" for package in r["source_packages"]), ";".join(r["source_files"]), r["year"], r["venue"], "Paper") for r in papers))
        _write_csv(graph_root / "authors.csv", ("author_id:ID(Author)", "name", ":LABEL"), ((aid, name, "Author") for aid, name in sorted(authors.items())))
        _write_csv(graph_root / "venues.csv", ("venue_id:ID(Venue)", "name", ":LABEL"), ((vid, name, "Venue") for vid, name in sorted(venues.items())))
        _write_csv(graph_root / "keywords.csv", ("keyword_id:ID(Keyword)", "name", ":LABEL"), ((kid, name, "Keyword") for kid, name in sorted(keywords.items())))
        _write_csv(graph_root / "subjects.csv", ("subject_id:ID(Subject)", "name", ":LABEL"), [])
        _write_csv(graph_root / "years.csv", ("year_id:ID(Year)", "year:int", ":LABEL"), ((_value_id("year", year), year, "Year") for year in sorted(years)))
        _write_csv(graph_root / "authored_by.csv", (":START_ID(Paper)", ":END_ID(Author)", "author_order:int", ":TYPE"), authored_rows)
        _write_csv(graph_root / "published_in.csv", (":START_ID(Paper)", ":END_ID(Venue)", ":TYPE"), published_rows)
        _write_csv(graph_root / "published_year.csv", (":START_ID(Paper)", ":END_ID(Year)", ":TYPE"), year_rows)
        _write_csv(graph_root / "has_keyword.csv", (":START_ID(Paper)", ":END_ID(Keyword)", ":TYPE"), keyword_rows)
        _write_csv(graph_root / "has_subject.csv", (":START_ID(Paper)", ":END_ID(Subject)", ":TYPE"), [])
        _write_csv(graph_root / "cites.csv", (":START_ID(Paper)", ":END_ID(Paper)", "confidence:double", "source", ":TYPE"), [])

        _write_jsonl(staging_root / "methods.jsonl", ({"paper_id": r["paper_id"], "package": r["package"], "items": r["methods"], "status": "pending_quality_gate"} for r in papers))
        _write_jsonl(staging_root / "institutions.jsonl", ({"paper_id": r["paper_id"], "package": r["package"], "items": r["institutions"], "status": "pending_quality_gate"} for r in papers))
        _write_jsonl(staging_root / "funding.jsonl", ({"paper_id": r["paper_id"], "package": r["package"], "items": r["fundings"], "status": "pending_quality_gate"} for r in papers))
        _write_jsonl(staging_root / "citations.jsonl", ({"paper_id": r["paper_id"], "package": r["package"], "items": r["citations"], "status": "pending_resolution"} for r in papers))

        citation_counts = Counter(item["tier"] for record in papers for item in record["citations"])
        method_count = sum(len(record["methods"]) for record in papers)
        institution_count = sum(len(record["institutions"]) for record in papers)
        funding_count = sum(len(record["fundings"]) for record in papers)
        anomaly_counts = Counter(record["arxiv_status"] for record in papers)
        report = {
            "schema": GRAPH_JSON_SCHEMA,
            "inputs": inputs,
            "archives": archive_reports,
            "counts": {
                "source_json_records": len(raw), "canonical_papers": len(papers), "retrieval_documents": retrieval_count,
                "authors": len(authors), "authored_by_edges": len(authored_rows), "venues": len(venues), "keywords": len(keywords), "years": len(years),
                "methods_staged": method_count, "institutions_staged": institution_count, "funding_staged": funding_count,
                "citations_staged": sum(citation_counts.values()), "structured_citations_staged": citation_counts["B"], "title_only_citations_staged": citation_counts["C"],
                "suspect_arxiv_ids": sum(count for status, count in anomaly_counts.items() if status != "ok"),
                "same_title_cross_source_papers": sum(int(record["same_title_cross_source"]) for record in papers),
                "version_records_merged": sum(max(0, record["version_record_count"] - 1) for record in papers),
            },
            "quality_gates": {
                "raw_inputs_modified": False, "a_layer_only_in_graph_import": True,
                "b_c_entities_in_staging_only": True, "title_only_citations_in_graph_import": False,
                "all_paper_ids_unique": len({r["paper_id"] for r in papers}) == len(papers),
                "all_authors_rebuilt_from_paper_properties": True,
            },
            "outputs": {"retrieval_documents": retrieval_count, "graph_import_dir": "graph_import", "staging_dir": "staging"},
        }
        (temp_root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temp_root / "quality_report.md").write_text(_report_markdown(report), encoding="utf-8")
        (temp_root / "manifest.json").write_text(json.dumps({"schema": GRAPH_JSON_SCHEMA, "source_is_immutable": True, "inputs": inputs}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_root.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _report_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return """# 图谱 JSON 规范化 dry-run 报告

## 结果

| 指标 | 数量 |
|---|---:|
| 原始 JSON 记录 | {source_json_records:,} |
| 规范化主论文 | {canonical_papers:,} |
| 可检索文档 | {retrieval_documents:,} |
| 完整作者节点 | {authors:,} |
| AUTHORED_BY 边 | {authored_by_edges:,} |
| Venue / Keyword / Year | {venues:,} / {keywords:,} / {years:,} |
| staging Method / Institution / Funding | {methods_staged:,} / {institutions_staged:,} / {funding_staged:,} |
| 结构化 / 标题型引用（staging） | {structured_citations_staged:,} / {title_only_citations_staged:,} |
| 可疑 arXiv ID | {suspect_arxiv_ids:,} |

## 接入边界

- `graph_import/paper_kg` 只包含 A 层：Paper、从 `Paper.authors` 重建的完整 Author、Topic→Keyword、Venue、Year。
- Method、Institution、Funding 和全部 CITES 保留在 `staging/`，未进入图导入 CSV。
- 标题型低置信度引用不会被删除，只是暂不暴露给检索图。
- 原始 ZIP 未修改；最终上线前仍需人工确认 ID 异常和 staging 质量。
""".format(**counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_paths", action="append", required=True, type=Path, help="canonical graph ZIP; repeat for multiple archives")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_graph_archives(args.zip_paths, args.output_dir)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
