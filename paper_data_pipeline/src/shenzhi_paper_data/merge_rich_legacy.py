"""Merge a legacy Neo4j CSV bundle into a new-authoritative rich graph.

The rich candidate is never modified. New non-empty Paper properties and all
new rich relationships win; legacy data only fills missing Paper metadata and
adds historical coverage in the rich schema.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Sequence
import zipfile

from .build_rich_graph import EDGE_CONTRACT, EDGE_FILES, NODE_FILES, NODE_LABELS, SCHEMA, _schema_cypher, _import_script
from .prepare import file_sha256, is_placeholder_abstract

csv.field_size_limit(sys.maxsize)

PAPER_FIELDS = [
    "paper_id:ID(Paper)", "title", "abstract", "arxiv_id", "doi", "url", "pdf_url",
    "year:int", "external:boolean", "venue", "booktitle", "publisher", "authors_json",
    "keywords_json", "datasets_json", "code_resources_json", "research_problem", "motivation",
    "contributions_json", "result", "properties_json", "sources_json", ":LABEL",
]


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _stable_id(prefix: str, *values: str) -> str:
    material = "\x1f".join(values).encode("utf-8")
    return f"{prefix}:legacy:{hashlib.sha256(material).hexdigest()[:24]}"


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _json_list(value: str) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _zip_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(f"/paper_kg/{basename}.csv")]
    if len(matches) != 1:
        raise ValueError(f"expected one legacy {basename}.csv, got {matches}")
    return matches[0]


def _zip_rows(archive: zipfile.ZipFile, basename: str) -> Iterable[dict[str, str]]:
    with archive.open(_zip_member(archive, basename)) as raw:
        import io
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            yield from csv.DictReader(text)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def _legacy_paper(row: dict[str, str]) -> dict[str, object]:
    properties = {key: row.get(key, "") for key in ("dblp_key", "biburl", "type", "source", "source_file") if row.get(key)}
    return {
        "paper_id:ID(Paper)": row["paper_id:ID(Paper)"], "title": row.get("title", ""),
        "abstract": row.get("abstract", ""), "arxiv_id": row.get("arxiv_id", ""), "doi": row.get("doi", ""),
        "url": row.get("biburl", ""), "pdf_url": row.get("pdf_url", ""), "year:int": row.get("year:int", ""),
        "external:boolean": "false", "venue": row.get("venue", ""), "booktitle": "", "publisher": "",
        "authors_json": "[]", "keywords_json": "[]", "datasets_json": "[]", "code_resources_json": "[]",
        "research_problem": "", "motivation": "", "contributions_json": "[]", "result": "",
        "properties_json": json.dumps(properties, ensure_ascii=False, sort_keys=True),
        "sources_json": json.dumps({"legacy_candidate": True}, ensure_ascii=False, sort_keys=True), ":LABEL": "Paper",
    }


def _merge_paper(new: dict[str, str], old: dict[str, str]) -> dict[str, object]:
    merged: dict[str, object] = dict(new)
    legacy = _legacy_paper(old)
    for field in ("title", "abstract", "arxiv_id", "doi", "url", "pdf_url", "year:int", "venue", "booktitle", "publisher"):
        if not str(merged.get(field, "")).strip() and str(legacy.get(field, "")).strip():
            merged[field] = legacy[field]
    if _norm(str(merged.get("external:boolean", ""))) in {"true", "1", "yes"} and old.get("title") and not is_placeholder_abstract(old.get("abstract", "")):
        merged["external:boolean"] = "false"
    props = _json_object(str(merged.get("properties_json", "")))
    props.setdefault("legacy_fill", {k: v for k, v in _json_object(str(legacy["properties_json"])).items()})
    merged["properties_json"] = json.dumps(props, ensure_ascii=False, sort_keys=True)
    sources = _json_object(str(merged.get("sources_json", "")))
    sources["legacy_candidate_fill"] = True
    merged["sources_json"] = json.dumps(sources, ensure_ascii=False, sort_keys=True)
    return merged


def merge_rich_legacy(rich_dir: Path, legacy_zip: Path, output_dir: Path) -> dict:
    rich_dir, legacy_zip, output_dir = Path(rich_dir).resolve(), Path(legacy_zip).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not (rich_dir / "validation_report.json").is_file() or not legacy_zip.is_file():
        raise FileNotFoundError("rich validation report and legacy ZIP are required")
    validation = json.loads((rich_dir / "validation_report.json").read_text(encoding="utf-8"))
    if not validation.get("ready"):
        raise ValueError("rich candidate validation is not ready")

    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    kg = temp / "paper_kg"
    kg.mkdir(parents=True)
    try:
        new_paper_fields, new_papers = _read_rows(rich_dir / "paper_kg" / "papers.csv")
        if new_paper_fields != PAPER_FIELDS:
            raise ValueError("unexpected rich papers.csv contract")
        new_ids = {row["paper_id:ID(Paper)"] for row in new_papers}
        new_main_ids = {row["paper_id:ID(Paper)"] for row in new_papers if _norm(row.get("external:boolean", "")) not in {"true", "1", "yes"}}

        with zipfile.ZipFile(legacy_zip) as archive:
            old_papers = {row["paper_id:ID(Paper)"]: row for row in _zip_rows(archive, "papers")}
            merged_papers = [_merge_paper(row, old_papers[row["paper_id:ID(Paper)"]]) if row["paper_id:ID(Paper)"] in old_papers else row for row in new_papers]
            merged_papers.extend(_legacy_paper(row) for pid, row in old_papers.items() if pid not in new_ids)
            paper_ids = {str(row["paper_id:ID(Paper)"]) for row in merged_papers}
            node_counts: Counter[str] = Counter(Paper=len(merged_papers))
            _write_rows(kg / "papers.csv", PAPER_FIELDS, merged_papers)

            # Load rich entity nodes and map legacy entities by unique normalized name.
            entity_rows: dict[str, list[dict[str, str]]] = {}
            entity_fields: dict[str, list[str]] = {}
            name_maps: dict[str, dict[str, str]] = {}
            for label in NODE_LABELS[1:]:
                fields, rows = _read_rows(rich_dir / "paper_kg" / NODE_FILES[label])
                entity_fields[label], entity_rows[label] = fields, rows
                id_field = next(field for field in fields if ":ID(" in field)
                grouped: dict[str, list[str]] = defaultdict(list)
                for row in rows:
                    grouped[_norm(row.get("name", ""))].append(row[id_field])
                name_maps[label] = {name: ids[0] for name, ids in grouped.items() if name and len(ids) == 1}

            def add_legacy_entities(old_name: str, label: str, id_prefix: str) -> dict[str, str]:
                fields, rows = entity_fields[label], entity_rows[label]
                id_field = next(field for field in fields if ":ID(" in field)
                mapping: dict[str, str] = {}
                for old in _zip_rows(archive, old_name):
                    old_id = next(old[k] for k in old if ":ID(" in k)
                    name = old.get("name", "")
                    target = name_maps[label].get(_norm(name)) or _stable_id(id_prefix, old_id, name)
                    mapping[old_id] = target
                    if target not in {r[id_field] for r in rows}:
                        rows.append({id_field: target, "name": name,
                                     "properties_json": json.dumps({"legacy_id": old_id}, ensure_ascii=False, sort_keys=True),
                                     "sources_json": json.dumps({"legacy_candidate": True}, sort_keys=True), ":LABEL": label})
                return mapping

            author_map = add_legacy_entities("authors", "Author", "author")
            keyword_map = add_legacy_entities("keywords", "Topic", "topic")
            subject_map = add_legacy_entities("subjects", "Topic", "topic")
            venue_map = add_legacy_entities("venues", "Venue", "venue")

            # Conference nodes are the rich equivalent of legacy venue+year publication facts.
            conf_fields, conf_rows = entity_fields["Conference"], entity_rows["Conference"]
            conf_ids = {r["conference_id:ID(Conference)"] for r in conf_rows}
            conf_by_venue_year = {(_norm(r.get("venue", "")), r.get("year:int", "")): r["conference_id:ID(Conference)"] for r in conf_rows}
            old_year = {r["year_id:ID(Year)"]: r.get("year:int", "") for r in _zip_rows(archive, "years")}
            paper_venue = {r[":START_ID(Paper)"]: r[":END_ID(Venue)"] for r in _zip_rows(archive, "published_in")}
            paper_year = {r[":START_ID(Paper)"]: old_year.get(r[":END_ID(Year)"], "") for r in _zip_rows(archive, "published_year")}
            conference_for_paper: dict[str, str] = {}
            for pid, old_vid in paper_venue.items():
                vid, year = venue_map.get(old_vid), paper_year.get(pid, "")
                if not vid:
                    continue
                venue_name = next((r.get("name", "") for r in entity_rows["Venue"] if r["venue_id:ID(Venue)"] == vid), "")
                cid = conf_by_venue_year.get((_norm(venue_name), year)) or _stable_id("conference", vid, year)
                conference_for_paper[pid] = cid
                if cid not in conf_ids:
                    conf_ids.add(cid)
                    conf_rows.append({"conference_id:ID(Conference)": cid, "name": f"{venue_name} {year}".strip(), "venue": venue_name,
                                      "year:int": year, "booktitle": "", "publisher": "",
                                      "properties_json": json.dumps({"legacy": True}, sort_keys=True),
                                      "sources_json": json.dumps({"legacy_candidate": True}, sort_keys=True), ":LABEL": "Conference"})

            for label in NODE_LABELS[1:]:
                node_counts[label] = _write_rows(kg / NODE_FILES[label], entity_fields[label], entity_rows[label])

            edge_counts: Counter[str] = Counter()
            for edge_type in EDGE_CONTRACT:
                fields, rich_edges = _read_rows(rich_dir / "paper_kg" / EDGE_FILES[edge_type])
                seen = {(r[next(k for k in fields if ":START_ID(" in k)], r[next(k for k in fields if ":END_ID(" in k)], edge_type) for r in rich_edges}
                rows: list[dict[str, object]] = list(rich_edges)
                start_field = next(k for k in fields if ":START_ID(" in k)
                end_field = next(k for k in fields if ":END_ID(" in k)

                def append(source: str, target: str, extra: dict[str, object] | None = None) -> None:
                    key = (source, target, edge_type)
                    if not source or not target or key in seen:
                        return
                    seen.add(key)
                    row: dict[str, object] = {start_field: source, end_field: target, ":TYPE": edge_type,
                        "properties_json": "{}", "sources_json": json.dumps({"legacy_candidate": True}, sort_keys=True), "derived:boolean": "true"}
                    row.update(extra or {})
                    rows.append(row)

                if edge_type == "AUTHORED_BY":
                    for old in _zip_rows(archive, "authored_by"):
                        if old[":START_ID(Paper)"] not in new_main_ids:
                            append(old[":START_ID(Paper)"], author_map.get(old[":END_ID(Author)"], ""), {"position:int": old.get("author_order:int", "")})
                elif edge_type == "HAS_TOPIC":
                    for legacy_file, mapping, end_key in (("has_keyword", keyword_map, ":END_ID(Keyword)"), ("has_subject", subject_map, ":END_ID(Subject)")):
                        for old in _zip_rows(archive, legacy_file):
                            if old[":START_ID(Paper)"] not in new_main_ids:
                                append(old[":START_ID(Paper)"], mapping.get(old[end_key], ""))
                elif edge_type == "PUBLISHED_IN":
                    for pid, cid in conference_for_paper.items():
                        if pid not in new_main_ids:
                            append(pid, cid)
                elif edge_type == "PART_OF":
                    for pid, cid in conference_for_paper.items():
                        append(cid, venue_map.get(paper_venue.get(pid, ""), ""))
                elif edge_type == "CITES":
                    for old in _zip_rows(archive, "cites"):
                        source, target = old[":START_ID(Paper)"], old[":END_ID(Paper)"]
                        if source not in new_main_ids and source in paper_ids and target in paper_ids:
                            append(source, target, {"source": old.get("source", "legacy")})
                edge_counts[edge_type] = _write_rows(kg / EDGE_FILES[edge_type], fields, rows)

        (temp / "constraints.cypher").write_text(_schema_cypher(), encoding="utf-8")
        script = temp / "import_neo4j_admin.sh"
        script.write_text(_import_script(), encoding="utf-8")
        script.chmod(0o755)

        retrieval_count = 0
        with (temp / "retrieval_documents.jsonl").open("w", encoding="utf-8") as handle:
            for row in merged_papers:
                if not row.get("title") or is_placeholder_abstract(str(row.get("abstract", ""))) or _norm(str(row.get("external:boolean", ""))) in {"true", "1", "yes"}:
                    continue
                doc = {"paper_id": row["paper_id:ID(Paper)"], "title": row.get("title", ""), "abstract": row.get("abstract", ""),
                       "year": row.get("year:int", ""), "conference": row.get("venue", ""),
                       "authors": _json_list(str(row.get("authors_json", "[]"))), "keywords": _json_list(str(row.get("keywords_json", "[]")))}
                doc["search_text"] = "\n".join(str(doc[k]) for k in ("title", "abstract", "conference"))
                handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                retrieval_count += 1

        outputs = {}
        for path in sorted(p for p in temp.rglob("*") if p.is_file() and p.name not in {"manifest.json", "quality_report.json", "validation_report.json"}):
            rel = path.relative_to(temp).as_posix()
            outputs[rel] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        quality = {"schema": SCHEMA, "merge_policy": "new_authoritative_legacy_fill_v1", "nodes_by_label": dict(node_counts),
                   "edges_by_type": dict(edge_counts), "counts": {"new_papers": len(new_ids), "legacy_papers": len(old_papers),
                   "overlapping_papers": len(new_ids & old_papers.keys()), "merged_papers": len(merged_papers), "retrieval_documents": retrieval_count},
                   "quality_gates": {"new_graph_authoritative": True, "legacy_only_fills_missing_or_historical": True}}
        (temp / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {"schema": SCHEMA, "merge_policy": quality["merge_policy"], "inputs": [
            {"path": str(rich_dir / "manifest.json"), "bytes": (rich_dir / "manifest.json").stat().st_size, "sha256": file_sha256(rich_dir / "manifest.json")},
            {"path": str(legacy_zip), "bytes": legacy_zip.stat().st_size, "sha256": file_sha256(legacy_zip)}], "outputs": outputs}
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp.replace(output_dir)
        return quality
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich-dir", required=True, type=Path)
    parser.add_argument("--legacy-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = merge_rich_legacy(args.rich_dir, args.legacy_zip, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
