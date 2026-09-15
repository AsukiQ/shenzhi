"""Build a deduplicated rich-schema Neo4j import candidate from canonical graphs.

The source archives are immutable.  Node IDs and relationship endpoints come
from the canonical JSON exports.  The only derived graph records are missing
authors/``AUTHORED_BY`` relationships reconstructed from a main Paper node's
ordered ``authors`` property; they are explicitly marked as derived.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import zipfile

from .prepare import file_sha256, is_placeholder_abstract, normalize_text, normalized_key


SCHEMA = "shenzhi_neo4j_rich_import_v1"
NODE_LABELS = (
    "Paper",
    "Author",
    "Institution",
    "Method",
    "Funding",
    "Topic",
    "Conference",
    "Venue",
)
EDGE_CONTRACT: dict[str, tuple[str, str]] = {
    "AUTHORED_BY": ("Paper", "Author"),
    "AFFILIATED_WITH": ("Author", "Institution"),
    "PROPOSES": ("Paper", "Method"),
    "USES_AS_BASELINE": ("Paper", "Method"),
    "FUNDED_BY": ("Paper", "Funding"),
    "HAS_TOPIC": ("Paper", "Topic"),
    "PUBLISHED_IN": ("Paper", "Conference"),
    "PART_OF": ("Conference", "Venue"),
    "CITES": ("Paper", "Paper"),
}
NODE_FILES = {
    "Paper": "papers.csv",
    "Author": "authors.csv",
    "Institution": "institutions.csv",
    "Method": "methods.csv",
    "Funding": "fundings.csv",
    "Topic": "topics.csv",
    "Conference": "conferences.csv",
    "Venue": "venues.csv",
}
EDGE_FILES = {
    "AUTHORED_BY": "authored_by.csv",
    "AFFILIATED_WITH": "affiliated_with.csv",
    "PROPOSES": "proposes.csv",
    "USES_AS_BASELINE": "uses_as_baseline.csv",
    "FUNDED_BY": "funded_by.csv",
    "HAS_TOPIC": "has_topic.csv",
    "PUBLISHED_IN": "published_in.csv",
    "PART_OF": "part_of.csv",
    "CITES": "cites.csv",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clean_json(value: Any) -> Any:
    """Normalize JSON-compatible values without inventing semantic content."""
    if isinstance(value, dict):
        return {normalize_text(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, str):
        return normalize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return normalize_text(value)


def _value_score(value: Any) -> tuple[int, int, str]:
    encoded = _stable_json(value)
    if value in (None, "", [], {}):
        return (0, 0, encoded)
    if isinstance(value, dict):
        return (4, len(encoded), encoded)
    if isinstance(value, list):
        return (3, len(encoded), encoded)
    if isinstance(value, str):
        return (2, len(value), encoded)
    return (1, len(encoded), encoded)


def _merge_values(field: str, values: Sequence[Any]) -> Any:
    nonempty = [value for value in values if value not in (None, "", [], {})]
    if not nonempty:
        if any(isinstance(value, list) for value in values):
            return []
        if any(isinstance(value, dict) for value in values):
            return {}
        return ""
    encoded_values = {_stable_json(value): value for value in nonempty}
    if len(encoded_values) == 1:
        return next(iter(encoded_values.values()))
    if field == "external" and any(value is False for value in nonempty):
        return False
    if all(isinstance(value, list) for value in nonempty):
        if field == "authors":
            return max(nonempty, key=lambda value: (len(value), len(_stable_json(value)), _stable_json(value)))
        merged: list[Any] = []
        seen: set[str] = set()
        for value in sorted(nonempty, key=_stable_json):
            for item in value:
                key = _stable_json(item)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        return merged
    counts = Counter(_stable_json(value) for value in nonempty)
    return encoded_values[max(counts, key=lambda key: (counts[key], _value_score(encoded_values[key])))]


def _merge_properties(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    fields = sorted(set().union(*(record.keys() for record in records)))
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for field in fields:
        values = [_clean_json(record.get(field)) for record in records]
        distinct = {_stable_json(value) for value in values if value not in (None, "", [], {})}
        if len(distinct) > 1:
            conflicts.append(field)
        merged[field] = _merge_values(field, values)
    return merged, conflicts


def _slug(value: str, limit: int = 56) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", normalize_text(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")[:limit] or "unknown"


def _derived_author_id(name: str) -> str:
    key = normalized_key(name)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"author:derived:{_slug(name)}:{digest}"


def _safe_node_id(node_id: str, label: str) -> str:
    """Keep IDs compact enough for Neo4j indexes while preserving identity."""
    if len(node_id.encode("utf-8")) <= 512:
        return node_id
    digest = hashlib.sha256(f"{label}|{node_id}".encode("utf-8")).hexdigest()
    return f"{label.lower()}:canonical-hash:{digest}"


def _as_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _as_int(value: Any) -> str | int:
    if value in (None, "") or isinstance(value, bool):
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def _as_float(value: Any) -> str | float:
    if value in (None, "") or isinstance(value, bool):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _list_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_text(item) for item in value if normalize_text(item)]


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
            count += 1
    return count


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")
            count += 1
    return count


def _property_json(props: Mapping[str, Any]) -> str:
    return _stable_json(props)


def _source_json(sources: Iterable[tuple[str, str]]) -> str:
    materialized = sorted(set(sources))
    archive_counts = Counter(archive for archive, _member in materialized)
    return _stable_json({
        "archives": dict(sorted(archive_counts.items())),
        "occurrence_count": len(materialized),
        "sample_members": [
            {"archive": archive, "member": member}
            for archive, member in materialized[:10]
        ],
        "sample_truncated": len(materialized) > 10,
    })


def _paper_search_text(props: Mapping[str, Any], context: Mapping[str, list[str]]) -> str:
    fields = [
        ("TITLE", normalize_text(props.get("title"))),
        ("ABSTRACT", normalize_text(props.get("abstract"))),
        ("AUTHORS", "; ".join(_list_strings(props.get("authors")))),
        ("KEYWORDS", "; ".join(_list_strings(props.get("keywords")))),
        ("TOPICS", "; ".join(context.get("topics", []))),
        ("METHODS", "; ".join(context.get("methods", []))),
        ("CONFERENCE", "; ".join(context.get("conferences", []))),
        ("VENUE", normalize_text(props.get("venue"))),
        ("YEAR", str(_as_int(props.get("year")) or "")),
        ("PROBLEM", normalize_text(props.get("research_problem"))),
        ("MOTIVATION", normalize_text(props.get("motivation"))),
        ("CONTRIBUTIONS", "; ".join(_list_strings(props.get("contributions")))),
        ("DATASETS", "; ".join(_list_strings(props.get("datasets")))),
        ("RESULT", normalize_text(props.get("result"))),
    ]
    return "\n".join(f"[{label}] {value}" for label, value in fields if value)


def _node_rows(label: str, records: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[str, ...], Iterable[tuple[Any, ...]]]:
    common = ("properties_json", "sources_json", ":LABEL")
    if label == "Paper":
        header = (
            "paper_id:ID(Paper)", "title", "abstract", "arxiv_id", "doi", "url",
            "pdf_url", "year:int", "external:boolean", "venue", "booktitle", "publisher",
            "authors_json", "keywords_json", "datasets_json", "code_resources_json",
            "research_problem", "motivation", "contributions_json", "result", *common,
        )
        rows = (
            (
                node_id, p.get("title", ""), p.get("abstract", ""), p.get("arxiv_id", ""),
                p.get("doi", ""), p.get("url", ""), p.get("pdf_url", ""), _as_int(p.get("year")),
                _as_bool(p.get("external")), p.get("venue", ""), p.get("booktitle", ""),
                p.get("publisher", ""), _stable_json(p.get("authors") or []),
                _stable_json(p.get("keywords") or []), _stable_json(p.get("datasets") or []),
                _stable_json(p.get("code_resources") or []), p.get("research_problem", ""),
                p.get("motivation", ""), _stable_json(p.get("contributions") or []),
                p.get("result", ""), _property_json(p), record["sources_json"], "Paper",
            )
            for node_id, record in sorted(records.items())
            for p in (record["properties"],)
        )
        return header, rows
    id_name = {
        "Author": "author_id:ID(Author)",
        "Institution": "institution_id:ID(Institution)",
        "Method": "method_id:ID(Method)",
        "Funding": "funding_id:ID(Funding)",
        "Topic": "topic_id:ID(Topic)",
        "Conference": "conference_id:ID(Conference)",
        "Venue": "venue_id:ID(Venue)",
    }[label]
    if label == "Author":
        header = (id_name, "name", "position:int", "is_first_author:boolean", "is_corresponding_author:boolean", "derived:boolean", *common)
        rows = (
            (node_id, p.get("name", ""), _as_int(p.get("position")), _as_bool(p.get("is_first_author")),
             _as_bool(p.get("is_corresponding_author")), _as_bool(p.get("derived")),
             _property_json(p), record["sources_json"], label)
            for node_id, record in sorted(records.items()) for p in (record["properties"],)
        )
        return header, rows
    if label == "Method":
        header = (id_name, "name", "content", *common)
        rows = ((node_id, p.get("name", ""), p.get("content", ""), _property_json(p), record["sources_json"], label)
                for node_id, record in sorted(records.items()) for p in (record["properties"],))
        return header, rows
    if label == "Conference":
        header = (id_name, "name", "venue", "year:int", "booktitle", "publisher", *common)
        rows = ((node_id, p.get("name", ""), p.get("venue", ""), _as_int(p.get("year")), p.get("booktitle", ""),
                 p.get("publisher", ""), _property_json(p), record["sources_json"], label)
                for node_id, record in sorted(records.items()) for p in (record["properties"],))
        return header, rows
    header = (id_name, "name", *common)
    rows = ((node_id, p.get("name", ""), _property_json(p), record["sources_json"], label)
            for node_id, record in sorted(records.items()) for p in (record["properties"],))
    return header, rows


def _edge_rows(edge_type: str, records: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> tuple[tuple[str, ...], Iterable[tuple[Any, ...]]]:
    start_label, end_label = EDGE_CONTRACT[edge_type]
    common = ("properties_json", "sources_json", "derived:boolean", ":TYPE")
    if edge_type == "AUTHORED_BY":
        header = (f":START_ID({start_label})", f":END_ID({end_label})", "position:int", "is_first_author:boolean", "is_corresponding_author:boolean", *common)
        rows = ((source, target, _as_int(p.get("position")), _as_bool(p.get("is_first_author")),
                 _as_bool(p.get("is_corresponding_author")), _property_json(p), record["sources_json"],
                 _as_bool(p.get("derived")), edge_type)
                for (source, target, _), record in sorted(records.items()) for p in (record["properties"],))
        return header, rows
    if edge_type == "AFFILIATED_WITH":
        header = (f":START_ID({start_label})", f":END_ID({end_label})", "author_position:int", "is_first_author:boolean", "is_corresponding_author:boolean", *common)
        rows = ((source, target, _as_int(p.get("author_position")), _as_bool(p.get("is_first_author")),
                 _as_bool(p.get("is_corresponding_author")), _property_json(p), record["sources_json"],
                 _as_bool(p.get("derived")), edge_type)
                for (source, target, _), record in sorted(records.items()) for p in (record["properties"],))
        return header, rows
    if edge_type == "CITES":
        header = (f":START_ID({start_label})", f":END_ID({end_label})", "confidence:double", "source", *common)
        rows = ((source, target, _as_float(p.get("confidence")), p.get("source", ""), _property_json(p),
                 record["sources_json"], _as_bool(p.get("derived")), edge_type)
                for (source, target, _), record in sorted(records.items()) for p in (record["properties"],))
        return header, rows
    header = (f":START_ID({start_label})", f":END_ID({end_label})", *common)
    rows = ((source, target, _property_json(p), record["sources_json"], _as_bool(p.get("derived")), edge_type)
            for (source, target, _), record in sorted(records.items()) for p in (record["properties"],))
    return header, rows


def _schema_cypher() -> str:
    constraints = [
        f"CREATE CONSTRAINT rich_{label.lower()}_id IF NOT EXISTS FOR ({label.lower()}_id_node:{label}) REQUIRE {label.lower()}_id_node.{label.lower()}_id IS UNIQUE;"
        for label in NODE_LABELS
        if label != "Paper"
    ]
    constraints.insert(0, "CREATE CONSTRAINT rich_paper_id IF NOT EXISTS FOR (paper_id_node:Paper) REQUIRE paper_id_node.paper_id IS UNIQUE;")
    indexes = [
        "CREATE INDEX rich_paper_year IF NOT EXISTS FOR (paper_year_node:Paper) ON (paper_year_node.year);",
        "CREATE INDEX rich_paper_arxiv IF NOT EXISTS FOR (paper_arxiv_node:Paper) ON (paper_arxiv_node.arxiv_id);",
        "CREATE INDEX rich_paper_doi IF NOT EXISTS FOR (paper_doi_node:Paper) ON (paper_doi_node.doi);",
    ]
    for label in NODE_LABELS[1:]:
        indexes.append(f"CREATE INDEX rich_{label.lower()}_name IF NOT EXISTS FOR ({label.lower()}_name_node:{label}) ON ({label.lower()}_name_node.name);")
    return "\n".join(constraints + indexes) + "\n"


def _import_script() -> str:
    # Labels and relationship types are already provided by each CSV's
    # ``:LABEL``/``:TYPE`` column, so file-only options avoid CLI ambiguity.
    node_args = " \\\n".join(f"  --nodes=paper_kg/{NODE_FILES[label]}" for label in NODE_LABELS)
    edge_args = " \\\n".join(f"  --relationships=paper_kg/{EDGE_FILES[edge_type]}" for edge_type in EDGE_CONTRACT)
    return f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 NEO4J_ADMIN DATABASE" >&2
  exit 2
fi

neo4j_admin=$1
database=$2
cd "$(dirname "$0")"
"$neo4j_admin" database import full "$database" \\
  --overwrite-destination=false \\
  --id-type=string \\
  --input-encoding=UTF-8 \\
  --multiline-fields=true \\
  --strict=true \\
  --bad-tolerance=0 \\
  --skip-bad-relationships=false \\
  --skip-duplicate-nodes=false \\
{node_args} \\
{edge_args}
"""


def build_rich_graph(zip_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    paths = sorted({Path(path).resolve() for path in zip_paths}, key=lambda path: path.name)
    output_dir = Path(output_dir).resolve()
    if not paths:
        raise ValueError("at least one canonical graph ZIP is required")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("all canonical graph ZIP inputs must exist")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    node_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_occurrences: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    input_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    raw_node_counts: Counter[str] = Counter()
    raw_edge_counts: Counter[str] = Counter()
    main_papers_by_archive: Counter[str] = Counter()
    raw_dangling = 0

    for path in paths:
        archive_name = path.name
        archive_nodes: Counter[str] = Counter()
        archive_edges: Counter[str] = Counter()
        json_files = 0
        malformed = 0
        with zipfile.ZipFile(path) as archive:
            members = sorted(member for member in archive.namelist() if member.lower().endswith(".json"))
            for member in members:
                json_files += 1
                try:
                    payload = json.loads(archive.read(member))
                    graph = payload.get("graph") if isinstance(payload, dict) else None
                    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
                        raise ValueError("expected graph.nodes and graph.edges arrays")
                    local_labels: dict[str, str] = {}
                    local_ids: dict[str, str] = {}
                    for node in graph["nodes"]:
                        if not isinstance(node, dict):
                            raise ValueError("node is not an object")
                        raw_node_id = normalize_text(node.get("id"))
                        node_id = raw_node_id
                        label = normalize_text(node.get("label"))
                        if not node_id or label not in NODE_LABELS:
                            raise ValueError(f"invalid node id/label: {node_id!r}/{label!r}")
                        previous = local_labels.get(raw_node_id)
                        if previous and previous != label:
                            raise ValueError(f"node ID has two labels in one graph: {raw_node_id}")
                        safe_id = _safe_node_id(raw_node_id, label)
                        local_labels[raw_node_id] = label
                        local_ids[raw_node_id] = safe_id
                        props = _clean_json(node.get("properties") or {})
                        if not isinstance(props, dict):
                            raise ValueError("node properties must be an object")
                        if safe_id != raw_node_id:
                            props.setdefault("canonical_id", raw_node_id)
                        node_occurrences[safe_id].append({"label": label, "properties": props, "source": (archive_name, member)})
                        archive_nodes[label] += 1
                        raw_node_counts[label] += 1
                        if label == "Paper" and props.get("external") is False:
                            main_papers_by_archive[archive_name] += 1
                    for edge in graph["edges"]:
                        if not isinstance(edge, dict):
                            raise ValueError("edge is not an object")
                        raw_source = normalize_text(edge.get("source"))
                        raw_target = normalize_text(edge.get("target"))
                        source = local_ids.get(raw_source, raw_source)
                        target = local_ids.get(raw_target, raw_target)
                        edge_type = normalize_text(edge.get("type"))
                        if not source or not target or edge_type not in EDGE_CONTRACT:
                            raise ValueError(f"invalid edge: {source!r}/{target!r}/{edge_type!r}")
                        if raw_source not in local_labels or raw_target not in local_labels:
                            raw_dangling += 1
                        expected = EDGE_CONTRACT[edge_type]
                        actual = (local_labels.get(raw_source), local_labels.get(raw_target))
                        if actual != expected:
                            raise ValueError(f"edge contract mismatch {edge_type}: expected {expected}, got {actual}")
                        props = _clean_json(edge.get("properties") or {})
                        if not isinstance(props, dict):
                            raise ValueError("edge properties must be an object")
                        edge_occurrences[(source, target, edge_type)].append({"properties": props, "source": (archive_name, member)})
                        archive_edges[edge_type] += 1
                        raw_edge_counts[edge_type] += 1
                except Exception as exc:  # keep a complete audit trail; readiness fails below
                    malformed += 1
                    errors.append({"archive": archive_name, "member": member, "error": f"{type(exc).__name__}: {exc}"})
        input_reports.append({
            "path": str(path), "archive": archive_name, "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "json_files": json_files, "malformed_json_files": malformed,
            "raw_nodes": dict(sorted(archive_nodes.items())), "raw_edges": dict(sorted(archive_edges.items())),
            "main_papers": main_papers_by_archive[archive_name],
        })

    merged_nodes: dict[str, dict[str, Any]] = {}
    node_conflict_counts: Counter[str] = Counter()
    node_conflict_examples: list[dict[str, Any]] = []
    for node_id, occurrences in sorted(node_occurrences.items()):
        labels = {record["label"] for record in occurrences}
        if len(labels) != 1:
            errors.append({"archive": "<merged>", "member": node_id, "error": f"node label conflict: {sorted(labels)}"})
            continue
        label = next(iter(labels))
        property_records = [record["properties"] for record in occurrences]
        props, conflicts = _merge_properties(property_records)
        # A referenced paper can later arrive as a main paper.  Main-paper
        # metadata is authoritative and must not be replaced by a longer raw
        # bibliography string from an earlier external citation record.
        if label == "Paper":
            main_records = [record for record in property_records if record.get("external") is False]
            if main_records:
                main_props, _main_conflicts = _merge_properties(main_records)
                for field, value in main_props.items():
                    if value not in (None, "", [], {}):
                        props[field] = value
                props["external"] = False
        for field in conflicts:
            node_conflict_counts[f"{label}.{field}"] += 1
        if conflicts and len(node_conflict_examples) < 100:
            node_conflict_examples.append({"node_id": node_id, "label": label, "fields": conflicts, "occurrences": len(occurrences)})
        merged_nodes[node_id] = {
            "label": label, "properties": props,
            "sources": {record["source"] for record in occurrences},
        }

    merged_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_conflict_counts: Counter[str] = Counter()
    edge_conflict_examples: list[dict[str, Any]] = []
    for edge_key, occurrences in sorted(edge_occurrences.items()):
        props, conflicts = _merge_properties([record["properties"] for record in occurrences])
        for field in conflicts:
            edge_conflict_counts[f"{edge_key[2]}.{field}"] += 1
        if conflicts and len(edge_conflict_examples) < 100:
            edge_conflict_examples.append({"source": edge_key[0], "target": edge_key[1], "type": edge_key[2], "fields": conflicts, "occurrences": len(occurrences)})
        merged_edges[edge_key] = {"properties": props, "sources": {record["source"] for record in occurrences}}

    # Copy deterministic conference context onto main papers for efficient
    # filtering while retaining the canonical Paper→Conference→Venue path.
    papers_enriched_from_conference = 0
    for (paper_id, conference_id, edge_type), _record in sorted(merged_edges.items()):
        if edge_type != "PUBLISHED_IN":
            continue
        paper = merged_nodes.get(paper_id)
        conference = merged_nodes.get(conference_id)
        if not paper or not conference or paper["label"] != "Paper" or conference["label"] != "Conference":
            continue
        if paper["properties"].get("external") is not False:
            continue
        changed = False
        for field in ("year", "venue", "booktitle", "publisher"):
            if paper["properties"].get(field) in (None, "", [], {}) and conference["properties"].get(field) not in (None, "", [], {}):
                paper["properties"][field] = conference["properties"][field]
                changed = True
        if changed:
            papers_enriched_from_conference += 1

    # Complete the author list for main conference papers.  Raw graphs retain
    # only the first author edge, while Paper.authors contains the ordered list.
    author_name_ids: dict[str, set[str]] = defaultdict(set)
    for node_id, record in merged_nodes.items():
        if record["label"] == "Author":
            name_key = normalized_key(record["properties"].get("name"))
            if name_key:
                author_name_ids[name_key].add(node_id)
    raw_author_ids_by_paper_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (source, target, edge_type), _record in merged_edges.items():
        if edge_type != "AUTHORED_BY" or target not in merged_nodes:
            continue
        name_key = normalized_key(merged_nodes[target]["properties"].get("name"))
        if name_key:
            raw_author_ids_by_paper_name[(source, name_key)].add(target)
    derived_author_nodes = 0
    derived_authored_edges = 0
    ambiguous_author_names = 0
    for paper_id, paper in sorted(merged_nodes.items()):
        if paper["label"] != "Paper" or paper["properties"].get("external") is not False:
            continue
        for position, name in enumerate(_list_strings(paper["properties"].get("authors")), start=1):
            name_key = normalized_key(name)
            raw_candidates = sorted(raw_author_ids_by_paper_name.get((paper_id, name_key), ()))
            candidates = raw_candidates or sorted(author_name_ids.get(name_key, ()))
            if len(candidates) > 1:
                ambiguous_author_names += 1
            author_id = candidates[0] if candidates else _derived_author_id(name)
            if not candidates:
                if author_id not in merged_nodes:
                    merged_nodes[author_id] = {
                        "label": "Author",
                        "properties": {"name": name, "derived": True},
                        "sources": set(paper["sources"]),
                    }
                    author_name_ids[name_key].add(author_id)
                    derived_author_nodes += 1
            edge_key = (paper_id, author_id, "AUTHORED_BY")
            derived_props = {
                "position": position,
                "is_first_author": position == 1,
                "is_corresponding_author": False,
                "derived": True,
                "source": "Paper.authors",
            }
            if edge_key in merged_edges:
                existing = merged_edges[edge_key]
                props, _ = _merge_properties([existing["properties"], derived_props])
                # A raw relationship remains raw even when supplemented.
                props["derived"] = bool(existing["properties"].get("derived", False))
                existing["properties"] = props
                existing["sources"].update(paper["sources"])
            else:
                merged_edges[edge_key] = {"properties": derived_props, "sources": set(paper["sources"])}
                derived_authored_edges += 1

    nodes_by_label: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in NODE_LABELS}
    for node_id, record in merged_nodes.items():
        record["sources_json"] = _source_json(record["sources"])
        nodes_by_label[record["label"]][node_id] = record
    edges_by_type: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {edge_type: {} for edge_type in EDGE_CONTRACT}
    dangling_edges: list[dict[str, str]] = []
    contract_mismatches: list[dict[str, str]] = []
    for edge_key, record in merged_edges.items():
        source, target, edge_type = edge_key
        if source not in merged_nodes or target not in merged_nodes:
            dangling_edges.append({"source": source, "target": target, "type": edge_type})
            continue
        expected = EDGE_CONTRACT[edge_type]
        actual = (merged_nodes[source]["label"], merged_nodes[target]["label"])
        if actual != expected:
            contract_mismatches.append({"source": source, "target": target, "type": edge_type, "actual": str(actual)})
            continue
        record["sources_json"] = _source_json(record["sources"])
        edges_by_type[edge_type][edge_key] = record

    main_paper_ids = {
        node_id for node_id, record in nodes_by_label["Paper"].items()
        if record["properties"].get("external") is False
    }
    paper_context: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (source, target, edge_type), _record in merged_edges.items():
        if source not in main_paper_ids or target not in merged_nodes:
            continue
        target_props = merged_nodes[target]["properties"]
        name = normalize_text(target_props.get("name"))
        if edge_type == "HAS_TOPIC" and name:
            paper_context[source]["topics"].append(name)
        elif edge_type in {"PROPOSES", "USES_AS_BASELINE"} and name:
            paper_context[source]["methods"].append(name)
        elif edge_type == "PUBLISHED_IN" and name:
            paper_context[source]["conferences"].append(name)
    retrieval_rows = []
    for paper_id in sorted(main_paper_ids):
        props = merged_nodes[paper_id]["properties"]
        if not normalize_text(props.get("title")) or is_placeholder_abstract(props.get("abstract")):
            continue
        context = {key: list(dict.fromkeys(values)) for key, values in paper_context.get(paper_id, {}).items()}
        retrieval_rows.append({
            "paper_id": paper_id,
            "source_id": normalize_text(props.get("arxiv_id")),
            "title": normalize_text(props.get("title")),
            "abstract": normalize_text(props.get("abstract")),
            "authors": _list_strings(props.get("authors")),
            "conference": (context.get("conferences") or [normalize_text(props.get("booktitle"))])[0],
            "venue": normalize_text(props.get("venue")),
            "year": _as_int(props.get("year")) or None,
            "subjects": [],
            "keywords": _list_strings(props.get("keywords")),
            "topics": context.get("topics", []),
            "methods": context.get("methods", []),
            "search_text": _paper_search_text(props, context),
        })

    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        graph_root = temp_root / "paper_kg"
        report_root = temp_root / "reports"
        graph_root.mkdir(parents=True)
        report_root.mkdir(parents=True)
        written_nodes: dict[str, int] = {}
        written_edges: dict[str, int] = {}
        for label in NODE_LABELS:
            header, rows = _node_rows(label, nodes_by_label[label])
            written_nodes[label] = _write_csv(graph_root / NODE_FILES[label], header, rows)
        for edge_type in EDGE_CONTRACT:
            header, rows = _edge_rows(edge_type, edges_by_type[edge_type])
            written_edges[edge_type] = _write_csv(graph_root / EDGE_FILES[edge_type], header, rows)
        retrieval_count = _write_jsonl(temp_root / "retrieval_documents.jsonl", retrieval_rows)
        (temp_root / "constraints.cypher").write_text(_schema_cypher(), encoding="utf-8")
        import_script = temp_root / "import_neo4j_admin.sh"
        import_script.write_text(_import_script(), encoding="utf-8")
        import_script.chmod(0o755)

        report = {
            "schema": SCHEMA,
            "created_from": "canonical graph JSON archives",
            "inputs": input_reports,
            "counts": {
                "input_archives": len(paths),
                "source_json_files": sum(item["json_files"] for item in input_reports),
                "malformed_json_files": len(errors),
                "raw_nodes": sum(raw_node_counts.values()),
                "raw_edges": sum(raw_edge_counts.values()),
                "unique_nodes": sum(written_nodes.values()),
                "unique_edges": sum(written_edges.values()),
                "duplicate_node_rows_removed": sum(raw_node_counts.values()) - len(node_occurrences),
                "duplicate_edge_rows_removed": sum(raw_edge_counts.values()) - len(edge_occurrences),
                "main_papers": len(main_paper_ids),
                "external_papers": written_nodes["Paper"] - len(main_paper_ids),
                "retrieval_documents": retrieval_count,
                "derived_author_nodes": derived_author_nodes,
                "derived_authored_by_edges": derived_authored_edges,
                "papers_enriched_from_conference": papers_enriched_from_conference,
                "ambiguous_author_name_reuses": ambiguous_author_names,
                "raw_dangling_edges": raw_dangling,
                "final_dangling_edges": len(dangling_edges),
                "relationship_contract_mismatches": len(contract_mismatches),
            },
            "nodes_by_label": written_nodes,
            "raw_nodes_by_label": dict(sorted(raw_node_counts.items())),
            "edges_by_type": written_edges,
            "raw_edges_by_type": dict(sorted(raw_edge_counts.items())),
            "duplicates": {
                "node_ids_with_multiple_occurrences": sum(len(items) > 1 for items in node_occurrences.values()),
                "edge_triples_with_multiple_occurrences": sum(len(items) > 1 for items in edge_occurrences.values()),
            },
            "property_conflicts": {
                "node_field_counts": dict(sorted(node_conflict_counts.items())),
                "edge_field_counts": dict(sorted(edge_conflict_counts.items())),
                "node_examples_file": "reports/node_property_conflict_examples.json",
                "edge_examples_file": "reports/edge_property_conflict_examples.json",
            },
            "quality_gates": {
                "source_archives_immutable": True,
                "all_json_parsed": not errors,
                "all_node_ids_unique": sum(written_nodes.values()) == len(merged_nodes),
                "all_edge_triples_unique": sum(written_edges.values()) == len(merged_edges),
                "no_dangling_edges": not dangling_edges,
                "all_relationship_labels_match_contract": not contract_mismatches,
                "all_eight_node_labels_present": all(written_nodes[label] > 0 for label in NODE_LABELS),
                "all_nine_edge_types_present": all(written_edges[edge_type] > 0 for edge_type in EDGE_CONTRACT),
                "neo4j_import_candidate_ready": False,
            },
            "notes": {
                "dataset_and_code": "Paper.datasets and Paper.code_resources are retained as JSON properties; canonical inputs do not model Dataset/Code as nodes.",
                "institutions": "Only source AFFILIATED_WITH facts are retained; missing affiliations are not inferred.",
                "authors": "Missing main-paper authors are deterministically reconstructed from Paper.authors and marked derived=true.",
            },
        }
        report["quality_gates"]["neo4j_import_candidate_ready"] = all(
            value for key, value in report["quality_gates"].items()
            if key not in {"source_archives_immutable", "neo4j_import_candidate_ready"}
        )
        (temp_root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (report_root / "node_property_conflict_examples.json").write_text(json.dumps(node_conflict_examples, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (report_root / "edge_property_conflict_examples.json").write_text(json.dumps(edge_conflict_examples, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (report_root / "malformed_or_rejected_records.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (report_root / "dangling_edges.json").write_text(json.dumps(dangling_edges, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (report_root / "relationship_contract_mismatches.json").write_text(json.dumps(contract_mismatches, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "inputs": [{key: item[key] for key in ("path", "archive", "bytes", "sha256")} for item in input_reports],
            "node_files": NODE_FILES,
            "relationship_files": EDGE_FILES,
            "identity": {"nodes": "canonical id", "relationships": ["source", "target", "type"]},
            "outputs": {},
        }
        tracked_outputs = [
            *(f"paper_kg/{NODE_FILES[label]}" for label in NODE_LABELS),
            *(f"paper_kg/{EDGE_FILES[edge_type]}" for edge_type in EDGE_CONTRACT),
            "retrieval_documents.jsonl", "constraints.cypher", "import_neo4j_admin.sh",
        ]
        for relative in tracked_outputs:
            output_path = temp_root / relative
            manifest["outputs"][relative] = {
                "bytes": output_path.stat().st_size,
                "sha256": file_sha256(output_path),
            }
        (temp_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_root.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_paths", action="append", type=Path, required=True, help="canonical graph ZIP; repeat for every archive")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_rich_graph(args.zip_paths, args.output_dir)
    print(json.dumps({"counts": report["counts"], "quality_gates": report["quality_gates"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["quality_gates"]["neo4j_import_candidate_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
