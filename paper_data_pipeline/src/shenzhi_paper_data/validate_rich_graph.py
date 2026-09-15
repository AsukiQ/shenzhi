"""Validate a rich-schema Neo4j CSV candidate without changing it."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .build_rich_graph import EDGE_CONTRACT, EDGE_FILES, NODE_FILES, NODE_LABELS, SCHEMA
from .prepare import file_sha256


csv.field_size_limit(sys.maxsize)


def _id_column(header: list[str], marker: str) -> str:
    columns = [column for column in header if marker in column]
    if len(columns) != 1:
        raise ValueError(f"expected one {marker} column, got {columns}")
    return columns[0]


def validate_rich_graph(candidate_dir: Path, *, verify_inputs: bool = True) -> dict[str, Any]:
    root = Path(candidate_dir).resolve()
    manifest_path = root / "manifest.json"
    quality_path = root / "quality_report.json"
    if not manifest_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError("candidate requires manifest.json and quality_report.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append(f"manifest schema mismatch: {manifest.get('schema')!r}")

    output_hash_mismatches: list[str] = []
    for relative, expected in sorted((manifest.get("outputs") or {}).items()):
        path = root / relative
        if not path.is_file():
            output_hash_mismatches.append(f"missing:{relative}")
            continue
        if path.stat().st_size != int(expected.get("bytes", -1)):
            output_hash_mismatches.append(f"bytes:{relative}")
        if file_sha256(path) != expected.get("sha256"):
            output_hash_mismatches.append(f"sha256:{relative}")

    input_hash_mismatches: list[str] = []
    if verify_inputs:
        for item in manifest.get("inputs") or []:
            path = Path(item["path"])
            if not path.is_file():
                input_hash_mismatches.append(f"missing:{path}")
                continue
            if path.stat().st_size != int(item.get("bytes", -1)):
                input_hash_mismatches.append(f"bytes:{path}")
            if file_sha256(path) != item.get("sha256"):
                input_hash_mismatches.append(f"sha256:{path}")

    node_labels: dict[str, str] = {}
    node_counts: Counter[str] = Counter()
    duplicate_node_ids = 0
    invalid_property_json = 0
    for label in NODE_LABELS:
        path = root / "paper_kg" / NODE_FILES[label]
        if not path.is_file():
            errors.append(f"missing node file: {path.name}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                errors.append(f"missing CSV header: {path.name}")
                continue
            try:
                id_column = _id_column(reader.fieldnames, ":ID(")
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            for row in reader:
                node_id = row.get(id_column, "")
                if not node_id:
                    errors.append(f"blank node ID in {path.name}")
                    continue
                if node_id in node_labels:
                    duplicate_node_ids += 1
                else:
                    node_labels[node_id] = label
                node_counts[label] += 1
                try:
                    properties = json.loads(row.get("properties_json", ""))
                    sources = json.loads(row.get("sources_json", ""))
                    if not isinstance(properties, dict) or not isinstance(sources, dict):
                        raise ValueError("JSON columns require objects")
                except (json.JSONDecodeError, ValueError):
                    invalid_property_json += 1

    edge_counts: Counter[str] = Counter()
    duplicate_edge_triples = 0
    dangling_edges = 0
    contract_mismatches = 0
    invalid_edge_property_json = 0
    seen_edges: set[tuple[str, str, str]] = set()
    for edge_type, (start_label, end_label) in EDGE_CONTRACT.items():
        path = root / "paper_kg" / EDGE_FILES[edge_type]
        if not path.is_file():
            errors.append(f"missing relationship file: {path.name}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                errors.append(f"missing CSV header: {path.name}")
                continue
            try:
                start_column = _id_column(reader.fieldnames, ":START_ID(")
                end_column = _id_column(reader.fieldnames, ":END_ID(")
                type_column = _id_column(reader.fieldnames, ":TYPE")
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            for row in reader:
                source, target, actual_type = row.get(start_column, ""), row.get(end_column, ""), row.get(type_column, "")
                triple = (source, target, actual_type)
                if triple in seen_edges:
                    duplicate_edge_triples += 1
                else:
                    seen_edges.add(triple)
                if source not in node_labels or target not in node_labels:
                    dangling_edges += 1
                elif node_labels[source] != start_label or node_labels[target] != end_label or actual_type != edge_type:
                    contract_mismatches += 1
                edge_counts[edge_type] += 1
                try:
                    properties = json.loads(row.get("properties_json", ""))
                    sources = json.loads(row.get("sources_json", ""))
                    if not isinstance(properties, dict) or not isinstance(sources, dict):
                        raise ValueError("JSON columns require objects")
                except (json.JSONDecodeError, ValueError):
                    invalid_edge_property_json += 1

    expected_nodes = expected_quality.get("nodes_by_label") or {}
    expected_edges = expected_quality.get("edges_by_type") or {}
    count_mismatches = {
        "nodes": {
            label: {"expected": expected_nodes.get(label), "actual": node_counts[label]}
            for label in NODE_LABELS if expected_nodes.get(label) != node_counts[label]
        },
        "edges": {
            edge_type: {"expected": expected_edges.get(edge_type), "actual": edge_counts[edge_type]}
            for edge_type in EDGE_CONTRACT if expected_edges.get(edge_type) != edge_counts[edge_type]
        },
    }
    ready = not any((
        errors,
        output_hash_mismatches,
        input_hash_mismatches,
        duplicate_node_ids,
        duplicate_edge_triples,
        dangling_edges,
        contract_mismatches,
        invalid_property_json,
        invalid_edge_property_json,
        count_mismatches["nodes"],
        count_mismatches["edges"],
    ))
    return {
        "schema": "shenzhi_neo4j_rich_validation_v1",
        "candidate": str(root),
        "ready": ready,
        "node_counts": dict(node_counts),
        "edge_counts": dict(edge_counts),
        "checks": {
            "duplicate_node_ids": duplicate_node_ids,
            "duplicate_edge_triples": duplicate_edge_triples,
            "dangling_edges": dangling_edges,
            "relationship_contract_mismatches": contract_mismatches,
            "invalid_node_json_columns": invalid_property_json,
            "invalid_edge_json_columns": invalid_edge_property_json,
            "output_hash_mismatches": output_hash_mismatches,
            "input_hash_mismatches": input_hash_mismatches,
            "count_mismatches": count_mismatches,
            "errors": errors,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--skip-input-hashes", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_rich_graph(args.candidate_dir, verify_inputs=not args.skip_input_hashes)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
