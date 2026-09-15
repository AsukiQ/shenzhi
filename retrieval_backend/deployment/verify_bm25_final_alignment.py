#!/usr/bin/env python3
"""Compare a BM25 SQLite index with active papers in a rich-graph archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import tarfile


PAPERS_MEMBER = "merged_rich_graph/paper_kg/papers.csv"


def _is_true(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    args = parser.parse_args()

    excluded = set(args.exclude_id)
    graph_ids: set[str] = set()
    active_ids: set[str] = set()
    with tarfile.open(args.archive, "r:gz") as archive:
        raw = archive.extractfile(PAPERS_MEMBER)
        if raw is None:
            raise RuntimeError(f"archive member not found: {PAPERS_MEMBER}")
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            id_column = next((name for name in reader.fieldnames or [] if name.startswith("paper_id")), None)
            external_column = next((name for name in reader.fieldnames or [] if name.startswith("external")), None)
            if not id_column or not external_column:
                raise RuntimeError(f"unexpected papers header: {reader.fieldnames}")
            for row in reader:
                paper_id = str(row.get(id_column) or "").strip()
                if not paper_id:
                    continue
                graph_ids.add(paper_id)
                if not _is_true(row.get(external_column)) and paper_id not in excluded:
                    active_ids.add(paper_id)

    connection = sqlite3.connect(f"file:{args.db}?mode=ro&immutable=1", uri=True)
    try:
        index_ids = {str(row[0]) for row in connection.execute("SELECT paper_id FROM papers")}
    finally:
        connection.close()

    missing_in_graph = sorted(index_ids - graph_ids)
    inactive_in_index = sorted(index_ids - active_ids)
    active_missing_index = sorted(active_ids - index_ids)
    report = {
        "graph_papers": len(graph_ids),
        "active_graph_papers_after_exclusions": len(active_ids),
        "index_papers": len(index_ids),
        "missing_in_graph": len(missing_in_graph),
        "inactive_or_quarantined_in_index": len(inactive_in_index),
        "active_graph_papers_without_retrieval_document": len(active_missing_index),
        "excluded_ids": sorted(excluded),
        "examples": {
            "missing_in_graph": missing_in_graph[:20],
            "inactive_or_quarantined_in_index": inactive_in_index[:20],
            "active_without_document": active_missing_index[:20],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if missing_in_graph or inactive_in_index:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
