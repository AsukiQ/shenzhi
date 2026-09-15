#!/usr/bin/env python3
"""Generate record-level Paper identity candidates without mutating the graph."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import tarfile
import unicodedata


PAPERS_MEMBER = "merged_rich_graph/paper_kg/papers.csv"
FIELDS = ("paper_id", "title", "arxiv_id", "doi", "year", "venue", "url", "pdf_url")


def _text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _is_true(value: object) -> bool:
    return _text(value).casefold() in {"1", "true", "yes"}


def normalize_arxiv(value: object) -> str:
    text = _text(value).casefold()
    text = re.sub(r"^arxiv\s*:\s*", "", text)
    text = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip(" /.")


def normalize_title(value: object) -> str:
    return "".join(character for character in _text(value).casefold() if character.isalnum())


def normalize_doi(value: object) -> str:
    text = _text(value).casefold()
    text = re.sub(r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)", "", text)
    return text.strip().rstrip(".,;)")


def _paper_rows(archive_path: str):
    with tarfile.open(archive_path, "r:gz") as archive:
        raw = archive.extractfile(PAPERS_MEMBER)
        if raw is None:
            raise RuntimeError(f"archive member not found: {PAPERS_MEMBER}")
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            fieldnames = reader.fieldnames or []
            columns = {
                field: next((name for name in fieldnames if name == field or name.startswith(f"{field}:")), None)
                for field in FIELDS
            }
            if columns["paper_id"] is None:
                raise RuntimeError(f"unexpected papers header: {fieldnames}")
            external_column = next((name for name in fieldnames if name.startswith("external")), None)
            if external_column is None:
                raise RuntimeError(f"external column missing: {fieldnames}")
            for row in reader:
                record = {field: _text(row.get(column)) if column else "" for field, column in columns.items()}
                record["external"] = _is_true(row.get(external_column))
                yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    normalizers = {
        "normalized_arxiv": ("arxiv_id", normalize_arxiv),
        "normalized_title": ("title", normalize_title),
        "normalized_doi": ("doi", normalize_doi),
    }
    first_ids: dict[str, dict[str, str]] = {kind: {} for kind in normalizers}
    group_ids: dict[str, dict[str, list[str]]] = {kind: {} for kind in normalizers}
    active_papers = 0

    for record in _paper_rows(args.archive):
        if record["external"]:
            continue
        active_papers += 1
        paper_id = record["paper_id"]
        for kind, (field, normalizer) in normalizers.items():
            key = normalizer(record[field])
            if not key:
                continue
            if key in group_ids[kind]:
                group_ids[kind][key].append(paper_id)
            elif key in first_ids[kind]:
                group_ids[kind][key] = [first_ids[kind].pop(key), paper_id]
            else:
                first_ids[kind][key] = paper_id

    candidate_ids = {
        paper_id
        for groups in group_ids.values()
        for members in groups.values()
        for paper_id in members
    }
    records_by_id = {
        record["paper_id"]: record
        for record in _paper_rows(args.archive)
        if record["paper_id"] in candidate_ids
    }

    report_groups: dict[str, list[dict[str, object]]] = {}
    summary: dict[str, dict[str, int]] = {}
    for kind, groups in group_ids.items():
        items = []
        extra_nodes = 0
        for key, ids in groups.items():
            records = [records_by_id[paper_id] for paper_id in ids]
            conflicts = []
            for field in ("title", "arxiv_id", "doi", "year", "venue"):
                values = sorted({_text(record[field]) for record in records if _text(record[field])})
                if len(values) > 1:
                    conflicts.append({"field": field, "values": values})
            invalid_identity_value = kind == "normalized_doi" and key == "10.18653/v1"
            if invalid_identity_value:
                recommendation = "invalid_identifier_quarantine_only"
            elif kind == "normalized_arxiv" and not conflicts:
                recommendation = "strong_candidate_manual_edge_and_property_review"
            else:
                recommendation = "manual_review_required"
            items.append(
                {
                    "normalized_key": key,
                    "member_count": len(records),
                    "invalid_identity_value": invalid_identity_value,
                    "recommendation": recommendation,
                    "property_conflicts": conflicts,
                    "members": records,
                }
            )
            extra_nodes += len(records) - 1
        items.sort(key=lambda item: (-int(item["member_count"]), str(item["normalized_key"])))
        report_groups[kind] = items
        summary[kind] = {
            "candidate_groups": len(items),
            "candidate_nodes": sum(int(item["member_count"]) for item in items),
            "extra_nodes_if_every_group_were_one_identity": extra_nodes,
        }

    report = {
        "generated_at": "2026-08-21T20:15:00+08:00",
        "source": str(Path(args.archive).resolve()),
        "active_papers_scanned": active_papers,
        "mode": "review_only",
        "automatic_merges_applied": 0,
        "warning": "Candidates are not merge instructions. Rewire and deduplicate all incident edges, preserve new-authoritative properties, and review conflicts before any merge.",
        "summary": summary,
        "groups": report_groups,
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
