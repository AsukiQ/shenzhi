#!/usr/bin/env python3
"""Generate review-only loose-normalization entity candidate groups."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import tarfile
import unicodedata


ENTITIES = {
    "merged_rich_graph/paper_kg/authors.csv": ("Author", "author_id"),
    "merged_rich_graph/paper_kg/topics.csv": ("Topic", "topic_id"),
    "merged_rich_graph/paper_kg/methods.csv": ("Method", "method_id"),
    "merged_rich_graph/paper_kg/institutions.csv": ("Institution", "institution_id"),
    "merged_rich_graph/paper_kg/fundings.csv": ("Funding", "funding_id"),
}


def display_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def loose_key(value: str) -> str:
    return "".join(character for character in display_key(value) if character.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-members", type=int, default=100)
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[dict[str, str]]]] = {
        label: {} for label, _id_prefix in ENTITIES.values()
    }
    node_counts = {label: 0 for label in grouped}

    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            config = ENTITIES.get(member.name)
            if config is None:
                continue
            label, id_prefix = config
            raw = archive.extractfile(member)
            if raw is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            reader = csv.DictReader(line.decode("utf-8") for line in raw)
            id_column = next(
                (name for name in reader.fieldnames or [] if name.startswith(id_prefix)),
                None,
            )
            if id_column is None or "name" not in (reader.fieldnames or []):
                raise RuntimeError(f"unexpected {label} header: {reader.fieldnames}")
            for row in reader:
                entity_id = str(row.get(id_column) or "").strip()
                name = str(row.get("name") or "").strip()
                if not entity_id or not name:
                    continue
                node_counts[label] += 1
                key = loose_key(name)
                if not key:
                    continue
                grouped[label].setdefault(key, []).append({"id": entity_id, "name": name})

    result_groups: dict[str, list[dict[str, object]]] = {}
    summary: dict[str, dict[str, int]] = {}
    for label, groups in grouped.items():
        candidates = []
        candidate_nodes = 0
        truncated_groups = 0
        for key, members in groups.items():
            if len(members) < 2:
                continue
            candidate_nodes += len(members)
            spellings = {display_key(member["name"]) for member in members}
            risk = "exact_name_multiple_ids" if len(spellings) == 1 else "punctuation_or_spacing_collision"
            ordered = sorted(members, key=lambda item: (display_key(item["name"]), item["id"]))
            truncated = len(ordered) > args.max_members
            truncated_groups += int(truncated)
            candidates.append(
                {
                    "normalized_key": key,
                    "risk": risk,
                    "member_count": len(ordered),
                    "members": ordered[: args.max_members],
                    "members_truncated": truncated,
                }
            )
        candidates.sort(key=lambda group: (-int(group["member_count"]), str(group["normalized_key"])))
        result_groups[label] = candidates
        summary[label] = {
            "nodes_scanned": node_counts[label],
            "candidate_groups": len(candidates),
            "candidate_nodes": candidate_nodes,
            "truncated_groups": truncated_groups,
        }

    report = {
        "generated_at": "2026-08-21T20:05:00+08:00",
        "source": str(Path(args.archive).resolve()),
        "mode": "review_only",
        "automatic_merges_applied": 0,
        "normalization": "Unicode NFKC + casefold + remove all non-alphanumeric characters",
        "warning": "A shared normalized key is not proof of identity. Examples such as ACE and ACE++ must not be merged automatically.",
        "summary": summary,
        "groups": result_groups,
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
