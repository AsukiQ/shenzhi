"""Prepare a deterministic, deduplicated Neo4j bulk-import directory.

The source ZIP is treated as immutable.  Node CSV files are validated and copied;
relationship CSV files are deduplicated by ``(start_id, end_id, type)`` so the
offline importer has the same one-edge semantics as the source ``MERGE`` scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any
import zipfile


SCHEMA_VERSION = "shenzhi_neo4j_import_v1"
NODE_FILES = ("papers", "authors", "venues", "keywords", "subjects", "years")
RELATION_FILES = (
    "authored_by",
    "published_in",
    "published_year",
    "has_keyword",
    "has_subject",
    "cites",
)
CYPHER_FILES = (
    "constraints.cypher",
    "import_graph_shell.cypher",
    "import_citations_shell.cypher",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_by_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ZIP member ending with {suffix!r}: {matches}")
    return matches[0]


def _reader(archive: zipfile.ZipFile, member: str):
    raw = archive.open(member, "r")
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return raw, text, csv.reader(text)


def _prepare_node(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> dict[str, Any]:
    raw, text, reader = _reader(archive, member)
    try:
        header = next(reader, None)
        if not header or not any(":ID(" in field for field in header):
            raise ValueError(f"node CSV lacks an ID column: {member}")
        id_index = next(index for index, field in enumerate(header) if ":ID(" in field)
        seen: set[str] = set()
        row_count = 0
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(header)
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"node CSV column mismatch in {member} at row {row_count + 2}")
                node_id = str(row[id_index]).strip()
                if not node_id:
                    raise ValueError(f"empty node ID in {member} at row {row_count + 2}")
                if node_id in seen:
                    raise ValueError(f"duplicate node ID in {member}: {node_id}")
                seen.add(node_id)
                writer.writerow(row)
                row_count += 1
    finally:
        text.close()
        raw.close()
    return {
        "kind": "node",
        "source_member": member,
        "row_count": row_count,
        "output_sha256": file_sha256(destination),
    }


def _prepare_relationship(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    sqlite_path: Path,
) -> dict[str, Any]:
    raw, text, reader = _reader(archive, member)
    conn = sqlite3.connect(sqlite_path)
    try:
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty relationship CSV: {member}")
        try:
            start_index = next(i for i, field in enumerate(header) if field.startswith(":START_ID"))
            end_index = next(i for i, field in enumerate(header) if field.startswith(":END_ID"))
            type_index = header.index(":TYPE")
        except (StopIteration, ValueError) as exc:
            raise ValueError(f"relationship CSV has no start/end/type columns: {member}") from exc
        conn.execute(
            "CREATE TABLE rel ("
            "edge_key TEXT PRIMARY KEY, first_seen INTEGER NOT NULL, row_json TEXT NOT NULL)"
        )
        source_rows = 0
        conflicting_duplicates = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"relationship CSV column mismatch in {member} at row {source_rows + 2}"
                )
            start_id = str(row[start_index]).strip()
            end_id = str(row[end_index]).strip()
            relation_type = str(row[type_index]).strip()
            if not start_id or not end_id or not relation_type:
                raise ValueError(
                    f"empty relationship endpoint/type in {member} at row {source_rows + 2}"
                )
            edge_key = "\0".join((start_id, end_id, relation_type))
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            existing = conn.execute(
                "SELECT row_json FROM rel WHERE edge_key = ?", (edge_key,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO rel(edge_key, first_seen, row_json) VALUES (?, ?, ?)",
                    (edge_key, source_rows, encoded),
                )
            else:
                conflicting_duplicates += int(str(existing[0]) != encoded)
                # Source Cypher uses MERGE followed by SET, so the last duplicate row
                # controls relationship properties while first-seen order stays stable.
                conn.execute(
                    "UPDATE rel SET row_json = ? WHERE edge_key = ?",
                    (encoded, edge_key),
                )
            source_rows += 1
            if source_rows % 100_000 == 0:
                conn.commit()
        conn.commit()
        output_rows = int(conn.execute("SELECT count(*) FROM rel").fetchone()[0])
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(header)
            for (row_json,) in conn.execute(
                "SELECT row_json FROM rel ORDER BY first_seen"
            ):
                writer.writerow(json.loads(row_json))
    finally:
        conn.close()
        text.close()
        raw.close()
        sqlite_path.unlink(missing_ok=True)
    return {
        "kind": "relationship",
        "source_member": member,
        "source_row_count": source_rows,
        "row_count": output_rows,
        "duplicate_row_count": source_rows - output_rows,
        "conflicting_duplicate_count": conflicting_duplicates,
        "output_sha256": file_sha256(destination),
    }


def prepare_import(
    source_zip: str | Path,
    output_dir: str | Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    source_zip = Path(source_zip).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if output_dir.exists() and any(output_dir.iterdir()) and not replace:
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        csv_root = temporary_root / "paper_kg"
        csv_root.mkdir(parents=True)
        files: dict[str, Any] = {}
        with zipfile.ZipFile(source_zip) as archive:
            for stem in NODE_FILES:
                member = _member_by_suffix(archive, f"/paper_kg/{stem}.csv")
                files[f"{stem}.csv"] = _prepare_node(
                    archive, member, csv_root / f"{stem}.csv"
                )
            for stem in RELATION_FILES:
                member = _member_by_suffix(archive, f"/paper_kg/{stem}.csv")
                files[f"{stem}.csv"] = _prepare_relationship(
                    archive,
                    member,
                    csv_root / f"{stem}.csv",
                    temporary_root / f".{stem}.sqlite",
                )
            for filename in CYPHER_FILES:
                member = _member_by_suffix(archive, f"/{filename}")
                target = temporary_root / filename
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                files[filename] = {
                    "kind": "cypher",
                    "source_member": member,
                    "output_sha256": file_sha256(target),
                }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_zip": str(source_zip),
            "source_zip_sha256": file_sha256(source_zip),
            "node_files": list(NODE_FILES),
            "relationship_files": list(RELATION_FILES),
            "files": files,
            "source_is_immutable": True,
            "relationship_identity": ["start_id", "end_id", "type"],
            "duplicate_semantics": "last_properties_first_position_v1",
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            existing_manifest = output_dir / "manifest.json"
            if any(output_dir.iterdir()) and not existing_manifest.is_file():
                raise RuntimeError(
                    "refusing to replace a non-empty directory without a preparation manifest"
                )
            shutil.rmtree(output_dir)
        temporary_root.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    report = prepare_import(args.source_zip, args.output_dir, replace=args.replace)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
