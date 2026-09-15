"""Safely merge an A-layer graph import directory into the existing CSV bundle.

The old import ZIP is the production baseline.  New rows are appended only
when their canonical IDs/relationship keys are absent.  Conflicting properties
for an existing key fail closed; the input ZIP and addition directory are never
modified.  Empty additions with a legacy header (notably ``cites.csv``) are
accepted and keep the baseline header.
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
from typing import Any, Sequence
import zipfile


NODE_FILES = ("papers", "authors", "venues", "keywords", "subjects", "years")
RELATION_FILES = (
    "authored_by",
    "published_in",
    "published_year",
    "has_keyword",
    "has_subject",
    "cites",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_row(row: Sequence[str]) -> str:
    return json.dumps(list(row), ensure_ascii=False, separators=(",", ":"))


def _row_hash(row: Sequence[str]) -> str:
    return hashlib.sha256(_stable_row(row).encode("utf-8")).hexdigest()


def _clean_header(header: Sequence[str]) -> list[str]:
    return [str(field).lstrip("\ufeff") for field in header]


def _zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one ZIP member ending with {suffix!r}, found {matches}")
    return matches[0]


def _csv_reader(archive: zipfile.ZipFile, member: str):
    raw = archive.open(member, "r")
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return raw, text, csv.reader(text)


def _addition_path(root: Path, stem: str) -> Path:
    candidates = [root / "paper_kg" / f"{stem}.csv", root / f"{stem}.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"addition CSV not found for {stem}: {candidates}")


def _read_addition_header(path: Path) -> tuple[list[str], csv.reader, io.TextIOBase]:
    text = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(text)
    header = next(reader, None)
    if not header:
        text.close()
        raise ValueError(f"empty addition CSV: {path}")
    return _clean_header(header), reader, text


def _copy_base_support_files(
    archive: zipfile.ZipFile,
    output_root: Path,
    base_prefix: str,
) -> list[str]:
    copied: list[str] = []
    for member in archive.namelist():
        if member.endswith("/") or "/paper_kg/" in member:
            continue
        if member.startswith(base_prefix + "/"):
            target = output_root / Path(member).name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            copied.append(target.name)
    return copied


def _merge_node(
    archive: zipfile.ZipFile,
    base_member: str,
    addition_path: Path,
    destination: Path,
    conn: sqlite3.Connection,
    stem: str,
) -> dict[str, Any]:
    raw, text, reader = _csv_reader(archive, base_member)
    add_header, add_reader, add_text = _read_addition_header(addition_path)
    try:
        base_header = _clean_header(next(reader, None) or [])
        if base_header != add_header:
            first_addition = next(add_reader, None)
            if first_addition is not None:
                raise ValueError(f"header mismatch for non-empty node CSV {stem}")
            add_empty = True
        else:
            first_addition = None
            add_empty = False
        id_index = next((i for i, field in enumerate(base_header) if ":ID(" in field), None)
        if id_index is None:
            raise ValueError(f"node CSV has no ID column: {stem}")
        table = f'node_{stem.replace("-", "_")}'
        conn.execute(f'CREATE TABLE "{table}" (node_id TEXT PRIMARY KEY, row_hash TEXT NOT NULL)')
        base_rows = added_rows = duplicate_rows = conflicts = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(base_header)
            for row in reader:
                if len(row) != len(base_header):
                    raise ValueError(f"column mismatch in base node CSV {stem} at row {base_rows + 2}")
                node_id = row[id_index].strip()
                if not node_id:
                    raise ValueError(f"empty node ID in base CSV {stem}")
                digest = _row_hash(row)
                conn.execute(f'INSERT INTO "{table}" VALUES (?, ?)', (node_id, digest))
                writer.writerow(row)
                base_rows += 1
            if not add_empty:
                pending = []
                if first_addition is not None:
                    pending.append(first_addition)
                pending.extend(add_reader)
                for row in pending:
                    if len(row) != len(base_header):
                        raise ValueError(f"column mismatch in addition node CSV {stem}")
                    node_id = row[id_index].strip()
                    if not node_id:
                        raise ValueError(f"empty node ID in addition CSV {stem}")
                    digest = _row_hash(row)
                    existing = conn.execute(f'SELECT row_hash FROM "{table}" WHERE node_id = ?', (node_id,)).fetchone()
                    if existing is not None:
                        duplicate_rows += 1
                        if existing[0] != digest:
                            conflicts += 1
                        continue
                    conn.execute(f'INSERT INTO "{table}" VALUES (?, ?)', (node_id, digest))
                    writer.writerow(row)
                    added_rows += 1
        if conflicts:
            raise ValueError(f"{conflicts} conflicting duplicate node rows in {stem}")
        return {"base_rows": base_rows, "added_rows": added_rows, "duplicate_rows": duplicate_rows, "conflicting_rows": conflicts}
    finally:
        add_text.close()
        text.close()
        raw.close()


def _merge_relation(
    archive: zipfile.ZipFile,
    base_member: str,
    addition_path: Path,
    destination: Path,
    conn: sqlite3.Connection,
    stem: str,
) -> dict[str, Any]:
    raw, text, reader = _csv_reader(archive, base_member)
    add_header, add_reader, add_text = _read_addition_header(addition_path)
    try:
        base_header = _clean_header(next(reader, None) or [])
        first_addition = None
        if base_header != add_header:
            first_addition = next(add_reader, None)
            if first_addition is not None:
                raise ValueError(f"header mismatch for non-empty relation CSV {stem}")
            add_empty = True
        else:
            add_empty = False
        try:
            start_index = next(i for i, field in enumerate(base_header) if field.startswith(":START_ID"))
            end_index = next(i for i, field in enumerate(base_header) if field.startswith(":END_ID"))
            type_index = base_header.index(":TYPE")
        except (StopIteration, ValueError) as exc:
            raise ValueError(f"relation CSV lacks start/end/type columns: {stem}") from exc
        table = f'rel_{stem.replace("-", "_")}'
        conn.execute(f'CREATE TABLE "{table}" (edge_key TEXT PRIMARY KEY, row_hash TEXT NOT NULL, origin TEXT NOT NULL)')
        base_rows = added_rows = duplicate_rows = conflicts = dangling = 0
        base_duplicate_rows = base_conflicts = addition_duplicate_rows = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(base_header)

            def consume(row: Sequence[str], *, is_addition: bool) -> None:
                nonlocal base_rows, added_rows, duplicate_rows, conflicts, dangling
                nonlocal base_duplicate_rows, base_conflicts, addition_duplicate_rows
                if len(row) != len(base_header):
                    raise ValueError(f"column mismatch in {stem}")
                start_id = str(row[start_index]).strip()
                end_id = str(row[end_index]).strip()
                relation_type = str(row[type_index]).strip()
                if not start_id or not end_id or not relation_type:
                    raise ValueError(f"empty endpoint/type in {stem}")
                if not conn.execute("SELECT 1 FROM all_node_ids WHERE node_id = ?", (start_id,)).fetchone():
                    dangling += 1
                if not conn.execute("SELECT 1 FROM all_node_ids WHERE node_id = ?", (end_id,)).fetchone():
                    dangling += 1
                key = "\0".join((start_id, end_id, relation_type))
                digest = _row_hash(row)
                existing = conn.execute(f'SELECT row_hash, origin FROM "{table}" WHERE edge_key = ?', (key,)).fetchone()
                if existing is not None:
                    duplicate_rows += 1
                    if is_addition:
                        addition_duplicate_rows += 1
                        if existing[0] != digest:
                            conflicts += 1
                    else:
                        # The baseline bundle can contain duplicate rows.  The
                        # original Cypher importer uses MERGE + SET, so the
                        # last baseline row controls properties while the edge
                        # keeps its first position in the CSV output.
                        base_duplicate_rows += 1
                        if existing[0] != digest:
                            base_conflicts += 1
                        conn.execute(f'UPDATE "{table}" SET row_hash = ?, origin = ? WHERE edge_key = ?', (digest, "base", key))
                    return
                conn.execute(f'INSERT INTO "{table}" VALUES (?, ?, ?)', (key, digest, "addition" if is_addition else "base"))
                writer.writerow(row)
                if is_addition:
                    added_rows += 1
                else:
                    base_rows += 1

            for row in reader:
                consume(row, is_addition=False)
            if not add_empty:
                if first_addition is not None:
                    consume(first_addition, is_addition=True)
                for row in add_reader:
                    consume(row, is_addition=True)
        if conflicts:
            raise ValueError(f"{conflicts} conflicting duplicate relation rows in {stem}")
        return {
            "base_rows": base_rows,
            "added_rows": added_rows,
            "duplicate_rows": duplicate_rows,
            "base_duplicate_rows": base_duplicate_rows,
            "base_conflicting_duplicate_rows": base_conflicts,
            "addition_duplicate_rows": addition_duplicate_rows,
            "conflicting_rows": conflicts,
            "dangling_endpoint_count": dangling,
        }
    finally:
        add_text.close()
        text.close()
        raw.close()


def merge_import_bundle(
    base_zip: str | Path,
    addition_dir: str | Path,
    output_dir: str | Path,
    *,
    output_zip: str | Path | None = None,
) -> dict[str, Any]:
    base_zip = Path(base_zip).resolve()
    addition_dir = Path(addition_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not base_zip.is_file():
        raise FileNotFoundError(base_zip)
    if not addition_dir.is_dir():
        raise FileNotFoundError(addition_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_zip is not None and Path(output_zip).exists():
        raise FileExistsError(output_zip)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    db_path = temp_root / ".merge.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE all_node_ids (node_id TEXT PRIMARY KEY)")
    counts: dict[str, Any] = {"nodes": {}, "relations": {}}
    try:
        bundle_root = temp_root / "root" / "paper_kg_neo4j_import_merged"
        paper_kg = bundle_root / "paper_kg"
        paper_kg.mkdir(parents=True)
        with zipfile.ZipFile(base_zip) as archive:
            papers_member = _zip_member(archive, "/paper_kg/papers.csv")
            base_prefix = papers_member.split("/paper_kg/", 1)[0]
            _copy_base_support_files(archive, bundle_root, base_prefix)
            for stem in NODE_FILES:
                base_member = _zip_member(archive, f"/paper_kg/{stem}.csv")
                result = _merge_node(archive, base_member, _addition_path(addition_dir, stem), paper_kg / f"{stem}.csv", conn, stem)
                counts["nodes"][stem] = result
                table = f'node_{stem}'
                conn.execute(f'INSERT OR IGNORE INTO all_node_ids SELECT node_id FROM "{table}"')
            conn.commit()
            for stem in RELATION_FILES:
                base_member = _zip_member(archive, f"/paper_kg/{stem}.csv")
                counts["relations"][stem] = _merge_relation(archive, base_member, _addition_path(addition_dir, stem), paper_kg / f"{stem}.csv", conn, stem)
                conn.commit()
        if any(item["conflicting_rows"] for group in counts.values() for item in group.values()):
            raise ValueError("merge produced conflicting duplicate rows")
        if any(item["dangling_endpoint_count"] for item in counts["relations"].values()):
            raise ValueError("merge produced dangling relationship endpoints")
        manifest = {
            "schema": "shenzhi_neo4j_import_merged_v1",
            "base_zip": str(base_zip),
            "base_zip_sha256": file_sha256(base_zip),
            "addition_dir": str(addition_dir),
            "output_is_additive": True,
            "source_inputs_immutable": True,
            "counts": counts,
            "quality_gates": {
                "conflicting_duplicate_rows": 0,
                "baseline_conflicting_duplicate_rows": sum(item.get("base_conflicting_duplicate_rows", 0) for item in counts["relations"].values()),
                "dangling_endpoint_count": 0,
                "old_logical_edges_preserved": True,
                "baseline_duplicate_rows_deduped": sum(item.get("base_duplicate_rows", 0) for item in counts["relations"].values()),
                "a_layer_only": True,
            },
        }
        alignment_report = addition_dir / "alignment_report.json"
        if alignment_report.is_file():
            manifest["addition_alignment_report"] = {
                "path": str(alignment_report),
                "sha256": file_sha256(alignment_report),
                "quarantined_paper_count": len(json.loads(alignment_report.read_text(encoding="utf-8")).get("quarantined_papers", {})),
            }
        (bundle_root / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_root.replace(output_dir)
        # The SQLite deduplication database is an implementation detail and
        # must never be shipped in the import bundle.
        (output_dir / ".merge.sqlite").unlink(missing_ok=True)
        if output_zip is not None:
            output_zip = Path(output_zip).resolve()
            output_zip.parent.mkdir(parents=True, exist_ok=True)
            zip_temp = output_zip.with_name(f".{output_zip.name}.tmp")
            with zipfile.ZipFile(zip_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file():
                        # Keep the same ``root/paper_kg_neo4j_import_*`` layout
                        # expected by the existing Neo4j deployment scripts.
                        archive.write(path, path.relative_to(output_dir).as_posix())
            zip_temp.replace(output_zip)
            manifest["output_zip"] = str(output_zip)
            manifest["output_zip_sha256"] = file_sha256(output_zip)
            (output_dir / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        conn.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", required=True, type=Path)
    parser.add_argument("--addition-dir", required=True, type=Path, help="graph_import or graph_import/paper_kg directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-zip", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = merge_import_bundle(args.base_zip, args.addition_dir, args.output_dir, output_zip=args.output_zip)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
