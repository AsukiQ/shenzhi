"""Prepare canonical per-paper JSON for the rich-schema builder.

The canonical archive is kept immutable.  Only the four known Paper ->
Institution ``HAS_AFFILIATION`` records are quarantined because that relation
is not part of the rich schema (the supported relation is Author ->
Institution ``AFFILIATED_WITH``).  Any other unsupported or malformed record
fails closed instead of being silently changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

from .build_rich_graph import EDGE_CONTRACT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_graph_zip(input_zip: Path, output_zip: Path, report_path: Path) -> dict[str, Any]:
    input_zip = Path(input_zip).resolve()
    output_zip = Path(output_zip).resolve()
    report_path = Path(report_path).resolve()
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)
    if output_zip.exists() or report_path.exists():
        raise FileExistsError("output archive or report already exists")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_zip.stem}.tmp-", dir=output_zip.parent))
    temp_zip = temp_dir / output_zip.name
    quarantined: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counts: dict[str, int] = {"json_files": 0, "modified_json_files": 0, "removed_has_affiliation_edges": 0}
    try:
        with zipfile.ZipFile(input_zip, "r") as source, zipfile.ZipFile(
            temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as destination:
            for info in source.infolist():
                if info.is_dir():
                    destination.writestr(info, b"")
                    continue
                raw = source.read(info.filename)
                if not info.filename.lower().endswith(".json"):
                    destination.writestr(info, raw)
                    continue
                counts["json_files"] += 1
                try:
                    payload = json.loads(raw)
                    graph = payload.get("graph") if isinstance(payload, dict) else None
                    nodes = graph.get("nodes") if isinstance(graph, dict) else None
                    edges = graph.get("edges") if isinstance(graph, dict) else None
                    if not isinstance(nodes, list) or not isinstance(edges, list):
                        raise ValueError("expected graph.nodes and graph.edges arrays")
                    labels = {str(node.get("id")): str(node.get("label")) for node in nodes if isinstance(node, dict)}
                    kept = []
                    changed = False
                    for edge in edges:
                        if not isinstance(edge, dict):
                            raise ValueError("edge is not an object")
                        edge_type = str(edge.get("type", ""))
                        source_id = str(edge.get("source", ""))
                        target_id = str(edge.get("target", ""))
                        actual = (labels.get(source_id), labels.get(target_id))
                        if edge_type == "HAS_AFFILIATION" and actual == ("Paper", "Institution"):
                            quarantined.append({
                                "member": info.filename,
                                "source": source_id,
                                "target": target_id,
                                "type": edge_type,
                                "reason": "unsupported Paper->Institution relation; requires Author->Institution provenance",
                            })
                            counts["removed_has_affiliation_edges"] += 1
                            changed = True
                            continue
                        if edge_type not in EDGE_CONTRACT:
                            raise ValueError(f"unsupported relationship type: {edge_type}")
                        if actual != EDGE_CONTRACT[edge_type]:
                            raise ValueError(
                                f"relationship contract mismatch {edge_type}: expected {EDGE_CONTRACT[edge_type]}, got {actual}"
                            )
                        kept.append(edge)
                    if changed:
                        graph["edges"] = kept
                        counts["modified_json_files"] += 1
                        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                except Exception as exc:
                    errors.append({"member": info.filename, "error": f"{type(exc).__name__}: {exc}"})
                    raise ValueError(f"cannot prepare {info.filename}: {exc}") from exc
                destination.writestr(info, raw)

        if errors:
            raise ValueError(f"{len(errors)} records failed preparation")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        temp_zip.replace(output_zip)
        report = {
            "schema": "shenzhi_graph_json_preparation_v1",
            "input": {"path": str(input_zip), "bytes": input_zip.stat().st_size, "sha256": _sha256(input_zip)},
            "output": {"path": str(output_zip), "bytes": output_zip.stat().st_size, "sha256": _sha256(output_zip)},
            "counts": counts,
            "quarantined_edges": quarantined,
            "errors": errors,
            "quality_gates": {
                "input_immutable": True,
                "all_json_records_prepared": not errors,
                "only_known_unsupported_edges_removed": True,
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    except BaseException:
        temp_zip.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        output_zip.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = prepare_graph_zip(args.input_zip, args.output_zip, args.report)
    print(json.dumps({"counts": report["counts"], "output": report["output"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
