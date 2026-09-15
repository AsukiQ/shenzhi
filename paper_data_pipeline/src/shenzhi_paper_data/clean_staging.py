"""Clean trusted-source staging entities without deleting raw evidence.

This is a conservative normalization pass: factual source values are retained,
obvious extraction artifacts are rejected, and only citation edges that can be
uniquely aligned to a Paper are emitted as import candidates.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


METHOD_STOPWORDS = {
    "a", "an", "the", "we", "our", "ours", "it", "this", "that",
    "propose", "proposes", "proposed", "position", "method", "approach",
}


def normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).strip()


def key(value: Any) -> str:
    return normalize(value).casefold()


def arxiv_key(value: Any) -> str:
    value = key(value)
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value).removesuffix(".pdf")
    return re.sub(r"v\d+$", "", value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_paper_indexes(papers_csv: Path) -> tuple[set[str], dict[str, str], dict[str, str]]:
    paper_ids: set[str] = set()
    arxiv_to_id: dict[str, str] = {}
    doi_to_id: dict[str, str] = {}
    with papers_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            paper_id = normalize(row.get("paper_id:ID(Paper)"))
            paper_ids.add(paper_id)
            a = arxiv_key(row.get("arxiv_id"))
            if a:
                arxiv_to_id.setdefault(a, paper_id)
            d = key(row.get("doi"))
            if d:
                doi_to_id.setdefault(d.removeprefix("https://doi.org/").removeprefix("doi:"), paper_id)
    return paper_ids, arxiv_to_id, doi_to_id


def _method_clean(item: Mapping[str, Any]) -> tuple[str, str]:
    name = normalize(item.get("name"))
    lowered = key(name)
    if not name:
        return "rejected", "empty"
    if lowered in METHOD_STOPWORDS or (len(name) == 1 and not name.isalnum()):
        return "rejected", "obvious_generic_token"
    if len(name) > 240 or len(name.split()) > 28:
        return "quarantine", "sentence_like_method"
    if "\n" in str(item.get("name", "")):
        return "quarantine", "multiline_method"
    return "accepted", ""


def _institution_parts(name: str) -> tuple[list[str], str]:
    # Split before ``normalize`` so an explicit newline remains a separator.
    raw_name = unicodedata.normalize("NFKC", str(name or "")).strip()
    name = normalize(raw_name)
    if not name:
        return [], "empty"
    # Semicolon, pipe and newlines are reliable list separators. Commas are
    # deliberately preserved because "Department, University, Country" is a
    # single affiliation, not three institutions.
    parts = [normalize(part) for part in re.split(r"\s*(?:;|\||\r?\n)\s*", raw_name) if normalize(part)]
    if len(parts) > 1:
        return parts, "split_explicit_separator"
    if len(name) > 320:
        return [name], "quarantine_long_compound"
    if "," in name:
        return [name], "compound_preserved"
    return [name], ""


def _funding_clean(item: Mapping[str, Any]) -> tuple[str, str]:
    props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
    name = normalize(props.get("name") or item.get("name"))
    if not name:
        return "", "empty"
    if len(name) > 500:
        return name, "quarantine_long_funding"
    return name, ""


def clean_staging(
    staging_dir: str | Path,
    merged_papers_csv: str | Path,
    alignment_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    staging_dir = Path(staging_dir).resolve()
    merged_papers_csv = Path(merged_papers_csv).resolve()
    alignment_report = Path(alignment_report).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for stem in ("methods", "institutions", "funding", "citations"):
        if not (staging_dir / f"{stem}.jsonl").is_file():
            raise FileNotFoundError(staging_dir / f"{stem}.jsonl")
    paper_ids, arxiv_to_id, doi_to_id = _load_paper_indexes(merged_papers_csv)
    alignment = json.loads(alignment_report.read_text(encoding="utf-8"))
    quarantined_sources = set(alignment.get("quarantined_papers", {}))

    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        temp_root.mkdir(exist_ok=True)
        counts = Counter()
        rejected: list[dict[str, Any]] = []
        methods_out: list[dict[str, Any]] = []
        method_seen: set[tuple[str, str, str]] = set()
        for row in _read_jsonl(staging_dir / "methods.jsonl"):
            for item in row.get("items", []):
                status, reason = _method_clean(item)
                name = normalize(item.get("name"))
                record = {"paper_id": row.get("paper_id"), "package": row.get("package"), "name": name, "content": normalize(item.get("content")), "relation": normalize(item.get("relation")), "raw": item, "status": status, "reason": reason}
                if status == "accepted":
                    dedup_key = (key(row.get("paper_id")), key(name), key(item.get("relation")))
                    if dedup_key not in method_seen:
                        method_seen.add(dedup_key)
                        methods_out.append(record)
                        counts["methods_accepted"] += 1
                    else:
                        counts["methods_duplicate"] += 1
                else:
                    rejected.append({"kind": "method", **record})
                    counts[f"methods_{status}"] += 1

        institutions_out: list[dict[str, Any]] = []
        institution_seen: set[tuple[str, str]] = set()
        for row in _read_jsonl(staging_dir / "institutions.jsonl"):
            for item in row.get("items", []):
                parts, note = _institution_parts(item.get("name"))
                for part in parts:
                    dedup_key = (key(part), key(row.get("paper_id")))
                    if dedup_key in institution_seen:
                        counts["institutions_duplicate"] += 1
                        continue
                    institution_seen.add(dedup_key)
                    status = "quarantine" if note == "quarantine_long_compound" else "accepted"
                    record = {"paper_id": row.get("paper_id"), "package": row.get("package"), "author_node_id": item.get("author_node_id"), "author_name": normalize(item.get("author_name")), "name": part, "normalization_note": note, "raw": item, "status": status}
                    institutions_out.append(record)
                    counts[f"institutions_{status}"] += 1

        funding_out: list[dict[str, Any]] = []
        funding_seen: set[tuple[str, str]] = set()
        for row in _read_jsonl(staging_dir / "funding.jsonl"):
            for item in row.get("items", []):
                name, note = _funding_clean(item)
                if not name:
                    rejected.append({"kind": "funding", "paper_id": row.get("paper_id"), "package": row.get("package"), "raw": item, "status": "rejected", "reason": note})
                    counts["funding_rejected"] += 1
                    continue
                status = "quarantine" if note else "accepted"
                dedup_key = (key(row.get("paper_id")), key(name))
                if dedup_key in funding_seen:
                    counts["funding_duplicate"] += 1
                    continue
                funding_seen.add(dedup_key)
                funding_out.append({"paper_id": row.get("paper_id"), "package": row.get("package"), "name": name, "normalization_note": note, "raw": item, "status": status})
                counts[f"funding_{status}"] += 1

        citations_out: list[dict[str, Any]] = []
        citations_quarantine: list[dict[str, Any]] = []
        cite_seen: set[tuple[str, str]] = set()
        for row in _read_jsonl(staging_dir / "citations.jsonl"):
            source_id = normalize(row.get("paper_id"))
            for item in row.get("items", []):
                target_arxiv = arxiv_key(item.get("target_arxiv_id"))
                target_id = arxiv_to_id.get(target_arxiv, "") if target_arxiv else ""
                confidence = item.get("confidence")
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None
                if source_id in quarantined_sources:
                    reason = "source_paper_quarantined"
                elif source_id not in paper_ids:
                    reason = "source_paper_not_in_merged_graph"
                elif not target_arxiv:
                    reason = "title_only_unresolved"
                elif not target_id:
                    reason = "target_arxiv_not_in_merged_graph"
                elif confidence is None or confidence < 0.85:
                    reason = "low_confidence"
                else:
                    reason = ""
                record = {"source_paper_id": source_id, "target_paper_id": target_id, "target_arxiv_id": target_arxiv, "target_title": normalize(item.get("target_title")), "confidence": confidence, "source": normalize(item.get("source")), "raw": item, "status": "accepted" if not reason else "quarantine", "reason": reason}
                if reason:
                    citations_quarantine.append(record)
                    counts[f"citations_{reason}"] += 1
                    continue
                dedup_key = (source_id, target_id)
                if dedup_key in cite_seen:
                    counts["citations_duplicate"] += 1
                    continue
                cite_seen.add(dedup_key)
                citations_out.append(record)
                counts["citations_accepted"] += 1

        _write_jsonl(temp_root / "methods_clean.jsonl", methods_out)
        _write_jsonl(temp_root / "institutions_clean.jsonl", institutions_out)
        _write_jsonl(temp_root / "funding_clean.jsonl", funding_out)
        _write_jsonl(temp_root / "citations_clean.jsonl", citations_out)
        _write_jsonl(temp_root / "citations_quarantine.jsonl", citations_quarantine)
        _write_jsonl(temp_root / "rejected.jsonl", rejected)
        with (temp_root / "cites_import.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow((":START_ID(Paper)", ":END_ID(Paper)", "confidence:double", "source", ":TYPE"))
            for item in citations_out:
                writer.writerow((item["source_paper_id"], item["target_paper_id"], item["confidence"], item["source"], "CITES"))
        report = {
            "schema": "shenzhi_staging_clean_v1",
            "inputs": {"staging_dir": str(staging_dir), "merged_papers_csv": str(merged_papers_csv), "alignment_report": str(alignment_report)},
            "counts": dict(sorted(counts.items())),
            "quality_gates": {
                "raw_staging_modified": False,
                "all_accepted_citations_have_source_and_target_paper": all(item["source_paper_id"] in paper_ids and item["target_paper_id"] in paper_ids for item in citations_out),
                "accepted_citation_confidence_min": min((item["confidence"] for item in citations_out), default=None),
                "obvious_method_noise_not_accepted": True,
                "raw_values_retained_in_clean_records": True,
            },
            "next_graph_contract": {
                "methods": "ready_for_incremental_import",
                "institutions": "ready_for_incremental_import",
                "funding": "ready_for_incremental_import",
                "cites_import": "ready_for_incremental_import",
            },
        }
        (temp_root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_root.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--merged-papers-csv", required=True, type=Path)
    parser.add_argument("--alignment-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = clean_staging(args.staging_dir, args.merged_papers_csv, args.alignment_report, args.output_dir)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
