from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import zipfile


PLACEHOLDER_ABSTRACTS = {
    "",
    "n/a",
    "na",
    "no abstract was provided.",
    "no summary was provided.",
    "none",
    "null",
    "tba",
    "todo",
}

LIST_FIELDS = {"authors", "editor", "keywords", "subjects"}
ORDERED_LIST_FIELDS = {"authors", "editor"}
INTEGER_FIELDS = {"year"}
CORE_FIELDS = (
    "title",
    "abstract",
    "authors",
    "conference",
    "year",
    "subjects",
    "keywords",
    "pdf_url",
    "doi",
    "type",
    "editor",
    "booktitle",
    "pages",
    "publisher",
    "biburl",
    "bibsource",
    "source",
)


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def normalize_source_id(value: Any) -> str:
    return normalize_text(value)


def normalized_key(value: Any) -> str:
    return " ".join(normalize_text(value).casefold().split())


def is_placeholder_abstract(value: Any) -> bool:
    return normalized_key(value) in PLACEHOLDER_ABSTRACTS


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonblank(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _mode_value(values: Sequence[Any], *, prefer_longer: bool = False) -> Any:
    nonblank = [value for value in values if _nonblank(value)]
    if not nonblank:
        return ""
    counts = Counter(_stable_json(value) for value in nonblank)
    decoded: dict[str, Any] = {}
    for value in nonblank:
        decoded.setdefault(_stable_json(value), value)

    def score(encoded: str) -> tuple[int, int, str]:
        value = decoded[encoded]
        length = len(normalize_text(value)) if prefer_longer else 0
        return counts[encoded], length, encoded

    return decoded[max(counts, key=score)]


def _choose_abstract(values: Sequence[Any]) -> str:
    candidates = [normalize_text(value) for value in values if _nonblank(value)]
    if not candidates:
        return ""
    return max(
        candidates,
        key=lambda value: (
            not is_placeholder_abstract(value),
            len(value),
            value,
        ),
    )


def collapse_repeated_sequence(values: Sequence[str]) -> list[str]:
    """Collapse exact whole-list repetition while preserving author order.

    For example, [A, B, C, A, B, C] becomes [A, B, C]. This intentionally
    does not remove isolated repeated names because two distinct authors may
    legitimately share a normalized display name.
    """

    sequence = list(values)
    for period in range(1, len(sequence) // 2 + 1):
        if len(sequence) % period:
            continue
        block = sequence[:period]
        if block * (len(sequence) // period) == sequence:
            return block
    return sequence


def _choose_ordered_list(values: Sequence[Any]) -> list[str]:
    candidates: list[list[str]] = []
    for value in values:
        if not isinstance(value, list):
            continue
        cleaned = collapse_repeated_sequence(
            [normalize_text(item) for item in value if normalize_text(item)]
        )
        if cleaned:
            candidates.append(cleaned)
    if not candidates:
        return []
    counts = Counter(_stable_json(candidate) for candidate in candidates)
    decoded = {encoded: candidate for encoded, candidate in zip(counts, candidates)}
    selected = max(
        counts,
        key=lambda encoded: (
            counts[encoded],
            len(decoded[encoded]),
            encoded,
        ),
    )
    return list(decoded[selected])


def _merge_unordered_lists(values: Sequence[Any]) -> list[str]:
    merged: dict[str, str] = {}
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            cleaned = normalize_text(item)
            if cleaned:
                key = normalized_key(cleaned)
                current = merged.get(key)
                # Prefer a more informative spelling; preserve the first spelling
                # for equal-length variants (e.g. "Graph" over later "graph").
                if current is None or len(cleaned) > len(current):
                    merged[key] = cleaned
    return [merged[key] for key in sorted(merged)]


def merge_records(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    if not records:
        raise ValueError("cannot merge an empty record group")
    source_ids = {normalize_source_id(record.get("arxiv_id")) for record in records}
    if "" in source_ids or len(source_ids) != 1:
        raise ValueError("all records in a merge group require one nonempty source id")

    conflicts: set[str] = set()
    all_fields = set().union(*(record.keys() for record in records))
    for field in all_fields:
        values = {_stable_json(record.get(field)) for record in records}
        if len(values) > 1:
            conflicts.add(field)

    merged: dict[str, Any] = {}
    for field in CORE_FIELDS:
        values = [record.get(field) for record in records]
        if field == "abstract":
            merged[field] = _choose_abstract(values)
        elif field in ORDERED_LIST_FIELDS:
            merged[field] = _choose_ordered_list(values)
        elif field in LIST_FIELDS:
            merged[field] = _merge_unordered_lists(values)
        elif field in INTEGER_FIELDS:
            value = _mode_value(values)
            merged[field] = None if value == "" else int(value)
        else:
            merged[field] = normalize_text(
                _mode_value(values, prefer_longer=field in {"title", "booktitle"})
            )

    source_id = next(iter(source_ids))
    source_files = sorted(
        {
            normalize_text(record.get("source_file"))
            for record in records
            if normalize_text(record.get("source_file"))
        }
    )
    source_paper_ids = sorted(
        {
            normalize_text(record.get("paper_id"))
            for record in records
            if normalize_text(record.get("paper_id"))
        }
    )
    merged.update(
        {
            "arxiv_id": source_id,
            "source_id": source_id,
            "source_file": source_files[0] if source_files else "",
            "source_files": source_files,
            "source_paper_ids": source_paper_ids,
            "source_record_count": len(records),
        }
    )
    return merged, conflicts


def generated_paper_id(title: Any, source_id: Any) -> str:
    source = normalize_source_id(source_id)
    if not source:
        raise ValueError("generated paper ids require a nonempty source id")
    ascii_source = (
        unicodedata.normalize("NFKD", source)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_source).strip("_")[:48] or "source"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"paper:new:{slug}:{digest}"


def build_search_text(record: Mapping[str, Any]) -> str:
    authors = "; ".join(record.get("authors") or [])
    keywords = "; ".join(record.get("keywords") or [])
    subjects = "; ".join(record.get("subjects") or [])
    return "\n".join(
        (
            f"[TITLE] {record.get('title', '')}",
            f"[ABSTRACT] {record.get('abstract', '')}",
            f"[AUTHORS] {authors}",
            f"[KEYWORDS] {keywords}",
            f"[SUBJECTS] {subjects}",
            f"[CONFERENCE] {record.get('conference', '')}",
            f"[YEAR] {record.get('year', '')}",
        )
    )


def _find_papers_csv_member(archive: zipfile.ZipFile) -> str:
    matches = [name for name in archive.namelist() if name.endswith("/paper_kg/papers.csv")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one paper_kg/papers.csv in ZIP, found {len(matches)}"
        )
    return matches[0]


def load_canonical_ids(zip_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    mapping: dict[str, str] = {}
    row_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        member = _find_papers_csv_member(archive)
        with archive.open(member) as raw, io.TextIOWrapper(
            raw, encoding="utf-8-sig", newline=""
        ) as text:
            reader = csv.DictReader(text)
            required = {"paper_id:ID(Paper)", "arxiv_id"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"ZIP papers.csv is missing columns: {sorted(required)}")
            for row in reader:
                row_count += 1
                source_id = normalize_source_id(row.get("arxiv_id"))
                paper_id = normalize_text(row.get("paper_id:ID(Paper)"))
                if not source_id or not paper_id:
                    raise ValueError(f"ZIP papers.csv row {row_count} has an empty id")
                previous = mapping.get(source_id)
                if previous is not None and previous != paper_id:
                    raise ValueError(f"source id {source_id!r} maps to multiple paper ids")
                mapping[source_id] = paper_id
    if len(mapping) != row_count:
        raise ValueError("ZIP papers.csv contains duplicate source ids")
    return mapping, {"member": member, "rows": row_count}


def load_json_records(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("paper JSON must be an array of objects")
    return data


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _field_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    fields = sorted(set().union(*(record.keys() for record in records)))
    result: dict[str, dict[str, int]] = {}
    for field in fields:
        present = sum(field in record for record in records)
        nonblank = sum(_nonblank(record.get(field)) for record in records)
        result[field] = {
            "present": present,
            "missing": len(records) - present,
            "nonblank": nonblank,
            "blank": present - nonblank,
        }
    return result


def _quality_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    inputs = report["inputs"]
    return f"""# 深知论文数据质量报告

## 输入

- JSON：`{inputs['json']['path']}`
- ZIP：`{inputs['zip']['path']}`

## 核心结果

| 指标 | 数量 |
|---|---:|
| JSON 原始记录 | {counts['json_records']:,} |
| 唯一来源论文 | {counts['unique_source_ids']:,} |
| 合并掉的重复记录 | {counts['duplicate_records_merged']:,} |
| 沿用 ZIP canonical ID | {counts['canonical_ids_reused']:,} |
| 新生成 canonical ID | {counts['canonical_ids_generated']:,} |
| 可进入摘要检索索引 | {counts['retrieval_documents']:,} |
| 仅有占位摘要 | {counts['placeholder_only_papers']:,} |

## ID 和去重规则

- 主去重键：规范化后的 `arxiv_id`（本数据中的来源记录 ID）。
- ZIP 已有论文：沿用 `papers.csv` 的 `paper_id:ID(Paper)`。
- JSON 新增论文：使用 `paper:new:<source-id-slug>:<source-id-sha256>`。
- 同标题但来源 ID 不同的记录不会自动合并。
- 作者/编辑数组如果是完整序列重复（如 `[A,B,A,B]`），会折叠为一份；孤立同名不会删除。
- 占位摘要论文保留在 `papers_clean.jsonl`，但不写入 `retrieval_documents.jsonl`。

## 当前限制

- 当前数据仍没有结构化的 Institution、Method、Funding 节点。
- `subjects` 是会议栏目/轨道信息，不等同于 Method。
- 本步骤只处理 Paper 元数据和检索文档，不改写 ZIP 中的图关系 CSV。
"""


def prepare_dataset(json_path: Path, zip_path: Path, output_dir: Path) -> dict[str, Any]:
    json_path = json_path.resolve()
    zip_path = zip_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not json_path.is_file() or not zip_path.is_file():
        raise FileNotFoundError("both JSON and ZIP input files must exist")

    canonical_ids, zip_info = load_canonical_ids(zip_path)
    source_records = load_json_records(json_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(source_records, start=1):
        source_id = normalize_source_id(record.get("arxiv_id"))
        if not source_id:
            raise ValueError(f"JSON record {index} has an empty arxiv_id/source id")
        grouped[source_id].append(record)

    conflict_counts: Counter[str] = Counter()
    duplicate_group_count = 0
    cleaned: list[dict[str, Any]] = []
    used_paper_ids: set[str] = set()
    aligned = 0
    generated = 0
    placeholder_only = 0

    for source_id in sorted(grouped):
        records = grouped[source_id]
        merged, conflicts = merge_records(records)
        if len(records) > 1:
            duplicate_group_count += 1
            conflict_counts.update(conflicts)

        paper_id = canonical_ids.get(source_id)
        if paper_id is None:
            paper_id = generated_paper_id(merged.get("title"), source_id)
            generated += 1
            id_source = "generated"
        else:
            aligned += 1
            id_source = "zip_canonical"
        if paper_id in used_paper_ids:
            raise ValueError(f"duplicate output paper id: {paper_id}")
        used_paper_ids.add(paper_id)

        eligible = not is_placeholder_abstract(merged.get("abstract"))
        if not eligible:
            placeholder_only += 1
        merged = {
            "paper_id": paper_id,
            "canonical_id_source": id_source,
            **merged,
            "retrieval_eligible": eligible,
            "retrieval_exclusion_reason": "" if eligible else "placeholder_abstract",
        }
        cleaned.append(merged)

    missing_json_sources = sorted(set(canonical_ids) - set(grouped))
    if missing_json_sources:
        raise ValueError(
            f"{len(missing_json_sources)} ZIP source ids are absent from JSON; first: "
            f"{missing_json_sources[:3]}"
        )
    cleaned.sort(key=lambda record: record["paper_id"])
    title_groups: Counter[str] = Counter(
        normalized_key(record.get("title")) for record in cleaned
    )
    same_title_different_source_groups = sum(
        count > 1 for title, count in title_groups.items() if title
    )
    same_title_different_source_extra_records = sum(
        count - 1 for title, count in title_groups.items() if title and count > 1
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        papers_path = temp_path / "papers_clean.jsonl"
        retrieval_path = temp_path / "retrieval_documents.jsonl"
        mapping_path = temp_path / "paper_id_map.csv"
        report_path = temp_path / "quality_report.json"
        markdown_path = temp_path / "quality_report.md"

        paper_count = _write_jsonl(papers_path, cleaned)
        retrieval_count = _write_jsonl(
            retrieval_path,
            (
                {
                    "paper_id": record["paper_id"],
                    "source_id": record["source_id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "authors": record["authors"],
                    "conference": record["conference"],
                    "year": record["year"],
                    "subjects": record["subjects"],
                    "keywords": record["keywords"],
                    "search_text": build_search_text(record),
                }
                for record in cleaned
                if record["retrieval_eligible"]
            ),
        )
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("source_id", "paper_id", "canonical_id_source"),
            )
            writer.writeheader()
            for record in cleaned:
                writer.writerow(
                    {
                        "source_id": record["source_id"],
                        "paper_id": record["paper_id"],
                        "canonical_id_source": record["canonical_id_source"],
                    }
                )

        report: dict[str, Any] = {
            "schema": "shenzhi_paper_data_quality_v1",
            "inputs": {
                "json": {
                    "path": str(json_path),
                    "bytes": json_path.stat().st_size,
                    "sha256": file_sha256(json_path),
                },
                "zip": {
                    "path": str(zip_path),
                    "bytes": zip_path.stat().st_size,
                    "sha256": file_sha256(zip_path),
                    **zip_info,
                },
            },
            "counts": {
                "json_records": len(source_records),
                "unique_source_ids": len(grouped),
                "duplicate_source_groups": duplicate_group_count,
                "duplicate_records_merged": len(source_records) - len(grouped),
                "clean_papers": paper_count,
                "canonical_ids_reused": aligned,
                "canonical_ids_generated": generated,
                "retrieval_documents": retrieval_count,
                "placeholder_only_papers": placeholder_only,
                "same_title_different_source_groups": same_title_different_source_groups,
                "same_title_different_source_extra_records": (
                    same_title_different_source_extra_records
                ),
            },
            "duplicate_field_conflicts": dict(sorted(conflict_counts.items())),
            "input_field_coverage": _field_coverage(source_records),
            "invariants": {
                "all_zip_source_ids_found_in_json": True,
                "paper_ids_unique": len(used_paper_ids) == len(cleaned),
                "source_ids_unique_after_merge": len(grouped) == len(cleaned),
                "retrieval_documents_have_real_abstracts": True,
                "original_inputs_modified": False,
            },
            "outputs": {
                "papers_clean.jsonl": {
                    "rows": paper_count,
                    "sha256": file_sha256(papers_path),
                },
                "retrieval_documents.jsonl": {
                    "rows": retrieval_count,
                    "sha256": file_sha256(retrieval_path),
                },
                "paper_id_map.csv": {
                    "rows": paper_count,
                    "sha256": file_sha256(mapping_path),
                },
            },
        }
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        markdown_path.write_text(_quality_markdown(report), encoding="utf-8")
        temp_path.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate Shenzhi paper JSON, reuse canonical Neo4j paper ids, "
            "and produce retrieval-ready JSONL documents."
        )
    )
    parser.add_argument("--json", required=True, type=Path, help="source paper JSON")
    parser.add_argument("--zip", required=True, type=Path, help="Neo4j import ZIP")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new output directory; the command refuses to overwrite it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_dataset(args.json, args.zip, args.output_dir)
    counts = report["counts"]
    print(
        "prepared "
        f"{counts['clean_papers']} unique papers and "
        f"{counts['retrieval_documents']} retrieval documents in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
