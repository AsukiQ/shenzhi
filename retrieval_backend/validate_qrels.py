"""Validate human paper qrels and report whether annotation is complete."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def validate(path: str | Path, *, require_complete: bool = False) -> dict:
    query_count = 0
    judgment_count = 0
    unlabeled_count = 0
    labels: Counter[int] = Counter()
    query_ids: set[str] = set()
    errors: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            query = str(row.get("query_text") or "").strip()
            if not query_id or not query:
                errors.append(f"line {line_number}: missing query_id/query_text")
                continue
            if query_id in query_ids:
                errors.append(f"line {line_number}: duplicate query_id={query_id}")
            query_ids.add(query_id)
            query_count += 1
            judgments = row.get("judgments") or []
            if not isinstance(judgments, list) or not judgments:
                errors.append(f"line {line_number}: judgments must be a non-empty list")
                continue
            seen: set[str] = set()
            for item in judgments:
                paper_id = str((item or {}).get("paper_id") or "").strip()
                if not paper_id:
                    errors.append(f"line {line_number}: empty paper_id")
                    continue
                if paper_id in seen:
                    errors.append(f"line {line_number}: duplicate paper_id={paper_id}")
                seen.add(paper_id)
                judgment_count += 1
                relevance = item.get("relevance")
                if relevance is None:
                    unlabeled_count += 1
                elif isinstance(relevance, bool) or int(relevance) not in {0, 1, 2, 3}:
                    errors.append(
                        f"line {line_number}: relevance must be null or 0-3 for {paper_id}"
                    )
                else:
                    labels[int(relevance)] += 1
    if require_complete and unlabeled_count:
        errors.append(f"annotation is incomplete: {unlabeled_count} null judgments")
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return {
        "status": "ok",
        "annotation_status": "complete" if unlabeled_count == 0 else "awaiting_human_labels",
        "query_count": query_count,
        "judgment_count": judgment_count,
        "unlabeled_count": unlabeled_count,
        "label_counts": {str(key): value for key, value in sorted(labels.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate(args.qrels, require_complete=args.require_complete)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
