"""Create a deterministic paper-relevance annotation batch from user queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_search import PaperSearchIndex, SearchFilters


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def build_batch(
    index: PaperSearchIndex,
    rows,
    *,
    candidates_per_query: int,
) -> list[dict[str, Any]]:
    output = []

    def string_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value if str(item).strip()]

    for offset, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or f"human-{offset:05d}")
        query = str(row.get("query_text") or row.get("query") or "").strip()
        if not query:
            continue
        raw_filters = row.get("filters") or {}
        filters = SearchFilters(
            year_gte=raw_filters.get("year_gte"),
            year_lte=raw_filters.get("year_lte"),
            conference=string_list(raw_filters.get("conference")),
            author=string_list(raw_filters.get("author")),
            keyword=string_list(raw_filters.get("keyword")),
            subject=string_list(raw_filters.get("subject")),
        )
        results, _state = index.search(
            query,
            filters=filters,
            top_k=candidates_per_query,
        )
        candidate_generation = "bm25_topk_v1"
        if not results:
            fallback_hits = index.recall("", limit=5000)
            results = index.fetch_results(
                [hit.paper_id for hit in fallback_hits],
                filters=filters,
                top_k=candidates_per_query,
                score_by_id={hit.paper_id: hit.score for hit in fallback_hits},
                retrieval_mode="fallback_recent_metadata",
            )
            candidate_generation = "fallback_recent_metadata_v1"
        output.append(
            {
                "schema_version": "shenzhi_paper_qrels_annotation_v1",
                "query_id": query_id,
                "query_text": query,
                "query_source": str(row.get("query_source") or "human"),
                "split": str(row.get("split") or "human_eval"),
                "filters": raw_filters,
                "candidate_generation": candidate_generation,
                "judgments": [
                    {
                        "paper_id": result.paper_id,
                        "title": result.title,
                        "conference": result.conference,
                        "year": result.year,
                        "abstract": result.abstract,
                        "relevance": None,
                        "note": "",
                    }
                    for result in results
                ],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-per-query", type=int, default=20)
    args = parser.parse_args()
    batch = build_batch(
        PaperSearchIndex(args.db),
        read_jsonl(args.queries),
        candidates_per_query=max(1, min(args.candidates_per_query, 100)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "query_count": len(batch), "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
