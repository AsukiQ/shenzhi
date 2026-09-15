"""Verify a live Neo4j import and the paper-filter adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j_filter import Neo4jHttpPaperFilter
from paper_search import SearchFilters


COUNT_CYPHER = """
MATCH (p:Paper)
WITH count(p) AS papers
MATCH (a:Author)
WITH papers, count(a) AS authors
MATCH ()-[r]->()
RETURN papers, authors, count(r) AS relationships
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http-uri", default="http://127.0.0.1:17474")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="unused")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--expected-paper-count", type=int, default=107200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    graph = Neo4jHttpPaperFilter(
        http_uri=args.http_uri,
        user=args.user,
        password=args.password,
        database=args.database,
        timeout_seconds=60.0,
    )
    health = graph.healthcheck()
    counts = graph._run(COUNT_CYPHER, {})
    if not counts:
        raise RuntimeError("Neo4j count query returned no rows")
    if int(health["paper_count"]) != int(args.expected_paper_count):
        raise RuntimeError(
            f"unexpected Paper count: {health['paper_count']} != {args.expected_paper_count}"
        )
    sample = graph._run(
        "MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue) "
        "RETURN p.paper_id AS paper_id, v.name AS venue, p.year AS year "
        "ORDER BY p.paper_id LIMIT 1",
        {},
    )
    if not sample:
        raise RuntimeError("Neo4j sample Paper→Venue query returned no rows")
    row = sample[0]
    filtered = graph.filter_candidate_ids(
        [str(row["paper_id"]), "paper:not-in-original-graph"],
        filters=SearchFilters(
            year_gte=int(row["year"]) if row.get("year") is not None else None,
            conference=[str(row["venue"])],
        ),
    )
    if str(row["paper_id"]) not in filtered:
        raise RuntimeError("Neo4j parameterized graph filter rejected its known sample")
    report = {
        "status": "ok",
        "health": health,
        "counts": counts[0],
        "sample": row,
        "filtered_candidate_ids": filtered,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
