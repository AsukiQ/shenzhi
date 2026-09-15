"""Apply the source Neo4j constraints/indexes and verify their counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j_filter import Neo4jHttpPaperFilter


def cypher_statements(text: str) -> list[str]:
    statements = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line)
        joined = "\n".join(current)
        while ";" in joined:
            statement, joined = joined.split(";", 1)
            if statement.strip():
                statements.append(statement.strip())
        current = [joined] if joined.strip() else []
    if current and "\n".join(current).strip():
        statements.append("\n".join(current).strip())
    return statements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-uri", default="http://127.0.0.1:17474")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--cypher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    graph = Neo4jHttpPaperFilter(
        http_uri=args.http_uri,
        user=args.user,
        password=args.password,
        database=args.database,
        timeout_seconds=120.0,
    )
    statements = cypher_statements(Path(args.cypher).read_text(encoding="utf-8"))
    if not statements:
        raise ValueError("schema Cypher file contains no statements")
    for statement in statements:
        graph._run(statement, {})
    constraints = graph._run(
        "SHOW CONSTRAINTS YIELD name RETURN count(name) AS count",
        {},
    )
    indexes = graph._run(
        "SHOW INDEXES YIELD name RETURN count(name) AS count",
        {},
    )
    report = {
        "status": "ok",
        "database": args.database,
        "statement_count": len(statements),
        "constraint_count": int(constraints[0]["count"]),
        "index_count": int(indexes[0]["count"]),
        "health": graph.healthcheck(),
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
