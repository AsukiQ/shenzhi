"""Neo4j candidate filtering for the deployed final graph schema."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib import error, request

try:
    from .paper_search import SearchFilters
except ImportError:
    from paper_search import SearchFilters


RELATION_TYPES: dict[str, tuple[str, ...]] = {
    "authors": ("AUTHORED_BY",),
    "keywords": ("HAS_TOPIC",),
    "subjects": ("HAS_TOPIC",),
    "venue_year": ("PUBLISHED_IN",),
}


FILTER_CYPHER = """
UNWIND range(0, size($candidate_ids) - 1) AS candidate_index
WITH candidate_index, $candidate_ids[candidate_index] AS requested_id
OPTIONAL MATCH (p:Paper {paper_id: requested_id})
WHERE p IS NULL OR (
  ($year_gte IS NULL OR p.year >= $year_gte)
  AND ($year_lte IS NULL OR p.year <= $year_lte)
  AND (size($conference) = 0 OR EXISTS {
    MATCH (p)-[:PUBLISHED_IN]->(c:Conference)
    WHERE any(wanted IN $conference
      WHERE toLower(coalesce(c.name, '')) CONTAINS toLower(wanted)
         OR toLower(coalesce(c.venue, '')) CONTAINS toLower(wanted))
  })
  AND (size($author) = 0 OR EXISTS {
    MATCH (p)-[:AUTHORED_BY]->(a:Author)
    WHERE any(wanted IN $author
      WHERE toLower(coalesce(a.name, '')) CONTAINS toLower(wanted))
  })
  AND (size($keyword) = 0 OR EXISTS {
    MATCH (p)-[:HAS_TOPIC]->(k:Topic)
    WHERE any(wanted IN $keyword
      WHERE toLower(coalesce(k.name, '')) CONTAINS toLower(wanted))
  })
  AND (size($subject) = 0 OR EXISTS {
    MATCH (p)-[:HAS_TOPIC]->(s:Topic)
    WHERE any(wanted IN $subject
      WHERE toLower(coalesce(s.name, '')) CONTAINS toLower(wanted))
  })
)
RETURN requested_id AS paper_id, candidate_index
ORDER BY candidate_index
""".strip()

EXPANSION_CYPHER = """
UNWIND range(0, size($seed_ids) - 1) AS seed_index
WITH $seed_ids[seed_index] AS seed_id, seed_index
MATCH (seed:Paper {paper_id: seed_id})
MATCH (seed)-[r]->(node)
WHERE type(r) IN $relation_types
WITH node, min(seed_index) AS first_seed
ORDER BY first_seed, coalesce(node.name, node.year, node.paper_id, '')
WITH collect(DISTINCT node)[0..$value_limit] AS nodes
UNWIND nodes AS node
OPTIONAL MATCH (other:Paper)-[back]->(node)
WHERE type(back) IN $relation_types
WITH node, collect(DISTINCT other.paper_id)[0..$paper_limit] AS paper_ids
RETURN coalesce(node.name, toString(node.year), node.paper_id) AS value,
       paper_ids
ORDER BY value
""".strip()


class Neo4jPaperFilter:
    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        timeout_seconds: float = 3.0,
        driver: Any | None = None,
    ) -> None:
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "neo4j Python driver is required when Neo4j filtering is enabled"
                ) from exc
            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=float(timeout_seconds),
            )
        self.driver = driver
        self.database = database

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def healthcheck(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            row = session.run(
                "MATCH (p:Paper) RETURN count(p) AS paper_count"
            ).single()
        return {
            "status": "ok",
            "paper_count": int(row["paper_count"]),
            "database": self.database,
        }

    def filter_candidate_ids(
        self,
        candidate_ids: list[str],
        *,
        filters: SearchFilters,
    ) -> list[str]:
        if not candidate_ids:
            return []
        params = {
            "candidate_ids": list(dict.fromkeys(candidate_ids)),
            "year_gte": filters.year_gte,
            "year_lte": filters.year_lte,
            "conference": list(filters.conference),
            "author": list(filters.author),
            "keyword": list(filters.keyword),
            "subject": list(filters.subject),
        }
        with self.driver.session(database=self.database) as session:
            rows = session.run(FILTER_CYPHER, **params)
            return [str(row["paper_id"]) for row in rows]

    def expand_from_papers(
        self,
        seed_ids: list[str],
        *,
        relation: str,
        value_limit: int = 128,
        paper_limit: int = 100,
        slot_index: int | None = None,
    ) -> tuple[list[tuple[str, list[str]]], list[str]]:
        relation_types = RELATION_TYPES.get(str(relation))
        if not relation_types:
            raise ValueError(f"unsupported Neo4j expansion relation: {relation}")
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                EXPANSION_CYPHER,
                seed_ids=list(dict.fromkeys(seed_ids)),
                relation_types=list(relation_types),
                value_limit=max(1, int(value_limit)),
                paper_limit=max(1, int(paper_limit)),
            )
            materialized = [
                (str(row["value"]), [str(item) for item in (row["paper_ids"] or [])])
                for row in rows
            ]
        if slot_index is not None:
            slot = int(slot_index)
            if not 0 <= slot < len(materialized):
                return [], []
            materialized = [materialized[slot]]
        return materialized, [value for value, _ids in materialized]


class Neo4jHttpPaperFilter:
    """Neo4j transactional HTTP adapter using only the Python standard library."""

    def __init__(
        self,
        *,
        http_uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        timeout_seconds: float = 3.0,
    ) -> None:
        self.endpoint = (
            str(http_uri).rstrip("/")
            + f"/db/{str(database).strip() or 'neo4j'}/tx/commit"
        )
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        self.authorization = f"Basic {token}"
        self.database = database
        self.timeout_seconds = float(timeout_seconds)

    def _run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        payload = json.dumps(
            {
                "statements": [
                    {
                        "statement": statement,
                        "parameters": parameters,
                        "resultDataContents": ["row"],
                    }
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Neo4j HTTP query failed ({exc.code}): {body[:1000]}") from exc
        failures = decoded.get("errors") or []
        if failures:
            raise RuntimeError(f"Neo4j HTTP query error: {failures[0]}")
        results = decoded.get("results") or []
        if not results:
            return []
        columns = list(results[0].get("columns") or [])
        return [
            dict(zip(columns, item.get("row") or []))
            for item in results[0].get("data") or []
        ]

    def close(self) -> None:
        return None

    def healthcheck(self) -> dict[str, Any]:
        rows = self._run(
            "MATCH (p:Paper) RETURN count(p) AS paper_count",
            {},
        )
        if not rows:
            raise RuntimeError("Neo4j HTTP healthcheck returned no rows")
        return {
            "status": "ok",
            "paper_count": int(rows[0]["paper_count"]),
            "database": self.database,
        }

    def filter_candidate_ids(
        self,
        candidate_ids: list[str],
        *,
        filters: SearchFilters,
    ) -> list[str]:
        if not candidate_ids:
            return []
        rows = self._run(
            FILTER_CYPHER,
            {
                "candidate_ids": list(dict.fromkeys(candidate_ids)),
                "year_gte": filters.year_gte,
                "year_lte": filters.year_lte,
                "conference": list(filters.conference),
                "author": list(filters.author),
                "keyword": list(filters.keyword),
                "subject": list(filters.subject),
            },
        )
        return [str(row["paper_id"]) for row in rows]

    def expand_from_papers(
        self,
        seed_ids: list[str],
        *,
        relation: str,
        value_limit: int = 128,
        paper_limit: int = 100,
        slot_index: int | None = None,
    ) -> tuple[list[tuple[str, list[str]]], list[str]]:
        relation_types = RELATION_TYPES.get(str(relation))
        if not relation_types:
            raise ValueError(f"unsupported Neo4j expansion relation: {relation}")
        rows = self._run(
            EXPANSION_CYPHER,
            {
                "seed_ids": list(dict.fromkeys(seed_ids)),
                "relation_types": list(relation_types),
                "value_limit": max(1, int(value_limit)),
                "paper_limit": max(1, int(paper_limit)),
            },
        )
        materialized = [
            (str(row["value"]), [str(item) for item in (row["paper_ids"] or [])])
            for row in rows
        ]
        if slot_index is not None:
            slot = int(slot_index)
            if not 0 <= slot < len(materialized):
                return [], []
            materialized = [materialized[slot]]
        return materialized, [value for value, _ids in materialized]
