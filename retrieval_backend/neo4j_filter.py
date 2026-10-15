"""Neo4j candidate filtering for the deployed final graph schema."""

from __future__ import annotations

import base64
import json
import time
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
MATCH (p:Paper {paper_id: requested_id})
WHERE (
  ($record_type = 'all' AND coalesce(p.record_type,
      CASE WHEN coalesce(p.external, false) THEN 'external_full' ELSE 'canonical' END)
      IN ['canonical', 'external_full'])
  OR ($record_type <> 'all' AND coalesce(p.record_type,
      CASE WHEN coalesce(p.external, false) THEN 'external_full' ELSE 'canonical' END)
      = $record_type)
)
AND (
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
  AND (size($institution) = 0 OR EXISTS {
    MATCH (p)-[:AUTHORED_BY]->(a:Author)-[:AFFILIATED_WITH]-(i:Institution)
    WHERE any(wanted IN $institution
      WHERE toLower(coalesce(i.name, '')) CONTAINS toLower(wanted)
         OR toLower(coalesce(i.institution_id, '')) CONTAINS toLower(wanted))
  })
  AND (size($funding) = 0 OR EXISTS {
    MATCH (p)-[:FUNDED_BY]->(f:Funding)
    WHERE any(wanted IN $funding
      WHERE toLower(coalesce(f.name, '')) CONTAINS toLower(wanted)
         OR toLower(coalesce(f.funding_id, '')) CONTAINS toLower(wanted))
  })
)
RETURN requested_id AS paper_id, candidate_index
ORDER BY candidate_index
""".strip()

EXPANSION_CYPHER = """
UNWIND range(0, size($seed_ids) - 1) AS seed_index
WITH $seed_ids[seed_index] AS seed_id, seed_index
MATCH (seed:Paper {paper_id: seed_id})
WHERE coalesce(seed.record_type,
      CASE WHEN coalesce(seed.external, false) THEN 'external_full' ELSE 'canonical' END)
      IN ['canonical', 'external_full']
MATCH (seed)-[r]->(node)
WHERE type(r) IN $relation_types
WITH node, min(seed_index) AS first_seed
ORDER BY first_seed, coalesce(node.name, node.year, node.paper_id, '')
WITH collect(DISTINCT node)[0..$value_limit] AS nodes
UNWIND nodes AS node
OPTIONAL MATCH (other:Paper)-[back]->(node)
WHERE type(back) IN $relation_types
  AND coalesce(other.record_type,
      CASE WHEN coalesce(other.external, false) THEN 'external_full' ELSE 'canonical' END)
      IN ['canonical', 'external_full']
WITH node, collect(DISTINCT other.paper_id)[0..$paper_limit] AS paper_ids
RETURN coalesce(node.name, toString(node.year), node.paper_id) AS value,
       paper_ids
ORDER BY value
""".strip()


class Neo4jPaperFilter:
    skip_empty_filter = True

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
        self._health_cache: tuple[float, dict[str, Any]] | None = None

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def healthcheck(self) -> dict[str, Any]:
        cached = self._health_cache
        if cached and time.monotonic() - cached[0] < 15.0:
            return dict(cached[1])
        with self.driver.session(database=self.database) as session:
            row = session.run(
                """CALL { MATCH (p:Paper) RETURN count(p) AS paper_count }
                   CALL { MATCH (f:Funding) RETURN count(f) AS funding_count }
                   RETURN paper_count, funding_count"""
            ).single()
        result = {
            "status": "ok",
            "paper_count": int(row["paper_count"]),
            "funding_count": int(row.get("funding_count", 0)),
            "database": self.database,
        }
        self._health_cache = (time.monotonic(), result)
        return dict(result)

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
            "institution": list(filters.institution),
            "funding": list(filters.funding),
            "record_type": filters.record_type,
        }
        with self.driver.session(database=self.database) as session:
            rows = session.run(FILTER_CYPHER, **params)
            return [str(row["paper_id"]) for row in rows]

    def institution_paper_page(
        self, institution: str, *, offset: int = 0, limit: int = 10
    ) -> tuple[list[str], int]:
        """Page papers linked to an institution without lexical/dense recall."""
        institution = str(institution or "").strip()
        offset, limit = int(offset), int(limit)
        if not institution:
            raise ValueError("institution is required")
        if offset < 0 or limit < 1 or limit > 100:
            raise ValueError("offset must be >= 0 and limit must be between 1 and 100")
        match = """
        MATCH (p:Paper)-[:AUTHORED_BY]->(:Author)-[:AFFILIATED_WITH]-(i:Institution)
        WHERE toLower(coalesce(i.name, '')) CONTAINS toLower($institution)
           OR toLower(coalesce(i.institution_id, '')) CONTAINS toLower($institution)
        RETURN DISTINCT p
        """
        with self.driver.session(database=self.database) as session:
            total_row = session.run(
                """MATCH (p:Paper)-[:AUTHORED_BY]->(:Author)-[:AFFILIATED_WITH]-(i:Institution)
                   WHERE toLower(coalesce(i.name, '')) CONTAINS toLower($institution)
                      OR toLower(coalesce(i.institution_id, '')) CONTAINS toLower($institution)
                   RETURN count(DISTINCT p) AS total""",
                institution=institution,
            ).single()
            rows = session.run(
                match.replace("RETURN DISTINCT p", "RETURN DISTINCT p.paper_id AS paper_id ORDER BY paper_id SKIP $offset LIMIT $limit"),
                institution=institution, offset=offset, limit=limit,
            )
            ids = [str(row["paper_id"]) for row in rows]
        return ids, int(total_row["total"] if total_row else 0)

    def scholar_search(self, query: str = "", *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Search authors with indexed name lookup and bounded paper counts."""
        query = str(query or '').strip()
        limit, offset = int(limit), int(offset)
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be >= 0")
        if not query:
            statement = """
            MATCH (a:Author)
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   count { (a)<-[:AUTHORED_BY]-() } AS paper_count
            ORDER BY paper_count DESC, name, scholar_id
            SKIP $offset LIMIT $limit
            """
        elif query:
            statement = """
            CALL db.index.fulltext.queryNodes("author_name_fulltext", $query)
            YIELD node AS a, score
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
            WITH a, score, count(DISTINCT p) AS paper_count
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   paper_count, score
            ORDER BY score DESC, paper_count DESC, name, scholar_id
            SKIP $offset LIMIT $limit
            """
        else:
            statement = """
            MATCH (a:Author)
            WHERE toLower(coalesce(a.name, '')) CONTAINS toLower($query)
               OR toLower(coalesce(a.author_id, '')) CONTAINS toLower($query)
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
            WITH a, count(DISTINCT p) AS paper_count
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   paper_count
            ORDER BY paper_count DESC, name, scholar_id
            SKIP $offset LIMIT $limit
            """
        return self._run(statement, {"query": query, "limit": limit, "offset": offset})

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

    skip_empty_filter = True

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
        self._scholar_cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
        self._entity_cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
        self._health_cache: tuple[float, dict[str, Any]] | None = None

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

    @staticmethod
    def _is_missing_fulltext_index(exc: Exception) -> bool:
        message = str(exc).casefold()
        return "indexnotfound" in message or ("fulltext" in message and "not found" in message)

    def funding_search(self, query: str = "", *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Search funding entities, counting distinct linked papers."""
        limit, offset = int(limit), int(offset)
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be >= 0")
        key = ("funding", str(query or "").strip().casefold(), limit, offset)
        cached = self._entity_cache.get(key)
        if cached and time.monotonic() - cached[0] < 30.0:
            return [dict(row) for row in cached[1]]
        parameters = {"query": str(query or '').strip(), "limit": limit, "offset": offset}
        fallback = """
            MATCH (f:Funding)
            WHERE $query = '' OR toLower(coalesce(f.name, '')) CONTAINS toLower($query)
               OR toLower(coalesce(f.funding_id, '')) CONTAINS toLower($query)
            OPTIONAL MATCH (p:Paper)-[:FUNDED_BY]->(f)
            RETURN f.funding_id AS funding_id, coalesce(f.name, f.funding_id) AS name,
                   count(DISTINCT p) AS paper_count
            ORDER BY paper_count DESC, name, funding_id
            SKIP $offset LIMIT $limit
        """
        statement = fallback
        if parameters["query"]:
            statement = """
                CALL db.index.fulltext.queryNodes("funding_name_fulltext", $query)
                YIELD node AS f, score
                OPTIONAL MATCH (p:Paper)-[:FUNDED_BY]->(f)
                RETURN f.funding_id AS funding_id, coalesce(f.name, f.funding_id) AS name,
                       count(DISTINCT p) AS paper_count, max(score) AS score
                ORDER BY score DESC, paper_count DESC, name, funding_id
                SKIP $offset LIMIT $limit
            """
        try:
            rows = self._run(statement, parameters)
        except RuntimeError as exc:
            if statement == fallback or not self._is_missing_fulltext_index(exc):
                raise
            rows = self._run(fallback, parameters)
        self._entity_cache[key] = (time.monotonic(), [dict(row) for row in rows])
        while len(self._entity_cache) > 256:
            self._entity_cache.pop(next(iter(self._entity_cache)))
        return rows

    def institution_search(self, query: str = "", *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        query = str(query or '').strip()
        bounded_limit = max(1, min(100, int(limit)))
        bounded_offset = max(0, int(offset))
        key = ("institution", query.casefold(), bounded_limit, bounded_offset)
        cached = self._entity_cache.get(key)
        if cached and time.monotonic() - cached[0] < 30.0:
            return [dict(row) for row in cached[1]]
        parameters = {"query": query, "offset": bounded_offset, "limit": bounded_limit}
        fallback = """
            MATCH (i:Institution)
            WHERE $query = '' OR toLower(coalesce(i.name, '')) CONTAINS toLower($query)
               OR toLower(coalesce(i.institution_id, '')) CONTAINS toLower($query)
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(:Author)-[:AFFILIATED_WITH]-(i)
            RETURN coalesce(i.institution_id, i.id) AS institution_id,
                   coalesce(i.name, i.institution_id, i.id) AS name,
                   count(DISTINCT p) AS paper_count
            ORDER BY paper_count DESC, name, institution_id
            SKIP $offset LIMIT $limit
        """
        statement = fallback
        if query:
            statement = """
                CALL db.index.fulltext.queryNodes("institution_name_fulltext", $query)
                YIELD node AS i, score
                OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(:Author)-[:AFFILIATED_WITH]-(i)
                RETURN coalesce(i.institution_id, i.id) AS institution_id,
                       coalesce(i.name, i.institution_id, i.id) AS name,
                       count(DISTINCT p) AS paper_count, max(score) AS score
                ORDER BY score DESC, paper_count DESC, name, institution_id
                SKIP $offset LIMIT $limit
            """
        try:
            rows = self._run(statement, parameters)
        except RuntimeError as exc:
            if statement == fallback or not self._is_missing_fulltext_index(exc):
                raise
            rows = self._run(fallback, parameters)
        self._entity_cache[key] = (time.monotonic(), [dict(row) for row in rows])
        while len(self._entity_cache) > 256:
            self._entity_cache.pop(next(iter(self._entity_cache)))
        return rows

    def scholar_search(self, query: str = "", *, subject: str = "", funding: str = "", limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Find authors by name, topic, and/or funding substring."""
        query = str(query or '').strip()
        subject = str(subject or '').strip()
        funding = str(funding or '').strip()
        cache_key = (query.casefold(), subject.casefold(), funding.casefold(), limit, offset)
        cached = self._scholar_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 30.0:
            return [dict(row) for row in cached[1]]
        if subject and not query and not funding:
            statement = """
            MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
            WHERE toLower(coalesce(t.name, '')) CONTAINS toLower($subject)
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            WITH a, count(DISTINCT p) AS paper_count
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name, paper_count
            ORDER BY paper_count DESC, name, scholar_id SKIP $offset LIMIT $limit
            """
        elif funding and not query and not subject:
            statement = """
            MATCH (p:Paper)-[:FUNDED_BY]->(f:Funding)
            WHERE toLower(coalesce(f.name, '')) CONTAINS toLower($funding)
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            WITH a, count(DISTINCT p) AS paper_count
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name, paper_count
            ORDER BY paper_count DESC, name, scholar_id SKIP $offset LIMIT $limit
            """
        elif not query and not subject and not funding:
            statement = """
            MATCH (a:Author)
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   count { (a)<-[:AUTHORED_BY]-() } AS paper_count
            ORDER BY paper_count DESC, name, scholar_id SKIP $offset LIMIT $limit
            """
        elif query and not subject and not funding:
            statement = """
            CALL db.index.fulltext.queryNodes("author_name_fulltext", $query)
            YIELD node AS a, score
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
            WITH a, score, count(DISTINCT p) AS paper_count
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   paper_count, score
            ORDER BY score DESC, paper_count DESC, name, scholar_id
            SKIP $offset LIMIT $limit
            """
        else:
            statement = """
            MATCH (a:Author)
            WHERE ($query = '' OR toLower(coalesce(a.name, '')) CONTAINS toLower($query)
                   OR toLower(coalesce(a.author_id, '')) CONTAINS toLower($query))
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
            WITH a, collect(DISTINCT p) AS papers
            WHERE ($subject = '' OR any(paper IN papers WHERE EXISTS {
              MATCH (paper)-[:HAS_TOPIC]->(t:Topic)
              WHERE toLower(coalesce(t.name, '')) CONTAINS toLower($subject)
            }))
            AND ($funding = '' OR any(paper IN papers WHERE EXISTS {
              MATCH (paper)-[:FUNDED_BY]->(f:Funding)
              WHERE toLower(coalesce(f.name, '')) CONTAINS toLower($funding)
            }))
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name, size(papers) AS paper_count
            ORDER BY paper_count DESC, name, scholar_id SKIP $offset LIMIT $limit
            """
        parameters = {
            "query": query,
            "subject": subject,
            "funding": funding,
            "offset": max(0, int(offset)),
            "limit": max(1, min(100, int(limit))),
        }
        try:
            rows = self._run(statement, parameters)
        except RuntimeError as exc:
            if not (query and not subject and not funding) or not self._is_missing_fulltext_index(exc):
                raise
            rows = self._run(
                """
                MATCH (a:Author)
                WHERE toLower(coalesce(a.name, '')) CONTAINS toLower($query)
                   OR toLower(coalesce(a.author_id, '')) CONTAINS toLower($query)
                OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
                WITH a, count(DISTINCT p) AS paper_count
                RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                       paper_count
                ORDER BY paper_count DESC, name, scholar_id SKIP $offset LIMIT $limit
                """,
                parameters,
            )
        self._scholar_cache[cache_key] = (time.monotonic(), [dict(row) for row in rows])
        while len(self._scholar_cache) > 256:
            self._scholar_cache.pop(next(iter(self._scholar_cache)))
        return rows

    def scholar_detail(self, scholar_id: str, *, paper_limit: int = 50) -> dict[str, Any] | None:
        rows = self._run(
            """
            MATCH (a:Author {author_id: $scholar_id})
            OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
            WITH a, collect(DISTINCT p) AS papers
            OPTIONAL MATCH (a)-[:AFFILIATED_WITH]-(i:Institution)
            WITH a, papers, collect(DISTINCT coalesce(i.name, i.institution_id)) AS institutions
            UNWIND CASE WHEN size(papers) = 0 THEN [null] ELSE papers END AS p
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(c:Conference)
            OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (p)-[:FUNDED_BY]->(f:Funding)
            WITH a, papers, institutions, collect(DISTINCT coalesce(c.name, c.venue)) AS conferences,
                 collect(DISTINCT t.name) AS topics, collect(DISTINCT f.name) AS funding
            UNWIND CASE WHEN size(papers) = 0 THEN [null] ELSE papers END AS pp
            OPTIONAL MATCH (pp)-[:AUTHORED_BY]->(co:Author)
            WITH a, papers, institutions, conferences, topics, funding,
                 collect(DISTINCT CASE WHEN co.author_id <> a.author_id THEN {scholar_id: co.author_id, name: co.name} END) AS coauthors
            RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
                   institutions, size(papers) AS paper_count,
                   [x IN papers WHERE x IS NOT NULL | x.year] AS years,
                   conferences, topics, funding, coauthors,
                   [x IN papers WHERE x IS NOT NULL | {paper_id:x.paper_id, title:x.title, year:x.year}] [0..$paper_limit] AS papers
            """,
            {"scholar_id": str(scholar_id), "paper_limit": max(1, min(200, int(paper_limit)))},
        )
        return rows[0] if rows else None

    def healthcheck(self) -> dict[str, Any]:
        cached = self._health_cache
        if cached and time.monotonic() - cached[0] < 15.0:
            return dict(cached[1])
        rows = self._run(
            """
            CALL { MATCH (p:Paper) RETURN count(p) AS paper_count }
            CALL { MATCH (f:Funding) RETURN count(f) AS funding_count }
            RETURN paper_count, funding_count
            """,
            {},
        )
        if not rows:
            raise RuntimeError("Neo4j HTTP healthcheck returned no rows")
        result = {
            "status": "ok",
            "paper_count": int(rows[0]["paper_count"]),
            "funding_count": int(rows[0]["funding_count"]),
            "database": self.database,
        }
        self._health_cache = (time.monotonic(), result)
        return dict(result)

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
                "institution": list(filters.institution),
                "funding": list(filters.funding),
            "record_type": filters.record_type,
            },
        )
        return [str(row["paper_id"]) for row in rows]

    def institution_paper_page(
        self, institution: str, *, offset: int = 0, limit: int = 10
    ) -> tuple[list[str], int]:
        """Page papers linked to an institution without lexical/dense recall."""
        institution = str(institution or "").strip()
        offset, limit = int(offset), int(limit)
        if not institution:
            raise ValueError("institution is required")
        if offset < 0 or limit < 1 or limit > 100:
            raise ValueError("offset must be >= 0 and limit must be between 1 and 100")
        base = """
        MATCH (p:Paper)-[:AUTHORED_BY]->(:Author)-[:AFFILIATED_WITH]-(i:Institution)
        WHERE toLower(coalesce(i.name, '')) CONTAINS toLower($institution)
           OR toLower(coalesce(i.institution_id, '')) CONTAINS toLower($institution)
        """
        total_rows = self._run(base + "RETURN count(DISTINCT p) AS total", {"institution": institution})
        rows = self._run(
            base + "RETURN DISTINCT p.paper_id AS paper_id ORDER BY paper_id SKIP $offset LIMIT $limit",
            {"institution": institution, "offset": offset, "limit": limit},
        )
        return [str(row["paper_id"]) for row in rows], int(total_rows[0]["total"] if total_rows else 0)

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
