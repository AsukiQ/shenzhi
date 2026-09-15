"""可复现的论文检索原型。

默认只依赖 Python 标准库和 SQLite FTS5。它把
``retrieval_documents.jsonl`` 建成一个本地英文 FTS5 索引，支持关键词/BM25、
标题加权和年份/会议/作者/关键词等结构化过滤。中文请求在进入索引前统一由
查询改写服务转成英文；后续接入 LSTR 时，只需替换 ``rank``，不用改变 paper_id
和返回协议。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    row_id INTEGER PRIMARY KEY,
    paper_id TEXT NOT NULL UNIQUE,
    source_id TEXT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    conference TEXT,
    year INTEGER,
    authors_json TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    subjects_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id UNINDEXED,
    title,
    abstract,
    keywords,
    subjects,
    tokenize='unicode61'
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_conference ON papers(conference);
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL,
    value TEXT NOT NULL,
    value_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (paper_id, value_key)
);
CREATE TABLE IF NOT EXISTS paper_keywords (
    paper_id TEXT NOT NULL,
    value TEXT NOT NULL,
    value_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (paper_id, value_key)
);
CREATE TABLE IF NOT EXISTS paper_subjects (
    paper_id TEXT NOT NULL,
    value TEXT NOT NULL,
    value_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (paper_id, value_key)
);
CREATE INDEX IF NOT EXISTS idx_paper_authors_value ON paper_authors(value_key, paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_keywords_value ON paper_keywords(value_key, paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_subjects_value ON paper_subjects(value_key, paper_id);
"""

RELATED_TABLES = {
    "authors": "paper_authors",
    "keywords": "paper_keywords",
    "subjects": "paper_subjects",
}

# Small, auditable corrections for common English paper-search typos. This is
# intentionally independent from Chinese rewriting and remains a low-weight
# fallback channel.
FUZZY_TOKEN_ALIASES = {
    "knowlege": "knowledge",
    "knoweldge": "knowledge",
    "graf": "graph",
    "grahp": "graph",
    "retrival": "retrieval",
    "retrievel": "retrieval",
    "netwrok": "network",
    "neuralnet": "neural network",
    "langauge": "language",
    "transfomer": "transformer",
    "sematic": "semantic",
    "reccomendation": "recommendation",
}


def _tokens(text: str) -> list[str]:
    """生成适合英文论文 FTS5 索引的词元。

    中文请求在进入这个索引前统一经过中译英改写；这里不再维护第二套
    中文 alias/BM25 索引，避免出现两套互相竞争的静态召回逻辑。
    """
    text = str(text or "").lower()
    result = re.findall(r"[a-z0-9][a-z0-9_+.-]*", text)
    return result


def _fts_text(value: Any) -> str:
    return " ".join(_tokens(str(value or "")))


def rewrite_fuzzy_tokens(query: str) -> str:
    """Apply only the small audited typo map, preserving unknown tokens."""

    tokens = str(query or "").split()
    return " ".join(FUZZY_TOKEN_ALIASES.get(token.casefold(), token) for token in tokens)


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value if str(x).strip()]


def _str_list(value: Any) -> list[str]:
    """兼容 CLI/HTTP 中的单字符串或字符串数组。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if str(x).strip()]


@dataclass
class SearchFilters:
    year_gte: int | None = None
    year_lte: int | None = None
    conference: list[str] = field(default_factory=list)
    author: list[str] = field(default_factory=list)
    keyword: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)


@dataclass
class SearchState:
    """LSTR 风格的可序列化检索记忆；原型阶段由规则更新。"""

    query: str
    step: int = 0
    visited_ids: list[str] = field(default_factory=list)
    executed_operations: list[str] = field(default_factory=list)
    last_result_ids: list[str] = field(default_factory=list)
    satisfied_constraints: list[str] = field(default_factory=list)
    remaining_constraints: list[str] = field(default_factory=list)
    failed_operations: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    paper_id: str
    title: str
    abstract: str
    conference: str | None
    year: int | None
    authors: list[str]
    keywords: list[str]
    subjects: list[str]
    score: float
    rank: int
    source_scores: dict[str, float] = field(default_factory=dict)
    retrieval_mode: str = "bm25"
    # Placeholder for the future PDF/figure extraction pipeline. Keeping the
    # field stable lets clients integrate before image data is available.
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateHit:
    paper_id: str
    score: float
    rank: int
    source: str = "bm25"


class PaperSearchIndex:
    """SQLite FTS5 索引；数据库文件是可删除重建的派生文件。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._local = threading.local()

    def connect(self, *, writable: bool = False) -> sqlite3.Connection:
        if writable:
            conn = sqlite3.connect(self.db_path)
        else:
            path = Path(self.db_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"paper index not found: {path}")
            conn = sqlite3.connect(
                f"{path.as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
        conn.row_factory = sqlite3.Row
        if writable:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        else:
            conn.execute("PRAGMA query_only=ON")
        return conn

    def read_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "read_connection", None)
        if conn is None:
            conn = self.connect()
            self._local.read_connection = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "read_connection", None)
        if conn is not None:
            conn.close()
            del self._local.read_connection

    def healthcheck(self) -> dict[str, Any]:
        conn = self.read_connection()
        paper_count = int(conn.execute("SELECT count(*) FROM papers").fetchone()[0])
        return {
            "status": "ok",
            "backend": "sqlite_fts5",
            "paper_count": paper_count,
            "db_path": str(Path(self.db_path).resolve()),
        }

    def build(self, documents_path: str | Path, *, replace: bool = False) -> int:
        path = Path(documents_path)
        self.close()
        conn = self.connect(writable=True)
        try:
            existing = int(conn.execute("SELECT count(*) FROM papers").fetchone()[0])
            if existing and not replace:
                raise RuntimeError(
                    f"index already contains {existing} papers; pass replace=True to rebuild "
                    "so stale FTS rows cannot accumulate"
                )
            if replace:
                # Remove the retired Chinese alias index from databases built
                # by older releases. New schemas never create it.
                conn.execute("DROP TABLE IF EXISTS paper_aliases_zh_fts")
                conn.execute("DELETE FROM paper_authors")
                conn.execute("DELETE FROM paper_keywords")
                conn.execute("DELETE FROM paper_subjects")
                conn.execute("DELETE FROM papers_fts")
                conn.execute("DELETE FROM papers")
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    paper_id = str(doc["paper_id"])
                    authors = _json_list(doc.get("authors"))
                    keywords = _json_list(doc.get("keywords"))
                    subjects = _json_list(doc.get("subjects"))
                    year = doc.get("year")
                    year = int(year) if year not in (None, "") else None
                    cur = conn.execute(
                        """INSERT OR REPLACE INTO papers
                        (paper_id,source_id,title,abstract,conference,year,authors_json,keywords_json,subjects_json)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        (paper_id, doc.get("source_id"), doc.get("title", ""),
                         doc.get("abstract", ""), doc.get("conference"), year,
                         json.dumps(authors, ensure_ascii=False), json.dumps(keywords, ensure_ascii=False),
                         json.dumps(subjects, ensure_ascii=False)),
                    )
                    # ``replace`` 模式下索引为空；同一批文档的 paper_id 已由
                    # 数据准备阶段保证唯一。为避免对 12 万条记录逐条扫描 FTS，
                    # 增量更新请先用 ``build(..., replace=True)`` 重建索引。
                    conn.execute(
                        "INSERT INTO papers_fts(paper_id,title,abstract,keywords,subjects) VALUES (?,?,?,?,?)",
                        (paper_id, _fts_text(doc.get("title")), _fts_text(doc.get("abstract")),
                         _fts_text(" ".join(keywords)), _fts_text(" ".join(subjects))),
                    )
                    for table, values in (
                        ("paper_authors", authors),
                        ("paper_keywords", keywords),
                        ("paper_subjects", subjects),
                    ):
                        for position, value in enumerate(values):
                            normalized = str(value).strip()
                            if not normalized:
                                continue
                            conn.execute(
                                f"INSERT OR IGNORE INTO {table}(paper_id,value,value_key,position) "
                                "VALUES (?,?,?,?)",
                                (paper_id, normalized, normalized.casefold(), position),
                            )
                    count += 1
                    if count % 2000 == 0:
                        conn.commit()
            conn.commit()
            return count
        finally:
            conn.close()

    def backfill_relationship_tables(self, *, replace: bool = False) -> dict[str, int]:
        """Populate normalized relation tables for an index built by an older release."""

        self.close()
        conn = self.connect(writable=True)
        inserted = {"authors": 0, "keywords": 0, "subjects": 0}
        try:
            if replace:
                for table in RELATED_TABLES.values():
                    conn.execute(f"DELETE FROM {table}")
            rows = conn.execute(
                "SELECT paper_id,authors_json,keywords_json,subjects_json FROM papers "
                "ORDER BY row_id"
            )
            for row_index, row in enumerate(rows, start=1):
                paper_id = str(row["paper_id"])
                for relation, raw_json in (
                    ("authors", row["authors_json"]),
                    ("keywords", row["keywords_json"]),
                    ("subjects", row["subjects_json"]),
                ):
                    table = RELATED_TABLES[relation]
                    for position, value in enumerate(json.loads(raw_json)):
                        normalized = str(value).strip()
                        if not normalized:
                            continue
                        cursor = conn.execute(
                            f"INSERT OR IGNORE INTO {table}(paper_id,value,value_key,position) "
                            "VALUES (?,?,?,?)",
                            (paper_id, normalized, normalized.casefold(), position),
                        )
                        inserted[relation] += int(cursor.rowcount > 0)
                if row_index % 5000 == 0:
                    conn.commit()
            conn.commit()
            return inserted
        finally:
            conn.close()

    @staticmethod
    def _match(value: str | None, wanted: Iterable[str]) -> bool:
        value = (value or "").casefold()
        return any(str(item).casefold() in value for item in wanted)

    @classmethod
    def _passes_filters(cls, row: sqlite3.Row, filters: SearchFilters) -> bool:
        authors = json.loads(row["authors_json"])
        keywords = json.loads(row["keywords_json"])
        subjects = json.loads(row["subjects_json"])
        if filters.year_gte is not None and (
            row["year"] is None or row["year"] < filters.year_gte
        ):
            return False
        if filters.year_lte is not None and (
            row["year"] is None or row["year"] > filters.year_lte
        ):
            return False
        if filters.conference and not cls._match(row["conference"], filters.conference):
            return False
        if filters.author and not any(cls._match(author, filters.author) for author in authors):
            return False
        if filters.keyword and not any(cls._match(keyword, filters.keyword) for keyword in keywords):
            return False
        if filters.subject and not any(cls._match(subject, filters.subject) for subject in subjects):
            return False
        return True

    def recall(self, query: str, *, limit: int = 1000) -> list[CandidateHit]:
        """Return BM25 candidates without applying metadata/graph filters."""

        limit = max(1, min(int(limit), 5000))
        tokens = _tokens(query)
        conn = self.read_connection()
        if tokens:
            match = " OR ".join(
                '"' + token.replace('"', '""') + '"'
                for token in dict.fromkeys(tokens)
            )
            rows = conn.execute(
                "SELECT paper_id, bm25(papers_fts, 5.0, 1.0, 3.0, 2.0) AS bm "
                "FROM papers_fts WHERE papers_fts MATCH ? ORDER BY bm LIMIT ?",
                (match, limit),
            ).fetchall()
            return [
                CandidateHit(
                    paper_id=str(row["paper_id"]),
                    score=max(0.0, -float(row["bm"])),
                    rank=index + 1,
                )
                for index, row in enumerate(rows)
            ]
        # An empty/unsupported query must not silently return the newest
        # papers.  In the translation-first design this is also the safe
        # behavior when a Chinese rewrite is unavailable.
        return []

    def recall_prefix(self, query: str, *, limit: int = 300) -> list[CandidateHit]:
        """Bounded typo/partial-word recall using FTS5 prefix terms.

        This secondary channel tolerates a missing suffix or a small audited
        typo without turning every short query into a broad table scan.
        """

        limit = max(1, min(int(limit), 1000))
        token_groups: list[list[str]] = []
        for token in dict.fromkeys(
            re.findall(r"[a-z0-9][a-z0-9_+.-]*", str(query or "").lower())
        ):
            correction = FUZZY_TOKEN_ALIASES.get(token)
            if correction:
                token_groups.append(_tokens(correction))
                continue
            if len(token) < 4:
                continue
            prefix_len = max(3, len(token) - 3)
            token_groups.append([token[:prefix_len]])
        if not token_groups:
            return []
        conn = self.read_connection()
        groups = []
        for alternatives in token_groups:
            terms = []
            for term in dict.fromkeys(alternatives):
                escaped = term.replace('"', '""')
                terms.append(
                    '"' + escaped + '"'
                    + ("" if term in FUZZY_TOKEN_ALIASES.values() else "*")
                )
            groups.append("(" + " OR ".join(terms) + ")")
        match = " AND ".join(groups)
        rows = conn.execute(
            "SELECT paper_id, bm25(papers_fts, 5.0, 1.0, 3.0, 2.0) AS bm "
            "FROM papers_fts WHERE papers_fts MATCH ? ORDER BY bm LIMIT ?",
            (match, limit),
        ).fetchall()
        return [
            CandidateHit(
                paper_id=str(row["paper_id"]),
                score=max(0.0, -float(row["bm"])),
                rank=index + 1,
                source="bm25_fuzzy",
            )
            for index, row in enumerate(rows)
        ]

    def browse(self, *, filters: SearchFilters | None = None, limit: int = 1000) -> list[CandidateHit]:
        """Return a deterministic candidate list for a pure metadata query.

        This is deliberately separate from :meth:`recall`: an empty semantic
        query must never be treated as a BM25 query (or silently replaced by
        an arbitrary "latest papers" fallback).  Results are ordered by
        publication year and row id, then the normal local/Neo4j filters are
        still applied by the retrieval pipeline.
        """

        filters = filters or SearchFilters()
        limit = max(1, min(int(limit), 5000))
        conn = self.read_connection()
        rows = conn.execute(
            "SELECT * FROM papers ORDER BY year IS NULL, year DESC, row_id DESC"
        ).fetchall()
        hits: list[CandidateHit] = []
        for row in rows:
            if not self._passes_filters(row, filters):
                continue
            rank = len(hits) + 1
            hits.append(
                CandidateHit(
                    paper_id=str(row["paper_id"]),
                    score=1.0 / float(rank),
                    rank=rank,
                    source="filtered",
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def fetch_results(
        self,
        paper_ids: Iterable[str],
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        score_by_id: Mapping[str, float] | None = None,
        source_scores_by_id: Mapping[str, Mapping[str, float]] | None = None,
        retrieval_mode: str = "bm25",
    ) -> list[SearchResult]:
        """Hydrate an ordered candidate list and apply local metadata filters."""

        ordered_ids = list(dict.fromkeys(str(item) for item in paper_ids if str(item)))
        if not ordered_ids:
            return []
        filters = filters or SearchFilters()
        scores = score_by_id or {}
        source_scores = source_scores_by_id or {}
        conn = self.read_connection()
        by_id: dict[str, sqlite3.Row] = {}
        # Stay below SQLite's common host-parameter limit.
        for start in range(0, len(ordered_ids), 800):
            chunk = ordered_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT * FROM papers WHERE paper_id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            by_id.update({str(row["paper_id"]): row for row in rows})

        results: list[SearchResult] = []
        limit = None if top_k is None else max(1, int(top_k))
        for paper_id in ordered_ids:
            row = by_id.get(paper_id)
            if row is None or not self._passes_filters(row, filters):
                continue
            results.append(
                SearchResult(
                    paper_id=paper_id,
                    title=str(row["title"]),
                    abstract=str(row["abstract"]),
                    conference=row["conference"],
                    year=row["year"],
                    authors=json.loads(row["authors_json"]),
                    keywords=json.loads(row["keywords_json"]),
                    subjects=json.loads(row["subjects_json"]),
                    score=float(scores.get(paper_id, 0.0)),
                    rank=len(results) + 1,
                    source_scores={
                        str(key): float(value)
                        for key, value in source_scores.get(paper_id, {}).items()
                    },
                    retrieval_mode=str(retrieval_mode),
                )
            )
            if limit is not None and len(results) >= limit:
                break
        return results

    def related_values(
        self,
        paper_ids: Iterable[str],
        *,
        relation: str,
        limit: int = 12,
    ) -> list[str]:
        """Return deterministic relation values carried by ordered seed papers."""

        table = RELATED_TABLES.get(str(relation))
        if table is None:
            raise ValueError(f"unsupported paper relation: {relation}")
        ordered_ids = list(dict.fromkeys(str(item) for item in paper_ids if str(item)))
        if not ordered_ids:
            return []
        conn = self.read_connection()
        by_paper: dict[str, list[sqlite3.Row]] = {}
        for start in range(0, len(ordered_ids), 800):
            chunk = ordered_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT paper_id,value,value_key,position FROM {table} "
                f"WHERE paper_id IN ({placeholders}) ORDER BY position,value_key",
                tuple(chunk),
            ).fetchall()
            for row in rows:
                by_paper.setdefault(str(row["paper_id"]), []).append(row)
        output: list[str] = []
        seen: set[str] = set()
        for paper_id in ordered_ids:
            for row in by_paper.get(paper_id, []):
                key = str(row["value_key"])
                if key in seen:
                    continue
                seen.add(key)
                output.append(str(row["value"]))
                if len(output) >= max(1, int(limit)):
                    return output
        return output

    def expand_related(
        self,
        paper_ids: Iterable[str],
        *,
        relation: str,
        limit: int = 100,
        value_limit: int = 12,
        slot_index: int | None = None,
    ) -> tuple[list[CandidateHit], list[str]]:
        """Expand papers by exact shared Author/Keyword/Subject values."""

        table = RELATED_TABLES.get(str(relation))
        if table is None:
            raise ValueError(f"unsupported paper relation: {relation}")
        seeds = list(dict.fromkeys(str(item) for item in paper_ids if str(item)))
        values = self.related_values(seeds, relation=relation, limit=value_limit)
        if not seeds or not values:
            return [], values
        selected_values = values
        if slot_index is not None:
            slot = int(slot_index)
            if not 0 <= slot < len(values):
                return [], []
            selected_values = [values[slot]]
        keys = [value.casefold() for value in selected_values]
        conn = self.read_connection()
        value_marks = ",".join("?" for _ in keys)
        seed_marks = ",".join("?" for _ in seeds)
        rows = conn.execute(
            f"SELECT m.paper_id, COUNT(DISTINCT m.value_key) AS shared, "
            "COALESCE(p.year, -1) AS paper_year "
            f"FROM {table} AS m JOIN papers AS p ON p.paper_id=m.paper_id "
            f"WHERE m.value_key IN ({value_marks}) "
            f"AND m.paper_id NOT IN ({seed_marks}) "
            "GROUP BY m.paper_id "
            "ORDER BY shared DESC, paper_year DESC, m.paper_id "
            "LIMIT ?",
            (*keys, *seeds, max(1, min(int(limit), 1000))),
        ).fetchall()
        return [
            CandidateHit(
                paper_id=str(row["paper_id"]),
                score=float(row["shared"]),
                rank=index + 1,
                source=f"expand_{relation}",
            )
            for index, row in enumerate(rows)
        ], selected_values

    def expand_venue_year(
        self,
        paper_ids: Iterable[str],
        *,
        limit: int = 100,
        seed_limit: int = 20,
        year_window: int = 1,
        value_limit: int = 128,
        slot_index: int | None = None,
    ) -> tuple[list[CandidateHit], list[str]]:
        """Expand within seed venues and nearby publication years."""

        seeds = list(dict.fromkeys(str(item) for item in paper_ids if str(item)))[
            : max(1, int(seed_limit))
        ]
        if not seeds:
            return [], []
        conn = self.read_connection()
        marks = ",".join("?" for _ in seeds)
        seed_rows = conn.execute(
            f"SELECT paper_id,conference,year FROM papers WHERE paper_id IN ({marks})",
            tuple(seeds),
        ).fetchall()
        pairs = sorted(
            {
                (str(row["conference"]), int(row["year"]))
                for row in seed_rows
                if row["conference"] and row["year"] is not None
            }
        )
        if not pairs:
            return [], []
        pairs = pairs[: max(1, int(value_limit))]
        selected_pairs = pairs
        if slot_index is not None:
            slot = int(slot_index)
            if not 0 <= slot < len(pairs):
                return [], []
            selected_pairs = [pairs[slot]]
        venues = sorted({conference for conference, _year in selected_pairs})
        venue_marks = ",".join("?" for _ in venues)
        seed_marks = ",".join("?" for _ in seeds)
        min_year = min(year for _conference, year in selected_pairs) - max(0, int(year_window))
        max_year = max(year for _conference, year in selected_pairs) + max(0, int(year_window))
        rows = conn.execute(
            f"SELECT paper_id,conference,year FROM papers "
            f"WHERE conference IN ({venue_marks}) AND year BETWEEN ? AND ? "
            f"AND paper_id NOT IN ({seed_marks})",
            (*venues, min_year, max_year, *seeds),
        ).fetchall()
        scored: list[tuple[str, float, int]] = []
        for row in rows:
            conference = str(row["conference"] or "")
            year = int(row["year"])
            distance = min(
                abs(year - seed_year)
                for seed_venue, seed_year in selected_pairs
                if seed_venue == conference
            )
            score = 2.0 if distance == 0 else 1.0 / float(1 + distance)
            scored.append((str(row["paper_id"]), score, year))
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        selected = scored[: max(1, min(int(limit), 1000))]
        return [
            CandidateHit(paper_id, score, rank, "expand_venue_year")
            for rank, (paper_id, score, _year) in enumerate(selected, start=1)
        ], [f"{conference}:{year}" for conference, year in selected_pairs]

    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        state: SearchState | None = None,
    ) -> tuple[list[SearchResult], SearchState]:
        filters = filters or SearchFilters()
        top_k = max(1, min(int(top_k), 100))
        candidates = self.recall(query, limit=1000)
        score_by_id = {candidate.paper_id: candidate.score for candidate in candidates}
        results = self.fetch_results(
            [candidate.paper_id for candidate in candidates],
            filters=filters,
            top_k=top_k,
            score_by_id=score_by_id,
            source_scores_by_id={
                candidate.paper_id: {"bm25": candidate.score}
                for candidate in candidates
            },
        )
        new_state = state or SearchState(query=query)
        new_state.query = query
        new_state.step += 1
        new_state.executed_operations.append("RECALL_PAPER")
        new_state.last_result_ids = [result.paper_id for result in results]
        return results, new_state


def state_to_dict(state: SearchState) -> dict[str, Any]:
    return asdict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/search Shenzhi paper FTS5 index")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--documents", required=True)
    b.add_argument("--db", required=True)
    b.add_argument("--replace", action="store_true")
    m = sub.add_parser("migrate-relations")
    m.add_argument("--db", required=True)
    m.add_argument("--replace", action="store_true")
    s = sub.add_parser("search")
    s.add_argument("--db", required=True)
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--year-gte", type=int)
    s.add_argument("--year-lte", type=int)
    s.add_argument("--conference", action="append", default=[])
    s.add_argument("--author", action="append", default=[])
    s.add_argument("--keyword", action="append", default=[])
    args = parser.parse_args()
    index = PaperSearchIndex(args.db)
    if args.command == "build":
        print(json.dumps({"indexed": index.build(args.documents, replace=args.replace)}, ensure_ascii=False))
        return
    if args.command == "migrate-relations":
        print(
            json.dumps(
                {"status": "ok", "inserted": index.backfill_relationship_tables(replace=args.replace)},
                ensure_ascii=False,
            )
        )
        return
    results, state = index.search(args.query, filters=SearchFilters(args.year_gte, args.year_lte, args.conference, args.author, args.keyword), top_k=args.top_k)
    print(json.dumps({"results": [asdict(r) for r in results], "state": state_to_dict(state)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
