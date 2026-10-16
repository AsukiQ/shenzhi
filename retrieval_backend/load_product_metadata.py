"""Upsert curated Topic and Scholar metadata into the shared paper SQLite.

Input format:
{
  "topics": [{"id": ..., "nameZh": ..., "nameEn": ..., ...,
              "aliases": ["raw subject"]}],
  "scholars": [{"scholar_id": ..., "name_zh": ..., ...,
                "affiliations": [{"institution": ..., "position": ...}],
                "annual_citations": [{"year": 2026, "count": 3}]}]
}

The loader never invents missing values. Re-running it is idempotent for the
same IDs and replaces only the metadata belonging to those IDs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

try:
    from .product_schema import ensure_product_schema
except ImportError:
    from product_schema import ensure_product_schema


def _text(value):
    return str(value or "").strip()


def _topic(row):
    topic_id = _text(row.get("id"))
    if not topic_id:
        raise ValueError("topic.id is required")
    query = _text(row.get("subjectQuery") or row.get("subject_query"))
    aliases = row.get("aliases") or []
    aliases = list(dict.fromkeys(_text(value) for value in aliases if _text(value)))
    if query and query.casefold() not in {value.casefold() for value in aliases}:
        aliases.insert(0, query)
    return {
        "id": topic_id,
        "image_path": _text(row.get("imagePath") or row.get("image_path")) or "/images/topics/default.png",
        "name_zh": _text(row.get("nameZh") or row.get("name_zh")),
        "name_en": _text(row.get("nameEn") or row.get("name_en")),
        "description_zh": _text(row.get("descriptionZh") or row.get("description_zh")),
        "description_en": _text(row.get("descriptionEn") or row.get("description_en")),
        "category": _text(row.get("category")),
        "subject_query": query,
        "enabled": int(bool(row.get("enabled", True))),
        "sort_order": int(row.get("sortOrder", row.get("sort_order", 0)) or 0),
        "updated_at": row.get("updatedAt") or row.get("updated_at"),
        "aliases": aliases,
    }


def _scholar(row):
    scholar_id = _text(row.get("scholar_id") or row.get("scholarId"))
    if not scholar_id:
        raise ValueError("scholar.scholar_id is required")
    metrics = row.get("metrics") or {}
    citation_meta = row.get("citation_meta") or row.get("citationMeta") or {}
    affiliations = []
    for item in row.get("affiliations") or []:
        institution = _text(item.get("institution"))
        if institution:
            affiliations.append((institution, _text(item.get("position"))))
    annual = []
    for item in row.get("annual_citations") or row.get("annualCitations") or []:
        year = int(item["year"])
        count = int(item["count"])
        if count < 0:
            raise ValueError("annual citation count cannot be negative")
        annual.append((year, count))
    return {
        "scholar_id": scholar_id,
        "name_zh": row.get("name_zh", row.get("nameZh")),
        "name_en": row.get("name_en", row.get("nameEn")),
        "avatar_path": row.get("avatar_path", row.get("avatarPath")),
        "bio_zh": row.get("bio_zh", row.get("bioZh")),
        "bio_en": row.get("bio_en", row.get("bioEn")),
        "citation_count": metrics.get("citation_count", metrics.get("citationCount")),
        "citation_source": citation_meta.get("source") or row.get("citation_source"),
        "citation_coverage": citation_meta.get("coverage") or row.get("citation_coverage"),
        "citation_as_of": citation_meta.get("as_of") or row.get("citation_as_of"),
        "updated_at": row.get("updated_at", row.get("updatedAt")),
        "affiliations": affiliations,
        "annual": annual,
    }


def load(db_path: Path, input_path: Path) -> dict[str, int]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    topics = [_topic(item) for item in payload.get("topics", [])]
    scholars = [_scholar(item) for item in payload.get("scholars", [])]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        ensure_product_schema(conn)
        with conn:
            for topic in topics:
                conn.execute(
                    """INSERT INTO topic_catalog
                    (id,image_path,name_zh,name_en,description_zh,description_en,
                     category,subject_query,enabled,sort_order,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                      image_path=excluded.image_path, name_zh=excluded.name_zh,
                      name_en=excluded.name_en, description_zh=excluded.description_zh,
                      description_en=excluded.description_en, category=excluded.category,
                      subject_query=excluded.subject_query, enabled=excluded.enabled,
                      sort_order=excluded.sort_order, updated_at=excluded.updated_at""",
                    tuple(topic[key] for key in (
                        "id", "image_path", "name_zh", "name_en", "description_zh",
                        "description_en", "category", "subject_query", "enabled",
                        "sort_order", "updated_at",
                    )),
                )
                conn.execute("DELETE FROM topic_aliases WHERE topic_id=?", (topic["id"],))
                conn.executemany(
                    "INSERT INTO topic_aliases(topic_id,alias,alias_key) VALUES (?,?,?)",
                    [(topic["id"], alias, alias.casefold()) for alias in topic["aliases"]],
                )
            for scholar in scholars:
                conn.execute(
                    """INSERT INTO scholar_profiles
                    (scholar_id,name_zh,name_en,avatar_path,bio_zh,bio_en,citation_count,
                     citation_source,citation_coverage,citation_as_of,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(scholar_id) DO UPDATE SET
                      name_zh=excluded.name_zh, name_en=excluded.name_en,
                      avatar_path=excluded.avatar_path, bio_zh=excluded.bio_zh,
                      bio_en=excluded.bio_en, citation_count=excluded.citation_count,
                      citation_source=excluded.citation_source,
                      citation_coverage=excluded.citation_coverage,
                      citation_as_of=excluded.citation_as_of,
                      updated_at=excluded.updated_at""",
                    tuple(scholar[key] for key in (
                        "scholar_id", "name_zh", "name_en", "avatar_path", "bio_zh",
                        "bio_en", "citation_count", "citation_source", "citation_coverage",
                        "citation_as_of", "updated_at",
                    )),
                )
                conn.execute("DELETE FROM scholar_affiliations WHERE scholar_id=?", (scholar["scholar_id"],))
                conn.executemany(
                    "INSERT INTO scholar_affiliations(scholar_id,institution,position) VALUES (?,?,?)",
                    [(scholar["scholar_id"], institution, position) for institution, position in scholar["affiliations"]],
                )
                conn.execute("DELETE FROM scholar_citations WHERE scholar_id=?", (scholar["scholar_id"],))
                conn.executemany(
                    "INSERT INTO scholar_citations(scholar_id,year,count) VALUES (?,?,?)",
                    [(scholar["scholar_id"], year, count) for year, count in scholar["annual"]],
                )
        return {"topics": len(topics), "scholars": len(scholars)}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(load(args.db, args.input), ensure_ascii=False))


if __name__ == "__main__":
    main()

