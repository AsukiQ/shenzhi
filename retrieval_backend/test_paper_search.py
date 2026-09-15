import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from paper_search import PaperSearchIndex, SearchFilters


class PaperSearchTest(unittest.TestCase):
    def test_build_search_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {"paper_id": "paper:a", "source_id": "a", "title": "图神经网络用于知识图谱补全", "abstract": "graph neural network knowledge graph completion", "conference": "AAAI", "year": 2024, "authors": ["张三"], "keywords": ["GNN"], "subjects": []},
                        {"paper_id": "paper:b", "source_id": "b", "title": "医学图像分割", "abstract": "medical image segmentation", "conference": "MICCAI", "year": 2023, "authors": ["李四"], "keywords": ["segmentation"], "subjects": []},
                    ]
                ),
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "index.db")
            self.assertEqual(index.build(docs), 2)
            conn = index.connect()
            try:
                self.assertIsNone(
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE name = 'paper_aliases_zh_fts'"
                    ).fetchone()
                )
            finally:
                conn.close()
            # PaperSearchIndex is the English lexical layer; Chinese requests
            # are rewritten before reaching it by the HTTP/hybrid adapter.
            results, state = index.search("knowledge graph completion", filters=SearchFilters(conference=["AAAI"]), top_k=5)
            self.assertEqual([x.paper_id for x in results], ["paper:a"])
            self.assertEqual(state.executed_operations, ["RECALL_PAPER"])
            self.assertEqual(state.last_result_ids, ["paper:a"])

    def test_year_and_author_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(json.dumps({"paper_id": "p", "title": "test", "abstract": "test", "conference": "AAAI", "year": 2020, "authors": ["Alice"], "keywords": [], "subjects": []}) + "\n")
            index = PaperSearchIndex(root / "index.db")
            index.build(docs)
            self.assertEqual(index.search("test", filters=SearchFilters(year_gte=2021))[0], [])
            self.assertEqual(len(index.search("test", filters=SearchFilters(author=["alice"]))[0]), 1)

    def test_search_connections_are_read_only_and_concurrent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps(
                    {
                        "paper_id": "p",
                        "title": "graph completion",
                        "abstract": "knowledge graph completion",
                        "authors": [],
                        "keywords": [],
                        "subjects": [],
                    }
                )
                + "\n"
            )
            index = PaperSearchIndex(root / "index.db")
            index.build(docs)
            first = index.connect()
            second = index.connect()
            try:
                self.assertEqual(first.execute("PRAGMA query_only").fetchone()[0], 1)
                self.assertEqual(second.execute("SELECT count(*) FROM papers").fetchone()[0], 1)
                with self.assertRaises(sqlite3.OperationalError):
                    first.execute("DELETE FROM papers")
            finally:
                first.close()
                second.close()

    def test_relation_and_venue_year_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "seed", "title": "seed", "abstract": "seed", "conference": "AAAI", "year": 2024, "authors": ["Alice"], "keywords": ["graph"], "subjects": ["Data Mining"]},
                        {"paper_id": "author", "title": "author", "abstract": "other", "conference": "CVPR", "year": 2025, "authors": ["Alice"], "keywords": ["vision"], "subjects": ["Vision"]},
                        {"paper_id": "keyword", "title": "keyword", "abstract": "other", "conference": "AAAI", "year": 2023, "authors": ["Bob"], "keywords": ["graph"], "subjects": ["Data Mining"]},
                        {"paper_id": "venue", "title": "venue", "abstract": "other", "conference": "AAAI", "year": 2025, "authors": ["Carol"], "keywords": ["language"], "subjects": ["NLP"]},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "index.db")
            index.build(docs)
            author_hits, authors = index.expand_related(
                ["seed"], relation="authors", limit=10
            )
            self.assertEqual(authors, ["Alice"])
            self.assertEqual([row.paper_id for row in author_hits], ["author"])
            keyword_hits, keywords = index.expand_related(
                ["seed"], relation="keywords", limit=10
            )
            self.assertEqual(keywords, ["graph"])
            self.assertEqual([row.paper_id for row in keyword_hits], ["keyword"])
            subject_hits, subjects = index.expand_related(
                ["seed"], relation="subjects", limit=10
            )
            self.assertEqual(subjects, ["Data Mining"])
            self.assertEqual([row.paper_id for row in subject_hits], ["keyword"])
            venue_hits, pairs = index.expand_venue_year(["seed"], limit=10)
            self.assertEqual(pairs, ["AAAI:2024"])
            self.assertEqual(
                [row.paper_id for row in venue_hits], ["venue", "keyword"]
            )

            conn = index.connect(writable=True)
            try:
                conn.execute("DELETE FROM paper_authors")
                conn.execute("DELETE FROM paper_keywords")
                conn.execute("DELETE FROM paper_subjects")
                conn.commit()
            finally:
                conn.close()
            report = index.backfill_relationship_tables()
            self.assertEqual(report["authors"], 4)
            self.assertEqual(report["keywords"], 4)
            self.assertEqual(report["subjects"], 4)
            self.assertEqual(
                [row.paper_id for row in index.expand_related(
                    ["seed"], relation="authors", limit=10
                )[0]],
                ["author"],
            )


if __name__ == "__main__":
    unittest.main()
