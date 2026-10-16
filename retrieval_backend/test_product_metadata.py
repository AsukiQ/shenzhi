import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from load_product_metadata import load
from paper_search import PaperSearchIndex
from rebuild_search_index import rebuild


class ProductMetadataTest(unittest.TestCase):
    def test_overlay_is_idempotent_and_survives_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "papers.db"
            docs = root / "docs.jsonl"
            docs.write_text(json.dumps({
                "paper_id": "paper:test",
                "source_id": "test",
                "title": "Product metadata paper",
                "abstract": "A paper about deep learning",
                "year": 2026,
                "subjects": ["deep learning"],
                "authors": ["Alice"],
                "keywords": [],
                "funding": [],
                "image_id": "cover.png",
            }) + "\n", encoding="utf-8")
            self.assertEqual(PaperSearchIndex(source).build(docs), 1)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "topics": [{
                    "id": "deep-learning",
                    "nameZh": "深度学习",
                    "nameEn": "Deep Learning",
                    "descriptionZh": "一种机器学习方法。",
                    "descriptionEn": "A machine learning method.",
                    "category": "机器学习",
                    "subjectQuery": "deep learning",
                    "aliases": ["deep learning"],
                    "enabled": True,
                }],
                "scholars": [{
                    "scholar_id": "author:test",
                    "name_en": "Alice",
                    "affiliations": [{"institution": "Example University", "position": "Professor"}],
                    "annual_citations": [{"year": 2026, "count": 4}],
                }],
            }), encoding="utf-8")
            self.assertEqual(load(source, metadata), {"topics": 1, "scholars": 1})
            self.assertEqual(load(source, metadata), {"topics": 1, "scholars": 1})
            target = root / "rebuilt.db"
            self.assertEqual(rebuild(source, target), 1)
            conn = sqlite3.connect(target)
            try:
                self.assertEqual(conn.execute("SELECT name_en FROM topic_catalog").fetchone()[0], "Deep Learning")
                self.assertEqual(conn.execute("SELECT alias FROM topic_aliases").fetchone()[0], "deep learning")
                self.assertEqual(conn.execute("SELECT position FROM scholar_affiliations").fetchone()[0], "Professor")
                self.assertEqual(conn.execute("SELECT count FROM scholar_citations").fetchone()[0], 4)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

