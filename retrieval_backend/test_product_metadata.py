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
                    "affiliation_history": [{"institution": "Example University", "years": [2022, 2024], "basis": "publication_affiliation"}],
                    "research_impact": {"h_index": 3, "i10_index": 0, "as_of": None},
                    "affiliations": [{"institution": "Example University", "position": "Professor"}] * 2,
                    "annual_citations": [{"year": 2026, "count": 4}],
                }],
            }), encoding="utf-8")
            self.assertEqual(load(source, metadata), {"topics": 1, "scholars": 1})
            self.assertEqual(load(source, metadata), {"topics": 1, "scholars": 1})
            with sqlite3.connect(source) as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM scholar_affiliations').fetchone()[0], 1)
                conn.execute("INSERT INTO scholar_paper_links VALUES ('author:test','paper:test','OpenAlex DOI')")
            target = root / "rebuilt.db"
            self.assertEqual(rebuild(source, target), 1)
            conn = sqlite3.connect(target)
            try:
                self.assertEqual(conn.execute("SELECT name_en FROM topic_catalog").fetchone()[0], "Deep Learning")
                self.assertEqual(conn.execute("SELECT alias FROM topic_aliases").fetchone()[0], "deep learning")
                self.assertEqual(conn.execute("SELECT position FROM scholar_affiliations").fetchone()[0], "Professor")
                self.assertEqual(conn.execute("SELECT count FROM scholar_citations").fetchone()[0], 4)
                self.assertEqual(conn.execute('SELECT paper_id FROM scholar_paper_links').fetchone()[0], 'paper:test')
                details = conn.execute('SELECT affiliation_history_json,research_impact_json FROM scholar_details').fetchone()
                self.assertEqual(json.loads(details[0])[0]['years'], [2022, 2024])
                self.assertEqual(json.loads(details[1])['i10_index'], 0)
                self.assertIsNone(json.loads(details[1])['as_of'])
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

