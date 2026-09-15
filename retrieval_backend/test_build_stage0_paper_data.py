import json
import tempfile
import unittest
from pathlib import Path

from build_stage0_paper_data import build


class Stage0PaperDataTest(unittest.TestCase):
    def test_build_is_deterministic_and_has_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "docs.jsonl"
            docs = [
                {"paper_id": "paper:a", "source_id": "a", "title": "Graph completion", "abstract": "Graph neural networks help completion.", "conference": "AAAI", "year": 2024, "keywords": ["gnn"], "authors": [], "retrieval_eligible": True},
                {"paper_id": "paper:b", "source_id": "b", "title": "Graph ranking", "abstract": "Ranking graphs.", "conference": "AAAI", "year": 2024, "keywords": ["graph"], "authors": [], "retrieval_eligible": True},
            ]
            source.write_text("\n".join(json.dumps(x) for x in docs) + "\n", encoding="utf-8")
            report = build(source, root / "out", negative_count=1)
            self.assertEqual(report["paper_count"], 2)
            rows = [json.loads(x) for x in (root / "out" / "retrieval.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r["positive_skill_id"] in {"paper:a", "paper:b"} for r in rows))
            self.assertTrue(all(r["positive_skill_id"] not in r["negative_skill_ids"] for r in rows))
            self.assertEqual(set(report["splits"]), {"train", "dev", "test"} & set(report["splits"]))


if __name__ == "__main__":
    unittest.main()
