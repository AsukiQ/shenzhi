import json
from pathlib import Path
import tempfile
import unittest

from evaluate_bm25_parallel import evaluate_parallel
from paper_search import PaperSearchIndex


class ParallelBm25EvaluationTest(unittest.TestCase):
    def test_exact_metrics_match_expected_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "a", "title": "graph completion", "abstract": "", "authors": [], "keywords": [], "subjects": []},
                        {"paper_id": "b", "title": "vision", "abstract": "image", "authors": [], "keywords": [], "subjects": []},
                    )
                )
                + "\n"
            )
            db = root / "papers.db"
            PaperSearchIndex(db).build(docs)
            report = evaluate_parallel(
                db_path=db,
                rows=[
                    {"query_id": "1", "query_text": "graph", "positive_paper_id": "a"},
                    {"query_id": "2", "query_text": "vision", "positive_paper_id": "b"},
                ],
                top_k=2,
                workers=2,
            )
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["metrics"]["mrr"], 1.0)
            self.assertEqual(report["metrics"]["recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
