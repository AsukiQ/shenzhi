import json
import tempfile
import unittest
from pathlib import Path

from evaluate_graph_ablation import run_ablation
from paper_search import PaperSearchIndex


class GraphAblationTest(unittest.TestCase):
    def test_operational_ablation_runs_without_manufactured_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "paper_id": "seed",
                            "title": "unique graph paper",
                            "abstract": "retrieval",
                            "authors": ["Alice"],
                        },
                        {
                            "paper_id": "related",
                            "title": "different work",
                            "abstract": "reasoning",
                            "authors": ["Alice"],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            db = root / "papers.db"
            PaperSearchIndex(db).build(docs)
            report = run_ablation(
                db=db,
                rows=[{"query_id": "q1", "query": "unique graph 找同作者的其他论文"}],
                top_k=2,
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["protocol"], "shenzhi_graph_ablation_v1")
            self.assertIsNone(report["quality"])
            self.assertGreaterEqual(
                report["graph"]["operation_counts"].get(
                    "EXPAND_PAPER_AUTHORS_SQLITE", 0
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()

