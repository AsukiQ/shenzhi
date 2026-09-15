import unittest

from evaluate_retrieval import evaluate_rows
from paper_search import SearchResult, SearchState


class FakeEngine:
    def search(self, query, *, filters=None, top_k):
        rankings = {
            "q1": ["a", "b"],
            "q2": ["x", "c"],
        }
        results = [
            SearchResult(
                paper_id=paper_id,
                title=paper_id,
                abstract="",
                conference=None,
                year=None,
                authors=[],
                keywords=[],
                subjects=[],
                score=1.0,
                rank=index + 1,
            )
            for index, paper_id in enumerate(rankings[query][:top_k])
        ]
        return results, SearchState(query=query)


class RetrievalEvaluationTest(unittest.TestCase):
    def test_metrics(self):
        report = evaluate_rows(
            FakeEngine(),
            [
                {"query_id": "1", "query_text": "q1", "positive_skill_id": "a"},
                {"query_id": "2", "query_text": "q2", "positive_skill_id": "c"},
            ],
            top_k=5,
        )
        self.assertEqual(report["query_count"], 2)
        self.assertAlmostEqual(report["metrics"]["mrr"], 0.75)
        self.assertEqual(report["metrics"]["recall_at_1"], 0.5)
        self.assertEqual(report["metrics"]["recall_at_5"], 1.0)

    def test_multi_positive_recall_is_not_only_a_hit_rate(self):
        report = evaluate_rows(
            FakeEngine(),
            [
                {
                    "query_id": "1",
                    "query_text": "q1",
                    "positive_paper_ids": ["a", "missing"],
                }
            ],
            top_k=5,
        )
        self.assertEqual(report["metrics"]["hit_at_5"], 1.0)
        self.assertEqual(report["metrics"]["recall_at_5"], 0.5)


if __name__ == "__main__":
    unittest.main()
