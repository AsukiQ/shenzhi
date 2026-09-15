import json
import tempfile
import unittest
from pathlib import Path

from paper_search import PaperSearchIndex, SearchFilters
from retrieval_pipeline import (
    HybridPaperSearch,
    StaticRunDenseRetriever,
    reciprocal_rank_fusion,
)
from paper_search import CandidateHit


class FixedReranker:
    def rerank(self, _query, candidates):
        return sorted(
            [(row.paper_id, 2.0 if row.paper_id == "a" else 1.0) for row in candidates],
            key=lambda item: -item[1],
        )


class RecordingReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, candidates):
        self.queries.append(query)
        return [(candidate.paper_id, 1.0) for candidate in candidates]


class RecordingGraphFilter:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.calls = []

    def filter_candidate_ids(self, candidate_ids, *, filters):
        self.calls.append((list(candidate_ids), filters))
        return [paper_id for paper_id in candidate_ids if paper_id in self.allowed]


class UnavailableGraphFilter:
    def healthcheck(self):
        raise ConnectionError("Neo4j unavailable")


class RetrievalPipelineTest(unittest.TestCase):
    def test_optional_graph_failure_reports_degraded_but_remains_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "graph retrieval"}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            engine = HybridPaperSearch(index, graph_filter=UnavailableGraphFilter())
            health = engine.healthcheck()
            readiness = engine.readiness()
            self.assertEqual(health["status"], "degraded")
            self.assertEqual(health["graph"]["status"], "error")
            self.assertEqual(health["graph"]["error"], "ConnectionError")
            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["status"], "degraded")

    def test_chinese_query_uses_english_rewrite_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "knowledge graph completion", "abstract": "graph neural network"}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            results, state = HybridPaperSearch(index).search(
                "知识图谱补全",
                query_variants={"translated_query": "knowledge graph completion"},
                top_k=1,
            )
            self.assertEqual(results[0].paper_id, "p")
            self.assertIn("RECALL_PAPER_BM25", state.executed_operations)
            self.assertNotIn("RECALL_PAPER_BM25_ZH", state.executed_operations)

    def test_chinese_query_without_rewrite_does_not_return_recent_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "knowledge graph completion", "abstract": "graph neural network"}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            results, state = HybridPaperSearch(index).search("一个未翻译的中文问题")
            self.assertEqual(results, [])
            self.assertIn("QUERY_REWRITE_ZH_EN:unavailable", state.failed_operations)

    def test_reranker_receives_translated_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "knowledge graph completion"}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            reranker = RecordingReranker()
            results, _state = HybridPaperSearch(index, reranker=reranker).search(
                "知识图谱补全",
                query_variants={"translated_query": "knowledge graph completion"},
                top_k=1,
            )
            self.assertEqual([row.paper_id for row in results], ["p"])
            self.assertEqual(reranker.queries, ["knowledge graph completion"])

    def test_rrf_prefers_cross_source_candidate(self):
        ordered, scores, components = reciprocal_rank_fusion(
            {
                "bm25": [CandidateHit("a", 8.0, 1), CandidateHit("b", 7.0, 2)],
                "dense": [CandidateHit("b", 0.9, 1), CandidateHit("c", 0.8, 2)],
            },
            rank_constant=60,
        )
        self.assertEqual(ordered[0], "b")
        self.assertGreater(scores["b"], scores["a"])
        self.assertIn("dense_raw", components["b"])
        self.assertIn("bm25_raw", components["b"])

    def test_hybrid_filter_rerank_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "a", "title": "graph completion", "abstract": "knowledge graph", "conference": "AAAI", "year": 2024, "authors": ["Alice"], "keywords": ["graph"], "subjects": []},
                        {"paper_id": "b", "title": "graph ranking", "abstract": "ranking", "conference": "AAAI", "year": 2024, "authors": ["Bob"], "keywords": ["rank"], "subjects": []},
                        {"paper_id": "c", "title": "vision", "abstract": "images", "conference": "CVPR", "year": 2023, "authors": ["Carol"], "keywords": ["vision"], "subjects": []},
                    )
                ) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            graph = RecordingGraphFilter({"a", "b"})
            engine = HybridPaperSearch(
                index,
                dense_retriever=StaticRunDenseRetriever(
                    {"graph": [("b", 0.9), ("c", 0.8)]}
                ),
                reranker=FixedReranker(),
                graph_filter=graph,
                recall_k=10,
                rerank_k=10,
            )
            results, state = engine.search(
                "graph",
                filters=SearchFilters(conference=["AAAI"]),
                top_k=2,
            )
            self.assertEqual({result.paper_id for result in results}, {"a", "b"})
            self.assertEqual(results[0].paper_id, "a")
            self.assertIn("rrf", results[0].source_scores)
            self.assertIn("reranker", results[0].source_scores)
            self.assertEqual(
                state.executed_operations,
                [
                    "RECALL_PAPER_BM25",
                    "RECALL_PAPER_STAGE0",
                    "FILTER_PAPER_NEO4J",
                    "RERANK_PAPER_SKILLROUTER",
                ],
            )
            self.assertEqual(len(graph.calls), 1)


if __name__ == "__main__":
    unittest.main()
