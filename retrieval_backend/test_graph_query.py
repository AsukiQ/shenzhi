import json
import tempfile
import unittest
from pathlib import Path

from graph_query import infer_graph_query_plan
from paper_search import PaperSearchIndex, SearchFilters
from retrieval_pipeline import HybridPaperSearch


class FakeGraph:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.expansion_calls = []

    def filter_candidate_ids(self, candidate_ids, *, filters):
        if self.fail:
            raise RuntimeError("neo4j unavailable")
        return list(candidate_ids)

    def expand_from_papers(
        self,
        seed_ids,
        *,
        relation,
        value_limit,
        paper_limit,
        slot_index=None,
    ):
        if self.fail:
            raise RuntimeError("neo4j unavailable")
        self.expansion_calls.append((list(seed_ids), relation, value_limit, paper_limit, slot_index))
        if relation == "authors":
            return [("Alice", ["related"])], ["Alice"]
        return [], []


class GraphQueryTest(unittest.TestCase):
    def test_planner_only_triggers_on_relation_language_or_filters(self):
        self.assertFalse(infer_graph_query_plan("knowledge graph completion").enabled)
        authors = infer_graph_query_plan("找同作者的其他论文")
        self.assertEqual(authors.expansion_relation, "authors")
        self.assertEqual(authors.trigger, "relation_language")
        filtered = infer_graph_query_plan(
            "graph papers", SearchFilters(conference=["AAAI"])
        )
        self.assertTrue(filtered.filter_requested)
        self.assertEqual(filtered.expansion_relation, None)

    def test_single_step_uses_graph_expansion_only_for_relation_query(self):
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
                            "abstract": "graph retrieval",
                            "authors": ["Alice"],
                        },
                        {
                            "paper_id": "related",
                            "title": "alice related work",
                            "abstract": "different wording",
                            "authors": ["Alice"],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            graph = FakeGraph()
            engine = HybridPaperSearch(index, graph_filter=graph, recall_k=10)

            ordinary, ordinary_state = engine.search("unique graph", top_k=2)
            self.assertEqual([row.paper_id for row in ordinary], ["seed"])
            self.assertEqual(graph.expansion_calls, [])
            self.assertNotIn("EXPAND_PAPER_AUTHORS_NEO4J", ordinary_state.executed_operations)

            related, related_state = engine.search("unique graph 找同作者的其他论文", top_k=2)
            self.assertEqual({row.paper_id for row in related}, {"seed", "related"})
            self.assertTrue(any(call[1] == "authors" for call in graph.expansion_calls))
            self.assertIn("EXPAND_PAPER_AUTHORS_NEO4J", related_state.executed_operations)
            self.assertTrue(any("graph_authors" in row.retrieval_mode for row in related))

    def test_graph_failure_falls_back_to_local_relation_tables(self):
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
                            "abstract": "graph retrieval",
                            "authors": ["Alice"],
                        },
                        {
                            "paper_id": "related",
                            "title": "alice related work",
                            "abstract": "different wording",
                            "authors": ["Alice"],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            engine = HybridPaperSearch(index, graph_filter=FakeGraph(fail=True), recall_k=10)
            results, state = engine.search("unique graph 找同作者的其他论文", top_k=2)
            self.assertEqual({row.paper_id for row in results}, {"seed", "related"})
            self.assertIn("EXPAND_PAPER_AUTHORS_SQLITE_FALLBACK", state.executed_operations)
            self.assertTrue(any(item.startswith("EXPAND_PAPER_AUTHORS_NEO4J:") for item in state.failed_operations))

    def test_graph_intent_feature_flag_preserves_text_only_baseline(self):
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
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            graph = FakeGraph()
            engine = HybridPaperSearch(
                index,
                graph_filter=graph,
                graph_intent_enabled=False,
                recall_k=10,
            )
            results, state = engine.search(
                "unique graph 找同作者的其他论文", top_k=2
            )
            self.assertEqual([row.paper_id for row in results], ["seed"])
            self.assertEqual(graph.expansion_calls, [])
            self.assertFalse(
                any(operation.startswith("EXPAND_PAPER") for operation in state.executed_operations)
            )


if __name__ == "__main__":
    unittest.main()
