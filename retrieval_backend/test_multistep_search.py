import json
import tempfile
import unittest
from pathlib import Path

from multistep_search import (
    EXPAND_AUTHORS,
    EXPAND_KEYWORDS,
    SEARCH_PAPERS,
    STOP,
    MultiStepPaperSearch,
    PlannedActionPolicy,
    action_skill_id,
)
from paper_search import PaperSearchIndex
from retrieval_pipeline import HybridPaperSearch
from neo4j_filter import FILTER_CYPHER


class MultiStepSearchTest(unittest.TestCase):
    def test_graph_expansion_adapter_is_used_when_configured(self):
        class FakeGraph:
            def __init__(self):
                self.calls = []

            def expand_from_papers(self, seed_ids, *, relation, value_limit, paper_limit, slot_index=None):
                self.calls.append((list(seed_ids), relation, value_limit, paper_limit, slot_index))
                return [("Alice", ["author-paper"])], ["Alice"]

            def filter_candidate_ids(self, candidate_ids, *, filters):
                return list(candidate_ids)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "seed", "title": "unique", "abstract": "unique", "authors": ["Alice"]}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            graph = FakeGraph()
            engine = HybridPaperSearch(index, graph_filter=graph)
            executor = MultiStepPaperSearch(engine, candidate_limit=10)
            _results, state = executor.run(
                "unique",
                policy=PlannedActionPolicy([SEARCH_PAPERS, action_skill_id(EXPAND_AUTHORS, 0), STOP]),
                max_steps=3,
            )
            self.assertTrue(any(call[1] == "authors" for call in graph.calls))
            self.assertIn("author-paper", state.events[1].result_ids)
    def test_executes_replayable_search_expand_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "seed", "title": "uniqueterm knowledge graph completion", "abstract": "graph completion", "conference": "AAAI", "year": 2024, "authors": ["Alice"], "keywords": ["graph"], "subjects": ["Data Mining"]},
                        {"paper_id": "author", "title": "other work", "abstract": "representation", "conference": "CVPR", "year": 2025, "authors": ["Alice"], "keywords": ["vision"], "subjects": ["Vision"]},
                        {"paper_id": "keyword", "title": "graph reasoning", "abstract": "reasoning", "conference": "IJCAI", "year": 2023, "authors": ["Bob"], "keywords": ["graph"], "subjects": ["Data Mining"]},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            executor = MultiStepPaperSearch(
                HybridPaperSearch(index, recall_k=10),
                expansion_seed_k=2,
                candidate_limit=10,
            )
            results, state = executor.run(
                "uniqueterm",
                policy=PlannedActionPolicy(
                    [
                        SEARCH_PAPERS,
                        action_skill_id(EXPAND_AUTHORS, 0),
                        action_skill_id(EXPAND_KEYWORDS, 1),
                        STOP,
                    ]
                ),
                top_k=10,
                max_steps=6,
            )
            self.assertTrue(state.stopped)
            self.assertEqual(state.executed_actions, [
                SEARCH_PAPERS,
                EXPAND_AUTHORS,
                EXPAND_KEYWORDS,
                STOP,
            ])
            self.assertEqual(state.events[1].bound_values, ["Alice"])
            self.assertIn("author", state.events[1].new_result_ids)
            self.assertIn("graph", state.events[2].bound_values)
            self.assertIn("keyword", [row.paper_id for row in results])
            self.assertTrue(all(event.result_text for event in state.events))

    def test_rule_policy_stops_after_initial_search_without_requested_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "graph", "abstract": "graph", "authors": ["Alice"], "keywords": ["graph"], "subjects": []}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            executor = MultiStepPaperSearch(HybridPaperSearch(index, recall_k=10))
            _results, state = executor.run("graph", max_steps=4)
            self.assertEqual(state.executed_actions, [SEARCH_PAPERS, STOP])

    def test_rule_policy_preserves_explicit_relation_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "seed", "title": "unique graph", "abstract": "graph", "authors": ["Alice"], "keywords": ["graph"], "subjects": ["ML"]},
                        {"paper_id": "author", "title": "author work", "abstract": "work", "authors": ["Alice"], "keywords": ["vision"], "subjects": ["Vision"]},
                        {"paper_id": "subject", "title": "subject work", "abstract": "work", "authors": ["Bob"], "keywords": ["language"], "subjects": ["ML"]},
                    )
                ) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            executor = MultiStepPaperSearch(HybridPaperSearch(index, recall_k=10), candidate_limit=10)
            _results, state = executor.run(
                "unique graph same author and related field",
                max_steps=4,
            )
            self.assertEqual(state.relation_plan, ["authors", "subjects"])
            self.assertEqual(state.executed_actions[:3], [SEARCH_PAPERS, EXPAND_AUTHORS, "EXPAND_SUBJECTS"])

    def test_multistep_graph_failure_falls_back_without_stopping_episode(self):
        class FailingGraph:
            def expand_from_papers(self, *_args, **_kwargs):
                raise RuntimeError("neo4j unavailable")

            def filter_candidate_ids(self, *_args, **_kwargs):
                raise RuntimeError("neo4j unavailable")

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
            executor = MultiStepPaperSearch(
                HybridPaperSearch(index, graph_filter=FailingGraph(), recall_k=10),
                candidate_limit=10,
            )
            results, state = executor.run(
                "unique graph",
                policy=PlannedActionPolicy(
                    [SEARCH_PAPERS, action_skill_id(EXPAND_AUTHORS, 0), STOP]
                ),
                max_steps=3,
                top_k=10,
            )
            self.assertTrue(state.stopped)
            self.assertEqual(state.stop_reason, "max_steps")
            self.assertIn("related", [row.paper_id for row in results])
            self.assertFalse(any(event.status == "error" for event in state.events))
            self.assertTrue(
                any(item.startswith("expand_authors_neo4j:") for item in state.failures)
            )
            self.assertTrue(
                any(item.startswith("graph_filter:") for item in state.failures)
            )


if __name__ == "__main__":
    unittest.main()
