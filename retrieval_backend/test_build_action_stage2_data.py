import json
import tempfile
import unittest
from pathlib import Path

from multistep_search import ACTION_SKILLS
from build_action_stage2_data import (
    _branch_pair,
    _canonical_event,
    _causal_state,
    _current_state,
    _state_components,
    build,
    read_jsonl,
)
from multistep_search import (
    EXPAND_AUTHORS,
    EXPAND_KEYWORDS,
    EXPAND_SUBJECTS,
    SEARCH_PAPERS,
    STOP,
)
from paper_search import PaperSearchIndex


def _catalog_payload():
    skill_ids = [str(row["skill_id"]) for row in ACTION_SKILLS]
    import hashlib

    digest = hashlib.sha256(
        json.dumps(skill_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "inventory_catalog_id": "test-action-pool",
        "inventory_catalog_digest": digest,
        "inventory_pool_size": len(skill_ids),
        "inventory_source": "test",
        "runtime_visible_skill_ids": skill_ids,
    }


class ActionStage2DataTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        docs = root / "docs.jsonl"
        docs.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "paper_id": "seed",
                        "title": "unique graph retrieval",
                        "abstract": "knowledge graph search",
                        "authors": ["Alice"],
                        "keywords": ["graph"],
                        "subjects": ["retrieval"],
                        "conference": "AAAI",
                        "year": 2024,
                    },
                    {
                        "paper_id": "author",
                        "title": "author graph work",
                        "abstract": "related retrieval",
                        "authors": ["Alice"],
                        "keywords": ["vision"],
                        "subjects": ["vision"],
                        "conference": "CVPR",
                        "year": 2025,
                    },
                    {
                        "paper_id": "keyword",
                        "title": "graph keyword work",
                        "abstract": "related retrieval",
                        "authors": ["Bob"],
                        "keywords": ["graph"],
                        "subjects": ["retrieval"],
                        "conference": "AAAI",
                        "year": 2023,
                    },
                    {
                        "paper_id": "subject",
                        "title": "retrieval subject work",
                        "abstract": "related retrieval",
                        "authors": ["Carol"],
                        "keywords": ["language"],
                        "subjects": ["retrieval"],
                        "conference": "EMNLP",
                        "year": 2024,
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        db = root / "papers.db"
        index = PaperSearchIndex(db)
        index.build(docs)
        stage0 = root / "stage0"
        stage0.mkdir()
        (stage0 / "skills.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in ACTION_SKILLS) + "\n",
            encoding="utf-8",
        )
        (stage0 / "inventory_catalogs.jsonl").write_text(
            json.dumps(_catalog_payload(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (stage0 / "data_contract.json").write_text("{}\n", encoding="utf-8")
        queries = root / "queries.jsonl"
        queries.write_text(
            "\n".join(
                json.dumps({"query_id": f"q{i}", "query_text": f"unique graph {i}"})
                for i in range(12)
            )
            + "\n",
            encoding="utf-8",
        )
        return db, stage0, queries

    def test_state_and_event_serialization_are_deterministic(self):
        components = _state_components("goal", 0, 4)
        current = _current_state(components)
        self.assertEqual(current, _current_state(dict(components)))
        event = {
            "step_index": 0,
            "target_skill_id": SEARCH_PAPERS,
            "action_text": "SEARCH_PAPERS(query='goal')",
            "actual_result_text": "{\"status\": \"ok\"}",
            "actual_result_executed": True,
        }
        canonical = _canonical_event(event)
        self.assertTrue(canonical["result_executed"])
        self.assertIn("causal_prefix:", _causal_state(current, [canonical]))

    def test_branch_pair_has_reproducible_completion_contrast(self):
        def row(trajectory, step, target, action):
            return {
                "trajectory_id": trajectory,
                "step_index": step,
                "target_skill_id": target,
                "action_text": action,
                "actual_result_text": "result-" + action,
            }

        a = [
            row("a", 0, SEARCH_PAPERS, "search"),
            row("a", 1, EXPAND_AUTHORS + "@000", "author"),
            row("a", 2, EXPAND_KEYWORDS + "@000", "keyword"),
        ]
        b = [
            row("b", 0, SEARCH_PAPERS, "search"),
            row("b", 1, EXPAND_KEYWORDS + "@000", "keyword"),
            row("b", 2, EXPAND_AUTHORS + "@000", "author"),
        ]
        pair = _branch_pair(a, b, pair_id="pair/0")
        self.assertEqual(pair["first_history_divergence_index"], 1)
        self.assertNotEqual(pair["history_a_digest"], pair["history_b_digest"])
        self.assertEqual(pair["causal_verification"]["shared_prefix_event_count"], 1)

    def test_build_emits_contiguous_ordinary_and_pair_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, stage0, queries = self._write_fixture(root)
            output = root / "out"
            report = build(
                db_path=db,
                query_source=queries,
                stage0_data_dir=stage0,
                output_dir=output,
                train_queries=1,
                dev_queries=1,
                train_pair_queries=1,
                dev_pair_queries=1,
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["counts"]["ordinary_train_trajectories"], 1)
            rows = list(read_jsonl(output / "trajectory_train.jsonl"))
            self.assertEqual([row["step_index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(rows[0]["capabilities"]["ordered_next_tool"], False)
            self.assertTrue(all(row["capabilities"]["ordered_next_tool"] for row in rows[1:]))
            pair_rows = list(read_jsonl(output / "causal_pair_support_train.jsonl"))
            self.assertTrue(pair_rows)
            self.assertTrue(all(row["pair_support_only"] for row in pair_rows))
            self.assertTrue(
                all(row["target_skill_id"] in _catalog_payload()["runtime_visible_skill_ids"] for row in rows + pair_rows)
            )


if __name__ == "__main__":
    unittest.main()
