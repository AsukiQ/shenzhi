import json
import sys
import tempfile
import unittest
from pathlib import Path

from build_vnext_stage0_paper_data import build, file_sha256


CLSTR_SOURCE = Path(
    "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/"
    "clstr-qwen06-native-sync-smoke-source"
)


class VNextStage0PaperDataTest(unittest.TestCase):
    def test_build_passes_current_vnext_contract_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills.jsonl"
            retrieval = root / "retrieval.jsonl"
            skills.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "skill_id": candidate_id,
                            "name": title,
                            "description": title,
                            "body": title,
                        }
                    )
                    for candidate_id, title in (
                        ("paper:a", "Graph completion"),
                        ("paper:b", "Graph ranking"),
                        ("paper:c", "Vision model"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            retrieval.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "query_text": query,
                            "positive_skill_id": positive,
                            "variant": "fixture",
                            "split": split,
                            "provenance": {
                                "paper_id": positive,
                                "label_quality": "weak",
                            },
                        }
                    )
                    for query_id, query, positive, split in (
                        ("q1", "find graph completion", "paper:a", "train"),
                        ("q2", "find graph ranking", "paper:b", "dev"),
                        ("q3", "find vision models", "paper:c", "test"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            report = build(skills, retrieval, root / "out")
            self.assertEqual(report["counts"]["query_splits"], {"dev": 1, "test": 1, "train": 1})
            self.assertEqual(
                (root / "out" / "retrieval_train.jsonl").read_text(),
                (root / "out" / "static_route_train.jsonl").read_text(),
            )
            self.assertNotIn(
                "q3",
                (root / "out" / "retrieval_train.jsonl").read_text()
                + (root / "out" / "retrieval_dev.jsonl").read_text(),
            )

            if str(CLSTR_SOURCE) not in sys.path:
                sys.path.insert(0, str(CLSTR_SOURCE))
            from clstr.history_channel import audit_history_channel_rows
            from clstr.vnext_stage0_train import _filter_rows
            from clstr.vnext_training import (
                load_inventory_catalogs,
                require_verified_data_contract,
            )

            row = json.loads(
                (root / "out" / "retrieval_train.jsonl").read_text().splitlines()[0]
            )
            audit = audit_history_channel_rows(
                [row],
                require_explicit_current=True,
                require_structured_current=True,
                require_explicit_causal=True,
            )
            self.assertEqual(audit["status"], "ok")
            self.assertEqual(
                _filter_rows([row], {"paper:a", "paper:b", "paper:c"}, kind="retrieval")[0]["_vnext_positive_skill_ids"],
                ["paper:a"],
            )
            catalogs = load_inventory_catalogs(root / "out" / "inventory_catalogs.jsonl")
            self.assertEqual(len(catalogs), 1)
            verified = require_verified_data_contract(
                root / "out" / "data_contract.json",
                {
                    "training_skills": root / "out" / "skills.jsonl",
                    "retrieval_rows": root / "out" / "retrieval_train.jsonl",
                    "retrieval_dev_rows": root / "out" / "retrieval_dev.jsonl",
                    "static_route_rows": root / "out" / "static_route_train.jsonl",
                    "static_route_dev_rows": root / "out" / "static_route_dev.jsonl",
                    "inventory_catalogs": root / "out" / "inventory_catalogs.jsonl",
                },
            )
            self.assertEqual(verified["status"], "ok")

            first_hash = file_sha256(root / "out" / "data_contract.json")
            build(skills, retrieval, root / "out")
            self.assertEqual(first_hash, file_sha256(root / "out" / "data_contract.json"))


if __name__ == "__main__":
    unittest.main()
