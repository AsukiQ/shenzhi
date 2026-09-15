import json
import tempfile
import unittest
from pathlib import Path

from build_action_stage0_data import build


class ActionStage0DataTest(unittest.TestCase):
    def test_build_is_balanced_and_digest_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data"
            report = build(output, train_per_action=3, dev_per_action=2, seed=7)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["counts"]["skill_count"], 515)
            self.assertEqual(report["counts"]["retrieval_train_rows"], 1545)
            self.assertEqual(report["counts"]["retrieval_dev_rows"], 1030)
            rows = [json.loads(line) for line in (output / "retrieval_train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1545)
            self.assertTrue(all(row["zero_history_causal_equals_current"] for row in rows))
            self.assertEqual(
                len({row["future_label_skill_ids"][0] for row in rows}),
                515,
            )


if __name__ == "__main__":
    unittest.main()
