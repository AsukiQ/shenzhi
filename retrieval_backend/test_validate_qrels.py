import json
from pathlib import Path
import tempfile
import unittest

from validate_qrels import validate


class ValidateQrelsTest(unittest.TestCase):
    def test_reports_unlabeled_and_can_require_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query_text": "graph",
                        "judgments": [
                            {"paper_id": "a", "relevance": 3},
                            {"paper_id": "b", "relevance": None},
                        ],
                    }
                )
                + "\n"
            )
            report = validate(path)
            self.assertEqual(report["annotation_status"], "awaiting_human_labels")
            self.assertEqual(report["unlabeled_count"], 1)
            with self.assertRaises(ValueError):
                validate(path, require_complete=True)


if __name__ == "__main__":
    unittest.main()
