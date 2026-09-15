from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from shenzhi_paper_data.clean_staging import clean_staging


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class CleanStagingTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        staging = root / "staging"
        staging.mkdir()
        _write_jsonl(
            staging / "methods.jsonl",
            [
                {
                    "paper_id": "p1",
                    "package": "TEST",
                    "items": [
                        {"name": "method", "content": "generic", "relation": "USES_METHOD"},
                        {"name": "Graph Neural Network", "content": "body", "relation": "USES_METHOD"},
                        {"name": "This is a sentence describing a method " * 5, "content": "body", "relation": "USES_METHOD"},
                    ],
                }
            ],
        )
        _write_jsonl(
            staging / "institutions.jsonl",
            [
                {
                    "paper_id": "p1",
                    "package": "TEST",
                    "items": [
                        {"name": "Department; University | Country\nResearch Lab", "author_node_id": "a1", "author_name": "Alice"},
                        {"name": "Department, University, Country", "author_node_id": "a1", "author_name": "Alice"},
                    ],
                }
            ],
        )
        _write_jsonl(
            staging / "funding.jsonl",
            [
                {
                    "paper_id": "p1",
                    "package": "TEST",
                    "items": [{"properties": {"name": "  National  Science Foundation  "}, "name": "ignored"}],
                }
            ],
        )
        _write_jsonl(
            staging / "citations.jsonl",
            [
                {
                    "paper_id": "p1",
                    "items": [
                        {"target_arxiv_id": "2601.00002v1", "target_title": "Target", "confidence": 0.9, "source": "exact"},
                        {"target_title": "Title only", "confidence": 0.99, "source": "title"},
                        {"target_arxiv_id": "9999.99999", "target_title": "Missing", "confidence": 0.99, "source": "id"},
                        {"target_arxiv_id": "2601.00002", "target_title": "Low confidence", "confidence": 0.5, "source": "weak"},
                    ],
                },
                {
                    "paper_id": "p-quarantined",
                    "items": [{"target_arxiv_id": "2601.00002", "target_title": "Target", "confidence": 0.99, "source": "source quarantined"}],
                },
            ],
        )

        papers = root / "papers.csv"
        with papers.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["paper_id:ID(Paper)", "arxiv_id", "doi"])
            writer.writerow(["p1", "2601.00001v1", ""])
            writer.writerow(["p2", "2601.00002v1", ""])

        alignment = root / "alignment_report.json"
        alignment.write_text(
            json.dumps({"quarantined_papers": {"p-quarantined": {"reason": "test"}}}),
            encoding="utf-8",
        )
        return staging, papers, alignment

    def test_cleaning_filters_noise_and_preserves_raw_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging, papers, alignment = self._fixture(Path(temporary))
            output = Path(temporary) / "clean"
            report = clean_staging(staging, papers, alignment, output)

            methods = [json.loads(line) for line in (output / "methods_clean.jsonl").read_text().splitlines()]
            self.assertEqual([item["name"] for item in methods], ["Graph Neural Network"])
            self.assertEqual(report["counts"]["methods_rejected"], 1)
            self.assertEqual(report["counts"]["methods_quarantine"], 1)
            self.assertEqual(methods[0]["raw"]["name"], "Graph Neural Network")

            institutions = [json.loads(line) for line in (output / "institutions_clean.jsonl").read_text().splitlines()]
            names = {item["name"] for item in institutions}
            self.assertEqual(names, {"Department", "University", "Country", "Research Lab", "Department, University, Country"})

            funding = [json.loads(line) for line in (output / "funding_clean.jsonl").read_text().splitlines()]
            self.assertEqual(funding[0]["name"], "National Science Foundation")
            self.assertEqual(funding[0]["raw"]["properties"]["name"], "  National  Science Foundation  ")

    def test_citation_quarantine_and_accepted_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging, papers, alignment = self._fixture(Path(temporary))
            output = Path(temporary) / "clean"
            report = clean_staging(staging, papers, alignment, output)

            accepted = [json.loads(line) for line in (output / "citations_clean.jsonl").read_text().splitlines()]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["source_paper_id"], "p1")
            self.assertEqual(accepted[0]["target_paper_id"], "p2")
            self.assertTrue(report["quality_gates"]["all_accepted_citations_have_source_and_target_paper"])

            quarantined = [json.loads(line) for line in (output / "citations_quarantine.jsonl").read_text().splitlines()]
            reasons = {item["reason"] for item in quarantined}
            self.assertIn("title_only_unresolved", reasons)
            self.assertIn("target_arxiv_not_in_merged_graph", reasons)
            self.assertIn("low_confidence", reasons)
            self.assertIn("source_paper_quarantined", reasons)

            with (output / "cites_import.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], ":START_ID(Paper)")


if __name__ == "__main__":
    unittest.main()
