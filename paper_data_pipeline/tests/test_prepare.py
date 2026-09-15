from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from shenzhi_paper_data.prepare import (
    build_search_text,
    collapse_repeated_sequence,
    generated_paper_id,
    is_placeholder_abstract,
    merge_records,
    prepare_dataset,
)


class MergeTests(unittest.TestCase):
    def test_merge_prefers_real_longer_abstract_and_unions_keywords(self) -> None:
        records = [
            {
                "arxiv_id": "x@source",
                "paper_id": "Example",
                "title": "Example",
                "abstract": "No summary was provided.",
                "authors": ["A"],
                "keywords": ["Graph"],
                "subjects": ["Track A"],
                "conference": "AAAI",
                "year": 2025,
                "doi": "",
                "source_file": "a.json",
            },
            {
                "arxiv_id": "x@source",
                "paper_id": "Example",
                "title": "Example",
                "abstract": "A complete abstract for retrieval.",
                "authors": ["A", "B"],
                "keywords": ["graph", "retrieval"],
                "subjects": ["Track B"],
                "conference": "AAAI",
                "year": 2025,
                "doi": "10.1/example",
                "source_file": "b.json",
            },
        ]
        merged, conflicts = merge_records(records)
        self.assertEqual(merged["abstract"], "A complete abstract for retrieval.")
        self.assertEqual(merged["authors"], ["A", "B"])
        self.assertEqual(merged["keywords"], ["Graph", "retrieval"])
        self.assertEqual(merged["subjects"], ["Track A", "Track B"])
        self.assertEqual(merged["source_record_count"], 2)
        self.assertEqual(merged["source_files"], ["a.json", "b.json"])
        self.assertIn("abstract", conflicts)

    def test_scalar_mode_keeps_the_actual_most_common_value(self) -> None:
        merged, _ = merge_records(
            [
                {
                    "arxiv_id": "x",
                    "title": "T",
                    "abstract": "A",
                    "authors": [],
                    "keywords": [],
                    "subjects": [],
                    "conference": "AAAI",
                    "year": 2025,
                    "doi": "",
                },
                {
                    "arxiv_id": "x",
                    "title": "T",
                    "abstract": "A",
                    "authors": [],
                    "keywords": [],
                    "subjects": [],
                    "conference": "AAAI",
                    "year": 2025,
                    "doi": "10/example",
                },
                {
                    "arxiv_id": "x",
                    "title": "T",
                    "abstract": "A",
                    "authors": [],
                    "keywords": [],
                    "subjects": [],
                    "conference": "ACL",
                    "year": 2025,
                    "doi": "",
                },
            ]
        )
        self.assertEqual(merged["conference"], "AAAI")
        self.assertEqual(merged["doi"], "10/example")

    def test_generated_id_is_stable_and_source_specific(self) -> None:
        first = generated_paper_id("Graph Retrieval!", "source:1")
        self.assertEqual(first, generated_paper_id("Renamed Paper", "source:1"))
        self.assertNotEqual(first, generated_paper_id("Graph Retrieval!", "source:2"))
        self.assertTrue(first.startswith("paper:new:source_1:"))

    def test_collapse_only_whole_repeated_author_sequence(self) -> None:
        self.assertEqual(
            collapse_repeated_sequence(["A", "B", "A", "B"]),
            ["A", "B"],
        )
        self.assertEqual(
            collapse_repeated_sequence(["A", "B", "A"]),
            ["A", "B", "A"],
        )

    def test_placeholder_detection(self) -> None:
        self.assertTrue(is_placeholder_abstract(" No summary was provided. "))
        self.assertFalse(is_placeholder_abstract("An actual abstract."))

    def test_search_text_has_expected_sections(self) -> None:
        text = build_search_text(
            {
                "title": "T",
                "abstract": "A",
                "authors": ["Alice"],
                "keywords": ["KG"],
                "subjects": ["Main"],
                "conference": "AAAI",
                "year": 2025,
            }
        )
        self.assertIn("[TITLE] T", text)
        self.assertIn("[AUTHORS] Alice", text)
        self.assertIn("[CONFERENCE] AAAI", text)


class EndToEndTests(unittest.TestCase):
    def _write_zip(self, path: Path) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=("paper_id:ID(Paper)", "title", "arxiv_id"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "paper_id:ID(Paper)": "paper:canonical:1",
                "title": "Existing",
                "arxiv_id": "existing@source",
            }
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("root/paper_kg/papers.csv", buffer.getvalue())

    def test_prepare_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            json_path = root / "papers.json"
            zip_path = root / "graph.zip"
            output = root / "out"
            rows = [
                {
                    "paper_id": "Existing",
                    "arxiv_id": "existing@source",
                    "title": "Existing",
                    "abstract": "No summary was provided.",
                    "authors": ["Alice"],
                    "conference": "AAAI",
                    "year": 2025,
                    "subjects": ["Main"],
                    "keywords": ["KG"],
                    "pdf_url": "x",
                    "source": "venue",
                    "source_file": "a.json",
                },
                {
                    "paper_id": "New",
                    "arxiv_id": "new@source",
                    "title": "New Paper",
                    "abstract": "Useful abstract.",
                    "authors": ["Bob"],
                    "conference": "ACL",
                    "year": 2024,
                    "subjects": ["Long"],
                    "keywords": ["NLP"],
                    "pdf_url": "y",
                    "source": "venue",
                    "source_file": "b.json",
                },
            ]
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            self._write_zip(zip_path)

            report = prepare_dataset(json_path, zip_path, output)
            self.assertEqual(report["counts"]["clean_papers"], 2)
            self.assertEqual(report["counts"]["canonical_ids_reused"], 1)
            self.assertEqual(report["counts"]["canonical_ids_generated"], 1)
            self.assertEqual(report["counts"]["retrieval_documents"], 1)

            papers = [
                json.loads(line)
                for line in (output / "papers_clean.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            by_source = {row["source_id"]: row for row in papers}
            self.assertEqual(by_source["existing@source"]["paper_id"], "paper:canonical:1")
            self.assertFalse(by_source["existing@source"]["retrieval_eligible"])
            self.assertTrue(by_source["new@source"]["paper_id"].startswith("paper:new:"))

            retrieval_lines = (output / "retrieval_documents.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(retrieval_lines), 1)

            with self.assertRaises(FileExistsError):
                prepare_dataset(json_path, zip_path, output)


if __name__ == "__main__":
    unittest.main()
