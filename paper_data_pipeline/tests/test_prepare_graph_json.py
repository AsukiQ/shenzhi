from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from shenzhi_paper_data.prepare_graph_json import (
    normalize_arxiv_id,
    prepare_graph_archives,
)


def _payload(arxiv: str, title: str, authors: list[str], *, version: str = "") -> dict:
    root_id = f"paper:{arxiv}{version}"
    author_nodes = [
        {
            "id": f"author:{name.lower().replace(' ', '_')}",
            "label": "Author",
            "properties": {"name": name, "position": index},
        }
        for index, name in enumerate(authors, 1)
    ]
    nodes = [
        {
            "id": root_id,
            "label": "Paper",
            "properties": {
                "arxiv_id": f"{arxiv}{version}",
                "title": title,
                "abstract": "A useful abstract for retrieval.",
                "authors": authors,
                "keywords": ["Graph retrieval"],
                "external": False,
            },
        },
        *author_nodes,
        {"id": "topic:graph_retrieval", "label": "Topic", "properties": {"name": "Graph retrieval"}},
        {"id": "venue:test", "label": "Venue", "properties": {"name": "TEST"}},
        {"id": "conference:test_2026", "label": "Conference", "properties": {"name": "TEST 2026", "venue": "TEST", "year": 2026}},
    ]
    edges = [
        {"source": root_id, "target": node["id"], "type": "AUTHORED_BY", "properties": {"position": index}}
        for index, node in enumerate(author_nodes, 1)
    ]
    edges.extend(
        [
            {"source": root_id, "target": "topic:graph_retrieval", "type": "HAS_TOPIC", "properties": {}},
            {"source": root_id, "target": "conference:test_2026", "type": "PUBLISHED_IN", "properties": {}},
            {"source": "conference:test_2026", "target": "venue:test", "type": "PART_OF", "properties": {}},
        ]
    )
    return {"graph": {"nodes": nodes, "edges": edges}, "audit": {}}


class GraphJsonAdapterTests(unittest.TestCase):
    def test_normalize_arxiv_flags_short_2026_id(self) -> None:
        self.assertEqual(normalize_arxiv_id("2607.2912")[2], "suspect_missing_leading_zero")
        self.assertEqual(normalize_arxiv_id("2607.02912v1")[1], "2607.02912")

    def test_dry_run_builds_a_layer_and_keeps_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "TEST2026_graph.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("TEST2026_graph/2601.00001v1.json", json.dumps(_payload("2601.00001", "Same paper", ["Alice", "Bob"], version="v1")))
                archive.writestr("TEST2026_graph/2601.00001v2.json", json.dumps(_payload("2601.00001", "Same paper", ["Alice", "Bob"], version="v2")))
            output = root / "out"
            report = prepare_graph_archives([archive_path], output)
            self.assertEqual(report["counts"]["source_json_records"], 2)
            self.assertEqual(report["counts"]["canonical_papers"], 1)
            self.assertEqual(report["counts"]["version_records_merged"], 1)
            self.assertTrue((output / "staging" / "citations.jsonl").is_file())
            with (output / "graph_import" / "paper_kg" / "authors.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 3)
            self.assertTrue((output / "retrieval_documents.jsonl").read_text(encoding="utf-8").strip())
            with self.assertRaises(FileExistsError):
                prepare_graph_archives([archive_path], output)


if __name__ == "__main__":
    unittest.main()
