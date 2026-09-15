from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from shenzhi_paper_data.build_rich_graph import build_rich_graph
from shenzhi_paper_data.validate_rich_graph import validate_rich_graph


def _graph_payload(*, cited_title: str = "Referenced work") -> dict:
    nodes = [
        {"id": "paper:2601.00001v1", "label": "Paper", "properties": {
            "arxiv_id": "2601.00001v1", "title": "A rich graph paper",
            "abstract": "A useful abstract.", "authors": ["Alice", "Bob"],
            "keywords": ["graphs"], "datasets": ["D1"], "external": False,
        }},
        {"id": "paper:ref", "label": "Paper", "properties": {
            "title": cited_title, "external": True, "year": 2024,
        }},
        {"id": "author:alice", "label": "Author", "properties": {"name": "Alice", "position": 1}},
        {"id": "institution:u", "label": "Institution", "properties": {"name": "University U"}},
        {"id": "method:m", "label": "Method", "properties": {"name": "Method M", "content": "Details"}},
        {"id": "method:b", "label": "Method", "properties": {"name": "Baseline B"}},
        {"id": "funding:f", "label": "Funding", "properties": {"name": "Grant F"}},
        {"id": "topic:t", "label": "Topic", "properties": {"name": "Graph retrieval"}},
        {"id": "conference:test_2026", "label": "Conference", "properties": {
            "name": "TEST 2026", "venue": "TEST", "year": 2026,
        }},
        {"id": "venue:test", "label": "Venue", "properties": {"name": "TEST"}},
    ]
    edges = [
        {"source": "paper:2601.00001v1", "target": "author:alice", "type": "AUTHORED_BY", "properties": {"position": 1}},
        {"source": "author:alice", "target": "institution:u", "type": "AFFILIATED_WITH", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "method:m", "type": "PROPOSES", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "method:b", "type": "USES_AS_BASELINE", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "funding:f", "type": "FUNDED_BY", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "topic:t", "type": "HAS_TOPIC", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "conference:test_2026", "type": "PUBLISHED_IN", "properties": {}},
        {"source": "conference:test_2026", "target": "venue:test", "type": "PART_OF", "properties": {}},
        {"source": "paper:2601.00001v1", "target": "paper:ref", "type": "CITES", "properties": {"confidence": 0.9}},
    ]
    return {"graph": {"nodes": nodes, "edges": edges}, "audit": {}}


class RichGraphBuilderTests(unittest.TestCase):
    def test_builds_all_labels_and_edges_and_reconstructs_authors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "canonical_TEST_2026.zip"
            second = root / "canonical_TEST2_2026.zip"
            with zipfile.ZipFile(first, "w") as archive:
                archive.writestr("test/a.json", json.dumps(_graph_payload()))
            # Re-exporting the same graph exercises cross-archive canonical
            # node and relationship deduplication.
            with zipfile.ZipFile(second, "w") as archive:
                archive.writestr("test2/b.json", json.dumps(_graph_payload()))
            output = root / "rich"
            report = build_rich_graph([second, first], output)

            self.assertTrue(report["quality_gates"]["neo4j_import_candidate_ready"])
            self.assertEqual(report["counts"]["input_archives"], 2)
            self.assertEqual(report["nodes_by_label"]["Paper"], 2)
            self.assertEqual(report["nodes_by_label"]["Author"], 2)
            self.assertEqual(report["counts"]["derived_author_nodes"], 1)
            self.assertEqual(report["edges_by_type"]["AUTHORED_BY"], 2)
            self.assertEqual(report["counts"]["derived_authored_by_edges"], 1)
            self.assertEqual(report["counts"]["papers_enriched_from_conference"], 1)

            with (output / "paper_kg" / "papers.csv").open(encoding="utf-8", newline="") as handle:
                papers = list(csv.DictReader(handle))
            main = next(row for row in papers if row["paper_id:ID(Paper)"] == "paper:2601.00001v1")
            self.assertEqual(main["year:int"], "2026")
            self.assertEqual(main["external:boolean"], "false")
            self.assertEqual(json.loads(main["datasets_json"]), ["D1"])

            docs = [json.loads(line) for line in (output / "retrieval_documents.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(docs), 1)
            self.assertIn("Method M", docs[0]["search_text"])
            self.assertIn("Graph retrieval", docs[0]["search_text"])
            self.assertTrue((output / "constraints.cypher").is_file())
            self.assertTrue((output / "import_neo4j_admin.sh").is_file())
            import_script = (output / "import_neo4j_admin.sh").read_text(encoding="utf-8")
            self.assertIn("--nodes=paper_kg/papers.csv", import_script)
            self.assertIn("--relationships=paper_kg/cites.csv", import_script)
            self.assertIn("--multiline-fields=true", import_script)
            validation = validate_rich_graph(output)
            self.assertTrue(validation["ready"])
            self.assertEqual(validation["checks"]["dangling_edges"], 0)
            with self.assertRaises(FileExistsError):
                build_rich_graph([first], output)


if __name__ == "__main__":
    unittest.main()
