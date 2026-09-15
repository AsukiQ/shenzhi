from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from shenzhi_paper_data.merge_neo4j_import import merge_import_bundle
from shenzhi_paper_data.align_neo4j_addition import align_addition


NODE_HEADERS = {
    "papers": ["paper_id:ID(Paper)", "title"],
    "authors": ["author_id:ID(Author)", "name"],
    "venues": ["venue_id:ID(Venue)", "name"],
    "keywords": ["keyword_id:ID(Keyword)", "name"],
    "subjects": ["subject_id:ID(Subject)", "name"],
    "years": ["year_id:ID(Year)", "year:int"],
}
REL_HEADERS = {
    "authored_by": [":START_ID(Paper)", ":END_ID(Author)", ":TYPE"],
    "published_in": [":START_ID(Paper)", ":END_ID(Venue)", ":TYPE"],
    "published_year": [":START_ID(Paper)", ":END_ID(Year)", ":TYPE"],
    "has_keyword": [":START_ID(Paper)", ":END_ID(Keyword)", ":TYPE"],
    "has_subject": [":START_ID(Paper)", ":END_ID(Subject)", ":TYPE"],
    "cites": [":START_ID(Paper)", ":END_ID(Paper)", "source", ":TYPE"],
}


def _csv_text(header: list[str], rows: list[list[str]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


class MergeImportTests(unittest.TestCase):
    def test_additive_merge_preserves_base_and_rejects_dangling_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_zip = root / "base.zip"
            addition = root / "addition" / "paper_kg"
            addition.mkdir(parents=True)
            base_nodes = {
                "papers": [["paper:old", "Old"]],
                "authors": [["author:old", "Alice"]],
                "venues": [["venue:old", "Old Venue"]],
                "keywords": [["keyword:old", "old"]],
                "subjects": [],
                "years": [["year:2025", "2025"]],
            }
            base_relations = {
                "authored_by": [["paper:old", "author:old", "AUTHORED_BY"]],
                "published_in": [["paper:old", "venue:old", "PUBLISHED_IN"]],
                "published_year": [["paper:old", "year:2025", "PUBLISHED_YEAR"]],
                "has_keyword": [["paper:old", "keyword:old", "HAS_KEYWORD"]],
                "has_subject": [],
                "cites": [],
            }
            with zipfile.ZipFile(base_zip, "w") as archive:
                archive.writestr("root/paper_kg_neo4j_import/constraints.cypher", "// constraints\n")
                for stem, rows in base_nodes.items():
                    archive.writestr(
                        f"root/paper_kg_neo4j_import/paper_kg/{stem}.csv",
                        _csv_text(NODE_HEADERS[stem], rows),
                    )
                for stem, rows in base_relations.items():
                    archive.writestr(
                        f"root/paper_kg_neo4j_import/paper_kg/{stem}.csv",
                        _csv_text(REL_HEADERS[stem], rows),
                    )
            addition_nodes = {
                "papers": [["paper:new", "New"]],
                "authors": [["author:new", "Bob"]],
                "venues": [["venue:new", "New Venue"]],
                "keywords": [["keyword:new", "new"]],
                "subjects": [],
                "years": [["year:2026", "2026"]],
            }
            addition_relations = {
                "authored_by": [["paper:new", "author:new", "AUTHORED_BY"]],
                "published_in": [["paper:new", "venue:new", "PUBLISHED_IN"]],
                "published_year": [["paper:new", "year:2026", "PUBLISHED_YEAR"]],
                "has_keyword": [["paper:new", "keyword:new", "HAS_KEYWORD"]],
                "has_subject": [],
                "cites": [],
            }
            for stem, rows in addition_nodes.items():
                (addition / f"{stem}.csv").write_text(_csv_text(NODE_HEADERS[stem], rows), encoding="utf-8")
            for stem, rows in addition_relations.items():
                (addition / f"{stem}.csv").write_text(_csv_text(REL_HEADERS[stem], rows), encoding="utf-8")
            output_dir = root / "merged"
            output_zip = root / "merged.zip"
            report = merge_import_bundle(base_zip, addition, output_dir, output_zip=output_zip)
            self.assertEqual(report["quality_gates"]["conflicting_duplicate_rows"], 0)
            self.assertEqual(report["quality_gates"]["dangling_endpoint_count"], 0)
            with zipfile.ZipFile(output_zip) as archive:
                names = archive.namelist()
                self.assertTrue(all(".merge.sqlite" not in name for name in names))
                papers = archive.read(
                    "root/paper_kg_neo4j_import_merged/paper_kg/papers.csv"
                ).decode()
            self.assertIn("paper:old,Old", papers)
            self.assertIn("paper:new,New", papers)

    def test_alignment_quarantines_title_and_short_arxiv_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_zip = root / "base.zip"
            addition = root / "addition"
            addition.mkdir()
            with zipfile.ZipFile(base_zip, "w") as archive:
                archive.writestr(
                    "root/paper_kg_neo4j_import/paper_kg/papers.csv",
                    _csv_text(
                        ["paper_id:ID(Paper)", "title", "doi", "arxiv_id"],
                        [["paper:old", "Existing", "", ""]],
                    ),
                )
                for stem, header in NODE_HEADERS.items():
                    if stem == "papers":
                        continue
                    archive.writestr(
                        f"root/paper_kg_neo4j_import/paper_kg/{stem}.csv",
                        _csv_text(header, []),
                    )
            for stem, header in NODE_HEADERS.items():
                if stem == "papers":
                    rows = [
                        ["paper:title", "Existing", "", ""],
                        ["paper:short", "Short", "", "2607.1234"],
                    ]
                else:
                    rows = []
                (addition / f"{stem}.csv").write_text(_csv_text(header if stem != "papers" else ["paper_id:ID(Paper)", "title", "doi", "arxiv_id"], rows), encoding="utf-8")
            for stem, header in REL_HEADERS.items():
                (addition / f"{stem}.csv").write_text(_csv_text(header, []), encoding="utf-8")
            output = root / "aligned"
            report = align_addition(base_zip, addition, output)
            self.assertEqual(len(report["quarantined_papers"]), 2)
            reasons = {item["reason"] for item in report["quarantined_papers"].values()}
            self.assertEqual(reasons, {"title_only_match", "suspect_arxiv_format"})


if __name__ == "__main__":
    unittest.main()
