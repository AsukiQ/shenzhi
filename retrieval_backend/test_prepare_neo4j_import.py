import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from prepare_neo4j_import import NODE_FILES, RELATION_FILES, prepare_import


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


class PrepareNeo4jImportTest(unittest.TestCase):
    def test_deduplicates_relationships_and_keeps_last_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                for stem in NODE_FILES:
                    group = stem[:-1].title() or "Node"
                    archive.writestr(
                        f"root/pkg/paper_kg/{stem}.csv",
                        csv_bytes([[f"id:ID({group})", ":LABEL"], [f"{stem}:1", group]]),
                    )
                for stem in RELATION_FILES:
                    archive.writestr(
                        f"root/pkg/paper_kg/{stem}.csv",
                        csv_bytes(
                            [
                                [":START_ID(Paper)", ":END_ID(Paper)", "value", ":TYPE"],
                                ["p1", "p2", "first", "REL"],
                                ["p1", "p2", "last", "REL"],
                            ]
                        ),
                    )
                for filename in (
                    "constraints.cypher",
                    "import_graph_shell.cypher",
                    "import_citations_shell.cypher",
                ):
                    archive.writestr(f"root/pkg/{filename}", "RETURN 1;\n")
            output = root / "prepared"
            report = prepare_import(source, output)
            self.assertTrue(report["source_is_immutable"])
            rel = report["files"]["authored_by.csv"]
            self.assertEqual(rel["source_row_count"], 2)
            self.assertEqual(rel["row_count"], 1)
            self.assertEqual(rel["duplicate_row_count"], 1)
            self.assertEqual(rel["conflicting_duplicate_count"], 1)
            with (output / "paper_kg/authored_by.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[1][2], "last")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "shenzhi_neo4j_import_v1")


if __name__ == "__main__":
    unittest.main()
