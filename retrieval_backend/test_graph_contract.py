import json
import unittest
from pathlib import Path

from neo4j_filter import RELATION_TYPES


class GraphContractTest(unittest.TestCase):
    def test_contract_matches_runtime_relation_mapping(self):
        path = Path(__file__).parent / "data" / "graph_retrieval_contract.v1.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            {
                key: tuple(value)
                for key, value in contract["expansion_relations"].items()
            },
            RELATION_TYPES,
        )
        self.assertEqual(contract["canonical_paper_id"], "Paper.paper_id")
        self.assertEqual(
            contract["schema_version"],
            "shenzhi_final_graph_retrieval_contract_v1",
        )
        self.assertEqual(contract["source_snapshot"]["node_count"], 661378)
        self.assertEqual(contract["source_snapshot"]["constraint_count"], 8)
        self.assertEqual(contract["source_snapshot"]["index_count"], 20)
        self.assertEqual(contract["relations"]["PUBLISHED_IN"]["end"], "Conference")
        self.assertEqual(contract["relations"]["HAS_TOPIC"]["end"], "Topic")
        self.assertEqual(
            contract["relations"]["CITES"]["retrieval_roles"],
            ["stored_not_exposed"],
        )
        self.assertTrue(
            contract["forbidden_source"]["path_suffix"].endswith("output_graphs.zip")
        )


if __name__ == "__main__":
    unittest.main()
