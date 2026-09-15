import unittest

from apply_neo4j_schema import cypher_statements


class ApplyNeo4jSchemaTest(unittest.TestCase):
    def test_splits_comments_and_multiline_statements(self):
        rows = cypher_statements(
            """
            // comment
            CREATE CONSTRAINT x IF NOT EXISTS
            FOR (n:Paper) REQUIRE n.paper_id IS UNIQUE;

            CREATE TEXT INDEX y IF NOT EXISTS FOR (n:Paper) ON (n.title);
            """
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("REQUIRE", rows[0])
        self.assertIn("TEXT INDEX", rows[1])


if __name__ == "__main__":
    unittest.main()
