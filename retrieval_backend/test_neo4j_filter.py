import unittest

from neo4j_filter import (
    EXPANSION_CYPHER,
    FILTER_CYPHER,
    RELATION_TYPES,
    Neo4jHttpPaperFilter,
    Neo4jPaperFilter,
)
from paper_search import SearchFilters


class FakeResult(list):
    def single(self):
        return self[0]


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **params):
        self.driver.calls.append((query, params))
        if "count(p)" in query:
            return FakeResult([{"paper_count": 7}])
        if query == EXPANSION_CYPHER:
            return FakeResult([{"value": "Alice", "paper_ids": ["p2", "p3"]}])
        allowed = self.driver.allowed
        return FakeResult(
            [{"paper_id": paper_id} for paper_id in params["candidate_ids"] if paper_id in allowed]
        )


class FakeDriver:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.calls = []

    def session(self, **_kwargs):
        return FakeSession(self)


class Neo4jPaperFilterTest(unittest.TestCase):
    def test_final_graph_schema_is_used_by_filters_and_expansion(self):
        self.assertIn("(c:Conference)", FILTER_CYPHER)
        self.assertIn("c.venue", FILTER_CYPHER)
        self.assertEqual(FILTER_CYPHER.count("[:HAS_TOPIC]"), 2)
        self.assertNotIn("HAS_KEYWORD", FILTER_CYPHER)
        self.assertNotIn("HAS_SUBJECT", FILTER_CYPHER)
        self.assertEqual(RELATION_TYPES["keywords"], ("HAS_TOPIC",))
        self.assertEqual(RELATION_TYPES["subjects"], ("HAS_TOPIC",))
        self.assertEqual(RELATION_TYPES["venue_year"], ("PUBLISHED_IN",))

    def test_filter_is_parameterized_and_preserves_order(self):
        driver = FakeDriver({"b", "c"})
        graph = Neo4jPaperFilter(uri="bolt://unused", user="u", password="p", driver=driver)
        result = graph.filter_candidate_ids(
            ["c", "a", "b"],
            filters=SearchFilters(year_gte=2020, conference=["AAAI"]),
        )
        self.assertEqual(result, ["c", "b"])
        query, params = driver.calls[0]
        self.assertEqual(query, FILTER_CYPHER)
        self.assertNotIn("AAAI", query)
        self.assertEqual(params["conference"], ["AAAI"])
        self.assertEqual(params["candidate_ids"], ["c", "a", "b"])

    def test_healthcheck(self):
        graph = Neo4jPaperFilter(uri="bolt://unused", user="u", password="p", driver=FakeDriver(set()))
        self.assertEqual(graph.healthcheck()["paper_count"], 7)

    def test_driver_graph_expansion_is_parameterized(self):
        driver = FakeDriver(set())
        graph = Neo4jPaperFilter(uri="bolt://unused", user="u", password="p", driver=driver)
        rows, values = graph.expand_from_papers(
            ["p1"], relation="authors", value_limit=5, paper_limit=10, slot_index=0
        )
        self.assertEqual(values, ["Alice"])
        self.assertEqual(rows, [("Alice", ["p2", "p3"])])
        query, params = driver.calls[0]
        self.assertEqual(query, EXPANSION_CYPHER)
        self.assertEqual(params["relation_types"], ["AUTHORED_BY"])
        self.assertEqual(params["seed_ids"], ["p1"])

    def test_http_adapter_maps_columns_and_rows(self):
        graph = Neo4jHttpPaperFilter(
            http_uri="http://localhost:7474",
            user="neo4j",
            password="secret",
        )
        calls = []

        def fake_run(statement, parameters):
            calls.append((statement, parameters))
            return [{"paper_id": "b"}]

        graph._run = fake_run
        result = graph.filter_candidate_ids(
            ["a", "b"],
            filters=SearchFilters(keyword=["graph"]),
        )
        self.assertEqual(result, ["b"])
        self.assertEqual(calls[0][1]["keyword"], ["graph"])

    def test_http_graph_expansion_maps_values(self):
        graph = Neo4jHttpPaperFilter(
            http_uri="http://localhost:7474",
            user="neo4j",
            password="secret",
        )
        calls = []

        def fake_run(statement, parameters):
            calls.append((statement, parameters))
            return [{"value": "kg", "paper_ids": ["p2"]}]

        graph._run = fake_run
        rows, values = graph.expand_from_papers(["p1"], relation="keywords")
        self.assertEqual(rows, [("kg", ["p2"])])
        self.assertEqual(values, ["kg"])
        self.assertEqual(calls[0][0], EXPANSION_CYPHER)
        self.assertEqual(calls[0][1]["relation_types"], ["HAS_TOPIC"])


if __name__ == "__main__":
    unittest.main()
