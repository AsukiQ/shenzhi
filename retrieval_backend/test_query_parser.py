import unittest
from pathlib import Path

from query_parser import QueryParser, merge_filters
from paper_search import SearchFilters


class QueryParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser(Path(__file__).parent / "data" / "domain_glossary.v1.json")

    def test_year_conference_and_glossary_topic(self):
        parsed = self.parser.parse("2020年以后在AAAI发表的图神经网络论文")
        self.assertEqual(parsed.filters.year_gte, 2020)
        self.assertEqual(parsed.filters.conference, ["AAAI"])
        self.assertIn("图神经网络", parsed.semantic_query)
        self.assertEqual(parsed.graph_query, parsed.original_query)

    def test_relation_phrase_is_graph_intent_not_author_name(self):
        parsed = self.parser.parse("找知识图谱补全中同作者的其他论文")
        self.assertEqual(parsed.filters.author, [])
        self.assertEqual(parsed.semantic_query, "知识图谱补全中")
        relations = [row for row in parsed.extracted if row["type"] == "relation"]
        self.assertEqual(relations[0]["value"], "同作者的其他论文")

    def test_explicit_authors(self):
        self.assertEqual(
            self.parser.parse("作者是Alice Smith的图神经网络论文").filters.author,
            ["Alice Smith"],
        )
        self.assertEqual(
            self.parser.parse("作者是张三的论文").filters.author,
            ["张三"],
        )

    def test_unknown_explicit_term_stays_in_semantic_query(self):
        parsed = self.parser.parse("关键词是一个未收录术语")
        self.assertEqual(parsed.filters.keyword, [])
        self.assertIn("未收录术语", parsed.semantic_query)

    def test_explicit_api_filters_override_natural_language(self):
        parsed = self.parser.parse("2020年以后在AAAI发表的论文")
        merged = merge_filters(parsed.filters, SearchFilters(year_gte=2023, conference=["ICML"]))
        self.assertEqual(merged.year_gte, 2023)
        self.assertEqual(merged.conference, ["ICML"])


if __name__ == "__main__":
    unittest.main()
