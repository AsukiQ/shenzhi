import json
import tempfile
import unittest
from pathlib import Path

from query_rewrite import QueryRewriteService


class QueryRewriteTest(unittest.TestCase):
    def _service(self):
        return QueryRewriteService(Path(__file__).parent / "data" / "domain_glossary.v1.json")

    def test_embedded_chinese_term_drops_request_filler(self):
        result = self._service().rewrite("请找近三年的知识图谱补全论文")
        self.assertEqual(result.translated_query, "knowledge graph completion")
        self.assertFalse(result.fallback)

    def test_expanded_rag_term(self):
        result = self._service().rewrite("检索增强生成")
        self.assertEqual(result.translated_query, "retrieval augmented generation")
        self.assertFalse(result.fallback)

    def test_mixed_latin_query_is_kept(self):
        result = self._service().rewrite("请找 RAG 论文")
        self.assertEqual(result.translated_query, "RAG")
        self.assertFalse(result.fallback)


if __name__ == "__main__":
    unittest.main()
