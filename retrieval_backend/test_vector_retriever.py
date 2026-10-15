import unittest
from collections import OrderedDict
import threading

from paper_search import CandidateHit
from vector_retriever import ZillizDenseRetriever


class _Embedder:
    def __init__(self):
        self.calls = 0

    def embed(self, _texts):
        self.calls += 1
        return [[0.1, 0.2]]


class _Client:
    def __init__(self):
        self.calls = 0

    def search(self, **_kwargs):
        self.calls += 1
        return [[{"paper_id": "p", "distance": 0.9}]]


class DenseRetrieverCacheTest(unittest.TestCase):
    def test_repeated_query_uses_ttl_cache(self):
        retriever = object.__new__(ZillizDenseRetriever)
        retriever.collection = "papers"
        retriever.embedder = _Embedder()
        retriever.client = _Client()
        retriever.cache_size = 8
        retriever.cache_ttl_seconds = 60.0
        retriever._cache = OrderedDict()
        retriever._cache_lock = threading.Lock()

        first = retriever.recall("Graph", limit=10)
        second = retriever.recall(" graph ", limit=10)

        self.assertEqual(first, [CandidateHit("p", 0.9, 1, "dense")])
        self.assertEqual(second, first)
        self.assertEqual(retriever.embedder.calls, 1)
        self.assertEqual(retriever.client.calls, 1)

    def test_zero_cache_keeps_provider_behavior(self):
        retriever = object.__new__(ZillizDenseRetriever)
        retriever.collection = "papers"
        retriever.embedder = _Embedder()
        retriever.client = _Client()
        retriever.cache_size = 0
        retriever.cache_ttl_seconds = 60.0
        retriever._cache = OrderedDict()
        retriever._cache_lock = threading.Lock()

        retriever.recall("graph", limit=10)
        retriever.recall("graph", limit=10)

        self.assertEqual(retriever.embedder.calls, 2)
        self.assertEqual(retriever.client.calls, 2)


if __name__ == "__main__":
    unittest.main()
