"""Optional Qwen Embedding + Zilliz query adapter.

This module is intentionally independent from the live HTTP server. It returns
the same ``CandidateHit`` objects used by ``retrieval_pipeline.py`` so it can be
wired into RRF after a smoke test, while BM25 remains the safe fallback.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any

try:
    from .paper_search import CandidateHit
    from .vector_ingest import EmbeddingClient, _zilliz_client
except ImportError:
    from paper_search import CandidateHit
    from vector_ingest import EmbeddingClient, _zilliz_client


class ZillizDenseRetriever:
    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        collection: str | None = None,
        embedding_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        timeout: float = 60.0,
        cache_size: int | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        self.component_name = "zilliz_dense"
        self.collection = collection or os.getenv("ZILLIZ_COLLECTION", "papers_qwen3_embedding_8b")
        self.embedder = EmbeddingClient(
            embedding_url or os.environ["QWEN_EMBEDDING_URL"],
            embedding_api_key or os.environ["QWEN_EMBEDDING_API_KEY"],
            embedding_model or os.getenv("QWEN_EMBEDDING_MODEL", "qwen3-embedding-8b"),
            timeout=timeout,
            dimensions=(int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "0")) or None),
        )
        self.client = _zilliz_client(uri or os.environ["ZILLIZ_URI"], token or os.environ["ZILLIZ_TOKEN"])
        self.cache_size = max(
            0,
            int(
                cache_size
                if cache_size is not None
                else os.getenv("DENSE_QUERY_CACHE_SIZE", "256")
            ),
        )
        self.cache_ttl_seconds = max(
            0.0,
            float(
                cache_ttl_seconds
                if cache_ttl_seconds is not None
                else os.getenv("DENSE_QUERY_CACHE_TTL_SECONDS", "300")
            ),
        )
        self._cache: OrderedDict[tuple[str, int], tuple[float, list[CandidateHit]]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        """Drop cached dense results after an embedding/index refresh."""

        with self._cache_lock:
            self._cache.clear()

    def _cached_recall(self, key: tuple[str, int]) -> list[CandidateHit] | None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            timestamp, hits = cached
            if now - timestamp >= self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return list(hits)

    def _store_recall(self, key: tuple[str, int], hits: list[CandidateHit]) -> None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), list(hits))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def healthcheck(self) -> dict[str, Any]:
        try:
            exists = bool(self.client.has_collection(collection_name=self.collection))
            with self._cache_lock:
                cache_entries = len(self._cache)
            return {
                "status": "ready" if exists else "missing_collection",
                "collection": self.collection,
                "cache_size": self.cache_size,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "cache_entries": cache_entries,
            }
        except Exception as exc:
            return {"status": "error", "collection": self.collection, "error": f"{type(exc).__name__}: {exc}"}

    def recall(self, query: str, *, limit: int) -> list[CandidateHit]:
        bounded_limit = max(1, min(int(limit), 1000))
        cache_key = (str(query).strip().casefold(), bounded_limit)
        cached = self._cached_recall(cache_key)
        if cached is not None:
            return cached
        vector = self.embedder.embed([str(query)])[0]
        rows = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="embedding",
            limit=max(1, min(bounded_limit * 3, 1000)),
            output_fields=["paper_id", "chunk_index"],
            search_params={"metric_type": "COSINE"},
        )
        hits = rows[0] if rows else []
        grouped: dict[str, list[float]] = {}
        for hit in hits:
            if isinstance(hit, dict):
                paper_id = hit.get("paper_id") or hit.get("entity", {}).get("paper_id")
                score = hit.get("distance", hit.get("score", 0.0))
            else:
                paper_id = getattr(hit, "id", None)
                score = getattr(hit, "distance", 0.0)
            if paper_id:
                grouped.setdefault(str(paper_id), []).append(float(score))
        scored = []
        for paper_id, scores in grouped.items():
            scores.sort(reverse=True)
            value = scores[0] + (0.25 * scores[1] if len(scores) > 1 else 0.0)
            scored.append((paper_id, value))
        scored.sort(key=lambda x: (-x[1], x[0]))
        hits = [CandidateHit(pid, score, rank, "dense") for rank, (pid, score) in enumerate(scored[:bounded_limit], 1)]
        self._store_recall(cache_key, hits)
        return hits
