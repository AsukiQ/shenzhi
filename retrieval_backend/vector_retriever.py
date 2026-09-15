"""Optional Qwen Embedding + Zilliz query adapter.

This module is intentionally independent from the live HTTP server. It returns
the same ``CandidateHit`` objects used by ``retrieval_pipeline.py`` so it can be
wired into RRF after a smoke test, while BM25 remains the safe fallback.
"""

from __future__ import annotations

import os
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

    def healthcheck(self) -> dict[str, Any]:
        try:
            exists = bool(self.client.has_collection(collection_name=self.collection))
            return {"status": "ready" if exists else "missing_collection", "collection": self.collection}
        except Exception as exc:
            return {"status": "error", "collection": self.collection, "error": f"{type(exc).__name__}: {exc}"}

    def recall(self, query: str, *, limit: int) -> list[CandidateHit]:
        vector = self.embedder.embed([str(query)])[0]
        rows = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="embedding",
            limit=max(1, min(int(limit) * 3, 1000)),
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
        return [CandidateHit(pid, score, rank, "dense") for rank, (pid, score) in enumerate(scored[:max(1, int(limit))], 1)]
