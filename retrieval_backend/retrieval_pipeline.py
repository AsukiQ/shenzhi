"""深知论文检索的可插拔后端主链。

主链保持四层彼此独立：BM25 召回、vNext Stage0 dense 召回、Qwen/SkillRouter
重排和 Neo4j 硬过滤。BM25 始终可单独运行；其余组件只在显式配置后启用。
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any, Iterable, Mapping, Protocol

try:
    from .paper_search import (
        CandidateHit,
        PaperSearchIndex,
        SearchFilters,
        SearchResult,
        SearchState,
        rewrite_fuzzy_tokens,
    )
    from .query_rewrite import contains_cjk
    from .graph_query import infer_graph_query_plan
except ImportError:
    from paper_search import (
        CandidateHit,
        PaperSearchIndex,
        SearchFilters,
        SearchResult,
        SearchState,
        rewrite_fuzzy_tokens,
    )
    from query_rewrite import contains_cjk
    from graph_query import infer_graph_query_plan


class DenseRetriever(Protocol):
    def recall(self, query: str, *, limit: int) -> list[CandidateHit]: ...


class CandidateReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[tuple[str, float]]: ...


class GraphFilter(Protocol):
    def filter_candidate_ids(
        self,
        candidate_ids: list[str],
        *,
        filters: SearchFilters,
    ) -> list[str]: ...


def reciprocal_rank_fusion(
    rankings: Mapping[str, Iterable[CandidateHit]],
    *,
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """Fuse independent rankings without assuming comparable raw scores."""

    if int(rank_constant) < 0:
        raise ValueError("RRF rank_constant must be nonnegative")
    weights = weights or {}
    totals: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    best_rank: dict[str, int] = {}
    for source, raw_hits in rankings.items():
        source_weight = float(weights.get(source, 1.0))
        if source_weight < 0.0:
            raise ValueError("RRF source weights must be nonnegative")
        seen: set[str] = set()
        for fallback_rank, hit in enumerate(raw_hits, start=1):
            paper_id = str(hit.paper_id)
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            rank = max(1, int(hit.rank or fallback_rank))
            contribution = source_weight / float(int(rank_constant) + rank)
            totals[paper_id] = totals.get(paper_id, 0.0) + contribution
            best_rank[paper_id] = min(best_rank.get(paper_id, rank), rank)
            values = components.setdefault(paper_id, {})
            values[f"{source}_raw"] = float(hit.score)
            values[f"{source}_rank"] = float(rank)
            values[f"{source}_rrf"] = contribution
    ordered = sorted(
        totals,
        key=lambda paper_id: (-totals[paper_id], best_rank[paper_id], paper_id),
    )
    for paper_id in ordered:
        components.setdefault(paper_id, {})["rrf"] = totals[paper_id]
    return ordered, totals, components


class Stage0DenseRetriever:
    """Lazy adapter over a canonical CLSTR vNext Stage0 checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        skills_path: str | Path,
        clstr_source: str | Path,
        model_cache_dir: str | Path,
        base_model_name_override: str | Path | None = None,
        device: str | None = None,
        belief_top_k: int = 64,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.skills_path = Path(skills_path)
        self.clstr_source = Path(clstr_source)
        self.model_cache_dir = Path(model_cache_dir)
        self.base_model_name_override = (
            None
            if base_model_name_override is None
            else str(base_model_name_override)
        )
        self.requested_device = device
        self.belief_top_k = int(belief_top_k)
        self._model = None
        self._torch = None
        self._skill_ids: list[str] = []
        self.load_report: dict[str, Any] | None = None
        self.load_error: str | None = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Stage0 checkpoint not found: {self.checkpoint_path}")
        if not self.skills_path.is_file():
            raise FileNotFoundError(f"Stage0 skills not found: {self.skills_path}")
        if str(self.clstr_source) not in sys.path:
            sys.path.insert(0, str(self.clstr_source))
        try:
            import torch
            from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint

            device = torch.device(
                self.requested_device
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model, _config, report = build_clstr_model_from_stage0_checkpoint(
                self.checkpoint_path,
                self.skills_path,
                self.model_cache_dir,
                base_model_name_override=self.base_model_name_override,
            )
            model.to(device).eval()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise
        self._model = model
        self._torch = torch
        self._device = device
        self._skill_ids = [
            str(row.get("skill_id") or row.get("canonical_skill_id") or "")
            for row in model.skills
        ]
        self.load_report = report
        self.load_error = None

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ready" if self._model is not None else "error" if self.load_error else "not_loaded",
            "loaded": self._model is not None,
            "error": self.load_error,
        }

    def recall(self, query: str, *, limit: int) -> list[CandidateHit]:
        with self._lock:
            self._load()
            model = self._model
            torch = self._torch
            assert model is not None and torch is not None
            limit = max(1, min(int(limit), len(self._skill_ids)))
            with torch.no_grad():
                state = f"goal: {str(query).strip()}"
                h_t = model.encode_states([state])
                legal = torch.ones(
                    (1, len(self._skill_ids)),
                    dtype=torch.bool,
                    device=h_t.device,
                )
                belief = model.vnext_initial_belief(
                    h_t,
                    legal,
                    top_k=self.belief_top_k,
                )
                unified_query = model.vnext.static_query(h_t, belief)
                logits = model.vnext_full_pool_logits(
                    unified_query,
                    head="recall",
                )[0]
                scores, indices = torch.topk(logits.float(), k=limit)
        return [
            CandidateHit(
                paper_id=self._skill_ids[int(index)],
                score=float(score),
                rank=rank,
                source="dense",
            )
            for rank, (score, index) in enumerate(
                zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()),
                start=1,
            )
        ]


class SkillRouterPaperReranker:
    """Use the local SkillRouter/Qwen3 reranker on a bounded candidate list."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        clstr_source: str | Path,
        batch_size: int = 4,
        max_length: int = 2048,
        max_document_chars: int = 1800,
        device: str | None = None,
    ) -> None:
        if str(clstr_source) not in sys.path:
            sys.path.insert(0, str(clstr_source))
        from clstr.toolbench_qwen_reranker import (
            Qwen3RerankerConfig,
            Qwen3RerankerScorer,
        )

        instruction = (
            "Given a paper-search query, judge whether the candidate paper is relevant. "
            "Use the title, abstract and metadata; answer yes only for a useful search result."
        )
        self.scorer = Qwen3RerankerScorer(
            Qwen3RerankerConfig(
                model_name_or_path=str(model_path),
                batch_size=int(batch_size),
                max_length=int(max_length),
                max_skill_chars=int(max_document_chars),
                instruction=instruction,
                device=device,
            )
        )
        self.max_document_chars = max(200, int(max_document_chars))
        self._lock = threading.RLock()

    def _document(self, result: SearchResult) -> str:
        text = "\n".join(
            [
                f"paper_id: {result.paper_id}",
                f"title: {result.title}",
                f"abstract: {result.abstract}",
                f"conference: {result.conference or ''}",
                f"year: {result.year or ''}",
                f"authors: {'; '.join(result.authors)}",
                f"keywords: {'; '.join(result.keywords)}",
                f"subjects: {'; '.join(result.subjects)}",
            ]
        )
        return text[: self.max_document_chars]

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[tuple[str, float]]:
        with self._lock:
            scores = self.scorer.score_pairs(
                queries=[query] * len(candidates),
                documents=[self._document(candidate) for candidate in candidates],
            )
        ranked = [
            (candidate.paper_id, float(score), index)
            for index, (candidate, score) in enumerate(zip(candidates, scores))
        ]
        ranked.sort(key=lambda item: (-item[1], item[2]))
        return [(paper_id, score) for paper_id, score, _index in ranked]


class HybridPaperSearch:
    def __init__(
        self,
        lexical_index: PaperSearchIndex,
        *,
        dense_retriever: DenseRetriever | None = None,
        reranker: CandidateReranker | None = None,
        graph_filter: GraphFilter | None = None,
        lexical_weight: float = 1.0,
        fuzzy_weight: float = 0.35,
        dense_weight: float = 1.0,
        rrf_rank_constant: int = 60,
        recall_k: int = 1000,
        rerank_k: int = 50,
        graph_weight: float = 0.8,
        graph_seed_k: int = 12,
        graph_expand_k: int = 100,
        graph_value_limit: int = 16,
        graph_intent_enabled: bool = True,
    ) -> None:
        self.lexical_index = lexical_index
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.graph_filter = graph_filter
        self.lexical_weight = float(lexical_weight)
        self.fuzzy_weight = max(0.0, float(fuzzy_weight))
        self.dense_weight = float(dense_weight)
        self.rrf_rank_constant = int(rrf_rank_constant)
        self.recall_k = max(1, int(recall_k))
        self.rerank_k = max(1, int(rerank_k))
        self.graph_weight = max(0.0, float(graph_weight))
        self.graph_seed_k = max(1, int(graph_seed_k))
        self.graph_expand_k = max(1, min(int(graph_expand_k), 1000))
        self.graph_value_limit = max(1, min(int(graph_value_limit), 128))
        self.graph_intent_enabled = bool(graph_intent_enabled)

    def healthcheck(self) -> dict[str, Any]:
        components = ["bm25"]
        if self.dense_retriever is not None:
            components.append(getattr(self.dense_retriever, "component_name", "stage0_dense"))
        if self.reranker is not None:
            components.append("skillrouter_reranker")
        graph_health = None
        if self.graph_filter is not None:
            components.append("neo4j_filter")
            checker = getattr(self.graph_filter, "healthcheck", None)
            if callable(checker):
                try:
                    graph_health = checker()
                except Exception as exc:
                    graph_health = {
                        "status": "error",
                        "error": type(exc).__name__,
                    }
        dense_health = None
        if self.dense_retriever is not None:
            checker = getattr(self.dense_retriever, "healthcheck", None)
            dense_health = checker() if callable(checker) else {"status": "unknown"}
        graph_degraded = bool(
            graph_health is not None and graph_health.get("status") != "ok"
        )
        return {
            "status": "degraded" if graph_degraded else "ok",
            "components": components,
            "lexical": self.lexical_index.healthcheck(),
            "graph": graph_health,
            "dense": dense_health,
        }

    def readiness(self) -> dict[str, Any]:
        health = self.healthcheck()
        dense = health.get("dense")
        ready = dense is None or dense.get("status") == "ready"
        if not ready:
            status = "not_ready"
        elif health.get("status") == "degraded":
            status = "degraded"
        else:
            status = "ok"
        return {**health, "status": status, "ready": ready}

    def _graph_hits_from_rows(
        self,
        rows: Iterable[tuple[str, Iterable[str]]],
        *,
        seed_ids: Iterable[str],
        relation: str,
        source: str,
    ) -> list[CandidateHit]:
        seed_set = {str(item) for item in seed_ids}
        counts: dict[str, int] = {}
        for _value, paper_ids in rows:
            for paper_id in paper_ids:
                resolved = str(paper_id)
                if resolved and resolved not in seed_set:
                    counts[resolved] = counts.get(resolved, 0) + 1
        ordered = sorted(counts, key=lambda item: (-counts[item], item))[
            : self.graph_expand_k
        ]
        return [
            CandidateHit(
                paper_id=paper_id,
                score=float(counts[paper_id]),
                rank=index + 1,
                source=source,
            )
            for index, paper_id in enumerate(ordered)
        ]

    def _recall_graph_relation(
        self,
        rankings: dict[str, list[CandidateHit]],
        relation: str,
        *,
        seed_ids: list[str],
        operations: list[str],
        state: SearchState,
    ) -> str | None:
        """Add one bounded graph/relation ranking, with SQLite fallback."""

        if not seed_ids:
            return None
        graph = self.graph_filter
        expand = getattr(graph, "expand_from_papers", None)
        if callable(expand):
            try:
                rows, _values = expand(
                    seed_ids,
                    relation=relation,
                    value_limit=self.graph_value_limit,
                    paper_limit=self.graph_expand_k,
                )
                source = f"graph_{relation}"
                rankings[source] = self._graph_hits_from_rows(
                    rows,
                    seed_ids=seed_ids,
                    relation=relation,
                    source=source,
                )
                operations.append(f"EXPAND_PAPER_{relation.upper()}_NEO4J")
                return source
            except Exception as exc:
                state.failed_operations.append(
                    f"EXPAND_PAPER_{relation.upper()}_NEO4J:{type(exc).__name__}"
                )
                operations.append(f"EXPAND_PAPER_{relation.upper()}_SQLITE_FALLBACK")

        try:
            if relation == "venue_year":
                hits, _values = self.lexical_index.expand_venue_year(
                    seed_ids,
                    limit=self.graph_expand_k,
                    value_limit=self.graph_value_limit,
                )
            else:
                hits, _values = self.lexical_index.expand_related(
                    seed_ids,
                    relation=relation,
                    limit=self.graph_expand_k,
                    value_limit=self.graph_value_limit,
                )
            source = f"relation_{relation}"
            rankings[source] = [
                CandidateHit(
                    paper_id=hit.paper_id,
                    score=hit.score,
                    rank=hit.rank,
                    source=source,
                )
                for hit in hits
            ]
            if not any(
                operation == f"EXPAND_PAPER_{relation.upper()}_SQLITE_FALLBACK"
                for operation in operations
            ):
                operations.append(f"EXPAND_PAPER_{relation.upper()}_SQLITE")
            return source
        except Exception as exc:
            state.failed_operations.append(
                f"EXPAND_PAPER_{relation.upper()}_SQLITE:{type(exc).__name__}"
            )
            operations.append(f"EXPAND_PAPER_{relation.upper()}_SKIPPED")
            return None

    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        state: SearchState | None = None,
        apply_reranker: bool = True,
        query_variants: Mapping[str, Any] | None = None,
    ) -> tuple[list[SearchResult], SearchState]:
        filters = filters or SearchFilters()
        top_k = max(1, min(int(top_k), 100))
        new_state = state or SearchState(query=query)
        new_state.query = query
        variants = query_variants or {}
        semantic_query = str(variants.get("semantic_query") or "").strip()
        raw_translated_query = str(variants.get("translated_query") or "").strip()
        rewrite_fallback = bool(variants.get("fallback"))
        translated_query = raw_translated_query
        if contains_cjk(query) and (
            not translated_query
            or translated_query == query
            or rewrite_fallback
        ):
            translated_query = ""
        # ``semantic_query`` is the parser's text after removing explicit
        # metadata/relation phrases. Chinese can be sent directly to the
        # dense provider even when no remote rewrite service is configured.
        if translated_query:
            retrieval_query = translated_query
        elif semantic_query and not contains_cjk(semantic_query):
            retrieval_query = semantic_query
        elif not contains_cjk(query):
            retrieval_query = semantic_query or query
        else:
            # Mixed Chinese/English requests can still make a useful lexical
            # seed from their explicit Latin terms. Pure Chinese is handled
            # by the dense path below.
            mixed_text = semantic_query or (
                "" if bool(variants.get("structured_only")) else query
            )
            retrieval_query = mixed_text if re.search(r"[A-Za-z]{2,}", mixed_text) else ""
        rankings: dict[str, list[CandidateHit]] = {}
        operations: list[str] = []
        lexical_query = retrieval_query
        if lexical_query and not contains_cjk(lexical_query):
            corrected_query = rewrite_fuzzy_tokens(lexical_query)
            if corrected_query != lexical_query:
                lexical_query = corrected_query
                operations.append("REWRITE_PAPER_BM25_FUZZY")
        has_graph_constraints = bool(
            filters.year_gte is not None
            or filters.year_lte is not None
            or filters.conference
            or filters.author
            or filters.keyword
            or filters.subject
            or filters.institution
        )
        # There is one static lexical route. Chinese requests are rewritten
        # before entering it; a pure metadata request uses an explicit browse
        # route instead of pretending that an empty query is BM25.
        if retrieval_query:
            rankings["bm25"] = self.lexical_index.recall(
                lexical_query, limit=self.recall_k
            )
            operations.append("RECALL_PAPER_BM25")
        elif has_graph_constraints and bool(variants.get("structured_only")):
            rankings["filtered"] = self.lexical_index.browse(
                filters=filters, limit=self.recall_k
            )
            operations.append("BROWSE_PAPER_FILTERED")
        else:
            rankings["bm25"] = []
            operations.append("RECALL_PAPER_BM25")
        # Prefix recall remains a bounded, low-weight fallback. It is only
        # enabled for the single English retrieval route and only contributes
        # candidates absent from exact BM25 results.
        if retrieval_query and not contains_cjk(retrieval_query):
            fuzzy_hits = self.lexical_index.recall_prefix(
                lexical_query, limit=self.recall_k
            )
            exact_ids = {hit.paper_id for hit in rankings.get("bm25", [])}
            if fuzzy_hits and any(hit.paper_id not in exact_ids for hit in fuzzy_hits):
                rankings["bm25_fuzzy"] = fuzzy_hits
                operations.append("RECALL_PAPER_BM25_FUZZY")
        # Dense providers such as Qwen support Chinese directly. If the
        # optional translation service is unavailable, use the original
        # Chinese query instead of silently disabling semantic recall.
        dense_query = retrieval_query or (query if contains_cjk(query) else "")
        if self.dense_retriever is not None and dense_query:
            try:
                rankings["dense"] = self.dense_retriever.recall(
                    dense_query,
                    limit=self.recall_k,
                )
                operations.append(
                    "RECALL_PAPER_ZILLIZ" if getattr(self.dense_retriever, "component_name", "") == "zilliz_dense"
                    else "RECALL_PAPER_STAGE0"
                )
            except Exception as exc:
                # Dense is an enrichment path; a transient provider/proxy
                # failure must never turn a usable BM25 request into HTTP 500.
                new_state.failed_operations.append(
                    f"RECALL_PAPER_DENSE:{type(exc).__name__}"
                )

        base_weights = {
            "bm25": self.lexical_weight,
            "dense": self.dense_weight,
            "filtered": self.lexical_weight,
            "bm25_fuzzy": self.fuzzy_weight,
        }
        graph_query = str(variants.get("graph_query") or query)
        graph_plan = infer_graph_query_plan(graph_query, filters)
        graph_source: str | None = None
        if self.graph_intent_enabled and graph_plan.expansion_relation is not None:
            seed_ids, _seed_scores, _seed_components = reciprocal_rank_fusion(
                rankings,
                weights=base_weights,
                rank_constant=self.rrf_rank_constant,
            )
            graph_source = self._recall_graph_relation(
                rankings,
                graph_plan.expansion_relation,
                seed_ids=seed_ids[: self.graph_seed_k],
                operations=operations,
                state=new_state,
            )
        ordered_ids, fused_scores, components = reciprocal_rank_fusion(
            rankings,
            weights={
                **base_weights,
                **{
                    source: self.graph_weight
                    for source in rankings
                    if source.startswith("graph_")
                },
                **{
                    source: self.lexical_weight
                    for source in rankings
                    if source.startswith("relation_")
                },
            },
            rank_constant=self.rrf_rank_constant,
        )

        if self.graph_filter is not None and has_graph_constraints:
            try:
                ordered_ids = self.graph_filter.filter_candidate_ids(
                    ordered_ids,
                    filters=filters,
                )
                operations.append("FILTER_PAPER_NEO4J")
            except Exception as exc:
                # Text/local metadata filtering remains available if Neo4j has
                # a transient error.  The failure is exposed for auditability.
                new_state.failed_operations.append(
                    f"FILTER_PAPER_NEO4J:{type(exc).__name__}"
                )
                operations.append("FILTER_PAPER_NEO4J_FALLBACK")

        mode_parts = ["filtered" if "filtered" in rankings else "bm25"]
        if "bm25_fuzzy" in rankings:
            mode_parts.append("fuzzy")
        if "dense" in rankings:
            mode_parts.append(
                "zilliz_dense" if getattr(self.dense_retriever, "component_name", "") == "zilliz_dense"
                else "stage0_dense"
            )
        if graph_source is not None:
            mode_parts.append(graph_source)
        candidate_limit = max(
            top_k,
            self.rerank_k if self.reranker is not None and apply_reranker else top_k,
        )
        candidates = self.lexical_index.fetch_results(
            ordered_ids,
            filters=filters,
            top_k=candidate_limit,
            score_by_id=fused_scores,
            source_scores_by_id=components,
            retrieval_mode="+".join(mode_parts),
        )
        reranker_query = translated_query or (
            semantic_query
            if semantic_query and not contains_cjk(semantic_query)
            else retrieval_query
            if retrieval_query and not contains_cjk(retrieval_query)
            else ""
        )
        # A metadata-only browse has no semantic English input; keep its
        # deterministic year/row ordering instead of sending the raw Chinese
        # request to an English reranker.
        if self.reranker is not None and candidates and apply_reranker and reranker_query:
            reranked = self.reranker.rerank(
                reranker_query, candidates[: self.rerank_k]
            )
            rerank_score = dict(reranked)
            rerank_ids = [paper_id for paper_id, _score in reranked]
            tail_ids = [
                candidate.paper_id
                for candidate in candidates
                if candidate.paper_id not in rerank_score
            ]
            by_id = {candidate.paper_id: candidate for candidate in candidates}
            candidates = [by_id[paper_id] for paper_id in [*rerank_ids, *tail_ids]]
            for candidate in candidates:
                if candidate.paper_id in rerank_score:
                    candidate.source_scores["reranker"] = rerank_score[candidate.paper_id]
                    candidate.score = rerank_score[candidate.paper_id]
                candidate.retrieval_mode += "+skillrouter_reranker"
            operations.append("RERANK_PAPER_SKILLROUTER")

        results = candidates[:top_k]
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        new_state.step += 1
        new_state.executed_operations.extend(operations)
        new_state.last_result_ids = [result.paper_id for result in results]
        return results, new_state


class DensePaperSearch:
    """Dense-only backend with the same result/state protocol as the hybrid chain."""

    def __init__(
        self,
        lexical_index: PaperSearchIndex,
        dense_retriever: DenseRetriever,
        *,
        graph_filter: GraphFilter | None = None,
        recall_k: int = 1000,
    ) -> None:
        self.lexical_index = lexical_index
        self.dense_retriever = dense_retriever
        self.graph_filter = graph_filter
        self.recall_k = max(1, int(recall_k))

    def healthcheck(self) -> dict[str, Any]:
        graph_health = None
        if self.graph_filter is not None:
            checker = getattr(self.graph_filter, "healthcheck", None)
            if callable(checker):
                graph_health = checker()
        return {
            "status": "ok",
            "components": [
                "stage0_dense",
                *(["neo4j_filter"] if self.graph_filter is not None else []),
            ],
            "lexical": self.lexical_index.healthcheck(),
            "graph": graph_health,
        }

    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        state: SearchState | None = None,
    ) -> tuple[list[SearchResult], SearchState]:
        filters = filters or SearchFilters()
        top_k = max(1, min(int(top_k), 100))
        hits = self.dense_retriever.recall(query, limit=self.recall_k)
        ordered_ids = [hit.paper_id for hit in hits]
        operations = ["RECALL_PAPER_STAGE0"]
        has_graph_constraints = bool(
            filters.year_gte is not None
            or filters.year_lte is not None
            or filters.conference
            or filters.author
            or filters.keyword
            or filters.subject
            or filters.institution
        )
        if self.graph_filter is not None and has_graph_constraints:
            ordered_ids = self.graph_filter.filter_candidate_ids(
                ordered_ids,
                filters=filters,
            )
            operations.append("FILTER_PAPER_NEO4J")
        scores = {hit.paper_id: hit.score for hit in hits}
        results = self.lexical_index.fetch_results(
            ordered_ids,
            filters=filters,
            top_k=top_k,
            score_by_id=scores,
            source_scores_by_id={
                hit.paper_id: {
                    "dense_raw": hit.score,
                    "dense_rank": float(hit.rank),
                }
                for hit in hits
            },
            retrieval_mode="stage0_dense",
        )
        new_state = state or SearchState(query=query)
        new_state.query = query
        new_state.step += 1
        new_state.executed_operations.extend(operations)
        new_state.last_result_ids = [result.paper_id for result in results]
        return results, new_state


class StaticRunDenseRetriever:
    """Small deterministic adapter for tests and offline run replay."""

    def __init__(self, rows_by_query: Mapping[str, Iterable[tuple[str, float]]]):
        self.rows_by_query = {
            str(query): [(str(paper_id), float(score)) for paper_id, score in rows]
            for query, rows in rows_by_query.items()
        }

    def recall(self, query: str, *, limit: int) -> list[CandidateHit]:
        return [
            CandidateHit(paper_id, score, rank, "dense")
            for rank, (paper_id, score) in enumerate(
                self.rows_by_query.get(query, [])[: max(1, int(limit))],
                start=1,
            )
        ]


def results_payload(results: list[SearchResult], state: SearchState) -> dict[str, Any]:
    return {
        "results": [asdict(result) for result in results],
        "state": asdict(state),
    }
