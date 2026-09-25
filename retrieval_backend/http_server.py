"""无第三方依赖的 JSON HTTP 包装，方便前端联调。"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse, unquote

try:  # 作为模块运行和直接运行两种方式都支持
    from .paper_search import PaperSearchIndex, SearchFilters, state_to_dict
    from .retrieval_pipeline import (
        HybridPaperSearch,
        SkillRouterPaperReranker,
        Stage0DenseRetriever,
    )
    from .vector_retriever import ZillizDenseRetriever
    from .neo4j_filter import Neo4jHttpPaperFilter, Neo4jPaperFilter
    from .multistep_search import MultiStepPaperSearch, Stage2ActionPolicy, multistep_payload
    from .query_rewrite import QueryRewriteService
    from .query_parser import QueryParser, merge_filters
except ImportError:
    from paper_search import PaperSearchIndex, SearchFilters, state_to_dict
    from retrieval_pipeline import (
        HybridPaperSearch,
        SkillRouterPaperReranker,
        Stage0DenseRetriever,
    )
    from vector_retriever import ZillizDenseRetriever
    from neo4j_filter import Neo4jHttpPaperFilter, Neo4jPaperFilter
    from multistep_search import MultiStepPaperSearch, Stage2ActionPolicy, multistep_payload
    from query_rewrite import QueryRewriteService
    from query_parser import QueryParser, merge_filters


def _str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if str(x).strip()]


def _int_or_none(value):
    if value in (None, ""):
        return None
    return int(value)


def _one(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key, [])
    return values[0] if values else None


def _request_payload(path: str, body: bytes = b"") -> dict:
    if body:
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload
    parsed = urlparse(path)
    params = parse_qs(parsed.query)
    return {
        "query": _one(params, "q") or _one(params, "query") or "",
        "top_k": int(_one(params, "top_k") or 10),
        "offset": int(_one(params, "offset") or 0),
        "limit": int(_one(params, "limit") or 10),
        "year_gte": int(_one(params, "year_gte")) if _one(params, "year_gte") else None,
        "year_lte": int(_one(params, "year_lte")) if _one(params, "year_lte") else None,
        "conference": params.get("conference", []),
        "author": params.get("author", []),
        "keyword": params.get("keyword", []),
        "subject": params.get("subject", []),
        "institution": params.get("institution", []),
    }


def make_handler(
    index,
    stage2_policy=None,
    *,
    graph=None,
    max_body_bytes: int = 1_048_576,
    request_timeout: float = 30.0,
    auth_token: str | None = None,
    query_rewriter: QueryRewriteService | None = None,
    query_parser: QueryParser | None = None,
):
    multistep = MultiStepPaperSearch(index)
    parser_instance = query_parser or QueryParser()
    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(request_timeout)

        def _authorized(self) -> bool:
            if not auth_token:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {auth_token}"

        def _require_auth(self) -> bool:
            if self._authorized():
                return True
            self._send(401, {"error": "unauthorized"}, extra_headers={"WWW-Authenticate": "Bearer"})
            return False

        def _send(self, status: int, payload: dict, *, extra_headers: dict[str, str] | None = None) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if not self._require_auth():
                return
            request_path = urlparse(self.path).path
            # Demo/legacy compatibility: accept the documented API prefixes.
            if request_path.startswith('/api/retrieval/'):
                request_path = request_path[len('/api/retrieval'):]
            elif request_path.startswith('/api/knowledge/'):
                request_path = request_path[len('/api'):]
            elif request_path == '/api/kg/search':
                request_path = '/search'
            if request_path == "/knowledge/papers/summary":
                try:
                    health = index.healthcheck()
                    lexical = health.get("lexical", health)
                    self._send(200, {"paper_count": int(lexical.get("paper_count", 0))})
                except Exception as exc:
                    self._send(500, {"error": "paper summary failed", "type": type(exc).__name__})
                return
            if request_path == "/knowledge/research-assets/summary":
                try:
                    health = graph.healthcheck() if graph is not None else {}
                    count = int(health.get("funding_count", 0))
                    self._send(200, {"research_asset_count": count})
                except Exception as exc:
                    self._send(500, {"error": "research asset summary failed", "type": type(exc).__name__})
                return
            if request_path in {"/scholars/search"}:
                if graph is None or not hasattr(graph, "scholar_search"):
                    self._send(503, {"error": "scholar search unavailable"})
                    return
                params = parse_qs(urlparse(self.path).query)
                query = _one(params, "q") or _one(params, "query") or ""
                subject = _one(params, "subject") or ""
                funding = _one(params, "funding") or ""
                try:
                    rows = graph.scholar_search(query, subject=subject, funding=funding, limit=int(_one(params, "limit") or 20), offset=int(_one(params, "offset") or 0))
                    self._send(200, {"results": rows, "query": query})
                except (ValueError, TypeError) as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": "scholar search failed", "type": type(exc).__name__})
                return
            if request_path == "/fundings/search":
                if graph is None or not hasattr(graph, "funding_search"):
                    self._send(503, {"error": "funding search unavailable"})
                    return
                params = parse_qs(urlparse(self.path).query)
                query = (_one(params, "q") or _one(params, "query") or "").strip()
                try:
                    rows = graph.funding_search(
                        query, limit=int(_one(params, "limit") or 20),
                        offset=int(_one(params, "offset") or 0),
                    )
                    self._send(200, {"results": rows, "query": query})
                except (ValueError, TypeError) as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": "funding search failed", "type": type(exc).__name__})
                return
            if request_path == "/institutions/search":
                if graph is None or not hasattr(graph, "institution_search"):
                    self._send(503, {"error": "institution search unavailable"})
                    return
                params = parse_qs(urlparse(self.path).query)
                query = _one(params, "q") or _one(params, "query") or ""
                try:
                    rows = graph.institution_search(
                        query,
                        limit=int(_one(params, "limit") or 20),
                        offset=int(_one(params, "offset") or 0),
                    )
                    self._send(200, {"results": rows, "query": query})
                except (ValueError, TypeError) as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": "institution search failed", "type": type(exc).__name__})
                return
            if request_path in {"/search/by-subject", "/search/by-funding", "/search/by-institution"}:
                params = parse_qs(urlparse(self.path).query)
                field_name = ("subject" if request_path.endswith("by-subject") else
                              "institution" if request_path.endswith("by-institution") else "funding")
                value = _one(params, field_name) or _one(params, "q") or ""
                if not value:
                    self._send(400, {"error": field_name + " is required"})
                    return
                if request_path == "/search/by-subject" and ("offset" in params or "limit" in params):
                    try:
                        offset = int(_one(params, "offset") or 0)
                        limit = int(_one(params, "limit") or 10)
                        lexical = getattr(index, "lexical_index", index)
                        results, total = lexical.subject_page(value, offset=offset, limit=limit)
                        self._send(200, {"results": [r.__dict__ for r in results], "total": total})
                    except (ValueError, TypeError) as exc:
                        self._send(400, {"error": str(exc)})
                    except Exception as exc:
                        self._send(500, {"error": "subject pagination failed", "type": type(exc).__name__})
                    return
                payload = {"query": value, "top_k": int(_one(params, "top_k") or 10)}
                payload[field_name] = [value]
                self._search(payload)
                return
            if request_path.startswith("/scholars/"):
                if graph is None or not hasattr(graph, "scholar_detail"):
                    self._send(503, {"error": "scholar search unavailable"})
                    return
                scholar_id = unquote(request_path[len("/scholars/"):].strip("/"))
                if not scholar_id:
                    self._send(400, {"error": "scholar_id is required"})
                    return
                try:
                    result = graph.scholar_detail(scholar_id)
                    self._send(200 if result else 404, result or {"error": "scholar not found", "scholar_id": scholar_id})
                except Exception as exc:
                    self._send(500, {"error": "scholar detail failed", "type": type(exc).__name__})
                return
            if request_path.startswith("/papers/") and request_path.endswith("/images"):
                paper_id = unquote(request_path[len("/papers/") : -len("/images")].strip("/"))
                if not paper_id:
                    self._send(400, {"error": "paper_id is required"})
                    return
                lexical = getattr(index, "lexical_index", index)
                self._send(200, {"paper_id": paper_id, "image_id": lexical.image_id_for(paper_id)})
                return
            if request_path in {"/health", "/ready"}:
                try:
                    checker = getattr(index, "readiness" if request_path == "/ready" else "healthcheck", None)
                    payload = checker() if callable(checker) else {"status": "ok"}
                    if stage2_policy is not None:
                        payload["stage2"] = {
                            "status": "ready" if stage2_policy.enabled else "fallback",
                            "enabled": stage2_policy.enabled,
                            "load_error": stage2_policy.load_error,
                        }
                    status = 200 if request_path == "/health" or payload.get("ready", True) else 503
                    self._send(status, payload)
                except Exception as exc:  # health must always return JSON
                    self._send(503, {"status": "error", "error": type(exc).__name__})
                return
            if request_path == "/query-rewrite":
                try:
                    result = query_rewriter.rewrite(_request_payload(self.path).get("query", "")) if query_rewriter else None
                    self._send(200, result.as_dict() if result else {"error": "rewrite disabled"})
                except Exception as exc:
                    self._send(500, {"error": "query rewrite error", "type": type(exc).__name__})
                return
            if request_path not in {"/search", "/multistep-search"}:
                self._send(404, {"error": "use GET /search, /multistep-search, /scholars/search, /scholars/{scholar_id}, /institutions/search, /query-rewrite, /papers/{paper_id}/images or /health"})
                return
            try:
                payload = _request_payload(self.path)
                if request_path == "/multistep-search":
                    self._multistep_search(payload)
                else:
                    self._search(payload)
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": "internal retrieval error", "type": type(exc).__name__})

        def do_POST(self):  # noqa: N802
            if not self._require_auth():
                return
            request_path = urlparse(self.path).path
            if request_path.startswith('/api/retrieval/'):
                request_path = request_path[len('/api/retrieval'):]
            elif request_path == '/api/kg/search':
                request_path = '/search'
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > max_body_bytes:
                self._send(413, {"error": "request body too large"})
                return
            if request_path == "/query-rewrite":
                try:
                    payload = _request_payload(self.path, self.rfile.read(length))
                    result = query_rewriter.rewrite(payload.get("query", "")) if query_rewriter else None
                    self._send(200, result.as_dict() if result else {"error": "rewrite disabled"})
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                return
            if request_path not in {"/search", "/multistep-search"}:
                self._send(404, {"error": "use POST /search or /multistep-search"})
                return
            try:
                payload = _request_payload(self.path, self.rfile.read(length))
                if request_path == "/multistep-search":
                    self._multistep_search(payload)
                else:
                    self._search(payload)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": "internal retrieval error", "type": type(exc).__name__})

        def _search(self, payload: dict) -> None:
            query = str(payload.get("query", payload.get("q", "")))
            paged = "offset" in payload or "limit" in payload
            offset = int(payload.get("offset", 0) or 0)
            limit = int(payload.get("limit", payload.get("top_k", 10)) or 10)
            if offset < 0:
                raise ValueError("offset must be >= 0")
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
            parsed = parser_instance.parse(query)
            explicit_filters = SearchFilters(
                year_gte=_int_or_none(payload.get("year_gte")),
                year_lte=_int_or_none(payload.get("year_lte")),
                conference=_str_list(payload.get("conference")),
                author=_str_list(payload.get("author")),
                keyword=_str_list(payload.get("keyword")),
                subject=_str_list(payload.get("subject")),
                institution=_str_list(payload.get("institution")),
            )
            filters = merge_filters(parsed.filters, explicit_filters)
            rewrite = query_rewriter.rewrite(parsed.semantic_query) if query_rewriter else None
            variants = rewrite.as_dict() if rewrite else {}
            variants.update(
                {
                    "original_query": parsed.original_query,
                    "semantic_query": parsed.semantic_query,
                    "graph_query": parsed.graph_query,
                    "structured_only": not bool(parsed.semantic_query),
                }
            )
            search_k = min(1000, offset + limit) if paged else int(payload.get("top_k", 10))
            results, state = index.search(
                query,
                filters=filters,
                top_k=search_k,
                query_variants=variants,
            )
            if paged:
                page = results[offset:offset + limit]
                response = {
                    "results": [r.__dict__ for r in page],
                    "total": len(results),
                    "total_exact": len(results) < 1000,
                    "state": state_to_dict(state),
                }
            else:
                response = {"results": [r.__dict__ for r in results], "state": state_to_dict(state)}
            response["query_parse"] = parsed.as_dict()
            if rewrite:
                response["query_rewrite"] = {
                    **rewrite.as_dict(),
                    "request_query": query,
                }
            self._send(200, response)

        def _multistep_search(self, payload: dict) -> None:
            query = str(payload.get("query", payload.get("q", ""))).strip()
            parsed = parser_instance.parse(query)
            explicit_filters = SearchFilters(
                year_gte=_int_or_none(payload.get("year_gte")),
                year_lte=_int_or_none(payload.get("year_lte")),
                conference=_str_list(payload.get("conference")),
                author=_str_list(payload.get("author")),
                keyword=_str_list(payload.get("keyword")),
                subject=_str_list(payload.get("subject")),
                institution=_str_list(payload.get("institution")),
            )
            filters = merge_filters(parsed.filters, explicit_filters)
            rewrite = query_rewriter.rewrite(parsed.semantic_query) if query_rewriter else None
            variants = rewrite.as_dict() if rewrite else {}
            variants.update(
                {
                    "original_query": parsed.original_query,
                    "semantic_query": parsed.semantic_query,
                    "graph_query": parsed.graph_query,
                    "structured_only": not bool(parsed.semantic_query),
                }
            )
            results, state = multistep.run(
                query,
                filters=filters,
                top_k=int(payload.get("top_k", 10)),
                max_steps=int(payload.get("max_steps", 6)),
                policy=(stage2_policy.new_episode() if stage2_policy is not None else None),
                query_variants=variants,
            )
            response = multistep_payload(results, state)
            response["query_parse"] = parsed.as_dict()
            if rewrite:
                response["query_rewrite"] = {
                    **rewrite.as_dict(),
                    "request_query": query,
                }
            self._send(200, response)

        def log_message(self, *_args):
            return

    return Handler


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        handler,
        *,
        max_workers: int = 16,
        worker_slot_wait: float = 0.1,
    ):
        self._request_slots = threading.BoundedSemaphore(max(1, int(max_workers)))
        self._worker_slot_wait = max(0.0, float(worker_slot_wait))
        super().__init__(server_address, handler)

    def process_request(self, request, client_address):
        if not self._request_slots.acquire(timeout=self._worker_slot_wait):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                )
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-body-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--worker-slot-wait", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--auth-token-env")
    parser.add_argument("--warmup-query")
    parser.add_argument("--glossary")
    parser.add_argument("--translator-url")
    parser.add_argument("--translator-timeout", type=float, default=0.8)
    parser.add_argument("--stage0-checkpoint")
    parser.add_argument("--stage0-skills")
    parser.add_argument("--stage0-base-model")
    parser.add_argument("--clstr-source")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--stage2-skills")
    parser.add_argument("--stage2-clstr-source")
    parser.add_argument("--stage2-device", default="cuda")
    parser.add_argument("--stage2-selection", default="adaptive", choices=("adaptive", "static", "dynamic"))
    parser.add_argument("--stage2-coarse-k", type=int, default=500)
    parser.add_argument("--stage2-dynamic-extra-k", type=int, default=64)
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--recall-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--fuzzy-weight", type=float, default=0.35)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--zilliz-dense", action="store_true", help="enable Qwen Embedding + Zilliz dense recall")
    parser.add_argument("--zilliz-collection", default=os.getenv("ZILLIZ_COLLECTION", "paper_embedding_chunks_v1_1024"))
    parser.add_argument("--graph-weight", type=float, default=0.8)
    parser.add_argument("--graph-seed-k", type=int, default=12)
    parser.add_argument("--graph-expand-k", type=int, default=100)
    parser.add_argument("--graph-value-limit", type=int, default=16)
    parser.add_argument("--disable-graph-intent", action="store_true")
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-http-uri")
    parser.add_argument("--neo4j-user")
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--neo4j-timeout", type=float, default=3.0)
    args = parser.parse_args()
    dense_enabled = any((args.stage0_checkpoint, args.stage0_skills))
    if dense_enabled and not all(
        (
            args.stage0_checkpoint,
            args.stage0_skills,
            args.clstr_source,
            args.model_cache_dir,
        )
    ):
        parser.error(
            "Stage0 dense retrieval requires --stage0-checkpoint, --stage0-skills, "
            "--clstr-source and --model-cache-dir"
        )
    dense = (
        Stage0DenseRetriever(
            checkpoint_path=args.stage0_checkpoint,
            skills_path=args.stage0_skills,
            clstr_source=args.clstr_source,
            model_cache_dir=args.model_cache_dir,
            base_model_name_override=args.stage0_base_model,
        )
        if dense_enabled
        else None
    )
    zilliz_dense = None
    if args.zilliz_dense or os.getenv("ZILLIZ_DENSE_ENABLED", "").lower() in {"1", "true", "yes"}:
        try:
            zilliz_dense = ZillizDenseRetriever(collection=args.zilliz_collection)
            health = zilliz_dense.healthcheck()
            if health.get("status") != "ready":
                print(f"zilliz dense unavailable; keeping BM25 fallback: {health}", flush=True)
                zilliz_dense = None
            else:
                print(f"zilliz dense enabled: collection={args.zilliz_collection}", flush=True)
        except Exception as exc:
            print(f"zilliz dense unavailable; keeping BM25 fallback: {type(exc).__name__}: {exc}", flush=True)
            zilliz_dense = None
    if args.reranker_model and not args.clstr_source:
        parser.error("--reranker-model requires --clstr-source")
    reranker = (
        SkillRouterPaperReranker(
            model_path=args.reranker_model,
            clstr_source=args.clstr_source,
            batch_size=args.reranker_batch_size,
        )
        if args.reranker_model
        else None
    )
    graph = None
    if args.neo4j_uri and args.neo4j_http_uri:
        parser.error("choose one of --neo4j-uri or --neo4j-http-uri")
    if args.neo4j_uri or args.neo4j_http_uri:
        if not args.neo4j_user:
            parser.error("Neo4j filtering requires --neo4j-user")
        password = os.environ.get(args.neo4j_password_env)
        if not password:
            parser.error(
                f"Neo4j password environment variable is empty: {args.neo4j_password_env}"
            )
        graph = (
            Neo4jPaperFilter(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=password,
                database=args.neo4j_database,
                timeout_seconds=args.neo4j_timeout,
            )
            if args.neo4j_uri
            else Neo4jHttpPaperFilter(
                http_uri=args.neo4j_http_uri,
                user=args.neo4j_user,
                password=password,
                database=args.neo4j_database or "neo4j",
                timeout_seconds=args.neo4j_timeout,
            )
        )
        graph.healthcheck()
    engine = HybridPaperSearch(
        PaperSearchIndex(args.db),
        dense_retriever=zilliz_dense or dense,
        reranker=reranker,
        graph_filter=graph,
        lexical_weight=args.bm25_weight,
        fuzzy_weight=args.fuzzy_weight,
        dense_weight=args.dense_weight,
        recall_k=args.recall_k,
        rerank_k=args.rerank_k,
        graph_weight=args.graph_weight,
        graph_seed_k=args.graph_seed_k,
        graph_expand_k=args.graph_expand_k,
        graph_value_limit=args.graph_value_limit,
        graph_intent_enabled=not args.disable_graph_intent,
    )
    stage2_enabled = any((args.stage2_checkpoint, args.stage2_skills, args.stage2_clstr_source))
    if stage2_enabled and not all((args.stage2_checkpoint, args.stage2_skills, args.stage2_clstr_source)):
        parser.error("Stage2 policy requires --stage2-checkpoint, --stage2-skills and --stage2-clstr-source")
    stage2_policy = (
        Stage2ActionPolicy(
            checkpoint_path=args.stage2_checkpoint,
            skills_path=args.stage2_skills,
            clstr_source=args.stage2_clstr_source,
            device=args.stage2_device,
            route_mode=args.stage2_selection,
            coarse_k=args.stage2_coarse_k,
            dynamic_extra_k=args.stage2_dynamic_extra_k,
        )
        if stage2_enabled
        else None
    )
    auth_token = os.environ.get(args.auth_token_env) if args.auth_token_env else None
    if args.auth_token_env and not auth_token:
        parser.error(f"authentication token environment variable is empty: {args.auth_token_env}")
    query_rewriter = QueryRewriteService(
        args.glossary,
        translator_url=args.translator_url,
        timeout_seconds=args.translator_timeout,
    ) if args.glossary or args.translator_url else None
    query_parser = QueryParser(args.glossary)
    if args.warmup_query:
        warmup_parsed = query_parser.parse(args.warmup_query)
        warmup_rewrite = query_rewriter.rewrite(warmup_parsed.semantic_query) if query_rewriter else None
        warmup_variants = warmup_rewrite.as_dict() if warmup_rewrite else {}
        warmup_variants.update(
            {
                "original_query": warmup_parsed.original_query,
                "semantic_query": warmup_parsed.semantic_query,
                "graph_query": warmup_parsed.graph_query,
                "structured_only": not bool(warmup_parsed.semantic_query),
            }
        )
        engine.search(
            args.warmup_query,
            top_k=1,
            apply_reranker=False,
            filters=warmup_parsed.filters,
            query_variants=warmup_variants,
        )
    server = BoundedThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            engine,
            stage2_policy,
            max_body_bytes=max(1, args.max_body_bytes),
            request_timeout=max(1.0, args.request_timeout),
            auth_token=auth_token,
            graph=graph,
            query_rewriter=query_rewriter,
            query_parser=query_parser,
        ),
        max_workers=args.max_workers,
        worker_slot_wait=args.worker_slot_wait,
    )
    enabled = ["bm25"]
    if zilliz_dense is not None:
        enabled.append("zilliz_dense")
    elif dense is not None:
        enabled.append("stage0_dense")
    if reranker is not None:
        enabled.append("skillrouter_reranker")
    if graph is not None:
        enabled.append("neo4j_filter")
    if stage2_policy is not None:
        enabled.append("clstr_stage2" if stage2_policy.enabled else "clstr_stage2_fallback")
    print(f"retrieval components: {','.join(enabled)}", flush=True)
    print(f"paper search server listening on http://{args.host}:{args.port}/search", flush=True)
    try:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
