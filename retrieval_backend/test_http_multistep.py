import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from http.server import ThreadingHTTPServer

from http_server import BoundedThreadingHTTPServer, make_handler
from query_rewrite import QueryRewriteService
from query_parser import QueryParser
from paper_search import PaperSearchIndex
from retrieval_pipeline import HybridPaperSearch
from multistep_search import Stage2ActionPolicy


class UnavailableGraph:
    def healthcheck(self):
        raise ConnectionError("Neo4j unavailable")

    def filter_candidate_ids(self, _candidate_ids, *, filters):
        raise ConnectionError("Neo4j unavailable")


class HttpMultistepTest(unittest.TestCase):
    def _serve(self, engine, **handler_options):
        server = BoundedThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(engine, **handler_options), max_workers=2
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_http_limits_auth_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(json.dumps({"paper_id": "p", "title": "graph"}) + "\n")
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            server, thread = self._serve(
                HybridPaperSearch(index), max_body_bytes=32, auth_token="test-secret"
            )
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/health")
                self.assertEqual(conn.getresponse().status, 401)
                conn.close()

                headers = {"Authorization": "Bearer test-secret"}
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/ready", headers=headers)
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())["ready"])
                conn.close()

                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("POST", "/search", body=b"x" * 33, headers=headers)
                self.assertEqual(conn.getresponse().status, 413)
                conn.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_http_remains_ready_and_searchable_when_optional_graph_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps(
                    {
                        "paper_id": "p",
                        "title": "graph retrieval",
                        "conference": "AAAI",
                    }
                )
                + "\n"
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            server, thread = self._serve(
                HybridPaperSearch(index, graph_filter=UnavailableGraph())
            )
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/ready")
                response = conn.getresponse()
                readiness = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertTrue(readiness["ready"])
                self.assertEqual(readiness["status"], "degraded")
                conn.close()

                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                body = json.dumps(
                    {"query": "graph retrieval", "conference": ["AAAI"]}
                ).encode()
                conn.request(
                    "POST",
                    "/search",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual([row["paper_id"] for row in payload["results"]], ["p"])
                self.assertIn(
                    "FILTER_PAPER_NEO4J_FALLBACK",
                    payload["state"]["executed_operations"],
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_query_rewrite_endpoint_and_search_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(json.dumps({"paper_id": "p", "title": "knowledge graph completion"}) + "\n")
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            rewriter = QueryRewriteService(
                Path(__file__).parent / "data" / "domain_glossary.v1.json"
            )
            server, thread = self._serve(HybridPaperSearch(index), query_rewriter=rewriter)
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/query-rewrite?q=%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E8%A1%A5%E5%85%A8")
                rewrite = json.loads(conn.getresponse().read())
                self.assertEqual(rewrite["translated_query"], "knowledge graph completion")
                conn.close()
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                body = json.dumps({"query": "知识图谱补全", "top_k": 1}).encode()
                conn.request("POST", "/search", body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["results"][0]["paper_id"], "p")
                self.assertEqual(payload["query_rewrite"]["source"], "glossary")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_multistep_endpoint_returns_executable_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "a", "title": "unique graph", "abstract": "graph", "authors": ["Alice"], "keywords": ["kg"], "subjects": ["retrieval"]},
                        {"paper_id": "b", "title": "alice work", "abstract": "related", "authors": ["Alice"], "keywords": ["vision"], "subjects": ["vision"]},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(HybridPaperSearch(index)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                body = json.dumps({"query": "unique graph", "max_steps": 4}).encode()
                conn.request("POST", "/multistep-search", body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertTrue(payload["results"])
                self.assertIn("query_parse", payload)
                events = payload["state"]["events"]
                self.assertEqual(events[0]["action"], "SEARCH_PAPERS")
                self.assertEqual(events[-1]["action"], "STOP")
                self.assertTrue(all(event["result_text"] for event in events))
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_http_parses_chinese_metadata_and_uses_filtered_browse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"paper_id": "a", "title": "graph paper", "conference": "AAAI", "year": 2024, "authors": ["Alice Smith"]},
                        {"paper_id": "b", "title": "graph paper", "conference": "ICML", "year": 2024, "authors": ["Alice Smith"]},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            server, thread = self._serve(
                HybridPaperSearch(index),
                query_rewriter=QueryRewriteService(
                    Path(__file__).parent / "data" / "domain_glossary.v1.json"
                ),
                query_parser=QueryParser(
                    Path(__file__).parent / "data" / "domain_glossary.v1.json"
                ),
            )
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                body = json.dumps({"query": "2020年以后在AAAI发表的论文", "top_k": 10}).encode()
                conn.request("POST", "/search", body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual([row["paper_id"] for row in payload["results"]], ["a"])
                self.assertEqual(payload["query_parse"]["filters"]["conference"], ["AAAI"])
                self.assertIn("BROWSE_PAPER_FILTERED", payload["state"]["executed_operations"])
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_stage2_policy_falls_back_cleanly_when_model_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs.jsonl"
            docs.write_text(
                json.dumps({"paper_id": "p", "title": "graph", "abstract": "graph"}) + "\n",
                encoding="utf-8",
            )
            index = PaperSearchIndex(root / "papers.db")
            index.build(docs)
            # An explicitly configured but invalid model path must not make the
            # multistep endpoint unavailable; the policy is fail-safe by design.
            policy = Stage2ActionPolicy(
                checkpoint_path=str(root / "missing.pt"),
                skills_path=str(root / "missing.jsonl"),
                clstr_source=str(root / "missing-source"),
                device="cpu",
            )
            self.assertFalse(policy.enabled)
            self.assertIsNotNone(policy.load_error)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(HybridPaperSearch(index), policy),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                body = json.dumps({"query": "graph", "max_steps": 2}).encode()
                conn.request("POST", "/multistep-search", body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["state"]["events"][0]["action"], "SEARCH_PAPERS")
                conn.close()

                conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/health")
                response = conn.getresponse()
                health = json.loads(response.read().decode())
                self.assertEqual(health["stage2"]["status"], "fallback")
                self.assertTrue(health["stage2"]["load_error"])
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
