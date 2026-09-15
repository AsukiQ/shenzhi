"""End-to-end JSON HTTP smoke for health, search and graph constraints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request


def call_json(url: str, *, payload=None, auth_token: str | None = None):
    encoded = None
    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=encoded, headers=headers, method="POST" if encoded else "GET")
    with request.urlopen(req, timeout=120.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-uri", default="http://127.0.0.1:8080")
    parser.add_argument("--expect-component", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--auth-token-env")
    args = parser.parse_args()
    base = args.base_uri.rstrip("/")
    auth_token = os.environ.get(args.auth_token_env) if args.auth_token_env else None
    health_status, health = call_json(f"{base}/health", auth_token=auth_token)
    if health_status != 200 or health.get("status") != "ok":
        raise RuntimeError(f"retrieval healthcheck failed: {health}")
    components = set(health.get("components") or [])
    missing = set(args.expect_component) - components
    if missing:
        raise RuntimeError(f"retrieval healthcheck lacks components: {sorted(missing)}")
    ready_status, ready = call_json(f"{base}/ready", auth_token=auth_token)
    if ready_status != 200 or not ready.get("ready", True):
        raise RuntimeError(f"retrieval readiness failed: {ready}")
    search_status, search = call_json(
        f"{base}/search",
        payload={
            "query": "knowledge graph completion with graph neural networks",
            "top_k": 5,
            "year_gte": 2020,
            "conference": ["AAAI"],
        }, auth_token=auth_token,
    )
    if search_status != 200 or not search.get("results"):
        raise RuntimeError(f"retrieval search smoke failed: {search}")
    operations = search.get("state", {}).get("executed_operations") or []
    if "neo4j_filter" in components and "FILTER_PAPER_NEO4J" not in operations:
        raise RuntimeError("Neo4j is enabled but the constrained search did not execute its filter")
    graph_expansion = None
    if "neo4j_filter" in components:
        expand_status, graph_expansion = call_json(
            f"{base}/search",
            payload={
                "query": "knowledge graph completion same authors other papers",
                "top_k": 5,
            },
            auth_token=auth_token,
        )
        if expand_status != 200 or not graph_expansion.get("results"):
            raise RuntimeError(f"Neo4j expansion search failed: {graph_expansion}")
        expand_operations = (
            graph_expansion.get("state", {}).get("executed_operations") or []
        )
        if "EXPAND_PAPER_AUTHORS_NEO4J" not in expand_operations:
            raise RuntimeError(
                "Neo4j is enabled but the relation query did not execute author expansion"
            )
    report = {
        "status": "ok",
        "base_uri": base,
        "health": health,
        "readiness": ready,
        "result_count": len(search["results"]),
        "top_result": search["results"][0],
        "operations": operations,
        "graph_expansion": graph_expansion,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
