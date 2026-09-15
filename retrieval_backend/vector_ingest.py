"""Stream papers from SQLite, embed them with an OpenAI-compatible API, and upsert to Zilliz.

The command is deliberately conservative: it never loads the full SQLite table in
memory, supports a small ``--limit``/``--dry-run`` mode, and stores resumable state.
Credentials are read only from environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def paper_text(row: sqlite3.Row, *, max_abstract_chars: int | None = None) -> str:
    """Build the canonical text sent to the document embedding model."""
    title = str(row["title"] or "").strip()
    abstract = str(row["abstract"] or "").strip()
    # A small number of records contain full-text/XML-like payloads rather
    # than abstracts (90k+ chars observed). Keep requests within provider
    # limits while preserving the title and the beginning of the abstract.
    if max_abstract_chars is not None and len(abstract) > max_abstract_chars:
        abstract = abstract[:max_abstract_chars].rstrip() + " …"
    conference = str(row["conference"] or "").strip()
    year = str(row["year"] or "").strip()
    parts = [f"Title: {title}"]
    if abstract:
        parts += ["", f"Abstract: {abstract}"]
    metadata = []
    if conference:
        metadata.append(f"Conference: {conference}")
    if year:
        metadata.append(f"Year: {year}")
    if metadata:
        parts += ["", "\n".join(metadata)]
    return "\n".join(parts)


def paper_chunks(row: sqlite3.Row, *, chunk_chars: int = 60000, overlap_chars: int = 6000) -> list[str]:
    """Split complete paper text only when it exceeds the safe API window.

    The API boundary was measured at roughly 32K tokens; 60K English-heavy
    characters is a conservative practical target. Chunking is character based
    because the provider does not expose its tokenizer.
    """
    full = paper_text(row)
    if len(full) <= chunk_chars:
        return [full]
    step = max(1, chunk_chars - overlap_chars)
    return [full[start:start + chunk_chars] for start in range(0, len(full), step)]


def _json_list(value: str) -> list[str]:
    try:
        data = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    return [str(data)] if str(data).strip() else []


def _truncate_utf8(value: Any, max_bytes: int) -> str:
    text = str(value or "")
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def iter_papers(db_path: str, *, after_row_id: int = 0, limit: int | None = None) -> Iterable[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = """SELECT row_id,paper_id,title,abstract,conference,year,
                         authors_json,keywords_json,subjects_json
                  FROM papers WHERE row_id > ? ORDER BY row_id"""
        params: list[Any] = [after_row_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in conn.execute(sql, params):
            yield row
    finally:
        conn.close()


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        import requests  # type: ignore
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except ImportError:  # pragma: no cover - requests is present in the current image
        import urllib.request
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


class EmbeddingClient:
    def __init__(self, url: str, api_key: str, model: str, timeout: float = 60.0, retries: int = 3, dimensions: int | None = None):
        # Accept either a complete endpoint or an OpenAI-compatible `/v1` base.
        normalized_url = url.rstrip("/")
        if normalized_url.endswith("/v1"):
            normalized_url += "/embeddings"
        self.url, self.api_key, self.model, self.dimensions = normalized_url, api_key, model, dimensions
        self.timeout, self.retries = timeout, retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = int(self.dimensions)
        instruction = os.getenv("QWEN_EMBEDDING_INSTRUCTION", "").strip()
        if instruction:
            payload["instruction"] = instruction
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                body = _post_json(self.url, payload, self.api_key, self.timeout)
                # Support both OpenAI-compatible responses
                # (``data:[{index,embedding}]``) and DashScope-style Qwen
                # responses (``output.embeddings:[{text_index,embedding}]``).
                data = body.get("data")
                if isinstance(data, list):
                    data = sorted(data, key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0)
                    vectors = [x.get("embedding") if isinstance(x, dict) else None for x in data]
                else:
                    output = body.get("output") if isinstance(body, dict) else None
                    data = output.get("embeddings") if isinstance(output, dict) else None
                    if not isinstance(data, list):
                        raise ValueError("embedding response missing data/ output.embeddings list")
                    data = sorted(data, key=lambda x: x.get("text_index", 0) if isinstance(x, dict) else 0)
                    vectors = [x.get("embedding") if isinstance(x, dict) else None for x in data]
                if len(vectors) != len(texts) or any(not isinstance(v, list) or not v for v in vectors):
                    raise ValueError("embedding response length/dimension mismatch")
                return vectors
            except Exception as exc:  # retry transient API failures
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"embedding API failed after {self.retries} attempts: {last_error}")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_row_id": 0, "dimension": None, "processed": 0, "failed": 0}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {"last_row_id": 0, "dimension": None, "processed": 0, "failed": 0}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _zilliz_client(uri: str, token: str):
    try:
        from pymilvus import MilvusClient  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pymilvus is required for upload; install it in the service environment") from exc
    return MilvusClient(uri=uri, token=token)


def ensure_collection(client: Any, name: str, dimension: int) -> None:
    from pymilvus import DataType  # type: ignore
    if client.has_collection(collection_name=name):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=768, is_primary=True)
    schema.add_field("paper_id", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=4096)
    schema.add_field("conference", DataType.VARCHAR, max_length=512)
    schema.add_field("year", DataType.INT64)
    schema.add_field("has_abstract", DataType.BOOL)
    schema.add_field("text_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def upload(args: argparse.Namespace) -> int:
    db = args.db or os.getenv("PAPER_SEARCH_DB", "papers_fts.db")
    state_path = Path(args.state)
    state = _load_state(state_path) if args.resume else {"last_row_id": 0, "dimension": None, "processed": 0, "failed": 0}
    url = os.getenv("QWEN_EMBEDDING_URL", "").strip()
    key = os.getenv("QWEN_EMBEDDING_API_KEY", "").strip()
    model = os.getenv("QWEN_EMBEDDING_MODEL", "qwen3-embedding-8b").strip()
    if not url or not key:
        raise SystemExit("QWEN_EMBEDDING_URL and QWEN_EMBEDDING_API_KEY are required (keep them in an untracked env file)")
    requested_dimensions = int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "0")) or None
    client = EmbeddingClient(url, key, model, timeout=args.timeout, dimensions=requested_dimensions)
    zclient = _zilliz_client(os.environ["ZILLIZ_URI"], os.environ["ZILLIZ_TOKEN"])
    failed_path = Path(args.failed)
    batch: list[tuple[sqlite3.Row, int, str]] = []
    pending_papers: list[int] = []
    total_seen = 0
    for row in iter_papers(db, after_row_id=int(state.get("last_row_id", 0)), limit=args.limit):
        chunks = paper_chunks(row)
        paper_rows = [(row, i, text) for i, text in enumerate(chunks)]
        # Keep all chunks of a paper together when possible, so its checkpoint
        # advances atomically. Flush complete papers in API-sized batches.
        if batch and len(batch) + len(paper_rows) > args.batch_size:
            _process_batch(batch, client, zclient, args.collection, state, state_path, failed_path, advance_checkpoint=False)
            state["last_row_id"] = max(pending_papers)
            _save_state(state_path, state)
            batch = []
            pending_papers = []
        if len(paper_rows) <= args.batch_size:
            batch.extend(paper_rows)
            pending_papers.append(int(row["row_id"]))
        else:
            # Very long papers may contain more chunks than one API request.
            # Process all their chunks, then advance the checkpoint only after
            # the complete paper has been uploaded successfully.
            for start in range(0, len(paper_rows), args.batch_size):
                _process_batch(paper_rows[start:start + args.batch_size], client, zclient, args.collection, state, state_path, failed_path, advance_checkpoint=False)
            state["last_row_id"] = int(row["row_id"])
            _save_state(state_path, state)
        total_seen += 1
    if batch:
        _process_batch(batch, client, zclient, args.collection, state, state_path, failed_path, advance_checkpoint=False)
        state["last_row_id"] = max(pending_papers)
        _save_state(state_path, state)
    print(json.dumps({"processed": state["processed"], "failed": state["failed"], "last_row_id": state["last_row_id"], "seen_this_run": total_seen}, ensure_ascii=False))
    return 0


def _process_batch(rows: list[tuple[sqlite3.Row, int, str]], embedder: EmbeddingClient, zclient: Any, collection: str, state: dict[str, Any], state_path: Path, failed_path: Path, advance_checkpoint: bool = True) -> None:
    texts = [text for _row, _idx, text in rows]
    succeeded = False
    try:
        vectors = embedder.embed(texts)
        dim = len(vectors[0])
        if state.get("dimension") not in (None, dim):
            raise ValueError(f"embedding dimension changed: {state.get('dimension')} -> {dim}")
        ensure_collection(zclient, collection, dim)
        entities = []
        for (row, chunk_index, text), vector in zip(rows, vectors):
            entities.append({
                "chunk_id": f"{row['paper_id']}#chunk:{chunk_index}", "paper_id": row["paper_id"], "chunk_index": chunk_index,
                "title": _truncate_utf8(row["title"], 4096),
                "conference": _truncate_utf8(row["conference"], 512),
                "year": int(row["year"] or 0), "has_abstract": bool(str(row["abstract"] or "").strip()),
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "embedding": vector,
            })
        zclient.upsert(collection_name=collection, data=entities)
        state["dimension"] = dim
        state["processed"] += len(rows)
        succeeded = True
    except Exception as exc:
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("a", encoding="utf-8") as fh:
            for row, chunk_index, _text in rows:
                fh.write(json.dumps({"row_id": row["row_id"], "paper_id": row["paper_id"], "chunk_index": chunk_index, "error": str(exc)}, ensure_ascii=False) + "\n")
        state["failed"] += len(rows)
        # Do not advance the resumable checkpoint on a failed batch. Exiting
        # here makes `--resume` retry the exact batch instead of silently
        # skipping papers after a transient API/Zilliz outage.
        _save_state(state_path, state)
        raise RuntimeError(f"batch failed at row_id={rows[0][0]['row_id']}; checkpoint not advanced: {exc}") from exc
    finally:
        if succeeded:
            if advance_checkpoint:
                state["last_row_id"] = int(rows[-1][0]["row_id"])
            _save_state(state_path, state)


def dry_run(args: argparse.Namespace) -> int:
    db = args.db or os.getenv("PAPER_SEARCH_DB", "papers_fts.db")
    rows = list(iter_papers(db, limit=args.limit))
    for row in rows[: args.show]:
        text = paper_text(row)
        print(json.dumps({"row_id": row["row_id"], "paper_id": row["paper_id"], "chars": len(text), "text_hash": hashlib.sha256(text.encode()).hexdigest(), "preview": text[:300]}, ensure_ascii=False))
    print(json.dumps({"rows": len(rows), "db": str(Path(db).resolve())}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite papers_fts.db path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    parser.add_argument("--collection", default=os.getenv("ZILLIZ_COLLECTION", "papers_qwen3_embedding_8b"))
    parser.add_argument("--state", default=os.getenv("VECTOR_INGEST_STATE", "data/vector_ingest_state.json"))
    parser.add_argument("--failed", default=os.getenv("VECTOR_INGEST_FAILED", "data/vector_ingest_failed.jsonl"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("QWEN_EMBEDDING_TIMEOUT", "60")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 128:
        parser.error("--batch-size must be between 1 and 128")
    return dry_run(args) if args.dry_run else upload(args)


if __name__ == "__main__":
    sys.exit(main())
