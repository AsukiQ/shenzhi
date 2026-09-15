#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _request_cache_key(info: Any) -> str:
    payload = {
        "category": str(info.category),
        "tool_name": str(info.tool_name),
        "api_name": str(info.api_name),
        "tool_input": info.tool_input,
        "strip": str(info.strip),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_trace(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_api_doc(namespace: dict[str, Any], config: dict[str, Any], info: Any):
    tool_name, standard_category, api_name, _ = namespace[
        "prepare_tool_name_and_url"
    ](info)
    original_tool_name = namespace["standardize"](str(info.tool_name))
    tool_path = (
        Path(config["tools_folder"])
        / standard_category
        / f"{original_tool_name.split('_for_')[0]}.json"
    )
    if not tool_path.is_file():
        raise LookupError(f"tool_document_not_found:{standard_category}/{tool_path.name}")
    tool_record = json.loads(tool_path.read_text(encoding="utf-8"))
    api_rows = [
        row
        for row in tool_record.get("api_list") or []
        if api_name == namespace["standardize"](str(row.get("name") or ""))
    ]
    if not api_rows:
        raise LookupError(
            f"api_document_not_found:{standard_category}/{tool_path.name}/{api_name}"
        )
    api_doc = {
        "tool_description": str(tool_record.get("tool_description") or ""),
        "api_info": api_rows,
    }
    return tool_name, standard_category, api_name, api_doc


def _simulate(namespace: dict[str, Any], config: dict[str, Any], info: Any) -> dict[str, Any]:
    tool_input = info.tool_input
    if isinstance(tool_input, str):
        if tool_input.strip():
            tool_input = json.loads(tool_input)
        else:
            tool_input = {}
    if not isinstance(tool_input, dict):
        raise ValueError("tool_input_must_be_json_object")
    _, _, prepared_api_name, _ = namespace["prepare_tool_name_and_url"](info)
    if prepared_api_name == "chat_with_user":
        return {"error": "", "response": "Chat with user."}
    _, standard_category, api_name, api_doc = _load_api_doc(
        namespace, config, info
    )
    original_tool_name = str(info.tool_name)
    real_tool_name = original_tool_name.split("_for_", 1)[0]
    data = {
        "category": standard_category,
        "tool_name": real_tool_name,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": "",
    }
    result = namespace["fake_response_function_with_trained_simulator"](
        tool_input, data, api_doc
    )
    if isinstance(result, dict):
        return result
    parsed = json.loads(result)
    if not isinstance(parsed, dict):
        raise ValueError("simulator_response_must_be_json_object")
    return parsed


def _build_runtime_app(
    *,
    namespace: dict[str, Any],
    config: dict[str, Any],
    trace_path: Path,
    max_attempts: int,
    initial_backoff_seconds: float,
):
    from fastapi import FastAPI

    app = FastAPI()
    info_type = namespace["Info"]
    cache: dict[str, dict[str, Any]] = {}
    cache_lock = threading.Lock()
    trace_lock = threading.Lock()
    attempts = max(1, int(max_attempts))
    initial_backoff = max(0.0, float(initial_backoff_seconds))

    def get_virtual_response(info):
        cache_key = _request_cache_key(info)
        with cache_lock:
            cached = cache.get(cache_key)
        if cached is not None:
            _append_trace(
                trace_path,
                {
                    "event": "mirrorapi_request",
                    "request_sha256": cache_key,
                    "category": str(info.category),
                    "tool_name": str(info.tool_name),
                    "api_name": str(info.api_name),
                    "cache_hit": True,
                    "status": "ok",
                    "attempt_count": 0,
                },
                trace_lock,
            )
            return dict(cached)

        last_error_type = ""
        attempt_count = 0
        for attempt in range(1, attempts + 1):
            attempt_count = attempt
            try:
                response = _simulate(namespace, config, info)
                with cache_lock:
                    cache[cache_key] = dict(response)
                _append_trace(
                    trace_path,
                    {
                        "event": "mirrorapi_request",
                        "request_sha256": cache_key,
                        "category": str(info.category),
                        "tool_name": str(info.tool_name),
                        "api_name": str(info.api_name),
                        "cache_hit": False,
                        "status": "ok",
                        "attempt_count": attempt,
                    },
                    trace_lock,
                )
                return response
            except LookupError as exc:
                last_error_type = type(exc).__name__
                break
            except Exception as exc:  # isolate provider/parser infrastructure faults
                last_error_type = type(exc).__name__
                if attempt < attempts and initial_backoff > 0:
                    time.sleep(initial_backoff * (2 ** (attempt - 1)))
        _append_trace(
            trace_path,
            {
                "event": "mirrorapi_request",
                "request_sha256": cache_key,
                "category": str(info.category),
                "tool_name": str(info.tool_name),
                "api_name": str(info.api_name),
                "cache_hit": False,
                "status": "isolated_error",
                "attempt_count": attempt_count,
                "exception_type": last_error_type or "unknown_error",
            },
            trace_lock,
        )
        return {
            "error": (
                "MirrorAPI simulator request failed after bounded retries "
                f"({last_error_type or 'unknown_error'})"
            ),
            "response": "",
        }

    # With postponed annotations, spelling ``info: info_type`` inside this local
    # scope leaves FastAPI an unresolvable string. Attach the real Pydantic class
    # before registering the route instead.
    get_virtual_response.__annotations__["info"] = info_type
    app.post("/virtual")(get_virtual_response)
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_root", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--tools_folder", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=12001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--initial_backoff_seconds", type=float, default=1.0)
    parser.add_argument("--trace_path")
    args = parser.parse_args()

    api_key = str(args.api_key)
    if api_key == "EMPTY":
        api_key = os.environ.get("MIRROR_SERVER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MirrorAPI requires --api_key or protected MIRROR_SERVER_API_KEY"
        )

    server_root = Path(args.server_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    tools_folder = Path(args.tools_folder).resolve()
    entrypoint = server_root / "main_mirrorapi.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    if not tools_folder.is_dir():
        raise FileNotFoundError(tools_folder)
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_path).resolve() if args.trace_path else work_dir / "requests.jsonl"
    config = {
        "api_key": api_key,
        "api_base": str(args.api_base),
        "temperature": float(args.temperature),
        "tools_folder": str(tools_folder),
        "port": int(args.port),
        "model": str(args.model),
    }
    config_path = work_dir / "config_mirrorapi.yml"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    os.chdir(work_dir)
    sys.path.insert(0, str(server_root))
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            namespace = runpy.run_path(
                str(entrypoint), run_name="stabletoolbench_mirrorapi_runtime"
            )
        app = _build_runtime_app(
            namespace=namespace,
            config=config,
            trace_path=trace_path,
            max_attempts=int(args.max_attempts),
            initial_backoff_seconds=float(args.initial_backoff_seconds),
        )
        print(
            json.dumps(
                {
                    "event": "mirrorapi_start",
                    "model": str(args.model),
                    "port": int(args.port),
                    "tools_folder": str(tools_folder),
                    "temperature": float(args.temperature),
                    "credentials_redacted": True,
                    "trace_path": str(trace_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=int(args.port))
    finally:
        config_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
