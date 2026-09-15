#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any, Callable


def sanitize_openai_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a standards-compliant copy of a legacy ToolBench tool schema."""

    sanitized = copy.deepcopy(tool)
    function = sanitized.get("function")
    if not isinstance(function, dict):
        return sanitized
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return sanitized
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    required: list[str] = []
    for raw_name in parameters.get("required") or []:
        name = str(raw_name).strip()
        if name and name in properties and name not in required:
            required.append(name)
    parameters["required"] = required
    parameters.pop("optional", None)
    for raw_schema in properties.values():
        if not isinstance(raw_schema, dict):
            continue
        if "example_value" in raw_schema and "default" not in raw_schema:
            raw_schema["default"] = raw_schema["example_value"]
        raw_schema.pop("example_value", None)
    return sanitized


def build_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    stop: Any = None,
    temperature: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_messages: list[dict[str, Any]] = []
    for raw_message in messages:
        if raw_message.get("valid", True) is False:
            continue
        message = dict(raw_message)
        message.pop("function_call", None)
        clean_messages.append(message)
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": clean_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    payload.update(dict(extra or {}))
    # The release protocol is deterministic even if an upstream caller forwards
    # its own sampling kwargs through **extra.
    payload["temperature"] = float(temperature)
    if stop is not None:
        payload["stop"] = stop
    if tools is not None:
        payload["tools"] = [sanitize_openai_tool_schema(item) for item in tools]
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def make_bounded_chat_completion_request(
    *,
    temperature: float,
    max_attempts: int,
    initial_backoff_seconds: float,
    client_factory: Callable[..., Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    attempts = max(1, int(max_attempts))
    initial_backoff = max(0.0, float(initial_backoff_seconds))

    def request(
        key,
        base_url,
        messages,
        tools=None,
        tool_choice=None,
        key_pos=None,
        model="gpt-3.5-turbo",
        stop=None,
        process_id=0,
        **extra,
    ):
        del key_pos
        factory = client_factory
        if factory is None:
            from openai import OpenAI

            factory = OpenAI
        payload = build_completion_payload(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            temperature=temperature,
            extra=extra,
        )
        last_error_type = ""
        for attempt in range(1, attempts + 1):
            try:
                client = factory(base_url=base_url, api_key=key) if base_url else factory(api_key=key)
                response = client.chat.completions.create(**payload)
                if hasattr(response, "model_dump"):
                    return dict(response.model_dump())
                if hasattr(response, "dict"):
                    return dict(response.dict())
                return dict(response)
            except Exception as exc:  # bounded infrastructure isolation
                last_error_type = type(exc).__name__
                print(
                    json.dumps(
                        {
                            "event": "stabletoolbench_executor_retry",
                            "process_id": int(process_id),
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "exception_type": last_error_type,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if attempt < attempts and initial_backoff > 0:
                    time.sleep(initial_backoff * (2 ** (attempt - 1)))
        return {
            "error": (
                "StableToolBench executor request failed after bounded retries "
                f"({last_error_type or 'unknown_error'})"
            ),
            "total_tokens": 0,
        }

    return request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run StableToolBench generation with valid JSON schemas, deterministic "
            "executor decoding, and bounded non-interactive retries."
        )
    )
    parser.add_argument("--tool_root_dir", required=True)
    parser.add_argument("--backbone_model", default="chatgpt_function")
    parser.add_argument("--chatgpt_model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--openai_key", default="EMPTY")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_path", default="")
    parser.add_argument("--max_observation_length", type=int, default=1024)
    parser.add_argument("--max_source_sequence_length", type=int, default=4096)
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--single_chain_max_step", type=int, default=50)
    parser.add_argument("--max_query_count", type=int, default=100000)
    parser.add_argument(
        "--observ_compress_method",
        choices=("truncate", "filter", "random"),
        default="truncate",
    )
    parser.add_argument("--method", default="CoT@1")
    parser.add_argument("--input_query_file", required=True)
    parser.add_argument("--output_answer_file", required=True)
    parser.add_argument("--toolbench_key", default="")
    parser.add_argument("--rapidapi_key", default="")
    parser.add_argument("--use_rapidapi_key", action="store_true")
    parser.add_argument("--api_customization", action="store_true")
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--corpus_tsv_path", default="")
    parser.add_argument("--retrieval_model_path", default="")
    parser.add_argument("--executor_temperature", type=float, default=0.0)
    parser.add_argument("--request_max_attempts", type=int, default=3)
    parser.add_argument("--request_initial_backoff_seconds", type=float, default=1.0)
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    if args.overwrite:
        import shutil

        shutil.rmtree(args.output_answer_file, ignore_errors=True)
    from toolbench.inference.Downstream_tasks import rapidapi_multithread
    from toolbench.inference.LLM import chatgpt_function_model

    original_schema_builder = rapidapi_multithread.rapidapi_wrapper.api_json_to_openai_json

    def valid_schema_builder(self, api_json, standard_tool_name):
        schema, category_name, api_name = original_schema_builder(
            self, api_json, standard_tool_name
        )
        return sanitize_openai_tool_schema(schema), category_name, api_name

    rapidapi_multithread.rapidapi_wrapper.api_json_to_openai_json = valid_schema_builder
    chatgpt_function_model.chat_completion_request = make_bounded_chat_completion_request(
        temperature=float(args.executor_temperature),
        max_attempts=int(args.request_max_attempts),
        initial_backoff_seconds=float(args.request_initial_backoff_seconds),
    )
    rapidapi_multithread.pipeline_runner(args).run()


def main() -> int:
    run_from_args(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
