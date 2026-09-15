from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from scripts.run_stabletoolbench_generation import (
    build_completion_payload,
    make_bounded_chat_completion_request,
    sanitize_openai_tool_schema,
)
from scripts.run_stabletoolbench_mirrorapi_server import (
    _build_runtime_app,
    _request_cache_key,
)


class _Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: str | dict
    strip: str
    toolbench_key: str


def _tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "example_value": "Paris",
                    }
                },
                "required": ["", "city", "missing", "city"],
                "optional": [""],
            },
        },
    }


def test_schema_sanitizer_and_temperature_lock() -> None:
    sanitized = sanitize_openai_tool_schema(_tool_schema())
    parameters = sanitized["function"]["parameters"]
    assert parameters["required"] == ["city"]
    assert "optional" not in parameters
    assert parameters["properties"]["city"]["default"] == "Paris"
    assert "example_value" not in parameters["properties"]["city"]
    payload = build_completion_payload(
        model="executor",
        messages=[
            {"role": "user", "content": "safe"},
            {"role": "user", "content": "drop", "valid": False},
        ],
        tools=[_tool_schema()],
        temperature=0.0,
        extra={"temperature": 0.9},
    )
    assert payload["temperature"] == 0.0
    assert len(payload["messages"]) == 1
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["city"]


def test_bounded_completion_retries_without_secret_leak() -> None:
    calls = {"count": 0}

    class _Completions:
        def create(self, **_kwargs):
            calls["count"] += 1
            raise RuntimeError("PRIVATE_EXECUTOR_SECRET")

    def factory(**_kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    request = make_bounded_chat_completion_request(
        temperature=0.0,
        max_attempts=2,
        initial_backoff_seconds=0.0,
        client_factory=factory,
    )
    captured = io.StringIO()
    with redirect_stdout(captured):
        result = request(
            "PRIVATE_API_KEY",
            "https://provider.invalid/v1",
            [{"role": "user", "content": "hello"}],
        )
    assert calls["count"] == 2
    assert result["total_tokens"] == 0
    output = captured.getvalue()
    assert "PRIVATE_EXECUTOR_SECRET" not in output
    assert "PRIVATE_API_KEY" not in output


def _endpoint(app):
    return next(route.endpoint for route in app.routes if route.path == "/virtual")


def test_mirror_cache_missing_doc_and_trace_redaction(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    category = tools / "weather"
    category.mkdir(parents=True)
    (category / "forecast.json").write_text(
        json.dumps(
            {
                "tool_description": "weather tool",
                "api_list": [
                    {
                        "name": "today",
                        "description": "today's forecast",
                        "required_parameters": [],
                        "optional_parameters": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    simulator_calls = {"count": 0}
    simulator_data: list[dict] = []

    def prepare(info):
        return info.tool_name, info.category, info.api_name, ""

    def simulate(_tool_input, data, _api_doc):
        simulator_calls["count"] += 1
        simulator_data.append(dict(data))
        return {"error": "", "response": "sunny"}

    namespace = {
        "Info": _Info,
        "prepare_tool_name_and_url": prepare,
        "standardize": lambda value: str(value).lower().replace(" ", "_"),
        "fake_response_function_with_trained_simulator": simulate,
    }
    trace_path = tmp_path / "trace.jsonl"
    app = _build_runtime_app(
        namespace=namespace,
        config={"tools_folder": str(tools)},
        trace_path=trace_path,
        max_attempts=3,
        initial_backoff_seconds=0.0,
    )
    endpoint = _endpoint(app)
    first = _Info(
        category="weather",
        tool_name="forecast",
        api_name="today",
        tool_input={"city": "PRIVATE_CITY"},
        strip="",
        toolbench_key="PRIVATE_TOOLBENCH_KEY_A",
    )
    second = first.copy(update={"toolbench_key": "PRIVATE_TOOLBENCH_KEY_B"})
    assert _request_cache_key(first) == _request_cache_key(second)
    assert endpoint(first)["response"] == "sunny"
    assert endpoint(second)["response"] == "sunny"
    assert simulator_calls["count"] == 1
    assert "toolbench_key" not in simulator_data[0]

    missing = first.copy(update={"api_name": "missing"})
    error = endpoint(missing)
    assert error["response"] == ""
    assert "LookupError" in error["error"]
    trace = trace_path.read_text(encoding="utf-8")
    for secret in (
        "PRIVATE_CITY",
        "PRIVATE_TOOLBENCH_KEY_A",
        "PRIVATE_TOOLBENCH_KEY_B",
    ):
        assert secret not in trace
    rows = [json.loads(line) for line in trace.splitlines()]
    assert rows[-1]["status"] == "isolated_error"
    assert rows[-1]["attempt_count"] == 1
