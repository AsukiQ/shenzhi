from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_example_path(path: Path) -> bool:
    return any(part.lower() in {"data_example", "solvable_queries_example"} for part in path.parts)


def _standardize_category(category: str) -> str:
    text = str(category).replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in text or "," in text:
        text = text.replace(" ", "_").replace(",", "_")
    return text.replace("__", "_")


def _standardize(value: str) -> str:
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", str(value))
    text = re.sub(r"(_)\1+", "_", text).lower().strip("_")
    if text and text[0].isdigit():
        text = f"get_{text}"
    return text


def _change_name(value: str) -> str:
    return f"is_{value}" if value in {"from", "class", "return", "false", "true", "id", "and"} else value


def _api_key(api: dict[str, Any]) -> str:
    return _change_name(_standardize(str(api.get("api_name") or api.get("name") or "")))


def _tool_description(tool_name: str, apis: list[dict[str, Any]]) -> str:
    for api in apis:
        description = str(api.get("api_description") or api.get("description") or "").strip()
        if description:
            return description
    return str(tool_name)


def _tool_record(tool_name: str, apis: list[dict[str, Any]]) -> dict[str, Any]:
    api_records = []
    seen_api_names: set[str] = set()
    for api in apis:
        key = _api_key(api)
        if not key or key in seen_api_names:
            continue
        seen_api_names.add(key)
        api_records.append(
            {
                "name": str(api.get("api_name") or api.get("name") or key),
                "description": str(api.get("api_description") or api.get("description") or ""),
                "required_parameters": list(api.get("required_parameters") or []),
                "optional_parameters": list(api.get("optional_parameters") or []),
                "method": api.get("method", "GET"),
                **({"template_response": api["template_response"]} if "template_response" in api else {}),
            }
        )
    return {
        "tool_name": str(tool_name),
        "tool_description": _tool_description(tool_name, apis),
        "api_list": api_records,
    }


def build_stabletoolbench_toolenv_from_queries(
    *,
    query_file: str | Path,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    query_file = Path(query_file)
    output_root = Path(output_root)
    manifest_path = Path(manifest_path) if manifest_path is not None else output_root.parent / "toolenv_manifest.json"

    blockers: list[str] = []
    if not query_file.is_file():
        blockers.append("missing_query_file")
    if _is_example_path(query_file):
        blockers.append("query_file_is_example_not_official")
    if _is_example_path(output_root):
        blockers.append("output_root_is_example_not_official")

    if blockers:
        report = {
            "status": "action_required",
            "blockers": blockers,
            "query_file": str(query_file),
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "query_count": 0,
            "tool_count": 0,
            "api_count": 0,
            "written_tool_files": 0,
            "metric_scope": "StableToolBench G3 toolenv subset construction; prerequisite for raw answer generation.",
        }
        if manifest_path is not None:
            _write_json(manifest_path, report)
        return report

    queries = json.loads(query_file.read_text(encoding="utf-8"))
    tools: "OrderedDict[tuple[str, str], list[dict[str, Any]]]" = OrderedDict()
    api_ref_count = 0
    for row in queries:
        for api in row.get("api_list") or []:
            category = str(api.get("category_name") or "Unknown")
            tool_name = str(api.get("tool_name") or "unknown")
            tools.setdefault((category, tool_name), []).append(api)
            api_ref_count += 1

    written = 0
    api_count = 0
    tool_file_samples: list[str] = []
    for (category, tool_name), apis in tools.items():
        category_dir = output_root / _standardize_category(category)
        tool_path = category_dir / f"{_standardize(tool_name)}.json"
        record = _tool_record(tool_name, apis)
        api_count += len(record["api_list"])
        _write_json(tool_path, record)
        written += 1
        if len(tool_file_samples) < 5:
            tool_file_samples.append(str(tool_path))

    report = {
        "status": "ok",
        "blockers": [],
        "query_file": str(query_file),
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "query_count": len(queries),
        "api_ref_count": api_ref_count,
        "tool_count": len(tools),
        "api_count": api_count,
        "written_tool_files": written,
        "tool_file_samples": tool_file_samples,
        "metric_scope": "StableToolBench G3 toolenv subset construction; prerequisite for raw answer generation.",
    }
    _write_json(manifest_path, report)
    return report
