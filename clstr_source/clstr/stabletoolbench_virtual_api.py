from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


METRIC_SCOPE = "StableToolBench virtual API readiness for official ToolBench-G3 raw generation."

MODE_CONFIG = {
    "mirrorapi": {
        "script": "main_mirrorapi.py",
        "config": "config_mirrorapi.yml",
    },
    "mirrorapi_cache": {
        "script": "main_mirrorapi_cache.py",
        "config": "config_mirrorapi_cache.yml",
    },
    "gpt_cache": {
        "script": "main.py",
        "config": "config.yml",
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_status(path: Path) -> dict[str, Any]:
    try:
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        error = None
    except OSError as exc:
        exists = False
        size = 0
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "error": error,
    }


def _dir_status(path: Path) -> dict[str, Any]:
    try:
        exists = path.is_dir()
        error = None
    except OSError as exc:
        exists = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "exists": exists,
        "error": error,
    }


def _provided(value: Any) -> bool:
    return bool(str(value or "").strip())


def _is_data_example_path(path: Path) -> bool:
    return any(part.lower() in {"data_example", "solvable_queries_example"} for part in path.parts)


def _resolve_path(path_text: Any, *, base_dir: Path) -> Path | None:
    if not _provided(path_text):
        return None
    path = Path(str(path_text))
    return path if path.is_absolute() else base_dir / path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _tool_json_files(tool_root_dir: Path | None) -> list[Path]:
    if tool_root_dir is None:
        return []
    try:
        if not tool_root_dir.is_dir():
            return []
        return sorted(path for path in tool_root_dir.glob("*/*.json") if path.is_file())
    except OSError:
        return []


def _load_yaml_config(config_path: Path) -> tuple[dict[str, Any], str | None]:
    if not config_path.is_file():
        return {}, None
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {}, f"invalid_yaml_config: {exc}"
    if not isinstance(payload, dict):
        return {}, "yaml_config_must_be_mapping"
    return payload, None


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(config)
    if _provided(redacted.get("api_key")) and redacted.get("api_key") != "EMPTY":
        redacted["api_key"] = "${API_KEY}"
    return redacted


def _server_command(server_dir: Path, server_script: str) -> list[str]:
    return ["cd", str(server_dir), "&&", "python", server_script]


def audit_stabletoolbench_virtual_api(
    *,
    stabletoolbench_root: str | Path,
    mode: str = "mirrorapi",
    autodl_tmp_root: str | Path = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    stabletoolbench_root = Path(stabletoolbench_root)
    autodl_tmp_root = Path(autodl_tmp_root)

    mode_spec = MODE_CONFIG.get(mode)
    server_dir = stabletoolbench_root / "server"
    server_script = server_dir / mode_spec["script"] if mode_spec else server_dir / "<unknown>"
    config_path = server_dir / mode_spec["config"] if mode_spec else server_dir / "<unknown>"
    config, config_error = _load_yaml_config(config_path)
    tools_folder = _resolve_path(config.get("tools_folder"), base_dir=server_dir)
    cache_folder = _resolve_path(config.get("cache_folder"), base_dir=server_dir)
    log_file = _resolve_path(config.get("log_file"), base_dir=server_dir)
    port = config.get("port")
    service_url = f"http://127.0.0.1:{port}/virtual" if _provided(port) else ""
    tool_files = _tool_json_files(tools_folder)

    blockers: list[str] = []
    if mode_spec is None:
        blockers.append("unsupported_virtual_api_mode")
    if not stabletoolbench_root.is_dir():
        blockers.append("missing_stabletoolbench_root")
    if not server_dir.is_dir():
        blockers.append("missing_server_dir")
    if not server_script.is_file():
        blockers.append("missing_server_script")
    if not config_path.is_file():
        blockers.append("missing_server_config")
    if config_error:
        blockers.append(config_error.split(":", 1)[0])
    if not _provided(port):
        blockers.append("missing_port")

    if tools_folder is None:
        blockers.append("missing_tools_folder_config")
    else:
        if not _inside(tools_folder, autodl_tmp_root):
            blockers.append("tools_folder_outside_autodl_tmp")
        if _is_data_example_path(tools_folder):
            blockers.append("tools_folder_is_data_example_not_official")
        try:
            tools_folder_exists = tools_folder.is_dir()
        except OSError:
            tools_folder_exists = False
        if not tools_folder_exists:
            blockers.append("missing_tools_folder")
        elif not tool_files:
            blockers.append("no_tool_json_files")

    if mode in {"mirrorapi", "mirrorapi_cache"}:
        if not _provided(config.get("api_base")):
            blockers.append("missing_api_base_for_mirrorapi")
        if not _provided(config.get("api_key")):
            blockers.append("missing_api_key_for_mirrorapi")
        if not _provided(config.get("model")):
            blockers.append("missing_model_for_mirrorapi")
        if not _provided(config.get("temperature")):
            blockers.append("missing_temperature_for_mirrorapi")
    elif mode == "gpt_cache":
        if not _provided(config.get("api_key")):
            blockers.append("missing_api_key_for_gpt_cache")
        if not _provided(config.get("model")):
            blockers.append("missing_model_for_gpt_cache")
        if not _provided(config.get("toolbench_url")):
            blockers.append("missing_toolbench_url_for_gpt_cache")
        if cache_folder is None:
            blockers.append("missing_cache_folder_config")
        if log_file is None:
            blockers.append("missing_log_file_config")

    report = {
        "status": "ok" if not blockers else "action_required",
        "can_start_virtual_api": not blockers,
        "blockers": blockers,
        "stabletoolbench_root": str(stabletoolbench_root),
        "mode": mode,
        "server_dir": str(server_dir),
        "server_script": str(server_script),
        "config_path": str(config_path),
        "service_url": service_url,
        "port": port,
        "metric_scope": METRIC_SCOPE,
        "autodl_tmp_root": str(autodl_tmp_root),
        "resolved_config": {
            "tools_folder": str(tools_folder) if tools_folder is not None else "",
            "cache_folder": str(cache_folder) if cache_folder is not None else "",
            "log_file": str(log_file) if log_file is not None else "",
            "api_base": str(config.get("api_base") or ""),
            "model": str(config.get("model") or ""),
            "temperature": config.get("temperature"),
            "toolbench_url": str(config.get("toolbench_url") or ""),
            "api_key_provided": _provided(config.get("api_key")),
        },
        "config": _redacted_config(config),
        "tool_json_file_count": len(tool_files),
        "tool_json_file_samples": [str(path) for path in tool_files[:5]],
        "required_paths": {
            "server_dir": _dir_status(server_dir),
            "server_script": _file_status(server_script),
            "server_config": _file_status(config_path),
            "tools_folder": _dir_status(tools_folder) if tools_folder is not None else {"path": "", "exists": False},
            "cache_folder": _dir_status(cache_folder) if cache_folder is not None else {"path": "", "exists": False},
            "log_file_parent": _dir_status(log_file.parent) if log_file is not None else {"path": "", "exists": False},
        },
        "reproducible_server_command": _server_command(server_dir, mode_spec["script"]) if mode_spec else [],
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report
