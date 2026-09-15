from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRIC_SCOPE = "StableToolBench raw answer generation; prerequisite for answer conversion and SoPR."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_status(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def _dir_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_dir(),
    }


def _path_for_command(path: Path, command_root: Path) -> Path:
    return path if path.is_absolute() else command_root / path


def _tool_json_files(tool_root_dir: Path) -> list[Path]:
    if not tool_root_dir.is_dir():
        return []
    return sorted(path for path in tool_root_dir.glob("*/*.json") if path.is_file())


def _is_data_example_path(path: Path) -> bool:
    return any(part.lower() in {"data_example", "solvable_queries_example"} for part in path.parts)


def _provided(value: str | None) -> bool:
    return bool(str(value or "").strip())


def _raw_generation_command(
    *,
    stabletoolbench_root: Path,
    tool_root_dir: Path,
    input_query_file: Path,
    output_answer_file: Path,
    backbone_model: str,
    chatgpt_model: str,
    model_path: str,
    method: str,
    service_url: str,
    toolbench_key_provided: bool,
    base_url: str,
    max_observation_length: int,
    single_chain_max_step: int,
    max_query_count: int,
    num_thread: int,
) -> list[str]:
    command = [
        "cd",
        str(stabletoolbench_root),
        "&&",
        f"SERVICE_URL={service_url}",
        "python",
        "toolbench/inference/qa_pipeline_multithread.py",
        "--tool_root_dir",
        str(tool_root_dir),
        "--backbone_model",
        backbone_model,
        "--chatgpt_model",
        chatgpt_model,
        "--base_url",
        base_url,
        "--openai_key",
        "${OPENAI_KEY}",
        "--model_path",
        model_path,
        "--max_observation_length",
        str(max_observation_length),
        "--single_chain_max_step",
        str(single_chain_max_step),
        "--max_query_count",
        str(max_query_count),
        "--method",
        method,
        "--input_query_file",
        str(input_query_file),
        "--output_answer_file",
        str(output_answer_file),
        "--toolbench_key",
        "${TOOLBENCH_KEY}" if toolbench_key_provided else "",
        "--num_thread",
        str(num_thread),
    ]
    return command


def audit_stabletoolbench_raw_generation(
    *,
    stabletoolbench_root: str | Path,
    tool_root_dir: str | Path,
    raw_answer_path: str | Path,
    candidate_model: str,
    test_set: str = "G3_instruction",
    input_query_file: str | Path | None = None,
    method: str = "CLSTR@1",
    backbone_model: str = "chatgpt_function",
    openai_key: str | None = None,
    model_path: str | Path | None = None,
    service_url: str | None = None,
    toolbench_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    chatgpt_model: str = "gpt-4-turbo-2024-04-09",
    max_observation_length: int = 1024,
    single_chain_max_step: int = 50,
    max_query_count: int = 200,
    num_thread: int = 1,
    command_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    stabletoolbench_root = Path(stabletoolbench_root)
    tool_root_dir = Path(tool_root_dir)
    raw_answer_path = Path(raw_answer_path)
    command_root = Path.cwd() if command_root is None else Path(command_root)
    model_path_text = str(model_path or "")
    service_url_text = str(service_url or "")

    qa_pipeline = stabletoolbench_root / "toolbench" / "inference" / "qa_pipeline_multithread.py"
    official_input_query_file = stabletoolbench_root / "solvable_queries" / "test_instruction" / f"{test_set}.json"
    input_query_file = Path(input_query_file) if input_query_file is not None else official_input_query_file
    output_answer_file = raw_answer_path / candidate_model / test_set
    tool_files = _tool_json_files(tool_root_dir)
    uses_official_full_api_list = input_query_file == official_input_query_file
    input_query_source = "official" if uses_official_full_api_list else "derived"

    blockers: list[str] = []
    if not stabletoolbench_root.is_dir():
        blockers.append("missing_stabletoolbench_root")
    if not qa_pipeline.is_file():
        blockers.append("missing_qa_pipeline_multithread_py")
    if not input_query_file.is_file():
        blockers.append("missing_input_query_file")
    if not tool_root_dir.is_dir():
        blockers.append("missing_tool_root_dir")
    elif not tool_files:
        blockers.append("no_tool_json_files")
    if _is_data_example_path(tool_root_dir):
        blockers.append("tool_root_dir_is_data_example_not_official")
    if not _provided(service_url_text):
        blockers.append("missing_service_url_for_stabletoolbench_virtual_api")

    if backbone_model == "chatgpt_function":
        if not _provided(openai_key):
            blockers.append("missing_openai_key_for_chatgpt_function")
    elif backbone_model in {"toolllama", "toolllama_lora", "toolllama_vllm"}:
        if not _provided(model_path_text):
            blockers.append("missing_model_path_for_toolllama")
        elif not Path(model_path_text).exists():
            blockers.append("missing_model_path")
    elif backbone_model == "davinci":
        if not _provided(openai_key):
            blockers.append("missing_openai_key_for_davinci")

    report = {
        "status": "ok" if not blockers else "action_required",
        "can_generate_raw_answers": not blockers,
        "blockers": blockers,
        "stabletoolbench_root": str(stabletoolbench_root),
        "tool_root_dir": str(tool_root_dir),
        "raw_answer_path": str(raw_answer_path),
        "output_answer_file": str(output_answer_file),
        "input_query_file": str(input_query_file),
        "official_input_query_file": str(official_input_query_file),
        "input_query_source": input_query_source,
        "uses_official_full_api_list": uses_official_full_api_list,
        "candidate_model": candidate_model,
        "test_set": test_set,
        "method": method,
        "backbone_model": backbone_model,
        "chatgpt_model": chatgpt_model,
        "base_url": base_url,
        "service_url": service_url_text,
        "model_path": model_path_text,
        "openai_key_provided": _provided(openai_key),
        "toolbench_key_provided": _provided(toolbench_key),
        "max_observation_length": max_observation_length,
        "single_chain_max_step": single_chain_max_step,
        "max_query_count": max_query_count,
        "num_thread": num_thread,
        "command_root": str(command_root),
        "metric_scope": METRIC_SCOPE,
        "tool_json_file_count": len(tool_files),
        "tool_json_file_samples": [str(path) for path in tool_files[:5]],
        "required_paths": {
            "qa_pipeline_multithread.py": _file_status(qa_pipeline),
            "input_query_file": _file_status(input_query_file),
            "tool_root_dir": _dir_status(tool_root_dir),
            "output_answer_parent": _dir_status(output_answer_file.parent),
        },
        "reproducible_command": _raw_generation_command(
            stabletoolbench_root=stabletoolbench_root,
            tool_root_dir=_path_for_command(tool_root_dir, command_root),
            input_query_file=input_query_file,
            output_answer_file=_path_for_command(output_answer_file, command_root),
            backbone_model=backbone_model,
            chatgpt_model=chatgpt_model,
            model_path=model_path_text,
            method=method,
            service_url=service_url_text or "${SERVICE_URL}",
            toolbench_key_provided=_provided(toolbench_key),
            base_url=base_url,
            max_observation_length=max_observation_length,
            single_chain_max_step=single_chain_max_step,
            max_query_count=max_query_count,
            num_thread=num_thread,
        ),
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report
