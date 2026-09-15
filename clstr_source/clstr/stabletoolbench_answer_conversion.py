from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


METRIC_SCOPE = "StableToolBench answer format conversion; prerequisite for SoPR, not pass-rate."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _path_for_command(path: Path, command_root: Path) -> Path:
    return path if path.is_absolute() else command_root / path


def _file_status(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def _dir_status(path: Path) -> dict[str, Any]:
    exists = path.is_dir()
    return {
        "path": str(path),
        "exists": exists,
    }


def _raw_answer_files(answer_dir: Path, method: str) -> tuple[list[Path], list[Path]]:
    if not answer_dir.is_dir():
        return [], []
    raw_files = sorted(path for path in answer_dir.iterdir() if path.is_file() and path.suffix == ".json")
    matching = [path for path in raw_files if method in path.name]
    return raw_files, matching


def _normalize_openai_tool_call_messages(value: Any) -> tuple[Any, int]:
    if isinstance(value, list):
        normalized: list[Any] = []
        changed = 0
        for item in value:
            clean, count = _normalize_openai_tool_call_messages(item)
            normalized.append(clean)
            changed += count
        return normalized, changed
    if isinstance(value, dict):
        normalized_dict: dict[str, Any] = {}
        changed = 0
        for key, item in value.items():
            clean, count = _normalize_openai_tool_call_messages(item)
            normalized_dict[str(key)] = clean
            changed += count
        # StableToolBench's legacy converter dispatches on key presence before
        # it inspects OpenAI's newer ``tool_calls`` field.  A serialized null
        # legacy field therefore crashes the converter even when there is no
        # tool call and the message should be treated as ordinary assistant
        # text.  Empty/null ``tool_calls`` has the symmetric problem: the
        # upstream branch runs but never creates a node.  Remove both inert
        # representations in the normalized copy while preserving real legacy
        # and modern calls exactly.
        if normalized_dict.get("function_call", object()) is None:
            normalized_dict.pop("function_call", None)
            changed += 1
        if "tool_calls" in normalized_dict and not normalized_dict["tool_calls"]:
            normalized_dict.pop("tool_calls")
            changed += 1
        return normalized_dict, changed
    return value, 0


def _materialize_normalized_answers(
    *, source_dir: Path, normalized_dir: Path
) -> tuple[int, int]:
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for old_path in normalized_dir.glob("*.json"):
        old_path.unlink()
    file_count = 0
    change_count = 0
    for source_path in sorted(source_dir.glob("*.json")):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        normalized, changed = _normalize_openai_tool_call_messages(payload)
        _write_json(normalized_dir / source_path.name, normalized)
        file_count += 1
        change_count += changed
    return file_count, change_count


def _convert_command(
    *,
    stabletoolbench_root: Path,
    answer_dir: Path,
    method: str,
    output_file: Path,
) -> list[str]:
    return [
        "cd",
        str(stabletoolbench_root / "toolbench" / "tooleval"),
        "&&",
        "python",
        "convert_to_answer_format.py",
        "--answer_dir",
        str(answer_dir),
        "--method",
        method,
        "--output",
        str(output_file),
    ]


def audit_stabletoolbench_answer_conversion(
    *,
    stabletoolbench_root: str | Path,
    raw_answer_path: str | Path,
    converted_answer_path: str | Path,
    candidate_model: str,
    test_set: str = "G3_instruction",
    method: str = "CLSTR@1",
    command_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    stabletoolbench_root = Path(stabletoolbench_root)
    raw_answer_path = Path(raw_answer_path)
    converted_answer_path = Path(converted_answer_path)
    command_root = Path.cwd() if command_root is None else Path(command_root)

    answer_dir = raw_answer_path / candidate_model / test_set
    output_file = converted_answer_path / candidate_model / f"{test_set}.json"
    convert_to_answer_format_py = stabletoolbench_root / "toolbench" / "tooleval" / "convert_to_answer_format.py"
    raw_files, matching_files = _raw_answer_files(answer_dir, method)

    blockers: list[str] = []
    if not stabletoolbench_root.is_dir():
        blockers.append("missing_stabletoolbench_root")
    if not convert_to_answer_format_py.is_file():
        blockers.append("missing_convert_to_answer_format_py")
    if not answer_dir.is_dir():
        blockers.append("missing_raw_answer_dir")
    if not matching_files:
        blockers.append("no_raw_answer_files_matching_method")

    answer_dir_for_command = _path_for_command(answer_dir, command_root)
    output_file_for_command = _path_for_command(output_file, command_root)
    report = {
        "status": "ok" if not blockers else "action_required",
        "can_convert_answers": not blockers,
        "blockers": blockers,
        "stabletoolbench_root": str(stabletoolbench_root),
        "raw_answer_path": str(raw_answer_path),
        "converted_answer_path": str(converted_answer_path),
        "answer_dir": str(answer_dir),
        "converted_answer_file": str(output_file),
        "candidate_model": candidate_model,
        "test_set": test_set,
        "method": method,
        "command_root": str(command_root),
        "metric_scope": METRIC_SCOPE,
        "raw_answer_file_count": len(raw_files),
        "matching_raw_answer_file_count": len(matching_files),
        "required_paths": {
            "convert_to_answer_format.py": _file_status(convert_to_answer_format_py),
            "answer_dir": _dir_status(answer_dir),
            "converted_answer_parent": _dir_status(output_file.parent),
        },
        "matching_raw_answer_file_samples": [str(path) for path in matching_files[:5]],
        "reproducible_command": _convert_command(
            stabletoolbench_root=stabletoolbench_root,
            answer_dir=answer_dir_for_command,
            method=method,
            output_file=output_file_for_command,
        ),
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def convert_stabletoolbench_answers(
    *,
    stabletoolbench_root: str | Path,
    raw_answer_path: str | Path,
    converted_answer_path: str | Path,
    candidate_model: str,
    test_set: str = "G3_instruction",
    method: str = "CLSTR@1",
    command_root: str | Path | None = None,
    output_path: str | Path | None = None,
    run_convert: bool = False,
) -> dict[str, Any]:
    report = audit_stabletoolbench_answer_conversion(
        stabletoolbench_root=stabletoolbench_root,
        raw_answer_path=raw_answer_path,
        converted_answer_path=converted_answer_path,
        candidate_model=candidate_model,
        test_set=test_set,
        method=method,
        command_root=command_root,
        output_path=output_path,
    )
    if not run_convert or report["status"] != "ok":
        return report

    stabletoolbench_root = Path(stabletoolbench_root)
    output_file = Path(report["converted_answer_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_dir = output_file.parent / ".normalized_raw" / test_set
    normalized_file_count, normalized_message_count = _materialize_normalized_answers(
        source_dir=Path(report["answer_dir"]),
        normalized_dir=normalized_dir,
    )
    command = [
        "python",
        "convert_to_answer_format.py",
        "--answer_dir",
        str(_path_for_command(normalized_dir, Path(report["command_root"]))),
        "--method",
        method,
        "--output",
        str(_path_for_command(output_file, Path(report["command_root"]))),
    ]
    proc = subprocess.run(
        command,
        cwd=stabletoolbench_root / "toolbench" / "tooleval",
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        **report,
        "normalization_schema": "openai_inert_call_fields_v2",
        "normalized_answer_dir": str(normalized_dir),
        "normalized_file_count": normalized_file_count,
        "normalized_message_count": normalized_message_count,
        "reproducible_command": ["cd", str(stabletoolbench_root / "toolbench" / "tooleval"), "&&", *command],
        "conversion_ran": True,
        "conversion_returncode": proc.returncode,
        "conversion_stdout": proc.stdout[-4000:],
        "conversion_stderr": proc.stderr[-4000:],
        "converted_answer_exists_after": output_file.is_file(),
    }
    if proc.returncode != 0:
        report["status"] = "action_required"
        report["can_convert_answers"] = False
        report["blockers"] = [*report["blockers"], "official_conversion_failed"]
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report
