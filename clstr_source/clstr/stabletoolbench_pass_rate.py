from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRIC_SCOPE = "StableToolBench Solvable Pass Rate; not static routing recall."


def _file_status(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _command_path(path: Path, command_root: Path) -> Path:
    return path if path.is_absolute() else command_root / path


def _has_valid_api_key(api_pool_file: Path) -> tuple[bool, str | None, int]:
    if not api_pool_file.is_file():
        return False, None, 0
    try:
        payload = json.loads(api_pool_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid_api_pool_json: {exc}", 0
    if not isinstance(payload, list):
        return False, "api_pool_json_must_be_a_list", 0

    valid_count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        api_key = item.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            valid_count += 1
    return valid_count > 0, None, valid_count


def _command(
    *,
    stabletoolbench_root: Path,
    converted_answer_path: Path,
    save_path: Path,
    candidate_model: str,
    test_ids_dir: Path,
    evaluator: str,
    max_eval_threads: int,
    evaluate_times: int,
    test_set: str,
) -> list[str]:
    return [
        "cd",
        str(stabletoolbench_root / "toolbench" / "tooleval"),
        "&&",
        "python",
        "eval_pass_rate.py",
        "--converted_answer_path",
        str(converted_answer_path),
        "--save_path",
        str(save_path / candidate_model),
        "--reference_model",
        candidate_model,
        "--test_ids",
        str(test_ids_dir),
        "--evaluator",
        evaluator,
        "--max_eval_threads",
        str(max_eval_threads),
        "--evaluate_times",
        str(evaluate_times),
        "--test_set",
        test_set,
    ]


def audit_stabletoolbench_pass_rate(
    *,
    stabletoolbench_root: str | Path,
    converted_answer_path: str | Path,
    api_pool_file: str | Path | None = None,
    candidate_model: str,
    test_set: str = "G3_instruction",
    save_path: str | Path = "outputs/toolbench_g3/stabletoolbench_pass_rate",
    test_ids_dir: str | Path | None = None,
    evaluator: str = "tooleval_gpt-3.5-turbo_default",
    max_eval_threads: int = 1,
    evaluate_times: int = 3,
    command_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    stabletoolbench_root = Path(stabletoolbench_root)
    converted_answer_path = Path(converted_answer_path)
    save_path = Path(save_path)
    api_pool_file = Path(api_pool_file) if api_pool_file is not None else stabletoolbench_root / "openai_key.json"
    test_ids_dir = Path(test_ids_dir) if test_ids_dir is not None else stabletoolbench_root / "solvable_queries" / "test_query_ids"
    command_root = Path.cwd() if command_root is None else Path(command_root)
    converted_answer_path_for_command = _command_path(converted_answer_path, command_root)
    save_path_for_command = _command_path(save_path, command_root)

    eval_pass_rate_py = stabletoolbench_root / "toolbench" / "tooleval" / "eval_pass_rate.py"
    convert_to_answer_format_py = stabletoolbench_root / "toolbench" / "tooleval" / "convert_to_answer_format.py"
    test_ids_file = test_ids_dir / f"{test_set}.json"
    converted_answer_file = converted_answer_path / candidate_model / f"{test_set}.json"
    evaluator_config = stabletoolbench_root / "toolbench" / "tooleval" / "evaluators" / evaluator / "config.yaml"

    required_files = {
        "eval_pass_rate.py": _file_status(eval_pass_rate_py),
        "convert_to_answer_format.py": _file_status(convert_to_answer_format_py),
        "test_ids": _file_status(test_ids_file),
        "converted_answer_file": _file_status(converted_answer_file),
        "api_pool_file": _file_status(api_pool_file),
        "evaluator_config": _file_status(evaluator_config),
    }

    blockers: list[str] = []
    if not stabletoolbench_root.is_dir():
        blockers.append("missing_stabletoolbench_root")
    if not eval_pass_rate_py.is_file():
        blockers.append("missing_eval_pass_rate_py")
    if not convert_to_answer_format_py.is_file():
        blockers.append("missing_convert_to_answer_format_py")
    if not test_ids_file.is_file():
        blockers.append("missing_test_ids_file")
    if not converted_answer_file.is_file():
        blockers.append("missing_converted_answer_file")
    if not evaluator_config.is_file():
        blockers.append("missing_evaluator_config")
    if not api_pool_file.is_file():
        blockers.append("missing_api_pool_file")
        api_key_valid = False
        api_pool_error = None
        valid_api_key_count = 0
    else:
        api_key_valid, api_pool_error, valid_api_key_count = _has_valid_api_key(api_pool_file)
        if api_pool_error:
            blockers.append(api_pool_error.split(":", 1)[0])
        if not api_key_valid:
            blockers.append("api_pool_has_no_valid_api_key")

    report = {
        "status": "ok" if not blockers else "action_required",
        "can_run_pass_rate": not blockers,
        "blockers": blockers,
        "stabletoolbench_root": str(stabletoolbench_root),
        "converted_answer_path": str(converted_answer_path),
        "converted_answer_file": str(converted_answer_file),
        "api_pool_file": str(api_pool_file),
        "test_ids_dir": str(test_ids_dir),
        "test_ids_file": str(test_ids_file),
        "save_path": str(save_path),
        "candidate_model": candidate_model,
        "test_set": test_set,
        "evaluator": evaluator,
        "max_eval_threads": max_eval_threads,
        "evaluate_times": evaluate_times,
        "command_root": str(command_root),
        "metric_scope": METRIC_SCOPE,
        "required_files": required_files,
        "api_pool": {
            "has_valid_api_key": api_key_valid,
            "valid_api_key_count": valid_api_key_count,
            "error": api_pool_error,
        },
        "reproducible_command": _command(
            stabletoolbench_root=stabletoolbench_root,
            converted_answer_path=converted_answer_path_for_command,
            save_path=save_path_for_command,
            candidate_model=candidate_model,
            test_ids_dir=test_ids_dir,
            evaluator=evaluator,
            max_eval_threads=max_eval_threads,
            evaluate_times=evaluate_times,
            test_set=test_set,
        ),
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report
