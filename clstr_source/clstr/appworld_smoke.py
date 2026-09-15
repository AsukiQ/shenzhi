from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize_task_splits(data_dir: Path, sample_size: int = 5) -> dict[str, dict[str, Any]]:
    datasets_dir = data_dir / "datasets"
    if not datasets_dir.exists():
        return {}
    summary: dict[str, dict[str, Any]] = {}
    for path in sorted(datasets_dir.glob("*.txt")):
        task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary[path.stem] = {
            "count": len(task_ids),
            "sample_task_ids": task_ids[:sample_size],
        }
    return summary


def run_appworld_smoke(
    output_dir: str | Path | None = None,
    module_name: str = "appworld",
    attempt_reset: bool = False,
    appworld_root: str | Path | None = None,
    appworld_cache: str | Path | None = None,
) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        report = {
            "status": "blocked",
            "blocker": "appworld_module_missing",
            "module_name": module_name,
            "import_ok": False,
            "reset_attempted": False,
            "recommended_next_step": "Install AppWorld on the login node or prepare a project-local environment before running AppWorld harness smoke.",
        }
        if output_dir is not None:
            _write_json(Path(output_dir) / "blocker_report.json", report)
        return report

    module = importlib.import_module(module_name)
    environment_module = f"{module_name}.environment"
    environment_spec = importlib.util.find_spec(environment_module)
    report: dict[str, Any] = {
        "status": "ok",
        "module_name": module_name,
        "import_ok": True,
        "module_file": getattr(module, "__file__", None),
        "module_version": getattr(module, "__version__", None),
        "environment_module_available": environment_spec is not None,
        "reset_attempted": False,
        "reset_ok": None,
    }

    root_value = appworld_root or os.environ.get("APPWORLD_ROOT")
    cache_value = appworld_cache or os.environ.get("APPWORLD_CACHE")
    if root_value:
        root_path = Path(root_value)
        data_dir = root_path / "data"
        report.update(
            {
                "appworld_root": str(root_path),
                "data_dir": str(data_dir),
                "data_dir_exists": data_dir.exists(),
            }
        )
        if not data_dir.exists():
            report["status"] = "partial"
            report["blocker"] = "appworld_data_missing"
            report["recommended_next_step"] = (
                "Run appworld download data on the login node with APPWORLD_ROOT set under autodl-tmp, "
                "then rerun scripts/run_appworld_smoke.py."
            )
        else:
            report["task_splits"] = _summarize_task_splits(data_dir)
    if cache_value:
        report["appworld_cache"] = str(Path(cache_value))

    if attempt_reset:
        report["reset_attempted"] = True
        report["reset_ok"] = False
        report["reset_error"] = "automatic reset is intentionally not implemented until AppWorld local data paths are known"
        report["status"] = "partial"

    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / "report.json", report)
        if report["status"] != "ok":
            _write_json(out / "blocker_report.json", report)
    return report
