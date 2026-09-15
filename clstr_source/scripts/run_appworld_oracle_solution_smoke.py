#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_routing import write_json


def run_oracle_solution_smoke(
    *,
    appworld_root: str | Path,
    appworld_cache: str | Path,
    output_dir: str | Path,
    task_id: str = "82e2fac_1",
    experiment_name: str = "clstr_appworld_oracle_solution_smoke",
) -> dict:
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ["APPWORLD_CACHE"] = str(appworld_cache)
    os.environ.setdefault("IPYTHONDIR", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython")
    Path(appworld_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)

    from appworld.environment import AppWorld

    root = Path(appworld_root)
    solution_path = root / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py"
    report = {
        "status": "ok",
        "role": "oracle_runtime_sanity_check_not_a_model_result",
        "task_id": task_id,
        "experiment_name": experiment_name,
        "solution_path": str(solution_path),
        "appworld_root": str(root),
    }
    if not solution_path.exists():
        report.update({"status": "blocked", "blocker": "compiled_solution_missing"})
        write_json(Path(output_dir) / "report.json", report)
        return report

    world = AppWorld(
        task_id,
        experiment_name=experiment_name,
        max_interactions=3,
        timeout_seconds=20,
        load_ground_truth=True,
        ground_truth_mode="minimal",
        show_api_response_schemas=False,
    )
    try:
        code = solution_path.read_text(encoding="utf-8") + "\n\nsolution(apis, requester)\n"
        output = world.execute(code)
        completed = bool(world.task_completed())
        tracker = world.evaluate(suppress_errors=True)
        report.update(
            {
                "execute_output": str(output),
                "task_completed": completed,
                "evaluate_available": tracker is not None,
                "success": completed,
            }
        )
    except Exception as exc:
        report.update({"status": "blocked", "blocker": f"{type(exc).__name__}: {exc}"})
    finally:
        world.close()
    write_json(Path(output_dir) / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AppWorld official compiled-solution oracle smoke.")
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--appworld_cache", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld")
    parser.add_argument("--output_dir", default="outputs/appworld_oracle_solution_smoke")
    parser.add_argument("--task_id", default="82e2fac_1")
    args = parser.parse_args()
    report = run_oracle_solution_smoke(
        appworld_root=args.appworld_root,
        appworld_cache=args.appworld_cache,
        output_dir=args.output_dir,
        task_id=args.task_id,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
