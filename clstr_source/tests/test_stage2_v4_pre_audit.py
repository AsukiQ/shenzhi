from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

from clstr.stage2_v4_pre_audit import classify_transition_row, run_stage2_v4_pre_audit, summarize_transition_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _load_cli_module():
    path = Path("scripts/audit_clstr_stage2_v4_precheck.py")
    spec = importlib.util.spec_from_file_location("audit_clstr_stage2_v4_precheck", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_transition_row_detects_top5_degradation_and_current_skill_stickiness():
    row = {
        "benchmark": "toolbench_g3",
        "skill_id": "toolbench-g3/weather/get-current-weather",
        "next_skill_id": "toolbench-g3/weather/get-forecast",
        "top1_skill_id": "toolbench-g3/weather/get-current-weather",
        "stage0_gold_rank": 2,
        "stage2_gold_rank": 12,
        "top_skill_ids": [
            "toolbench-g3/weather/get-current-weather",
            "toolbench-g3/weather/search-city",
        ],
    }

    flags = classify_transition_row(row)

    assert flags["stage0_top5_stage2_miss"] is True
    assert flags["stage2_worse_than_stage0"] is True
    assert flags["top1_is_current_skill"] is True
    assert flags["top1_same_provider_as_gold"] is True
    assert flags["top1_same_root_namespace_as_gold"] is True
    assert flags["self_transition"] is False


def test_summarize_transition_rows_separates_toolbench_and_traject_failure_modes():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "skill_id": "toolbench-g3/a/current",
            "next_skill_id": "toolbench-g3/a/gold",
            "top1_skill_id": "toolbench-g3/a/current",
            "stage0_gold_rank": 1,
            "stage2_gold_rank": 20,
            "stage2_hit@5": False,
            "step_index": 1,
        },
        {
            "benchmark": "traject_bench",
            "skill_id": "traject/x",
            "next_skill_id": "traject/y",
            "top1_skill_id": "traject/z",
            "stage0_gold_rank": 80,
            "stage2_gold_rank": 30,
            "stage2_hit@5": False,
            "step_index": 0,
        },
        {
            "benchmark": "traject_bench",
            "skill_id": "traject/x",
            "next_skill_id": "traject/y",
            "top1_skill_id": "traject/y",
            "stage0_gold_rank": 50,
            "stage2_gold_rank": 3,
            "stage2_hit@5": True,
            "step_index": 2,
        },
    ]

    summary = summarize_transition_rows(rows)

    assert summary["overall"]["row_count"] == 3
    assert summary["by_benchmark"]["toolbench_g3"]["stage0_top5_stage2_miss_count"] == 1
    assert summary["by_benchmark"]["toolbench_g3"]["top1_is_current_skill_fraction"] == 1.0
    assert summary["by_benchmark"]["toolbench_g3"]["top1_same_root_namespace_as_gold_fraction"] == 1.0
    assert summary["by_benchmark"]["traject_bench"]["stage0_low_rank_count"] == 2
    assert summary["by_benchmark"]["traject_bench"]["stage2_improved_vs_stage0_fraction"] == 1.0
    assert summary["by_benchmark_transition_type"]["toolbench_g3::switch"]["row_count"] == 1
    assert summary["by_benchmark_transition_type"]["traject_bench::switch"]["row_count"] == 2


def test_run_stage2_v4_pre_audit_writes_report_and_context_examples(tmp_path):
    rows = [
        {
            "row_index": 5,
            "benchmark": "toolbench_g3",
            "task_id": "task-a",
            "trajectory_id": "traj-a",
            "step_index": 0,
            "skill_id": "toolbench-g3/weather/current",
            "next_skill_id": "toolbench-g3/weather/forecast",
            "top1_skill_id": "toolbench-g3/weather/current",
            "top_skill_ids": ["toolbench-g3/weather/current"],
            "stage0_gold_rank": 1,
            "stage2_gold_rank": 25,
            "stage2_hit@5": False,
            "action_text": "weather.current({})",
            "next_action_text": "weather.forecast({})",
        }
    ]
    trajectories = [
        {"benchmark": "toolbench_g3", "task_id": "other", "step_index": 0},
        {
            "benchmark": "toolbench_g3",
            "task_id": "task-a",
            "trajectory_id": "traj-a",
            "step_index": 0,
            "goal_text": "get weather",
            "state_text": "goal: get weather",
            "history_text": "",
            "next_observation_text": "forecast result",
        },
    ]
    diagnostics_path = tmp_path / "rows.jsonl"
    trajectories_path = tmp_path / "trajectories.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(diagnostics_path, rows)
    _write_jsonl(trajectories_path, trajectories)

    report = run_stage2_v4_pre_audit(
        row_diagnostics_path=diagnostics_path,
        trajectories_path=trajectories_path,
        output_dir=output_dir,
        benchmarks=["toolbench_g3"],
        example_limit=5,
        benchmark_caps={"toolbench_g3": -1},
    )

    assert report["status"] == "ok"
    assert report["summary"]["by_benchmark"]["toolbench_g3"]["stage0_top5_stage2_miss_count"] == 1
    assert (output_dir / "stage2_v4_pre_audit_report.json").exists()
    markdown = (output_dir / "stage2_v4_pre_audit_report.md").read_text(encoding="utf-8")
    assert "By Benchmark x Transition Type" in markdown
    examples = [
        json.loads(line)
        for line in (output_dir / "stage2_v4_pre_audit_examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert examples[0]["case"] == "stage0_top5_stage2_miss"
    assert examples[0]["context"]["goal_text"] == "get weather"


def test_cli_benchmark_caps_parser_accepts_equals_and_colon():
    module = _load_cli_module()

    caps = module._benchmark_caps("toolbench_g3=-1,traject_bench:-1,alfworld=10000")

    assert caps == {"toolbench_g3": -1, "traject_bench": -1, "alfworld": 10000}


def test_cli_help_runs_without_external_pythonpath():
    result = subprocess.run(
        [sys.executable, "scripts/audit_clstr_stage2_v4_precheck.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Audit Stage2 v4 failure modes" in result.stdout
