import json
from pathlib import Path

from scripts.build_clstr_controller_reports import build_controller_reports


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_controller_reports_use_real_scienceworld_and_webshop_gate_metrics(tmp_path):
    outputs = tmp_path / "outputs"
    _write_json(
        outputs / "alfworld_eval" / "clstr_routing_init_baseline" / "metrics.json",
        {"status": "ok", "success_rate": 0.0, "average_reward": 0.0, "average_goal_condition_points": 0.0, "average_episode_steps": 50.0, "episodes": 274},
    )
    _write_json(
        outputs / "alfworld_eval" / "clstr_policy_gate_valid_seen" / "metrics.json",
        {"status": "ok", "success_rate": 0.0, "average_reward": 0.0, "average_goal_condition_points": 0.0, "average_episode_steps": 50.0, "episodes": 20},
    )
    _write_json(
        outputs / "alfworld_eval" / "clstr_full_base_gate" / "metrics.json",
        {"status": "ok", "success_rate": 0.0, "average_reward": 0.0, "average_goal_condition_points": 0.0, "average_episode_steps": 50.0, "episodes": 20},
    )
    _write_json(
        outputs / "alfworld_eval" / "clstr_controller_gate_valid_seen" / "metrics.json",
        {"status": "ok", "success_rate": 0.0, "average_reward": 0.0, "average_goal_condition_points": 0.0, "average_episode_steps": 50.0, "episodes": 20},
    )
    (outputs / "alfworld_eval" / "clstr_controller_gate_valid_seen" / "run.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "steps": 50.0,
                "action_trace": ["go to fridge 1"],
                "chosen_reason_trace": ["max_final_score"],
                "component_score_trace": [
                    [
                        {
                            "action": "go to fridge 1",
                            "policy_score": 0.8,
                            "transition_score": 0.6,
                            "belief_score": 0.61,
                            "stop_probability": 0.02,
                            "loop_penalty": 0.0,
                            "final_score": 1.1,
                        }
                    ]
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(outputs / "scienceworld_eval" / "harness_smoke_report.json", {"status": "ok", "smoke_success": True})
    _write_json(
        outputs / "scienceworld_eval" / "clstr_controller_gate" / "metrics.json",
        {"status": "ok", "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 10.0, "episodes": 1},
    )
    _write_json(outputs / "webshop_eval" / "harness_smoke_report.json", {"status": "ok", "smoke_success": True})
    _write_json(
        outputs / "webshop_eval" / "clstr_controller_gate" / "metrics.json",
        {"status": "ok", "closed_loop_evaluated": True, "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 10.0, "episodes": 1},
    )
    _write_json(
        outputs / "dbbench_eval" / "harness_audit_report.json",
        {
            "status": "blocked_no_closed_loop_eval_harness",
            "closed_loop_evaluated": False,
            "missing_items": ["official_closed_loop_harness"],
            "reproducible_commands": {"audit": "python -B scripts/run_dbbench_clstr_eval.py --output_path outputs/dbbench_eval/harness_audit_report.json"},
        },
    )

    summary = build_controller_reports(
        output_root=outputs,
        output_table_path=outputs / "clstr_controller_comparison_table.md",
        output_summary_path=outputs / "clstr_controller_comparison_summary.json",
        output_paper_md_path=outputs / "clstr_controller_paper_report.md",
        output_paper_json_path=outputs / "clstr_controller_paper_report.json",
    )

    table = (outputs / "clstr_controller_comparison_table.md").read_text(encoding="utf-8")
    paper = (outputs / "clstr_controller_paper_report.md").read_text(encoding="utf-8")
    rows = {row["method"]: row for row in summary["rows"]}

    assert rows["scienceworld_controller_gate"]["status"] == "ok"
    assert rows["scienceworld_controller_gate"]["closed_loop_evaluated"] is True
    assert rows["webshop_controller_gate"]["status"] == "ok"
    assert rows["webshop_controller_gate"]["closed_loop_evaluated"] is True
    assert rows["dbbench_controller_gate"]["status"] == "blocked_no_closed_loop_eval_harness"
    assert summary["component_trace_example"]["chosen_action"] == "go to fridge 1"
    assert summary["component_trace_example"]["top_component_scores"][0]["transition_score"] == 0.6
    assert "blocked by missing official harness" not in table.lower()
    assert "ScienceWorld: official smoke OK; gate closed-loop eval ran for 1 episode(s)" in paper
    assert "WebShop: official smoke OK; gate closed-loop eval ran for 1 episode(s)" in paper
    assert "DBBench: blocked_no_closed_loop_eval_harness" in paper
    assert "Component Trace Example" in paper
