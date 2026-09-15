import json
import subprocess
import sys
from pathlib import Path

from clstr.eval_matrix_readiness import audit_eval_matrix_readiness


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_toolret_eval(root: Path) -> None:
    write_jsonl(root / "queries.jsonl", [{"query_id": "q1", "query_text": "find weather"}])
    write_jsonl(root / "skills.jsonl", [{"skill_id": "s1", "description": "weather"}])
    write_jsonl(root / "qrels.jsonl", [{"query_id": "q1", "skill_id": "s1", "relevance": 1}])


def make_traject_eval(root: Path) -> None:
    write_jsonl(
        root / "queries.jsonl",
        [
            {
                "query_id": "tq1",
                "query_text": "step 1",
                "trajectory_id": "traj1",
                "step_index": 0,
                "trajectory_type": "sequential",
            }
        ],
    )
    write_jsonl(root / "skills.jsonl", [{"skill_id": "ts1", "description": "tool"}])
    write_jsonl(root / "qrels.jsonl", [{"query_id": "tq1", "skill_id": "ts1", "relevance": 1}])


def make_toolbench_g3(root: Path, source_root: Path) -> None:
    write_jsonl(root / "skills.jsonl", [{"skill_id": "tb1", "description": "tool"}])
    write_jsonl(
        root / "retrieval.jsonl",
        [{"query_id": "g3q1", "query_text": "call tool", "positive_skill_id": "tb1"}],
    )
    write_jsonl(
        root / "trajectories.jsonl",
        [{"skill_id": "tb1", "next_skill_id": "tb1", "benchmark": "toolbench_g3"}],
    )
    touch(source_root / "instruction/G3_query.json", "{}")
    touch(source_root / "answer/G3_answer/1.json", "{}")


def make_stabletoolbench(root: Path, converted_root: Path, candidate_model: str = "clstr_toolbench_g3") -> None:
    touch(root / "toolbench/tooleval/eval_pass_rate.py")
    touch(root / "toolbench/tooleval/convert_to_answer_format.py")
    touch(root / "toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/config.yaml")
    write_json(root / "solvable_queries/test_query_ids/G3_instruction.json", {"1": 0})
    write_json(root / "openai_key.json", [{"api_key": "sk-test"}])
    write_json(converted_root / candidate_model / "G3_instruction.json", {"1": {"answer": {}}})


def make_stage4_quality(output_dir: Path, checkpoint: Path, *, steps: int = 6) -> None:
    latest = output_dir / "checkpoints/latest.pt"
    metrics = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    touch(checkpoint)
    touch(latest)
    touch(loss_curve, "<svg>loss</svg>")
    write_jsonl(
        metrics,
        [
            {
                "step": step,
                "loss": 1.0 / step,
                "stage4_act_loss": 1.0 / step,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, steps + 1)
        ],
    )
    write_json(
        output_dir / "train_stdout.json",
        {
            "status": "ok",
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "training_objective": "transition_conditioned_next_skill_ce",
            "training_regime": "offline_train_split_transition_conditioned_next_skill",
            "checkpoint": str(checkpoint),
            "training_metrics_path": str(metrics),
            "loss_curve_path": str(loss_curve),
            "latest_checkpoint": str(latest),
            "valid_or_test_used_for_training": False,
            "on_policy_rollout_used": False,
            "data_report": {
                "stage4_rows": 24,
                "candidate_source": "stage0_topm_online",
                "positive_injected_rows": 0,
                "stage0_candidate_rows": 24,
                "stage0_candidate_handoff": {
                    "enabled": True,
                    "candidate_source": "stage0_topm_online",
                    "top_m": 350,
                    "positive_missing_policy": "skip",
                    "query_mode": "skillrouter_state",
                    "injected_positive_rows": 0,
                },
                "benchmark_filter_enabled": True,
                "benchmark_counts": {"alfworld": 6, "toolbench_g3": 6, "traject_bench": 6, "webshop": 6},
            },
            "freeze_report": {
                "frozen_routing_foundation": True,
                "train_transition": True,
                "trainable_modules": ["trans_head", "action_proj", "transition"],
            },
            "checkpoint_init_report": {
                "partial_load_mode": "stage0_routing_plus_act_init_heads_for_joint_stage4",
                "protect_routing_foundation": True,
                "head_overrides_routing_keys": ["transition.obs_proj.weight"],
                "skipped_head_routing_foundation_keys": ["skill_table.E"],
            },
            "last_metrics": {
                "stage4_act_loss": 0.1,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            },
        },
    )


def test_eval_matrix_blocks_official_main_table_when_only_proxy_outputs_exist(tmp_path):
    checkpoint = tmp_path / "outputs/stage4/checkpoints/clstr_stage4_act-step3000.pt"
    touch(checkpoint)
    make_toolret_eval(tmp_path / "data/toolret_eval")
    make_traject_eval(tmp_path / "data/traject_eval")
    make_toolbench_g3(tmp_path / "data/toolbench_g3", tmp_path / "ToolBench/data")

    proxy_path = tmp_path / "outputs/traject_proxy/traject_sequence_proxy_metrics.json"
    write_json(proxy_path, {"status": "ok", "metric_scope": "TRAJECT selection-only sequence proxy"})
    toolret_run = tmp_path / "outputs/toolret/run.tsv"
    touch(toolret_run, "q1 Q0 s1 1 1.0 clstr\n")
    toolbench_routing = tmp_path / "outputs/toolbench_g3/routing_eval.json"
    write_json(toolbench_routing, {"status": "ok", "metric_scope": "static routing"})

    report = audit_eval_matrix_readiness(
        stage4_checkpoint=checkpoint,
        toolret_eval_dir=tmp_path / "data/toolret_eval",
        toolret_run_path=toolret_run,
        traject_eval_dir=tmp_path / "data/traject_eval",
        traject_sequence_proxy_path=proxy_path,
        toolbench_g3_data_dir=tmp_path / "data/toolbench_g3",
        toolbench_g3_source_root=tmp_path / "ToolBench/data",
        toolbench_g3_routing_report=toolbench_routing,
        stabletoolbench_root=tmp_path / "StableToolBench",
        converted_answer_path=tmp_path / "outputs/converted",
        candidate_model="clstr_toolbench_g3",
    )

    assert report["status"] == "action_required"
    assert report["official_main_table_ready"] is False
    assert report["proxy_or_routing_eval_ready"] is True
    assert report["main_table"]["traject"]["official_ready"] is False
    assert report["main_table"]["traject"]["proxy_ready"] is True
    assert "traject_official_outputs_missing" in report["blockers"]
    assert "toolbench_g3_pass_rate_not_ready" in report["blockers"]
    assert "stage4_quality_gate_not_ok" in report["blockers"]
    assert report["main_table"]["toolbench_g3"]["routing_proxy_ready"] is True
    assert report["main_table"]["toolbench_g3"]["official_ready"] is False


def test_eval_matrix_ready_when_all_main_official_artifacts_exist(tmp_path):
    checkpoint = tmp_path / "outputs/stage4/checkpoints/clstr_stage4_act-step3000.pt"
    make_stage4_quality(tmp_path / "outputs/stage4", checkpoint)
    make_toolret_eval(tmp_path / "data/toolret_eval")
    make_traject_eval(tmp_path / "data/traject_eval")
    make_toolbench_g3(tmp_path / "data/toolbench_g3", tmp_path / "ToolBench/data")
    make_stabletoolbench(tmp_path / "StableToolBench", tmp_path / "outputs/converted")

    toolret_run = tmp_path / "outputs/toolret/run.tsv"
    touch(toolret_run, "q1 Q0 s1 1 1.0 clstr\n")
    traject_official = tmp_path / "outputs/traject_official/metrics.json"
    write_json(
        traject_official,
        {
            "status": "ok",
            "metrics": {"EM": 1.0, "Inclusion": 1.0, "Usage": 1.0, "Traj-Satisfy": 1.0, "Acc": 1.0},
        },
    )
    appworld_plot = tmp_path / "outputs/appworld_combination/plot.json"
    write_json(appworld_plot, {"status": "ok", "token_cost_reduction": 0.31, "task_success_drop": 0.0})

    report = audit_eval_matrix_readiness(
        stage4_checkpoint=checkpoint,
        toolret_eval_dir=tmp_path / "data/toolret_eval",
        toolret_run_path=toolret_run,
        traject_eval_dir=tmp_path / "data/traject_eval",
        traject_official_metrics_path=traject_official,
        toolbench_g3_data_dir=tmp_path / "data/toolbench_g3",
        toolbench_g3_source_root=tmp_path / "ToolBench/data",
        stabletoolbench_root=tmp_path / "StableToolBench",
        converted_answer_path=tmp_path / "outputs/converted",
        candidate_model="clstr_toolbench_g3",
        appworld_combination_report=appworld_plot,
        min_stage4_steps=6,
    )

    assert report["status"] == "ready"
    assert report["official_main_table_ready"] is True
    assert report["main_table"]["toolret"]["official_ready"] is True
    assert report["main_table"]["traject"]["official_ready"] is True
    assert report["main_table"]["toolbench_g3"]["official_ready"] is True
    assert report["secondary"]["appworld"]["ready"] is True
    assert report["blockers"] == []


def test_eval_matrix_requires_stage4_quality_gate_not_checkpoint_only(tmp_path):
    checkpoint = tmp_path / "outputs/stage4/checkpoints/clstr_stage4_act-step3000.pt"
    touch(checkpoint)
    make_toolret_eval(tmp_path / "data/toolret_eval")
    make_traject_eval(tmp_path / "data/traject_eval")
    make_toolbench_g3(tmp_path / "data/toolbench_g3", tmp_path / "ToolBench/data")
    make_stabletoolbench(tmp_path / "StableToolBench", tmp_path / "outputs/converted")
    toolret_run = tmp_path / "outputs/toolret/run.tsv"
    touch(toolret_run, "q1 Q0 s1 1 1.0 clstr\n")
    traject_official = tmp_path / "outputs/traject_official/metrics.json"
    write_json(
        traject_official,
        {
            "status": "ok",
            "metrics": {"EM": 1.0, "Inclusion": 1.0, "Usage": 1.0, "Traj-Satisfy": 1.0, "Acc": 1.0},
        },
    )
    appworld_plot = tmp_path / "outputs/appworld_combination/plot.json"
    write_json(appworld_plot, {"status": "ok"})

    report = audit_eval_matrix_readiness(
        stage4_checkpoint=checkpoint,
        toolret_eval_dir=tmp_path / "data/toolret_eval",
        toolret_run_path=toolret_run,
        traject_eval_dir=tmp_path / "data/traject_eval",
        traject_official_metrics_path=traject_official,
        toolbench_g3_data_dir=tmp_path / "data/toolbench_g3",
        toolbench_g3_source_root=tmp_path / "ToolBench/data",
        stabletoolbench_root=tmp_path / "StableToolBench",
        converted_answer_path=tmp_path / "outputs/converted",
        candidate_model="clstr_toolbench_g3",
        appworld_combination_report=appworld_plot,
        min_stage4_steps=6,
    )

    assert report["official_main_table_ready"] is False
    assert "stage4_quality_gate_not_ok" in report["blockers"]
    assert report["stage4_quality_gate"]["status"] == "action_required"


def test_eval_matrix_readiness_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_path = tmp_path / "eval_matrix.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_eval_matrix_readiness.py",
            "--stage4_checkpoint",
            str(tmp_path / "missing.pt"),
            "--toolret_eval_dir",
            str(tmp_path / "missing_toolret"),
            "--traject_eval_dir",
            str(tmp_path / "missing_traject"),
            "--toolbench_g3_data_dir",
            str(tmp_path / "missing_toolbench"),
            "--stabletoolbench_root",
            str(tmp_path / "missing_stabletoolbench"),
            "--converted_answer_path",
            str(tmp_path / "converted"),
            "--candidate_model",
            "clstr_toolbench_g3",
            "--output_path",
            str(output_path),
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
    assert "missing_stage4_checkpoint" in report["blockers"]
