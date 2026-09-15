import json
import subprocess
import sys
from pathlib import Path

import pytest

from clstr.traject_sequence_proxy import evaluate_traject_sequence_proxy


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluate_traject_sequence_proxy_reports_step_and_trajectory_metrics(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    run_path = tmp_path / "run.tsv"
    output_dir = tmp_path / "proxy"
    write_jsonl(
        queries_path,
        [
            {"query_id": "traj-a::0", "trajectory_id": "traj-a", "step_index": 0, "trajectory_type": "sequential"},
            {"query_id": "traj-a::1", "trajectory_id": "traj-a", "step_index": 1, "trajectory_type": "sequential"},
            {"query_id": "traj-b::0", "trajectory_id": "traj-b", "step_index": 0, "trajectory_type": "parallel"},
            {"query_id": "traj-b::1", "trajectory_id": "traj-b", "step_index": 1, "trajectory_type": "parallel"},
        ],
    )
    write_jsonl(
        qrels_path,
        [
            {"query_id": "traj-a::0", "skill_id": "tool/weather", "relevance": 1},
            {"query_id": "traj-a::1", "skill_id": "tool/map", "relevance": 1},
            {"query_id": "traj-b::0", "skill_id": "tool/search", "relevance": 1},
            {"query_id": "traj-b::1", "skill_id": "tool/summarize", "relevance": 1},
        ],
    )
    run_path.write_text(
        "\n".join(
            [
                "traj-a::0 Q0 tool/weather 1 4.0 clstr",
                "traj-a::1 Q0 tool/wrong 1 4.0 clstr",
                "traj-b::0 Q0 tool/summarize 1 4.0 clstr",
                "traj-b::1 Q0 tool/search 1 3.0 clstr",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_traject_sequence_proxy(
        queries_path=queries_path,
        qrels_path=qrels_path,
        run_path=run_path,
        output_dir=output_dir,
        run_format="trec",
        method="clstr",
    )

    assert report["status"] == "ok"
    assert report["metrics"]["step_top1_accuracy"] == pytest.approx(0.25)
    assert report["metrics"]["ordered_exact_match"] == pytest.approx(0.0)
    assert report["metrics"]["unordered_exact_match"] == pytest.approx(0.5)
    assert report["metrics"]["inclusion"] == pytest.approx(0.75)
    assert report["by_trajectory_type"]["sequential"]["inclusion"] == pytest.approx(0.5)
    assert report["by_trajectory_type"]["parallel"]["unordered_exact_match"] == pytest.approx(1.0)
    assert (output_dir / "traject_sequence_proxy_metrics.json").exists()


def test_evaluate_traject_sequence_proxy_cli_writes_report(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    run_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "cli_proxy"
    write_jsonl(
        queries_path,
        [{"query_id": "traj::0", "trajectory_id": "traj", "step_index": 0, "trajectory_type": "sequential"}],
    )
    write_jsonl(qrels_path, [{"query_id": "traj::0", "skill_id": "tool/a", "relevance": 1}])
    write_jsonl(run_path, [{"query_id": "traj::0", "ranked_skill_ids": ["tool/a"], "scores": [1.0]}])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_traject_sequence_proxy.py",
            "--queries_path",
            str(queries_path),
            "--qrels_path",
            str(qrels_path),
            "--run_path",
            str(run_path),
            "--output_dir",
            str(output_dir),
            "--run_format",
            "jsonl",
            "--method",
            "clstr",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["metrics"]["ordered_exact_match"] == pytest.approx(1.0)
    assert report["metric_scope"] == "TRAJECT selection-only sequence proxy"
