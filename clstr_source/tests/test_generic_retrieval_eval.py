import json
import subprocess
import sys
from pathlib import Path

import pytest

from clstr.retrieval_metrics import (
    compute_ranking_metrics,
    evaluate_retrieval_run,
    load_jsonl_run,
    load_qrels_jsonl,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compute_ranking_metrics_reports_toolret_style_metrics():
    qrels = {
        "q1": {"d1": 1, "d3": 1},
        "q2": {"d2": 1},
    }
    run = {
        "q1": [("d2", 3.0), ("d1", 2.0), ("d3", 1.0)],
        "q2": [("d4", 2.0), ("d5", 1.0), ("d6", 0.5)],
        "q3": [("d1", 1.0)],
    }

    metrics = compute_ranking_metrics(qrels, run, k_values=(2, 3))

    assert metrics["evaluated_queries"] == 2
    assert metrics["missing_run_queries"] == 0
    assert metrics["Recall@2"] == pytest.approx(0.25)
    assert metrics["Recall@3"] == pytest.approx(0.5)
    assert metrics["MAP@3"] == pytest.approx(0.291667)
    assert metrics["NDCG@3"] == pytest.approx(0.346713)


def test_loaders_accept_unified_qrels_and_jsonl_ranked_outputs(tmp_path):
    qrels_path = tmp_path / "qrels.jsonl"
    run_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        qrels_path,
        [
            {"query_id": "q1", "skill_id": "s1", "relevance": 1},
            {"query_id": "q1", "tool_id": "s2", "relevance": 0},
            {"query_id": "q2", "doc_id": "s3", "relevance": 2},
        ],
    )
    write_jsonl(
        run_path,
        [
            {"query_id": "q1", "ranked_skill_ids": ["s0", "s1"], "scores": [0.9, 0.8]},
            {"query_id": "q2", "ranked_tool_ids": ["s3"], "scores": [0.7]},
        ],
    )

    qrels = load_qrels_jsonl(qrels_path)
    run = load_jsonl_run(run_path)

    assert qrels == {"q1": {"s1": 1}, "q2": {"s3": 2}}
    assert run == {"q1": [("s0", 0.9), ("s1", 0.8)], "q2": [("s3", 0.7)]}


def test_evaluate_retrieval_run_writes_report_for_cli_and_programmatic_use(tmp_path):
    qrels_path = tmp_path / "qrels.jsonl"
    run_path = tmp_path / "run.tsv"
    output_dir = tmp_path / "eval"
    write_jsonl(
        qrels_path,
        [
            {"query_id": "q1", "skill_id": "s1", "relevance": 1},
            {"query_id": "q2", "skill_id": "s2", "relevance": 1},
        ],
    )
    run_path.write_text(
        "\n".join(
            [
                "q1 Q0 s0 1 3.0 clstr",
                "q1 Q0 s1 2 2.0 clstr",
                "q2 Q0 s2 1 4.0 clstr",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_retrieval_run(
        qrels_path=qrels_path,
        run_path=run_path,
        output_dir=output_dir,
        run_format="trec",
        k_values=(1, 2),
        benchmark="toolret",
        method="clstr",
    )

    assert report["status"] == "ok"
    assert report["benchmark"] == "toolret"
    assert report["method"] == "clstr"
    assert report["metrics"]["Recall@1"] == pytest.approx(0.5)
    assert report["metrics"]["Recall@2"] == pytest.approx(1.0)
    assert (output_dir / "metrics.json").exists()

    cli_output = tmp_path / "cli_eval"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_retrieval_run.py",
            "--qrels_path",
            str(qrels_path),
            "--run_path",
            str(run_path),
            "--output_dir",
            str(cli_output),
            "--run_format",
            "trec",
            "--k_values",
            "1",
            "2",
            "--benchmark",
            "toolret",
            "--method",
            "clstr",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )
    cli_report = json.loads(proc.stdout)
    assert cli_report["metrics"]["MAP@2"] == pytest.approx(0.75)
    assert (cli_output / "metrics.json").exists()
