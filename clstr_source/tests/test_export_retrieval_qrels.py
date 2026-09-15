import json
import subprocess
import sys
from pathlib import Path

from clstr.retrieval_qrels import export_retrieval_qrels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_retrieval_qrels_normalizes_positive_and_negative_labels(tmp_path):
    source_path = tmp_path / "retrieval.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source": "toolret_training",
                "query_id": "q1",
                "positive_skill_id": "s1",
                "negative_skill_ids": ["s2", "s3"],
                "provenance": {"split": "train"},
            },
            {
                "source": "skillret",
                "query_id": "q2",
                "positive_skill_id": "s4",
                "negative_skill_ids": [],
                "provenance": json.dumps({"split": "train"}),
            },
        ],
    )

    report = export_retrieval_qrels(
        retrieval_path=source_path,
        output_path=qrels_path,
        include_negatives=True,
    )

    assert report["status"] == "ok"
    assert report["source_rows"] == 2
    assert report["positive_qrels"] == 2
    assert report["negative_qrels"] == 2
    assert read_jsonl(qrels_path) == [
        {"query_id": "q1", "skill_id": "s1", "relevance": 1, "source": "toolret_training", "split": "train"},
        {"query_id": "q1", "skill_id": "s2", "relevance": 0, "source": "toolret_training", "split": "train"},
        {"query_id": "q1", "skill_id": "s3", "relevance": 0, "source": "toolret_training", "split": "train"},
        {"query_id": "q2", "skill_id": "s4", "relevance": 1, "source": "skillret", "split": "train"},
    ]


def test_export_retrieval_qrels_filters_sources_and_unsafe_splits(tmp_path):
    source_path = tmp_path / "retrieval.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source": "toolret_training",
                "query_id": "train-q",
                "positive_skill_id": "tool-a",
                "provenance": {"split": "train"},
            },
            {
                "source": "traject_bench",
                "query_id": "public-q",
                "positive_skill_id": "tool-b",
                "provenance": {"split": "public_train_or_eval_unlabeled"},
            },
            {
                "source": "skillret",
                "query_id": "other-q",
                "positive_skill_id": "skill-a",
                "provenance": {"split": "train"},
            },
        ],
    )

    report = export_retrieval_qrels(
        retrieval_path=source_path,
        output_path=qrels_path,
        allowed_sources={"toolret_training", "traject_bench"},
        allowed_splits={"train"},
    )

    assert report["status"] == "ok"
    assert report["positive_qrels"] == 1
    assert report["skipped_by_source"] == {"skillret": 1}
    assert report["skipped_by_split"] == {"public_train_or_eval_unlabeled": 1}
    assert read_jsonl(qrels_path) == [
        {
            "query_id": "train-q",
            "skill_id": "tool-a",
            "relevance": 1,
            "source": "toolret_training",
            "split": "train",
        }
    ]


def test_export_retrieval_qrels_cli_writes_report(tmp_path):
    source_path = tmp_path / "retrieval.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(
        source_path,
        [
            {
                "source": "toolret_training",
                "query_id": "q1",
                "positive_skill_id": "s1",
                "provenance": {"split": "train"},
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/export_retrieval_qrels.py",
            "--retrieval_path",
            str(source_path),
            "--output_path",
            str(qrels_path),
            "--report_path",
            str(report_path),
            "--allowed_sources",
            "toolret_training",
            "--allowed_splits",
            "train",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["qrels_path"] == str(qrels_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert read_jsonl(qrels_path)[0]["skill_id"] == "s1"
