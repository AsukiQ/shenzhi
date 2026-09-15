import json
import subprocess
import sys
from pathlib import Path

from clstr.toolbench_g3_routing_eval import prepare_toolbench_g3_routing_eval


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_prepare_toolbench_g3_routing_eval_deduplicates_queries_and_aggregates_qrels(tmp_path):
    retrieval_path = tmp_path / "data/toolbench_g3/retrieval.jsonl"
    output_dir = tmp_path / "data/toolbench_g3_routing_eval"
    write_jsonl(
        retrieval_path,
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-1",
                "query_text": "Find cocktails and news.",
                "positive_skill_id": "toolbench-g3/cocktail/list",
                "provenance": {"split": "train_or_released_g3"},
            },
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-1",
                "query_text": "Find cocktails and news.",
                "positive_skill_id": "toolbench-g3/web/newssearch",
                "provenance": {"split": "train_or_released_g3"},
            },
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-1",
                "query_text": "Find cocktails and news.",
                "positive_skill_id": "toolbench-g3/web/newssearch",
                "provenance": {"split": "train_or_released_g3"},
            },
        ],
    )

    report = prepare_toolbench_g3_routing_eval(retrieval_path=retrieval_path, output_dir=output_dir)

    assert report["status"] == "ok"
    assert report["source_rows"] == 3
    assert report["query_count"] == 1
    assert report["positive_qrels"] == 2
    assert read_jsonl(output_dir / "queries.jsonl") == [
        {
            "query_id": "toolbench-g3-1",
            "query_text": "Find cocktails and news.",
            "source": "toolbench_g3",
            "split": "train_or_released_g3",
        }
    ]
    assert read_jsonl(output_dir / "qrels.jsonl") == [
        {
            "query_id": "toolbench-g3-1",
            "skill_id": "toolbench-g3/cocktail/list",
            "relevance": 1,
            "source": "toolbench_g3",
            "split": "train_or_released_g3",
        },
        {
            "query_id": "toolbench-g3-1",
            "skill_id": "toolbench-g3/web/newssearch",
            "relevance": 1,
            "source": "toolbench_g3",
            "split": "train_or_released_g3",
        },
    ]


def test_prepare_toolbench_g3_routing_eval_cli_writes_report(tmp_path):
    retrieval_path = tmp_path / "data/toolbench_g3/retrieval.jsonl"
    output_dir = tmp_path / "data/toolbench_g3_routing_eval"
    write_jsonl(
        retrieval_path,
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-2",
                "query_text": "Find stocks.",
                "positive_skill_id": "toolbench-g3/stocks/ohlc",
                "provenance": {"split": "train_or_released_g3"},
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_toolbench_g3_routing_eval.py",
            "--retrieval_path",
            str(retrieval_path),
            "--output_dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    assert report["queries_path"] == str(output_dir / "queries.jsonl")
    assert report["qrels_path"] == str(output_dir / "qrels.jsonl")
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == report
