from __future__ import annotations

import json
from pathlib import Path
import subprocess

from clstr.toolrex_training_data import build_toolrex_unified_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_toolrex_unified_data_merges_retrieval_and_qrels_sources(tmp_path: Path):
    retrieval_root = tmp_path / "toolret_training"
    _write_jsonl(
        retrieval_root / "skills.jsonl",
        [
            {"skill_id": "toolret/weather", "name": "weather", "description": "Get weather by city."},
            {"skill_id": "toolret/news", "name": "news", "description": "Search news articles."},
        ],
    )
    _write_jsonl(
        retrieval_root / "retrieval.jsonl",
        [
            {
                "query_id": "r1",
                "query_text": "Find the weather and latest news for Paris.",
                "positive_skill_id": "toolret/weather",
                "negative_skill_ids": ["toolret/news"],
                "provenance": {"source_dataset": "ToolRet", "split": "train"},
            }
        ],
    )

    qrels_root = tmp_path / "toolbench_g3"
    _write_jsonl(
        qrels_root / "skills.jsonl",
        [
            {"skill_id": "g3/cocktails", "name": "cocktails", "description": "List cocktails."},
        ],
    )
    _write_jsonl(
        qrels_root / "queries.jsonl",
        [{"query_id": "q1", "query": "List cocktail recipes for a party."}],
    )
    _write_jsonl(
        qrels_root / "qrels.jsonl",
        [{"query_id": "q1", "skill_id": "g3/cocktails", "relevance": 1, "split": "train"}],
    )

    output_dir = tmp_path / "toolrex_unified"
    report = build_toolrex_unified_data(
        source_roots=[retrieval_root, qrels_root],
        output_dir=output_dir,
    )

    skills = _read_jsonl(output_dir / "skill_pool.jsonl")
    trajectories = _read_jsonl(output_dir / "trajectories.jsonl")
    retrieval = _read_jsonl(output_dir / "retrieval.jsonl")

    assert report["status"] == "ok"
    assert report["skill_count"] == 3
    assert report["trajectory_count"] == 2
    assert report["source_reports"]["toolret_training"]["query_positive_count"] == 1
    assert report["source_reports"]["toolbench_g3"]["query_positive_count"] == 1
    assert {row["skill_id"] for row in skills} == {"toolret/weather", "toolret/news", "g3/cocktails"}
    assert {row["next_skill_id"] for row in trajectories} == {"toolret/weather", "g3/cocktails"}
    assert all(row["provenance"]["split"] == "train" for row in trajectories)
    assert len(retrieval) == 2


def test_toolrex_builder_cli_and_sbatch_expose_sources_and_progress():
    cli = Path("scripts/build_bge_toolrex_training_data.py")
    sbatch = Path("scripts/sbatch/run_build_bge_toolrex_training_data.sh")

    assert cli.is_file()
    help_result = subprocess.run(
        ["/data/home/scyb713/run/miniconda3/envs/xzf/bin/python", str(cli), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--source_roots" in help_result.stdout
    assert "--output_dir" in help_result.stdout
    assert "progress.json" in cli.read_text(encoding="utf-8")
    script = sbatch.read_text(encoding="utf-8")
    assert "SOURCE_ROOTS=${SOURCE_ROOTS:-data/toolret_training,data/toolbench_g3}" in script
    assert "scripts/build_bge_toolrex_training_data.py" in script
    assert "tee \"${OUTPUT_DIR}/stdout.log\"" in script


def test_toolrex_embedding_cli_uses_tool_embed_method_not_sr_adapter():
    cli = Path("scripts/run_bge_toolrex_embedding_full_train.py")
    sbatch = Path("scripts/sbatch/run_bge_toolrex_embedding_full_train.sh")

    assert cli.is_file()
    result = subprocess.run(
        ["/data/home/scyb713/run/miniconda3/envs/xzf/bin/python", str(cli), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--data_root" in result.stdout
    assert "--prepared_corpus_dir" in result.stdout
    assert "--negatives_per_query" in result.stdout
    text = cli.read_text(encoding="utf-8")
    assert 'method_name="bge_tool_embed_full_encoder_finetune"' in text
    assert 'checkpoint_prefix="bge-m3-tool-embed"' in text
    assert 'doc_template="skillret_official_full_text"' in text
    assert 'training_objective="multi_positive_inbatch_infonce_with_uniform_catalog_negatives"' in text
    assert "hard_negative_top_k=0" in text
    assert "max_hard_negative_queries=0" in text
    assert "adapter" not in text.lower()

    script = sbatch.read_text(encoding="utf-8")
    assert "BATCH_SIZE=${BATCH_SIZE:-64}" in script
    assert "NEGATIVES_PER_QUERY=${NEGATIVES_PER_QUERY:-5}" in script
    assert "scripts/run_bge_toolrex_embedding_full_train.py" in script
    assert "--resume_checkpoint_path" in script
    assert "PREPARED_CORPUS_DIR=${PREPARED_CORPUS_DIR:-}" in script
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in script
    assert "/autodl-tmp/clstr}" not in script
