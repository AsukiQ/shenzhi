from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from clstr.bge_toolrex_reranker_train import build_toolrank_pairwise_pairs


def test_build_toolrank_pairwise_pairs_creates_true_false_examples() -> None:
    skills = [
        {"skill_id": "skill/a", "name": "A", "description": "alpha", "body": "do alpha"},
        {"skill_id": "skill/b", "name": "B", "description": "beta", "body": "do beta"},
        {"skill_id": "skill/c", "name": "C", "description": "gamma", "body": "do gamma"},
    ]
    queries = [
        {"query_id": "q0", "query": "need alpha", "positive_skill_ids": ["skill/a"]},
        {"query_id": "q1", "query": "need missing", "positive_skill_ids": ["skill/missing"]},
    ]
    ranked = {
        "q0": ["skill/a", "skill/b", "skill/c"],
        "q1": ["skill/b", "skill/c"],
    }

    pairs, report = build_toolrank_pairwise_pairs(
        queries=queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=3,
        negatives_per_query=2,
    )

    assert report["pair_count"] == 3
    assert report["positive_pair_count"] == 1
    assert report["negative_pair_count"] == 2
    assert report["skipped_reasons"]["positive_missing_from_skills"] == 1
    assert [(row["skill_id"], row["label"]) for row in pairs] == [
        ("skill/a", 1),
        ("skill/b", 0),
        ("skill/c", 0),
    ]


def test_bge_toolrex_reranker_train_cli_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_bge_toolrex_reranker_train.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--tool_embed_output_dir" in result.stdout
    assert "--tool_embed_checkpoint_path" in result.stdout
    assert "--reranker_model_path" in result.stdout
    assert "--negatives_per_query" in result.stdout
    assert "--resume_checkpoint_path" in result.stdout
    assert "--candidate_cache_dir" in result.stdout


def test_bge_toolrex_reranker_sbatch_uses_pairwise_objective_and_batch64() -> None:
    script = Path("scripts/sbatch/run_bge_toolrex_reranker_train.sh").read_text(encoding="utf-8")

    assert "scripts/run_bge_toolrex_reranker_train.py" in script
    assert "TOOL_EMBED_OUTPUT_DIR" in script
    assert "TOOL_EMBED_CHECKPOINT_PATH" in script
    assert "BATCH_SIZE=${BATCH_SIZE:-64}" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "CANDIDATE_CACHE_DIR=${CANDIDATE_CACHE_DIR:-${PROJECT_ROOT}/outputs/shared_exact_candidate_cache}" in script
    assert "--candidate_cache_dir" in script
    assert "models/BAAI/bge-reranker-v2-m3" in script
    assert "ADAPTER" not in script
    assert "listwise" not in script.lower()
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in script
    assert "/autodl-tmp/clstr}" not in script
