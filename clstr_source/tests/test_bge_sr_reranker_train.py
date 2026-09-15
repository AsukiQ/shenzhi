from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.bge_reranker import bge_reranker_score_tensor
from clstr.bge_sr_reranker_train import (
    _group_batch_to_pairs,
    _listwise_batch_loss,
    _rank_candidates,
    _write_reranker_progress,
    build_listwise_reranker_groups,
)


class _RaggedTokenizer:
    def __call__(self, queries, docs, **_kwargs):
        assert len(queries) == len(docs)
        return {"input_ids": torch.arange(len(queries)).view(-1, 1)}


class _RaggedModel(torch.nn.Module):
    def forward(self, input_ids):
        return type("Output", (), {"logits": input_ids.float()})()


def test_bge_reranker_score_tensor_keeps_gradients_for_single_logit():
    logits = torch.tensor([[1.0], [-2.0]], requires_grad=True)

    scores = bge_reranker_score_tensor(logits)
    scores.sum().backward()

    assert torch.allclose(scores, torch.tensor([1.0, -2.0]))
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.ones_like(logits))


def test_build_listwise_reranker_groups_skips_rows_without_positive_candidate():
    skills = [
        {"skill_id": "skill/a", "name": "A", "description": "alpha", "body": "do alpha"},
        {"skill_id": "skill/b", "name": "B", "description": "beta", "body": "do beta"},
        {"skill_id": "skill/c", "name": "C", "description": "gamma", "body": "do gamma"},
    ]
    queries = [
        {"query_id": "q0", "query": "need beta", "positive_skill_ids": ["skill/b"]},
        {"query_id": "q1", "query": "need gamma", "positive_skill_ids": ["skill/c"]},
    ]
    ranked = {
        "q0": ["skill/a", "skill/b", "skill/c"],
        "q1": ["skill/a", "skill/b"],
    }

    groups, report = build_listwise_reranker_groups(
        queries=queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=2,
    )

    assert report["group_count"] == 1
    assert report["skipped_reasons"]["positive_not_in_topk"] == 1
    assert groups[0]["query_id"] == "q0"
    assert groups[0]["candidate_skill_ids"] == ["skill/a", "skill/b"]
    assert groups[0]["label_index"] == 1


def test_build_listwise_reranker_groups_preserves_multiple_positives():
    skills = [
        {"skill_id": "skill/a", "name": "A"},
        {"skill_id": "skill/b", "name": "B"},
        {"skill_id": "skill/c", "name": "C"},
    ]
    groups, _report = build_listwise_reranker_groups(
        queries=[
            {
                "query_id": "q0",
                "query": "need b or c",
                "positive_skill_ids": ["skill/b", "skill/c"],
            }
        ],
        skills=skills,
        ranked_skill_ids_by_query={"q0": ["skill/a", "skill/b", "skill/c"]},
        top_k=3,
    )

    assert groups[0]["positive_candidate_indices"] == [1, 2]


def test_write_reranker_progress_writes_progress_json(tmp_path):
    progress = _write_reranker_progress(
        tmp_path,
        stage="training",
        step=5,
        max_steps=20,
        extra={"loss": 0.75},
    )

    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "training"
    assert saved["step"] == 5
    assert saved["max_steps"] == 20
    assert saved["loss"] == 0.75
    assert progress == saved


def test_rank_candidates_respects_per_query_catalogs():
    ranked = _rank_candidates(
        query_ids=["q0", "q1"],
        query_embs=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        skill_embs=torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0]]),
        skill_ids=["skill/a", "skill/b", "skill/c"],
        top_k=2,
        batch_size=2,
        candidate_skill_ids_by_query=[
            ["skill/b", "skill/c"],
            ["skill/a", "skill/c"],
        ],
    )

    assert ranked == {
        "q0": ["skill/b", "skill/c"],
        "q1": ["skill/a", "skill/c"],
    }


def test_listwise_reranker_loss_supports_ragged_local_catalogs():
    groups = [
        {
            "query": "q0",
            "candidate_texts": ["a", "b"],
            "label_index": 1,
            "positive_candidate_indices": [1],
        },
        {
            "query": "q1",
            "candidate_texts": ["a", "b", "c"],
            "label_index": 0,
            "positive_candidate_indices": [0, 2],
        },
    ]

    queries, docs, positive_mask, sizes = _group_batch_to_pairs(groups)
    loss, metrics = _listwise_batch_loss(
        _RaggedModel(),
        _RaggedTokenizer(),
        torch.device("cpu"),
        groups,
        max_length=32,
    )

    assert len(queries) == len(docs) == 5
    assert sizes == [2, 3]
    assert positive_mask.shape == (2, 3)
    assert torch.isfinite(loss)
    assert metrics["recall@1"] == 1.0


def test_bge_sr_reranker_train_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_bge_sr_reranker_train.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--sr_embedding_output_dir" in result.stdout
    assert "--sr_embedding_checkpoint_path" in result.stdout
    assert "--reranker_model_path" in result.stdout
    assert "--top_k" in result.stdout
    assert "--resume_checkpoint_path" in result.stdout
    assert "--candidate_cache_dir" in result.stdout


def test_bge_sr_reranker_train_sbatch_uses_full_sr_embedding_checkpoint_not_adapter():
    script = Path("scripts/sbatch/run_bge_sr_reranker_train.sh").read_text(encoding="utf-8")

    assert "scripts/run_bge_sr_reranker_train.py" in script
    assert "SR_EMBEDDING_OUTPUT_DIR" in script
    assert "SR_EMBEDDING_CHECKPOINT_PATH" in script
    assert "--sr_embedding_checkpoint_path" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "--resume_checkpoint_path" in script
    assert "CANDIDATE_CACHE_DIR=${CANDIDATE_CACHE_DIR:-${PROJECT_ROOT}/outputs/shared_exact_candidate_cache}" in script
    assert "--candidate_cache_dir" in script
    assert "ADAPTER_CHECKPOINT_PATH" not in script
    assert "models/BAAI/bge-reranker-v2-m3" in script
    assert "stdout.log" in script
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in script
    assert "/autodl-tmp/clstr}" not in script


def test_bge_sr_reranker_train_uses_training_monitor_for_curves():
    source = Path("clstr/bge_sr_reranker_train.py").read_text(encoding="utf-8")

    assert "TrainingMonitor" in source
    assert "optimizer_state_dict" in source
    assert "resume_checkpoint_path" in source
