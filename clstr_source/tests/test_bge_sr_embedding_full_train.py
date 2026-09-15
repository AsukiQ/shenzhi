from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.bge_sr_embedding_full_train import (
    _build_embedding_training_monitor,
    _copy_bge_pooling_config,
    _ensure_finite_loss,
    _ensure_finite_gradients,
    _ensure_finite_tensor,
    _load_embedding_resume_state,
    _mine_hard_negative_indices,
    _multi_positive_legal_nll,
    _prepared_sampling_groups,
    _prepared_source_balanced_batch,
    _select_semantic_hard_negative_queries,
    _should_build_semantic_hard_negatives,
    _write_embedding_progress,
    sample_negative_skill_ids,
)
from clstr.baseline_corpus_types import UnifiedSkillRouterQuery


def test_mine_hard_negative_indices_excludes_positive_indices():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.1],
            [0.5, 0.4, 0.3, 0.2],
        ]
    )

    mined = _mine_hard_negative_indices(scores, [{0, 1}, {2}], top_k=2)

    assert mined == [[2, 3], [0, 1]]


def test_should_build_semantic_hard_negatives_can_disable_mining_for_toolrex():
    assert _should_build_semantic_hard_negatives(hard_negative_top_k=32, max_hard_negative_queries=100) is True
    assert _should_build_semantic_hard_negatives(hard_negative_top_k=0, max_hard_negative_queries=100) is False
    assert _should_build_semantic_hard_negatives(hard_negative_top_k=32, max_hard_negative_queries=0) is False


def test_sample_negative_skill_ids_excludes_all_positive_aliases():
    skills = ["skill/a", "skill/b", "skill/c", "skill/d"]
    hard = {"q0": ["skill/a", "skill/c", "skill/d"]}

    sampled = sample_negative_skill_ids(
        query_id="q0",
        positive_skill_ids={"skill/a", "skill/b"},
        all_skill_ids=skills,
        hard_negatives_by_query=hard,
        count=2,
        rng_seed=7,
    )

    assert sampled == ["skill/c", "skill/d"]


def test_sample_negative_skill_ids_never_leaves_query_catalog():
    sampled = sample_negative_skill_ids(
        query_id="q0",
        positive_skill_ids={"skill/a"},
        all_skill_ids=["skill/a", "skill/b", "skill/c", "skill/d"],
        candidate_skill_ids=["skill/a", "skill/c"],
        hard_negatives_by_query={"q0": ["skill/b", "skill/c", "skill/d"]},
        count=3,
        rng_seed=7,
    )

    assert sampled == ["skill/c"]


def test_sample_negative_skill_ids_global_pool_is_unique_and_excludes_positives():
    sampled = sample_negative_skill_ids(
        query_id="q0",
        positive_skill_ids={"skill/a", "skill/b"},
        all_skill_ids=[f"skill/{index}" for index in range(100)] + ["skill/a", "skill/b"],
        candidate_skill_ids=None,
        hard_negatives_by_query={},
        count=8,
        rng_seed=7,
    )

    assert len(sampled) == len(set(sampled)) == 8
    assert not set(sampled) & {"skill/a", "skill/b"}


def test_semantic_hard_negative_query_cap_balances_sources_and_capabilities():
    queries = [
        UnifiedSkillRouterQuery(
            f"{kind}-{source}-{index}",
            "query",
            source,
            ["skill/a"],
            [0],
            source_id=source,
            kind=kind,
        )
        for kind in ("retrieval", "static_route")
        for source in ("tau2", "toolsandbox")
        for index in range(3)
    ]

    selected, report = _select_semantic_hard_negative_queries(
        queries,
        max_queries=4,
        source_balanced=True,
        seed=17,
    )

    assert len(selected) == 4
    assert {(query.kind, query.source_id) for query in selected} == {
        ("retrieval", "tau2"),
        ("retrieval", "toolsandbox"),
        ("static_route", "tau2"),
        ("static_route", "toolsandbox"),
    }
    assert report["protocol"] == "capability_first_source_round_robin_cap_v2"
    assert report["selected_query_count"] == 4


def test_mine_hard_negative_indices_respects_legal_candidates():
    scores = torch.tensor([[0.99, 0.8, 0.7, 0.6]])

    mined = _mine_hard_negative_indices(
        scores,
        [{0}],
        top_k=2,
        legal_indices_by_query=[{0, 2, 3}],
    )

    assert mined == [[2, 3]]


def test_multi_positive_legal_nll_ignores_illegal_in_batch_documents():
    logits = torch.tensor([[2.0, 1.0, 100.0]])
    positive = torch.tensor([[True, True, False]])
    legal = torch.tensor([[True, True, False]])

    loss = _multi_positive_legal_nll(logits, positive, legal)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_prepared_sampling_balances_capabilities_and_sources():
    queries = [
        UnifiedSkillRouterQuery("r0", "r0", "b", ["s"], [0], source_id="a", kind="retrieval"),
        UnifiedSkillRouterQuery("r1", "r1", "b", ["s"], [0], source_id="b", kind="retrieval"),
        UnifiedSkillRouterQuery("s0", "s0", "b", ["s"], [0], source_id="a", kind="static_route"),
        UnifiedSkillRouterQuery("s1", "s1", "b", ["s"], [0], source_id="b", kind="static_route"),
    ]
    groups = _prepared_sampling_groups(queries)

    batch = _prepared_source_balanced_batch(
        groups,
        global_batch_index=0,
        batch_size=4,
        seed=13,
    )

    assert sum(query.kind == "retrieval" for query in batch) == 2
    assert sum(query.kind == "static_route" for query in batch) == 2
    assert {query.source_id for query in batch if query.kind == "retrieval"} == {"a", "b"}
    assert {query.source_id for query in batch if query.kind == "static_route"} == {"a", "b"}


def test_prepared_sampling_alternates_the_odd_batch_slot():
    queries = [
        UnifiedSkillRouterQuery("r0", "r0", "b", ["s"], [0], source_id="a", kind="retrieval"),
        UnifiedSkillRouterQuery("s0", "s0", "b", ["s"], [0], source_id="a", kind="static_route"),
    ]
    groups = _prepared_sampling_groups(queries)

    first = _prepared_source_balanced_batch(
        groups,
        global_batch_index=0,
        batch_size=3,
        seed=13,
    )
    second = _prepared_source_balanced_batch(
        groups,
        global_batch_index=1,
        batch_size=3,
        seed=13,
    )

    combined = first + second
    assert sum(query.kind == "retrieval" for query in combined) == 3
    assert sum(query.kind == "static_route" for query in combined) == 3


def test_copy_bge_pooling_config_preserves_cls_pooling(tmp_path):
    source = tmp_path / "bge-m3"
    pooling = source / "1_Pooling"
    pooling.mkdir(parents=True)
    (pooling / "config.json").write_text(
        '{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}\n',
        encoding="utf-8",
    )
    target = tmp_path / "checkpoint"

    copied = _copy_bge_pooling_config(source, target)

    assert copied is True
    saved = json.loads((target / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    assert saved["pooling_mode_cls_token"] is True


def test_write_embedding_progress_writes_progress_json(tmp_path):
    progress = _write_embedding_progress(
        tmp_path,
        stage="training",
        step=2,
        max_steps=5,
        extra={"loss": 1.0},
    )

    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "training"
    assert saved["step"] == 2
    assert saved["max_steps"] == 5
    assert saved["loss"] == 1.0
    assert progress == saved


def test_bge_sr_embedding_training_monitor_writes_curves(tmp_path):
    monitor = _build_embedding_training_monitor(
        tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=2,
    )

    assert (tmp_path / "loss_curve.svg").exists()
    assert (tmp_path / "diagnostic_curves.svg").exists()

    monitor.record({"step": 1, "loss": 1.0, "train_recall@1": 0.25})

    rows = (tmp_path / "training_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["loss"] == 1.0


def test_load_embedding_resume_state_reads_training_state(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "bge-m3-sr-emb-step400"
    checkpoint.mkdir(parents=True)
    torch.save(
        {
            "step": 400,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "last_metrics": {"loss": 1.5},
        },
        checkpoint / "training_state.pt",
    )

    resume = _load_embedding_resume_state(checkpoint)

    assert resume["step"] == 400
    assert resume["checkpoint_path"] == str(checkpoint)
    assert resume["optimizer_state_dict"] == {"state": {}, "param_groups": []}
    assert resume["last_metrics"]["loss"] == 1.5


def test_ensure_finite_loss_writes_blocker_before_raising(tmp_path):
    loss = torch.tensor(float("nan"))

    try:
        _ensure_finite_loss(loss, output_dir=tmp_path, step=7)
    except FloatingPointError as exc:
        assert "non-finite loss" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FloatingPointError")

    blocker = json.loads((tmp_path / "blocker_report.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked"
    assert blocker["step"] == 7
    assert progress["status"] == "blocked"


def test_ensure_finite_tensor_writes_named_blocker_before_raising(tmp_path):
    values = torch.tensor([0.0, float("inf")])

    try:
        _ensure_finite_tensor(values, output_dir=tmp_path, step=11, name="query_embeddings")
    except FloatingPointError as exc:
        assert "non-finite query_embeddings" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FloatingPointError")

    blocker = json.loads((tmp_path / "blocker_report.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked"
    assert blocker["reason"] == "non-finite query_embeddings"
    assert blocker["step"] == 11
    assert progress["status"] == "blocked"
    assert progress["reason"] == "non-finite query_embeddings"


def test_ensure_finite_gradients_can_zero_small_nonfinite_fraction(tmp_path):
    model = torch.nn.Linear(4, 2)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.tensor([float("nan"), 1.0])

    report = _ensure_finite_gradients(
        model,
        output_dir=tmp_path,
        step=3,
        action="zero",
        max_nonfinite_fraction=0.5,
    )

    assert report["nonfinite_gradient_count"] == 1
    assert report["sanitized_gradient_count"] == 1
    assert torch.isfinite(model.bias.grad).all()
    assert model.bias.grad[0].item() == 0.0


def test_bge_sr_embedding_full_train_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_bge_sr_embedding_full_train.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--encoder_model_path" in result.stdout
    assert "--negatives_per_query" in result.stdout
    assert "--hard_negative_top_k" in result.stdout
    assert "--output_dir" in result.stdout
    assert "--resume_checkpoint_path" in result.stdout
    assert "--nonfinite_gradient_action" in result.stdout
    assert "--max_nonfinite_gradient_fraction" in result.stdout
    assert "--prepared_corpus_dir" in result.stdout


def test_bge_sr_embedding_full_sbatch_defaults_to_float32_full_finetune():
    script = Path("scripts/sbatch/run_bge_sr_embedding_full_train.sh").read_text(encoding="utf-8")

    assert "TORCH_DTYPE=${TORCH_DTYPE:-float32}" in script
    assert "BATCH_SIZE=${BATCH_SIZE:-64}" in script
    assert "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}" in script
    assert "--gradient_accumulation_steps" in script
    assert "USE_BF16_AUTOCAST=${USE_BF16_AUTOCAST:-0}" in script
    assert "MINING_USE_BF16_AUTOCAST=${MINING_USE_BF16_AUTOCAST:-1}" in script
    assert "--mining_use_bf16_autocast" in script
    assert "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "--resume_checkpoint_path" in script
    assert "NONFINITE_GRADIENT_ACTION=${NONFINITE_GRADIENT_ACTION:-error}" in script
    assert "MAX_NONFINITE_GRADIENT_FRACTION=${MAX_NONFINITE_GRADIENT_FRACTION:-0.0}" in script
    assert "--nonfinite_gradient_action" in script
    assert "--max_nonfinite_gradient_fraction" in script
    assert "PREPARED_CORPUS_DIR=${PREPARED_CORPUS_DIR:-}" in script
    assert "--prepared_corpus_dir" in script
    assert "scripts/run_bge_sr_embedding_full_train.py" in script
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in script
    assert "/autodl-tmp/clstr}" not in script


def test_bge_sr_embedding_full_cli_exposes_separate_mining_autocast_flag():
    result = subprocess.run(
        [sys.executable, "scripts/run_bge_sr_embedding_full_train.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--mining_use_bf16_autocast" in result.stdout
    assert "--no-mining_use_bf16_autocast" in result.stdout
