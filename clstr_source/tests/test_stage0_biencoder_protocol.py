from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import clstr.retrieval_warmup as retrieval_warmup
from clstr.encoders import SkillTable, pool_hidden
from clstr.retrieval_warmup import (
    STAGE0_BIENCODER_DEFAULTS,
    compute_unified_static_warmup_loss,
    _save_named_stage0_checkpoint,
    _parameter_anchor_loss,
    _load_stage0_resume_state,
    _snapshot_trainable_parameters,
    run_stage0_biencoder_train,
    stage0_biencoder_scores,
)


def test_stage0_named_checkpoint_persists_step_zero_optimizer_and_frozen_metadata(tmp_path):
    payload = {
        "stage": "clstr_unified_retrieval_v2",
        "step": 0,
        "setup_phase": "training_started",
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "checkpoint_excludes_frozen_backbone": True,
        "config": {
            "base_model_name": "models/Qwen3-Embedding-0.6B",
            "freeze_backbone": True,
            "route_scorer": "unified_memory",
            "state_query_prompt_version": "clstr_causal_state_v1",
            "state_query_instruction": (
                "Given an agent task or current execution state and interaction history, "
                "retrieve the skill or tool document most useful for the next action."
            ),
            "state_query_max_chars": 2000,
            "state_query_truncation": "head_tail_v1",
        },
    }

    path = _save_named_stage0_checkpoint(
        checkpoint_dir=tmp_path,
        stage_name="clstr_unified_retrieval_v2",
        step=0,
        payload=payload,
    )

    loaded = torch.load(path, map_location="cpu")
    assert path.name == "clstr_unified_retrieval_v2-step0.pt"
    assert loaded["step"] == 0
    assert loaded["optimizer_state_dict"] == payload["optimizer_state_dict"]
    assert loaded["checkpoint_excludes_frozen_backbone"] is True


def test_stage0_biencoder_defaults_match_skillrouter_protocol():
    assert STAGE0_BIENCODER_DEFAULTS["encoder_pooling"] == "last_token"
    assert STAGE0_BIENCODER_DEFAULTS["cross_encoder_pooling"] == "last_token"
    assert STAGE0_BIENCODER_DEFAULTS["tokenizer_padding_side"] == "left"
    assert STAGE0_BIENCODER_DEFAULTS["projection_init"] == "identity"
    assert STAGE0_BIENCODER_DEFAULTS["skill_table_adapter_init"] == "identity"
    assert STAGE0_BIENCODER_DEFAULTS["normalize_embeddings"] is True
    assert STAGE0_BIENCODER_DEFAULTS["skill_text_format"] == "skillret_official"
    assert STAGE0_BIENCODER_DEFAULTS["query_text_format"] == "skillrouter"
    assert STAGE0_BIENCODER_DEFAULTS["state_query_prompt_version"] is None
    assert STAGE0_BIENCODER_DEFAULTS["data_format"] == "unified_v2"
    assert STAGE0_BIENCODER_DEFAULTS["sampling_strategy"] == "handoff_balanced"
    assert STAGE0_BIENCODER_DEFAULTS["use_cross_encoder"] is False
    assert STAGE0_BIENCODER_DEFAULTS["explicit_negative_loss_weight"] == 0.0
    assert STAGE0_BIENCODER_DEFAULTS["explicit_negative_margin"] == 0.1
    assert STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_loss_weight"] == 0.0
    assert STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_margin"] == 0.1
    assert STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_top_k"] == 32


def test_unified_v2_loader_preserves_explicit_negative_skill_ids(tmp_path):
    data_root = tmp_path / "unified"
    data_root.mkdir()
    (data_root / "skill_pool.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"skill_id": "skill/a", "name": "A"},
                {"skill_id": "skill/b", "name": "B"},
                {"skill_id": "skill/c", "name": "C"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_root / "retrieval.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "query_id": "q1",
                    "source": "source-a",
                    "query_text": "find the right tool",
                    "positive_skill_id": "skill/a",
                    "negative_skill_ids": ["skill/b"],
                    "split": "train",
                },
                {
                    "query_id": "q1",
                    "source": "source-a",
                    "query_text": "find the right tool",
                    "positive_skill_id": "skill/a",
                    "negative_skill_ids": ["skill/b", "skill/c"],
                    "split": "train",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _skills, queries, positives = retrieval_warmup.load_unified_v2_retrieval_rows(data_root)

    assert positives == {"q1": ["skill/a", "skill/a"]}
    assert queries[0]["negative_skill_ids"] == ["skill/b", "skill/c"]


def test_stage0_filter_resolves_negative_indices_and_excludes_positive_ids():
    queries = [
        {
            "query_id": "q1",
            "query": "find the right tool",
            "positive_skill_ids": ["skill/a"],
            "negative_skill_ids": ["skill/a", "skill/b", "missing"],
        }
    ]
    positives = {"q1": ["skill/a"]}
    skill_id_to_idx = {"skill/a": 0, "skill/b": 1}

    filtered = retrieval_warmup._filter_queries_to_skill_pool(
        queries,
        positives,
        skill_id_to_idx,
        max_queries=None,
    )

    assert filtered[0]["positive_indices"] == [0]
    assert filtered[0]["negative_indices"] == [1]
    assert filtered[0]["resolved_negative_skill_ids"] == ["skill/b"]


def test_stage0_explicit_negative_margin_loss_penalizes_high_scored_negatives():
    logits = torch.tensor([[1.0, 1.4, -1.0]])

    loss, report = retrieval_warmup._explicit_negative_margin_loss(
        logits,
        positive_indices=[[0]],
        negative_indices=[[1, 2]],
        margin=0.1,
    )

    assert torch.allclose(loss, torch.tensor(0.25))
    assert report["explicit_negative_rows"] == 1
    assert report["explicit_negative_pair_count"] == 2


def test_stage0_mined_hard_negative_margin_loss_mines_top_non_positive_logits():
    logits = torch.tensor([[1.0, 1.4, 0.7, -1.0]])

    loss, report = retrieval_warmup._mined_hard_negative_margin_loss(
        logits,
        positive_indices=[[0]],
        top_k=2,
        margin=0.1,
    )

    assert torch.allclose(loss, torch.tensor(0.25))
    assert report["mined_hard_negative_rows"] == 1
    assert report["mined_hard_negative_pair_count"] == 2


def test_stage0_handoff_balanced_sampler_splits_trajectory_current_and_next():
    rows = [
        {
            "query_id": "traj-current",
            "source_dataset": "unused",
            "metadata": {
                "source": "trajectory_derived_traject_bench",
                "provenance": {"target": "current"},
            },
        },
        {
            "query_id": "traj-next",
            "source_dataset": "unused",
            "metadata": {
                "source": "trajectory_derived_traject_bench",
                "provenance": {"target": "next"},
            },
        },
        {
            "query_id": "skillret",
            "source_dataset": "skillret",
            "metadata": {"source": "skillret"},
        },
    ]

    buckets = retrieval_warmup._build_training_buckets(rows, sampling_strategy="handoff_balanced")

    assert sorted(buckets) == [
        "skillret",
        "trajectory_derived_traject_bench:current",
        "trajectory_derived_traject_bench:next",
    ]


def test_stage0_training_bucket_counts_are_sorted_for_audit_reports():
    rows = [
        {"query_id": "skillret-1", "metadata": {"source": "skillret"}},
        {"query_id": "skillret-2", "metadata": {"source": "skillret"}},
        {
            "query_id": "traj-next",
            "metadata": {
                "source": "trajectory_derived_traject_bench",
                "provenance": {"target": "next"},
            },
        },
    ]
    buckets = retrieval_warmup._build_training_buckets(rows, sampling_strategy="handoff_balanced")

    counts = retrieval_warmup._training_bucket_counts(buckets)

    assert counts == {
        "skillret": 2,
        "trajectory_derived_traject_bench:next": 1,
    }


def test_stage0_handoff_tempered_sampler_caps_missing_positive_slots():
    rows = []
    base_sources = [
        "skillret",
        "toolret_training",
        "toolbench_g3",
        "traject_bench",
        "trajectory_derived_alfworld",
        "trajectory_derived_webshop",
        "trajectory_derived_toolbench_g3",
        "trajectory_derived_traject_bench",
        "trajectory_derived_other",
        "toolret_extra",
        "skillret_extra",
        "webshop_extra",
    ]
    correction_sources = [
        "toolbench_g3_stage0_balanced_missing_positive",
        "toolret_training_stage0_balanced_missing_positive",
        "traject_bench_stage0_balanced_missing_positive",
        "trajectory_derived_toolbench_g3_stage0_balanced_missing_positive",
        "trajectory_derived_traject_bench_stage0_balanced_missing_positive",
        "toolbench_g3_stage0_missing_positive",
        "traject_bench_stage0_missing_positive",
    ]
    for source in base_sources + correction_sources:
        for idx in range(3):
            rows.append({"query_id": f"{source}-{idx}", "metadata": {"source": source}})
    buckets = retrieval_warmup._build_training_buckets(rows, sampling_strategy="handoff_tempered")

    batch = retrieval_warmup._batch_queries_for_step(
        rows,
        step=1,
        batch_size=64,
        sampling_strategy="handoff_tempered",
        source_buckets=buckets,
        tempered_correction_fraction=0.2,
    )

    correction_count = sum(
        "stage0" in retrieval_warmup._query_source(row) and "missing_positive" in retrieval_warmup._query_source(row)
        for row in batch
    )
    assert correction_count == 13
    assert correction_count / len(batch) <= 0.21


def test_stage0_last_token_pooling_uses_final_token_for_left_padding():
    hidden = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]],
            [[0.0, 0.0], [2.0, 2.0], [8.0, 8.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])

    pooled = pool_hidden(hidden, mask, "last_token")

    assert torch.equal(pooled, torch.tensor([[9.0, 9.0], [8.0, 8.0]]))


def test_stage0_biencoder_scores_l2_normalize_query_and_skill_embeddings():
    query_embs = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    skill_embs = torch.tensor([[30.0, 0.0], [0.0, 40.0]])

    scores = stage0_biencoder_scores(query_embs, skill_embs)

    expected = F.normalize(query_embs, p=2, dim=-1) @ F.normalize(skill_embs, p=2, dim=-1).t()
    assert torch.allclose(scores, expected)
    assert torch.allclose(scores, torch.eye(2))


def test_stage0_identity_skill_table_score_matches_frozen_biencoder_cosine():
    skill_embs = torch.tensor([[2.0, 0.0], [0.0, 5.0]])

    def encode(texts: list[str]) -> torch.Tensor:
        return skill_embs[: len(texts)].clone()

    table = SkillTable(
        [{"name": "weather"}, {"name": "news"}],
        encode,
        d=2,
        adapter_init="identity",
        trainable=False,
        logit_scale_retr_init=0.0,
    )
    query_embs = torch.tensor([[10.0, 0.0], [0.0, 3.0]])

    table_scores = table.retrieval_logits(query_embs)
    frozen_scores = stage0_biencoder_scores(query_embs, table.E)

    assert torch.allclose(table_scores, frozen_scores, atol=1.0e-6)


def test_unified_static_warmup_loss_uses_initial_belief_and_unified_route_logits():
    class RecordingUnifiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query_embs = {
                "query zero": torch.tensor([1.0, 0.0]),
                "query one": torch.tensor([0.0, 1.0]),
            }
            self.skill_embs = torch.tensor(
                [
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [1.0, 1.0],
                ]
            )
            self.initial_belief_inputs = []
            self.unified_route_inputs = []

        def encode_states(self, texts):
            return torch.stack([self.query_embs[str(text)] for text in texts])

        def initial_belief(self, h_t, top_k=None):
            self.initial_belief_inputs.append((h_t.detach().clone(), top_k))
            return h_t + 1.0

        def unified_route_logits(self, h_t, m_t, candidate_rows=None):
            self.unified_route_inputs.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
            return (h_t + m_t) @ self.skill_embs.t()

    model = RecordingUnifiedModel()
    query_texts = ["query zero", "query one"]
    positive_indices = [[1, 2], [0]]

    loss, metrics = compute_unified_static_warmup_loss(
        model,
        query_texts,
        positive_indices,
        belief_top_k=2,
    )

    h = model.encode_states(query_texts)
    expected_logits = (h + h + 1.0) @ model.skill_embs.t()
    expected_loss = retrieval_warmup._multi_positive_nll(expected_logits, positive_indices)
    assert torch.allclose(loss, expected_loss)
    assert metrics["route_scorer"] == "unified_memory"
    assert metrics["unified_static_query_count"] == 2
    assert metrics["unified_static_recall_at_1"] == 0.5
    assert len(model.initial_belief_inputs) == 1
    assert model.initial_belief_inputs[0][1] == 2
    assert len(model.unified_route_inputs) == 1
    assert model.unified_route_inputs[0][2] is None


def test_stage0_biencoder_train_wrapper_forces_protocol_kwargs(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "model_config": kwargs}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        model_dim=1024,
        max_steps=5,
        use_cross_encoder=True,
    )

    assert report["status"] == "ok"
    assert captured["encoder_pooling"] == "last_token"
    assert captured["cross_encoder_pooling"] == "last_token"
    assert captured["tokenizer_padding_side"] == "left"
    assert captured["projection_init"] == "identity"
    assert captured["skill_table_adapter_init"] == "identity"
    assert captured["normalize_embeddings"] is True
    assert captured["skill_text_format"] == "skillret_official"
    assert captured["query_text_format"] == "skillrouter"
    assert captured["data_format"] == "unified_v2"
    assert captured["use_cross_encoder"] is False
    assert captured["max_steps"] == 5
    assert captured["retrieval_loss_mode"] == "multi_positive_nll"
    assert captured["sampling_strategy"] == "handoff_balanced"
    assert captured["tempered_correction_fraction"] == 0.2
    assert captured["expand_alias_positives"] is True
    assert captured["train_skill_embeddings"] is False
    assert captured["train_skill_bias"] is False
    assert captured["train_encoder_projection"] is False
    assert captured["train_skill_adapter"] is True
    assert captured["train_retrieval_scale"] is True
    assert captured["learning_rate"] <= 2.0e-5
    assert captured["gradient_accumulation_steps"] == 2
    assert captured["protocol_metadata"]["role"] == "SkillRouter-compatible bi-encoder coarse retriever"
    assert captured["protocol_metadata"]["retrieval_loss_mode"] == "multi_positive_nll"


def test_stage0_biencoder_train_wrapper_allows_bge_pooling_and_raw_queries(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "model_config": kwargs}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="models/BAAI/bge-m3",
        output_dir=tmp_path / "out",
        model_dim=1024,
        max_steps=5,
        encoder_pooling="cls",
        cross_encoder_pooling="cls",
        tokenizer_padding_side="right",
        query_text_format="raw",
    )

    assert report["status"] == "ok"
    assert captured["encoder_pooling"] == "cls"
    assert captured["cross_encoder_pooling"] == "cls"
    assert captured["tokenizer_padding_side"] == "right"
    assert captured["query_text_format"] == "raw"
    assert captured["protocol_metadata"]["encoder_pooling"] == "cls"
    assert captured["protocol_metadata"]["query_text_format"] == "raw"


def test_stage0_biencoder_records_explicit_causal_state_query_contract(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        retrieval_warmup,
        "run_skillret_retrieval_warmup",
        fake_run_skillret_retrieval_warmup,
    )

    run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=1,
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
    )

    assert captured["state_query_prompt_version"] == "clstr_causal_state_v1"
    assert captured["state_query_max_chars"] == 2000
    assert captured["state_query_truncation"] == "head_tail_v1"
    assert captured["protocol_metadata"]["state_query_prompt_version"] == "clstr_causal_state_v1"


def test_stage0_raw_query_texts_do_not_prepend_instruction():
    rows = [
        {
            "query": (
                "goal: inspect invoice\n"
                "observation: inbox\n"
                "previous_tools: open_mail"
            )
        }
    ]

    assert retrieval_warmup._stage0_raw_query_texts(rows) == [
        "goal: inspect invoice\nobservation: inbox"
    ]


def test_stage0_prompt_contract_maps_legacy_format_and_accepts_causal_override():
    legacy = retrieval_warmup._resolve_stage0_state_query_contract(
        query_text_format="skillrouter",
        state_query_prompt_version=None,
        state_query_max_chars=None,
        state_query_truncation=None,
    )
    causal = retrieval_warmup._resolve_stage0_state_query_contract(
        query_text_format="skillrouter",
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
    )

    assert legacy["state_query_prompt_version"] == "sr_task_description_v1"
    assert legacy["state_query_max_chars"] == 1500
    assert causal["state_query_prompt_version"] == "clstr_causal_state_v1"
    assert causal["state_query_max_chars"] == 2000
    assert causal["state_query_truncation"] == "head_tail_v1"


def test_stage0_cli_exposes_state_query_prompt_contract():
    proc = subprocess.run(
        [sys.executable, "scripts/run_clstr_stage0_biencoder_train.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--state_query_prompt_version" in proc.stdout
    assert "--state_query_max_chars" in proc.stdout
    assert "--state_query_truncation" in proc.stdout


def test_stage0_biencoder_train_wrapper_passes_full_backbone_training(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "model_config": kwargs}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        freeze_backbone=False,
        train_encoder_backbone=True,
        encoder_backbone_learning_rate=5.0e-6,
    )

    assert report["status"] == "ok"
    assert captured["freeze_backbone"] is False
    assert captured["train_encoder_backbone"] is True
    assert captured["encoder_backbone_learning_rate"] == 5.0e-6
    assert captured["protocol_metadata"]["train_encoder_backbone"] is True


def test_stage0_checkpoint_state_dict_includes_backbone_only_when_requested():
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.backbone = torch.nn.Linear(2, 2)
    model.encoder.proj = torch.nn.Linear(2, 2)
    model.skill_table = torch.nn.Module()
    model.skill_table.W = torch.nn.Linear(2, 2, bias=False)
    model.skill_table.E = torch.nn.Parameter(torch.ones(2, 2))

    filtered_state, filtered_report = retrieval_warmup._warmup_checkpoint_state_dict(
        model,
        include_encoder_backbone=False,
    )
    full_state, full_report = retrieval_warmup._warmup_checkpoint_state_dict(
        model,
        include_encoder_backbone=True,
    )

    assert "encoder.backbone.weight" not in filtered_state
    assert "encoder.backbone.weight" in full_state
    assert filtered_report["checkpoint_includes_encoder_backbone"] is False
    assert full_report["checkpoint_includes_encoder_backbone"] is True


def test_stage0_resume_state_reads_model_and_optimizer_state(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "step": 400,
            "model_state_dict": {"encoder.proj.weight": torch.ones(2, 2)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "metrics": {"loss": 1.25},
        },
        checkpoint,
    )

    resume = _load_stage0_resume_state(checkpoint)

    assert resume["step"] == 400
    assert resume["checkpoint_path"] == str(checkpoint)
    assert "encoder.proj.weight" in resume["model_state_dict"]
    assert resume["optimizer_state_dict"] == {"state": {}, "param_groups": []}
    assert resume["metrics"]["loss"] == 1.25


def test_stage0_biencoder_train_can_select_unified_memory_route_scorer(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "model_config": kwargs}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        route_scorer="unified_memory",
        belief_top_k=32,
    )

    assert report["status"] == "ok"
    assert captured["route_scorer"] == "unified_memory"
    assert captured["belief_top_k"] == 32
    assert captured["protocol_metadata"]["route_scorer"] == "unified_memory"
    assert captured["protocol_metadata"]["uses_h_only_prior_at_inference"] is False


def test_stage0_biencoder_train_records_required_coarse_recall_keys(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "metrics": {}}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        model_dim=1024,
        max_steps=5,
        top_k=100,
    )

    assert report["status"] == "ok"
    assert captured["metric_recall_ks"] == (20, 50, 100)


def test_stage0_biencoder_train_passes_tempered_sampler_fraction(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "metrics": {}}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        sampling_strategy="handoff_tempered",
        tempered_correction_fraction=0.15,
    )

    assert report["status"] == "ok"
    assert captured["sampling_strategy"] == "handoff_tempered"
    assert captured["tempered_correction_fraction"] == 0.15
    assert captured["protocol_metadata"]["tempered_correction_fraction"] == 0.15


def test_stage0_parameter_anchor_loss_penalizes_trainable_drift_only():
    model = torch.nn.Module()
    model.keep = torch.nn.Linear(2, 1, bias=False)
    model.frozen = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.keep.weight.fill_(1.0)
        model.frozen.weight.fill_(2.0)
    model.frozen.weight.requires_grad_(False)

    anchors = _snapshot_trainable_parameters(model)
    with torch.no_grad():
        model.keep.weight.add_(1.0)
        model.frozen.weight.add_(10.0)

    loss, report = _parameter_anchor_loss(model, anchors, weight=0.5)

    assert report["anchor_parameter_count"] == 1
    assert report["anchor_unweighted_loss"] == 1.0
    assert torch.allclose(loss, torch.tensor(0.5))


def test_stage0_parameter_anchor_snapshot_can_follow_optimizer_names():
    model = torch.nn.Module()
    model.optimized = torch.nn.Linear(2, 1, bias=False)
    model.grad_but_not_optimized = torch.nn.Linear(2, 1, bias=False)

    anchors = _snapshot_trainable_parameters(model, parameter_names={"optimized.weight"})

    assert sorted(anchors) == ["optimized.weight"]


def test_stage0_biencoder_train_passes_preservation_anchor(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "metrics": {}}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        preservation_anchor_weight=0.25,
    )

    assert report["status"] == "ok"
    assert captured["preservation_anchor_weight"] == 0.25
    assert captured["protocol_metadata"]["preservation_anchor_weight"] == 0.25


def test_stage0_biencoder_train_passes_explicit_negative_loss_settings(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "metrics": {}}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        explicit_negative_loss_weight=0.2,
        explicit_negative_margin=0.15,
    )

    assert report["status"] == "ok"
    assert captured["explicit_negative_loss_weight"] == 0.2
    assert captured["explicit_negative_margin"] == 0.15
    assert captured["protocol_metadata"]["explicit_negative_loss_weight"] == 0.2
    assert captured["protocol_metadata"]["explicit_negative_margin"] == 0.15


def test_stage0_biencoder_train_passes_mined_hard_negative_loss_settings(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "metrics": {}}

    monkeypatch.setattr(retrieval_warmup, "run_skillret_retrieval_warmup", fake_run_skillret_retrieval_warmup)

    report = run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=5,
        mined_hard_negative_loss_weight=0.3,
        mined_hard_negative_margin=0.2,
        mined_hard_negative_top_k=16,
    )

    assert report["status"] == "ok"
    assert captured["mined_hard_negative_loss_weight"] == 0.3
    assert captured["mined_hard_negative_margin"] == 0.2
    assert captured["mined_hard_negative_top_k"] == 16
    assert captured["protocol_metadata"]["mined_hard_negative_loss_weight"] == 0.3
    assert captured["protocol_metadata"]["mined_hard_negative_margin"] == 0.2
    assert captured["protocol_metadata"]["mined_hard_negative_top_k"] == 16


def test_stage0_biencoder_cli_help_and_sbatch_defaults():
    proc = subprocess.run(
        [sys.executable, "scripts/run_clstr_stage0_biencoder_train.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )
    script = Path("scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh").read_text(encoding="utf-8")
    env_script = Path("scripts/sbatch/_clstr_gpu_env.sh").read_text(encoding="utf-8")

    assert "--data_root" in proc.stdout
    assert "--model_name_or_path" in proc.stdout
    assert "--gradient_accumulation_steps" in proc.stdout
    assert "--sampling_strategy" in proc.stdout
    assert "--tempered_correction_fraction" in proc.stdout
    assert "--preservation_anchor_weight" in proc.stdout
    assert "--route_scorer" in proc.stdout
    assert "--belief_top_k" in proc.stdout
    assert "--train_encoder_backbone" in proc.stdout
    assert "--encoder_backbone_learning_rate" in proc.stdout
    assert "--resume_checkpoint_path" in proc.stdout
    assert "--explicit_negative_loss_weight" in proc.stdout
    assert "--explicit_negative_margin" in proc.stdout
    assert "--mined_hard_negative_loss_weight" in proc.stdout
    assert "--mined_hard_negative_margin" in proc.stdout
    assert "--mined_hard_negative_top_k" in proc.stdout
    assert "--expand_alias_positives" in proc.stdout
    assert "PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}" in script
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert 'source "$(dirname "$0")/_clstr_gpu_env.sh"' not in script
    assert "module load miniforge3" in env_script
    assert "module load cuda" in env_script
    assert "CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/scyb713/run/miniconda3/envs/xzf}" in env_script
    assert 'source activate "${CONDA_ENV_PATH}"' in env_script
    assert "export PYTHONUNBUFFERED=1" in env_script
    assert "scripts/run_clstr_stage0_biencoder_train.py" in script
    assert "RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}" in script
    assert "ROUTE_SCORER=${ROUTE_SCORER:-legacy_biencoder}" in script
    assert "--route_scorer \"${ROUTE_SCORER}\"" in script
    assert "--belief_top_k \"${BELIEF_TOP_K}\"" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}" in script
    assert "TEMPERED_CORRECTION_FRACTION=${TEMPERED_CORRECTION_FRACTION:-0.2}" in script
    assert "PRESERVATION_ANCHOR_WEIGHT=${PRESERVATION_ANCHOR_WEIGHT:-0.0}" in script
    assert "EXPLICIT_NEGATIVE_LOSS_WEIGHT=${EXPLICIT_NEGATIVE_LOSS_WEIGHT:-0.0}" in script
    assert "EXPLICIT_NEGATIVE_MARGIN=${EXPLICIT_NEGATIVE_MARGIN:-0.1}" in script
    assert "MINED_HARD_NEGATIVE_LOSS_WEIGHT=${MINED_HARD_NEGATIVE_LOSS_WEIGHT:-0.0}" in script
    assert "MINED_HARD_NEGATIVE_MARGIN=${MINED_HARD_NEGATIVE_MARGIN:-0.1}" in script
    assert "MINED_HARD_NEGATIVE_TOP_K=${MINED_HARD_NEGATIVE_TOP_K:-32}" in script
    assert "EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}" in script
    assert "--sampling_strategy" in script
    assert "--tempered_correction_fraction" in script
    assert "--preservation_anchor_weight" in script
    assert "--explicit_negative_loss_weight" in script
    assert "--explicit_negative_margin" in script
    assert "--mined_hard_negative_loss_weight" in script
    assert "--mined_hard_negative_margin" in script
    assert "--mined_hard_negative_top_k" in script
    assert "--no-expand_alias_positives" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-0}" in script
    assert "TRAIN_ENCODER_BACKBONE=${TRAIN_ENCODER_BACKBONE:-0}" in script
    assert "ENCODER_BACKBONE_LEARNING_RATE=${ENCODER_BACKBONE_LEARNING_RATE:-}" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-0}" in script
    assert "--train_encoder_backbone" in script
    assert "--encoder_backbone_learning_rate" in script
    assert "--resume_checkpoint_path" in script
    assert "ENCODER_POOLING=${ENCODER_POOLING:-last_token}" in script
    assert "CROSS_ENCODER_POOLING=${CROSS_ENCODER_POOLING:-last_token}" in script
    assert "TOKENIZER_PADDING_SIDE=${TOKENIZER_PADDING_SIDE:-left}" in script
    assert "QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-skillrouter}" in script
    assert "STATE_QUERY_PROMPT_VERSION=${STATE_QUERY_PROMPT_VERSION:-}" in script
    assert "STATE_QUERY_MAX_CHARS=${STATE_QUERY_MAX_CHARS:-}" in script
    assert "STATE_QUERY_TRUNCATION=${STATE_QUERY_TRUNCATION:-}" in script
    assert "--encoder_pooling \"${ENCODER_POOLING}\"" in script
    assert "--cross_encoder_pooling \"${CROSS_ENCODER_POOLING}\"" in script
    assert "--tokenizer_padding_side \"${TOKENIZER_PADDING_SIDE}\"" in script
    assert "--query_text_format \"${QUERY_TEXT_FORMAT}\"" in script
    assert "--state_query_prompt_version" in script
    assert "--state_query_max_chars" in script
    assert "--state_query_truncation" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final" in script
    assert "outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE" in script
