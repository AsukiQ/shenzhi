import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

import clstr.full_base_train as full_base_train_module
from clstr.model import RouteMemoryUtilityGate
from clstr.full_base_train import (
    STATE_QUERY_ROLE,
    TRANSITION_TEXT_ROLE,
    _apply_skill_text_format,
    _attach_auto_replay_prefixes,
    _attach_full_base_embedding_cache,
    _attach_policy_embedding_cache,
    _augment_rows_with_available_actions_planner_context,
    _belief_prediction,
    _build_loss_buckets,
    _cap_rows_by_benchmark,
    canonical_stage_loss_weights,
    _compute_full_base_loss,
    _counterfactual_memory_permutation,
    _current_skill_candidate_metrics,
    _embedding_cache_policy,
    _equivalent_skill_ids_by_skill_id,
    _filter_transition_candidates_by_inventory,
    _frozen_encoder_checkpoint_exclusion,
    _freeze_for_full_base,
    _multi_positive_listwise_nll,
    _read_jsonl,
    _reported_loss_weights,
    _retrieval_contrastive_loss_from_logits,
    _ranking_metrics_from_logits,
    _batch_composition_metrics,
    _batch_cached_or_encode,
    _raise_if_nonfinite_loss,
    _sample_full_base_batch,
    _select_stage0_handoff_training_subset,
    _replay_prefix_belief,
    _stage0_candidate_prior_scores_tensor,
    _stage0_rank_prior_logits_like,
    _load_stage0_handoff_cache,
    _load_routing_checkpoint_into_model,
    _skill_logits_and_memory,
    _stage0_handoff_cache_key,
    _stage0_handoff_rows_digest,
    _merge_stage0_handoff_cached_rows,
    _transition_positive_mask,
    _transition_skill_logits,
    _write_stage0_handoff_cache,
    run_clstr_full_base_train,
    run_legacy_clstr_full_base_train_from_routing_init,
    train_clstr_full_base_with_model,
)
from clstr.gated_temporal_reranker import GatedTemporalConfig, GatedTemporalReranker
from clstr.qwen_full_base_train import write_qwen_comparison_reports
from clstr.action_adapter import UniversalActionAdapter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_full_base_jsonl_reader_preserves_unicode_line_separator_inside_json_string(tmp_path):
    path = tmp_path / "train.jsonl"
    row = {"state_text": "first\u2028second\u0085third", "action_text": "act"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    assert _read_jsonl(path) == [row]


def test_full_base_defaults_to_legacy_jsonl_handoff_cache():
    import inspect

    parameter = inspect.signature(train_clstr_full_base_with_model).parameters[
        "stage0_handoff_cache_format"
    ]

    assert parameter.default == "legacy_jsonl"


def test_internal_frozen_qwen_backbone_is_excluded_without_legacy_external_metadata():
    class _Config:
        freeze_backbone = True
        base_model_name = "Qwen3-Embedding-0.6B"

    class _Model:
        config = _Config()

    assert _frozen_encoder_checkpoint_exclusion(_Model(), {}) is True


def test_canonical_stage2_loss_contract_excludes_dead_objectives():
    weights = canonical_stage_loss_weights("stage2")

    assert {key: value for key, value in weights.items() if value > 0.0} == {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
    }
    assert all(
        weights[key] == 0.0
        for key in (
            "routing",
            "hard_negative_margin",
            "Q_success",
            "transition_hard_negative_margin",
            "STOP",
            "counterfactual_utility",
        )
    )
    assert weights["counterfactual_history"] == 0.0


def test_counterfactual_memory_permutation_stays_within_benchmark_and_replay_length():
    rows = [
        {"source_benchmark": "toolbench_g3", "replay_prefix": [{"step": 0}]},
        {"source_benchmark": "toolbench_g3", "replay_prefix": [{"step": 1}]},
        {"source_benchmark": "tau2", "replay_prefix": [{"step": 0}]},
        {"source_benchmark": "toolbench_g3", "replay_prefix": []},
        {"source_benchmark": "toolbench_g3", "replay_prefix": [{"step": 2}]},
    ]

    permutation = _counterfactual_memory_permutation(rows, torch.device("cpu"))

    assert permutation.tolist() == [1, 4, -1, -1, 0]


def test_counterfactual_memory_donors_exclude_same_trajectory_and_same_target():
    rows = [
        {
            "benchmark": "toy",
            "trajectory_id": "a",
            "skill_id": "skill/current",
            "next_skill_id": "skill/x",
            "replay_prefix": [{"step": 0}],
        },
        {
            "benchmark": "toy",
            "trajectory_id": "b",
            "skill_id": "skill/current",
            "next_skill_id": "skill/x",
            "replay_prefix": [{"step": 1}],
        },
        {
            "benchmark": "toy",
            "trajectory_id": "c",
            "skill_id": "skill/current",
            "next_skill_id": "skill/y",
            "replay_prefix": [{"step": 2}],
        },
    ]

    permutation = _counterfactual_memory_permutation(rows, torch.device("cpu"))

    assert permutation.tolist() == [2, 2, 0]


def test_auto_replay_prefix_uses_history_free_router_state():
    rows = [
        {
            "trajectory_id": "t",
            "step_index": 0,
            "state_text": "goal: g\nobservation: first\nhistory: <empty>",
            "state_text_current": "goal: g\nobservation: first",
            "action_text": "act",
            "next_observation_text": "actual result",
            "skill_id": "skill/a",
        },
        {
            "trajectory_id": "t",
            "step_index": 1,
            "state_text": "goal: g\nobservation: second\nhistory: act",
            "state_text_current": "goal: g\nobservation: second",
            "action_text": "next",
            "next_observation_text": "next result",
            "skill_id": "skill/b",
        },
    ]

    prepared, _report = _attach_auto_replay_prefixes(rows, max_steps=1)

    assert prepared[1]["replay_prefix"][0]["observation_text"] == (
        "goal: g\nobservation: first"
    )


def test_canonical_stage2_rejects_removed_loss_override():
    with pytest.raises(ValueError, match="removed canonical stage2 loss: STOP"):
        canonical_stage_loss_weights("stage2", {"STOP": 0.1})

    weights = canonical_stage_loss_weights("stage2", {"L_policy": 0.15})
    assert weights["L_policy"] == pytest.approx(0.15)

    counterfactual = canonical_stage_loss_weights(
        "stage2", {"counterfactual_history": 1.0}
    )
    assert counterfactual["counterfactual_history"] == 1.0


def test_canonical_report_omits_removed_zero_weight_losses():
    assert _reported_loss_weights(canonical_stage_loss_weights("stage2")) == {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
    }

    experimental = canonical_stage_loss_weights("stage2")
    experimental["routing"] = 0.2
    assert _reported_loss_weights(experimental)["routing"] == 0.2


def test_embedding_cache_auto_skips_large_native_stage2_data():
    policy = _embedding_cache_policy(
        mode="auto",
        row_count=65_570,
        qwen_external_encoder=False,
        max_rows=20_000,
    )

    assert policy["cache_enabled"] is False
    assert policy["reason"] == "large_dataset_auto_skip"


def test_routing_checkpoint_loader_seeds_default_belief_calibration(tmp_path):
    checkpoint_path = tmp_path / "stage0.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "skill_table.logit_scale_retr": torch.tensor([0.25]),
                "skill_table.skill_bias_retr": torch.tensor([0.1, -0.2, 0.3]),
                "skill_table.logit_scale_belief": torch.tensor([torch.log(torch.tensor(0.2)).item()]),
                "skill_table.skill_bias_belief": torch.zeros(3),
            },
        },
        checkpoint_path,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skill_table = torch.nn.Module()
            self.skill_table.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))

    model = _Model()

    report = _load_routing_checkpoint_into_model(model, checkpoint_path)

    assert torch.allclose(model.skill_table.logit_scale_belief, torch.tensor([0.25]))
    assert torch.allclose(model.skill_table.skill_bias_belief, torch.tensor([0.1, -0.2, 0.3]))
    assert report["belief_calibration_seeded_from_retrieval"]["applied"] is True


def test_full_base_train_max_rows_limits_smoke_training_after_filters(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
        ],
    )
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "toy",
                "trajectory_id": "toy-traj",
                "step_index": 0,
                "state_text": "look mug",
                "action_text": "take mug",
                "expert_action": "take mug",
                "admissible_actions": ["take mug", "look drawer"],
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "next_observation_text": "drawer",
                "provenance": {"source_id": "toy-source", "split": "train"},
                "loss_mask": {"L_policy": True, "L_trans_skill_ce": True, "STOP": True},
            },
            {
                "benchmark": "toy",
                "trajectory_id": "toy-traj",
                "step_index": 1,
                "state_text": "look drawer",
                "action_text": "open drawer",
                "expert_action": "open drawer",
                "admissible_actions": ["open drawer"],
                "skill_id": "skill/b",
                "next_skill_id": "skill/a",
                "next_observation_text": "mug",
                "provenance": {"source_id": "toy-source", "split": "train"},
                "loss_mask": {"L_policy": True, "L_trans_skill_ce": True, "STOP": True},
            },
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=2),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        max_rows=1,
        learning_rate=1.0e-3,
        loss_weights={"L_policy": 1.0, "L_trans_skill_ce": 0.2, "STOP": 0.2, "routing": 0.0, "L_trans": 0.0, "belief": 0.0},
        embedding_cache_mode="never",
        allow_full_pool_stage2_debug=True,
    )

    assert report["status"] == "ok"
    assert report["sample_count"] == 1
    assert report["max_rows"] == 1
    assert report["max_rows_filter"]["source_rows"] == 2
    assert report["max_rows_filter"]["retained_rows"] == 1
    assert report["causal_next_state"]["attached_rows"] == 1
    assert report["loss_activation_counts"]["L_trans_skill_ce"] == 1
    assert Path(report["latest_checkpoint"]).exists()
    assert Path(report["training_metrics_path"]).exists()


def test_full_base_available_actions_augmentation_precedes_next_state_attachment(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/a", "name": "a"},
            {"skill_id": "skill/b", "name": "b"},
        ],
    )
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "toy",
                "trajectory_id": "toy-traj",
                "step_index": 0,
                "state_text": "current raw state",
                "action_text": "take mug",
                "expert_action": "take mug",
                "admissible_actions": ["take mug", "look"],
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "next_observation_text": "mug taken",
                "next_state_text": "successor raw state",
                "next_state_source": "source fixture",
                "provenance": {"source_id": "toy-source", "split": "train"},
                "loss_mask": {"L_policy": True, "L_trans_skill_ce": True},
            },
            {
                "benchmark": "toy",
                "trajectory_id": "toy-traj",
                "step_index": 1,
                "state_text": "successor raw state",
                "action_text": "open drawer",
                "expert_action": "open drawer",
                "admissible_actions": ["open drawer", "look"],
                "skill_id": "skill/b",
                "next_observation_text": "drawer open",
                "provenance": {"source_id": "toy-source", "split": "train"},
                "loss_mask": {"L_policy": True, "L_trans_skill_ce": False},
            },
        ],
    )
    original_attach = full_base_train_module._attach_adjacent_next_states
    captured = {}

    def recording_attach(rows):
        prepared, report = original_attach(rows)
        captured["successor_state_text"] = rows[1]["state_text"]
        captured["next_state_text"] = prepared[0]["next_state_text"]
        captured["next_state_source"] = prepared[0]["next_state_source"]
        return prepared, report

    monkeypatch.setattr(full_base_train_module, "_attach_adjacent_next_states", recording_attach)

    report = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=2),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        max_rows=1,
        learning_rate=1.0e-3,
        include_available_actions_in_state=True,
        embedding_cache_mode="never",
        allow_full_pool_stage2_debug=True,
    )

    assert "AVAILABLE ACTIONS" in captured["successor_state_text"]
    assert captured["next_state_text"] == captured["successor_state_text"]
    assert captured["next_state_source"] == "adjacent_trajectory_row"
    assert report["available_actions_planner_context"]["row_count"] == 1
    assert report["available_actions_planner_context"]["augmented_row_count"] == 1
    assert report["available_actions_planner_context"]["preparation_row_count"] == 2
    assert report["available_actions_planner_context"]["preparation_augmented_row_count"] == 2


def _row_with_losses(row_id: str, enabled_losses: list[str]) -> dict:
    return {
        "task_id": row_id,
        "state_text": f"state {row_id}",
        "action_text": f"action {row_id}",
        "loss_mask": {key: key in enabled_losses for key in ["L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"]},
    }


def test_full_base_sampler_keeps_sparse_policy_examples_active_after_file_prefix():
    rows = [_row_with_losses(f"policy-{idx}", ["L_policy", "L_trans_skill_ce", "STOP"]) for idx in range(2)]
    rows.extend(_row_with_losses(f"aux-{idx}", ["L_trans", "belief", "STOP", "routing"]) for idx in range(50))
    loss_weights = {
        "L_policy": 1.0,
        "L_trans": 0.02,
        "L_trans_skill_ce": 0.2,
        "belief": 0.05,
        "STOP": 0.2,
        "routing": 0.2,
    }

    sampled = [
        _sample_full_base_batch(rows, step_idx=step_idx, batch_size=4, loss_weights=loss_weights)
        for step_idx in range(20, 30)
    ]

    assert all(any((row.get("loss_mask") or {}).get("L_policy") for row in batch) for batch in sampled)
    assert {row["task_id"] for batch in sampled for row in batch if (row.get("loss_mask") or {}).get("L_policy")} == {
        "policy-0",
        "policy-1",
    }


def test_apply_skill_text_format_switches_skill_table_serializer():
    model = _CountingEncodeModel(dim=3)

    class _SkillTable:
        skill_text_fn = None

    model.skill_table = _SkillTable()
    model.config = type("Config", (), {"skill_text_format": "clstr"})()
    model_config = {"skill_text_format": "clstr"}

    report = _apply_skill_text_format(model, model_config, "clstr_enriched")
    text = model.skill_table.skill_text_fn(
        {
            "name": "skill",
            "description": "desc",
            "body": "body text",
            "positive_action_examples": ["go to fridge 1"],
        }
    )

    assert report["skill_text_format"] == "clstr_enriched"
    assert model_config["skill_text_format"] == "clstr_enriched"
    assert model.config.skill_text_format == "clstr_enriched"
    assert "body text" in text
    assert "go to fridge 1" in text


def test_canonical_full_base_train_rejects_legacy_routing_init_manifest(tmp_path):
    with pytest.raises(ValueError, match="canonical Stage2 requires routing_checkpoint_path"):
        run_clstr_full_base_train(
            train_path=tmp_path / "train.jsonl",
            skills_path=tmp_path / "skills.jsonl",
            output_dir=tmp_path / "out",
            routing_init_manifest=tmp_path / "legacy_manifest.json",
        )


def test_canonical_full_base_train_requires_stage1_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Stage2 canonical training requires stage1_checkpoint_path"):
        run_clstr_full_base_train(
            train_path=tmp_path / "train.jsonl",
            skills_path=tmp_path / "skills.jsonl",
            output_dir=tmp_path / "out",
            routing_checkpoint_path=tmp_path / "stage0.pt",
        )


def test_legacy_full_base_train_wrapper_is_explicitly_marked(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    manifest_path = tmp_path / "legacy_manifest.json"
    _write_jsonl(train_path, [])
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a"}])
    manifest_path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    class FakeModel(torch.nn.Module):
        pass

    def fake_build_model_from_routing_init(routing_init_manifest, skills, output_dir):
        return (
            FakeModel(),
            {"d": 8, "base_model_name": "legacy-test"},
            {
                "routing_init_manifest": str(routing_init_manifest),
                "skill_count": len(skills),
            },
        )

    captured = {}

    def fake_train_with_model(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "routing_init": kwargs["routing_report"],
        }

    monkeypatch.setattr("clstr.full_base_train._build_model_from_routing_init", fake_build_model_from_routing_init)
    monkeypatch.setattr("clstr.full_base_train.train_clstr_full_base_with_model", fake_train_with_model)

    report = run_legacy_clstr_full_base_train_from_routing_init(
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        routing_init_manifest=manifest_path,
        max_steps=1,
        batch_size=1,
    )

    assert report["status"] == "ok"
    assert report["legacy_routing_init_manifest_used"] is True
    assert report["not_canonical_stage2_unified_mainline"] is True
    assert captured["routing_checkpoint_path"] is None
    assert captured["routing_report"]["legacy_routing_init_manifest_used"] is True
    assert captured["routing_report"]["not_canonical_stage2_unified_mainline"] is True


def test_full_base_sampler_reuses_precomputed_loss_buckets():
    rows = [_row_with_losses("policy-0", ["L_policy", "L_trans_skill_ce"])]
    rows.extend(_row_with_losses(f"aux-{idx}", ["L_trans", "STOP"]) for idx in range(4))
    loss_weights = {
        "L_policy": 1.0,
        "L_trans": 0.02,
        "L_trans_skill_ce": 0.2,
        "belief": 0.05,
        "STOP": 0.2,
        "routing": 0.2,
    }

    buckets = _build_loss_buckets(rows, loss_weights)
    batch = _sample_full_base_batch(rows, step_idx=99, batch_size=3, loss_weights=loss_weights, loss_buckets=buckets)

    assert buckets["L_policy"] == [0]
    assert buckets["L_trans"] == [1, 2, 3, 4]
    assert any(row["task_id"] == "policy-0" for row in batch)


def test_full_base_random_sampler_is_seeded_and_balanced():
    rows = [_row_with_losses(f"policy-{idx}", ["L_policy", "L_trans_skill_ce"]) for idx in range(8)]
    rows.extend(_row_with_losses(f"trans-{idx}", ["L_trans", "belief", "STOP"]) for idx in range(24))
    loss_weights = {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
        "STOP": 0.1,
        "routing": 0.0,
    }
    buckets = _build_loss_buckets(rows, loss_weights)

    batch_a = _sample_full_base_batch(
        rows,
        step_idx=7,
        batch_size=4,
        loss_weights=loss_weights,
        loss_buckets=buckets,
        sampling_strategy="balanced_random",
        sampler_seed=13,
    )
    batch_b = _sample_full_base_batch(
        rows,
        step_idx=7,
        batch_size=4,
        loss_weights=loss_weights,
        loss_buckets=buckets,
        sampling_strategy="balanced_random",
        sampler_seed=13,
    )
    batch_c = _sample_full_base_batch(
        rows,
        step_idx=7,
        batch_size=4,
        loss_weights=loss_weights,
        loss_buckets=buckets,
        sampling_strategy="balanced_random",
        sampler_seed=29,
    )

    assert [row["task_id"] for row in batch_a] == [row["task_id"] for row in batch_b]
    assert [row["task_id"] for row in batch_a] != [row["task_id"] for row in batch_c]
    assert any((row.get("loss_mask") or {}).get("L_policy") for row in batch_a)
    assert any((row.get("loss_mask") or {}).get("L_trans_skill_ce") for row in batch_a)
    assert any((row.get("loss_mask") or {}).get("L_trans") for row in batch_a)


def test_full_base_random_sampler_avoids_large_shuffle(monkeypatch):
    rows = [_row_with_losses(f"policy-{idx}", ["L_policy", "L_trans_skill_ce", "L_trans", "STOP"]) for idx in range(128)]
    loss_weights = {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
        "STOP": 0.1,
        "routing": 0.0,
    }

    def fail_shuffle(*_args, **_kwargs):
        raise AssertionError("balanced_random sampler must not shuffle full buckets")

    monkeypatch.setattr(random.Random, "shuffle", fail_shuffle)

    batch = _sample_full_base_batch(
        rows,
        step_idx=3,
        batch_size=4,
        loss_weights=loss_weights,
        sampling_strategy="balanced_random",
        sampler_seed=17,
    )

    assert len(batch) == 4


def test_full_base_benchmark_transition_balanced_sampler_covers_small_self_groups():
    rows = []
    for idx in range(80):
        row = _row_with_losses(f"traject-switch-{idx}", ["L_policy", "L_trans_skill_ce", "L_trans", "STOP"])
        row.update({"benchmark": "traject_bench", "skill_id": "traject/search", "next_skill_id": f"traject/next-{idx}"})
        rows.append(row)
    for idx in range(4):
        row = _row_with_losses(f"toolbench-self-{idx}", ["L_policy", "L_trans_skill_ce", "L_trans", "STOP"])
        row.update({"benchmark": "toolbench_g3", "skill_id": "tool/use", "next_skill_id": "tool/use"})
        rows.append(row)
    for idx in range(4):
        row = _row_with_losses(f"alfworld-self-{idx}", ["L_policy", "L_trans_skill_ce", "L_trans", "STOP"])
        row.update({"benchmark": "alfworld", "skill_id": "alf/open", "next_skill_id": "alf/open"})
        rows.append(row)
    loss_weights = {
        "L_policy": 0.5,
        "L_trans": 0.5,
        "L_trans_skill_ce": 1.0,
        "belief": 0.0,
        "STOP": 0.1,
        "routing": 0.0,
    }

    sampled_groups = Counter()
    loss_buckets = _build_loss_buckets(rows, loss_weights)
    for step_idx in range(1, 25):
        batch = _sample_full_base_batch(
            rows,
            step_idx=step_idx,
            batch_size=4,
            loss_weights=loss_weights,
            loss_buckets=loss_buckets,
            sampling_strategy="benchmark_transition_balanced_random",
            sampler_seed=17,
        )
        for row in batch:
            relation = "self" if row["skill_id"] == row["next_skill_id"] else "switch"
            sampled_groups[f"{row['benchmark']}:{relation}"] += 1

    assert sampled_groups["toolbench_g3:self"] >= 20
    assert sampled_groups["alfworld:self"] >= 20
    assert sampled_groups["traject_bench:switch"] >= 20


def test_full_base_quota_sampler_makes_batch16_transition_and_policy_dense():
    rows = []
    for idx in range(24):
        row = _row_with_losses(f"switch-{idx}", ["L_trans_skill_ce", "L_trans", "STOP"])
        row.update(
            {
                "benchmark": "toolbench_g3" if idx % 2 == 0 else "traject_bench",
                "skill_id": "skill/current",
                "next_skill_id": f"skill/next-{idx}",
                "stage0_positive_injected": False,
            }
        )
        rows.append(row)
    for idx in range(24):
        row = _row_with_losses(f"self-{idx}", ["L_trans_skill_ce", "belief", "STOP"])
        row.update(
            {
                "benchmark": "alfworld" if idx % 2 == 0 else "webshop",
                "skill_id": "skill/self",
                "next_skill_id": "skill/self",
                "stage0_positive_injected": False,
            }
        )
        rows.append(row)
    for idx in range(24):
        row = _row_with_losses(f"policy-{idx}", ["L_policy"])
        row.update({"benchmark": "webshop", "skill_id": "skill/policy", "next_skill_id": "skill/policy"})
        rows.append(row)
    loss_weights = {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
        "STOP": 0.1,
        "routing": 0.0,
    }

    batch = _sample_full_base_batch(
        rows,
        step_idx=3,
        batch_size=16,
        loss_weights=loss_weights,
        loss_buckets=_build_loss_buckets(rows, loss_weights),
        sampling_strategy="benchmark_transition_quota_random",
        sampler_seed=17,
    )
    metrics = _batch_composition_metrics(batch)

    assert len(batch) == 16
    assert metrics["batch_transition_switch_real_rows"] >= 4
    assert metrics["batch_transition_real_rows"] >= 8
    assert metrics["batch_policy_rows"] >= 4
    assert metrics["batch_loss_mask_counts"]["L_trans_skill_ce"] >= 8
    assert metrics["batch_benchmark_counts"]["toolbench_g3"] > 0
    assert metrics["batch_benchmark_counts"]["traject_bench"] > 0


def test_stage0_handoff_subset_reports_quota_sampler_order():
    rows = []
    for idx in range(12):
        row = _row_with_losses(f"switch-{idx}", ["L_trans_skill_ce", "L_trans"])
        row.update({"benchmark": "toolbench_g3", "skill_id": "skill/a", "next_skill_id": f"skill/b-{idx}"})
        rows.append(row)
    for idx in range(12):
        row = _row_with_losses(f"policy-{idx}", ["L_policy"])
        row.update({"benchmark": "webshop", "skill_id": "skill/p", "next_skill_id": "skill/p"})
        rows.append(row)

    selected, report = _select_stage0_handoff_training_subset(
        rows,
        max_steps=3,
        batch_size=16,
        loss_weights={"L_policy": 0.2, "L_trans": 0.3, "L_trans_skill_ce": 1.0},
        sample_multiplier=1.0,
        sampling_strategy="benchmark_transition_quota_random",
        sampler_seed=17,
    )

    assert selected
    assert report["selection_order"] == "seeded_benchmark_transition_quota_training_sampler"
    assert report["sampling_strategy"] == "benchmark_transition_quota_random"


def test_stage0_handoff_quota_subset_reuses_grouped_buckets(monkeypatch):
    rows = []
    for idx in range(64):
        row = _row_with_losses(f"switch-{idx}", ["L_trans_skill_ce", "L_trans", "STOP"])
        row.update({"benchmark": "toolbench_g3", "skill_id": "skill/a", "next_skill_id": f"skill/b-{idx}"})
        rows.append(row)
    for idx in range(64):
        row = _row_with_losses(f"policy-{idx}", ["L_policy", "routing"])
        row.update({"benchmark": "webshop", "skill_id": "skill/p", "next_skill_id": "skill/p"})
        rows.append(row)
    for idx in range(8):
        row = _row_with_losses(f"belief-{idx}", ["belief", "L_trans", "STOP"])
        row.update({"benchmark": "alfworld", "skill_id": "skill/belief", "next_skill_id": f"skill/belief-next-{idx}"})
        rows.append(row)

    original = full_base_train_module._group_indices_by_transition_relation
    call_count = 0

    def counting_group(rows_arg, indices):
        nonlocal call_count
        call_count += 1
        return original(rows_arg, indices)

    monkeypatch.setattr(full_base_train_module, "_group_indices_by_transition_relation", counting_group)

    selected, report = _select_stage0_handoff_training_subset(
        rows,
        max_steps=20,
        batch_size=16,
        loss_weights={"L_policy": 0.2, "L_trans": 0.3, "L_trans_skill_ce": 1.0, "belief": 0.1, "STOP": 0.1},
        sample_multiplier=2.0,
        sampling_strategy="benchmark_transition_quota_random",
        sampler_seed=17,
    )

    assert selected
    assert report["sampling_strategy"] == "benchmark_transition_quota_random"
    assert call_count <= 8


def test_full_base_quota_sampler_prioritizes_sparse_belief_rows():
    rows = []
    for idx in range(128):
        row = _row_with_losses(f"trans-{idx}", ["L_trans", "STOP"])
        row.update({"benchmark": "alfworld", "skill_id": "skill/a", "next_skill_id": f"skill/b-{idx}"})
        rows.append(row)
    for idx in range(2):
        row = _row_with_losses(f"belief-{idx}", ["belief", "L_trans", "STOP"])
        row.update({"benchmark": "alfworld", "skill_id": "skill/belief", "next_skill_id": f"skill/belief-next-{idx}"})
        rows.append(row)
    for idx in range(16):
        row = _row_with_losses(f"policy-{idx}", ["L_policy"])
        row.update({"benchmark": "webshop", "skill_id": "skill/p", "next_skill_id": "skill/p"})
        rows.append(row)
    loss_weights = {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 0.0,
        "belief": 0.1,
        "STOP": 0.1,
        "routing": 0.0,
    }

    loss_buckets = _build_loss_buckets(rows, loss_weights)
    for step_idx in range(1, 6):
        batch = _sample_full_base_batch(
            rows,
            step_idx=step_idx,
            batch_size=16,
            loss_weights=loss_weights,
            loss_buckets=loss_buckets,
            sampling_strategy="benchmark_transition_quota_random",
            sampler_seed=17,
        )
        metrics = _batch_composition_metrics(batch)
        assert metrics["batch_loss_mask_counts"]["belief"] >= 1


def test_current_skill_candidate_metrics_track_keep_and_switch_errors():
    rows = [
        {"skill_id": "skill/a", "next_skill_id": "skill/a"},
        {"skill_id": "skill/a", "next_skill_id": "skill/b"},
    ]
    candidate_rows = [[0, 1], [1, 0]]
    logits = torch.tensor([[3.0, 1.0], [1.0, 4.0]])

    metrics = _current_skill_candidate_metrics(
        rows=rows,
        candidate_rows=candidate_rows,
        logits=logits,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1},
    )

    assert metrics["transition_current_skill_candidate_rows"] == 2.0
    assert metrics["transition_current_skill_keep_rows"] == 1.0
    assert metrics["transition_current_skill_switch_rows"] == 1.0
    assert metrics["transition_current_skill_keep_recall@1"] == 1.0
    assert metrics["transition_current_skill_switch_false_positive@1"] == 1.0
    assert metrics["transition_current_skill_mean_rank"] == 1.0


def test_full_base_policy_loss_uses_cached_candidate_embeddings():
    model = _CountingEncodeModel(dim=3)
    rows = [
        {
            "state_text": "state current",
            "action_text": "candidate expert action",
            "admissible_actions": ["candidate expert action", "candidate distractor action"],
            "expert_action": "candidate expert action",
            "loss_mask": {"L_policy": True},
        }
    ]
    cache_report = _attach_policy_embedding_cache(model, rows, encode_batch_size=4)
    model.encoded_texts.clear()

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=UniversalActionAdapter(3, hidden_dim=3),
        batch=rows,
        skill_id_to_idx={},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 1.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )

    assert cache_report["used"] is True
    assert cache_report["policy_rows"] == 1
    assert metrics["policy_ce_loss"] >= 0.0
    assert 0.0 <= metrics["policy_expert_recall@1"] <= 1.0
    assert "candidate expert action" not in model.encoded_texts
    assert "candidate distractor action" not in model.encoded_texts


def test_full_base_policy_loss_prefers_native_clstr_skill_head_over_legacy_adapter():
    model = _NativePolicyHeadModel()
    rows = [
        {
            "state_text": "state current",
            "action_text": "candidate expert action",
            "admissible_actions": ["candidate expert action", "candidate distractor action"],
            "expert_action": "candidate expert action",
            "loss_mask": {"L_policy": True},
        }
    ]
    _attach_policy_embedding_cache(model, rows, encode_batch_size=4)

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=_FailingActionAdapter(),
        batch=rows,
        skill_id_to_idx={},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 1.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )

    assert model.skill_head.calls == 1
    assert metrics["policy_head_type"] == "native_skill_head"
    assert metrics["policy_ce_loss"] < 0.01
    assert metrics["policy_expert_recall@1"] == 1.0


def test_full_base_q_success_loss_uses_candidate_labels_and_reports_metrics():
    model = _NativePolicyHeadModel()
    model.q_success_head = _LinearQSuccessHead()
    rows = [
        {
            "state_text": "state current",
            "action_text": "expert action",
            "admissible_actions": ["qwen wrong", "expert action"],
            "expert_action": "expert action",
            "q_success_labels": [0.0, 1.0],
            "q_success_label_weights": [0.5, 1.0],
            "loss_mask": {"L_policy": True, "Q_success": True},
        }
    ]
    _attach_policy_embedding_cache(model, rows, encode_batch_size=4)

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=_FailingActionAdapter(),
        batch=rows,
        skill_id_to_idx={},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "Q_success": 1.0,
            "hard_negative_margin": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
    )

    assert metrics["q_success_bce_loss"] > 0.0
    assert metrics["q_success_sample_count"] == 2
    assert metrics["weighted_loss_terms"]["Q_success"] > 0.0


def test_full_base_hard_negative_margin_penalizes_qwen_wrong_action_above_expert():
    model = _NativePolicyHeadModel()
    rows = [
        {
            "state_text": "state current",
            "action_text": "expert action",
            "admissible_actions": ["qwen wrong", "expert action"],
            "expert_action": "expert action",
            "hard_negative_action": "qwen wrong",
            "loss_mask": {"L_policy": True, "hard_negative_margin": True},
        }
    ]
    rows[0]["_policy_state_embedding"] = torch.zeros(3)
    rows[0]["_policy_candidate_embeddings"] = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    rows[0]["_policy_label"] = 1

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=_FailingActionAdapter(),
        batch=rows,
        skill_id_to_idx={},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "hard_negative_margin": 1.0,
            "Q_success": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
    )

    assert metrics["hard_negative_margin_loss"] >= 0.0
    assert metrics["hard_negative_pair_count"] == 1
    assert "hard_negative_margin" in metrics["weighted_loss_terms"]


def test_zero_weight_policy_auxiliaries_do_not_execute_forward_helpers(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("zero-weight loss helper executed")

    monkeypatch.setattr(full_base_train_module, "compute_q_success_loss", fail)
    monkeypatch.setattr(full_base_train_module, "_retrieval_contrastive_loss_from_logits", fail)
    model = _NativePolicyHeadModel()
    model.q_success_head = _LinearQSuccessHead()
    rows = [
        {
            "state_text": "state current",
            "action_text": "expert action",
            "admissible_actions": ["wrong action", "expert action"],
            "expert_action": "expert action",
            "hard_negative_action": "wrong action",
            "skill_id": "skill/a",
            "loss_mask": {
                "L_policy": True,
                "hard_negative_margin": True,
                "Q_success": True,
                "routing": True,
            },
        }
    ]

    _compute_full_base_loss(
        model=model,
        action_adapter=_FailingActionAdapter(),
        batch=rows,
        skill_id_to_idx={"skill/a": 0},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
    )


def test_available_actions_planner_context_augmentation_only_changes_policy_rows():
    rows = [
        {
            "state_text": "goal: heat apple\nobservation: kitchen",
            "admissible_actions": ["look", "open fridge 1"],
            "loss_mask": {"L_policy": True},
        },
        {
            "state_text": "goal: aux\nobservation: other",
            "admissible_actions": [],
            "loss_mask": {"L_policy": False, "L_trans": True},
        },
    ]

    augmented, report = _augment_rows_with_available_actions_planner_context(rows)

    assert report["enabled"] is True
    assert report["augmented_row_count"] == 1
    assert "AVAILABLE ACTIONS" in augmented[0]["state_text"]
    assert "1. look" in augmented[0]["state_text"]
    assert "CLSTR skill-routing query" in augmented[0]["state_text"]
    assert augmented[1]["state_text"] == rows[1]["state_text"]


def test_available_actions_augmentation_preserves_raw_successor_for_canonicalization():
    rows = _causal_adjacency_rows()
    rows[0]["next_state_text"] = "trusted successor state"
    rows[0]["_next_state_embedding"] = torch.tensor([99.0])
    rows[1]["loss_mask"] = {"L_policy": True}
    rows[1]["admissible_actions"] = ["open drawer", "look"]

    augmented, _report = _augment_rows_with_available_actions_planner_context(rows)
    prepared, adjacency_report = full_base_train_module._attach_adjacent_next_states(augmented)

    assert augmented[1]["_available_actions_base_state_text"] == "trusted successor state"
    assert "AVAILABLE ACTIONS" in augmented[1]["state_text"]
    assert all("_available_actions_base_state_text" not in row for row in prepared)
    assert prepared[0]["next_state_text"] == augmented[1]["state_text"]
    assert prepared[0]["next_state_source"] == "adjacent_trajectory_row"
    assert "_next_state_embedding" not in prepared[0]
    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert adjacency_report["materialized_rows"] == 1
    assert adjacency_report["skip_reasons"] == {}


def test_qwen_comparison_reports_include_full_likelihood_direct_baseline(tmp_path):
    metrics_dir = tmp_path / "outputs" / "alfworld_eval" / "qwen3_8b_likelihood_full"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "success_rate": 0.0,
                "average_reward": 0.0,
                "average_goal_condition_points": 0.0,
                "average_episode_steps": 50.0,
                "episodes": 274,
                "qwen_direct_baseline": True,
                "not_clstr_result": True,
            }
        ),
        encoding="utf-8",
    )

    report = write_qwen_comparison_reports(
        output_root=tmp_path / "outputs",
        output_table_path=tmp_path / "table.md",
        output_summary_path=tmp_path / "summary.json",
        output_paper_md=tmp_path / "paper.md",
        output_paper_json=tmp_path / "paper.json",
    )

    rows = {row["method"]: row for row in report["rows"]}
    assert "qwen3_8b_likelihood_full" in rows
    assert rows["qwen3_8b_likelihood_full"]["qwen_direct_baseline"] is True
    assert rows["qwen3_8b_likelihood_full"]["is_clstr"] is False
    assert "qwen3_8b_likelihood_full" in (tmp_path / "table.md").read_text(encoding="utf-8")


def test_qwen_comparison_reports_include_available_actions_diagnostics(tmp_path):
    eval_root = tmp_path / "outputs" / "alfworld_eval"
    direct_dir = eval_root / "qwen3_8b_chat_available_valid_seen_20"
    direct_dir.mkdir(parents=True)
    (direct_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "success_rate": 0.1,
                "average_reward": 0.1,
                "average_goal_condition_points": 0.1,
                "average_episode_steps": 46.25,
                "episodes": 20,
                "qwen_direct_baseline": True,
                "not_clstr_result": True,
            }
        ),
        encoding="utf-8",
    )
    direct_50_dir = eval_root / "qwen3_8b_chat_available_valid_seen_50"
    direct_50_dir.mkdir(parents=True)
    (direct_50_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "success_rate": 0.14,
                "average_reward": 0.14,
                "average_goal_condition_points": 0.0,
                "average_episode_steps": 44.7,
                "episodes": 50,
                "qwen_direct_baseline": True,
                "not_clstr_result": True,
            }
        ),
        encoding="utf-8",
    )
    direct_full_dir = eval_root / "qwen3_8b_chat_available_full"
    direct_full_dir.mkdir(parents=True)
    (direct_full_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "success_rate": 0.0,
                "average_reward": 0.0,
                "average_goal_condition_points": 0.0,
                "average_episode_steps": 0.0,
                "episodes": 0,
                "qwen_direct_baseline": True,
                "not_clstr_result": True,
                "not_closed_loop_success": True,
                "blocked_split_count": 2,
                "error": "CUDA out of memory",
            }
        ),
        encoding="utf-8",
    )
    clstr_dir = eval_root / "clstr_qwen3_8b_available_actions_gate_valid_seen_20"
    clstr_dir.mkdir(parents=True)
    (clstr_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "success_rate": 0.05,
                "average_reward": 0.05,
                "average_goal_condition_points": 0.05,
                "average_episode_steps": 48.0,
                "episodes": 20,
                "qwen_external_encoder": True,
                "clstr_native_act": True,
            }
        ),
        encoding="utf-8",
    )
    trained_clstr_dir = eval_root / "clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20"
    trained_clstr_dir.mkdir(parents=True)
    (trained_clstr_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "success_rate": 0.0,
                "average_reward": 0.0,
                "average_goal_condition_points": 0.0,
                "average_episode_steps": 50.0,
                "episodes": 20,
                "qwen_external_encoder": True,
                "clstr_native_act": True,
                "include_available_actions_in_state": True,
            }
        ),
        encoding="utf-8",
    )
    (trained_clstr_dir / "blocker_report.json").write_text(
        json.dumps({"status": "blocked", "stage": "clstr_qwen3_8b_available_actions_retrained_controller_gate"}),
        encoding="utf-8",
    )

    report = write_qwen_comparison_reports(
        output_root=tmp_path / "outputs",
        output_table_path=tmp_path / "table.md",
        output_summary_path=tmp_path / "summary.json",
        output_paper_md=tmp_path / "paper.md",
        output_paper_json=tmp_path / "paper.json",
    )

    rows = {row["method"]: row for row in report["rows"]}
    assert rows["qwen3_8b_chat_available_valid_seen_20"]["qwen_direct_baseline"] is True
    assert rows["qwen3_8b_chat_available_valid_seen_20"]["is_clstr"] is False
    assert rows["qwen3_8b_chat_available_valid_seen_50"]["qwen_direct_baseline"] is True
    assert rows["qwen3_8b_chat_available_full"]["qwen_direct_baseline"] is True
    assert rows["qwen3_8b_chat_available_full"]["is_clstr"] is False
    assert rows["qwen3_8b_chat_available_valid_seen_50"]["qwen_direct_baseline"] is True
    assert rows["qwen3_8b_chat_available_valid_seen_50"]["success_rate"] == 0.14
    assert rows["qwen3_8b_chat_available_full"]["status"] == "blocked"
    assert rows["qwen3_8b_chat_available_full"]["is_clstr"] is False
    assert rows["clstr_qwen3_8b_available_actions_gate_valid_seen_20"]["is_clstr"] is True
    assert rows["clstr_qwen3_8b_available_actions_gate_valid_seen_20"]["qwen_external_encoder"] is True
    assert rows["clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20"]["is_clstr"] is True
    assert rows["clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20"]["status"] == "blocked"
    assert rows["clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20"]["blocker_path"].endswith("blocker_report.json")
    assert rows["clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20"]["success_rate"] == 0.0
    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "qwen3_8b_chat_available_valid_seen_20" in table
    assert "qwen3_8b_chat_available_valid_seen_50" in table
    assert "qwen3_8b_chat_available_full" in table
    assert "qwen3_8b_chat_available_valid_seen_50" in table
    assert "qwen3_8b_chat_available_full" in table
    assert "clstr_qwen3_8b_available_actions_gate_valid_seen_20" in table
    assert "clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20" in table


def test_full_base_loss_uses_cached_replay_prefix_embeddings_without_reencoding_prefix_text():
    model = _CountingEncodeModel(dim=3)
    rows = [
        {
            "state_text": "current state",
            "action_text": "expert action",
            "next_observation_text": "target next observation",
            "skill_id": "skill",
            "replay_prefix": [
                {
                    "observation_text": "prefix initial observation",
                    "action_text": "prefix action",
                    "next_observation_text": "prefix next observation",
                    "skill_id": "skill",
                }
            ],
            "loss_mask": {"L_trans": True},
        }
    ]
    cache_report = _attach_full_base_embedding_cache(model, rows, encode_batch_size=8)
    model.encoded_texts.clear()

    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=rows,
        skill_id_to_idx={"skill": 0},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 1.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )

    assert cache_report["used"] is True
    assert "prefix initial observation" not in model.encoded_texts
    assert "prefix action" not in model.encoded_texts
    assert "prefix next observation" not in model.encoded_texts
    assert "current state" not in model.encoded_texts
    assert "expert action" not in model.encoded_texts
    assert "target next observation" not in model.encoded_texts


def test_full_base_embedding_cache_attaches_next_state_embedding():
    model = _CountingEncodeModel(dim=3)
    rows = [{"state_text": "current state", "next_state_text": "trusted successor state"}]

    report = _attach_full_base_embedding_cache(model, rows, encode_batch_size=8)

    assert report["used"] is True
    assert "trusted successor state" in model.encoded_texts
    assert torch.is_tensor(rows[0]["_next_state_embedding"])


def test_full_base_cache_uses_prompted_role_only_for_state_and_next_state():
    model = _RoleAwareCountingModel()
    rows = [
        {
            "state_text": "current state",
            "action_text": "current action",
            "next_observation_text": "new observation",
            "next_state_text": "successor state",
        }
    ]

    _attach_full_base_embedding_cache(model, rows, encode_batch_size=8)

    assert set(model.encoded_states) == {"current state", "successor state"}
    assert set(model.encoded_observations) == {"current action", "new observation"}


def test_batch_cached_or_encode_requires_explicit_state_or_transition_role():
    model = _RoleAwareCountingModel()
    rows = [{"state_text": "current state", "action_text": "current action"}]

    _batch_cached_or_encode(
        model,
        rows,
        "_state_embedding",
        "state_text",
        torch.device("cpu"),
        text_role=STATE_QUERY_ROLE,
    )
    _batch_cached_or_encode(
        model,
        rows,
        "_action_embedding",
        "action_text",
        torch.device("cpu"),
        text_role=TRANSITION_TEXT_ROLE,
    )

    assert model.encoded_states == ["current state"]
    assert model.encoded_observations == ["current action"]


def _causal_adjacency_rows() -> list[dict]:
    return [
        {
            "benchmark": "toy",
            "trajectory_id": "traj-a",
            "step_index": 0,
            "state_text": "current state",
            "action_text": "act",
            "next_observation_text": "observation after act",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "provenance": {"source_id": "source-a", "split": "train"},
            "loss_mask": {"L_policy": True, "L_trans_skill_ce": True, "STOP": True},
        },
        {
            "benchmark": "toy",
            "trajectory_id": "traj-a",
            "step_index": 1,
            "state_text": "trusted successor state",
            "skill_id": "skill/next",
            "provenance": {"source_id": "source-a", "split": "train"},
            "loss_mask": {"routing": True},
        },
    ]


def test_attach_adjacent_next_states_materializes_trusted_successor_state():
    assert hasattr(full_base_train_module, "_attach_adjacent_next_states")
    rows = _causal_adjacency_rows()

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert "next_state_text" not in rows[0]
    assert prepared[0]["next_state_text"] == "trusted successor state"
    assert prepared[0]["next_state_source"] == "adjacent_trajectory_row"
    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert report["row_count"] == 2
    assert report["attached_rows"] == 1
    assert report["skip_reasons"] == {}


@pytest.mark.parametrize("existing_value", [None, "", "   "])
def test_attach_adjacent_next_states_materializes_empty_existing_values(existing_value):
    rows = _causal_adjacency_rows()
    rows[0]["next_state_text"] = existing_value
    rows[0]["next_state_source"] = "stale source"
    rows[0]["_next_state_embedding"] = torch.tensor([99.0])

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["next_state_text"] == "trusted successor state"
    assert prepared[0]["next_state_source"] == "adjacent_trajectory_row"
    assert "_next_state_embedding" not in prepared[0]
    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert "causal_next_state_skip_reason" not in prepared[0]
    assert report["materialized_rows"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"step_index": 2}, "missing_adjacent_step"),
        ({"trajectory_id": "traj-b"}, "missing_adjacent_step"),
        ({"benchmark": "other"}, "missing_adjacent_step"),
        ({"provenance": {"source_id": "source-b", "split": "train"}}, "missing_adjacent_step"),
        ({"provenance": {"source_id": "source-a", "split": "dev"}}, "missing_adjacent_step"),
        ({"skill_id": "skill/other"}, "adjacent_skill_mismatch"),
        ({"state_text": "   "}, "empty_next_state"),
    ],
)
def test_attach_adjacent_next_states_masks_untrusted_successors(mutation, expected_reason):
    rows = _causal_adjacency_rows()
    rows[1].update(mutation)

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["loss_mask"] == {"L_policy": True, "L_trans_skill_ce": False, "STOP": True}
    assert prepared[0]["causal_next_state_skip_reason"] == expected_reason
    assert "next_state_text" not in prepared[0]
    assert report["skip_reasons"][expected_reason] == 1


def test_attach_adjacent_next_states_rejects_duplicate_step_indices():
    rows = _causal_adjacency_rows()
    rows.append(dict(rows[1]))

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is False
    assert prepared[0]["causal_next_state_skip_reason"] == "duplicate_step_index"
    assert report["skip_reasons"]["duplicate_step_index"] == 1


def test_attach_adjacent_next_states_rejects_non_integer_step_index():
    rows = _causal_adjacency_rows()
    rows[0]["step_index"] = "0.5"

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is False
    assert prepared[0]["causal_next_state_skip_reason"] == "invalid_step_index"
    assert report["skip_reasons"]["invalid_step_index"] == 1


@pytest.mark.parametrize(
    ("missing_key", "fallback", "reason"),
    [
        ("state_text", {"task_text": "must not fallback"}, "missing_state_text"),
        ("action_text", {"expert_action": "must not fallback"}, "missing_action_text"),
        (
            "next_observation_text",
            {"observation_text": "must not fallback"},
            "missing_next_observation_text",
        ),
        ("skill_id", {"canonical_skill_id": "skill/current"}, "missing_current_skill"),
        ("next_skill_id", {"next_action_text": "must not fallback"}, "missing_next_skill"),
    ],
)
def test_attach_adjacent_next_states_masks_missing_current_causal_fields(missing_key, fallback, reason):
    rows = _causal_adjacency_rows()
    rows[0].pop(missing_key)
    rows[0].update(fallback)
    rows[0]["next_state_text"] = "stale state"
    rows[0]["next_state_source"] = "stale source"

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["loss_mask"] == {"L_policy": True, "L_trans_skill_ce": False, "STOP": True}
    assert prepared[0]["causal_next_state_skip_reason"] == reason
    assert "next_state_text" not in prepared[0]
    assert "next_state_source" not in prepared[0]
    assert report["skip_reasons"][reason] == 1


def test_attach_adjacent_next_states_allows_missing_source_namespace_but_does_not_cross_nonempty_sources():
    rows = _causal_adjacency_rows()
    rows[0]["provenance"].pop("source_id")
    rows[1]["provenance"].pop("source_id")

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["next_state_text"] == "trusted successor state"
    assert report["attached_rows"] == 1

    rows = _causal_adjacency_rows()
    rows[1]["provenance"]["source_id"] = "other-source"
    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)
    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is False
    assert prepared[0]["causal_next_state_skip_reason"] == "missing_adjacent_step"
    assert report["skip_reasons"]["missing_adjacent_step"] == 1


def test_attach_adjacent_next_states_preserves_and_validates_existing_next_state():
    rows = _causal_adjacency_rows()
    rows[0]["next_state_text"] = "trusted successor state"

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert prepared[0]["next_state_text"] == "trusted successor state"
    assert prepared[0]["next_state_source"] == "adjacent_trajectory_row"
    assert report["attached_rows"] == 1
    assert report["preserved_rows"] == 1

    rows[0]["next_state_text"] = "stale successor state"
    rows[0]["next_state_source"] = "stale source"
    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)
    assert "next_state_text" not in prepared[0]
    assert "next_state_source" not in prepared[0]
    assert prepared[0]["loss_mask"]["L_trans_skill_ce"] is False
    assert prepared[0]["causal_next_state_skip_reason"] == "existing_next_state_mismatch"
    assert report["skip_reasons"]["existing_next_state_mismatch"] == 1
    assert "stale successor state" not in full_base_train_module._stage0_handoff_raw_query(prepared[0], target="next")


def test_attach_adjacent_next_states_only_disables_causal_ce_on_failure():
    rows = _causal_adjacency_rows()
    rows.pop()

    prepared, report = full_base_train_module._attach_adjacent_next_states(rows)

    assert len(prepared) == 1
    assert prepared[0]["loss_mask"] == {"L_policy": True, "L_trans_skill_ce": False, "STOP": True}
    assert prepared[0]["causal_next_state_skip_reason"] == "missing_adjacent_step"
    assert report["masked_rows"] == 1


def test_attach_auto_replay_prefixes_uses_same_trajectory_previous_steps_only():
    rows = [
        {
            "trajectory_id": "traj-a",
            "step_index": 2,
            "state_text": "a step 2",
            "action_text": "a action 2",
            "skill_id": "skill/a2",
        },
        {
            "trajectory_id": "traj-b",
            "step_index": 0,
            "state_text": "b step 0",
            "action_text": "b action 0",
            "skill_id": "skill/b0",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 0,
            "state_text": "a step 0",
            "action_text": "a action 0",
            "next_observation_text": "a observed 1",
            "skill_id": "skill/a0",
        },
        {
            "trajectory_id": "traj-a",
            "step_index": 1,
            "state_text": "a step 1",
            "action_text": "a action 1",
            "next_observation_text": "a observed 2",
            "skill_id": "skill/a1",
        },
    ]

    prepared, report = _attach_auto_replay_prefixes(rows, max_steps=2)

    assert prepared[0]["state_text"] == "a step 2"
    assert [step["observation_text"] for step in prepared[0]["replay_prefix"]] == ["a step 0", "a step 1"]
    assert [step["next_observation_text"] for step in prepared[0]["replay_prefix"]] == ["a observed 1", "a observed 2"]
    assert [step["skill_id"] for step in prepared[0]["replay_prefix"]] == ["skill/a0", "skill/a1"]
    assert "replay_prefix" not in prepared[1]
    assert report["rows_with_auto_replay_prefix"] == 2
    assert report["total_prefix_steps"] == 3
    assert report["max_prefix_len"] == 2


def test_attach_auto_replay_prefixes_respects_max_steps_and_existing_prefixes():
    rows = [
        {"trajectory_id": "traj", "step_index": 0, "state_text": "s0", "action_text": "a0", "skill_id": "k0"},
        {"trajectory_id": "traj", "step_index": 1, "state_text": "s1", "action_text": "a1", "skill_id": "k1"},
        {"trajectory_id": "traj", "step_index": 2, "state_text": "s2", "action_text": "a2", "skill_id": "k2"},
        {
            "trajectory_id": "traj",
            "step_index": 3,
            "state_text": "s3",
            "action_text": "a3",
            "skill_id": "k3",
            "replay_prefix": [{"observation_text": "official"}],
        },
    ]

    prepared, report = _attach_auto_replay_prefixes(rows, max_steps=1)

    assert [step["observation_text"] for step in prepared[2]["replay_prefix"]] == ["s1"]
    assert prepared[3]["replay_prefix"] == [{"observation_text": "official"}]
    assert report["rows_with_existing_replay_prefix"] == 1
    assert report["rows_with_auto_replay_prefix"] == 2
    assert report["max_prefix_len"] == 1


def test_attach_auto_replay_prefixes_requires_consecutive_steps():
    rows = [
        {
            "benchmark": "alfworld",
            "source_id": "source-a",
            "split": "train",
            "trajectory_id": "traj",
            "step_index": 0,
            "state_text": "s0",
            "action_text": "a0",
            "skill_id": "k0",
        },
        {
            "benchmark": "alfworld",
            "source_id": "source-a",
            "split": "train",
            "trajectory_id": "traj",
            "step_index": 2,
            "state_text": "s2",
            "action_text": "a2",
            "skill_id": "k2",
        },
    ]

    prepared, report = _attach_auto_replay_prefixes(rows, max_steps=2)

    assert "replay_prefix" not in prepared[1]
    assert report["rows_with_auto_replay_prefix"] == 0


def test_attach_auto_replay_prefixes_rejects_duplicate_steps():
    rows = [
        {
            "trajectory_id": "traj",
            "step_index": step_index,
            "state_text": state_text,
            "action_text": f"action {state_text}",
            "skill_id": f"skill/{state_text}",
        }
        for step_index, state_text in ((0, "s0"), (1, "s1-a"), (1, "s1-b"), (2, "s2"))
    ]

    prepared, report = _attach_auto_replay_prefixes(rows, max_steps=3)

    assert all("replay_prefix" not in row for row in prepared)
    assert report["rows_with_auto_replay_prefix"] == 0


@pytest.mark.parametrize("identity_field", ["benchmark", "source_id", "split"])
def test_attach_auto_replay_prefixes_does_not_cross_identity_boundaries(identity_field):
    first = {
        "benchmark": "alfworld",
        "source_id": "source-a",
        "split": "train",
        "trajectory_id": "traj",
        "step_index": 0,
        "state_text": "s0",
        "action_text": "a0",
        "skill_id": "k0",
    }
    second = {
        **first,
        "step_index": 1,
        "state_text": "s1",
        "action_text": "a1",
        "skill_id": "k1",
    }
    second[identity_field] = {
        "benchmark": "webshop",
        "source_id": "source-b",
        "split": "valid",
    }[identity_field]

    prepared, report = _attach_auto_replay_prefixes([first, second], max_steps=2)

    assert "replay_prefix" not in prepared[1]
    assert report["rows_with_auto_replay_prefix"] == 0
    assert report["trajectory_count"] == 2


class _TinyFullBaseModel(torch.nn.Module):
    def __init__(self, dim: int = 8, skill_count: int = 3):
        super().__init__()
        self.dim = dim
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = torch.nn.Linear(dim, skill_count, bias=False)
        self.stop_head = torch.nn.Sequential(torch.nn.Linear(dim * 2, dim), torch.nn.GELU(), torch.nn.Linear(dim, 1))
        self.transition = torch.nn.Linear(dim * 2, dim)
        self.gate = torch.nn.Linear(dim * 3, dim)

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            vec = torch.zeros(self.dim, device=self.device)
            for idx, token in enumerate(("mug", "drawer", "kitchen", "teleport", "look", "take", "open", "put")):
                if token in lowered:
                    vec[idx] += 1.0
            if vec.sum() == 0:
                vec[0] = 0.1
            rows.append(vec + self.anchor * 0.0)
        return torch.stack(rows)


class _CountingEncodeModel(torch.nn.Module):
    def __init__(self, dim: int = 3):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.dim = dim
        self.skill_table = None
        self.transition = None
        self.gate = None
        self.stop_head = None
        self.encoded_texts: list[str] = []

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        self.encoded_texts.extend(str(text) for text in texts)
        rows = []
        for text in texts:
            lowered = str(text).lower()
            vec = torch.zeros(self.dim, device=self.device)
            if "expert" in lowered:
                vec[0] = 1.0
            elif "distractor" in lowered:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            rows.append(vec + self.anchor * 0.0)
        return torch.stack(rows)

    def initial_belief(self, h, top_k=None):
        del top_k
        return h


class _RoleAwareCountingModel(_CountingEncodeModel):
    def __init__(self, dim: int = 3):
        super().__init__(dim=dim)
        self.encoded_states: list[str] = []
        self.encoded_observations: list[str] = []

    def encode_states(self, texts):
        self.encoded_states.extend(str(text) for text in texts)
        return super().encode_observations(texts)

    def encode_observations(self, texts):
        self.encoded_observations.extend(str(text) for text in texts)
        return super().encode_observations(texts)


class _RecordingNativeSkillHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, candidate_embs, belief):
        del belief
        self.calls += 1
        return candidate_embs[:, :, 0] * 8.0


class _NativePolicyHeadModel(_CountingEncodeModel):
    def __init__(self):
        super().__init__(dim=3)
        self.skill_head = _RecordingNativeSkillHead()


class _LinearQSuccessHead(torch.nn.Module):
    def forward(self, h, m, candidate_embs):
        del h, m
        return candidate_embs[:, :, 0] - candidate_embs[:, :, 1]


class _FailingActionAdapter(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("legacy UniversalActionAdapter should not be used when native skill_head exists")


class _RecordingGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred, memory, obs_emb):
        self.calls.append((pred.detach().clone(), memory.detach().clone(), obs_emb.detach().clone()))
        return torch.ones_like(pred)


class _BeliefModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _RecordingGate()


class _ObservationConditioningTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_inputs = []
        self.action_inputs = []

    def forward(self, m_obs, action_input, obs_emb):
        self.action_inputs.append(action_input.detach().clone())
        self.obs_inputs.append(obs_emb.detach().clone())
        return m_obs + obs_emb


class _ObservationConditioningSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _ObservationConditioningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _ObservationConditioningSkillTable()
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.action_proj.weight.copy_(torch.eye(3))
        self.transition = _ObservationConditioningTransition()
        self.gate = _RecordingGate()
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "next observation marker" in lowered:
                rows.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
            elif "action marker" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0


class _OracleLeakCheckModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.transition = self
        self.gate = _RecordingGate()
        self.skill_table = None
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "next observation" in lowered:
                rows.append(torch.tensor([9.0, 9.0], dtype=torch.float32))
            elif "expert action" in lowered:
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def __call__(self, m_obs, labels, action_emb):
        del labels, action_emb
        return m_obs + torch.tensor([[0.0, 0.5]], dtype=m_obs.dtype, device=m_obs.device)


class _AlwaysFirstSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(2))

    def logits(self, h):
        return torch.tensor([[10.0, -10.0]], dtype=h.dtype, device=h.device).expand(h.size(0), 2)


class _NextFirstCurrentSecondSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(2))

    def logits(self, h):
        first_is_next = h[:, 0] > 5.0
        logits = torch.empty(h.size(0), 2, dtype=h.dtype, device=h.device)
        logits[first_is_next] = torch.tensor([10.0, -10.0], dtype=h.dtype, device=h.device)
        logits[~first_is_next] = torch.tensor([-10.0, 10.0], dtype=h.dtype, device=h.device)
        return logits


class _SubspaceTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _AlwaysFirstSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            if "next observation" in str(text):
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def __call__(self, m_obs, labels, action_emb):
        del labels, action_emb
        return m_obs


class _GradientTargetSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(2))

    def logits(self, h):
        return torch.zeros(h.size(0), 2, dtype=h.dtype, device=h.device)


class _TargetStopGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pred = torch.nn.Parameter(torch.tensor([[0.2, 0.8]], dtype=torch.float32))
        self.skill_table = _GradientTargetSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None

    @property
    def device(self):
        return self.pred.device

    def encode_observations(self, texts):
        return torch.ones(len(texts), 2, dtype=torch.float32, device=self.device)

    def __call__(self, m_obs, labels, action_emb):
        del m_obs, labels
        return self.pred.expand(action_emb.size(0), -1)


class _BeliefGateSubspaceCheckModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.transition = self
        self.gate = _RecordingGate()
        self.skill_table = _NextFirstCurrentSecondSkillTable()
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            if "next observation" in str(text):
                rows.append(torch.tensor([9.0, 9.0], dtype=torch.float32))
            elif "expert action" in str(text):
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def __call__(self, m_obs, labels, action_emb):
        del labels, action_emb
        return m_obs


class _ReplaySkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(2))

    def logits(self, h):
        first = h[:, 0] > h[:, 1]
        logits = torch.empty(h.size(0), 2, dtype=h.dtype, device=h.device)
        logits[first] = torch.tensor([10.0, -10.0], dtype=h.dtype, device=h.device)
        logits[~first] = torch.tensor([-10.0, 10.0], dtype=h.dtype, device=h.device)
        return logits


class _RecordingTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, m_obs, labels, action_emb):
        del labels, action_emb
        self.calls.append(m_obs.detach().clone())
        return m_obs


class _FixedObservationGate(torch.nn.Module):
    def forward(self, pred, m_tilde_next, action_emb):
        del pred, action_emb
        return torch.ones_like(m_tilde_next)


class _ReplayPrefixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _ReplaySkillTable()
        self.transition = _RecordingTransition()
        self.gate = _FixedObservationGate()
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            text = str(text)
            if "prefix next" in text or "target next" in text:
                rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def initial_belief(self, h, top_k=None):
        del top_k
        logits = self.skill_table.logits(h)
        return torch.softmax(logits, dim=-1) @ self.skill_table.E.to(device=h.device, dtype=h.dtype)


class _TrainableReplaySkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(2), requires_grad=False)

    def logits(self, h):
        return h


class _TrainableReplayTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, m_obs, action_input, obs_emb):
        del obs_emb
        return m_obs + self.scale * action_input


class _TrainableReplayGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, m_tilde_next, obs_emb):
        del obs_emb
        return torch.sigmoid(self.logit).expand_as(pred + m_tilde_next)


class _TrainableReplayPrefixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.encoder_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.skill_table = _TrainableReplaySkillTable()
        self.initial_belief_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.initial_belief_head.weight.copy_(torch.eye(2))
        self.initial_belief_calls = 0
        self.transition = _TrainableReplayTransition()
        self.gate = _TrainableReplayGate()
        self.stop_head = None

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "initial" in lowered:
                rows.append(torch.tensor([3.0, 0.0], dtype=torch.float32))
            elif "action" in lowered:
                rows.append(torch.tensor([0.5, 0.0], dtype=torch.float32))
            elif "next" in lowered:
                rows.append(torch.tensor([0.0, 3.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) * self.encoder_scale + self.anchor * 0.0

    def initial_belief(self, h, top_k=None):
        del top_k
        self.initial_belief_calls += 1
        return self.initial_belief_head(h)


class _FixedPolicyAdapter(torch.nn.Module):
    def forward(self, state_embs, action_embs, candidate_mask):
        del state_embs, action_embs, candidate_mask
        return torch.tensor([[2.0, 0.0]], dtype=torch.float32)


class _SkillCeSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _SkillCeTransitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pred = torch.nn.Parameter(torch.tensor([[0.0, 6.0, 0.0]], dtype=torch.float32))
        self.skill_table = _SkillCeSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None

    @property
    def device(self):
        return self.pred.device

    def encode_observations(self, texts):
        return torch.zeros(len(texts), 3, dtype=torch.float32, device=self.device) + self.pred.sum() * 0.0

    def __call__(self, m_obs, labels, action_emb):
        del m_obs, labels
        return self.pred.expand(action_emb.size(0), -1)


class _MisleadingSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        del h
        return torch.tensor([[8.0, -8.0, 0.0]], dtype=torch.float32)


class _SkillCeTransHead(torch.nn.Module):
    def forward(self, pred, candidate_emb):
        del pred
        return candidate_emb[:, :, 0] * 5.0


class _TransHeadSkillCeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pred = torch.nn.Parameter(torch.zeros(1, 3))
        self.skill_table = _MisleadingSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None
        self.action_emb = torch.nn.Embedding(3, 2)
        with torch.no_grad():
            self.action_emb.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [3.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
        self.trans_head = _SkillCeTransHead()

    @property
    def device(self):
        return self.pred.device

    def encode_observations(self, texts):
        return torch.zeros(len(texts), 3, dtype=torch.float32, device=self.device) + self.pred.sum() * 0.0

    def __call__(self, m_obs, labels, action_emb):
        del m_obs, labels
        return self.pred.expand(action_emb.size(0), -1)


class _PriorResidualSkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3))

    def logits(self, h):
        return h


class _PriorResidualTransitionModel(torch.nn.Module):
    def __init__(self, *, prior: list[float], residual: list[float]):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _PriorResidualSkillTable()
        self.transition = self
        self.gate = None
        self.stop_head = None
        self.prior = torch.tensor([prior], dtype=torch.float32)
        self.residual = torch.tensor([residual], dtype=torch.float32)

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "action marker" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            elif "next observation marker" in lowered:
                rows.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
            else:
                rows.append(torch.zeros(3, dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def __call__(self, m_obs, action_input, obs_emb):
        del m_obs, action_input
        is_prior = torch.all(obs_emb == 0, dim=-1, keepdim=True).to(dtype=obs_emb.dtype)
        prior = self.prior.to(device=obs_emb.device, dtype=obs_emb.dtype).expand(obs_emb.size(0), -1)
        residual = self.residual.to(device=obs_emb.device, dtype=obs_emb.dtype).expand(obs_emb.size(0), -1)
        return is_prior * prior + (1.0 - is_prior) * residual


class _UnifiedMemorySkillTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(3), requires_grad=False)

    def retrieval_logits(self, h):
        return torch.zeros(h.size(0), 3, dtype=h.dtype, device=h.device)

    def belief_logits(self, h):
        return h @ self.E.to(device=h.device, dtype=h.dtype).t()


class _ZeroUnifiedMemoryGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred, observation_memory, observation_embedding):
        self.calls.append(
            (
                pred.detach().clone(),
                observation_memory.detach().clone(),
                observation_embedding.detach().clone(),
            )
        )
        return torch.zeros_like(pred)


class _UnifiedMemoryStage2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.skill_table = _UnifiedMemorySkillTable()
        self.transition = self
        self.gate = _ZeroUnifiedMemoryGate()
        self.stop_head = None
        self.unified_route_calls = []
        self.transition_calls = 0
        self.transition_inputs = []

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "next" in lowered:
                rows.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            else:
                rows.append(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def initial_belief(self, h_t, top_k=None):
        del top_k
        return h_t

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = torch.tensor([[0.0, -1.0, 8.0]], dtype=h_t.dtype, device=h_t.device).expand(h_t.size(0), -1)
        if candidate_rows is None:
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output

    def unified_route_full_logits(self, h_t, m_t):
        return self.unified_route_logits(h_t, m_t)

    def __call__(self, m_obs, action_input, obs_emb):
        self.transition_calls += 1
        self.transition_inputs.append(
            (m_obs.detach().clone(), action_input.detach().clone(), obs_emb.detach().clone())
        )
        return m_obs + action_input + obs_emb


class _MemorySensitiveUnifiedMemoryStage2Model(_UnifiedMemoryStage2Model):
    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = m_t @ self.skill_table.E.to(device=m_t.device, dtype=m_t.dtype).t()
        if candidate_rows is None:
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output

    def __call__(self, m_obs, action_input, obs_emb):
        del m_obs, action_input
        return obs_emb


class _CounterfactualUnifiedMemoryStage2Model(_UnifiedMemoryStage2Model):
    def __init__(self):
        super().__init__()
        self.unified_route_outputs = []

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "action alpha" in lowered:
                row = [0.8, 0.1, 0.0]
            elif "action beta" in lowered:
                row = [0.1, 0.0, 0.9]
            elif "observation alpha" in lowered:
                row = [0.0, 0.7, 0.2]
            elif "observation beta" in lowered:
                row = [0.2, 0.1, 0.8]
            elif "future alpha" in lowered:
                row = [0.2, 0.8, 0.1]
            elif "future beta" in lowered:
                row = [0.7, 0.1, 0.4]
            else:
                row = [1.0, 0.0, 0.0]
            rows.append(torch.tensor(row, dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        self.unified_route_calls.append((h_t.detach().clone(), m_t.detach().clone(), candidate_rows))
        full_logits = (h_t + m_t) @ self.skill_table.E.to(device=h_t.device, dtype=h_t.dtype).t()
        if candidate_rows is None:
            output = full_logits
        else:
            width = max(len(row) for row in candidate_rows)
            output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
            for row_idx, row in enumerate(candidate_rows):
                ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
                output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        self.unified_route_outputs.append(output.detach().clone())
        return output


class _GradientStage2Transition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(27, dtype=torch.float32).view(3, 9) / 50.0 + 0.02)

    def forward(self, m_obs, action_input, obs_emb):
        return self.linear(torch.cat([m_obs, action_input, obs_emb], dim=-1))


class _GradientStage2Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(27, dtype=torch.float32).view(3, 9) / 100.0 - 0.05)

    def forward(self, pred, observation_memory, observation_embedding):
        return torch.sigmoid(self.linear(torch.cat([pred, observation_memory, observation_embedding], dim=-1)))


class _GradientUnifiedMemoryStage2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.skill_table = _UnifiedMemorySkillTable()
        self.initial_belief_head = torch.nn.Linear(3, 3, bias=False)
        self.action_proj = torch.nn.Linear(3, 3, bias=False)
        self.transition = _GradientStage2Transition()
        self.gate = _GradientStage2Gate()
        self.unified_retriever = torch.nn.Linear(6, 3, bias=False)
        self.route_memory_utility_gate = RouteMemoryUtilityGate(3, initial_alpha=0.01)
        self.trans_head = torch.nn.Linear(6, 1, bias=False)
        self.stop_head = None
        with torch.no_grad():
            self.initial_belief_head.weight.copy_(torch.eye(3) + 0.1)
            self.action_proj.weight.copy_(torch.eye(3) * 0.7 + 0.05)
            self.unified_retriever.weight.copy_(
                torch.arange(18, dtype=torch.float32).view(3, 6) / 40.0 + 0.03
            )

    @property
    def device(self):
        return self.anchor.device

    def encode_observations(self, texts):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            if "current action" in lowered:
                row = [0.1, 1.0, 0.2]
            elif "next observation" in lowered:
                row = [0.0, 0.2, 1.0]
            elif "next full state" in lowered:
                row = [0.3, 0.4, 1.0]
            else:
                row = [1.0, 0.2, 0.1]
            rows.append(torch.tensor(row, dtype=torch.float32))
        return torch.stack(rows) + self.anchor * 0.0

    def initial_belief(self, h_t, top_k=None):
        del top_k
        return self.initial_belief_head(h_t)

    def unified_route_logits(self, h_t, m_t, candidate_rows=None):
        routed = self.unified_retriever(torch.cat([h_t, m_t], dim=-1))
        full_logits = routed @ self.skill_table.E.to(device=routed.device, dtype=routed.dtype).t()
        if candidate_rows is None:
            return full_logits
        width = max(len(row) for row in candidate_rows)
        output = torch.full((len(candidate_rows), width), -1000.0, dtype=h_t.dtype, device=h_t.device)
        for row_idx, row in enumerate(candidate_rows):
            ids = torch.tensor(row, dtype=torch.long, device=h_t.device)
            output[row_idx, : len(row)] = full_logits[row_idx].index_select(0, ids)
        return output

    def unified_route_full_logits(self, h_t, m_t):
        return self.unified_route_logits(h_t, m_t)

    def route_memory_alpha(self, h_t, static_memory, dynamic_memory, causal_update_count):
        raw = self.route_memory_utility_gate(h_t, static_memory, dynamic_memory)
        counts = causal_update_count.to(device=raw.device, dtype=raw.dtype)
        return torch.where(counts > 0, raw, torch.zeros_like(raw))


class _TrainSkillHead(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(dim * 3, 1)

    def forward(self, candidate_embs, memory):
        memory_exp = memory.unsqueeze(1).expand_as(candidate_embs)
        return self.linear(torch.cat([candidate_embs, memory_exp, candidate_embs * memory_exp], dim=-1)).squeeze(-1)


class _TrainTransHead(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(dim + dim, 1)

    def forward(self, pred, candidate_emb):
        pred_exp = pred.unsqueeze(1).expand(-1, candidate_emb.size(1), -1)
        return self.linear(torch.cat([pred_exp, candidate_emb], dim=-1)).squeeze(-1)


class _NativeFullBaseTrainModel(_TinyFullBaseModel):
    def __init__(self, dim: int = 8, skill_count: int = 3):
        super().__init__(dim=dim, skill_count=skill_count)
        self.skill_head = _TrainSkillHead(dim)
        self.action_emb = torch.nn.Embedding(skill_count + 1, dim)
        self.trans_head = _TrainTransHead(dim)


def test_belief_prediction_uses_next_observation_belief_as_correction_input():
    model = _BeliefModel()
    pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    current_memory = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    next_observation_belief = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    action_emb = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    belief = _belief_prediction(model, pred, next_observation_belief, action_emb)

    assert torch.equal(model.gate.calls[0][1], next_observation_belief)
    assert not torch.equal(model.gate.calls[0][1], current_memory)
    assert torch.equal(belief, next_observation_belief)


def test_full_base_transition_and_belief_condition_on_next_observation_embedding_not_action_text():
    model = _ObservationConditioningModel()
    batch = [
        {
            "state_text": "state marker",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans": True, "L_trans_skill_ce": True, "belief": True},
        }
    ]

    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 1.0,
            "L_trans_skill_ce": 1.0,
            "belief": 1.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    expected_next_observation = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    action_text_embedding = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    assert len(model.transition.obs_inputs) == 4
    assert len(model.transition.action_inputs) == 4
    assert sum(torch.allclose(obs_input, torch.zeros_like(expected_next_observation)) for obs_input in model.transition.obs_inputs) == 1
    assert sum(torch.allclose(obs_input, expected_next_observation) for obs_input in model.transition.obs_inputs) == 3
    assert any(not torch.allclose(action_input.float().view(1, -1), action_text_embedding) for action_input in model.transition.action_inputs)
    residual_action_inputs = [
        action_input
        for action_input, obs_input in zip(model.transition.action_inputs, model.transition.obs_inputs)
        if torch.allclose(obs_input, expected_next_observation)
    ]
    for action_input in residual_action_inputs:
        assert torch.allclose(action_input, action_text_embedding)
        assert not torch.allclose(action_input, expected_next_observation)
    assert torch.allclose(model.gate.calls[0][2], expected_next_observation)


def test_full_base_belief_loss_feeds_next_skill_subspace_to_gate_not_raw_observation():
    model = _BeliefGateSubspaceCheckModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "expert action",
            "next_observation_text": "next observation",
            "skill_id": "skill",
            "loss_mask": {"belief": True},
        }
    ]

    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill": 0},
        device=torch.device("cpu"),
    )

    gate_memory = model.gate.calls[0][1]
    assert torch.allclose(gate_memory, torch.tensor([[1.0, 0.0]]), atol=1.0e-4)
    assert not torch.equal(gate_memory, torch.tensor([[9.0, 9.0]]))


def test_full_base_transition_targets_next_skill_subspace_not_raw_next_observation():
    model = _SubspaceTargetModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "expert action",
            "next_observation_text": "next observation",
            "skill_id": "skill",
            "loss_mask": {"L_trans": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill": 0},
        device=torch.device("cpu"),
        loss_weights={"L_trans": 1.0, "L_policy": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-5)
    assert metrics["transition_cosine_loss"] == pytest.approx(0.0, abs=1.0e-5)


def test_full_base_transition_target_stop_gradient_does_not_update_skill_table():
    model = _TargetStopGradientModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "expert action",
            "next_observation_text": "next observation",
            "skill_id": "skill",
            "loss_mask": {"L_trans": True},
        }
    ]

    loss, _metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill": 0},
        device=torch.device("cpu"),
        loss_weights={"L_trans": 1.0, "L_policy": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )
    loss.backward()

    assert model.pred.grad is not None
    assert model.skill_table.E.grad is None


def test_full_base_loss_uses_official_replay_prefix_to_reconstruct_current_belief():
    model = _ReplayPrefixModel()
    batch = [
        {
            "state_text": "current state without replay would map to first skill",
            "action_text": "expert action",
            "next_observation_text": "target next observation",
            "skill_id": "skill",
            "replay_prefix": [
                {
                    "observation_text": "initial observation",
                    "action_text": "prefix action",
                    "next_observation_text": "prefix next observation",
                    "skill_id": "skill",
                }
            ],
            "loss_mask": {"L_trans": True},
        }
    ]

    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill": 0},
        device=torch.device("cpu"),
        loss_weights={"L_trans": 1.0, "L_policy": 0.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
    )

    current_loss_transition_input = model.transition.calls[-1]
    assert torch.allclose(current_loss_transition_input, torch.tensor([[0.0, 1.0]]), atol=1.0e-4)


def test_trainable_replay_prefix_backpropagates_through_short_unroll_heads():
    model = _TrainableReplayPrefixModel()
    replay_prefix = [
        {
            "observation_text": "initial observation",
            "action_text": "prefix action",
            "next_observation_text": "prefix next observation",
            "skill_id": "skill",
        }
    ]

    replayed, used = _replay_prefix_belief(
        model,
        replay_prefix,
        fallback_m=torch.zeros(1, 2),
        skill_id_to_idx={"skill": 0},
        skill_count=2,
        device=torch.device("cpu"),
        trainable=True,
    )
    loss = replayed[:, 0].sum()
    loss.backward()

    assert used is True
    assert model.transition.scale.grad is not None
    assert model.transition.scale.grad.abs().item() > 0
    assert model.gate.logit.grad is not None
    assert model.gate.logit.grad.abs().item() > 0


def test_trainable_replay_prefix_initializes_through_model_initial_belief():
    model = _TrainableReplayPrefixModel()
    replayed, used = _replay_prefix_belief(
        model,
        [
            {
                "observation_text": "initial observation",
                "action_text": "prefix action",
                "next_observation_text": "prefix next observation",
                "skill_id": "skill",
            }
        ],
        fallback_m=torch.full((1, 2), 99.0),
        skill_id_to_idx={"skill": 0},
        skill_count=2,
        device=torch.device("cpu"),
        trainable=True,
    )
    replayed.sum().backward()

    assert used is True
    assert model.initial_belief_calls == 1
    assert model.initial_belief_head.weight.grad is not None
    assert model.initial_belief_head.weight.grad.abs().sum().item() > 0
    assert model.encoder_scale.grad is None


def test_replay_prefix_requires_model_initial_belief():
    model = _TrainableReplayPrefixModel()
    model.initial_belief = None

    with pytest.raises(ValueError, match="causal replay requires model.initial_belief"):
        _replay_prefix_belief(
            model,
            [{"observation_text": "initial observation", "action_text": "prefix action", "skill_id": "skill"}],
            fallback_m=torch.zeros(1, 2),
            skill_id_to_idx={"skill": 0},
            skill_count=2,
            device=torch.device("cpu"),
            trainable=True,
        )


def test_full_base_stage0_handoff_uses_retrieval_logits_without_skill_table_forward():
    class RetrievalAndBeliefSkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.eye(2), requires_grad=False)

        def retrieval_logits(self, h):
            return h @ self.E.t()

        def belief_logits(self, h):
            del h
            return torch.tensor([[0.0, 4.0]], dtype=torch.float32)

    model = type("Model", (), {"skill_table": RetrievalAndBeliefSkillTable()})()

    logits, memory = _skill_logits_and_memory(model, torch.tensor([[1.0, 0.0]]), skill_count=2)

    assert torch.allclose(logits, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(memory, torch.softmax(model.skill_table.belief_logits(logits), dim=-1) @ model.skill_table.E)


def test_stage0_score_prior_calibrates_flat_real_scores_to_rank_prior_scale():
    rows = [
        {
            "stage0_next_candidate_skill_indices": [10, 11, 12, 13, 14],
            "stage0_next_candidate_skill_scores": [0.51, 0.50, 0.49, 0.48, 0.47],
        }
    ]
    candidate_rows = [[10, 11, 12, 13, 14]]

    prior = _stage0_candidate_prior_scores_tensor(
        rows,
        candidate_rows,
        width=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        calibration="rank_std",
    )
    rank_prior = _stage0_rank_prior_logits_like(torch.zeros(1, 5)).squeeze(0)

    assert prior is not None
    assert torch.argsort(prior[0], descending=True).tolist() == [0, 1, 2, 3, 4]
    assert torch.isclose(prior[0].std(unbiased=False), rank_prior.std(unbiased=False), atol=1.0e-5)
    assert float(prior[0, 0] - prior[0, -1]) > 1.0


def test_stage0_score_prior_is_disabled_by_default_for_rank_prior_mainline():
    prior = _stage0_candidate_prior_scores_tensor(
        [
            {
                "stage0_next_candidate_skill_indices": [0, 1, 2],
                "stage0_next_candidate_skill_scores": [0.0, 1.0, 9.0],
            }
        ],
        [[0, 1, 2]],
        width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prior is None


def test_stage0_score_prior_prefers_next_candidate_scores_for_transition_rows():
    rows = [
        {
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_next_candidate_skill_scores": [3.0, 2.0, 1.0],
            "stage0_candidate_skill_indices": [0, 1, 2],
            "stage0_candidate_skill_scores": [1.0, 2.0, 3.0],
        }
    ]

    prior = _stage0_candidate_prior_scores_tensor(
        rows,
        [[0, 1, 2]],
        width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        calibration="rank_std",
    )

    assert prior is not None
    assert torch.argsort(prior[0], descending=True).tolist() == [0, 1, 2]


def test_stage0_handoff_cache_reuses_rows_only_for_matching_manifest(tmp_path):
    cache_root = tmp_path / "handoff_cache"
    rows = [
        {
            "row_id": "r1",
            "stage0_candidate_skill_indices": [0, 1],
            "stage0_candidate_skill_scores": [1.0, 0.5],
        }
    ]
    report = {"enabled": True, "retained_rows": 1, "top_m": 2}
    key = _stage0_handoff_cache_key(
        rows_digest="rows-a",
        skills_digest="skills-a",
        checkpoint_digest="ckpt-a",
        top_m=2,
        positive_missing_policy="skip",
        query_mode="skillrouter_state",
        inventory_min_candidates=0,
        allow_full_pool_stage2_debug=False,
    )

    written = _write_stage0_handoff_cache(cache_root, key, rows, report)
    hit = _load_stage0_handoff_cache(cache_root, key)
    miss = _load_stage0_handoff_cache(cache_root, {**key, "top_m": 3})

    assert written["status"] == "written"
    assert hit is not None
    cached_rows, cached_report = hit
    assert cached_rows == rows
    assert cached_report["retained_rows"] == 1
    assert cached_report["cache_hit"] is True
    assert cached_report["cache_key"] == key["cache_key"]
    assert miss is None


def test_stage0_handoff_cache_key_tracks_effective_unified_static_scorer():
    base = {
        "rows_digest": "rows-a",
        "skills_digest": "skills-a",
        "checkpoint_digest": {"sha256": "checkpoint-a"},
        "top_m": 2,
        "positive_missing_policy": "skip",
        "query_mode": "skillrouter_state",
        "inventory_min_candidates": 0,
        "allow_full_pool_stage2_debug": False,
        "static_candidate_scorer": "unified_static",
        "initial_belief_top_k": 64,
        "route_scorer": "unified_memory",
        "state_text_format": "default",
        "skill_text_format": "default",
        "skill_embedding_digest": "embedding-a",
        "declared_pool_order_digest": "order-a",
        "candidate_selection_version": "stable_declared_pool_v1",
        "tie_break_policy": "declared_pool_index_ascending",
        "unified_static_scorer_digest": "scorer-a",
        "next_skill_pool_mode": "full_pool",
    }

    key = _stage0_handoff_cache_key(**base)
    changed_topk = _stage0_handoff_cache_key(**{**base, "initial_belief_top_k": 32})
    changed_order = _stage0_handoff_cache_key(**{**base, "declared_pool_order_digest": "order-b"})
    changed_selector = _stage0_handoff_cache_key(
        **{**base, "candidate_selection_version": "stable_declared_pool_v2"}
    )
    changed_pool_mode = _stage0_handoff_cache_key(
        **{**base, "next_skill_pool_mode": "stage0_candidates"}
    )

    assert key["version"] == "stage0_handoff_cache_v2_unified_static"
    assert key["cache_key"] != changed_topk["cache_key"]
    assert key["cache_key"] != changed_order["cache_key"]
    assert key["cache_key"] != changed_selector["cache_key"]
    assert key["cache_key"] != changed_pool_mode["cache_key"]


def test_stage0_handoff_cache_content_digests_track_file_and_tensor_bytes(tmp_path):
    checkpoint = tmp_path / "stage0.pt"
    checkpoint.write_bytes(b"first")
    first_file = full_base_train_module._path_content_digest(checkpoint)
    first_tensor = full_base_train_module._tensor_content_digest(torch.tensor([[1.0, 2.0]]))

    checkpoint.write_bytes(b"second")
    second_file = full_base_train_module._path_content_digest(checkpoint)
    second_tensor = full_base_train_module._tensor_content_digest(torch.tensor([[1.0, 3.0]]))

    assert first_file["sha256"] != second_file["sha256"]
    assert first_tensor != second_tensor


def test_stage0_handoff_rows_digest_ignores_runtime_replay_prefixes():
    base = {
        "row_id": "r1",
        "trajectory_id": "traj",
        "step_index": 2,
        "benchmark": "webshop",
        "skill_id": "skill/a",
        "next_skill_id": "skill/b",
        "state_text": "state",
        "next_observation_text": "next",
        "loss_mask": {"L_trans_skill_ce": True},
    }
    with_prefix = dict(base)
    with_prefix["replay_prefix"] = [{"skill_id": "skill/prev"}]
    with_other_next_state = {**base, "next_state_text": "different successor state"}

    assert _stage0_handoff_rows_digest([base]) == _stage0_handoff_rows_digest([with_prefix])
    assert _stage0_handoff_rows_digest([base]) != _stage0_handoff_rows_digest([with_other_next_state])


def test_stage0_handoff_rows_digest_tracks_visible_inventory():
    base = {
        "row_id": "r1",
        "benchmark": "toolbench_g3",
        "trajectory_id": "traj",
        "step_index": 0,
        "skill_id": "skill/a",
        "next_skill_id": "skill/b",
        "state_text": "state",
        "next_observation_text": "next",
        "loss_mask": {"L_trans_skill_ce": True},
    }

    inventory_a = {**base, "visible_inventory_skill_ids": ["skill/a", "skill/b"]}
    inventory_b = {**base, "visible_inventory_skill_ids": ["skill/a", "skill/c"]}

    assert _stage0_handoff_rows_digest([inventory_a]) != _stage0_handoff_rows_digest([inventory_b])


def test_stage0_handoff_next_query_uses_trusted_next_state():
    row = {
        "state_text": "current state",
        "action_text": "current action",
        "next_observation_text": "environment observation",
        "next_state_text": "trusted successor state",
    }

    assert full_base_train_module._stage0_handoff_raw_query(row, target="next") == "trusted successor state"


def test_stage0_handoff_cache_merge_preserves_current_replay_prefix():
    current_rows = [
        {
            "row_id": "r1",
            "state_text": "state",
            "replay_prefix": [{"skill_id": "skill/prev"}],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]
    cached_rows = [
        {
            "row_id": "r1",
            "state_text": "state",
            "stage0_candidate_skill_indices": [0, 1],
            "stage0_candidate_skill_scores": [2.0, 1.0],
            "loss_mask": {"L_trans_skill_ce": False},
        }
    ]

    merged_rows, report = _merge_stage0_handoff_cached_rows(current_rows, cached_rows)

    assert report["cached_rows"] == 1
    assert report["missing_current_rows"] == 0
    assert merged_rows[0]["replay_prefix"] == [{"skill_id": "skill/prev"}]
    assert merged_rows[0]["stage0_candidate_skill_indices"] == [0, 1]
    assert merged_rows[0]["loss_mask"] == {"L_trans_skill_ce": False}


def test_stage0_handoff_cache_identity_includes_next_state():
    shared = {
        "row_id": "r1",
        "trajectory_id": "traj",
        "step_index": 0,
        "benchmark": "toy",
        "skill_id": "skill/a",
        "next_skill_id": "skill/b",
        "state_text": "state",
        "next_observation_text": "observation",
    }
    current_rows = [
        {
            **shared,
            "next_state_text": "new successor",
            "replay_prefix": [{"skill_id": "skill/prev"}],
            "loss_mask": {"L_policy": True, "L_trans_skill_ce": False},
            "causal_next_state_skip_reason": "existing_next_state_mismatch",
        }
    ]
    cached_rows = [
        {
            **shared,
            "next_state_text": "stale successor",
            "next_state_source": "stale source",
            "stage0_candidate_skill_indices": [0, 1],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    merged_rows, report = _merge_stage0_handoff_cached_rows(current_rows, cached_rows)

    assert report["missing_current_rows"] == 1
    assert report["missing_cached_rows"] == 1
    assert report["valid"] is False
    assert merged_rows == []


def test_full_base_freeze_trains_belief_calibration_without_unfreezing_retrieval_foundation():
    class _SkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.ones(2, 2))
            self.W = torch.nn.Linear(2, 2, bias=False)
            self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_retr = torch.nn.Parameter(torch.zeros(2))
            self.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_belief = torch.nn.Parameter(torch.zeros(2))

    model = type(
        "Model",
        (torch.nn.Module,),
        {
            "__init__": lambda self: (
                torch.nn.Module.__init__(self),
                setattr(self, "skill_table", _SkillTable()),
                setattr(self, "transition", torch.nn.Linear(2, 2)),
                setattr(self, "gate", torch.nn.Linear(6, 2)),
                setattr(self, "skill_head", torch.nn.Linear(2, 1)),
            )[-1],
        },
    )()

    audit = _freeze_for_full_base(model)

    assert "skill_table.logit_scale_belief" in audit["trainable_modules"]
    assert "skill_table.skill_bias_belief" in audit["trainable_modules"]
    assert model.skill_table.logit_scale_belief.requires_grad
    assert model.skill_table.skill_bias_belief.requires_grad
    assert not model.skill_table.E.requires_grad
    assert not model.skill_table.W.weight.requires_grad
    assert not model.skill_table.logit_scale_retr.requires_grad
    assert not model.skill_table.skill_bias_retr.requires_grad


def test_full_base_transition_skill_logits_use_retrieval_logits_without_skill_table_forward():
    class RetrievalOnlySkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.eye(2), requires_grad=False)

        def retrieval_logits(self, h):
            return h @ self.E.t()

    model = type("Model", (), {"skill_table": RetrievalOnlySkillTable()})()

    logits, head_type = _transition_skill_logits(model, torch.tensor([[0.0, 1.0]]), skill_count=2)

    assert head_type == "skill_table_logits"
    assert torch.allclose(logits, torch.tensor([[0.0, 1.0]]))


def test_retrieval_contrastive_loss_excludes_positive_from_negative_set():
    logits = torch.tensor([[0.1, 5.0, 3.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)

    loss, audit = _retrieval_contrastive_loss_from_logits(
        logits,
        labels,
        num_negatives=3,
        hard_ratio=1.0,
        return_audit=True,
    )

    assert loss.item() == pytest.approx(
        torch.nn.functional.cross_entropy(torch.tensor([[5.0, 3.0, 2.0, 0.1]]), torch.tensor([0])).item()
    )
    assert audit["positive_indices"] == [1]
    assert 1 not in audit["negative_indices"][0]


def test_transition_inventory_mask_filters_cross_root_candidates_and_preserves_gold():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/weather/forecast",
        }
    ]
    skill_ids_by_idx = {
        0: "toolbench-g3/weather/current",
        1: "traject/weatherapi-com",
        2: "toolbench-g3/weather/forecast",
        3: "toolret/weather-lookup",
    }

    filtered, audit = _filter_transition_candidates_by_inventory(
        rows=rows,
        candidate_rows=[[1, 2, 3, 0]],
        labels=torch.tensor([2]),
        skill_ids_by_idx=skill_ids_by_idx,
        mode="auto",
    )

    assert filtered == [[2, 0]]
    assert audit["inventory_mask_applied_rows"] == 1
    assert audit["inventory_mask_removed_candidates"] == 2
    assert audit["inventory_mask_positive_missing_rows"] == 0


def test_transition_explicit_only_inventory_never_infers_target_root_namespace():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/weather/forecast",
        }
    ]
    skill_ids_by_idx = {
        0: "traject/weatherapi-com",
        1: "toolbench-g3/weather/forecast",
        2: "toolret/weather-lookup",
    }

    filtered, audit = _filter_transition_candidates_by_inventory(
        rows=rows,
        candidate_rows=[[0, 1, 2]],
        labels=torch.tensor([1]),
        skill_ids_by_idx=skill_ids_by_idx,
        mode="explicit_only",
    )

    assert filtered == [[0, 1, 2]]
    assert audit["inventory_mask_applied_rows"] == 0
    assert audit["inventory_mask_missing_rows"] == 1


def test_transition_inventory_mask_backfills_to_min_candidates_from_stage0_topm():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/weather/forecast",
        }
    ]
    skill_ids_by_idx = {
        0: "toolbench-g3/weather/current",
        1: "traject/weatherapi-com",
        2: "toolbench-g3/weather/forecast",
        3: "toolret/weather-lookup",
        4: "toolbench-g3/weather/alerts",
        5: "traject/calendar",
    }

    filtered, audit = _filter_transition_candidates_by_inventory(
        rows=rows,
        candidate_rows=[[1, 2, 3, 0, 4, 5]],
        labels=torch.tensor([2]),
        skill_ids_by_idx=skill_ids_by_idx,
        mode="auto",
        min_candidates=4,
    )

    assert filtered == [[2, 0, 4, 1]]
    assert audit["inventory_mask_applied_rows"] == 1
    assert audit["inventory_mask_backfilled_rows"] == 1
    assert audit["inventory_mask_backfilled_candidates"] == 1
    assert audit["inventory_mask_removed_candidates"] == 2
    assert audit["inventory_mask_positive_missing_rows"] == 0


def test_transition_stage0_topk_trajectory_prior_keeps_score_order_without_hard_filtering():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/weather/forecast",
            "available_skill_ids": ["toolbench-g3/weather/alerts"],
        }
    ]
    skill_ids_by_idx = {
        0: "traject/weatherapi-com",
        1: "toolret/weather-lookup",
        2: "toolbench-g3/weather/forecast",
        3: "toolbench-g3/weather/current",
        4: "toolbench-g3/weather/alerts",
        5: "traject/calendar",
    }

    filtered, audit = _filter_transition_candidates_by_inventory(
        rows=rows,
        candidate_rows=[[0, 1, 2, 3, 4, 5]],
        labels=torch.tensor([2]),
        skill_ids_by_idx=skill_ids_by_idx,
        mode="stage0_topk_trajectory_prior",
        min_candidates=4,
    )

    assert filtered == [[0, 1, 2, 4]]
    assert audit["inventory_mask_applied_rows"] == 1
    assert audit["inventory_mask_removed_candidates"] == 2
    assert audit["inventory_mask_positive_missing_rows"] == 0


def test_transition_stage0_topk_trajectory_prior_preserves_gold_inside_stage0_topm():
    rows = [
        {
            "benchmark": "toolbench_g3",
            "next_skill_id": "toolbench-g3/weather/forecast",
            "available_skill_ids": ["toolbench-g3/weather/alerts"],
        }
    ]
    skill_ids_by_idx = {
        0: "traject/weatherapi-com",
        1: "toolret/weather-lookup",
        2: "toolbench-g3/weather/current",
        3: "toolbench-g3/weather/alerts",
        4: "traject/calendar",
        5: "toolbench-g3/weather/forecast",
    }

    filtered, audit = _filter_transition_candidates_by_inventory(
        rows=rows,
        candidate_rows=[[0, 1, 2, 3, 4, 5]],
        labels=torch.tensor([5]),
        skill_ids_by_idx=skill_ids_by_idx,
        mode="stage0_topk_trajectory_prior",
        min_candidates=4,
    )

    assert filtered == [[0, 1, 3, 5]]
    assert audit["inventory_mask_applied_rows"] == 1
    assert audit["inventory_mask_removed_candidates"] == 2
    assert audit["inventory_mask_positive_missing_rows"] == 1


def test_transition_multi_positive_listwise_nll_uses_all_positive_candidates():
    logits = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)
    positive_mask = torch.tensor([[False, True, True]])

    loss = _multi_positive_listwise_nll(logits, positive_mask, reduction="none")

    expected = torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits[:, 1:], dim=-1)
    single_positive_ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1]), reduction="none")
    assert torch.allclose(loss, expected)
    assert loss.item() < single_positive_ce.item()


def test_transition_positive_mask_adds_explicit_equivalent_skill_ids():
    rows = [
        {
            "next_skill_id": "skill/gold",
            "positive_next_skill_ids": ["skill/equiv"],
        }
    ]
    candidate_rows = [[0, 1, 2]]
    labels = torch.tensor([1])
    skill_ids_by_idx = {0: "skill/negative", 1: "skill/gold", 2: "skill/equiv"}

    mask, counts = _transition_positive_mask(
        rows=rows,
        candidate_rows=candidate_rows,
        labels=labels,
        skill_ids_by_idx=skill_ids_by_idx,
        positive_mode="gold_plus_equivalent",
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [[False, True, True]]
    assert counts["transition_positive_mean_count"] == pytest.approx(2.0)


def test_equivalent_skill_ids_by_skill_id_uses_skill_pool_alias_groups():
    skills = [
        {
            "skill_id": "traject/email-tool",
            "canonical_skill_id": "traject/email-tool",
            "alias_skill_ids": ["traject/email-tool", "toolret/email-tool"],
        },
        {
            "skill_id": "toolret/email-tool",
            "canonical_skill_id": "traject/email-tool",
            "alias_skill_ids": ["traject/email-tool", "toolret/email-tool"],
        },
        {
            "skill_id": "traject/other",
            "canonical_skill_id": "traject/other",
            "alias_skill_ids": ["traject/other"],
        },
    ]

    equivalent = _equivalent_skill_ids_by_skill_id(skills)

    assert equivalent["traject/email-tool"] == ["toolret/email-tool"]
    assert equivalent["toolret/email-tool"] == ["traject/email-tool"]
    assert equivalent["traject/other"] == []


def test_transition_ranking_metrics_split_self_and_switch_rows():
    logits = torch.tensor([[3.0, 1.0], [0.0, 4.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    rows = [
        {"skill_id": "skill/a", "next_skill_id": "skill/a"},
        {"skill_id": "skill/a", "next_skill_id": "skill/b"},
    ]

    metrics = _ranking_metrics_from_logits(logits, labels, rows=rows)

    assert metrics["transition_skill_self_count"] == 1.0
    assert metrics["transition_skill_switch_count"] == 1.0
    assert metrics["transition_skill_self_recall@5"] == 1.0
    assert metrics["transition_skill_switch_recall@5"] == 1.0


def test_full_base_transition_inventory_mask_is_used_by_stage2_loss():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "benchmark": "toolbench_g3",
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "toolbench-g3/current",
            "next_skill_id": "toolbench-g3/next",
            "stage0_next_candidate_skill_indices": [2, 1, 0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={
            "toolbench-g3/current": 0,
            "toolbench-g3/next": 1,
            "traject/foreign": 2,
        },
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_inventory_mask_mode="auto",
    )

    assert metrics["transition_inventory_mask_mode"] == "auto"
    assert metrics["transition_inventory_mask_applied_rows"] == 1.0
    assert metrics["transition_inventory_mask_removed_candidates"] == 1.0
    assert metrics["transition_skill_ce_candidate_count"] == 2.0


def test_full_base_transition_listwise_loss_treats_equivalent_next_skill_as_positive():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/gold",
            "positive_next_skill_ids": ["skill/equiv"],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]
    common = {
        "model": model,
        "action_adapter": object(),
        "batch": batch,
        "skill_id_to_idx": {"skill/gold": 0, "skill/equiv": 1, "skill/current": 2},
        "device": torch.device("cpu"),
        "loss_weights": {
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        "transition_scoring_mode": "skill_prior_plus_action_observation_residual",
    }

    ce_loss, ce_metrics = _compute_full_base_loss(**common)
    listwise_loss, listwise_metrics = _compute_full_base_loss(
        **common,
        transition_loss_type="listwise_nll",
        transition_positive_mode="gold_plus_equivalent",
    )

    assert listwise_metrics["transition_loss_type"] == "listwise_nll"
    assert listwise_metrics["transition_positive_mode"] == "gold_plus_equivalent"
    assert listwise_metrics["transition_multi_positive_rows"] == 1.0
    assert listwise_metrics["transition_positive_mean_count"] == pytest.approx(2.0)
    assert listwise_metrics["transition_skill_recall@1"] == 1.0
    assert ce_metrics["transition_skill_recall@1"] == 0.0
    assert listwise_loss.item() < ce_loss.item()


def test_full_base_transition_listwise_loss_uses_skill_pool_alias_equivalents():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/gold",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]
    common = {
        "model": model,
        "action_adapter": object(),
        "batch": batch,
        "skill_id_to_idx": {"skill/gold": 0, "skill/equiv": 1, "skill/current": 2},
        "device": torch.device("cpu"),
        "loss_weights": {
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        "transition_scoring_mode": "skill_prior_plus_action_observation_residual",
    }

    listwise_loss, listwise_metrics = _compute_full_base_loss(
        **common,
        transition_loss_type="listwise_nll",
        transition_positive_mode="gold_plus_equivalent",
        equivalent_skill_ids_by_skill_id={"skill/gold": ["skill/equiv"]},
    )

    assert listwise_metrics["transition_multi_positive_rows"] == 1.0
    assert listwise_metrics["transition_positive_mean_count"] == pytest.approx(2.0)
    assert listwise_metrics["transition_skill_recall@1"] == 1.0
    assert listwise_loss.item() < 0.01


def test_compute_full_base_loss_applies_configurable_weights_to_policy_loss_only():
    model = _TinyFullBaseModel(dim=8, skill_count=3)
    batch = [
        {
            "state_text": "goal: put mug in drawer",
            "action_text": "take mug 1 from table 1",
            "admissible_actions": ["take mug 1 from table 1", "look"],
            "expert_action": "take mug 1 from table 1",
            "loss_mask": {"L_policy": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=_FixedPolicyAdapter(),
        batch=batch,
        skill_id_to_idx={},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 2.5, "L_trans": 0.1, "belief": 0.1, "STOP": 0.1, "routing": 0.2},
    )

    assert loss.item() == pytest.approx(metrics["policy_ce_loss"] * 2.5)
    assert metrics["weighted_loss_terms"] == {
        "L_policy": pytest.approx(metrics["policy_ce_loss"] * 2.5),
        "routing": 0.0,
        "STOP": 0.0,
        "L_trans": 0.0,
        "L_trans_skill_ce": 0.0,
        "transition_hard_negative_margin": 0.0,
        "counterfactual_utility": 0.0,
        "counterfactual_history": 0.0,
        "hard_negative_margin": 0.0,
        "Q_success": 0.0,
        "belief": 0.0,
    }


def test_full_base_transition_skill_ce_scores_whole_skill_pool_from_prior():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert loss.item() == pytest.approx(metrics["transition_skill_ce_loss"])
    assert metrics["transition_skill_ce_loss"] < 0.01
    assert metrics["transition_skill_recall@1"] == 1.0
    assert metrics["transition_skill_recall@5"] == 1.0
    assert metrics["transition_skill_mrr"] == 1.0
    assert metrics["transition_skill_ce_candidate_count"] == 3.0


def test_full_base_transition_skill_ce_prefers_native_trans_head_over_skill_table_logits():
    model = _TransHeadSkillCeModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert loss.item() == pytest.approx(metrics["transition_skill_ce_loss"])
    assert metrics["transition_skill_ce_loss"] < 0.01
    assert metrics["transition_skill_head_type"] == "native_trans_head"
    assert metrics["transition_skill_recall@1"] == 1.0


def test_transition_prior_residual_lambda_zero_uses_skill_prior_not_noisy_residual():
    model = _PriorResidualTransitionModel(prior=[0.0, 6.0, 0.0], residual=[6.0, 0.0, 0.0])
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        transition_residual_lambda=0.0,
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert metrics["transition_scoring_mode"] == "skill_prior_plus_action_observation_residual"
    assert metrics["transition_residual_lambda"] == 0.0
    assert metrics["transition_prior_skill_recall@1"] == 1.0
    assert metrics["transition_residual_skill_recall@1"] == 0.0
    assert metrics["transition_skill_recall@1"] == 1.0


def test_stage2_rank_prior_scoring_lambda_zero_preserves_stage0_candidate_order():
    model = _PriorResidualTransitionModel(prior=[0.0, 8.0, 0.0], residual=[8.0, 0.0, 0.0])
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [2, 1, 0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/wrong": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        transition_residual_lambda=0.0,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
    )

    assert metrics["transition_scoring_mode"] == "stage0_rank_prior_plus_transition_residual"
    assert metrics["transition_prior_skill_recall@1"] == 1.0
    assert metrics["transition_residual_skill_recall@1"] == 0.0
    assert metrics["transition_skill_recall@1"] == 1.0


def test_stage2_unified_memory_route_scorer_uses_post_action_memory_and_unified_logits():
    model = _UnifiedMemoryStage2Model()
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "next_state_text": "next full state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [2, 1, 0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/wrong": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        route_scorer="unified_memory",
    )

    assert model.transition_calls == 1
    assert len(model.gate.calls) == 1
    assert len(model.unified_route_calls) == 3
    dynamic_h, dynamic_m, dynamic_candidates = model.unified_route_calls[1]
    static_h, static_m, static_candidates = model.unified_route_calls[2]
    assert torch.equal(dynamic_h, model.encode_observations(["next full state"]))
    assert torch.equal(static_h, dynamic_h)
    assert dynamic_candidates == static_candidates == [[2, 1, 0]]
    assert not torch.equal(dynamic_m, static_m)
    _m_t, action_input, observation_input = model.transition_inputs[0]
    assert torch.equal(action_input, model.encode_observations(["action marker"]))
    assert torch.equal(observation_input, model.encode_observations(["next observation marker"]))
    assert metrics["route_scorer"] == "unified_memory"
    assert metrics["transition_skill_head_type"] == "unified_memory_retriever"
    assert metrics["transition_scoring_mode"] == "unified_memory"
    assert metrics["uses_stage0_prior_at_inference"] is False
    assert metrics["transition_skill_recall@1"] == 1.0
    assert metrics["transition_unified_dynamic_skill_recall@1"] == 1.0
    assert metrics["transition_unified_static_skill_recall@1"] == 1.0
    assert metrics["transition_post_action_update_rows"] == 1.0
    assert metrics["transition_next_state_rows"] == 1.0


def test_stage2_unified_memory_full_pool_routes_from_post_action_next_state():
    model = _UnifiedMemoryStage2Model()

    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "action marker",
                "next_observation_text": "next observation marker",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/wrong": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        route_scorer="unified_memory",
    )

    assert model.transition_calls == 1
    dynamic_h, dynamic_m, dynamic_candidates = model.unified_route_calls[-2]
    static_h, static_m, static_candidates = model.unified_route_calls[-1]
    assert dynamic_candidates is None
    assert static_candidates is None
    assert torch.equal(dynamic_h, model.encode_observations(["next full state"]))
    assert torch.equal(static_h, dynamic_h)
    assert not torch.equal(dynamic_m, static_m)


def test_stage2_unified_memory_reports_dynamic_static_rank_flips_and_counterfactual_utility():
    model = _MemorySensitiveUnifiedMemoryStage2Model()
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "next_state_text": "future state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "replay_prefix": [
                {
                    "observation_text": "current state",
                    "action_text": "action marker",
                    "next_observation_text": "next observation marker",
                    "skill_idx": 0,
                }
            ],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.5,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        route_scorer="unified_memory",
        trainable_replay_prefix=True,
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        counterfactual_gain_margin=1.5,
        counterfactual_scale=1.0,
    )

    assert metrics["transition_unified_static_skill_recall@1"] == 0.0
    assert metrics["transition_unified_dynamic_skill_recall@1"] == 1.0
    assert metrics["transition_unified_dynamic_vs_static_delta_mrr"] > 0.0
    assert metrics["transition_unified_positive_rank_improved_rows"] == 1.0
    assert metrics["transition_unified_positive_rank_worsened_rows"] == 0.0
    assert metrics["transition_unified_argmax_changed_rows"] == 1.0
    assert metrics["counterfactual_eligible_rows"] == 1.0
    assert metrics["counterfactual_utility_loss"] > 0.0
    assert metrics["weighted_loss_terms"]["counterfactual_utility"] > 0.0


def _stage2_counterfactual_outputs(**row_overrides):
    model = _CounterfactualUnifiedMemoryStage2Model()
    row = {
        "state_text": "current state",
        "action_text": "action alpha",
        "next_observation_text": "observation alpha",
        "next_state_text": "future alpha",
        "skill_id": "skill/current",
        "next_skill_id": "skill/next",
        "stage0_next_candidate_skill_indices": [0, 1, 2],
        "loss_mask": {"L_trans_skill_ce": True},
    }
    row.update(row_overrides)
    _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=[row],
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        route_scorer="unified_memory",
    )
    return {
        "dynamic_h": model.unified_route_calls[-2][0],
        "dynamic_m": model.unified_route_calls[-2][1],
        "static_h": model.unified_route_calls[-1][0],
        "static_m": model.unified_route_calls[-1][1],
        "dynamic_logits": model.unified_route_outputs[-2],
        "static_logits": model.unified_route_outputs[-1],
    }


def test_stage2_unified_memory_action_counterfactual_changes_dynamic_not_static_logits():
    baseline = _stage2_counterfactual_outputs(action_text="action alpha")
    changed = _stage2_counterfactual_outputs(action_text="action beta")

    assert not torch.equal(baseline["dynamic_m"], changed["dynamic_m"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert torch.equal(baseline["static_h"], changed["static_h"])
    assert torch.equal(baseline["static_m"], changed["static_m"])
    assert torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage2_unified_memory_observation_counterfactual_changes_dynamic_not_static_logits():
    baseline = _stage2_counterfactual_outputs(next_observation_text="observation alpha")
    changed = _stage2_counterfactual_outputs(next_observation_text="observation beta")

    assert not torch.equal(baseline["dynamic_m"], changed["dynamic_m"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert torch.equal(baseline["static_h"], changed["static_h"])
    assert torch.equal(baseline["static_m"], changed["static_m"])
    assert torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage2_unified_memory_next_state_counterfactual_changes_both_route_branches():
    baseline = _stage2_counterfactual_outputs(next_state_text="future alpha")
    changed = _stage2_counterfactual_outputs(next_state_text="future beta")

    assert not torch.equal(baseline["dynamic_h"], changed["dynamic_h"])
    assert not torch.equal(baseline["dynamic_logits"], changed["dynamic_logits"])
    assert not torch.equal(baseline["static_h"], changed["static_h"])
    assert not torch.equal(baseline["static_m"], changed["static_m"])
    assert not torch.equal(baseline["static_logits"], changed["static_logits"])


def test_stage2_safe_memory_objective_backpropagates_only_through_causal_modules():
    model = _GradientUnifiedMemoryStage2Model()
    audit = _freeze_for_full_base(model, route_scorer="unified_memory")
    loss, _metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_indices": [0, 1, 2],
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.05,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
    )
    loss.backward()

    for module in (
        model.transition,
        model.gate,
        model.action_proj,
        model.route_memory_utility_gate,
    ):
        grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert grads
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0
    for module in (model.initial_belief_head, model.unified_retriever):
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.grad is None for parameter in module.parameters())
    assert "route_memory_utility_gate" in audit["trainable_modules"]
    assert "initial_belief_head" not in audit["trainable_modules"]
    assert "unified_retriever" not in audit["trainable_modules"]
    assert all(parameter.grad is None for parameter in model.trans_head.parameters())


def test_stage2_full_pool_keeps_static_miss_and_backpropagates():
    model = _GradientUnifiedMemoryStage2Model()
    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_indices": [0, 1],
                "stage0_next_positive_hit": False,
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.05,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        counterfactual_gain_margin=0.1,
        counterfactual_safety_tolerance=0.01,
        counterfactual_gain_weight=1.0,
        counterfactual_safety_weight=1.0,
        counterfactual_scale=1.0,
    )
    loss.backward()

    assert metrics["transition_full_pool_rows"] == 1.0
    assert metrics["transition_static_miss_train_rows"] == 1.0
    assert metrics["transition_skill_ce_candidate_count"] == 3.0
    assert metrics["counterfactual_eligible_rows"] == 1.0
    assert "counterfactual_utility_loss" in metrics
    assert model.transition.linear.weight.grad is not None
    assert model.gate.linear.weight.grad is not None


def test_zero_weight_transition_margin_does_not_execute_helper(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("zero-weight transition margin executed")

    monkeypatch.setattr(
        full_base_train_module,
        "_transition_hard_negative_margin_loss",
        fail,
    )
    _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {
                    "L_trans_skill_ce": True,
                    "transition_hard_negative_margin": True,
                },
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
    )


def test_canonical_stage2_skips_legacy_counterfactual_fusion(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("legacy Stage2 counterfactual fusion executed")

    monkeypatch.setattr(full_base_train_module, "bounded_memory_fusion", fail)
    monkeypatch.setattr(full_base_train_module, "build_local_candidate_masks", fail)
    monkeypatch.setattr(full_base_train_module, "safe_local_route_objective", fail)

    _loss, metrics = _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
    )

    assert metrics["counterfactual_training_objective"] == "disabled_stage4_cmc_owned"
    assert metrics["counterfactual_utility_loss"] == 0.0


def test_stage2_counterfactual_history_uses_matched_replay_donors(monkeypatch):
    captured = {}

    def fake_history_loss(**kwargs):
        captured.update(kwargs)
        return kwargs["true_logits"].sum() * 0.0 + 0.5

    monkeypatch.setattr(
        full_base_train_module,
        "shuffled_history_utility_loss",
        fake_history_loss,
    )
    prefix_a = [{
        "observation_text": "previous state alpha",
        "action_text": "current action",
        "next_observation_text": "next observation",
        "skill_id": "skill/current",
    }]
    prefix_b = [{
        "observation_text": "next observation",
        "action_text": "previous action beta",
        "next_observation_text": "previous observation beta",
        "skill_id": "skill/other",
    }]
    rows = [
        {
            "source_benchmark": "toolbench_g3",
            "state_text": "current state alpha",
            "action_text": "current action",
            "next_observation_text": "next observation",
            "next_state_text": "next full state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "replay_prefix": prefix_a,
            "loss_mask": {"L_trans_skill_ce": True},
        },
        {
            "source_benchmark": "toolbench_g3",
            "state_text": "current state beta",
            "action_text": "current action",
            "next_observation_text": "next observation",
            "next_state_text": "next full state",
            "skill_id": "skill/other",
            "next_skill_id": "skill/next",
            "replay_prefix": prefix_b,
            "loss_mask": {"L_trans_skill_ce": True},
        },
    ]

    _loss, metrics = _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=rows,
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights(
            "stage2", {"counterfactual_history": 1.0}
        ),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        trainable_replay_prefix=True,
        counterfactual_history_margin=0.2,
    )

    assert captured["margin"] == pytest.approx(0.2)
    assert captured["true_logits"].shape == captured["shuffled_logits"].shape == (2, 3)
    assert not torch.equal(captured["true_logits"], captured["shuffled_logits"])
    assert metrics["counterfactual_history_rank_loss"] == pytest.approx(0.5)
    assert metrics["counterfactual_history_loss"] >= 0.5
    assert metrics["counterfactual_history_eligible_rows"] == 2.0
    assert metrics["weighted_loss_terms"]["counterfactual_history"] == pytest.approx(
        metrics["counterfactual_history_loss"]
    )


def test_stage2_counterfactual_history_has_zero_eligible_rows_without_donor(monkeypatch):
    def fail(**_kwargs):
        raise AssertionError("counterfactual history loss ran without a valid donor")

    monkeypatch.setattr(full_base_train_module, "shuffled_history_utility_loss", fail)
    _loss, metrics = _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=[{
            "source_benchmark": "toolbench_g3",
            "state_text": "current state",
            "action_text": "current action",
            "next_observation_text": "next observation",
            "next_state_text": "next full state",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "replay_prefix": [{
                "observation_text": "previous state",
                "action_text": "previous action",
                "next_observation_text": "previous observation",
                "skill_id": "skill/current",
            }],
            "loss_mask": {"L_trans_skill_ce": True},
        }],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights(
            "stage2", {"counterfactual_history": 1.0}
        ),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        trainable_replay_prefix=True,
    )

    assert metrics["counterfactual_history_rank_loss"] == 0.0
    assert metrics["counterfactual_history_eligible_rows"] == 0.0


def test_anchored_stage2_loss_compares_static_and_dynamic_against_step_zero_teacher(
    monkeypatch,
):
    class _Teacher:
        def full_logits(self, _model, h):
            return h.new_tensor([[2.0, 1.0, 0.0]]).expand(h.size(0), -1)

        def post_action_full_logits(
            self,
            _model,
            *,
            h_current,
            current_skill_labels,
            action_embeddings,
            observation_embeddings,
            h_next,
        ):
            del current_skill_labels, action_embeddings, observation_embeddings, h_next
            return h_current.new_tensor([[2.0, 1.0, 0.0]]).expand(
                h_current.size(0), -1
            )

    def fake_anchor(_model, _teacher, h, valid_mask, **_precomputed_logits):
        assert tuple(valid_mask.shape) == (1, 3)
        return h.sum() * 0.0 + 0.5

    monkeypatch.setattr(full_base_train_module, "static_route_anchor_kl", fake_anchor)
    loss, metrics = _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        static_route_teacher=_Teacher(),
        static_route_anchor_weight=0.1,
    )

    assert loss.requires_grad
    assert metrics["stage2_static_route_anchor_rows"] == 1.0
    assert metrics["stage2_static_route_anchor_loss"] == pytest.approx(0.5)
    assert metrics["stage2_static_route_anchor_weighted_loss"] == pytest.approx(0.05)
    assert metrics["stage2_static_mrr_delta_vs_teacher"] <= 1.0
    assert metrics["stage2_dynamic_mrr_delta_vs_step_zero"] > 0.0


def test_anchored_stage2_excludes_replayed_rows_from_static_anchor(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("replayed row entered zero-history static anchor")

    monkeypatch.setattr(full_base_train_module, "static_route_anchor_kl", fail)
    _loss, metrics = _compute_full_base_loss(
        model=_GradientUnifiedMemoryStage2Model(),
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "next_state_text": "next full state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "replay_prefix": [
                    {
                        "state_text": "previous state",
                        "action_text": "previous action",
                        "next_observation_text": "previous observation",
                        "skill_id": "skill/other",
                    }
                ],
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/other": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        transition_inventory_mask_mode="explicit_only",
        trainable_replay_prefix=True,
        static_route_teacher=object(),
        static_route_anchor_weight=0.1,
    )

    assert metrics["stage2_static_route_anchor_rows"] == 0.0


@pytest.mark.parametrize(
    ("step", "max_steps", "fraction", "expected"),
    [(0, 100, 0.1, 0.0), (5, 100, 0.1, 0.5), (10, 100, 0.1, 1.0), (20, 100, 0.0, 1.0)],
)
def test_counterfactual_warmup_scale_uses_global_step(step, max_steps, fraction, expected):
    assert full_base_train_module._counterfactual_warmup_scale(step, max_steps, fraction) == pytest.approx(expected)


@pytest.mark.parametrize(
    "argument",
    [
        "counterfactual_gain_margin",
        "counterfactual_safety_tolerance",
        "counterfactual_gain_weight",
        "counterfactual_safety_weight",
    ],
)
def test_stage2_rejects_negative_counterfactual_hyperparameter_without_eligible_rows(argument):
    kwargs = {
        "counterfactual_gain_margin": 0.1,
        "counterfactual_safety_tolerance": 0.01,
        "counterfactual_gain_weight": 1.0,
        "counterfactual_safety_weight": 1.0,
    }
    kwargs[argument] = -0.1

    with pytest.raises(ValueError, match=argument):
        _compute_full_base_loss(
            model=_UnifiedMemoryStage2Model(),
            action_adapter=object(),
            batch=[{"state_text": "current state", "loss_mask": {}}],
            skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
            device=torch.device("cpu"),
            route_scorer="unified_memory",
            next_skill_pool_mode="full_pool",
            transition_inventory_mask_mode="explicit_only",
            **kwargs,
        )


def test_stage2_unified_memory_ignores_legacy_gated_temporal_auxiliary_loss():
    model = _CounterfactualUnifiedMemoryStage2Model()

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=[
            {
                "state_text": "current state",
                "action_text": "action alpha",
                "next_observation_text": "observation alpha",
                "next_state_text": "future alpha",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "stage0_next_candidate_skill_indices": [0, 1, 2],
                "loss_mask": {"L_trans_skill_ce": True},
            }
        ],
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        gated_temporal_kl_alpha=1.0,
        gated_temporal_rank_drop_beta=1.0,
        route_scorer="unified_memory",
    )

    assert metrics["transition_scoring_mode"] == "unified_memory"
    assert metrics["gated_temporal_prior_kl_loss"] == 0.0
    assert metrics["gated_temporal_rank_drop_loss"] == 0.0
    assert metrics["gated_temporal_kl_alpha"] == 0.0
    assert metrics["gated_temporal_rank_drop_beta"] == 0.0


def test_nonfinite_loss_guard_writes_blocker_report(tmp_path):
    loss = torch.tensor(float("nan"))
    metrics = {"counterfactual_utility_loss": float("nan"), "route_scorer": "unified_memory"}

    with pytest.raises(FloatingPointError):
        _raise_if_nonfinite_loss(loss, metrics=metrics, step_idx=7, output_dir=tmp_path)

    report = json.loads((tmp_path / "blocker_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
    assert report["reason"] == "nonfinite_loss"
    assert report["step"] == 7
    assert report["metrics"]["counterfactual_utility_loss"] == "nan"
    assert report["metrics"]["route_scorer"] == "unified_memory"


def test_gated_temporal_scoring_mode_reports_gate_and_prior_losses():
    model = _NativeFullBaseTrainModel(dim=3, skill_count=3)
    model.gated_temporal_reranker = GatedTemporalReranker(GatedTemporalConfig(dim=3, lambda_max=0.5))
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [2, 1, 0],
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/wrong": 1, "skill/next": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        gated_temporal_lambda_max=0.5,
        gated_temporal_kl_alpha=0.03,
        gated_temporal_rank_drop_beta=0.05,
    )

    assert loss.requires_grad
    assert metrics["transition_scoring_mode"] == "stage0_prior_gated_temporal_residual"
    assert metrics["transition_skill_head_type"] == "stage0_prior_gated_temporal_residual"
    assert 0.0 <= metrics["gated_temporal_lambda_mean"] <= 0.5
    assert metrics["gated_temporal_prior_kl_loss"] >= 0.0
    assert metrics["gated_temporal_rank_drop_loss"] >= 0.0


def test_freeze_for_gated_temporal_mode_trains_only_new_reranker():
    model = _NativeFullBaseTrainModel(dim=3, skill_count=3)
    model.gated_temporal_reranker = GatedTemporalReranker(GatedTemporalConfig(dim=3, lambda_max=0.5))

    audit = _freeze_for_full_base(
        model,
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        freeze_gated_temporal_only=True,
    )

    assert audit["trainable_modules"] == ["gated_temporal_reranker"]
    assert any(param.requires_grad for param in model.gated_temporal_reranker.parameters())
    assert all(not param.requires_grad for param in model.transition.parameters())
    assert all(not param.requires_grad for param in model.skill_head.parameters())
    assert all(not param.requires_grad for param in model.trans_head.parameters())


def test_freeze_for_gated_temporal_mode_still_trains_belief_calibration():
    class _SkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.ones(2, 2))
            self.W = torch.nn.Linear(2, 2, bias=False)
            self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_retr = torch.nn.Parameter(torch.zeros(2))
            self.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_belief = torch.nn.Parameter(torch.zeros(2))

    model = _NativeFullBaseTrainModel(dim=2, skill_count=2)
    model.skill_table = _SkillTable()
    model.gated_temporal_reranker = GatedTemporalReranker(GatedTemporalConfig(dim=2, lambda_max=0.5))

    audit = _freeze_for_full_base(
        model,
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        freeze_gated_temporal_only=True,
    )

    assert "gated_temporal_reranker" in audit["trainable_modules"]
    assert "skill_table.logit_scale_belief" in audit["trainable_modules"]
    assert "skill_table.skill_bias_belief" in audit["trainable_modules"]
    assert model.skill_table.logit_scale_belief.requires_grad
    assert model.skill_table.skill_bias_belief.requires_grad
    assert not model.skill_table.E.requires_grad
    assert not model.skill_table.W.weight.requires_grad
    assert not model.skill_table.logit_scale_retr.requires_grad
    assert not model.skill_table.skill_bias_retr.requires_grad


def test_freeze_for_unified_memory_route_preserves_static_router_and_trains_utility_gate():
    model = _NativeFullBaseTrainModel(dim=3, skill_count=3)
    model.initial_belief_head = torch.nn.Linear(3, 3)
    model.unified_retriever = torch.nn.Linear(3, 3)
    model.route_memory_utility_gate = RouteMemoryUtilityGate(3, initial_alpha=0.01)

    audit = _freeze_for_full_base(model, route_scorer="unified_memory")

    assert "initial_belief_head" not in audit["trainable_modules"]
    assert "unified_retriever" not in audit["trainable_modules"]
    assert "route_memory_utility_gate" in audit["trainable_modules"]
    assert "trans_head" not in audit["trainable_modules"]
    assert all(not param.requires_grad for param in model.initial_belief_head.parameters())
    assert all(not param.requires_grad for param in model.unified_retriever.parameters())
    assert any(param.requires_grad for param in model.route_memory_utility_gate.parameters())
    assert all(not param.requires_grad for param in model.trans_head.parameters())


def test_freeze_for_anchored_unified_memory_trains_router_and_freezes_qwen_geometry():
    class _AnchoredSkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.eye(3))
            self.W = torch.nn.Linear(3, 3, bias=False)
            self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))

    model = _NativeFullBaseTrainModel(dim=3, skill_count=3)
    model.skill_table = _AnchoredSkillTable()
    model.initial_belief_head = torch.nn.Linear(3, 3)
    model.unified_retriever = torch.nn.Linear(3, 3)
    model.route_memory_utility_gate = RouteMemoryUtilityGate(3, initial_alpha=0.01)

    audit = _freeze_for_full_base(
        model,
        route_scorer="unified_memory",
        anchored_routing_foundation=True,
    )

    for module_name in ("initial_belief_head", "unified_retriever"):
        assert module_name in audit["trainable_modules"]
        assert all(
            parameter.requires_grad
            for parameter in getattr(model, module_name).parameters()
        )
    for parameter_name in (
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    ):
        assert parameter_name in audit["trainable_modules"]
    assert model.skill_table.logit_scale_belief.requires_grad
    assert model.skill_table.skill_bias_belief.requires_grad
    assert model.skill_table.E.requires_grad is False
    assert model.skill_table.W.weight.requires_grad is False
    assert model.skill_table.logit_scale_retr.requires_grad is False
    assert model.skill_table.skill_bias_retr.requires_grad is False
    assert all(
        not parameter.requires_grad
        for parameter in model.route_memory_utility_gate.parameters()
    )
    assert audit["anchored_routing_foundation"] is True
    assert audit["frozen_routing_foundation"] is False
    assert audit["qwen_and_skill_table_frozen"] is True


def test_canonical_stage2_freezes_unused_stop_and_q_success_heads():
    model = _NativeFullBaseTrainModel(dim=3, skill_count=3)
    model.initial_belief_head = torch.nn.Linear(3, 3)
    model.unified_retriever = torch.nn.Linear(3, 3)
    model.q_success_head = _LinearQSuccessHead()

    audit = _freeze_for_full_base(
        model,
        route_scorer="unified_memory",
        anchored_routing_foundation=True,
        active_loss_weights=canonical_stage_loss_weights("stage2"),
    )

    assert "stop_head" not in audit["trainable_modules"]
    assert "q_success_head" not in audit["trainable_modules"]
    assert all(not parameter.requires_grad for parameter in model.stop_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.q_success_head.parameters())


def test_freeze_for_unified_memory_ignores_legacy_gated_temporal_only_policy():
    model = _GradientUnifiedMemoryStage2Model()
    model.gated_temporal_reranker = GatedTemporalReranker(
        GatedTemporalConfig(dim=3, lambda_max=0.5)
    )

    audit = _freeze_for_full_base(
        model,
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        freeze_gated_temporal_only=True,
        route_scorer="unified_memory",
    )

    assert audit["freeze_gated_temporal_only"] is False
    for module_name in (
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ):
        module = getattr(model, module_name)
        assert module_name in audit["trainable_modules"]
        assert any(param.requires_grad for param in module.parameters())
    for module_name in ("initial_belief_head", "unified_retriever"):
        module = getattr(model, module_name)
        assert module_name not in audit["trainable_modules"]
        assert all(not param.requires_grad for param in module.parameters())
    assert "trans_head" not in audit["trainable_modules"]
    assert all(not param.requires_grad for param in model.trans_head.parameters())
    assert "gated_temporal_reranker" not in audit["trainable_modules"]
    assert all(not param.requires_grad for param in model.gated_temporal_reranker.parameters())


def test_transition_prior_residual_positive_lambda_allows_residual_to_correct_prior():
    model = _PriorResidualTransitionModel(prior=[6.0, 0.0, 0.0], residual=[0.0, 7.0, 0.0])
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        transition_residual_lambda=1.0,
        transition_scoring_mode="skill_prior_plus_action_observation_residual",
    )

    assert metrics["transition_residual_lambda"] == 1.0
    assert metrics["transition_prior_skill_recall@1"] == 0.0
    assert metrics["transition_residual_skill_recall@1"] == 1.0
    assert metrics["transition_skill_recall@1"] == 1.0


def test_v4_1b_action_observation_scoring_replays_legacy_transition_ce():
    model = _PriorResidualTransitionModel(prior=[6.0, 0.0, 0.0], residual=[0.0, 4.0, 0.0])
    batch = [
        {
            "state_text": "current state",
            "action_text": "action marker",
            "next_observation_text": "next observation marker",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "loss_mask": {"L_trans_skill_ce": True},
        }
    ]

    _loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={"L_policy": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 1.0, "belief": 0.0, "STOP": 0.0, "routing": 0.0},
        transition_scoring_mode="v4_1b_action_observation",
    )

    assert metrics["transition_scoring_mode"] == "v4_1b_action_observation"
    assert metrics["transition_skill_recall@1"] == 1.0


def test_transition_skill_ce_downweights_injected_stage0_candidates():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_positive_injected": False,
            "loss_mask": {"L_trans_skill_ce": True},
        },
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/current",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_positive_injected": True,
            "loss_mask": {"L_trans_skill_ce": True},
        },
    ]
    common = {
        "model": model,
        "action_adapter": object(),
        "batch": batch,
        "skill_id_to_idx": {"skill/current": 0, "skill/next": 1, "skill/other": 2},
        "device": torch.device("cpu"),
        "loss_weights": {
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "transition_hard_negative_margin": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
    }

    _unweighted_loss, unweighted_metrics = _compute_full_base_loss(**common)
    weighted_loss, weighted_metrics = _compute_full_base_loss(
        **common,
        transition_real_candidate_ce_multiplier=1.0,
        transition_injected_candidate_ce_multiplier=0.0,
    )

    assert weighted_metrics["transition_real_candidate_rows"] == 1.0
    assert weighted_metrics["transition_injected_candidate_rows"] == 1.0
    assert weighted_metrics["transition_skill_ce_loss"] < unweighted_metrics["transition_skill_ce_loss"]
    assert weighted_loss.item() == pytest.approx(weighted_metrics["transition_skill_ce_loss"])


def test_transition_hard_negative_margin_uses_real_stage0_candidates_only():
    model = _SkillCeTransitionModel()
    batch = [
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/current",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_positive_injected": False,
            "loss_mask": {"L_trans_skill_ce": True},
        },
        {
            "state_text": "current state",
            "action_text": "current expert action",
            "skill_id": "skill/current",
            "next_skill_id": "skill/current",
            "stage0_next_candidate_skill_indices": [0, 1, 2],
            "stage0_positive_injected": True,
            "loss_mask": {"L_trans_skill_ce": True},
        },
    ]

    loss, metrics = _compute_full_base_loss(
        model=model,
        action_adapter=object(),
        batch=batch,
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        device=torch.device("cpu"),
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "transition_hard_negative_margin": 0.1,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        transition_hard_negative_margin=1.0,
    )

    assert metrics["transition_hard_negative_pair_count"] == 1.0
    assert metrics["transition_hard_negative_margin_loss"] > 0.0
    assert metrics["weighted_loss_terms"]["transition_hard_negative_margin"] > 0.0
    assert loss.item() > metrics["transition_skill_ce_loss"]


def test_stage2_unified_memory_training_report_describes_causal_post_action_routing(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/current", "name": "current", "description": "current"},
            {"skill_id": "skill/other", "name": "other", "description": "other"},
            {"skill_id": "skill/next", "name": "next", "description": "next"},
        ],
    )
    shared_identity = {
        "benchmark": "toolbench_g3",
        "trajectory_id": "causal-report",
        "provenance": {"source_id": "causal-report-source", "split": "train"},
    }
    _write_jsonl(
        train_path,
        [
            {
                **shared_identity,
                "task_id": "causal-report:0",
                "step_index": 0,
                "state_text": "current state",
                "action_text": "current action",
                "next_observation_text": "next observation",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": True},
                "source_quality": "official_replay",
            },
            {
                **shared_identity,
                "task_id": "causal-report:1",
                "step_index": 1,
                "state_text": "next full state",
                "skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": False},
                "source_quality": "successor_only",
            },
        ],
    )

    saved_freeze_configs = []
    original_save_latest = full_base_train_module.TrainingMonitor.save_latest

    def recording_save_latest(monitor, payload):
        config = payload.get("transition_candidate_training")
        if isinstance(config, dict):
            saved_freeze_configs.append(dict(config))
        return original_save_latest(monitor, payload)

    monkeypatch.setattr(full_base_train_module.TrainingMonitor, "save_latest", recording_save_latest)
    monkeypatch.setattr(
        full_base_train_module.TrainingMonitor,
        "_should_write",
        staticmethod(lambda _step, _interval: True),
    )
    model = _GradientUnifiedMemoryStage2Model()
    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 3, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        loss_weights={
            "L_policy": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "counterfactual_utility": 0.0,
            "belief": 0.0,
            "STOP": 0.0,
            "routing": 0.0,
        },
        embedding_cache_mode="never",
        allow_full_pool_stage2_debug=True,
        transition_scoring_mode="stage0_prior_gated_temporal_residual",
        freeze_gated_temporal_only=True,
        route_scorer="unified_memory",
    )

    semantics = report["transition_input_semantics"]
    assert report["transition_objective"] == (
        "causal_post_action_unified_retrieval_plus_optional_next_belief_cosine"
    )
    assert semantics["score_function"] == "unified_route_logits(h_{t+1}, m_{t+1})"
    assert semantics["static_memory"] == "initial_belief(h_{t+1})"
    assert semantics["dynamic_memory"] == (
        "post_action_update_of_replayed_or_initial_m_t_for_every_causal_stage2_row"
    )
    assert semantics["action_channel"] == (
        "actual_action_text_embedding_updates_m_{t+1}_for_every_causal_stage2_row"
    )
    assert semantics["observation_channel"] == (
        "next_observation_text_embedding_updates_m_{t+1}_for_every_causal_stage2_row"
    )
    assert semantics["next_state_channel"] == (
        "next_state_text_embedding_is_h_{t+1}_for_routing_and_observation_correction"
    )
    assert semantics["post_action_update"] == "required_for_every_causal_stage2_next_skill_ce_row"
    assert not hasattr(model, "gated_temporal_reranker")
    assert len(saved_freeze_configs) >= 3
    for config in [*saved_freeze_configs, report["transition_candidate_training"]]:
        assert config["scoring_mode"] == "unified_memory"
        assert config["freeze_gated_temporal_only"] is False
        assert config["freeze_gated_temporal_only_requested"] is True

    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["transition_objective"] == report["transition_objective"]
    assert payload["transition_input_semantics"] == semantics


def test_train_clstr_full_base_with_model_respects_loss_masks_and_writes_checkpoint(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "alfworld/alfworld-object-picker", "name": "pick", "description": "pick"},
            {"skill_id": "scienceworld/scienceworld-room-navigator", "name": "nav", "description": "nav"},
            {"skill_id": "webshop/webshop-search-executor", "name": "search", "description": "search"},
        ],
    )
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "alf-1:0",
                "trajectory_id": "alf-1",
                "step_index": 0,
                "goal_text": "put mug in drawer",
                "task_text": "pick_and_place",
                "state_text": "goal: put mug in drawer\nobservation: You see a mug.",
                "history_text": "",
                "action_text": "take mug 1 from table 1",
                "admissible_actions": ["take mug 1 from table 1", "look"],
                "expert_action": "take mug 1 from table 1",
                "next_observation_text": "You pick up the mug.",
                "done": False,
                "reward": None,
                "skill_id": "alfworld/alfworld-object-picker",
                "next_skill_id": "scienceworld/scienceworld-room-navigator",
                "loss_mask": {
                    "L_policy": True,
                    "L_trans": True,
                    "L_trans_skill_ce": True,
                    "belief": False,
                    "STOP": True,
                    "routing": True,
                },
                "source_quality": "official_replay",
                "provenance": {"source_id": "alfworld_official_train_replay", "split": "train"},
            },
            {
                "benchmark": "scienceworld",
                "task_id": "sci-1:0",
                "goal_text": "find kitchen",
                "task_text": "chemistry",
                "state_text": "observation: You are in the foundry.",
                "history_text": "look around",
                "action_text": "teleport to kitchen",
                "admissible_actions": [],
                "expert_action": "teleport to kitchen",
                "next_observation_text": "You teleport to the kitchen.",
                "done": False,
                "reward": 0.0,
                "skill_id": "scienceworld/scienceworld-room-navigator",
                "next_skill_id": None,
                "loss_mask": {
                    "L_policy": False,
                    "L_trans": True,
                    "L_trans_skill_ce": False,
                    "belief": True,
                    "STOP": True,
                    "routing": True,
                },
                "source_quality": "aux_only",
                "provenance": {"source_id": "aux_skillnet_rebuilt", "split": "train"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "alf-1:1",
                "trajectory_id": "alf-1",
                "step_index": 1,
                "state_text": "goal: put mug in drawer\nobservation: You are holding the mug.",
                "skill_id": "scienceworld/scienceworld-room-navigator",
                "loss_mask": {"L_trans_skill_ce": False},
                "source_quality": "successor_only",
                "provenance": {"source_id": "alfworld_official_train_replay", "split": "train"},
            },
        ],
    )
    class _ReportSkillTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.zeros(3, 8))
            self.W = torch.nn.Linear(8, 8, bias=False)
            self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))
            with torch.no_grad():
                self.E[:, :3].copy_(torch.eye(3))
                self.W.weight.copy_(torch.eye(8))

        def retrieval_logits(self, h):
            return h @ self.E.t() + self.skill_bias_retr.unsqueeze(0)

        def belief_logits(self, h):
            return h @ self.E.t() + self.skill_bias_belief.unsqueeze(0)

    model = _NativeFullBaseTrainModel(dim=8, skill_count=3)
    model.skill_table = _ReportSkillTable()

    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=3,
        batch_size=2,
        max_rows=2,
        learning_rate=1.0e-2,
        seed=5,
        loss_weights={"L_policy": 1.0, "L_trans": 0.1, "belief": 0.1, "STOP": 0.2, "routing": 0.2},
        allow_full_pool_stage2_debug=True,
    )

    assert report["status"] == "ok"
    assert report["training_objective"] == "component_complete_masked_multi_loss"
    assert report["transition_objective"] == "supervised_next_skill_ce_plus_optional_next_belief_cosine"
    assert report["transition_input_semantics"]["uses_actual_action_text_when_available"] is True
    assert report["transition_input_semantics"]["uses_next_observation_as_observation"] is True
    assert report["transition_input_semantics"]["action_text_used_as_observation"] is False
    assert report["belief_memory_semantics"] == {
        "routing_logits_source": "skill_table.retrieval_logits",
        "memory_source": "skill_table.belief_logits_via_subspace_obs",
        "stage1_stage2_train_eval_memory_consistent": True,
        "belief_calibration_params_present": True,
        "belief_calibration_trainable": True,
    }
    assert report["policy_feature_semantics"] == {
        "train_candidate_encoder": "action_text_bi_encoder_embeddings",
        "train_context": "belief",
        "skill_candidate_policy_eval_interface": "removed",
        "skill_routing_eval_uses": "route_logits_from_candidates_stage0_retrieval_prior",
        "train_eval_default_aligned": True,
    }
    assert report["transition_skill_ce"]["enabled"] is True
    assert report["transition_skill_ce"]["input_semantics"] == report["transition_input_semantics"]
    assert report["transition_skill_ce"]["target"] == "next_skill_id"
    assert report["training_regime"] == "offline_replay_supervised_pretraining"
    assert report["on_policy_rollout_used"] is False
    assert report["grpo_policy_loss_used"] is False
    assert report["clstr_native_act_trained"] is True
    assert report["not_full_clstr_act"] is False
    assert report["qdoc_adapter_used"] is False
    assert report["valid_or_test_used_for_training"] is False
    assert report["loss_activation_counts"] == {
        "L_policy": 1,
        "hard_negative_margin": 0,
        "Q_success": 0,
        "L_trans": 2,
        "L_trans_skill_ce": 1,
        "transition_hard_negative_margin": 0,
        "counterfactual_utility": 0,
        "counterfactual_history": 0,
        "STOP": 2,
        "belief": 1,
        "routing": 2,
    }
    assert report["bucket_counts"] == {"aux_only": 1, "official_replay": 1}
    assert report["benchmark_counts"] == {"alfworld": 1, "scienceworld": 1}
    assert report["loss_weights"] == {
        "L_policy": 1.0,
        "hard_negative_margin": 0.2,
            "Q_success": 0.5,
            "L_trans": 0.1,
            "L_trans_skill_ce": 1.0,
            "transition_hard_negative_margin": 0.0,
            "counterfactual_utility": 0.0,
            "counterfactual_history": 0.0,
            "belief": 0.1,
            "STOP": 0.2,
            "routing": 0.2,
    }
    assert "metric_averages" in report
    assert "retrieval_contrastive_loss" in report["metric_averages"]
    assert report["retrieval_loss"] == {
        "type": "contrastive",
        "num_negatives": 32,
        "hard_ratio": 0.5,
        "positive_excluded_from_negatives": True,
        "legacy_routing_ce_loss_weight": 0.0,
        "routing_foundation_frozen": True,
        "trainable_effect": "monitor_only_no_encoder_or_skill_table_gradient",
        "claim_scope": "diagnostic_not_retrieval_improvement_evidence",
    }
    assert "transition_skill_ce_loss" in report["metric_averages"]
    assert report["metric_averages"]["loss"] != report["last_batch_metrics"]["loss"]
    assert report["policy_head_type"] == "native_skill_head"
    assert report["legacy_universal_action_adapter_role"] == "fallback_only"
    assert "skill_head" in report["trainable_modules"]
    assert "trans_head" in report["trainable_modules"]
    assert "universal_action_adapter" not in report["trainable_modules"]
    assert "transition" in report["trainable_modules"]
    assert Path(report["checkpoint"]).exists()
    assert Path(report["latest_checkpoint"]).exists()
    assert Path(report["training_metrics_path"]).exists()
    assert Path(report["loss_curve_path"]).exists()
    assert Path(report["setup_status_path"]).exists()
    assert (tmp_path / "out" / "train_report.json").exists()
    setup_phases = [
        json.loads(line)["phase"]
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for phase in [
        "data_loaded",
        "rows_filtered",
        "model_moved_to_device",
        "routing_checkpoint_loaded",
        "embedding_cache_prepared",
        "training_started",
    ]:
        assert phase in setup_phases
    metric_lines = [
        json.loads(line)
        for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step"] for row in metric_lines] == [1, 2, 3]
    latest_payload = torch.load(report["latest_checkpoint"], map_location="cpu")
    assert latest_payload["step"] == 3

    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["stage"] == "clstr_full_base_component_complete"
    assert payload["transition_objective"] == report["transition_objective"]
    assert payload["transition_input_semantics"] == report["transition_input_semantics"]
    assert payload["belief_memory_semantics"] == report["belief_memory_semantics"]
    assert payload["policy_feature_semantics"] == report["policy_feature_semantics"]
    assert payload["transition_skill_ce"]["enabled"] is True
    assert payload["qdoc_adapter_used"] is False
    assert payload["frozen_routing_foundation"] is True
    assert payload["clstr_native_act_trained"] is True
    assert payload["not_full_clstr_act"] is False
    assert payload["skills_path"] == str(skills_path)
    assert payload["train_path"] == str(train_path)
    assert payload["loss_activation_counts"]["L_trans"] == 2
    assert payload["loss_activation_counts"]["L_trans_skill_ce"] == 1
    assert payload["loss_weights"] == report["loss_weights"]
    assert payload["policy_head_type"] == "native_skill_head"
    assert payload["legacy_universal_action_adapter_role"] == "fallback_only"
    assert payload["metric_averages"] == report["metric_averages"]
    assert payload["checkpoint_excludes_frozen_routing_foundation"] is True
    assert all(not key.startswith("encoder.") for key in payload["model_state_dict"])
    assert all(
        not key.startswith("skill_table.")
        or key in {"skill_table.logit_scale_belief", "skill_table.skill_bias_belief"}
        for key in payload["model_state_dict"]
    )


def test_train_clstr_full_base_resume_checkpoint_continues_step_optimizer_and_counterfactual_warmup(
    tmp_path,
    monkeypatch,
):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/current", "name": "current", "description": "current"},
            {"skill_id": "skill/next", "name": "next", "description": "next"},
        ],
    )
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "train-1:0",
                "state_text": "current state",
                "action_text": "call current",
                "next_observation_text": "next state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {"L_trans_skill_ce": True, "L_trans": True, "STOP": True},
                "provenance": {"split": "train"},
            }
        ],
    )

    first = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=2),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "first",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        loss_weights={"L_trans_skill_ce": 1.0, "L_trans": 0.1, "STOP": 0.1, "routing": 0.0, "belief": 0.0, "L_policy": 0.0},
        allow_full_pool_stage2_debug=True,
    )
    first_payload = torch.load(first["latest_checkpoint"], map_location="cpu")
    assert first_payload["step"] == 1
    assert "optimizer_state_dict" in first_payload

    warmup_calls: list[tuple[int, int, float]] = []
    original_warmup_scale = full_base_train_module._counterfactual_warmup_scale

    def recording_warmup_scale(global_step, max_steps, fraction):
        warmup_calls.append((int(global_step), int(max_steps), float(fraction)))
        return original_warmup_scale(global_step, max_steps, fraction)

    monkeypatch.setattr(full_base_train_module, "_counterfactual_warmup_scale", recording_warmup_scale)

    second = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=2),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "second",
        max_steps=3,
        target_total_steps=3,
        batch_size=1,
        learning_rate=1.0e-3,
        loss_weights={"L_trans_skill_ce": 1.0, "L_trans": 0.1, "STOP": 0.1, "routing": 0.0, "belief": 0.0, "L_policy": 0.0},
        allow_full_pool_stage2_debug=True,
        resume_checkpoint_path=first["latest_checkpoint"],
    )

    assert second["resume"]["enabled"] is True
    assert second["resume"]["start_step"] == 1
    assert second["resume"]["trained_steps_this_run"] == 2
    assert second["resume"]["final_step"] == 3
    assert second["resume"]["target_total_steps"] == 3
    assert second["resume"]["optimizer_loaded"] is True
    assert warmup_calls[-2:] == [(2, 3, 0.1), (3, 3, 0.1)]
    metric_lines = [
        json.loads(line)
        for line in Path(second["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step"] for row in metric_lines] == [2, 3]
    latest_payload = torch.load(second["latest_checkpoint"], map_location="cpu")
    assert latest_payload["step"] == 3
    assert latest_payload["resume"]["optimizer_loaded"] is True


def test_full_base_report_distinguishes_offline_rollout_rows_from_online_rollout_training(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/current", "name": "current", "description": "current"},
            {"skill_id": "skill/next", "name": "next", "description": "next"},
        ],
    )
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "dagger-row:0",
                "state_text": "goal: put mug\nobservation: You see a mug.",
                "history_text": "",
                "action_text": "take mug 1 from table 1",
                "admissible_actions": ["take mug 1 from table 1", "look"],
                "expert_action": "take mug 1 from table 1",
                "next_observation_text": "You pick up the mug.",
                "done": False,
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "loss_mask": {
                    "L_policy": True,
                    "L_trans": True,
                    "L_trans_skill_ce": True,
                    "belief": True,
                    "STOP": True,
                    "routing": True,
                },
                "source_quality": "dagger_expert_corrected_rollout",
                "on_policy_rollout": True,
                "provenance": {"split": "train", "source_dataset": "qwen3_expert_corrected_rollout"},
            }
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=2),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={"qdoc_adapter_used": False},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        loss_weights={
            "L_policy": 0.2,
            "L_trans": 0.3,
            "L_trans_skill_ce": 1.0,
            "belief": 0.1,
            "STOP": 0.1,
            "routing": 0.1,
        },
        embedding_cache_mode="never",
        allow_full_pool_stage2_debug=True,
    )

    assert report["training_regime"] == "offline_replay_supervised_pretraining"
    assert report["on_policy_rollout_used"] is False
    assert report["offline_rollout_source_rows_used"] == 1
    assert report["offline_rollout_source_counts"] == {"dagger_expert_corrected_rollout": 1}
    assert report["offline_rollout_source_note"].startswith("Rows may originate from logged or corrected rollouts")

    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["on_policy_rollout_used"] is False
    assert payload["offline_rollout_source_rows_used"] == 1


def test_full_base_training_filters_to_allowed_benchmarks_and_records_audit(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    rows = []
    for benchmark in ("traject_bench", "alfworld", "toolbench_g3"):
        rows.append(
            {
                "benchmark": benchmark,
                "task_id": f"{benchmark}-1",
                "state_text": f"{benchmark} state",
                "action_text": "open expert",
                "admissible_actions": ["look", "open expert"],
                "expert_action": "open expert",
                "skill_id": "skill/a",
                "next_skill_id": "skill/a",
                "loss_mask": {
                    "L_policy": True,
                    "routing": False,
                    "L_trans": False,
                    "L_trans_skill_ce": False,
                    "belief": False,
                    "STOP": False,
                },
                "source_quality": "unit_test",
                "provenance": {"split": "train"},
            }
        )
    _write_jsonl(train_path, rows)

    report = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=1),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=2,
        learning_rate=1.0e-3,
        seed=19,
        loss_weights={"L_policy": 1.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        allow_full_pool_stage2_debug=True,
    )

    assert report["sample_count"] == 2
    assert report["benchmark_counts"] == {"toolbench_g3": 1, "traject_bench": 1}
    assert report["benchmark_filter"] == {
        "enabled": True,
        "allowed_benchmarks": ["toolbench_g3", "traject_bench"],
        "source_rows": 3,
        "retained_rows": 2,
        "skipped_rows": 1,
        "skipped_by_benchmark": {"alfworld": 1},
    }
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["benchmark_filter"] == report["benchmark_filter"]


def test_full_base_benchmark_caps_downsample_auxiliary_domains():
    rows = (
        [{"benchmark": "toolbench_g3", "task_id": f"tb-{idx}"} for idx in range(2)]
        + [{"benchmark": "traject_bench", "task_id": f"tr-{idx}"} for idx in range(3)]
        + [{"benchmark": "alfworld", "task_id": f"alf-{idx}"} for idx in range(3)]
        + [{"benchmark": "scienceworld", "task_id": f"sci-{idx}"} for idx in range(2)]
    )

    retained, report = _cap_rows_by_benchmark(
        rows,
        {
            "toolbench_g3": -1,
            "traject_bench": -1,
            "alfworld": 1,
            "scienceworld": 1,
        },
    )

    assert [row["task_id"] for row in retained] == [
        "tb-0",
        "tb-1",
        "tr-0",
        "tr-1",
        "tr-2",
        "alf-0",
        "sci-0",
    ]
    assert report == {
        "enabled": True,
        "benchmark_caps": {
            "alfworld": 1,
            "scienceworld": 1,
            "toolbench_g3": -1,
            "traject_bench": -1,
        },
        "source_rows": 10,
        "retained_rows": 7,
        "skipped_rows": 3,
        "retained_by_benchmark": {
            "alfworld": 1,
            "scienceworld": 1,
            "toolbench_g3": 2,
            "traject_bench": 3,
        },
        "skipped_by_benchmark": {
            "alfworld": 2,
            "scienceworld": 1,
        },
    }


def test_full_base_training_excludes_public_unlabeled_rows_by_default(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "train-safe",
                "state_text": "train state",
                "action_text": "open expert",
                "admissible_actions": ["open expert"],
                "expert_action": "open expert",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": True, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "official_replay",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "task_id": "public-unlabeled",
                "state_text": "public eval state",
                "action_text": "call public",
                "admissible_actions": ["call public"],
                "expert_action": "call public",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": True, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "traject_public_data",
                "provenance": {"split": "public_train_or_eval_unlabeled"},
            },
        ],
    )

    report = train_clstr_full_base_with_model(
        model=_NativeFullBaseTrainModel(dim=8, skill_count=1),
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=23,
        loss_weights={"L_policy": 1.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        allow_full_pool_stage2_debug=True,
    )

    assert report["sample_count"] == 1
    assert report["benchmark_counts"] == {"alfworld": 1}
    assert report["train_split_filter"]["retained_rows"] == 1
    assert report["train_split_filter"]["skipped_by_split"] == {"public_train_or_eval_unlabeled": 1}
    assert report["benchmark_filter"]["enabled"] is False
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["train_split_filter"] == report["train_split_filter"]


def test_qwen_external_checkpoint_excludes_frozen_backbone_state(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "qwen-1",
                "state_text": "state",
                "action_text": "open fridge",
                "admissible_actions": ["open fridge", "look"],
                "expert_action": "open fridge",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": True, "routing": True},
                "source_quality": "official_replay",
                "provenance": {"split": "train"},
            }
        ],
    )
    model = _NativeFullBaseTrainModel(dim=8, skill_count=1)
    model.qwen_external_metadata = {
        "qwen_external_encoder": True,
        "qwen_model_name_or_path": "models/Qwen3-8B",
        "qwen_frozen": True,
        "qwen_direct_generator": False,
    }

    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "models/Qwen3-8B", "qwen_external_encoder": True},
        routing_report={"qdoc_adapter_used": False, "qwen_external_encoder": True},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=7,
        loss_weights={"L_policy": 1.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        allow_full_pool_stage2_debug=True,
    )

    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["qwen_external_encoder"] is True
    assert payload["checkpoint_excludes_frozen_qwen_backbone"] is True
    assert all(not key.startswith("encoder.backbone.") for key in payload["model_state_dict"])
    assert all(not key.startswith("skill_table.encoder_fn.backbone.") for key in payload["model_state_dict"])


def test_qwen_external_checkpoint_filter_removes_duplicate_skill_table_encoder_refs():
    from clstr.full_base_train import _checkpoint_state_dict

    class _DuplicateQwenRefs:
        def state_dict(self):
            return {
                "encoder.backbone.layers.0.weight": torch.ones(1),
                "encoder.proj.weight": torch.ones(1),
                "skill_table.encoder_fn.backbone.layers.0.weight": torch.ones(1),
                "skill_table.encoder_fn.proj.weight": torch.ones(1),
                "skill_table.W.weight": torch.ones(1),
                "skill_table.E": torch.ones(1),
                "skill_head.weight": torch.ones(1),
                "trans_head.weight": torch.ones(1),
            }

    state, report = _checkpoint_state_dict(_DuplicateQwenRefs(), exclude_frozen_qwen_backbone=True)

    assert sorted(state) == ["skill_head.weight", "trans_head.weight"]
    assert "skill_table.encoder_fn.backbone." in report["excluded_state_key_prefixes"]


def test_full_base_checkpoint_filter_removes_frozen_routing_foundation_refs():
    from clstr.full_base_train import _checkpoint_state_dict

    class _FrozenFoundationRefs:
        def state_dict(self):
            return {
                "encoder.backbone.layers.0.weight": torch.ones(1),
                "encoder.proj.weight": torch.ones(1),
                "skill_table.encoder_fn.backbone.layers.0.weight": torch.ones(1),
                "skill_table.W.weight": torch.ones(1),
                "skill_table.E": torch.ones(1),
                "skill_table.logit_scale_retr": torch.ones(1),
                "skill_table.skill_bias_retr": torch.ones(2),
                "skill_table.logit_scale_belief": torch.ones(1) * 2.0,
                "skill_table.skill_bias_belief": torch.ones(2) * 3.0,
                "skill_head.net.0.weight": torch.ones(1),
                "transition.net.weight": torch.ones(1),
                "q_success_head.net.0.weight": torch.ones(1),
            }

    state, report = _checkpoint_state_dict(_FrozenFoundationRefs(), exclude_frozen_routing_foundation=True)

    assert sorted(state) == [
        "q_success_head.net.0.weight",
        "skill_head.net.0.weight",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
        "transition.net.weight",
    ]
    assert report["checkpoint_excludes_frozen_routing_foundation"] is True
    assert "encoder." in report["excluded_state_key_prefixes"]
    assert report["preserved_routing_calibration_keys"] == [
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    ]


class _WarmstartSkillTable(torch.nn.Module):
    def __init__(self, dim: int, skill_count: int):
        super().__init__()
        self.W = torch.nn.Linear(dim, dim, bias=False)
        self.E = torch.nn.Parameter(torch.zeros(skill_count, dim))
        self.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
        self.skill_bias_retr = torch.nn.Parameter(torch.zeros(skill_count))

    def logits(self, h):
        return self.W(h) @ self.E.t() + self.skill_bias_retr.unsqueeze(0)


def test_full_base_training_loads_stage1_routing_checkpoint_after_skill_table_rebuild(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "stage2-1",
                "state_text": "state",
                "action_text": "look",
                "admissible_actions": ["look"],
                "expert_action": "look",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": False, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "official_replay",
                "provenance": {"split": "train"},
            }
        ],
    )
    model = _NativeFullBaseTrainModel(dim=8, skill_count=1)
    model.skill_table = _WarmstartSkillTable(dim=8, skill_count=1)
    with torch.no_grad():
        model.skill_table.E.zero_()
        model.skill_table.W.weight.zero_()
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "skill_table.E": torch.full((1, 8), 0.75),
                "skill_table.W.weight": torch.eye(8) * 2.0,
                "skill_table.logit_scale_retr": torch.tensor([1.25]),
                "skill_table.skill_bias_retr": torch.tensor([0.5]),
                "encoder.backbone.layers.0.weight": torch.ones(3, 3),
                "skill_head.linear.weight": torch.ones(1, 24),
            },
        },
        checkpoint_path,
    )

    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=13,
        loss_weights={"L_policy": 0.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        routing_checkpoint_path=checkpoint_path,
        allow_full_pool_stage2_debug=True,
    )

    assert torch.allclose(model.skill_table.E, torch.full((1, 8), 0.75))
    assert torch.allclose(model.skill_table.W.weight, torch.eye(8) * 2.0)
    assert torch.allclose(model.skill_table.logit_scale_retr, torch.tensor([1.25]))
    assert torch.allclose(model.skill_table.skill_bias_retr, torch.tensor([0.5]))
    assert report["routing_checkpoint"]["loaded"] is True
    assert report["routing_checkpoint"]["path"] == str(checkpoint_path)
    assert set(report["routing_checkpoint"]["loaded_keys"]) >= {
        "skill_table.E",
        "skill_table.W.weight",
        "skill_table.logit_scale_retr",
        "skill_table.skill_bias_retr",
    }
    assert "encoder.backbone.layers.0.weight" in report["routing_checkpoint"]["skipped_keys"]
    assert "skill_head.linear.weight" in report["routing_checkpoint"]["skipped_keys"]
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["routing_checkpoint"]["loaded"] is True


def test_full_base_training_skips_deferred_skill_table_rebuild_when_stage1_checkpoint_has_embeddings(tmp_path):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    checkpoint_path = tmp_path / "stage1.pt"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": "stage2-skip-rebuild",
                "state_text": "state",
                "action_text": "look",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": False, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "official_replay",
                "provenance": {"split": "train"},
            }
        ],
    )
    model = _NativeFullBaseTrainModel(dim=8, skill_count=1)
    model.skill_table = _WarmstartSkillTable(dim=8, skill_count=1)
    model.config = type("Config", (), {"defer_skill_table_init": True, "skill_text_format": "clstr"})()
    calls = {"rebuild": 0}

    def rebuild_skill_table():
        calls["rebuild"] += 1
        raise AssertionError("Stage 2 should not rebuild skill embeddings when Stage 1 checkpoint contains skill_table.E")

    model.rebuild_skill_table = rebuild_skill_table
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {
                "skill_table.E": torch.full((1, 8), 0.5),
                "skill_table.W.weight": torch.eye(8),
            },
        },
        checkpoint_path,
    )

    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "tiny-test"},
        routing_report={},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=13,
        loss_weights={"L_policy": 0.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        routing_checkpoint_path=checkpoint_path,
        allow_full_pool_stage2_debug=True,
    )

    assert calls["rebuild"] == 0
    assert report["routing_checkpoint"]["loaded"] is True
    assert report["routing_checkpoint"]["skill_table_rebuild_skipped"] is True


def test_qwen_external_training_uses_small_embedding_cache_batches(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "alfworld",
                "task_id": "qwen-cache",
                "state_text": "state",
                "action_text": "look",
                "admissible_actions": ["look"],
                "expert_action": "look",
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": False, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "official_replay",
                "provenance": {"split": "train"},
            }
        ],
    )
    model = _NativeFullBaseTrainModel(dim=8, skill_count=1)
    model.qwen_external_metadata = {
        "qwen_external_encoder": True,
        "qwen_model_name_or_path": "models/Qwen3-8B",
        "qwen_frozen": True,
    }
    captured: dict[str, int] = {}

    def fake_attach_full_base(model, rows, encode_batch_size=256):
        captured["full_base"] = encode_batch_size
        return {"used": True, "encode_batch_size": encode_batch_size}

    def fake_attach_policy(model, rows, encode_batch_size=256):
        captured["policy"] = encode_batch_size
        return {"used": False, "encode_batch_size": encode_batch_size}

    monkeypatch.setattr("clstr.full_base_train._attach_full_base_embedding_cache", fake_attach_full_base)
    monkeypatch.setattr("clstr.full_base_train._attach_policy_embedding_cache", fake_attach_policy)

    train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "models/Qwen3-8B", "qwen_external_encoder": True},
        routing_report={"qdoc_adapter_used": False, "qwen_external_encoder": True},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=11,
        loss_weights={"L_policy": 0.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        allow_full_pool_stage2_debug=True,
    )

    assert captured == {"full_base": 8}


def test_qwen_external_training_skips_full_dataset_embedding_cache_for_large_stage2_data(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(skills_path, [{"skill_id": "skill/a", "name": "a", "description": "a"}])
    _write_jsonl(
        train_path,
        [
            {
                "benchmark": "traject_bench",
                "task_id": f"large-cache-{idx}",
                "state_text": f"state {idx}",
                "action_text": "look",
                "admissible_actions": ["look"],
                "expert_action": "look",
                "next_observation_text": "next",
                "done": False,
                "skill_id": "skill/a",
                "loss_mask": {"L_policy": False, "routing": False, "L_trans": False, "L_trans_skill_ce": False, "belief": False, "STOP": False},
                "source_quality": "traject_public_data",
                "provenance": {"split": "train"},
            }
            for idx in range(3)
        ],
    )
    model = _NativeFullBaseTrainModel(dim=8, skill_count=1)
    model.qwen_external_metadata = {
        "qwen_external_encoder": True,
        "qwen_model_name_or_path": "models/Qwen3-8B",
        "qwen_frozen": True,
    }

    def fail_attach_full_base(*args, **kwargs):
        raise AssertionError("large Qwen Stage 2 data should not pre-encode every replay embedding")

    def fail_attach_policy(*args, **kwargs):
        raise AssertionError("large Qwen Stage 2 data should not pre-encode every policy candidate embedding")

    monkeypatch.setattr("clstr.full_base_train._attach_full_base_embedding_cache", fail_attach_full_base)
    monkeypatch.setattr("clstr.full_base_train._attach_policy_embedding_cache", fail_attach_policy)

    report = train_clstr_full_base_with_model(
        model=model,
        model_config={"d": 8, "base_model_name": "models/Qwen3-8B", "qwen_external_encoder": True},
        routing_report={"qdoc_adapter_used": False, "qwen_external_encoder": True},
        train_path=train_path,
        skills_path=skills_path,
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        learning_rate=1.0e-3,
        seed=11,
        loss_weights={"L_policy": 0.0, "routing": 0.0, "L_trans": 0.0, "L_trans_skill_ce": 0.0, "belief": 0.0, "STOP": 0.0},
        embedding_cache_mode="auto",
        embedding_cache_max_rows=2,
        allow_full_pool_stage2_debug=True,
    )

    assert report["full_base_embedding_cache"]["used"] is False
    assert report["policy_embedding_cache"]["used"] is False
    assert report["full_base_embedding_cache"]["reason"] == "qwen_external_large_dataset_auto_skip"
    assert report["embedding_cache_policy"]["mode"] == "auto"
    assert report["embedding_cache_policy"]["cache_enabled"] is False
