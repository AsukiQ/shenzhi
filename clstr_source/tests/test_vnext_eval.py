from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from clstr.vnext_eval import (
    _aligned_actual_result,
    _batch_factual_latent_trace,
    _benchmark_event_contract,
    _candidate_multi_positive_rank,
    _candidate_rank,
    _candidate_recall_query,
    _corpus_source_contract,
    _evaluation_restore_contract,
    _equal_reciprocal_rank_fusion_logits,
    _exact_id_occurrence_count,
    _executed_action_text,
    _foundation_preservation_contract,
    _full_pool_rank,
    _materialize_causal_eval_timeline,
    _materialize_closed_set_foundation_records,
    _materialize_memories,
    _metrics,
    _multi_positive_rank,
    _interface_expert_decision,
    _load_unified_router_contract,
    _unified_router_release_wrapper,
    _prepare_eval_rows,
    _partition_benchmark_skill_append,
    _preserved_foundation_candidate_logits,
    _preflight_verified_toolbench_results,
    _score_alternative,
    run_vnext_benchmark_eval,
)
from clstr.vnext_unified_router import (
    UNIFIED_EXPERT_NAMES,
    UNIFIED_ROUTER_FEATURE_NAMES,
    UNIFIED_ROUTER_SCHEMA,
    UNIFIED_ROUTING_MODE_SOFT,
    UNIFIED_ROUTING_MODE_SPARSE,
    UnifiedThreeExpertRouter,
)
from clstr.matched_multibench_data import canonical_digest
from scripts.run_clstr_vnext_eval import _load_corpus


def _timeline_row(
    trajectory_id: str,
    step: int,
    *,
    timing: str,
    skill_id: str,
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "step_index": step,
        "_vnext_source_index": step,
        "_vnext_state_text": f"state-{step}",
        "_vnext_event_timing": timing,
        "_vnext_executed_skill_id": skill_id,
        "_vnext_positive_skill_ids": [skill_id],
        "_vnext_executed_action_text": f"call {skill_id}",
        "_vnext_actual_result_text": f"result {skill_id}",
    }


def test_vnext_eval_timeline_uses_causal_state_before_each_event() -> None:
    after_rows = [
        _timeline_row("after", 0, timing="after_decision", skill_id="a"),
        _timeline_row("after", 1, timing="after_decision", skill_id="b"),
    ]
    state_texts, report = _materialize_causal_eval_timeline(after_rows)
    assert after_rows[0]["_vnext_causal_state_text"] == "state-0"
    assert after_rows[0]["_vnext_event"]["state_text"] == "state-0"
    assert "event[0].skill_id: a" in after_rows[1]["_vnext_causal_state_text"]
    assert (
        after_rows[1]["_vnext_event"]["state_text"]
        == after_rows[1]["_vnext_causal_state_text"]
    )
    assert after_rows[1]["_vnext_causal_state_text"] in state_texts
    assert report["protocol"] == "causal_state_before_executed_event_v2"

    before_row = _timeline_row(
        "before",
        0,
        timing="before_decision",
        skill_id="executed-a",
    )
    _materialize_causal_eval_timeline([before_row])
    assert before_row["_vnext_event"]["state_text"] == "state-0"
    assert "event[0].skill_id: executed-a" in before_row[
        "_vnext_causal_state_text"
    ]


def test_vnext_candidate_probe_separates_current_and_causal_queries() -> None:
    current = torch.tensor([[3.0, 0.0]])
    causal = torch.tensor([[0.0, 4.0]])
    memory = torch.zeros_like(current)
    current_query = _candidate_recall_query(
        SimpleNamespace(),
        current,
        causal,
        memory,
        mode="raw_current",
    )
    causal_query = _candidate_recall_query(
        SimpleNamespace(),
        current,
        causal,
        memory,
        mode="raw_causal",
    )
    assert torch.equal(current_query, torch.tensor([[1.0, 0.0]]))
    assert torch.equal(causal_query, torch.tensor([[0.0, 1.0]]))


def test_support_aware_anchor_rejects_a_learned_router_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="mutually exclusive",
    ):
        run_vnext_benchmark_eval(
            benchmark="tau2",
            corpus=SimpleNamespace(source_rows=[], skills=[]),
            checkpoint_path=tmp_path / "stage2.pt",
            training_skills_path=tmp_path / "skills.jsonl",
            output_dir=tmp_path / "out",
            unified_router_path=tmp_path / "router.pt",
            support_aware_anchor=True,
        )


def test_vnext_closed_set_foundation_records_never_execute_recurrence() -> None:
    rows = [
        _timeline_row("closed", 0, timing="after_decision", skill_id="a"),
        _timeline_row("closed", 1, timing="after_decision", skill_id="b"),
    ]
    _materialize_causal_eval_timeline(
        rows,
        include_counterfactual_diagnostics=False,
    )
    records, report = _materialize_closed_set_foundation_records(
        SimpleNamespace(vnext=SimpleNamespace(d=3)),
        rows,
    )
    assert [record["replay_prefix_length"] for record in records] == [0, 1]
    assert all(record["uses_recurrent_m_t"] is False for record in records)
    assert all(record["mismatch_memory"] is None for record in records)
    assert report["source_replay_prefix_row_count"] == 1
    assert report["uses_recurrent_m_t_count"] == 0
    assert report["closed_set_recurrent_path_executed"] is False


def test_vnext_closed_set_skips_missing_recurrent_trace_for_foundation_scoring() -> None:
    model = SimpleNamespace(
        vnext=SimpleNamespace(
            synchronization_enabled=True,
            synchronization_trace_length=3,
        )
    )
    records = [{"factual_latent_trace": None}]
    assert _batch_factual_latent_trace(
        model,
        records,
        foundation_selected=True,
    ) is None
    with pytest.raises(RuntimeError, match="lacks its factual trace"):
        _batch_factual_latent_trace(
            model,
            records,
            foundation_selected=False,
        )


def test_vnext_closed_set_equal_rrf_preserves_tiny_rows_and_fuses_moderate() -> None:
    static = torch.tensor([[4.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
    semantic = torch.tensor([[1.0, 3.0, 4.0], [3.0, 1.0, 2.0]])
    valid = torch.ones_like(static, dtype=torch.bool)
    fused = _equal_reciprocal_rank_fusion_logits(
        static,
        semantic,
        valid,
        torch.tensor([True, False]),
    )
    assert fused[0].argsort(descending=True, stable=True).tolist() == [0, 2, 1]
    assert torch.equal(fused[1], static[1])


def test_vnext_preserved_foundation_scorer_uses_one_shared_rrf_contract() -> None:
    class FakeVNext:
        @staticmethod
        def static_query_components(states, _belief):
            static = torch.tensor(
                [[0.0, 1.0], [0.0, 1.0]],
                dtype=states.dtype,
            )
            return static, torch.zeros_like(static), static

    class FakeModel:
        vnext = FakeVNext()

        @staticmethod
        def vnext_initial_belief(states, legal, *, top_k):
            assert states.shape == (2, 2)
            assert legal.shape == (2, 9)
            assert top_k == 9
            return torch.zeros_like(states)

        @staticmethod
        def vnext_candidate_logits(query, candidate_ids, valid_mask, *, head):
            assert head == "route"
            embeddings = torch.zeros((9, 2), dtype=query.dtype)
            embeddings[0, 1] = 1.0
            embeddings[8, 0] = 1.0
            selected = embeddings.index_select(0, candidate_ids.reshape(-1)).view(
                query.size(0),
                candidate_ids.size(1),
                2,
            )
            logits = torch.einsum("bd,bcd->bc", query, selected)
            return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

    candidate_ids = torch.arange(9).unsqueeze(0).expand(2, -1)
    candidate_valid = torch.tensor(
        [[True] * 9, [True] * 8 + [False]],
        dtype=torch.bool,
    )
    logits, fusion_applied = _preserved_foundation_candidate_logits(
        FakeModel(),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        candidate_valid.clone(),
        candidate_ids,
        candidate_valid,
        belief_top_k=9,
    )
    assert fusion_applied.tolist() == [True, False]
    assert logits[0].argsort(descending=True, stable=True)[:2].tolist() == [0, 8]
    assert int(logits[1].argmax().item()) == 0
    assert logits[1, 8].item() == torch.finfo(torch.float32).min


def test_vnext_static_only_support_fuses_moderate_complete_pool() -> None:
    if not torch.cuda.is_available():
        return

    class FakeCache:
        def batch(self, texts, *, device):
            return torch.tensor([[1.0, 0.0] for _ in texts], device=device)

    class FakeVNext:
        d = 2

        @staticmethod
        def static_query_components(h_t, _static_memory):
            static = torch.tensor([[0.0, 1.0]], device=h_t.device).expand_as(h_t)
            return static, torch.zeros_like(h_t), static

    class FakeModel:
        vnext = FakeVNext()

        @staticmethod
        def vnext_initial_belief(h_t, _legal, *, top_k):
            assert top_k == 9
            return torch.zeros_like(h_t)

        @staticmethod
        def vnext_full_pool_logits(query, *, head):
            assert head == "recall"
            return torch.zeros((query.size(0), 9), device=query.device)

        @staticmethod
        def vnext_candidate_logits(query, candidate_ids, valid_mask, *, head):
            assert head == "route"
            embeddings = torch.zeros(
                (9, 2),
                device=query.device,
                dtype=query.dtype,
            )
            embeddings[0, 1] = 1.0
            embeddings[8, 0] = 1.0
            selected = embeddings.index_select(0, candidate_ids.reshape(-1)).view(
                query.size(0),
                candidate_ids.size(1),
                2,
            )
            logits = torch.einsum("bd,bcd->bc", query, selected)
            return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

    scored = _score_alternative(
        FakeModel(),
        [
            {
                "_vnext_state_text": "current",
                "_vnext_causal_state_text": "causal",
                "_vnext_legal_skill_ids": [f"s{index}" for index in range(9)],
                "_vnext_target_skill_id": "s8",
                "_vnext_positive_skill_ids": ["s8"],
            }
        ],
        torch.zeros((1, 2)),
        torch.zeros(1, dtype=torch.bool),
        history_depth=torch.zeros(1),
        observation_correction_count=torch.zeros(1),
        state_cache=FakeCache(),
        skill_id_to_idx={f"s{index}": index for index in range(9)},
        device=torch.device("cuda"),
        belief_top_k=9,
        coarse_k=500,
        compressed_m=64,
        use_static_query=True,
        candidate_query_mode="learned_causal",
        candidate_compression_mode="learned",
        final_k=100,
        static_only_execution=True,
    )
    assert scored["route_ranks"] == [2]
    assert scored["foundation_route_ranks"] == [2]
    assert scored["closed_set_semantic_fusion_applied"] == [True]
    assert scored["masked_equals_static"] is True


def test_vnext_direct_support_discards_recall_rank_order() -> None:
    from clstr.vnext_candidates import direct_natural_support

    support = direct_natural_support(
        torch.tensor([[7, 2, 5, 0]]),
        torch.tensor([[True, True, True, False]]),
        skill_count=8,
    )
    assert support.candidate_ids.tolist() == [[2, 5, 7, 7]]
    assert support.valid_mask.tolist() == [[True, True, True, False]]


def test_vnext_eval_restore_contract_is_strict_for_stage2() -> None:
    report = _evaluation_restore_contract(
        checkpoint_stage="clstr_vnext_stage2",
        checkpoint_state_keys=["skill_table.E"],
        model_state_keys=["skill_table.E", "vnext.memory_recall_query.weight"],
        loaded_keys=["skill_table.E"],
    )
    assert report["required_scope"] == "complete_stage2_model"
    assert report["missing_required_keys"] == [
        "vnext.memory_recall_query.weight"
    ]


def test_vnext_eval_restore_contract_allows_stage0_to_omit_stage2_heads() -> None:
    report = _evaluation_restore_contract(
        checkpoint_stage="clstr_vnext_stage0",
        checkpoint_state_keys=["skill_table.E", "vnext.static_query.weight"],
        model_state_keys=[
            "skill_table.E",
            "vnext.static_query.weight",
            "vnext.memory_recall_query.weight",
        ],
        loaded_keys=["skill_table.E", "vnext.static_query.weight"],
    )
    assert report["required_scope"] == "checkpoint_declared_static_diagnostic"
    assert report["missing_required_keys"] == []
    assert report["unrestored_model_canonical_keys"] == [
        "vnext.memory_recall_query.weight"
    ]


def test_vnext_eval_restore_contract_rejects_unloaded_stage0_tensor() -> None:
    report = _evaluation_restore_contract(
        checkpoint_stage="clstr_vnext_stage0",
        checkpoint_state_keys=["skill_table.E", "vnext.static_query.weight"],
        model_state_keys=["skill_table.E", "vnext.static_query.weight"],
        loaded_keys=["skill_table.E"],
    )
    assert report["missing_required_keys"] == ["vnext.static_query.weight"]


def test_vnext_interface_dispatch_uses_foundation_only_for_enumerated_closed_set() -> None:
    rows = [
        {
            "next_skill_id": "a",
            "candidate_next_skill_ids": ["a", "b"],
        },
        {
            "next_skill_id": "c",
            "candidate_next_skill_ids": ["c", "d", "e"],
        },
    ]
    closed = _interface_expert_decision(rows, coarse_k=500)
    assert closed["selected_expert"] == "preserved_closed_set_foundation"
    assert closed["maximum_row_pool_size"] == 3
    assert closed["uses_benchmark_identity"] is False

    open_pool = _interface_expert_decision(
        rows,
        coarse_k=500,
        declared_open_pool_size=67_317,
        declared_pool_protocol="checkpoint_plus_benchmark_full_pool",
    )
    assert open_pool["selected_expert"] == "adapted_static_plus_recurrent_dynamic"
    assert open_pool["reason"] == "declared_open_pool_exceeds_natural_candidate_budget"

    incomplete = _interface_expert_decision(
        [{"next_skill_id": "a"}],
        coarse_k=500,
    )
    assert incomplete["selected_expert"] == "adapted_static_plus_recurrent_dynamic"


def test_vnext_foundation_preservation_contract_binds_lineage(tmp_path: Path) -> None:
    skills = tmp_path / "skills.jsonl"
    skills.write_text('{"skill_id":"a"}\n{"skill_id":"b"}\n', encoding="utf-8")
    skills_sha = hashlib.sha256(skills.read_bytes()).hexdigest()
    config = {
        "base_model_name": "frozen-qwen",
        "frozen_backbone_snapshot_digest": "backbone-digest",
        "skill_text_format": "skillret_official",
        "state_query_prompt_version": "clstr_causal_state_v1",
        "freeze_backbone": True,
    }
    common = {
        "config": config,
        "run_contract": {"inputs": {"skills_sha256": skills_sha}},
    }
    stage2 = tmp_path / "stage2.pt"
    foundation = tmp_path / "stage0-step0.pt"
    torch.save(
        {
            **common,
            "stage": "clstr_vnext_stage2",
            "step": 1000,
            "model_state_dict": {
                "skill_table.E": torch.ones(2, 3),
                "vnext.memory_recall_query.weight": torch.ones(1),
                "vnext.route_expert_mixture.weight": torch.ones(1),
                "vnext.unified_route_query.weight": torch.ones(1),
            },
        },
        stage2,
    )
    torch.save(
        {
            **common,
            "stage": "clstr_vnext_stage0",
            "step": 0,
            "model_state_dict": {"skill_table.E": torch.ones(2, 3)},
        },
        foundation,
    )
    report = _foundation_preservation_contract(
        stage2_checkpoint_path=stage2,
        foundation_checkpoint_path=foundation,
        training_skills_path=skills,
    )
    assert report["protocol"] == "lineage_bound_stage0_step0_plus_release_stage2_v1"
    assert report["checkpoint_skill_count"] == 2
    assert report["foundation_checkpoint_step"] == 0


def test_vnext_unified_router_contract_binds_both_experts_without_benchmark_features(
    tmp_path: Path,
) -> None:
    stage2 = tmp_path / "stage2.pt"
    foundation = tmp_path / "foundation.pt"
    skills = tmp_path / "skills.jsonl"
    stage2.write_bytes(b"stage2")
    foundation.write_bytes(b"foundation")
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")
    router = UnifiedThreeExpertRouter()
    artifact = tmp_path / "router.pt"
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    release_selection = tmp_path / "stage2_release.json"
    original_selection = tmp_path / "stage2_selection.json"
    release_selection.write_text("{}\n", encoding="utf-8")
    original_selection.write_text("{}\n", encoding="utf-8")
    release_contract = {
        "schema_version": "clstr_vnext_stage2_open_pool_selection_v1",
        "selection_path": str(release_selection.resolve()),
        "selection_sha256": hashlib.sha256(release_selection.read_bytes()).hexdigest(),
        "source_commit": "approved-stage2-release-source",
        "selection_mode": "open_pool_full_pool",
        "selected_checkpoint_path": str(stage2.resolve()),
        "selected_checkpoint_sha256": hashlib.sha256(stage2.read_bytes()).hexdigest(),
        "original_stage2_selection_path": str(original_selection.resolve()),
        "original_stage2_selection_sha256": hashlib.sha256(
            original_selection.read_bytes()
        ).hexdigest(),
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
    }
    release_wrapper = _unified_router_release_wrapper(
        release_contract,
        router_source_commit=source_commit,
        foundation_checkpoint_path=foundation,
        foundation_checkpoint_sha256=hashlib.sha256(
            foundation.read_bytes()
        ).hexdigest(),
        training_skills_path=skills,
        training_skills_sha256=hashlib.sha256(skills.read_bytes()).hexdigest(),
    )
    torch.save(
        {
            "schema_version": UNIFIED_ROUTER_SCHEMA,
            "status": "ok",
            "blockers": [],
            "routing_mode": UNIFIED_ROUTING_MODE_SPARSE,
            "state_dict": router.state_dict(),
            "feature_names": list(UNIFIED_ROUTER_FEATURE_NAMES),
            "expert_names": list(UNIFIED_EXPERT_NAMES),
            "stage2_checkpoint_sha256": hashlib.sha256(stage2.read_bytes()).hexdigest(),
            "foundation_checkpoint_sha256": hashlib.sha256(
                foundation.read_bytes()
            ).hexdigest(),
            "training_skills_sha256": hashlib.sha256(skills.read_bytes()).hexdigest(),
            "selection": {
                "uses_benchmark_eval_rows": False,
                "uses_benchmark_identity_feature": False,
            },
            "training_contract": {
                "default_expert_prior": "foundation",
                "observation_grounding_feature": (
                    "observation_correction_fraction"
                ),
                "paired_calibration_protocol": (
                    "same_prefix_factual_and_action_only_result_suppressed_v1"
                ),
                "paired_views_share_trajectory_split": True,
                "foundation_expert_protocol": (
                    "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
                ),
                "expert_supervision_protocol": (
                    "class_balanced_sparse_minimum_regret_top1_v1"
                ),
                "expert_tie_break_order": list(UNIFIED_EXPERT_NAMES),
                "expert_utility": "reciprocal_rank_plus_0.5_top5",
                "oracle_expert_recall_min_rows": 20,
                "oracle_expert_recall_floor": 0.20,
                "calibration_no_regret_tolerances": {
                    "global_mrr": 0.005,
                    "global_recall_at_5": 0.01,
                    "family_mrr": 0.02,
                    "family_recall_at_5": 0.03,
                    "memory_evidence_mrr": 0.005,
                    "memory_evidence_recall_at_5": 0.01,
                },
            },
            "source_commit": source_commit,
            "stage2_release_wrapper": release_wrapper,
        },
        artifact,
    )
    loaded, report = _load_unified_router_contract(
        artifact,
        stage2_checkpoint_path=stage2,
        foundation_checkpoint_path=foundation,
        training_skills_path=skills,
        stage2_release_selection_contract=release_contract,
        device=torch.device("cpu"),
    )
    assert isinstance(loaded, UnifiedThreeExpertRouter)
    assert report["uses_benchmark_identity"] is False
    assert report["uses_source_identity"] is False
    assert report["routing_mode"] == UNIFIED_ROUTING_MODE_SPARSE
    assert report["stage2_release_wrapper"] == release_wrapper

    soft = torch.load(artifact, map_location="cpu")
    soft["routing_mode"] = UNIFIED_ROUTING_MODE_SOFT
    soft_artifact = tmp_path / "router-soft.pt"
    torch.save(soft, soft_artifact)
    with pytest.raises(ValueError, match="not sparse-deployment approved"):
        _load_unified_router_contract(
            soft_artifact,
            stage2_checkpoint_path=stage2,
            foundation_checkpoint_path=foundation,
            training_skills_path=skills,
            stage2_release_selection_contract=release_contract,
            device=torch.device("cpu"),
        )

    tampered = torch.load(artifact, map_location="cpu")
    tampered["stage2_release_wrapper"] = {
        **tampered["stage2_release_wrapper"],
        "stage2_release_selection_sha256": "tampered",
    }
    tampered_artifact = tmp_path / "router-tampered.pt"
    torch.save(tampered, tampered_artifact)
    with pytest.raises(ValueError, match="release wrapper differs"):
        _load_unified_router_contract(
            tampered_artifact,
            stage2_checkpoint_path=stage2,
            foundation_checkpoint_path=foundation,
            training_skills_path=skills,
            stage2_release_selection_contract=release_contract,
            device=torch.device("cpu"),
        )

    relaxed = torch.load(artifact, map_location="cpu")
    relaxed["training_contract"]["calibration_no_regret_tolerances"][
        "global_recall_at_5"
    ] = 0.02
    relaxed_artifact = tmp_path / "router-relaxed-tolerance.pt"
    torch.save(relaxed, relaxed_artifact)
    with pytest.raises(ValueError, match="calibration contract differs"):
        _load_unified_router_contract(
            relaxed_artifact,
            stage2_checkpoint_path=stage2,
            foundation_checkpoint_path=foundation,
            training_skills_path=skills,
            stage2_release_selection_contract=release_contract,
            device=torch.device("cpu"),
        )


def test_vnext_support_probe_is_exposed_by_cli_and_launcher() -> None:
    root = Path(__file__).parents[1]
    cli = root.joinpath("scripts/run_clstr_vnext_eval.py").read_text(
        encoding="utf-8"
    )
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_benchmark_eval.sh"
    ).read_text(encoding="utf-8")
    evaluator = root.joinpath("clstr/vnext_eval.py").read_text(encoding="utf-8")
    router = root.joinpath("clstr/vnext_unified_router.py").read_text(
        encoding="utf-8"
    )
    calibration = root.joinpath(
        "scripts/train_clstr_vnext_unified_router.py"
    ).read_text(encoding="utf-8")
    calibration_launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_unified_router_train.sh"
    ).read_text(encoding="utf-8")
    for option in (
        "--candidate_query_mode",
        "--candidate_compression_mode",
        "--final_k",
        "--toolbench_pool_scope",
        "--toolbench_native_skills_path",
    ):
        assert option in cli
        assert option in launcher
    assert "--allow_stage0_static_diagnostic" in cli
    assert "--allow_stage0_static_diagnostic" in launcher
    assert "--allow_static_reranker_diagnostic" in cli
    assert "--allow_static_reranker_diagnostic" in launcher
    assert "--closed_set_foundation_checkpoint_path" in cli
    assert "CLOSED_SET_FOUNDATION_CHECKPOINT_PATH" in launcher
    assert "--stage2_release_selection_path" in cli
    assert "STAGE2_RELEASE_SELECTION_PATH" in launcher
    assert "clstr_vnext_stage2_open_pool_selection_v1" in launcher
    assert "--unified_router_path" in cli
    assert "UNIFIED_ROUTER_PATH" in launcher
    assert "--support_aware_anchor" in cli
    assert "SUPPORT_AWARE_ANCHOR" in launcher
    assert "stage2_release_wrapper" in launcher
    assert "--stage2_release_selection_path" in calibration
    assert "STAGE2_RELEASE_SELECTION_PATH" in calibration_launcher
    assert "--stage2_release_selection_path" in calibration_launcher
    assert "sample_level_unified_sparse_three_expert_route_v6" in evaluator
    assert "sample_level_unified_foundation_static_memory_sparse_top1_v6" in evaluator
    assert "incomplete_support_dynamic_complete_support_top4_foundation_equal_rrf_tail_v2" in router
    assert "complete_support_equal_rrf_tail_else_static_or_recurrent_anchor" in evaluator
    assert '"support_aware_foundation_prefix_preserved_exact"' in evaluator
    assert '"support_aware_incomplete_route_all_exact"' in evaluator
    assert '"foundation_top_skill_ids"' in evaluator
    assert '"recurrent_memory_top_skill_ids"' in evaluator
    assert '"observation_grounding_feature"' in evaluator
    assert "same_prefix_factual_and_action_only_result_suppressed_v1" in evaluator
    assert "one_hot_sparse_top1_expert_selection" in evaluator
    assert "stage0_static_foundation_diagnostic" in evaluator
    assert '"clstr_vnext_candidate_compressor"' in evaluator
    assert '"successor_static_reranker_diagnostic"' in evaluator
    assert "Stage0 static diagnostic unexpectedly declares" in evaluator
    assert '"raw_current"' in evaluator
    assert '"raw_causal"' in evaluator
    assert '"direct"' in evaluator
    assert '"raw_dynamic_end_to_end_route"' in evaluator
    assert '"mixture_probability"' in evaluator
    assert '"candidate_query_is_route_input": False' in evaluator
    assert '"candidate_recall_scores_are_route_inputs": False' in evaluator
    assert '"candidate_positions_are_route_inputs": False' in evaluator
    assert 'default="learned_causal"' in cli
    assert 'default="learned"' in cli
    assert "COMPRESSED_M=${COMPRESSED_M:-64}" in launcher
    assert '"primary_route_scoring"' in evaluator
    assert (
        "history_depth_aware_soft_train_confidence_abstaining_hard_expert_route_final100_v7"
        in evaluator
    )
    assert (
        "history_depth_aware_soft_train_confidence_abstaining_hard_expert_route_final100_v7"
        in evaluator
    )
    assert '"observation_support_gate_is_benchmark_agnostic": bool(' in evaluator
    assert "deployment_history_mask" in evaluator
    assert "static_only_execution=foundation_selected" in evaluator
    assert '[float(record["replay_prefix_length"]) for record in batch_records]' in evaluator
    assert "history_depth=history_depth" in evaluator
    assert "clstr_vnext_checkpoint_native_eval_v20" in evaluator
    assert "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1" in evaluator
    assert "route_logits = route_scores.mixed_logits" in evaluator
    assert '"memory_changes_candidate_support": bool' in evaluator
    assert "route_states = causal_states" in evaluator
    assert '"static_route_context": "compact_causal_skill_action_history"' in evaluator
    assert '"memory_query_state": "structured_current_only"' in evaluator
    assert '"non_static_memory_metrics_release_eligible"' in evaluator
    assert "retrieval_requirement_aware_foundation_dispatch_v1" in evaluator
    assert '"uses_benchmark_identity": False' in evaluator
    assert 'metrics["deployment"]' in evaluator
    assert '"closed_set_recurrent_path_executed"' in evaluator
    assert '"deployment_uses_random_unrestored_stage2_heads": False' in evaluator
    assert '"deployment_uses_recurrent_m_t"' in evaluator
    assert "CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE = 8" in evaluator
    assert '"deployment_uses_closed_set_semantic_fusion"' in evaluator
    assert '"closed_set_semantic_fusion_applied_count"' in evaluator
    assert '"closed_set_semantic_fusion_expected_count"' in evaluator
    assert '"closed_set_semantic_fusion_count_exact"' in evaluator
    assert '"direct" if foundation_selected else candidate_compression_mode' in evaluator
    assert '"direct_causal_semantic"' in evaluator
    assert "preserved_closed_set_static_semantic_rrf_route_v1" in evaluator
    assert "TOOLBENCH_G3_NATIVE_SKILL_COUNT = 27_486" in evaluator
    assert (
        '"transition_state_protocol": "structured_current_state_v1"'
        in evaluator
    )


def test_vnext_eval_uses_benchmark_local_pool_and_excludes_no_call() -> None:
    rows, no_call, report = _prepare_eval_rows(
        benchmark="toolsandbox",
        source_rows=[
            {
                "trajectory_id": "t",
                "step_index": 0,
                "state_text_current": "goal: use tool a",
                "state_text": "goal: use tool a",
                "next_skill_id": "a",
                "equivalent_next_skill_ids": ["b"],
                "candidate_next_skill_ids": ["a", "b"],
            },
            {
                "trajectory_id": "stop",
                "step_index": 0,
                "state_text_current": "goal: no call",
                "state_text": "goal: no call",
                "route_target": "STOP",
            },
        ],
        final_skill_ids=["a", "b", "training-only"],
        max_eval_rows=None,
    )
    assert rows[0]["_vnext_legal_skill_ids"] == ["a", "b"]
    assert rows[0]["_vnext_positive_skill_ids"] == ["a", "b"]
    assert rows[0]["_vnext_pool_protocol"] == "benchmark_row_local_pool"
    assert len(no_call) == 1
    assert report["routing_row_count"] == 1
    assert report["no_call_row_count"] == 1


def test_vnext_trajectbench_uses_after_decision_results_and_multi_positives() -> None:
    rows, no_call, report = _prepare_eval_rows(
        benchmark="trajectbench",
        source_rows=[
            {
                "trajectory_id": "traject::parallel::Travel::q::0",
                "step_index": 0,
                "state_text_current": "goal: use weather and maps",
                "state_text": "goal: use weather and maps",
                "skill_id": "traject/__start__",
                "next_skill_id": "traject/weather",
                "equivalent_next_skill_ids": ["traject/maps"],
                "candidate_next_skill_ids": ["traject/weather", "traject/maps"],
                "target_action_text": "Weather(city=Paris)",
                "actual_result_skill_id": "traject/weather",
                "actual_result_text": "rain",
                "actual_result_executed": True,
            }
        ],
        final_skill_ids=["traject/weather", "traject/maps"],
        max_eval_rows=None,
    )
    assert no_call == []
    assert rows[0]["_vnext_positive_skill_ids"] == [
        "traject/weather",
        "traject/maps",
    ]
    assert rows[0]["_vnext_event_timing"] == "after_decision"
    assert rows[0]["_vnext_executed_skill_id"] == "traject/weather"
    assert rows[0]["_vnext_actual_result_text"] == "rain"
    assert report["pool_protocols"] == {"benchmark_row_local_pool": 1}


def test_vnext_toolbench_eval_uses_checkpoint_plus_benchmark_full_pool() -> None:
    rows, _no_call, _report = _prepare_eval_rows(
        benchmark="toolbench_g3",
        source_rows=[
            {
                "trajectory_id": "t",
                "step_index": 0,
                "state_text_current": "goal: choose b",
                "state_text": "goal: choose b",
                "skill_id": "training-a",
                "action_text": "training-a(x=1)",
                "next_skill_id": "b",
            }
        ],
        final_skill_ids=["training-a", "b", "benchmark-c"],
        max_eval_rows=None,
    )
    assert rows[0]["_vnext_legal_skill_ids"] == [
        "training-a",
        "b",
        "benchmark-c",
    ]
    assert rows[0]["_vnext_pool_protocol"] == "checkpoint_plus_benchmark_full_pool"
    assert rows[0]["_vnext_event_timing"] == "before_decision"
    assert rows[0]["_vnext_executed_skill_id"] == "training-a"


def test_vnext_toolbench_eval_can_use_declared_native_pool() -> None:
    rows, _no_call, report = _prepare_eval_rows(
        benchmark="toolbench_g3",
        source_rows=[
            {
                "trajectory_id": "t",
                "step_index": 0,
                "state_text_current": "goal: choose b",
                "state_text": "goal: choose b",
                "skill_id": "training-a",
                "action_text": "training-a(x=1)",
                "next_skill_id": "b",
            }
        ],
        final_skill_ids=["training-a", "b", "benchmark-c", "training-only"],
        toolbench_legal_skill_ids=["training-a", "b", "benchmark-c"],
        max_eval_rows=None,
    )
    assert rows[0]["_vnext_legal_skill_ids"] == [
        "training-a",
        "b",
        "benchmark-c",
    ]
    assert rows[0]["_vnext_pool_protocol"] == "benchmark_declared_native_pool"
    assert report["pool_protocols"] == {"benchmark_declared_native_pool": 1}


def test_vnext_toolbench_materializes_legacy_current_state_without_history() -> None:
    rows, _no_call, report = _prepare_eval_rows(
        benchmark="toolbench_g3",
        source_rows=[
            {
                "trajectory_id": "legacy-toolbench",
                "step_index": 1,
                "state_text": "goal: choose b\nhistory:\ncalled training-a",
                "skill_id": "training-a",
                "action_text": "training-a(x=1)",
                "next_skill_id": "b",
            }
        ],
        final_skill_ids=["training-a", "b"],
        max_eval_rows=None,
    )

    assert rows[0]["state_text_current"] == "goal: choose b"
    assert rows[0]["_vnext_state_text"] == "goal: choose b"
    assert "called training-a" not in rows[0]["_vnext_state_text"]
    assert report["history_channel"]["status"] == "ok"


def test_vnext_eval_fails_closed_without_explicit_current_state() -> None:
    try:
        _prepare_eval_rows(
            benchmark="toolsandbox",
            source_rows=[
                {
                    "trajectory_id": "legacy",
                    "step_index": 0,
                    "state_text": "goal: choose a\nhistory: leaked",
                    "next_skill_id": "a",
                    "candidate_next_skill_ids": ["a", "b"],
                }
            ],
            final_skill_ids=["a", "b"],
            max_eval_rows=None,
        )
    except ValueError as exc:
        assert "state_text_current" in str(exc)
    else:
        raise AssertionError("vNext evaluator must reject legacy history-bearing input")


def test_vnext_toolbench_replays_executed_result_before_successor_decision() -> None:
    event = _benchmark_event_contract(
        "toolbench_g3",
        {
            "skill_id": "tool-a",
            "action_text": "tool-a(x=1)",
            "next_skill_id": "tool-b",
            "next_observation_text": "actual result from tool-a",
            "observation_source": "actual_tool_result",
        },
    )
    assert event == {
        "timing": "before_decision",
        "skill_id": "tool-a",
        "action_text": "tool-a(x=1)",
        "result_text": "actual result from tool-a",
        "result_alignment": "toolbench_executed_row_schema",
    }


def test_vnext_toolbench_strict_result_contract_fails_before_scoring() -> None:
    row = {
        "benchmark": "toolbench_g3",
        "task_id": "tb::0",
        "trajectory_id": "tb",
        "step_index": 0,
        "state_text": "goal: choose b",
        "skill_id": "a",
        "action_text": "a(x=1)",
        "next_skill_id": "b",
        "next_observation_text": "real result",
    }
    try:
        _prepare_eval_rows(
            benchmark="toolbench_g3",
            source_rows=[row],
            final_skill_ids=["a", "b"],
            max_eval_rows=None,
            require_verified_toolbench_results=True,
        )
    except ValueError as exc:
        assert "exact verified executed-result provenance" in str(exc)
    else:
        raise AssertionError("strict ToolBench result evaluation must fail closed")


def test_vnext_toolbench_strict_result_contract_accepts_exact_event_binding() -> None:
    result = "real result"
    answer_path = "/fixture/G3_answer/tb.json"
    event_id = hashlib.sha256(
        json.dumps(
            {
                "answer_path": answer_path,
                "action_text": "a(x=1)",
                "result_text": result,
                "skill_id": "a",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    rows, _no_call, report = _prepare_eval_rows(
        benchmark="toolbench_g3",
        source_rows=[
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb::0",
                "trajectory_id": "tb",
                "step_index": 0,
                "state_text": "goal: choose b",
                "skill_id": "a",
                "action_text": "a(x=1)",
                "next_skill_id": "b",
                "next_observation_text": result,
                "actual_result_text": result,
                "actual_result_executed": True,
                "actual_result_skill_id": "a",
                "actual_result_event_id": event_id,
                "observation_source": "executed_trace:toolbench_g3",
                "provenance": {"answer_path": answer_path},
            }
        ],
        final_skill_ids=["a", "b"],
        max_eval_rows=None,
        require_verified_toolbench_results=True,
    )

    assert rows[0]["_vnext_actual_result_text"] == result
    contract = report["toolbench_verified_result_contract"]
    assert contract["verified_row_count"] == 1
    assert contract["verified_nonempty_result_row_count"] == 1


def test_vnext_toolbench_preflight_resolves_answer_path_alias(
    tmp_path: Path,
) -> None:
    real_answer = tmp_path / "real" / "answer.json"
    real_answer.parent.mkdir()
    real_answer.write_text("{}", encoding="utf-8")
    alias = tmp_path / "answer-alias.json"
    alias.symlink_to(real_answer)
    result = "real result"
    event_id = hashlib.sha256(
        json.dumps(
            {
                "answer_path": str(alias.resolve()),
                "action_text": "a(x=1)",
                "result_text": result,
                "skill_id": "a",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = _preflight_verified_toolbench_results(
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "tb-alias::0",
                "skill_id": "a",
                "action_text": "a(x=1)",
                "next_observation_text": result,
                "actual_result_text": result,
                "actual_result_executed": True,
                "actual_result_skill_id": "a",
                "actual_result_event_id": event_id,
                "observation_source": "executed_trace:toolbench_g3",
                "provenance": {"answer_path": str(alias)},
            }
        ],
        max_eval_rows=None,
    )
    assert report["verified_row_count"] == 1


def test_vnext_toolbench_raw_preflight_rejects_forged_event_identity() -> None:
    try:
        _preflight_verified_toolbench_results(
            [
                {
                    "task_id": "forged",
                    "skill_id": "a",
                    "action_text": "a(x=1)",
                    "next_observation_text": "result",
                    "actual_result_text": "result",
                    "actual_result_executed": True,
                    "actual_result_skill_id": "a",
                    "actual_result_event_id": "e" * 64,
                    "observation_source": "executed_trace:toolbench_g3",
                    "provenance": {"answer_path": "/fixture/answer.json"},
                }
            ],
            max_eval_rows=None,
        )
    except ValueError as exc:
        assert "exact verified executed-result provenance" in str(exc)
    else:
        raise AssertionError("forged ToolBench event identity must fail preflight")


def test_vnext_toolbench_preflight_runs_before_checkpoint_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import clstr.vnext_eval as module

    def forbidden_loader(**_kwargs):
        raise AssertionError("checkpoint loader must not run before ToolBench preflight")

    monkeypatch.setattr(module, "load_vnext_stage2_for_evaluation", forbidden_loader)
    corpus = SimpleNamespace(
        source_rows=[
            {
                "task_id": "invalid-before-load",
                "skill_id": "a",
                "action_text": "a(x=1)",
                "next_skill_id": "b",
                "next_observation_text": "result",
            }
        ],
        skills=[],
    )
    try:
        run_vnext_benchmark_eval(
            benchmark="toolbench_g3",
            corpus=corpus,
            checkpoint_path=tmp_path / "missing.pt",
            training_skills_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "out",
            require_verified_toolbench_results=True,
        )
    except ValueError as exc:
        assert "exact verified executed-result provenance" in str(exc)
    else:
        raise AssertionError("invalid ToolBench rows must fail before model loading")


def test_vnext_toolbench_native_pool_count_fails_before_checkpoint_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import clstr.vnext_eval as module

    def forbidden_loader(**_kwargs):
        raise AssertionError("checkpoint loader must not run for a wrong native pool")

    monkeypatch.setattr(module, "load_vnext_stage2_for_evaluation", forbidden_loader)
    native_skills = tmp_path / "native_skills.jsonl"
    native_skills.write_text(
        json.dumps({"skill_id": "only-one-skill"}) + "\n",
        encoding="utf-8",
    )
    corpus = SimpleNamespace(
        source_rows=[],
        skills=[{"skill_id": "global-skill"}],
    )
    try:
        run_vnext_benchmark_eval(
            benchmark="toolbench_g3",
            corpus=corpus,
            checkpoint_path=tmp_path / "missing.pt",
            training_skills_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "out",
            toolbench_pool_scope="native",
            toolbench_native_skills_path=native_skills,
        )
    except ValueError as exc:
        assert "exact declared 27486-skill ID pool" in str(exc)
    else:
        raise AssertionError("wrong ToolBench native pool must fail before loading")


def test_vnext_toolbench_native_file_is_only_a_legal_id_mask(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import clstr.vnext_eval as module

    global_skills = [
        {"skill_id": "a", "description": "CLSTR definition a"},
        {"skill_id": "b", "description": "CLSTR definition b"},
    ]
    native_skills = tmp_path / "native_skills.jsonl"
    native_skills.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"skill_id": "a", "description": "conflicting SR definition a"},
                {"skill_id": "b", "description": "conflicting SR definition b"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def capture_loader(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("loader captured")

    monkeypatch.setattr(module, "TOOLBENCH_G3_NATIVE_SKILL_COUNT", 2)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "load_vnext_stage2_for_evaluation", capture_loader)
    corpus = SimpleNamespace(source_rows=[], skills=global_skills)
    try:
        run_vnext_benchmark_eval(
            benchmark="toolbench_g3",
            corpus=corpus,
            checkpoint_path=tmp_path / "missing.pt",
            training_skills_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "out",
            toolbench_pool_scope="native",
            toolbench_native_skills_path=native_skills,
        )
    except RuntimeError as exc:
        assert str(exc) == "loader captured"
    else:
        raise AssertionError("native mask test must stop at the captured loader")
    assert captured["benchmark_skills"] == global_skills
    assert all(
        "conflicting SR definition" not in json.dumps(row)
        for row in captured["benchmark_skills"]
    )


def test_vnext_tau2_replays_target_action_only_after_current_decision() -> None:
    event = _benchmark_event_contract(
        "tau2",
        {
            "next_skill_id": "tau2/retail/get_order",
            "next_observation_text": "oracle_next_action_arguments: {order_id: 1}",
            "observation_source": "oracle_next_action_arguments",
        },
    )
    assert event["timing"] == "after_decision"
    assert event["skill_id"] == "tau2/retail/get_order"
    assert "oracle_next_action_arguments" in event["action_text"]
    assert event["result_text"] == ""


def test_vnext_eval_treats_oracle_arguments_as_action_not_result() -> None:
    row = {
        "next_skill_id": "tau2/domain/action",
        "next_observation_text": "oracle_next_action_arguments: {x: 1}",
        "observation_source": "oracle_next_action_arguments",
    }
    assert "oracle_next_action_arguments" in _executed_action_text(row)
    assert _aligned_actual_result(row) == ""
    row.update(
        {
            "actual_result_executed": True,
            "actual_result_skill_id": "tau2/domain/action",
            "actual_result_text": "real executed output",
        }
    )
    assert _aligned_actual_result(row) == "real executed output"


def test_vnext_memory_release_requires_a_prior_observed_result(
    monkeypatch,
) -> None:
    import clstr.vnext_eval as module

    rows = [
        _timeline_row("after", 0, timing="after_decision", skill_id="a"),
        _timeline_row("after", 1, timing="after_decision", skill_id="b"),
        _timeline_row("before", 0, timing="before_decision", skill_id="a"),
    ]
    for index, row in enumerate(rows):
        row["_vnext_source_index"] = index
        row["_vnext_legal_skill_ids"] = ["a", "b"]
        row["_vnext_target_skill_id"] = row["_vnext_executed_skill_id"]
    _materialize_causal_eval_timeline(rows)

    monkeypatch.setattr(
        module,
        "_initial_memory",
        lambda *_args, **_kwargs: torch.zeros((1, 2)),
    )
    monkeypatch.setattr(
        module,
        "_apply_event",
        lambda _model, memory, _event, **_kwargs: memory + 1,
    )

    records, report = _materialize_memories(
        SimpleNamespace(),
        rows,
        state_cache=None,
        action_cache=None,
        result_cache=None,
        skill_id_to_idx={"a": 0, "b": 1},
        device=torch.device("cpu"),
        belief_top_k=2,
    )
    by_trajectory_step = {
        (record["row"]["trajectory_id"], record["row"]["step_index"]): record
        for record in records
    }

    after_zero = by_trajectory_step[("after", 0)]
    after_one = by_trajectory_step[("after", 1)]
    before_zero = by_trajectory_step[("before", 0)]
    assert after_zero["uses_recurrent_m_t"] is False
    assert after_zero["uses_observation_supported_m_t"] is False
    assert after_one["uses_recurrent_m_t"] is True
    assert after_one["uses_observation_supported_m_t"] is True
    assert after_one["observed_result_correction_count"] == 1
    assert before_zero["uses_recurrent_m_t"] is True
    assert before_zero["uses_observation_supported_m_t"] is True
    assert report["actual_result_correction_count"] == 3
    assert report["observation_supported_route_count"] == 2


def test_vnext_native_eval_materializes_fixed_latent_traces(monkeypatch) -> None:
    import clstr.vnext_eval as module

    rows = [
        _timeline_row("native", 0, timing="after_decision", skill_id="a"),
        _timeline_row("native", 1, timing="after_decision", skill_id="b"),
    ]
    for index, row in enumerate(rows):
        row["_vnext_source_index"] = index
        row["_vnext_legal_skill_ids"] = ["a", "b"]
        row["_vnext_target_skill_id"] = row["_vnext_executed_skill_id"]
    _materialize_causal_eval_timeline(rows)

    class Core:
        synchronization_enabled = True
        synchronization_trace_length = 3

        def append_latent_trace(self, memory, latent_trace):
            return torch.cat((latent_trace, memory.unsqueeze(1)), dim=1)[:, -3:]

    model = SimpleNamespace(vnext=Core())
    monkeypatch.setattr(
        module,
        "_initial_memory",
        lambda *_args, **_kwargs: torch.zeros((1, 2)),
    )

    def apply(_model, memory, _event, *, latent_trace, **_kwargs):
        updated = memory + 1
        return updated, _model.vnext.append_latent_trace(updated, latent_trace)

    monkeypatch.setattr(module, "_apply_event", apply)
    records, report = _materialize_memories(
        model,
        rows,
        state_cache=None,
        action_cache=None,
        result_cache=None,
        skill_id_to_idx={"a": 0, "b": 1},
        device=torch.device("cpu"),
        belief_top_k=2,
    )
    assert report["native_synchronization_enabled"] is True
    assert report["native_synchronization_trace_length"] == 3
    assert records[0]["factual_latent_trace"].shape == (1, 3, 2)
    assert torch.equal(
        records[1]["factual_latent_trace"],
        torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]),
    )


def test_vnext_eval_ranking_is_deterministic_and_missing_candidate_is_zero_credit() -> None:
    logits = torch.tensor([[2.0, 2.0, 1.0]])
    legal = torch.tensor([[True, True, True]])
    ranks = _full_pool_rank(logits, torch.tensor([1]), legal)
    assert ranks.tolist() == [2]
    candidate_ranks = _candidate_rank(
        torch.tensor([[3.0, 2.0]]),
        torch.tensor([[0, 2]]),
        torch.tensor([[True, True]]),
        torch.tensor([1]),
    )
    assert candidate_ranks == [None]
    assert _metrics(candidate_ranks)["mrr"] == 0.0


def test_vnext_eval_multi_positive_rank_uses_best_legal_positive() -> None:
    logits = torch.tensor([[4.0, 3.0, 3.0, 2.0]])
    positive = torch.tensor([[False, False, True, True]])
    legal = torch.tensor([[True, True, True, True]])
    assert _multi_positive_rank(logits, positive, legal) == [3]
    assert _multi_positive_rank(
        logits,
        torch.zeros_like(positive),
        legal,
    ) == [None]
    assert _candidate_multi_positive_rank(
        torch.tensor([[4.0, 3.0, 2.0]]),
        torch.tensor([[0, 2, 3]]),
        torch.tensor([[True, True, True]]),
        positive,
    ) == [2]


def test_vnext_eval_rejects_equivalent_positive_outside_legal_pool() -> None:
    with pytest.raises(ValueError, match="positive set escapes"):
        _prepare_eval_rows(
            benchmark="toolsandbox",
            source_rows=[
                {
                    "trajectory_id": "t",
                    "step_index": 0,
                    "state_text_current": "goal: choose a or b",
                    "state_text": "goal: choose a or b",
                    "next_skill_id": "a",
                    "equivalent_next_skill_ids": ["b"],
                    "candidate_next_skill_ids": ["a"],
                }
            ],
            final_skill_ids=["a", "b"],
            max_eval_rows=None,
        )


def test_vnext_unseen_eval_training_leak_scan_matches_exact_ids_only() -> None:
    heldout = {"skill/heldout"}
    value = {
        "positive": "skill/heldout",
        "description": "mentions skill/heldout inside a longer sentence",
        "nested": ["seen", {"negative": "skill/heldout"}],
    }
    assert _exact_id_occurrence_count(value, heldout) == 2


def test_vnext_toolbench_source_contract_hashes_both_inputs(tmp_path: Path) -> None:
    trajectories = tmp_path / "eval.jsonl"
    skills = tmp_path / "skills.jsonl"
    trajectories.write_text('{"next_skill_id":"a"}\n', encoding="utf-8")
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")
    contract = _corpus_source_contract(
        "toolbench_g3",
        SimpleNamespace(
            report={
                "eval_trajectories_path": str(trajectories),
                "skills_path": str(skills),
            }
        ),
    )
    assert contract["protocol"] == "toolbench_exact_files_v1"
    assert contract["file_count"] == 2
    assert len(contract["contract_sha256"]) == 64


def test_vnext_trajectbench_source_contract_hashes_raw_public_files(
    tmp_path: Path,
) -> None:
    public_data = tmp_path / "public_data"
    tools = public_data / "tools/all_tools.json"
    query = public_data / "sequential/Travel/traj_query.json"
    tools.parent.mkdir(parents=True)
    query.parent.mkdir(parents=True)
    tools.write_text("[]\n", encoding="utf-8")
    query.write_text("[]\n", encoding="utf-8")
    contract = _corpus_source_contract(
        "trajectbench",
        SimpleNamespace(
            report={
                "public_data": str(public_data),
                "split_partition": "test",
                "split_policy": "deterministic_hash_trajectory_level_v1",
                "source_files": [str(tools), str(query)],
            }
        ),
    )
    assert contract["protocol"] == (
        "trajectbench_deterministic_heldout_global_inventory_v1"
    )
    assert contract["split_partition"] == "test"
    assert contract["file_count"] == 2


def test_vnext_eval_source_cannot_reenter_legacy_route_paths() -> None:
    source = Path(__file__).parents[1].joinpath("clstr", "vnext_eval.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "route_memory_utility_gate",
        "route_memory_candidate_utility_gate",
        "stop_head",
        "causal_update_count",
        "attach_stage0_topm_candidates",
        "candidate_gate_probability",
        "candidate_gate_role",
    ):
        assert forbidden not in source


def test_vnext_eval_verifies_overlapping_skill_definitions() -> None:
    serializer = lambda row: f"{row['name']}::{row.get('description', '')}"
    appended, overlaps = _partition_benchmark_skill_append(
        [{"skill_id": "a", "name": "A", "description": "same"}],
        [
            {"skill_id": "a", "name": "A", "description": "same"},
            {"skill_id": "b", "name": "B", "description": "new"},
        ],
        serializer=serializer,
    )
    assert [row["skill_id"] for row in appended] == ["b"]
    assert overlaps == ["a"]
    try:
        _partition_benchmark_skill_append(
            [{"skill_id": "a", "name": "A", "description": "old"}],
            [{"skill_id": "a", "name": "A", "description": "changed"}],
            serializer=serializer,
        )
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("conflicting benchmark skill definitions must fail closed")


def test_vnext_eval_loads_only_digest_bound_matched_test_rows(tmp_path: Path) -> None:
    rows_path = tmp_path / "tau2_test_rows.jsonl"
    skills_path = tmp_path / "tau2_skills.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "matched_data_split": "test",
                "state_text_current": "goal: refund order",
                "next_skill_id": "tau2/retail/refund",
                "candidate_next_skill_ids": ["tau2/retail/refund"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    skills_path.write_text(
        json.dumps({"skill_id": "tau2/retail/refund", "name": "refund"}) + "\n",
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "schema_version": "clstr_matched_multibench_union_v1",
        "status": "ok",
        "route_files": {
            "tau2_test": {
                "path": str(rows_path.resolve()),
                "sha256": digest(rows_path),
                "rows": 1,
            },
            "tau2_skills": {
                "path": str(skills_path.resolve()),
                "sha256": digest(skills_path),
                "rows": 1,
            },
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    manifest_path = tmp_path / "matched_union_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark, corpus = _load_corpus(
        SimpleNamespace(
            benchmark="tau2",
            prebuilt_source_rows_path=str(rows_path),
            prebuilt_skills_path=str(skills_path),
            matched_union_manifest_path=str(manifest_path),
            matched_split="test",
        )
    )
    assert benchmark == "tau2"
    assert corpus.source_rows[0]["matched_data_split"] == "test"
    assert corpus.report["matched_union_manifest_sha256"] == payload[
        "manifest_sha256"
    ]
