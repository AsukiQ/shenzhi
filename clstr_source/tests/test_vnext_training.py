from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from scripts.migrate_clstr_vnext_stage0_v5_segment_contract import (
    migrate_contract as migrate_stage0_v5_segment_contract,
)

from clstr.belief import InitialBelief
from clstr.model import UnifiedMemoryRetriever
from clstr.vnext_core import CLSTRVNextCore
from clstr.vnext_stage0_train import (
    _all_positive_topk_coverage,
    _exact_factual_route_metrics,
    _filter_rows,
    _partition_stage0_eligible_rows,
    _positive_set_cardinality_report,
    _ranking_metrics,
    _schedule_batches,
)
from clstr.vnext_training import (
    candidate_foundation_digest,
    capture_frozen_backbone_snapshot,
    configure_vnext_stage0,
    configure_vnext_candidate_compressor,
    configure_vnext_static_route_adapter,
    configure_vnext_stage2,
    derive_inventory_catalog_subset,
    immutable_run_contract,
    load_or_build_frozen_text_cache,
    load_frozen_text_cache_read_only,
    load_or_build_skill_embedding_cache,
    load_frozen_backbone_snapshot,
    load_or_capture_frozen_backbone_snapshot,
    require_canonical_vnext_checkpoint_state,
    require_canonical_trainability,
    require_matching_run_contract,
    mapped_legacy_stage0_state,
    require_verified_data_contract,
    semantic_source_id,
    state_digest,
    vnext_checkpoint_state,
    verify_frozen_backbone_contract,
    rewrite_inventory_catalog_references,
    seed_vnext_run,
    stable_stratified_cap_rows,
    static_foundation_digest,
    static_route_foundation_digest,
)


def test_legacy_stage0_mapper_transplants_only_static_foundation() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.proj = torch.nn.Linear(2, 2)
            self.skill_table = torch.nn.Module()
            self.skill_table.E = torch.nn.Parameter(torch.randn(3, 2))
            self.skill_table.W = torch.nn.Linear(2, 2, bias=False)
            self.skill_table.logit_scale_retr = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.zeros(3))
            self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
            self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.zeros(3))
            self.initial_belief_head = InitialBelief(2)
            self.vnext = CLSTRVNextCore(2, hidden_dim=2)

    model = Model()
    legacy_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("vnext.")
    }
    legacy_unified = UnifiedMemoryRetriever(2)
    legacy_state.update(
        {
            f"unified_retriever.{key}": value.detach().clone()
            for key, value in legacy_unified.state_dict().items()
        }
    )
    legacy_state["stop_head.weight"] = torch.randn(1, 2)
    mapped, report = mapped_legacy_stage0_state(
        model,
        legacy_state,
        legacy_skill_ids=["a", "b", "c"],
        current_skill_ids=["c", "a", "b"],
    )
    assert report["status"] == "ok"
    assert "stop_head.weight" not in mapped
    assert torch.equal(
        mapped["vnext.static_query.fuse.0.weight"],
        legacy_state["unified_retriever.fuse.0.weight"],
    )
    assert set(report["required_targets"]).issubset(mapped)
    assert report["skill_id_alignment"]["enabled"] is True
    assert torch.equal(mapped["skill_table.E"], legacy_state["skill_table.E"][[2, 0, 1]])


def test_frozen_backbone_snapshot_binds_weight_and_tokenizer_contents(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model.safetensors").write_bytes(b"weights-v1")
    (model_root / "config.json").write_text('{"hidden_size":2}\n', encoding="utf-8")
    (model_root / "tokenizer.json").write_text('{"version":1}\n', encoding="utf-8")
    manifest_path = tmp_path / "backbone_snapshot.json"
    captured = capture_frozen_backbone_snapshot(model_root)
    assert captured["contract"]["files"]
    resolved = load_or_capture_frozen_backbone_snapshot(
        model_root,
        manifest_path=manifest_path,
    )
    assert resolved["contract_digest"] == captured["contract"]["contract_digest"]
    verified = load_frozen_backbone_snapshot(
        manifest_path,
        expected_model_name_or_path=model_root,
        verify_all_hashes=True,
    )
    assert verified["verified_hash_count"] == 3
    embedded = verify_frozen_backbone_contract(
        verified["contract"],
        expected_model_name_or_path=model_root,
    )
    assert embedded["full_hash_verification"] is True
    (model_root / "tokenizer.json").write_text('{"version":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        load_frozen_backbone_snapshot(
            manifest_path,
            expected_model_name_or_path=model_root,
            verify_all_hashes=True,
        )


def test_semantic_source_prefers_persisted_adapter_provenance() -> None:
    assert semantic_source_id(
        {
            "benchmark": "benchmark-name",
            "source": "row-source",
            "provenance": {"source_id": "adapter-source"},
        }
    ) == "adapter-source"


def test_vnext_run_seed_reproduces_python_and_torch_initialization() -> None:
    first_report = seed_vnext_run(29)
    first_python = random.random()
    first_torch = torch.rand(4)
    second_report = seed_vnext_run(29)
    assert random.random() == first_python
    assert torch.equal(torch.rand(4), first_torch)
    assert first_report == second_report
    assert first_report["cudnn_benchmark"] is False
    assert first_report["cudnn_deterministic"] is True


def test_stage0_fails_closed_without_explicit_current_state() -> None:
    with pytest.raises(ValueError, match="state_text_current"):
        _filter_rows(
            [
                {
                    "state_text": "goal: legacy prompt\nhistory: leaked",
                    "required_tool_set_skill_ids": ["a"],
                }
            ],
            {"a"},
            kind="retrieval",
        )


@pytest.mark.parametrize("kind", ["retrieval", "static_route"])
def test_stage0_fails_closed_without_explicit_causal_events(kind: str) -> None:
    positive_field = (
        "required_tool_set_skill_ids"
        if kind == "retrieval"
        else "current_state_route_set_skill_ids"
    )
    with pytest.raises(ValueError, match="persisted events"):
        _filter_rows(
            [
                {
                    "source": "fixture",
                    "state_text_current": "goal: current only",
                    positive_field: ["a"],
                }
            ],
            {"a"},
            kind=kind,
        )


@pytest.mark.parametrize(
    ("kind", "positive_field"),
    [
        ("retrieval", "required_tool_set_skill_ids"),
        ("static_route", "current_state_route_set_skill_ids"),
    ],
)
def test_stage0_uses_causal_query_for_both_heads(
    kind: str,
    positive_field: str,
) -> None:
    rows = _filter_rows(
        [
            {
                "source": "fixture",
                "state_text_current": "goal: current only",
                "state_text_causal": "goal: current only\ncausal_prefix:\nevent[0]: prior",
                "causal_prefix_events": [
                    {
                        "step_index": 0,
                        "skill_id": "prior-skill",
                        "action_text": "call prior action",
                        "result_text": "raw result must not enter static query",
                        "result_executed": True,
                    }
                ],
                positive_field: ["a"],
            }
        ],
        {"a"},
        kind=kind,
    )
    assert "prior-skill" in rows[0]["_vnext_query"]
    assert "call prior action" in rows[0]["_vnext_query"]
    assert "raw result must not enter static query" not in rows[0]["_vnext_query"]
    assert rows[0]["_vnext_query_channel"] == "compact_causal_skill_action_v1"


def test_vnext_entrypoints_bootstrap_the_repository_root() -> None:
    root = Path(__file__).parents[1]
    for relative in (
        "scripts/run_clstr_vnext_stage0_train.py",
        "scripts/run_clstr_vnext_stage2_train.py",
        "scripts/run_clstr_vnext_precompute.py",
        "scripts/run_clstr_vnext_eval.py",
        "scripts/run_clstr_vnext_unseen_eval.py",
    ):
        source = root.joinpath(relative).read_text(encoding="utf-8")
        assert "ROOT = Path(__file__).resolve().parents[1]" in source
        assert "sys.path.insert(0, str(ROOT))" in source
        if "stage0_train" in relative or "stage2_train" in relative:
            assert '"--data_contract_path"' in source


def test_canonical_full_chain_uses_static_reranker_before_stage2() -> None:
    root = Path(__file__).parents[1]
    submit = root.joinpath("scripts/submit_clstr_vnext_full_chain.sh").read_text(
        encoding="utf-8"
    )
    assert "run_clstr_vnext_full_precompute.sh" in submit
    assert "run_clstr_vnext_full_stage0_segment.sh" in submit
    assert "run_clstr_vnext_candidate_compressor.sh" in submit
    assert "run_clstr_vnext_full_stage2_segment.sh" in submit
    assert "STATIC_RERANK_VALIDATION_INTERVAL=${STATIC_RERANK_VALIDATION_INTERVAL:-300}" in submit
    assert "VALIDATION_INTERVAL=${STATIC_RERANK_VALIDATION_INTERVAL}" in submit
    assert "stage1" not in submit.lower()
    assert "stage4" not in submit.lower()
    assert "DATA_MANIFEST_PATH must name a freshly audited causal-view manifest" in submit

    stage2 = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_full_stage2_segment.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_a800" in stage2
    assert "#SBATCH -p gpu_h100" not in stage2
    assert "gpu_a800,gpu_h100,gpu_h200" not in stage2
    assert "--max_horizon 16" in stage2
    assert (
        "FAMILY_FIRST_HORIZON_SAMPLING=${FAMILY_FIRST_HORIZON_SAMPLING:-0}"
        in stage2
    )
    assert "family_horizon_args=(--family_first_horizon_sampling)" in stage2
    assert '"${family_horizon_args[@]}"' in stage2
    assert "--max_dev_pairs_per_kind 0" in stage2
    assert "MAX_ORDINARY_DEV_ROWS=${MAX_ORDINARY_DEV_ROWS:-1024}" in stage2
    assert '--max_ordinary_dev_rows "${MAX_ORDINARY_DEV_ROWS}"' in stage2
    assert "--ordinary_mrr_noninferiority_tolerance 0.01" in stage2
    assert "candidate_recall_noninferiority_tolerance" not in stage2
    assert "--minimum_ordinary_dev_stratum_coverage 0.90" in stage2
    assert "--minimum_full_pool_clusters 20" in stage2
    assert "minimum_static_miss_clusters" not in stage2
    assert "--require_clean_source" in stage2
    assert (
        'STAGE0_OUTPUT_DIR=$(realpath -m "${STAGE0_OUTPUT_DIR:-${RUN_ROOT}/stage0}")'
        in stage2
    )
    assert "Stage0 output must remain inside RUN_ROOT" in stage2
    assert 'OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-${RUN_ROOT}/stage2}")' in stage2
    assert "writable OUTPUT_DIR must remain inside RUN_ROOT" in stage2
    assert "FINALIZE_FULL_CHAIN" in stage2
    assert 'report = json.loads((stage0_root / "train_report.json").read_text())' in stage2
    assert 'checkpoint = Path(str(report.get("checkpoint_path") or ""))' in stage2
    assert "Stage2 Stage0 report does not name its final checkpoint" in stage2
    assert "stage0_selection.json" in stage2
    assert "compressor_selection.json" in stage2
    assert "--candidate_checkpoint_path" in stage2
    assert "stage2_selection.json" in stage2
    assert "Stage2 resume requires a mechanically complete prior segment" in stage2
    assert 'selection.get("status") not in {"ok", "action_required"}' in stage2
    assert "Stage2 resume report/selection boundary is inconsistent" in stage2

    stage0 = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_full_stage0_segment.sh"
    ).read_text(encoding="utf-8")
    assert 'OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-${RUN_ROOT}/stage0}")' in stage0
    assert "writable OUTPUT_DIR must remain inside RUN_ROOT" in stage0
    assert 'BACKBONE_SNAPSHOT_PATH=$(realpath -m' in stage0
    assert '--backbone_snapshot_path "${BACKBONE_SNAPSHOT_PATH}"' in stage0
    assert "schedule_aware_preprojection_v1" in root.joinpath(
        "clstr/vnext_stage0_train.py"
    ).read_text(encoding="utf-8")
    for source in (
        root.joinpath("scripts/sbatch/run_clstr_vnext_full_precompute.sh").read_text(
            encoding="utf-8"
        ),
        stage0,
        stage2,
    ):
        assert "writable CACHE_ROOT must remain inside RUN_ROOT" in source
        assert 'RUN_ROOT=$(realpath -m "${RUN_ROOT}")' in source


def test_vnext_eval_launcher_requires_verified_toolbench_and_matched_heldout_rows() -> None:
    root = Path(__file__).parents[1]
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_benchmark_eval.sh"
    ).read_text(encoding="utf-8")
    exporter = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_toolbench_verified_export.sh"
    ).read_text(encoding="utf-8")
    assert "--require_verified_toolbench_results" in launcher
    assert 'STAGE2_OUTPUT_DIR=$(realpath -m "${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/stage2}")' in launcher
    assert "Stage2 selection is not release-ready" in launcher
    assert "explicit CHECKPOINT_PATH does not match the release-ready Stage2 selection" in launcher
    assert "TOOLBENCH_ROWS must name the verified v2 export" in launcher
    assert "TOOLBENCH_POOL_SCOPE=${TOOLBENCH_POOL_SCOPE:-global}" in launcher
    assert '--toolbench_pool_scope "${TOOLBENCH_POOL_SCOPE}"' in launcher
    assert "native ToolBench evaluation requires TOOLBENCH_NATIVE_SKILLS" in launcher
    assert '--toolbench_native_skills_path "${TOOLBENCH_NATIVE_SKILLS}"' in launcher
    assert "--tau2_task_split base" not in launcher
    assert "MATCHED_UNION_MANIFEST_PATH" in launcher
    assert "MATCHED_SPLIT=${MATCHED_SPLIT:-test}" in launcher
    assert "append_matched_eval_args toolsandbox" in launcher
    assert "append_matched_eval_args tau2" in launcher
    assert '--prebuilt_source_rows_path "${matched_artifacts[0]}"' in launcher
    assert '--prebuilt_skills_path "${matched_artifacts[1]}"' in launcher
    assert '--matched_union_manifest_path "${MATCHED_UNION_MANIFEST_PATH}"' in launcher
    assert '--matched_split "${MATCHED_SPLIT}"' in launcher
    assert "SOURCE_EVAL_TRAJECTORIES must name the SR/ToolREx-aligned official split" in exporter
    assert "--verification_skills_path" in exporter
    assert "scripts/import_toolbench_g3.py" in exporter
    assert "scripts/verify_toolbench_g3_official_eval_results.py" in exporter


def test_vnext_smoke_launchers_have_no_stale_artifact_defaults() -> None:
    root = Path(__file__).parents[1]
    stage0 = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_stage0_smoke.sh"
    ).read_text(encoding="utf-8")
    stage2 = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_stage2_smoke.sh"
    ).read_text(encoding="utf-8")
    for source in (stage0, stage2):
        assert "clstr_vnext_smoke_bundle_20260715" not in source
        assert "FROZEN_CACHE_DIR must identify" in source
    assert "--full_pool_chunk_size" not in stage0
    assert "BUNDLE_ROOT must come from the freshly audited causal manifest" in stage0
    assert "STAGE0_OUTPUT_DIR must be the matching Stage0 smoke" in stage2
    assert "CANDIDATE_OUTPUT_DIR must be the matching static-reranker smoke" in stage2
    assert 'compressor_selection.json' in stage2
    assert 'MAX_LENGTH=${MAX_LENGTH:-2048}' in stage0
    assert 'BELIEF_TOP_K=${BELIEF_TOP_K:-64}' in stage0
    assert '--max_length "${MAX_LENGTH}"' in stage0
    assert '--belief_top_k "${BELIEF_TOP_K}"' in stage0
    assert 'BELIEF_TOP_K=${BELIEF_TOP_K:-64}' in stage2
    assert '--belief_top_k "${BELIEF_TOP_K}"' in stage2

    precompute_source = root.joinpath("clstr/vnext_precompute.py").read_text(
        encoding="utf-8"
    )
    assert 'output_dir / "backbone_snapshot.json"' in precompute_source
    evaluator_source = root.joinpath("clstr/vnext_eval.py").read_text(
        encoding="utf-8"
    )
    assert "verify_frozen_backbone_contract" in evaluator_source

    stage0_source = root.joinpath("clstr/vnext_stage0_train.py").read_text(
        encoding="utf-8"
    )
    assert "Stage0 resume checkpoint lacks cumulative trainer_progress" in stage0_source
    assert 'output_dir / "trainer_progress.json"' in stage0_source
    assert "checkpoint_interval == validation_interval" in stage0_source
    assert "clstr_vnext_stage0_unified_static_run_v7" in stage0_source
    assert "model.vnext.static_query(h_t, b_t)" in stage0_source
    assert "unified_query.data_ptr()" not in stage0_source
    assert "deterministic_absolute_optimizer_step_v1" in stage0_source
    assert "scheduled_training_occurrence_count" not in stage0_source


def test_stage0_v5_segment_contract_migration_changes_only_segment_count() -> None:
    observed = immutable_run_contract(
        {
            "schema_version": "clstr_vnext_stage0_unified_static_run_v5",
            "inputs": {"rows_sha256": "rows"},
            "optimization": {
                "batch_size": 128,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2.0e-5,
                "scheduled_training_occurrence_count": 128000,
            },
            "source_contract_digest": "source",
        }
    )
    migrated, record = migrate_stage0_v5_segment_contract(
        observed,
        checkpoint_step=500,
        segment_end_step=5000,
    )
    assert record["old_scheduled_training_occurrence_count"] == 128000
    assert record["new_scheduled_training_occurrence_count"] == 1152000
    assert migrated["optimization"]["scheduled_training_occurrence_count"] == 1152000
    restored = __import__("copy").deepcopy(migrated)
    restored["optimization"]["scheduled_training_occurrence_count"] = 128000
    restored = immutable_run_contract(restored)
    assert restored == observed
    with pytest.raises(ValueError, match="completed step"):
        migrate_stage0_v5_segment_contract(
            observed,
            checkpoint_step=501,
            segment_end_step=5000,
        )


def test_stage0_schedule_balances_sources_and_resumes_exact_suffix() -> None:
    retrieval = [
        {"source": source, "row_id": f"r-{source}-{index}"}
        for source, count in (("ra", 1), ("rb", 5))
        for index in range(count)
    ]
    static = [
        {"benchmark": source, "row_id": f"s-{source}-{index}"}
        for source, count in (("sa", 7), ("sb", 1))
        for index in range(count)
    ]
    full, report = _schedule_batches(
        retrieval,
        static,
        start_step=1,
        max_steps=4,
        gradient_accumulation_steps=1,
        batch_size=4,
        seed=17,
    )
    resumed, _resume_report = _schedule_batches(
        retrieval,
        static,
        start_step=3,
        max_steps=4,
        gradient_accumulation_steps=1,
        batch_size=4,
        seed=17,
    )
    assert [row["row_id"] for batch in resumed for row in batch] == [
        row["row_id"] for batch in full[2:] for row in batch
    ]
    assert report["sampled_exposures"] == {
        "retrieval:ra": 4,
        "retrieval:rb": 4,
        "static_route:sa": 4,
        "static_route:sb": 4,
    }


def test_stratified_dev_cap_round_robins_sources_and_tasks() -> None:
    rows = [
        {
            "source": source,
            "task_id": task,
            "state_text_current": f"{source}-{task}-{index}",
        }
        for source, task, count in (
            ("large", "task-a", 8),
            ("large", "task-b", 2),
            ("small", "task-c", 1),
        )
        for index in range(count)
    ]
    selected, report = stable_stratified_cap_rows(rows, 3)
    assert {row["source"] for row in selected} == {"large", "small"}
    assert {row["task_id"] for row in selected if row["source"] == "large"} == {
        "task-a",
        "task-b",
    }
    assert report["protocol"] == "source_stratum_task_round_robin_v1"


def test_stage0_reports_frequency_weighted_exact_factual_route_metrics() -> None:
    report = _exact_factual_route_metrics(
        torch.tensor([[3.0, 2.0, 1.0]]),
        [
            {
                "current_state_route_factual_target_counts": {
                    "a": 1,
                    "b": 3,
                }
            }
        ],
        torch.tensor([[True, True, True]]),
        {"a": 0, "b": 1, "c": 2},
    )
    assert report["weight_sum"] == 4.0
    assert report["mrr_sum"] / report["weight_sum"] == 0.625
    assert report["recall_at_1_sum"] / report["weight_sum"] == 0.25
    assert report["recall_at_500_sum"] / report["weight_sum"] == 1.0


def test_stage0_reports_full_pool_recall_at_500_separately_from_top100() -> None:
    logits = torch.arange(600, 0, -1, dtype=torch.float32).unsqueeze(0)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    positive[0, 199] = True
    legal = torch.ones_like(positive)
    report = _ranking_metrics(logits, positive, legal)
    assert report["recall_at_100_sum"] == 0.0
    assert report["recall_at_500_sum"] == 1.0


def test_stage0_all_positive_topk_coverage_requires_every_target() -> None:
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    positive = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
        ]
    )
    legal = torch.ones_like(positive)

    report, complete = _all_positive_topk_coverage(
        logits,
        positive,
        legal,
        k=2,
    )

    assert report["positive_target_count"] == 4.0
    assert report["positive_target_hit_count"] == 3.0
    assert report["complete_row_count"] == 1.0
    assert complete.tolist() == [True, False]


def test_stage0_all_positive_topk_coverage_uses_canonical_ties() -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    positive = torch.tensor([[False, True, True]])
    legal = torch.ones_like(positive)

    report, complete = _all_positive_topk_coverage(
        logits,
        positive,
        legal,
        k=2,
    )

    assert report["positive_target_hit_count"] == 1.0
    assert report["complete_row_count"] == 0.0
    assert complete.tolist() == [False]


def test_stage0_reports_multi_positive_cardinality_by_source() -> None:
    report = _positive_set_cardinality_report(
        [
            {"source": "a", "_vnext_positive_skill_ids": ["x"]},
            {"source": "a", "_vnext_positive_skill_ids": ["x", "y", "x"]},
            {"source": "b", "_vnext_positive_skill_ids": ["z", "q"]},
        ]
    )

    assert report["positive_count_distribution"] == {"1": 1, "2": 2}
    assert report["multi_positive_row_count"] == 2
    assert report["multi_positive_by_source"] == {"a": 1, "b": 1}


def test_stage0_excludes_nondiscriminative_rows_from_train_and_dev() -> None:
    catalogs = {
        "pool": {
            "inventory_catalog_id": "pool",
            "inventory_catalog_digest": "digest",
            "runtime_visible_skill_ids": ["a", "b"],
        }
    }
    rows = [
        {
            "source": "fixture",
            "runtime_visible_catalog_id": "pool",
            "inventory_catalog_digest": "digest",
            "_vnext_positive_skill_ids": ["a"],
        },
        {
            "source": "fixture",
            "runtime_visible_catalog_id": "pool",
            "inventory_catalog_digest": "digest",
            "_vnext_positive_skill_ids": ["a", "b"],
        },
    ]
    eligible, report = _partition_stage0_eligible_rows(
        rows,
        catalogs,
        kind="static_route_dev",
    )
    assert eligible == [rows[0]]
    assert report["eligible_row_count"] == 1
    assert report["excluded_reasons"] == {"no_legal_negative": 1}


def test_state_digest_supports_scalar_and_bfloat16_tensors() -> None:
    model = torch.nn.Module()
    model.register_parameter("scalar", torch.nn.Parameter(torch.tensor(1.25)))
    model.register_buffer("bf16_vector", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    first = state_digest(model, include=lambda _name: True)
    second = state_digest(model, include=lambda _name: True)
    assert first == second
    assert len(first) == 64


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.backbone = torch.nn.Linear(2, 2)
        self.encoder.proj = torch.nn.Linear(2, 2)
        self.skill_table = torch.nn.Module()
        self.skill_table.W = torch.nn.Linear(2, 2)
        self.skill_table.E = torch.nn.Parameter(torch.randn(3, 2))
        self.skill_table.logit_scale_belief = torch.nn.Parameter(torch.zeros(1))
        self.skill_table.skill_bias_retr = torch.nn.Parameter(torch.ones(3))
        self.skill_table.skill_bias_belief = torch.nn.Parameter(torch.ones(3))
        self.skill_table.encoder_fn = self.encoder
        self.initial_belief_head = torch.nn.Linear(2, 2)
        self.stop_head = torch.nn.Linear(2, 1)
        self.route_memory_utility_gate = torch.nn.Linear(2, 1)
        self.vnext = torch.nn.Module()
        self.vnext.static_query = torch.nn.Linear(2, 2)
        self.vnext.unified_route_query = torch.nn.Linear(2, 2)
        self.vnext.route_skill_adapter = torch.nn.Linear(2, 2, bias=False)
        self.vnext.static_route_query_delta = torch.nn.Linear(2, 2)
        self.vnext.memory_recall_query = torch.nn.Linear(2, 2)
        self.vnext.memory_route_query = torch.nn.Linear(2, 2)
        self.vnext.candidate_route_gate = torch.nn.Linear(2, 2)
        self.vnext.route_expert_mixture = torch.nn.Linear(2, 2)
        self.vnext.candidate_compressor = torch.nn.Linear(2, 2)
        self.vnext.action_adapter = torch.nn.Linear(2, 2)
        self.vnext.result_adapter = torch.nn.Linear(2, 2)
        self.vnext.transition_delta = torch.nn.Linear(2, 2)
        self.vnext.correction_delta = torch.nn.Linear(2, 2)
        self.vnext.correction_gate = torch.nn.Linear(2, 1)
        self.vnext.transition_scale = torch.nn.Linear(1, 1)
        self.vnext.result_scale = torch.nn.Linear(1, 1)
        self.vnext.recall_scale = torch.nn.Linear(1, 1)
        self.vnext.route_scale = torch.nn.Linear(1, 1)


def test_stage0_trainability_excludes_backbone_skill_table_and_stop() -> None:
    model = _DummyModel()
    report = configure_vnext_stage0(model)
    require_canonical_trainability(report)
    assert report["legacy_stop_trainable"] is False
    assert report["encoder_projection_trainable"] is True
    assert report["skill_embeddings_trainable"] is False
    assert model.skill_table.skill_bias_retr.requires_grad is False
    assert torch.count_nonzero(model.skill_table.skill_bias_retr) == 3
    assert torch.count_nonzero(model.skill_table.skill_bias_belief) == 3
    assert report["belief_skill_bias_trainable"] is True
    assert any(name.startswith("vnext.static_query") for name in report["trainable_parameter_names"])


def test_stage2_trainability_contains_recurrent_proposal_and_direct_router_modules() -> None:
    model = _DummyModel()
    report = configure_vnext_stage2(model)
    require_canonical_trainability(report)
    assert report["trainable_parameter_names"]
    assert all(name.startswith("vnext.") for name in report["trainable_parameter_names"])
    assert not any(
        name.startswith("vnext.static_route_query_delta")
        for name in report["trainable_parameter_names"]
    )
    assert any(
        name.startswith("vnext.unified_route_query")
        for name in report["trainable_parameter_names"]
    )
    assert any(
        name.startswith("vnext.route_skill_adapter")
        for name in report["trainable_parameter_names"]
    )
    assert any(
        name.startswith("vnext.route_expert_mixture")
        for name in report["trainable_parameter_names"]
    )
    assert not any(
        name.startswith("vnext.candidate_route_gate")
        for name in report["trainable_parameter_names"]
    )
    assert not any(
        name.startswith("vnext.candidate_compressor")
        or name.startswith("vnext.memory_route_query")
        for name in report["trainable_parameter_names"]
    )
    assert report["encoder_backbone_trainable"] is False
    assert report["encoder_projection_trainable"] is False


def test_candidate_compressor_trainability_isolated_from_stage0_and_memory() -> None:
    model = _DummyModel()
    static_before = static_foundation_digest(model)
    candidate_before = candidate_foundation_digest(model)
    report = configure_vnext_candidate_compressor(model)
    require_canonical_trainability(report)
    assert report["trainable_parameter_names"] == [
        "vnext.candidate_compressor.bias",
        "vnext.candidate_compressor.weight",
    ]
    with torch.no_grad():
        model.vnext.candidate_compressor.weight.add_(1.0)
    assert static_foundation_digest(model) == static_before
    assert candidate_foundation_digest(model) != candidate_before


def test_static_route_adapter_trainability_isolated_from_stage0_and_memory() -> None:
    model = _DummyModel()
    static_before = static_foundation_digest(model)
    candidate_before = candidate_foundation_digest(model)
    report = configure_vnext_static_route_adapter(model)
    require_canonical_trainability(report)
    assert report["trainable_parameter_names"] == [
        "vnext.static_route_query_delta.bias",
        "vnext.static_route_query_delta.weight",
    ]
    with torch.no_grad():
        model.vnext.static_route_query_delta.weight.add_(1.0)
    assert static_foundation_digest(model) == static_before
    assert candidate_foundation_digest(model) != candidate_before


def test_static_route_foundation_digest_binds_frozen_route_adapter_only() -> None:
    model = _DummyModel()
    route_before = static_route_foundation_digest(model)
    with torch.no_grad():
        model.vnext.unified_route_query.weight.add_(1.0)
        model.vnext.route_skill_adapter.weight.add_(1.0)
    assert static_route_foundation_digest(model) == route_before
    with torch.no_grad():
        model.vnext.static_route_query_delta.weight.add_(1.0)
    assert static_route_foundation_digest(model) != route_before


def test_candidate_foundation_digest_binds_new_route_expert_mixture() -> None:
    model = _DummyModel()
    candidate_before = candidate_foundation_digest(model)
    with torch.no_grad():
        model.vnext.route_expert_mixture.weight.add_(1.0)
    assert candidate_foundation_digest(model) != candidate_before


def test_vnext_checkpoint_state_excludes_legacy_heads() -> None:
    state = vnext_checkpoint_state(_DummyModel())
    assert "skill_table.E" in state
    assert any(name.startswith("vnext.") for name in state)
    assert not any(name.startswith("stop_head.") for name in state)
    assert not any(name.startswith("route_memory_utility_gate.") for name in state)
    assert not any(name.startswith("skill_table.encoder_fn.") for name in state)
    assert not any("backbone." in name for name in state)
    assert require_canonical_vnext_checkpoint_state(state)["status"] == "ok"


def test_vnext_checkpoint_state_rejects_nested_encoder_alias() -> None:
    with pytest.raises(ValueError, match="noncanonical"):
        require_canonical_vnext_checkpoint_state(
            {"skill_table.encoder_fn.backbone.weight": torch.ones(1)}
        )


def test_inventory_subset_recomputes_digest_and_rewrites_row_reference() -> None:
    catalogs = {
        "global": {
            "inventory_catalog_id": "global",
            "inventory_catalog_digest": "parent-digest",
            "runtime_visible_skill_ids": ["a", "b", "c"],
            "inventory_pool_size": 3,
        }
    }
    derived, mapping, report = derive_inventory_catalog_subset(catalogs, {"a", "c"})
    derived_id = mapping["global"]
    assert derived_id != "global"
    assert derived[derived_id]["runtime_visible_skill_ids"] == ["a", "c"]
    assert derived[derived_id]["inventory_catalog_digest"] != "parent-digest"
    assert derived[derived_id]["inventory_parent_catalog_digest"] == "parent-digest"
    assert report["derived_subset_catalog_count"] == 1

    rows = rewrite_inventory_catalog_references(
        [
            {
                "runtime_visible_catalog_id": "global",
                "inventory_catalog_digest": "parent-digest",
            }
        ],
        mapping,
        derived,
    )
    assert rows[0]["runtime_visible_catalog_id"] == derived_id
    assert rows[0]["inventory_catalog_digest"] == derived[derived_id][
        "inventory_catalog_digest"
    ]
    assert rows[0]["inventory_parent_catalog_id"] == "global"


def test_inventory_subset_preserves_identity_when_catalog_is_complete() -> None:
    catalogs = {
        "local": {
            "inventory_catalog_id": "local",
            "inventory_catalog_digest": "digest",
            "runtime_visible_skill_ids": ["a", "b"],
            "inventory_pool_size": 2,
        }
    }
    derived, mapping, report = derive_inventory_catalog_subset(catalogs, {"a", "b", "extra"})
    assert mapping == {"local": "local"}
    assert derived["local"]["inventory_catalog_digest"] == "digest"
    assert report["derived_subset_catalog_count"] == 0


class _CacheModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.proj.weight.copy_(torch.eye(2))
        self.encoder.backbone = torch.nn.Module()
        self.encoder.backbone.config = SimpleNamespace(hidden_size=2)
        self.encoder.encode_backbone_pooled = self._encode
        self.encoder.project_pooled = self.encoder.proj
        self.config = SimpleNamespace(
            base_model_name="fixture/frozen",
            d=2,
            encoder_pooling="last_token",
            torch_dtype="float32",
            max_length=32,
            state_query_prompt_version="fixture-v1",
            state_query_max_chars=128,
            state_query_truncation="head_tail_v1",
            frozen_backbone_snapshot_digest="fixture-snapshot",
        )
        self.encoded_texts: list[str] = []
        self.encoded_batches: list[list[str]] = []

    def _encode(self, texts: list[str]) -> torch.Tensor:
        self.encoded_texts.extend(texts)
        self.encoded_batches.append(list(texts))
        return torch.tensor(
            [[float(len(text)), float(sum(ord(char) for char in text) % 97)] for text in texts]
        )

    def encode_states(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_observations(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    @staticmethod
    def _serialize_state_for_encoder(text: str) -> str:
        return text


class _SkillCacheModel(_CacheModel):
    def __init__(self) -> None:
        super().__init__()
        self.config.skill_text_format = "fixture"
        self.skill_table = torch.nn.Module()
        self.skill_table.E = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.skill_table.skill_text_fn = lambda skill: str(skill["text"])
        self.skill_table._skill_payload = lambda skill: skill
        self.skill_table.encoder_fn = self._encode
        self._vnext_skill_embedding_cache = {"stale": torch.ones(1)}


def test_persistent_skill_cache_resumes_ordered_shards(tmp_path) -> None:
    model = _SkillCacheModel()
    skills = [
        {"skill_id": "a", "text": "alpha"},
        {"skill_id": "b", "text": "beta"},
    ]
    first = load_or_build_skill_embedding_cache(
        model,
        skills,
        cache_root=tmp_path,
        batch_size=1,
        cache_shard_size=2,
    )
    expected = model.skill_table.E.detach().clone()
    assert first["shard_count"] == 1
    assert model.encoded_texts == ["alpha", "beta"]
    assert model.encoded_batches == [["alpha"], ["beta"]]
    assert model._vnext_skill_embedding_cache == {}

    model.encoded_texts.clear()
    model.encoded_batches.clear()
    model.skill_table.E.data.zero_()
    second = load_or_build_skill_embedding_cache(
        model,
        skills,
        cache_root=tmp_path,
        batch_size=1,
        cache_shard_size=2,
    )
    assert second["cache_identity"] == first["cache_identity"]
    assert model.encoded_texts == []
    assert model.encoded_batches == []
    assert torch.equal(model.skill_table.E, expected)


def test_persistent_frozen_cache_reuses_hits_and_extends_only_missing_texts(tmp_path) -> None:
    model = _CacheModel()
    first = load_or_build_frozen_text_cache(
        model,
        ["alpha", "beta", "alpha"],
        role="state",
        batch_size=2,
        cache_root=tmp_path,
    )
    assert first.persistent_hit_count == 0
    assert first.persistent_miss_count == 2
    assert model.encoded_texts == ["alpha", "beta"]
    first_manifest = Path(first.cache_tensor_path)
    assert first_manifest.name == "state.index.json"
    assert "alpha" not in first_manifest.read_text(encoding="utf-8")
    assert first.report()["shard_count"] == 1
    assert first.report()["online_projection"] is True

    model.encoded_texts.clear()
    second = load_or_build_frozen_text_cache(
        model,
        ["beta", "gamma"],
        role="state",
        batch_size=2,
        cache_root=tmp_path,
    )
    assert second.persistent_hit_count == 1
    assert second.persistent_miss_count == 1
    assert model.encoded_texts == ["gamma"]
    assert second.report()["shard_count"] == 2
    assert torch.equal(
        second.batch(["alpha", "gamma"], device=torch.device("cpu")),
        torch.tensor([[5.0, 33.0], [5.0, 30.0]]),
    )
    projected = second.batch(["alpha"], device=torch.device("cpu"))
    projected.sum().backward()
    assert model.encoder.proj.weight.grad is not None
    pooled = second.pooled_batch(["alpha"], device=torch.device("cpu"))
    assert torch.equal(pooled, torch.tensor([[5.0, 33.0]]))
    alternate = second.batch_with_projection(
        ["alpha"],
        device=torch.device("cpu"),
        projection_fn=lambda value: 2.0 * value,
    )
    assert torch.equal(alternate, torch.tensor([[10.0, 66.0]]))

    model.encoded_texts.clear()
    third = load_or_build_frozen_text_cache(
        model,
        ["gamma", "alpha"],
        role="state",
        batch_size=1,
        cache_root=tmp_path,
    )
    assert third.persistent_hit_count == 2
    assert third.persistent_miss_count == 0
    assert model.encoded_texts == []


def test_read_only_frozen_cache_reuses_complete_parent_without_writes(tmp_path) -> None:
    model = _CacheModel()
    built = load_or_build_frozen_text_cache(
        model,
        ["alpha", "beta"],
        role="state",
        batch_size=2,
        cache_root=tmp_path,
    )
    manifest_path = Path(built.cache_tensor_path)
    before = manifest_path.read_bytes()
    model.encoded_texts.clear()
    loaded = load_frozen_text_cache_read_only(
        model,
        ["beta", "alpha"],
        role="state",
        cache_root=tmp_path,
    )
    assert loaded.persistent_hit_count == 2
    assert loaded.persistent_miss_count == 0
    assert model.encoded_texts == []
    assert manifest_path.read_bytes() == before
    with pytest.raises(RuntimeError, match="read-only frozen text cache miss"):
        load_frozen_text_cache_read_only(
            model,
            ["gamma"],
            role="state",
            cache_root=tmp_path,
        )


def test_resume_contract_is_order_stable_and_fails_on_semantic_change() -> None:
    left = immutable_run_contract(
        {"inputs": {"b": "2", "a": "1"}, "optimization": {"lr": 1.0e-4}}
    )
    right = immutable_run_contract(
        {"optimization": {"lr": 1.0e-4}, "inputs": {"a": "1", "b": "2"}}
    )
    assert left == right
    require_matching_run_contract(left, right)

    changed = immutable_run_contract(
        {"inputs": {"a": "1", "b": "2"}, "optimization": {"lr": 2.0e-4}}
    )
    try:
        require_matching_run_contract(left, changed)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("semantic resume mismatch must fail closed")


def test_verified_data_contract_binds_path_and_digest(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"row": 1}\n', encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(rows.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "status": "ok",
                "schema_version": "fixture",
                "files": {"rows": {"path": str(rows), "sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "selected_rows.jsonl"
    alias.write_bytes(rows.read_bytes())
    report = require_verified_data_contract(manifest, {"rows": alias})
    assert report["status"] == "ok"
    assert report["verified_files"]["rows"]["content_alias"] is True
    alias.write_text('{"row": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        require_verified_data_contract(manifest, {"rows": alias})
