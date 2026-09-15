import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.data import validate_data_roles
from clstr.skillret import import_skillret_dataset
from clstr.retrieval_warmup import (
    _batch_queries_for_step,
    _build_source_buckets,
    _filter_queries_to_skill_pool,
    _skill_pool_alias_positive_ids_by_skill_id,
    load_unified_v2_retrieval_rows,
    run_skillret_retrieval_warmup,
)
from clstr.skillrouter_style import (
    export_skillrouter_style_official_run,
    run_skillrouter_style_finetune,
)
from clstr.clstr_retrieval_adapters import (
    CLSTRResidualListwiseRerankHead,
    _score_residual_rerank_candidates,
    build_listwise_rerank_candidates,
    export_clstr_qdoc_official_run,
    run_clstr_qdoc_rerank_warmup,
    run_clstr_qdoc_warmup,
)
from clstr.native_rerank import (
    CLSTRNativeResidualRerankHead,
    build_clstr_native_rerank_decision_report,
    build_native_routing_init_manifest,
    export_clstr_native_rerank_official_run,
    run_clstr_native_rerank_warmup,
    score_native_residual_rerank_candidates,
    write_clstr_native_rerank_design_report,
)
from clstr.skillret_official import (
    build_official_protocol_report,
    evaluate_official_run,
    export_lexical_official_run,
    resolve_clstr_export_config,
    write_skillret_serialization_report,
    write_official_comparison_table,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_skillret_source(root: Path) -> None:
    write_jsonl(
        root / "data" / "skills" / "train.jsonl",
        [
            {
                "id": "skill-a",
                "name": "spreadsheet-cleaner",
                "description": "Clean spreadsheet data",
                "skill_md": "# Spreadsheet Cleaner\nClean data.",
                "major": "data",
            },
            {
                "id": "skill-b",
                "name": "chart-maker",
                "description": "Make charts",
                "skill_md": "# Chart Maker\nDraw charts.",
                "major": "data",
            },
        ],
    )
    write_jsonl(
        root / "data" / "skills" / "test.jsonl",
        [
            {
                "id": "skill-c",
                "name": "pdf-helper",
                "description": "Work with PDFs",
                "skill_md": "# PDF Helper\nProcess PDFs.",
                "major": "documents",
            }
        ],
    )
    write_jsonl(
        root / "data" / "queries" / "train.jsonl",
        [
            {
                "id": "q-train",
                "query": "clean this spreadsheet",
                "skill_ids": ["skill-a"],
                "skill_names": ["spreadsheet-cleaner"],
                "k": 1,
            }
        ],
    )
    write_jsonl(
        root / "data" / "queries" / "test.jsonl",
        [
            {
                "id": "q-test",
                "query": "extract tables from a pdf",
                "skill_ids": ["skill-c"],
                "skill_names": ["pdf-helper"],
                "k": 1,
            }
        ],
    )
    write_jsonl(
        root / "data" / "qrels" / "train.jsonl",
        [{"query_id": "q-train", "skill_id": "skill-a", "relevance": 1}],
    )
    write_jsonl(
        root / "data" / "qrels" / "test.jsonl",
        [{"query_id": "q-test", "skill_id": "skill-c", "relevance": 1}],
    )


def test_import_skillret_dataset_emits_clstr_retrieval_files_and_reports(tmp_path):
    source = tmp_path / "skillret_source"
    output = tmp_path / "clstr_skillret"
    report_dir = tmp_path / "reports"
    make_skillret_source(source)

    manifest = import_skillret_dataset(
        source_root=source,
        output_dir=output,
        report_dir=report_dir,
    )

    assert manifest["source_dataset"] == "ThakiCloud/SKILLRET"
    assert manifest["data_role"] == "retrieval_pretraining_not_skillsbench_trajectory"
    assert manifest["splits"]["train"]["skill_count"] == 2
    assert manifest["splits"]["test"]["query_count"] == 1
    assert manifest["record_counts"]["qrels"] == 2
    assert (output / "skills.jsonl").exists()
    assert (output / "queries.jsonl").exists()
    assert (output / "qrels.jsonl").exists()
    assert (output / "manifest.json").exists()
    schema = json.loads((report_dir / "schema_report.json").read_text(encoding="utf-8"))
    assert schema["subsets"]["skills"]["train"]["fields"] == [
        "description",
        "id",
        "major",
        "name",
        "skill_md",
    ]


def test_data_roles_support_skillret_retrieval_without_treating_it_as_clean_router(tmp_path):
    root = tmp_path / "skillret"
    root.mkdir()
    for name in ["skills.jsonl", "queries.jsonl", "qrels.jsonl"]:
        (root / name).write_text("{}\n", encoding="utf-8")
    cfg = {
        "skillrouter_eval_root": str(tmp_path / "skillrouter_eval_core"),
        "skillsbench_root": str(tmp_path / "skillsbench"),
        "alfworld_root": str(tmp_path / "alfworld"),
        "leakage_audit_dir": str(tmp_path / "audit"),
        "clean_router_data_root": str(tmp_path / "clean_router"),
        "skillret_root": str(tmp_path / "raw_skillret"),
        "retrieval_warmup_data_root": str(root),
        "training_data_source": "skillret_retrieval",
    }

    roles = validate_data_roles(cfg, require_training_data=True)

    assert roles["skillret"]["role"] == "retrieval-pretraining-source"
    assert roles["retrieval_warmup_data"]["role"] == "training-retrieval-warmup-data"
    assert roles["training"]["source"] == "skillret_retrieval"


def test_skillret_retrieval_warmup_writes_checkpoint_and_report(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "warmup"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    write_jsonl(
        source / "data" / "skills" / "test.jsonl",
        [
            {
                "id": "skill-negative-first",
                "name": "unrelated",
                "description": "Unrelated first skill",
                "skill_md": "# Unrelated\nDo something else.",
                "major": "misc",
            },
            {
                "id": "skill-positive-second",
                "name": "pdf-helper",
                "description": "Work with PDFs",
                "skill_md": "# PDF Helper\nProcess PDFs.",
                "major": "documents",
            },
        ],
    )
    write_jsonl(
        source / "data" / "qrels" / "test.jsonl",
        [{"query_id": "q-test", "skill_id": "skill-positive-second", "relevance": 1}],
    )
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=16,
        top_k=2,
        max_skills=2,
        max_queries=1,
    )

    assert report["status"] == "ok"
    assert report["training_data_source"] == "skillret_retrieval"
    assert report["steps"] == 1
    assert report["metrics"]["recall_at_1"] >= 0.0
    assert report["metrics"]["reranker_loss"] >= 0.0
    assert Path(report["checkpoint"]).exists()
    assert (output_dir / "train_report.json").exists()


def test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream(tmp_path):
    data_root = tmp_path / "unified_v2"
    output_dir = tmp_path / "warmup"
    model_dir = tmp_path / "tiny-model"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolret/weather",
                "name": "Get Weather",
                "description": "Returns weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/news",
                "name": "News Search",
                "description": "Searches news by query.",
                "input_schema": {"query": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "q-weather",
                "query_text": "Need the current weather in Paris.",
                "positive_skill_id": "toolret/weather",
                "negative_skill_ids": ["toolret/news"],
            },
            {
                "source": "toolbench_g3",
                "query_id": "q-news",
                "query_text": "Find recent news about birthdays.",
                "positive_skill_id": "toolret/news",
                "negative_skill_ids": ["toolret/weather"],
            },
        ],
    )
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=16,
        top_k=2,
        max_skills=2,
        max_queries=1,
        data_format="unified_v2",
        use_cross_encoder=False,
    )

    assert report["status"] == "ok"
    assert report["training_data_source"] == "clstr_unified_retrieval_v2"
    assert report["data_format"] == "unified_v2"
    assert report["skill_count"] == 2
    assert report["query_count"] == 1
    assert report["split_protocol"]["query_count_before_max_queries"] == 2
    assert report["trainable_parameter_policy"]["skill_table_retrieval_path_trainable"] is True
    assert report["trainable_parameter_policy"]["trains_skill_table_E"] is True
    assert report["trainable_parameter_policy"]["trains_retrieval_scale_and_bias"] is True
    assert "skill_table.E" in report["trainable_parameter_policy"]["optimizer_parameter_names"]
    assert "skill_table.logit_scale_retr" in report["trainable_parameter_policy"]["optimizer_parameter_names"]
    assert "skill_table.skill_bias_retr" in report["trainable_parameter_policy"]["optimizer_parameter_names"]
    assert "skill_head" not in " ".join(report["trainable_parameter_policy"]["optimizer_parameter_names"])
    assert Path(report["checkpoint"]).exists()
    assert Path(report["latest_checkpoint"]).exists()
    assert Path(report["training_metrics_path"]).exists()
    assert Path(report["loss_curve_path"]).exists()
    assert Path(report["setup_status_path"]).exists()
    setup_events = [
        json.loads(line)
        for line in Path(report["setup_status_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    setup_phases = [event["phase"] for event in setup_events]
    expected_setup_phases = [
        "data_loaded",
        "selected_skills_written",
        "model_initialized",
        "skill_table_rebuild_started",
        "skill_table_rebuild_progress",
        "skill_table_rebuilt",
        "training_started",
    ]
    positions = [setup_phases.index(phase) for phase in expected_setup_phases]
    assert positions == sorted(positions)
    progress_events = [event for event in setup_events if event["phase"] == "skill_table_rebuild_progress"]
    assert progress_events
    assert progress_events[-1]["encoded_skill_count"] == report["skill_count"]
    metric_lines = [
        json.loads(line)
        for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step"] for row in metric_lines] == [1]
    latest_payload = torch.load(report["latest_checkpoint"], map_location="cpu")
    assert latest_payload["step"] == 1
    payload = torch.load(report["checkpoint"], map_location="cpu")
    assert payload["train_safety"]["unified_v2_train_safe_retrieval"] is True
    assert "public_train_or_eval_unlabeled" in payload["train_safety"]["excluded_retrieval_splits"]
    assert payload["checkpoint_excludes_frozen_backbone"] is True
    assert "skill_table.E" in payload["model_state_dict"]
    assert "skill_table.W.weight" in payload["model_state_dict"]
    assert "encoder.proj.weight" in payload["model_state_dict"]
    assert all(not key.startswith("encoder.backbone.") for key in payload["model_state_dict"])
    assert all(not key.startswith("skill_table.encoder_fn.") for key in payload["model_state_dict"])


def test_retrieval_warmup_can_initialize_from_stage0_checkpoint(tmp_path):
    data_root = tmp_path / "unified_v2"
    first_output_dir = tmp_path / "warmup_first"
    second_output_dir = tmp_path / "warmup_second"
    model_dir = tmp_path / "tiny-model"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolret/weather",
                "name": "Get Weather",
                "description": "Returns weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/news",
                "name": "News Search",
                "description": "Searches news by query.",
                "input_schema": {"query": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "q-weather",
                "query_text": "Need the current weather in Paris.",
                "positive_skill_id": "toolret/weather",
                "negative_skill_ids": ["toolret/news"],
            }
        ],
    )
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    first_report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=first_output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=16,
        top_k=2,
        max_skills=2,
        max_queries=1,
        data_format="unified_v2",
        use_cross_encoder=False,
    )
    init_payload = torch.load(first_report["checkpoint"], map_location="cpu")
    init_payload["model_state_dict"]["skill_table.skill_bias_retr"] = torch.full((2,), 2.5)
    init_checkpoint = tmp_path / "init_stage0.pt"
    torch.save(init_payload, init_checkpoint)

    second_report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=second_output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=16,
        top_k=2,
        max_skills=2,
        max_queries=1,
        data_format="unified_v2",
        use_cross_encoder=False,
        learning_rate=0.0,
        init_checkpoint_path=init_checkpoint,
    )

    assert second_report["init_checkpoint"]["loaded"] is True
    assert second_report["init_checkpoint"]["path"] == str(init_checkpoint)
    second_payload = torch.load(second_report["checkpoint"], map_location="cpu")
    assert torch.equal(
        second_payload["model_state_dict"]["skill_table.skill_bias_retr"],
        torch.full((2,), 2.5),
    )


def test_stage0_source_balanced_sampling_keeps_trajectory_sources_in_batches():
    queries = [
        {"query_id": f"toolret-{idx}", "metadata": {"source": "toolret_training"}}
        for idx in range(6)
    ] + [
        {"query_id": "alf-0", "metadata": {"source": "trajectory_derived_alfworld"}},
        {"query_id": "sci-0", "metadata": {"source": "trajectory_derived_scienceworld"}},
    ]

    batch_stride = _batch_queries_for_step(queries, step=1, batch_size=4)
    source_balanced = _batch_queries_for_step(
        queries,
        step=1,
        batch_size=4,
        sampling_strategy="source_balanced",
        source_buckets=_build_source_buckets(queries),
    )

    assert [row["query_id"] for row in batch_stride] == [
        "toolret-0",
        "toolret-1",
        "toolret-2",
        "toolret-3",
    ]
    assert {
        row["metadata"]["source"]
        for row in source_balanced
    } == {
        "toolret_training",
        "trajectory_derived_alfworld",
        "trajectory_derived_scienceworld",
    }


def test_retrieval_warmup_multi_positive_loss_does_not_penalize_equivalent_positive(tmp_path, monkeypatch):
    import clstr.model as model_module

    data_root = tmp_path / "unified_v2"
    output_dir = tmp_path / "warmup"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "skill/a", "name": "A", "description": "A"},
            {"skill_id": "skill/b", "name": "B", "description": "B"},
            {"skill_id": "skill/c", "name": "C", "description": "C"},
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "q-multi",
                "query_text": "equivalent positive query",
                "positive_skill_id": "skill/a",
            },
            {
                "source": "toolret_training",
                "query_id": "q-multi",
                "query_text": "equivalent positive query",
                "positive_skill_id": "skill/b",
            },
        ],
    )

    class FixedEncoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def forward(self, texts):
            rows = []
            for text in texts:
                if "query" in text:
                    rows.append([0.0, 1.0])
                elif "B" in text:
                    rows.append([0.0, 1.0])
                elif "C" in text:
                    rows.append([0.0, -1.0])
                else:
                    rows.append([1.0, 0.0])
            return torch.tensor(rows, dtype=torch.float32)

    monkeypatch.setattr(model_module, "StateEncoder", FixedEncoder)
    monkeypatch.setattr(model_module, "CrossEncoder", FixedEncoder)

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name="unused",
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=2,
        top_k=3,
        max_skills=3,
        max_queries=1,
        data_format="unified_v2",
        use_cross_encoder=False,
        skill_table_adapter_init="identity",
        normalize_embeddings=True,
        retrieval_loss_mode="multi_positive_nll",
        train_skill_embeddings=False,
        train_skill_bias=False,
        train_encoder_projection=False,
        train_skill_adapter=False,
        train_retrieval_scale=True,
    )

    metrics = json.loads(Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()[-1])
    assert metrics["positive_label_counts"] == [2]
    assert metrics["retrieval_loss_mode"] == "multi_positive_nll"
    assert metrics["recall_at_1"] == 1.0
    assert metrics["retriever_loss"] < 0.1
    assert report["trainable_parameter_policy"]["trains_skill_table_E"] is False
    assert report["trainable_parameter_policy"]["trains_skill_bias"] is False


def test_stage0_filter_queries_expands_skill_pool_alias_positives():
    skill_rows = [
        {
            "skill_id": "toolbench-g3/weather/get-weather",
            "canonical_skill_id": "toolbench-g3/weather/get-weather",
            "alias_skill_ids": [
                "toolbench-g3/weather/get-weather",
                "toolret/get-weather",
                "raw/weather",
            ],
        },
        {
            "skill_id": "toolret/get-weather",
            "canonical_skill_id": "toolbench-g3/weather/get-weather",
            "alias_skill_ids": [
                "toolbench-g3/weather/get-weather",
                "toolret/get-weather",
                "raw/weather",
            ],
        },
        {
            "skill_id": "toolret/news-search",
            "canonical_skill_id": "toolret/news-search",
            "alias_skill_ids": ["toolret/news-search"],
        },
    ]
    skill_id_to_idx = {row["skill_id"]: idx for idx, row in enumerate(skill_rows)}
    alias_positives = _skill_pool_alias_positive_ids_by_skill_id(skill_rows)

    queries = [
        {
            "query_id": "q-weather",
            "query": "Need weather.",
            "positive_skill_ids": ["raw/weather"],
        }
    ]
    usable = _filter_queries_to_skill_pool(
        queries,
        {"q-weather": ["raw/weather"]},
        skill_id_to_idx,
        max_queries=None,
        alias_positive_ids_by_skill_id=alias_positives,
    )

    assert len(usable) == 1
    assert usable[0]["positive_indices"] == [0, 1]
    assert usable[0]["resolved_positive_skill_ids"] == [
        "toolbench-g3/weather/get-weather",
        "toolret/get-weather",
    ]
    assert usable[0]["raw_positive_skill_ids"] == ["raw/weather"]
    assert usable[0]["stage0_alias_positive_expanded"] is True


def test_retrieval_warmup_accumulates_gradients_inside_each_optimizer_step(tmp_path, monkeypatch):
    import clstr.model as model_module

    data_root = tmp_path / "unified_v2"
    output_dir = tmp_path / "warmup"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "skill/0", "name": "Skill Zero", "description": "Zero"},
            {"skill_id": "skill/1", "name": "Skill One", "description": "One"},
            {"skill_id": "skill/2", "name": "Skill Two", "description": "Two"},
            {"skill_id": "skill/3", "name": "Skill Three", "description": "Three"},
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": f"q-{idx}",
                "query_text": f"query {idx}",
                "positive_skill_id": f"skill/{idx}",
            }
            for idx in range(4)
        ],
    )

    class FixedEncoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, texts):
            rows = []
            for text in texts:
                if "1" in text or "One" in text:
                    rows.append([0.0, 1.0, 0.0, 0.0])
                elif "2" in text or "Two" in text:
                    rows.append([0.0, 0.0, 1.0, 0.0])
                elif "3" in text or "Three" in text:
                    rows.append([0.0, 0.0, 0.0, 1.0])
                else:
                    rows.append([1.0, 0.0, 0.0, 0.0])
            return torch.tensor(rows, dtype=torch.float32)

    step_calls = []
    original_adamw = torch.optim.AdamW

    class CountingAdamW(original_adamw):
        def step(self, *args, **kwargs):
            step_calls.append(len(step_calls) + 1)
            return super().step(*args, **kwargs)

    monkeypatch.setattr(model_module, "StateEncoder", FixedEncoder)
    monkeypatch.setattr(model_module, "CrossEncoder", FixedEncoder)
    monkeypatch.setattr(torch.optim, "AdamW", CountingAdamW)

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name="unused",
        output_dir=output_dir,
        max_steps=2,
        batch_size=1,
        gradient_accumulation_steps=2,
        model_dim=4,
        top_k=4,
        max_skills=4,
        max_queries=4,
        data_format="unified_v2",
        use_cross_encoder=False,
        skill_table_adapter_init="identity",
        normalize_embeddings=True,
        train_skill_embeddings=False,
        train_skill_bias=False,
        train_encoder_projection=False,
        train_skill_adapter=True,
        train_retrieval_scale=True,
        shuffle_queries=False,
    )

    metrics = [
        json.loads(line)
        for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(step_calls) == 2
    assert [row["optimizer_step"] for row in metrics] == [1, 2]
    assert [row["query_ids"] for row in metrics] == [["q-0", "q-1"], ["q-2", "q-3"]]
    assert report["gradient_accumulation_steps"] == 2
    assert report["effective_batch_size"] == 2


def test_retrieval_warmup_batches_advance_by_batch_stride_for_full_corpus_coverage(tmp_path):
    data_root = tmp_path / "unified_v2"
    output_dir = tmp_path / "warmup"
    model_dir = tmp_path / "tiny-model"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": f"skill-{idx}",
                "name": f"Skill {idx}",
                "description": f"Handles task family {idx}.",
            }
            for idx in range(8)
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training" if idx < 4 else "toolbench_g3",
                "query_id": f"q-{idx}",
                "query_text": f"Need skill {idx}.",
                "positive_skill_id": f"skill-{idx}",
                "negative_skill_ids": [],
                "provenance": {"split": "train"},
            }
            for idx in range(8)
        ],
    )
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=4,
        batch_size=2,
        model_dim=16,
        top_k=2,
        max_skills=8,
        max_queries=None,
        data_format="unified_v2",
        use_cross_encoder=False,
        shuffle_queries=False,
    )

    metrics = [
        json.loads(line)
        for line in Path(report["training_metrics_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["query_ids"] for row in metrics] == [
        ["q-0", "q-1"],
        ["q-2", "q-3"],
        ["q-4", "q-5"],
        ["q-6", "q-7"],
    ]
    assert report["sampling_strategy"] == "batch_stride"


def test_unified_v2_retrieval_loader_reports_corrupt_jsonl_line(tmp_path):
    from clstr.retrieval_warmup import load_unified_v2_retrieval_rows

    data_root = tmp_path / "unified_v2"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [{"skill_id": "tool/a", "name": "A", "description": "A"}],
    )
    (data_root / "retrieval.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-ok",
                "query_text": "ok",
                "positive_skill_id": "tool/a",
            }
        )
        + "\n"
        + '{"query_id": "q-bad", "query_text": "unterminated\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"retrieval\.jsonl line 2"):
        load_unified_v2_retrieval_rows(data_root)


def test_unified_v2_retrieval_loader_excludes_public_unlabeled_rows_by_default(tmp_path):
    from clstr.retrieval_warmup import load_unified_v2_retrieval_rows

    data_root = tmp_path / "unified_v2"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "toolret/weather", "name": "weather", "description": "weather"},
            {"skill_id": "traject/weather", "name": "traject weather", "description": "traject weather"},
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "q-train",
                "query_text": "weather query",
                "positive_skill_id": "toolret/weather",
                "provenance": json.dumps({"split": "train"}),
            },
            {
                "source": "traject_bench",
                "query_id": "q-public",
                "query_text": "public benchmark query",
                "positive_skill_id": "traject/weather",
                "provenance": json.dumps({"split": "public_train_or_eval_unlabeled"}),
            },
        ],
    )

    _skills, queries, positives = load_unified_v2_retrieval_rows(data_root)

    assert [row["query_id"] for row in queries] == ["q-train"]
    assert positives == {"q-train": ["toolret/weather"]}
    assert queries[0]["metadata"]["source"] == "toolret_training"


def test_unified_v2_retrieval_loader_keeps_duplicate_query_from_different_source_as_separate_training_row(tmp_path):
    data_root = tmp_path / "unified_v2"
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "toolret/weather", "name": "Weather", "description": "Weather API"},
            {"skill_id": "toolret/news", "name": "News", "description": "News API"},
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "q-shared",
                "query_text": "weather query",
                "positive_skill_id": "toolret/weather",
                "provenance": {"split": "train"},
            },
            {
                "source": "toolret_training_stage0_balanced_missing_positive",
                "query_id": "q-shared",
                "query_text": "weather query",
                "positive_skill_id": "toolret/weather",
                "provenance": {"split": "train"},
            },
        ],
    )

    _skills, queries, positives = load_unified_v2_retrieval_rows(data_root)

    assert len(queries) == 2
    assert [row["metadata"]["source"] for row in queries] == [
        "toolret_training",
        "toolret_training_stage0_balanced_missing_positive",
    ]
    assert queries[1]["metadata"]["original_query_id"] == "q-shared"
    assert queries[1]["query_id"] != "q-shared"
    assert positives[queries[1]["query_id"]] == ["toolret/weather"]


def test_skillret_retrieval_warmup_can_use_protocol_train_split(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "warmup"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    write_jsonl(
        source / "data" / "queries" / "train.jsonl",
        [
            {
                "id": "q-keep",
                "query": "clean spreadsheet",
                "skill_ids": ["skill-a"],
                "skill_names": ["spreadsheet-cleaner"],
                "k": 1,
            },
            {
                "id": "q-dev",
                "query": "make a chart",
                "skill_ids": ["skill-b"],
                "skill_names": ["chart-maker"],
                "k": 1,
            },
        ],
    )
    write_jsonl(
        source / "data" / "qrels" / "train.jsonl",
        [
            {"query_id": "q-keep", "skill_id": "skill-a", "relevance": 1},
            {"query_id": "q-dev", "skill_id": "skill-b", "relevance": 1},
        ],
    )
    import_skillret_dataset(source, data_root, report_dir)
    (data_root / "splits.json").write_text(
        json.dumps(
            {
                "split_strategy": "test",
                "splits": {
                    "train": {"query_ids": ["q-keep"]},
                    "dev": {"query_ids": ["q-dev"]},
                    "test": {"query_ids": ["q-test"]},
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=16,
        top_k=2,
        max_skills=2,
        max_queries=None,
        split_path=data_root / "splits.json",
        train_split="train",
    )

    assert report["query_count"] == 1
    assert report["split_protocol"]["train_split"] == "train"
    assert report["split_protocol"]["query_count_before_max_queries"] == 1


def test_skillrouter_style_finetune_writes_adapter_checkpoint_without_clstr_heads(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "skillrouter_style"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_skillrouter_style_finetune(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
        temperature=0.05,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_style_finetune"
    assert report["training_objective"] == "full_pool_infonce"
    assert report["uses_clstr_heads"] is False
    assert report["uses_skillrouter_eval_labels"] is False
    assert report["temperature"] == 0.05
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.exists()
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["method"] == "skillrouter_style_finetune"
    assert payload["source_note"].startswith("CLSTR-side SkillRouter-style")
    assert "q_proj.weight" in payload["adapter_state_dict"]
    assert not any(key.startswith("belief") or key.startswith("transition") for key in payload["adapter_state_dict"])


def test_skillrouter_style_export_subset_does_not_peek_at_test_qrels(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    train_dir = tmp_path / "skillrouter_style"
    eval_dir = tmp_path / "official_eval" / "skillrouter_style_subset"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    write_jsonl(
        source / "data" / "skills" / "test.jsonl",
        [
            {
                "id": "skill-negative-first",
                "name": "unrelated",
                "description": "Unrelated first skill",
                "skill_md": "# Unrelated\nDo something else.",
                "major": "misc",
            },
            {
                "id": "skill-positive-second",
                "name": "pdf-helper",
                "description": "Work with PDFs",
                "skill_md": "# PDF Helper\nProcess PDFs.",
                "major": "documents",
            },
        ],
    )
    write_jsonl(
        source / "data" / "qrels" / "test.jsonl",
        [{"query_id": "q-test", "skill_id": "skill-positive-second", "relevance": 1}],
    )
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    train_report = run_skillrouter_style_finetune(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=train_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
    )

    run_report = export_skillrouter_style_official_run(
        data_root=data_root,
        base_model_name=str(model_dir),
        checkpoint_path=train_report["checkpoint"],
        output_dir=eval_dir,
        split="test",
        top_k=1,
        batch_size=2,
        max_length=128,
        max_skills=1,
    )

    predictions = (eval_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert run_report["skill_pool_selection"] == "first_n_no_qrel_peek"
    assert "skill-negative-first" in predictions
    assert "skill-positive-second" not in predictions


def test_skillrouter_style_export_uses_official_test_protocol(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    train_dir = tmp_path / "skillrouter_style"
    eval_dir = tmp_path / "official_eval" / "skillrouter_style_finetune"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    train_report = run_skillrouter_style_finetune(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=train_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
    )

    run_report = export_skillrouter_style_official_run(
        data_root=data_root,
        base_model_name=str(model_dir),
        checkpoint_path=train_report["checkpoint"],
        output_dir=eval_dir,
        split="test",
        top_k=15,
        batch_size=2,
        max_length=128,
    )
    metrics = evaluate_official_run(data_root, run_report["run_path"], eval_dir, split="test")

    assert run_report["method"] == "skillrouter_style_finetune"
    assert run_report["split"] == "test"
    assert run_report["skill_pool_selection"] == "all"
    assert metrics["protocol"] == "official_skillret_pytrec_eval"
    assert (eval_dir / "run.tsv").exists()
    assert (eval_dir / "predictions.jsonl").exists()


def test_clstr_qdoc_warmup_writes_adapter_checkpoint(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "clstr_qdoc"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    report = run_clstr_qdoc_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=output_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
        temperature=0.05,
    )

    assert report["status"] == "ok"
    assert report["method"] == "clstr_qdoc"
    assert report["training_objective"] == "full_pool_infonce"
    assert report["routing_foundation_role"] == "static_retrieval_not_closed_loop_core"
    assert report["uses_skillret_test_qrels"] is False
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.exists()
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["method"] == "clstr_qdoc"
    assert "q_adapter.weight" in payload["adapter_state_dict"]
    assert "d_adapter.weight" in payload["adapter_state_dict"]
    assert payload["source_note"].startswith("CLSTR q/doc adapter")


def test_clstr_qdoc_export_uses_official_test_protocol(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    train_dir = tmp_path / "clstr_qdoc"
    eval_dir = tmp_path / "official_eval" / "clstr_qdoc"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    train_report = run_clstr_qdoc_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=train_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
    )

    run_report = export_clstr_qdoc_official_run(
        data_root=data_root,
        base_model_name=str(model_dir),
        checkpoint_path=train_report["checkpoint"],
        output_dir=eval_dir,
        split="test",
        top_k=15,
        batch_size=2,
        max_length=128,
        run_name="clstr_qdoc_full",
    )
    metrics = evaluate_official_run(data_root, run_report["run_path"], eval_dir, split="test")

    assert run_report["method"] == "clstr_qdoc"
    assert run_report["split"] == "test"
    assert run_report["skill_pool_selection"] == "all"
    assert metrics["protocol"] == "official_skillret_pytrec_eval"
    assert (eval_dir / "run.tsv").exists()
    assert (eval_dir / "predictions.jsonl").exists()


def test_clstr_listwise_rerank_candidates_filter_false_negatives():
    query_ids = ["q1"]
    ranked_by_query = {
        "q1": [
            ("skill-a", 1.0),
            ("skill-b", 0.9),
            ("skill-c", 0.8),
            ("skill-d", 0.7),
        ]
    }
    positives_by_query = {"q1": ["skill-a", "skill-b"]}

    candidates = build_listwise_rerank_candidates(
        query_ids=query_ids,
        ranked_by_query=ranked_by_query,
        positives_by_query=positives_by_query,
        candidate_k=3,
    )

    assert len(candidates) == 1
    row = candidates[0]
    assert row["positive_skill_id"] == "skill-a"
    assert row["candidate_skill_ids"][0] == "skill-a"
    assert "skill-b" not in row["candidate_skill_ids"][1:]
    assert row["candidate_skill_ids"][1:] == ["skill-c", "skill-d"]


def test_clstr_qdoc_rerank_warmup_writes_listwise_checkpoint(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    qdoc_dir = tmp_path / "clstr_qdoc"
    rerank_dir = tmp_path / "clstr_qdoc_rerank"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    qdoc_report = run_clstr_qdoc_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=qdoc_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_batch_size=2,
    )

    report = run_clstr_qdoc_rerank_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        qdoc_checkpoint_path=qdoc_report["checkpoint"],
        output_dir=rerank_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        max_skills=2,
        max_queries=1,
        candidate_k=2,
        max_length=128,
        encode_batch_size=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "clstr_qdoc_rerank"
    assert report["training_objective"] == "residual_listwise_cross_entropy_over_retriever_topk"
    assert report["false_negative_filtering"] == "exclude_all_query_positive_skill_ids_from_negatives"
    assert report["uses_skillret_test_qrels"] is False
    assert report["model_config"]["reranker_type"] == "residual_embedding_pair_mlp_head"
    assert report["model_config"]["base_score_temperature"] == 0.05
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.exists()


def test_clstr_native_rerank_design_report_keeps_qdoc_out_of_mainline(tmp_path):
    report = write_clstr_native_rerank_design_report(tmp_path / "design_report.json")

    assert report["status"] == "ok"
    assert report["native_routing_components"]["state_query_adapter"] == "StateEncoder.proj"
    assert report["native_routing_components"]["skill_query_adapter"] == "SkillTable.W"
    assert report["native_routing_components"]["doc_side_table"] == "SkillTable.E"
    assert report["qdoc_adapter_policy"]["enters_mainline"] is False
    assert report["loss_taxonomy"]["native_residual_listwise_rerank"] == "L_retr"


def test_native_residual_reranker_zero_delta_preserves_native_scores():
    import torch

    reranker = CLSTRNativeResidualRerankHead(d=2)
    query_embs = torch.tensor([[1.0, 0.0]])
    skill_embs = torch.tensor([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]])
    candidate_indices = torch.tensor([[0, 1, 2]])
    base_scores = torch.tensor([[0.9, 0.3, 0.1]])

    scores = score_native_residual_rerank_candidates(
        reranker=reranker,
        query_embs=query_embs,
        skill_embs=skill_embs,
        candidate_indices=candidate_indices,
        base_scores=base_scores,
        device=torch.device("cpu"),
        base_score_temperature=1.0,
    )

    assert torch.allclose(scores, base_scores)
    assert scores.argmax(dim=-1).tolist() == [0]


def test_clstr_native_rerank_warmup_and_export_use_native_routing_without_qdoc(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    base_dir = tmp_path / "base_clstr"
    rerank_dir = tmp_path / "native_rerank"
    eval_dir = tmp_path / "native_eval"
    model_dir = tmp_path / "tiny-model"
    make_skillret_source(source)
    write_jsonl(
        source / "data" / "skills" / "test.jsonl",
        [
            {
                "id": "skill-negative-first",
                "name": "unrelated",
                "description": "Unrelated first skill",
                "skill_md": "# Unrelated\nDo something else.",
                "major": "misc",
            },
            {
                "id": "skill-positive-second",
                "name": "pdf-helper",
                "description": "Work with PDFs",
                "skill_md": "# PDF Helper\nProcess PDFs.",
                "major": "documents",
            },
        ],
    )
    write_jsonl(
        source / "data" / "qrels" / "test.jsonl",
        [{"query_id": "q-test", "skill_id": "skill-positive-second", "relevance": 1}],
    )
    import_skillret_dataset(source, data_root, report_dir)
    subprocess.run(
        [sys.executable, "scripts/make_tiny_hf_model.py", "--output_dir", str(model_dir)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )
    base_report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=str(model_dir),
        output_dir=base_dir,
        max_steps=1,
        batch_size=1,
        model_dim=32,
        top_k=2,
        max_skills=2,
        max_queries=1,
        max_length=128,
        skill_table_batch_size=2,
        skill_text_format="skillret_official",
        query_text_format="skillrouter",
        use_cross_encoder=False,
    )

    train_report = run_clstr_native_rerank_warmup(
        data_root=data_root,
        base_checkpoint_path=base_report["checkpoint"],
        output_dir=rerank_dir,
        max_steps=1,
        batch_size=1,
        candidate_k=2,
        max_skills=2,
        max_queries=1,
        learning_rate=1.0e-4,
    )

    assert train_report["status"] == "ok"
    assert train_report["training_objective"] == "L_retr_native_residual_listwise"
    assert train_report["loss_taxonomy"]["belongs_to"] == "L_retr"
    assert train_report["qdoc_adapter_used"] is False
    assert train_report["uses_skillret_test_qrels"] is False
    assert Path(train_report["checkpoint"]).exists()

    run_report = export_clstr_native_rerank_official_run(
        data_root=data_root,
        base_checkpoint_path=base_report["checkpoint"],
        rerank_checkpoint_path=train_report["checkpoint"],
        output_dir=eval_dir,
        split="test",
        top_k=1,
        batch_size=1,
        max_skills=1,
        run_name="clstr_native_rerank_full",
    )

    assert run_report["method"] == "clstr_native_rerank"
    assert run_report["qdoc_adapter_used"] is False
    assert run_report["loss_taxonomy"] == "L_retr"
    assert run_report["skill_pool_selection"] == "first_n_no_qrel_peek"
    assert (eval_dir / "run.tsv").exists()
    predictions = (eval_dir / "predictions.jsonl").read_text(encoding="utf-8")
    assert "q-test" in predictions


def test_native_routing_init_manifest_records_adoption_without_qdoc(tmp_path):
    base_metrics = {"NDCG@10": 0.5, "Recall@10": 0.5, "MAP@10": 0.5}
    native_metrics = {"NDCG@10": 0.51, "Recall@10": 0.52, "MAP@10": 0.5}
    qdoc_metrics = {"NDCG@10": 0.6, "Recall@10": 0.6, "MAP@10": 0.6}

    decision = build_clstr_native_rerank_decision_report(
        base_metrics=base_metrics,
        native_metrics=native_metrics,
        qdoc_metrics=qdoc_metrics,
        output_path=tmp_path / "decision.json",
    )
    base_checkpoint = tmp_path / "base.pt"
    native_checkpoint = tmp_path / "native.pt"
    base_checkpoint.write_bytes(b"base")
    native_checkpoint.write_bytes(b"native")
    manifest = build_native_routing_init_manifest(
        base_clstr_checkpoint=base_checkpoint,
        native_rerank_checkpoint=native_checkpoint,
        native_rerank_metrics=native_metrics,
        decision_report=decision,
        output_path=tmp_path / "manifest.json",
    )

    assert decision["native_rerank_adopted"] is True
    assert decision["qdoc_rerank_role"] == "static_ablation_not_mainline_init"
    assert decision["loss_taxonomy"]["listwise_rerank"] == "L_retr"
    assert manifest["native_rerank_adopted"] is True
    assert manifest["qdoc_adapter_used"] is False
    assert manifest["downstream_init_for"] == "auxiliary_trajectory_pretrain"
    assert manifest["native_rerank_checkpoint"] == str(native_checkpoint)
    assert manifest["base_clstr_checkpoint_sha256"]
    assert manifest["native_rerank_checkpoint_sha256"]


def test_residual_reranker_zero_delta_preserves_retriever_scores():
    import torch

    reranker = CLSTRResidualListwiseRerankHead(d=2)
    query_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    skill_embs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    candidate_indices = torch.tensor([[0, 2, 1], [1, 2, 0]])
    base_scores = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])

    scores = _score_residual_rerank_candidates(
        reranker=reranker,
        query_embs=query_embs,
        skill_embs=skill_embs,
        candidate_indices=candidate_indices,
        base_scores=base_scores,
        device=torch.device("cpu"),
        base_score_temperature=1.0,
        residual_weight=1.0,
    )

    assert torch.allclose(scores, base_scores)
    assert scores.argmax(dim=-1).tolist() == [0, 0]


def test_official_protocol_report_records_eval_contract(tmp_path):
    repo = tmp_path / "skillret_repo"
    (repo / "skillret").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "train" / "4gpu-qwen3-0.6b").mkdir(parents=True)
    (repo / "README.md").write_text("official implementation\npytrec_eval\n", encoding="utf-8")
    (repo / "skillret" / "eval.py").write_text("def trec_eval(): pass\n", encoding="utf-8")
    (repo / "scripts" / "run_eval_embedding.sh").write_text("eval_retrieval\n", encoding="utf-8")
    (repo / "scripts" / "run_eval_rerank.sh").write_text("eval_rerank\n", encoding="utf-8")

    report = build_official_protocol_report(
        official_repo=repo,
        output_path=tmp_path / "protocol_report.json",
    )

    assert report["status"] == "ok"
    assert report["official_repo"] == str(repo)
    assert report["eval_split"] == "test"
    assert "NDCG@10" in report["metric_names"]
    assert report["qrels_format"] == "query_id -> skill_id -> integer relevance"
    assert report["run_format"] == "TREC run TSV: query_id Q0 skill_id rank score run_name"
    assert report["supports_custom_run_file"] is True


def test_official_protocol_eval_writes_run_and_pytrec_metrics(tmp_path):
    source = tmp_path / "skillret_source"
    data_root = tmp_path / "skillret_data"
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "official_eval"
    make_skillret_source(source)
    import_skillret_dataset(source, data_root, report_dir)

    run_report = export_lexical_official_run(
        data_root=data_root,
        output_dir=output_dir / "lexical_official_smoke",
        split="test",
        top_k=15,
    )

    metrics = evaluate_official_run(
        data_root=data_root,
        run_path=Path(run_report["run_path"]),
        output_dir=output_dir / "lexical_official_smoke",
        split="test",
    )

    assert metrics["status"] == "ok"
    assert metrics["protocol"] == "official_skillret_pytrec_eval"
    assert metrics["query_count"] == 1
    assert metrics["NDCG@10"] == 1.0
    assert metrics["Recall@10"] == 1.0
    assert metrics["Completeness@10"] == 1.0
    assert metrics["MAP@10"] == 1.0
    assert (output_dir / "lexical_official_smoke" / "metrics.json").exists()
    assert (output_dir / "lexical_official_smoke" / "run.tsv").exists()
    predictions = (output_dir / "lexical_official_smoke" / "predictions.jsonl").read_text(encoding="utf-8")
    assert "q-test" in predictions
    assert "skill-c" in predictions


def test_official_comparison_table_uses_official_metric_columns(tmp_path):
    eval_root = tmp_path / "skillret_eval"
    for run_name, hit in [("skillrouter_frozen", 0.25), ("clstr_skillret_warmup", 0.5)]:
        run_dir = eval_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "method": run_name,
                    "paper_role": "paper_static_retrieval",
                    "split": "test",
                    "train_data": "none" if run_name == "skillrouter_frozen" else "SKILLRET train",
                    "model_or_checkpoint": "frozen" if run_name == "skillrouter_frozen" else "checkpoint.pt",
                    "NDCG@5": hit,
                    "NDCG@10": hit,
                    "Recall@10": hit,
                    "Completeness@10": hit,
                    "MAP@10": hit,
                    "query_count": 2,
                }
            ),
            encoding="utf-8",
        )

    summary = write_official_comparison_table(
        eval_root=eval_root,
        run_names=["skillrouter_frozen", "clstr_skillret_warmup"],
        output_table_path=eval_root / "comparison_table.md",
        output_summary_path=eval_root / "comparison_summary.json",
    )

    table = (eval_root / "comparison_table.md").read_text(encoding="utf-8")
    assert summary["status"] == "ok"
    assert "skillrouter_frozen" in table
    assert "clstr_skillret_warmup" in table
    assert "NDCG@10" in table
    assert "Completeness@10" in table
    assert "static retrieval/rerank evidence" in table
    assert "routing foundation rather than the CLSTR closed-loop contribution" in table
    assert "not a SkillsBench held-out closed-loop harness result" in table


def test_official_comparison_table_skips_non_paper_smoke_rows(tmp_path):
    eval_root = tmp_path / "skillret_eval"
    smoke_dir = eval_root / "clstr_skillret_warmup"
    full_dir = eval_root / "clstr_skillrouter_init_full"
    smoke_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    (smoke_dir / "metrics.json").write_text(
        json.dumps(
            {
                "method": "clstr_skillret_warmup",
                "paper_role": "non_paper_smoke",
                "model_or_checkpoint": "outputs/skillret_warmup/tiny-hf-model",
                "NDCG@10": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (full_dir / "metrics.json").write_text(
        json.dumps(
                {
                    "method": "clstr_skillrouter_init_full",
                    "paper_role": "paper_static_retrieval",
                    "split": "test",
                    "model_or_checkpoint": "SkillRouter initialized CLSTR",
                    "NDCG@10": 0.5,
                "Recall@10": 0.5,
                "Completeness@10": 0.5,
                "MAP@10": 0.5,
            }
        ),
        encoding="utf-8",
    )

    summary = write_official_comparison_table(
        eval_root=eval_root,
        run_names=["clstr_skillret_warmup", "clstr_skillrouter_init_full"],
        output_table_path=eval_root / "comparison_table.md",
        output_summary_path=eval_root / "comparison_summary.json",
    )

    table = (eval_root / "comparison_table.md").read_text(encoding="utf-8")
    assert "clstr_skillrouter_init_full" in table
    assert "clstr_skillret_warmup" not in table
    assert summary["skipped_non_paper_smoke"] == ["clstr_skillret_warmup"]


def test_official_comparison_table_skips_pilot_or_non_test_rows(tmp_path):
    eval_root = tmp_path / "skillret_eval"
    pilot_dir = eval_root / "pilot_run"
    dev_dir = eval_root / "dev_run"
    full_dir = eval_root / "full_run"
    for directory in [pilot_dir, dev_dir, full_dir]:
        directory.mkdir(parents=True)
    common = {
        "NDCG@10": 0.5,
        "Recall@10": 0.5,
        "Completeness@10": 0.5,
        "MAP@10": 0.5,
    }
    (pilot_dir / "metrics.json").write_text(
        json.dumps({"method": "pilot_run", "paper_role": "pilot", "split": "test", **common}),
        encoding="utf-8",
    )
    (dev_dir / "metrics.json").write_text(
        json.dumps({"method": "dev_run", "paper_role": "paper_static_retrieval", "split": "dev", **common}),
        encoding="utf-8",
    )
    (full_dir / "metrics.json").write_text(
        json.dumps({"method": "full_run", "paper_role": "paper_static_retrieval", "split": "test", **common}),
        encoding="utf-8",
    )

    summary = write_official_comparison_table(
        eval_root=eval_root,
        run_names=["pilot_run", "dev_run", "full_run"],
        output_table_path=eval_root / "comparison_table.md",
        output_summary_path=eval_root / "comparison_summary.json",
    )

    table = (eval_root / "comparison_table.md").read_text(encoding="utf-8")
    assert "full_run" in table
    assert "pilot_run" not in table
    assert "dev_run" not in table
    assert summary["skipped_non_paper_runs"] == ["pilot_run", "dev_run"]


def test_clstr_export_config_prefers_checkpoint_training_config(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch_payload = {
        "model_state_dict": {},
        "config": {
            "base_model_name": "/models/SkillRouter-Embedding-0.6B",
            "d": 256,
            "d_a": 64,
            "top_k": 50,
            "encoder_pooling": "last_token",
            "cross_encoder_pooling": "last_token",
            "tokenizer_padding_side": "left",
            "torch_dtype": "bfloat16",
            "freeze_backbone": True,
        },
    }
    import torch

    torch.save(torch_payload, checkpoint)

    resolved = resolve_clstr_export_config(
        base_model_name="outputs/skillret_warmup/tiny-hf-model",
        checkpoint_path=checkpoint,
        model_dim=16,
        top_k=15,
    )

    assert resolved["base_model_name"] == "/models/SkillRouter-Embedding-0.6B"
    assert resolved["d"] == 256
    assert resolved["top_k"] == 50
    assert resolved["encoder_pooling"] == "last_token"
    assert resolved["tokenizer_padding_side"] == "left"
    assert resolved["torch_dtype"] == "bfloat16"
    assert resolved["freeze_backbone"] is True


def test_skillret_serialization_report_records_official_skillrouter_format(tmp_path):
    report = write_skillret_serialization_report(
        output_path=tmp_path / "serialization_report.json",
        official_repo="/root/autodl-tmp/skillret_repo",
    )

    assert report["status"] == "ok"
    assert report["query_format"]["skillrouter"]["pooling"] == "last_token"
    assert report["query_format"]["skillrouter"]["padding_side"] == "left"
    assert "Instruct: Given a task description" in report["query_format"]["skillrouter"]["template"]
    assert report["doc_format"]["source"] == "official_skillret_embedding_text_for_skill"
