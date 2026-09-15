import json

import pytest

from clstr.qwen_route_eval import (
    build_qwen_route_method_name,
    embedding_model_family,
    is_reranker_disabled,
    reranker_family,
    load_qwen_route_corpus,
    metric_row_from_reranked_candidates,
    route_query_text,
    select_embedding_topk,
    summarize_route_metric_rows,
    _append_route_eval_prediction,
    _load_existing_route_predictions,
    _route_eval_row_key,
)


def test_missing_positive_after_embedding_topk_counts_as_zero_not_skipped():
    selected = select_embedding_topk(
        candidate_skill_ids=["skill/a", "skill/gold", "skill/b"],
        candidate_scores=[0.9, 0.1, 0.8],
        top_k=2,
    )

    assert selected == ["skill/a", "skill/b"]

    metric = metric_row_from_reranked_candidates(
        positive_skill_id="skill/gold",
        embedding_candidate_skill_ids=selected,
        reranked_skill_ids=["skill/a", "skill/b"],
    )

    assert metric["recall@1"] == 0.0
    assert metric["recall@5"] == 0.0
    assert metric["mrr"] == 0.0
    assert metric["positive_in_embedding_topk"] == 0.0
    assert metric["candidate_count"] == 2.0


def test_summarize_route_metric_rows_uses_source_denominator():
    summary = summarize_route_metric_rows(
        [
            {
                "recall@1": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
                "candidate_count": 2.0,
                "positive_in_embedding_topk": 1.0,
            },
            {
                "recall@1": 0.0,
                "recall@5": 0.0,
                "mrr": 0.0,
                "candidate_count": 2.0,
                "positive_in_embedding_topk": 0.0,
            },
        ],
        source_rows=4,
    )

    assert summary["retained_metrics"]["mrr"] == pytest.approx(0.5)
    assert summary["strict"]["strict_mrr"] == pytest.approx(0.25)
    assert summary["strict"]["retained_rows"] == 2.0
    assert summary["strict"]["source_rows"] == 4.0
    assert summary["embedding_recall@topk"] == pytest.approx(0.5)


def test_reranker_disabled_accepts_explicit_embedding_only_values():
    assert is_reranker_disabled(None)
    assert is_reranker_disabled("")
    assert is_reranker_disabled("none")
    assert is_reranker_disabled("embedding_only")
    assert not is_reranker_disabled("models/Qwen3-Reranker-0.6B")


def test_reranker_family_detects_bge_sequence_classifier():
    assert reranker_family("models/BAAI/bge-reranker-v2-m3") == "bge_sequence_classifier"
    assert reranker_family("BAAI/bge-reranker-v2-m3") == "bge_sequence_classifier"
    assert reranker_family("models/Qwen3-Reranker-0.6B") == "qwen3_causal_yes_no"


def test_reranker_family_detects_finetuned_bge_checkpoint_from_config(tmp_path):
    checkpoint = tmp_path / "bge_sr_rank-step2000"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["XLMRobertaForSequenceClassification"],
                "model_type": "xlm-roberta",
                "num_labels": 1,
            }
        ),
        encoding="utf-8",
    )

    assert reranker_family(checkpoint) == "bge_sequence_classifier"


def test_embedding_model_family_detects_bge_m3():
    assert embedding_model_family("models/BAAI/bge-m3") == "bge_m3"
    assert embedding_model_family("BAAI/bge-m3") == "bge_m3"
    assert embedding_model_family("models/Qwen3-Embedding-0.6B") == "qwen3_embedding"


def test_route_query_text_auto_uses_raw_state_for_bge_m3():
    row = {"state_text": "Need current weather for Boston."}

    text = route_query_text(
        row,
        benchmark="toolbench_g3",
        embedding_family="bge_m3",
        query_text_mode="auto",
    )

    assert text == "Need current weather for Boston."
    assert "Instruct:" not in text


def test_route_query_text_auto_keeps_instruct_prompt_for_qwen():
    row = {"state_text": "Need current weather for Boston."}

    text = route_query_text(
        row,
        benchmark="toolbench_g3",
        embedding_family="qwen3_embedding",
        query_text_mode="auto",
    )

    assert text.startswith("Instruct:")
    assert "Need current weather for Boston." in text


def test_build_qwen_route_method_name_reflects_model_size():
    assert (
        build_qwen_route_method_name(
            embedding_model_name_or_path="models/Qwen3-Embedding-8B",
            reranker_model_name_or_path="models/Qwen3-Reranker-8B",
        )
        == "qwen3_embedding_8b_plus_qwen3_reranker_8b"
    )
    assert (
        build_qwen_route_method_name(
            embedding_model_name_or_path="models/Tool-Embed-0.6B",
            reranker_model_name_or_path="none",
        )
        == "toolembed_0_6b_embedding_only_route_eval"
    )


def test_load_qwen_route_corpus_supports_prebuilt_trajectbench_rows(tmp_path):
    skills_path = tmp_path / "skills.jsonl"
    rows_path = tmp_path / "rows.jsonl"
    skills_path.write_text(
        "\n".join(
            [
                '{"skill_id":"skill/a","name":"Skill A"}',
                '{"skill_id":"skill/b","name":"Skill B"}',
                '{"skill_id":"skill/unused","name":"Unused"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows_path.write_text(
        '{"trajectory_id":"t1","step_index":0,"state_text":"choose b","next_skill_id":"skill/b",'
        '"candidate_next_skill_ids":["skill/a","skill/b"]}\n',
        encoding="utf-8",
    )

    corpus = load_qwen_route_corpus(
        benchmark="trajectbench",
        trajectbench_eval_rows_path=rows_path,
        trajectbench_skills_path=skills_path,
    )

    assert corpus.benchmark == "trajectbench"
    assert corpus.candidate_source == "row_candidates"
    assert [row["skill_id"] for row in corpus.skills] == ["skill/a", "skill/b"]
    assert corpus.source_rows[0]["next_skill_id"] == "skill/b"
    assert corpus.report["selected_skill_count"] == 2


def test_apply_embedding_adapter_projects_queries_and_skills(tmp_path):
    import torch

    from clstr.qwen_route_eval import apply_embedding_adapter

    adapter_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "method": "generic_embedding_adapter_finetune",
            "step": 3,
            "adapter_state_dict": {
                "q_proj.weight": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "d_proj.weight": torch.eye(2),
            },
            "config": {"backbone": "frozen"},
        },
        adapter_path,
    )

    query_embs = torch.tensor([[1.0, 0.0]])
    skill_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    projected_queries, projected_skills, report = apply_embedding_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_path,
    )

    scores = projected_queries @ projected_skills.T
    assert scores.argmax(dim=-1).item() == 1
    assert report["checkpoint_path"] == str(adapter_path)
    assert report["q_proj_shape"] == [2, 2]


def test_write_route_eval_outputs_adds_generic_report_alias(tmp_path):
    from clstr.qwen_route_eval import _write_route_eval_outputs

    report = {
        "status": "ok",
        "benchmark": "toolbench_g3",
        "strict": {"strict_mrr": 0.5},
        "retained_eval_rows": 2,
        "source_eval_rows": 2,
    }
    predictions = [{"task_id": "a", "ranked_skill_ids": ["skill/a"]}]

    paths = _write_route_eval_outputs(
        output_dir=tmp_path,
        report=report,
        predictions=predictions,
        save_predictions=True,
    )

    assert paths["legacy_report_path"].name == "qwen3_embedding_reranker_route_eval_report.json"
    assert paths["generic_report_path"].name == "route_eval_report.json"
    assert paths["legacy_report_path"].exists()
    assert paths["generic_report_path"].exists()
    assert paths["metrics_path"].exists()
    assert paths["legacy_predictions_path"].exists()
    assert paths["generic_predictions_path"].exists()


def test_write_route_eval_outputs_adds_blocker_report_for_action_required(tmp_path):
    import json

    from clstr.qwen_route_eval import _write_route_eval_outputs

    report = {
        "status": "action_required",
        "benchmark": "tau2",
        "blockers": ["no_source_or_retained_rows"],
        "source_eval_rows": 0,
        "retained_eval_rows": 0,
    }

    paths = _write_route_eval_outputs(
        output_dir=tmp_path,
        report=report,
        predictions=[],
        save_predictions=False,
    )

    assert paths["metrics_path"] is None
    assert paths["blocker_report_path"].exists()
    blocker = json.loads(paths["blocker_report_path"].read_text(encoding="utf-8"))
    assert blocker["status"] == "action_required"
    assert blocker["blockers"] == ["no_source_or_retained_rows"]


def test_route_eval_row_key_is_stable_for_resume():
    row = {
        "task_id": "task-1",
        "trajectory_id": "traj-9",
        "step_index": 3,
        "domain": "retail",
        "next_skill_id": "skill/a",
    }

    key = _route_eval_row_key(row, row_index=4)

    assert key == "idx=4|task=task-1|traj=traj-9|step=3|domain=retail|pos=skill/a"


def test_load_existing_route_predictions_uses_generic_then_legacy(tmp_path):
    generic = tmp_path / "route_eval_predictions.jsonl"
    legacy = tmp_path / "qwen3_embedding_reranker_predictions.jsonl"
    legacy.write_text(
        json.dumps({"row_key": "legacy", "metrics": {"mrr": 0.5}, "ranked_skill_ids": ["skill/l"]}) + "\n",
        encoding="utf-8",
    )
    generic.write_text(
        "\n".join(
            [
                json.dumps({"row_key": "done", "metrics": {"mrr": 1.0}, "ranked_skill_ids": ["skill/a"]}),
                json.dumps({"row_key": "skip-no-metrics", "ranked_skill_ids": ["skill/b"]}),
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_existing_route_predictions(tmp_path)

    assert sorted(loaded) == ["done"]
    assert loaded["done"]["metrics"]["mrr"] == 1.0


def test_append_route_eval_prediction_writes_incremental_aliases(tmp_path):
    prediction = {
        "row_key": "idx=1|task=t|traj=|step=|domain=|pos=skill/a",
        "task_id": "t",
        "next_skill_id": "skill/a",
        "metrics": {"mrr": 1.0},
    }

    _append_route_eval_prediction(tmp_path, prediction, save_predictions=True)

    generic_rows = [
        json.loads(line)
        for line in (tmp_path / "route_eval_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    legacy_rows = [
        json.loads(line)
        for line in (tmp_path / "qwen3_embedding_reranker_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert generic_rows == [prediction]
    assert legacy_rows == [prediction]
