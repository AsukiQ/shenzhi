import json

from scripts.update_bge_table_from_reports import route_report_to_table_row


def test_route_report_to_table_row_uses_strict_metrics(tmp_path):
    report_path = tmp_path / "route_eval_report.json"
    report = {
        "benchmark": "toolbench_g3",
        "method": "bge_m3_embedding_only_route_eval",
        "embedding_model_family": "bge_m3",
        "reranker_model_name_or_path": None,
        "source_eval_rows": 32,
        "retained_eval_rows": 30,
        "top_k": 100,
        "candidate_source": "full_pool",
        "strict": {
            "strict_recall@1": 0.125,
            "strict_recall@5": 0.25,
            "strict_mrr": 0.2,
        },
        "embedding_report": {
            "candidate_source": "full_pool",
            "top_k": 100,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    row = route_report_to_table_row(report_path)

    assert row["benchmark"] == "ToolBench-G3"
    assert row["method"] == "BGE-M3 retrieval-only"
    assert row["params"] == "0.6B"
    assert row["r1"] == "0.125000"
    assert row["r5"] == "0.250000"
    assert row["mrr"] == "0.200000"
    assert row["rows"] == "30/32 strict"
    assert row["candidate_scope"] == "full pool -> top100"
    assert row["report_path"] == str(report_path)


def test_route_report_to_table_row_labels_bge_reranker(tmp_path):
    report_path = tmp_path / "route_eval_report.json"
    report = {
        "benchmark": "tau2",
        "method": "bge_m3_plus_bge_reranker_v2_m3",
        "embedding_model_family": "bge_m3",
        "reranker_model_name_or_path": "models/BAAI/bge-reranker-v2-m3",
        "source_eval_rows": 32,
        "retained_eval_rows": 32,
        "candidate_source": "row_candidates",
        "strict": {
            "strict_recall@1": 0.1,
            "strict_recall@5": 0.4,
            "strict_mrr": 0.3,
        },
        "embedding_report": {
            "candidate_count_mean": 14.9,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    row = route_report_to_table_row(report_path)

    assert row["benchmark"] == "tau2"
    assert row["method"] == "BGE-M3 + BGE-reranker-v2-M3"
    assert row["params"] == "1.2B"
    assert row["candidate_scope"] == "row candidates"


def test_route_report_to_table_row_labels_bge_toolrex_embed(tmp_path):
    report_path = tmp_path / "route_eval_report.json"
    report = {
        "benchmark": "toolbench_g3",
        "method": "bge_m3_embedding_only_route_eval",
        "embedding_model_family": "bge_m3",
        "embedding_model_name_or_path": "outputs/bge_toolrex_embedding_full_train/run/checkpoints/bge-m3-tool-embed-step2000",
        "reranker_model_name_or_path": None,
        "source_eval_rows": 1362,
        "retained_eval_rows": 1362,
        "candidate_source": "full_pool",
        "top_k": 100,
        "strict": {
            "strict_recall@1": 0.11,
            "strict_recall@5": 0.22,
            "strict_mrr": 0.18,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    row = route_report_to_table_row(report_path)

    assert row["method"] == "BGE-Tool-Embed retrieval-only"
    assert row["params"] == "0.6B"


def test_route_report_to_table_row_labels_bge_toolrex_embed_rank(tmp_path):
    report_path = tmp_path / "route_eval_report.json"
    report = {
        "benchmark": "toolbench_g3",
        "method": "bge_m3_plus_bge_sequence_classifier",
        "embedding_model_family": "bge_m3",
        "embedding_model_name_or_path": "outputs/bge_toolrex_embedding_full_train/run/checkpoints/bge-m3-tool-embed-step2000",
        "reranker_model_name_or_path": "outputs/bge_toolrex_reranker_train/run/checkpoints/bge_tool_rank-step2000",
        "source_eval_rows": 1362,
        "retained_eval_rows": 1362,
        "candidate_source": "full_pool",
        "top_k": 100,
        "strict": {
            "strict_recall@1": 0.12,
            "strict_recall@5": 0.24,
            "strict_mrr": 0.19,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    row = route_report_to_table_row(report_path)

    assert row["method"] == "BGE-Tool-Embed + BGE-Tool-Rank"
    assert row["params"] == "1.2B"
