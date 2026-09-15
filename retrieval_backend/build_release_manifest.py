"""Build a fail-closed manifest for the reproducible Shenzhi retrieval release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-bm25-full", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    required = {
        "stage0_verification": root / "outputs/paper_vnext_stage0_skillrouter_full/verification_report.json",
        "gpu_eval_matrix": root / "outputs/gpu_eval_matrix.json",
        "http_service_smoke": root / "outputs/http_service_smoke.json",
        "http_graph_smoke": root / "outputs/http_graph_smoke.json",
        "bm25_http_smoke": root / "outputs/bm25_http_smoke.json",
        "graph_ablation_report": root / "outputs/graph_ablation_real_neo4j.json",
        "neo4j_import_manifest": root / "runtime/neo4j/import_ready/manifest.json",
        "neo4j_healthcheck": root / "outputs/neo4j_healthcheck.json",
        "neo4j_schema_report": root / "outputs/neo4j_schema_report.json",
        "neo4j_auth_smoke": root / "outputs/neo4j_auth_smoke.json",
        "stage0_data_contract": root / "derived/paper_vnext_stage0_v1/data_contract.json",
        "human_queries": root / "evaluation/human_queries_seed_v1.jsonl",
        "graph_queries": root / "evaluation/graph_queries_seed_v1.jsonl",
        "human_qrels_annotation": root / "evaluation/human_qrels_annotation_v1.jsonl",
        "qrels_instructions": root / "evaluation/QRELS_ANNOTATION.md",
        "human_qrels_status": root / "outputs/human_qrels_status.json",
        "backend_readme": root / "retrieval_backend/README.md",
        "project_status": root / "retrieval_backend/PROJECT_STATUS.md",
        "environment_example": root / "retrieval_backend/retrieval.env.example",
        "deployment_checksums": root / "retrieval_backend/deployment/SHA256SUMS",
        "build_index_script": root / "retrieval_backend/build_paper_index.sh",
        "run_service_script": root / "retrieval_backend/run_retrieval_service.sh",
        "run_neo4j_script": root / "retrieval_backend/run_neo4j_apptainer.sh",
        "verify_release_script": root / "retrieval_backend/verify_release.sh",
        "http_server_source": root / "retrieval_backend/http_server.py",
        "retrieval_pipeline_source": root / "retrieval_backend/retrieval_pipeline.py",
        "graph_query_source": root / "retrieval_backend/graph_query.py",
        "graph_ablation_source": root / "retrieval_backend/evaluate_graph_ablation.py",
        "graph_query_test": root / "retrieval_backend/test_graph_query.py",
        "graph_ablation_test": root / "retrieval_backend/test_evaluate_graph_ablation.py",
        "graph_retrieval_contract": root / "retrieval_backend/data/graph_retrieval_contract.v1.json",
        "graph_contract_test": root / "retrieval_backend/test_graph_contract.py",
        "paper_search_source": root / "retrieval_backend/paper_search.py",
        "neo4j_filter_source": root / "retrieval_backend/neo4j_filter.py",
        "release_manifest_builder": root / "retrieval_backend/build_release_manifest.py",
        # Stage2 is an optional multi-step extension.  Its artifacts are still
        # bound in the release so that the engineering result is reproducible,
        # while the manifest keeps the weak-bootstrap/evidence boundary explicit.
        "stage2_data_contract": root / "derived/action_vnext_stage2_v1/data_contract.json",
        "stage2_train_report": root / "outputs/action_vnext_stage2_skillrouter_full/train_report.json",
        "stage2_quality_gate": root / "outputs/action_vnext_stage2_skillrouter_full/stage2_quality_gate.json",
        "stage2_selection": root / "outputs/action_vnext_stage2_skillrouter_full/stage2_selection.json",
        "stage2_reselection_manifest": root / "outputs/action_vnext_stage2_skillrouter_full/stage2_reselection_manifest.json",
        "stage2_training_skills": root / "outputs/action_vnext_stage0_skillrouter_full/selected_skills.jsonl",
        "stage2_checkpoint": root / "outputs/action_vnext_stage2_skillrouter_full/checkpoints/clstr_vnext_stage2-step800.pt",
        "stage2_trajectory_train": root / "derived/action_vnext_stage2_v1/trajectory_train.jsonl",
        "stage2_trajectory_dev": root / "derived/action_vnext_stage2_v1/trajectory_dev.jsonl",
        "stage2_causal_branch_train": root / "derived/action_vnext_stage2_v1/causal_branch_pairs_train.jsonl",
        "stage2_causal_branch_dev": root / "derived/action_vnext_stage2_v1/causal_branch_pairs_dev.jsonl",
        "stage2_pair_support_train": root / "derived/action_vnext_stage2_v1/causal_pair_support_train.jsonl",
        "stage2_pair_support_dev": root / "derived/action_vnext_stage2_v1/causal_pair_support_dev.jsonl",
        "stage2_inventory_catalog": root / "derived/action_vnext_stage0_v1/inventory_catalogs.jsonl",
        "action_stage0_checkpoint": root / "outputs/action_vnext_stage0_skillrouter_full/checkpoints/clstr_vnext_stage0-step1000.pt",
        "action_stage0_quality_gate": root / "outputs/action_vnext_stage0_skillrouter_full/stage0_quality_gate.json",
        "action_stage0_selection": root / "outputs/action_vnext_stage0_skillrouter_full/stage0_selection.json",
        "stage2_multistep_executor": root / "retrieval_backend/multistep_search.py",
        "stage2_data_builder": root / "retrieval_backend/build_action_stage2_data.py",
        "stage2_http_test": root / "retrieval_backend/test_http_multistep.py",
        "stage2_graph_filter": root / "retrieval_backend/neo4j_filter.py",
        "stage2_online_smoke_source": root / "retrieval_backend/stage2_online_smoke.py",
        "stage2_online_smoke_script": root / "retrieval_backend/sbatch_stage2_online_smoke.sh",
        "stage2_policy_eval_source": root / "retrieval_backend/evaluate_stage2_policy.py",
        "stage2_policy_eval_script": root / "retrieval_backend/sbatch_stage2_policy_eval.sh",
        "stage2_policy_eval_report": root / "outputs/stage2_policy_eval.json",
    }
    bm25 = root / "outputs/bm25_test_full.json"
    if bm25.is_file() or not args.allow_missing_bm25_full:
        required["bm25_test_full"] = bm25
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release is missing required artifacts: {missing}")

    stage0 = load_json(required["stage0_verification"])
    gpu = load_json(required["gpu_eval_matrix"])
    http = load_json(required["http_service_smoke"])
    http_graph = load_json(required["http_graph_smoke"])
    bm25_http = load_json(required["bm25_http_smoke"])
    graph_ablation = load_json(required["graph_ablation_report"])
    neo4j = load_json(required["neo4j_healthcheck"])
    schema = load_json(required["neo4j_schema_report"])
    auth = load_json(required["neo4j_auth_smoke"])
    qrels = load_json(required["human_qrels_status"])
    stage2_contract = load_json(required["stage2_data_contract"])
    stage2_gate = load_json(required["stage2_quality_gate"])
    stage2_selection = load_json(required["stage2_selection"])
    stage2_reselection = load_json(required["stage2_reselection_manifest"])
    stage2_eval = load_json(required["stage2_policy_eval_report"])
    graph_contract = load_json(required["graph_retrieval_contract"])
    graph_snapshot = graph_contract["source_snapshot"]
    online_smoke_path = root / "outputs/stage2_online_smoke.json"
    stage2_online = load_json(online_smoke_path) if online_smoke_path.is_file() else None
    bm25_report = load_json(bm25) if bm25.is_file() else None
    if any(
        row.get("status") != "ok"
        for row in (
            stage0,
            gpu,
            http,
            http_graph,
            bm25_http,
            graph_ablation,
            neo4j,
            schema,
            auth,
            qrels,
            stage2_contract,
            stage2_gate,
            stage2_selection,
            stage2_reselection,
            stage2_eval,
        )
    ):
        raise ValueError("one or more release quality reports are not ok")
    if stage2_contract.get("label_quality") != "weak_bootstrap_executable_replay":
        raise ValueError("Stage2 contract must explicitly identify weak bootstrap labels")
    if stage2_contract.get("stage_contract") != "canonical_clstr_vnext_stage2":
        raise ValueError("Stage2 data contract is not canonical vNext Stage2")
    if not all(
        bool(stage2_gate.get(name))
        for name in (
            "causal_route_gate",
            "causal_safety_gate",
            "ordinary_safety_gate",
            "coarse_recall_gate",
            "gradient_health_gate",
        )
    ):
        raise ValueError("Stage2 quality gate is incomplete")
    if int(stage2_selection.get("selected_step") or 0) != 800:
        raise ValueError("Stage2 selected checkpoint step mismatch")
    if int(stage2_reselection.get("selected_step") or 0) != 800:
        raise ValueError("Stage2 reselection checkpoint step mismatch")
    if not stage2_reselection.get("selected_checkpoint_path"):
        raise ValueError("Stage2 reselection lacks selected checkpoint binding")
    if stage2_eval.get("protocol") != "shenzhi_stage2_frozen_dev_evaluation_v1":
        raise ValueError("Stage2 policy evaluation protocol mismatch")
    if stage2_eval.get("evidence_scope") != "weak_bootstrap_executable_replay_engineering_evaluation":
        raise ValueError("Stage2 policy evaluation overstates its evidence scope")
    if int(stage2_eval.get("trajectory_count") or 0) != 48:
        raise ValueError("Stage2 policy evaluation must cover the frozen 48-trajectory dev split")
    teacher = stage2_eval.get("teacher_forced") or {}
    closed = stage2_eval.get("closed_loop") or {}
    if int(teacher.get("decision_count") or 0) != 192:
        raise ValueError("Stage2 teacher-forced evaluation decision count mismatch")
    if not 0.0 <= float(teacher.get("top1_accuracy") or -1.0) <= float(teacher.get("top5_accuracy") or -1.0) <= 1.0:
        raise ValueError("Stage2 teacher-forced accuracy metrics are invalid")
    if int(closed.get("episode_count") or 0) != 48:
        raise ValueError("Stage2 closed-loop episode count mismatch")
    if float(closed.get("stopped_episode_rate") or 0.0) != 1.0:
        raise ValueError("Stage2 closed-loop evaluation did not terminate every episode")
    if float(closed.get("nonempty_result_rate") or 0.0) != 1.0:
        raise ValueError("Stage2 closed-loop evaluation returned empty episodes")
    if int(closed.get("illegal_action_count") or 0) != 0:
        raise ValueError("Stage2 closed-loop evaluation observed illegal actions")
    if int(closed.get("executor_failure_count") or 0) != 0:
        raise ValueError("Stage2 closed-loop evaluation observed executor failures")
    if stage2_online is not None:
        if stage2_online.get("protocol") != "shenzhi_stage2_native_online_smoke_v1":
            raise ValueError("Stage2 online smoke protocol mismatch")
        if int(stage2_online.get("skill_count") or 0) != 515:
            raise ValueError("Stage2 online smoke skill inventory mismatch")
        if int(stage2_online.get("illegal_action_count") or 0) != 0:
            raise ValueError("Stage2 online smoke observed illegal actions")
        if int(stage2_online.get("failure_count") or 0) != 0:
            raise ValueError("Stage2 online smoke observed executor failures")
        if not bool(stage2_online.get("event_count")) or not stage2_online.get("stop_reason"):
            raise ValueError("Stage2 online smoke did not terminate a real episode")
    stage2_counts = stage2_contract.get("counts") or {}
    if bm25_report is not None and bm25_report.get("status") != "ok":
        raise ValueError("full BM25 evaluation is not ok")
    expected_components = {
        "bm25", "stage0_dense", "skillrouter_reranker", "neo4j_filter"
    }
    actual_components = set((http.get("health") or {}).get("components") or [])
    if expected_components - actual_components:
        raise ValueError("HTTP smoke did not verify every release component")
    if set((bm25_http.get("health") or {}).get("components") or []) != {"bm25"}:
        raise ValueError("BM25 fallback smoke did not run in BM25-only mode")
    if (bm25_http.get("health") or {}).get("graph") is not None:
        raise ValueError("BM25 fallback smoke unexpectedly enabled Neo4j")
    if bm25_http.get("operations") != ["RECALL_PAPER_BM25"]:
        raise ValueError("BM25 fallback smoke executed unexpected operations")
    graph_components = set((http_graph.get("health") or {}).get("components") or [])
    if not {"bm25", "neo4j_filter"}.issubset(graph_components):
        raise ValueError("graph HTTP smoke did not enable BM25 and Neo4j")
    graph_expansion_operations = (
        (http_graph.get("graph_expansion") or {}).get("state", {}).get(
            "executed_operations"
        )
        or []
    )
    if "EXPAND_PAPER_AUTHORS_NEO4J" not in graph_expansion_operations:
        raise ValueError("graph HTTP smoke did not execute relation expansion")
    if graph_ablation.get("protocol") != "shenzhi_graph_ablation_v1":
        raise ValueError("graph ablation protocol mismatch")
    if graph_ablation.get("graph_enabled") is not True:
        raise ValueError("graph ablation did not use a live graph")
    if int((graph_ablation.get("graph") or {}).get("graph_operation_count") or 0) <= 0:
        raise ValueError("graph ablation did not execute graph operations")
    if (graph_ablation.get("graph") or {}).get("failure_counts"):
        raise ValueError("graph ablation observed graph operation failures")
    if graph_ablation.get("quality") is not None:
        raise ValueError("unlabeled graph smoke queries must not emit relevance metrics")
    if int(stage0.get("selected_step") or 0) != 1000:
        raise ValueError("release Stage0 did not select the 1000-step checkpoint")
    if int((neo4j.get("health") or {}).get("paper_count") or 0) != int(
        graph_snapshot["paper_count"]
    ):
        raise ValueError("release Neo4j Paper count mismatch")
    if int(schema.get("constraint_count") or 0) != int(
        graph_snapshot["constraint_count"]
    ) or int(schema.get("index_count") or 0) != int(graph_snapshot["index_count"]):
        raise ValueError("release Neo4j schema count mismatch")
    if int(auth.get("constraint_count") or 0) != int(
        graph_snapshot["constraint_count"]
    ) or int(auth.get("index_count") or 0) != int(graph_snapshot["index_count"]):
        raise ValueError("authenticated Neo4j smoke did not verify the release schema")
    if qrels.get("annotation_status") != "awaiting_human_labels":
        raise ValueError("human qrels status must explicitly report awaiting_human_labels")
    if int(qrels.get("unlabeled_count") or 0) <= 0:
        raise ValueError("human qrels must remain explicitly unlabeled until human review")
    gpu_reports = gpu.get("reports") or {}
    if set(gpu_reports) != {"dense", "hybrid", "hybrid_reranker"}:
        raise ValueError("GPU evaluation matrix is incomplete")
    if any(int(report.get("query_count") or 0) != 200 for report in gpu_reports.values()):
        raise ValueError("GPU evaluation matrix query count mismatch")
    if bm25_report is not None:
        if bm25_report.get("split") != "test" or int(bm25_report.get("query_count") or 0) != 24084:
            raise ValueError("full BM25 report does not cover the frozen test split")

    artifacts = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in sorted(required.items())
    }
    checkpoint = Path(stage0["selected_checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report = {
        "schema_version": "shenzhi_retrieval_release_v1",
        "status": "ok",
        "scope": "single_request_paper_retrieval_with_optional_graph_plus_optional_multistep_stage2",
        "stage2_status": "weak_bootstrap_engineering_only_no_real_user_generalization_evidence",
        "stage2": {
            "status": "ok",
            "label_quality": stage2_contract["label_quality"],
            "selected_step": int(stage2_selection["selected_step"]),
            "checkpoint": {
                "path": str(required["stage2_checkpoint"].resolve()),
                "bytes": required["stage2_checkpoint"].stat().st_size,
                "sha256": sha256(required["stage2_checkpoint"]),
            },
            "data_contract": {
                "schema_version": stage2_contract["schema_version"],
                "sha256": sha256(required["stage2_data_contract"]),
                "counts": stage2_counts,
            },
            "quality_gates": {
                name: bool(stage2_gate.get(name))
                for name in (
                    "causal_route_gate",
                    "causal_safety_gate",
                    "ordinary_safety_gate",
                    "coarse_recall_gate",
                    "gradient_health_gate",
                )
            },
            "open_pool_release": {
                "status": "not_released",
                "reason": "no_independent_heldout_open_pool_stratum",
            },
            "online_smoke": (
                {
                    "status": "ok",
                    "path": str(online_smoke_path.resolve()),
                    "sha256": sha256(online_smoke_path),
                    "protocol": stage2_online["protocol"],
                    "skill_count": int(stage2_online["skill_count"]),
                    "event_count": int(stage2_online["event_count"]),
                    "executed_actions": stage2_online.get("executed_actions") or [],
                    "stop_reason": stage2_online["stop_reason"],
                }
                if stage2_online is not None
                else {
                    "status": "pending_gpu_online_smoke",
                    "path": str(online_smoke_path.resolve()),
                    "reason": "GPU allocation was submitted but no completed report is available",
                }
            ),
            "frozen_dev_evaluation": {
                "path": str(required["stage2_policy_eval_report"].resolve()),
                "sha256": sha256(required["stage2_policy_eval_report"]),
                "evidence_scope": stage2_eval["evidence_scope"],
                "trajectory_count": int(stage2_eval["trajectory_count"]),
                "teacher_forced": teacher,
                "closed_loop": closed,
                "interpretation_limit": stage2_eval["interpretation_limit"],
            },
            "interpretation_limit": (
                "The checkpoint proves executable training/inference plumbing only; "
                "real user multi-step queries and relevance labels are still required."
            ),
        },
        "paper_document_count": int(http["health"]["lexical"]["paper_count"]),
        "neo4j_paper_count": int(http["health"]["graph"]["paper_count"]),
        "stage0_selected_step": int(stage0["selected_step"]),
        "stage0_checkpoint": {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "evaluation": {
            "bm25_frozen_test": (
                {
                    "query_count": int(bm25_report["query_count"]),
                    "metrics": bm25_report["metrics"],
                    "latency": bm25_report["latency"],
                    "wall_time_seconds": float(bm25_report["wall_time_seconds"]),
                }
                if bm25_report is not None
                else None
            ),
            "weak_test_gpu_matrix": {
                name: {
                    "query_count": int(row["query_count"]),
                    "metrics": row["metrics"],
                    "latency": row["latency"],
                }
                for name, row in sorted(gpu_reports.items())
            },
            "human_qrels": {
                "annotation_status": qrels["annotation_status"],
                "query_count": int(qrels["query_count"]),
                "judgment_count": int(qrels["judgment_count"]),
                "unlabeled_count": int(qrels["unlabeled_count"]),
            },
        },
        "neo4j_schema": {
            "constraints": int(schema["constraint_count"]),
            "indexes": int(schema["index_count"]),
            "authenticated_smoke": auth["status"],
        },
        "graph_retrieval": {
            "status": "ok",
            "contract": {
                "path": str(required["graph_retrieval_contract"].resolve()),
                "sha256": sha256(required["graph_retrieval_contract"]),
                "schema_version": graph_contract["schema_version"],
            },
            "live_http_operations": graph_expansion_operations,
            "ablation": {
                "protocol": graph_ablation["protocol"],
                "query_count": int(graph_ablation["graph"]["query_count"]),
                "graph_operation_count": int(
                    graph_ablation["graph"]["graph_operation_count"]
                ),
                "changed_query_count": int(
                    graph_ablation["graph"]["changed_query_count"]
                ),
                "added_result_count": int(
                    graph_ablation["graph"]["added_result_count"]
                ),
                "quality": None,
                "interpretation_limit": (
                    "Operational graph evidence only; the graph smoke queries have no "
                    "human relevance judgments."
                ),
            },
        },
        "deployment_smoke": {
            "full_components": sorted(actual_components),
            "bm25_fallback_components": ["bm25"],
            "graph_components": sorted(graph_components),
        },
        "artifacts": artifacts,
        "interpretation_limits": [
            "weak-bootstrap title/abstract queries are for regression, not sole generalization evidence",
            "human qrels candidates require independent human annotation before quality claims",
            "Institution/Method/Funding exist in the final graph but are not yet exposed as search filters",
            "graph smoke proves execution and candidate changes, not relevance improvement",
            "Stage2 weak-bootstrap trajectories are executable replays, not observed user behavior",
            "Stage2 open-pool release is fail-closed until an independent held-out open-pool stratum exists",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
