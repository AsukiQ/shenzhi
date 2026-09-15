from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from clstr.matched_history_data import load_matched_history_trajectories
from clstr.matched_history_encoders import build_matched_history_encoder
from clstr.matched_history_train import (
    MATCHED_HISTORY_TRAIN_SCHEMA,
    MatchedHistoryRouteScorer,
    _anchors,
    _collect_cache_texts,
    _evaluate,
)
from clstr.vnext_eval import load_vnext_stage2_for_evaluation
from clstr.vnext_training import (
    file_sha256,
    load_inventory_catalogs,
    load_or_build_frozen_text_cache,
)


MATCHED_HISTORY_TEST_DATA_SCHEMA = "clstr_matched_history_e1_test_data_v1"
MATCHED_HISTORY_TEST_RUN_SCHEMA = "clstr_matched_history_e1_test_run_v1"
MATCHED_HISTORY_TEST_METHODS = ("serialized", "gru", "transformer", "lstr")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if status.strip():
        raise RuntimeError("formal E1 test execution requires a clean source worktree")
    return {"source_commit": commit, "source_worktree_clean": True}


def _validate_test_data_binding(
    *,
    report_path: Path,
    rows_path: Path,
    catalogs_path: Path,
    train_report: dict[str, Any],
) -> dict[str, Any]:
    report = _read_json(report_path)
    if (
        report.get("schema_version") != MATCHED_HISTORY_TEST_DATA_SCHEMA
        or report.get("status") != "ok"
        or not bool(report.get("source_worktree_clean"))
        or not str(report.get("source_commit") or "")
    ):
        raise ValueError("E1 test-data report is incomplete")
    if Path(str(report.get("output_path") or "")).resolve() != rows_path:
        raise ValueError("E1 test rows differ from their data report")
    if str(report.get("output_sha256") or "") != file_sha256(rows_path):
        raise ValueError("E1 test rows changed after materialization")
    if Path(str(report.get("output_catalogs_path") or "")).resolve() != catalogs_path:
        raise ValueError("E1 test catalogs differ from their data report")
    if str(report.get("output_catalogs_sha256") or "") != file_sha256(catalogs_path):
        raise ValueError("E1 test catalogs changed after materialization")
    split_audit = report.get("split_audit") or {}
    if any(
        int(split_audit.get(key) or 0) != 0
        for key in ("train_dev_overlap", "train_test_overlap", "dev_test_overlap")
    ) or bool(split_audit.get("checkpoint_selection_uses_test")):
        raise ValueError("E1 test-data split audit is unsafe")
    report_inputs = report.get("inputs") or {}
    training_inputs = train_report.get("inputs") or {}
    for test_name, train_name in (
        ("train", "train_rows_sha256"),
        ("dev", "dev_rows_sha256"),
    ):
        observed = str((report_inputs.get(test_name) or {}).get("sha256") or "")
        if observed != str(training_inputs.get(train_name) or ""):
            raise ValueError(f"E1 test-data {test_name} binding differs from training")
    contract = report.get("contract") or {}
    if (
        not bool(contract.get("test_checkpoint_selection_forbidden"))
        or not bool(contract.get("history_bearing_decisions_only"))
        or int(contract.get("maximum_horizon_applied_by_evaluator") or 0) != 16
    ):
        raise ValueError("E1 test-data contract differs")
    return report


def _validate_checkpoint_binding(
    *,
    checkpoint_path: Path,
    train_report_path: Path,
    encoder_kind: str,
    seed: int,
    foundation_checkpoint_path: Path,
    skills_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(train_report_path)
    if (
        report.get("schema_version") != MATCHED_HISTORY_TRAIN_SCHEMA
        or report.get("status") != "ok"
    ):
        raise ValueError("E1 train report is incomplete")
    if str(report.get("encoder_kind") or "") != encoder_kind:
        raise ValueError("E1 test method differs from its train report")
    if int(report.get("seed") or -1) != int(seed):
        raise ValueError("E1 test seed differs from its train report")
    if Path(str(report.get("best_checkpoint_path") or "")).resolve() != checkpoint_path:
        raise ValueError("E1 test checkpoint is not the dev-selected checkpoint")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if str(report.get("best_checkpoint_sha256") or "") != checkpoint_sha256:
        raise ValueError("E1 test checkpoint digest differs from its train report")
    contract = report.get("contract") or {}
    expected_contract = {
        "max_horizon": 16,
        "support_k": 500,
        "belief_top_k": 64,
        "candidate_support": "bounded_last8_skill_action_frozen_static_top500",
        "same_shared_scorer": True,
        "frozen_foundation": True,
        "dynamic_candidate_extras": False,
        "selector": False,
        "history_decisions_only": True,
        "checkpoint_selection": "macro_dev_mrr_across_toolbench_g3_and_tau2",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"E1 test contract differs for {key}")
    foundation = (report.get("inputs") or {}).get("foundation") or {}
    if Path(str(foundation.get("checkpoint_path") or "")).resolve() != foundation_checkpoint_path:
        raise ValueError("E1 test foundation path differs from training")
    if str(foundation.get("checkpoint_sha256") or "") != file_sha256(
        foundation_checkpoint_path
    ):
        raise ValueError("E1 test foundation digest differs from training")
    if Path(str(foundation.get("training_skills_path") or "")).resolve() != skills_path:
        raise ValueError("E1 test skill table differs from training")
    if str(foundation.get("training_skills_sha256") or "") != file_sha256(skills_path):
        raise ValueError("E1 test skill-table digest differs from training")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != MATCHED_HISTORY_TRAIN_SCHEMA
        or str(payload.get("encoder_kind") or "") != encoder_kind
        or int(payload.get("seed") or -1) != int(seed)
        or int(payload.get("step") or 0) != int(report.get("best_step") or -1)
    ):
        raise ValueError("E1 test checkpoint payload differs from its selected report")
    if not isinstance(payload.get("encoder_state_dict"), dict) or not isinstance(
        payload.get("scorer_state_dict"), dict
    ):
        raise ValueError("E1 test checkpoint lacks encoder/scorer weights")
    return report, payload


def evaluate_matched_history_e1_test(
    *,
    encoder_kind: str,
    seed: int,
    foundation_checkpoint_path: str | Path,
    skills_path: str | Path,
    test_rows_path: str | Path,
    test_data_report_path: str | Path,
    inventory_catalogs_path: str | Path,
    checkpoint_path: str | Path,
    train_report_path: str | Path,
    output_dir: str | Path,
    frozen_cache_dir: str | Path,
    validation_batch_size: int = 32,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
) -> dict[str, Any]:
    if encoder_kind not in MATCHED_HISTORY_TEST_METHODS:
        raise ValueError(f"unsupported E1 test encoder: {encoder_kind}")
    if int(validation_batch_size) <= 0:
        raise ValueError("E1 test batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("matched-history E1 test evaluation requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    root = Path(__file__).resolve().parents[1]
    git_identity = _git_identity(root)

    foundation_path = Path(foundation_checkpoint_path).resolve()
    skills = Path(skills_path).resolve()
    test_rows = Path(test_rows_path).resolve()
    test_data_report = Path(test_data_report_path).resolve()
    catalogs_path = Path(inventory_catalogs_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    train_report_file = Path(train_report_path).resolve()
    required = (
        foundation_path,
        skills,
        test_rows,
        test_data_report,
        catalogs_path,
        checkpoint,
        train_report_file,
    )
    if any(not path.is_file() for path in required):
        raise ValueError("E1 test evaluation is missing an immutable input")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("E1 test output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    train_report, checkpoint_payload = _validate_checkpoint_binding(
        checkpoint_path=checkpoint,
        train_report_path=train_report_file,
        encoder_kind=encoder_kind,
        seed=int(seed),
        foundation_checkpoint_path=foundation_path,
        skills_path=skills,
    )
    data_report = _validate_test_data_binding(
        report_path=test_data_report,
        rows_path=test_rows,
        catalogs_path=catalogs_path,
        train_report=train_report,
    )
    if str(data_report.get("source_commit") or "") != str(
        git_identity["source_commit"]
    ):
        raise ValueError("E1 test data and evaluator do not share one source commit")
    started = time.perf_counter()
    foundation, _skills, skill_id_to_idx, foundation_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=foundation_path,
            training_skills_path=skills,
            benchmark_skills=[],
            device=device,
        )
    )
    foundation.eval()
    for parameter in foundation.parameters():
        parameter.requires_grad_(False)
    catalogs = load_inventory_catalogs(catalogs_path)
    test_anchors = _anchors(load_matched_history_trajectories(test_rows))
    all_selected = [
        item for collection in test_anchors.values() for item in collection
    ]
    state_texts, action_texts, result_texts, history_texts = _collect_cache_texts(
        all_selected
    )
    cache_root = Path(frozen_cache_dir).resolve()
    state_cache = load_or_build_frozen_text_cache(
        foundation,
        state_texts,
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    action_cache = load_or_build_frozen_text_cache(
        foundation,
        action_texts,
        role="action",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    result_cache = load_or_build_frozen_text_cache(
        foundation,
        result_texts,
        role="result",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    history_cache = (
        load_or_build_frozen_text_cache(
            foundation,
            history_texts,
            role="matched_history",
            batch_size=cache_batch_size,
            cache_root=cache_root,
            cache_shard_size=cache_shard_size,
        )
        if encoder_kind == "serialized"
        else None
    )

    d = int(foundation.vnext.d)
    encoder = build_matched_history_encoder(
        encoder_kind, d, max_horizon=16
    ).to(device)
    scorer = MatchedHistoryRouteScorer(d).to(device)
    encoder.load_state_dict(checkpoint_payload["encoder_state_dict"], strict=True)
    scorer.load_state_dict(checkpoint_payload["scorer_state_dict"], strict=True)
    encoder.eval()
    scorer.eval()
    for parameter in (*encoder.parameters(), *scorer.parameters()):
        parameter.requires_grad_(False)
    metrics, predictions = _evaluate(
        encoder,
        scorer,
        test_anchors,
        foundation=foundation,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        history_cache=history_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        max_horizon=16,
        support_k=500,
        belief_top_k=64,
        batch_size=int(validation_batch_size),
        collect_predictions=True,
    )
    expected_count = int(data_report.get("history_bearing_decision_count") or 0)
    if len(predictions) != expected_count or sum(
        int(metrics[benchmark]["count"]) for benchmark in ("toolbench_g3", "tau2")
    ) != expected_count:
        raise RuntimeError("E1 test prediction count differs from test-data report")
    predictions_path = output / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    report = {
        "schema_version": MATCHED_HISTORY_TEST_RUN_SCHEMA,
        "status": "ok",
        "encoder_kind": encoder_kind,
        "seed": int(seed),
        **git_identity,
        "contract": {
            "max_horizon": 16,
            "candidate_support": "bounded_last8_skill_action_frozen_static_top500",
            "support_k": 500,
            "belief_top_k": 64,
            "frozen_foundation": True,
            "frozen_dev_selected_checkpoint": True,
            "optimizer_steps": 0,
            "checkpoint_selection": "preselected_on_disjoint_e1_dev",
            "test_access": "single_frozen_evaluation",
            "history_decisions_only": True,
            "dynamic_candidate_extras": False,
            "selector": False,
        },
        "metrics": metrics,
        "test_prediction_count": len(predictions),
        "artifacts": {
            "test_predictions_path": str(predictions_path),
            "test_predictions_sha256": file_sha256(predictions_path),
        },
        "inputs": {
            "foundation": foundation_report,
            "foundation_checkpoint_path": str(foundation_path),
            "foundation_checkpoint_sha256": file_sha256(foundation_path),
            "skills_path": str(skills),
            "skills_sha256": file_sha256(skills),
            "test_rows_path": str(test_rows),
            "test_rows_sha256": file_sha256(test_rows),
            "test_data_report_path": str(test_data_report),
            "test_data_report_sha256": file_sha256(test_data_report),
            "inventory_catalogs_path": str(catalogs_path),
            "inventory_catalogs_sha256": file_sha256(catalogs_path),
            "selected_checkpoint_path": str(checkpoint),
            "selected_checkpoint_sha256": file_sha256(checkpoint),
            "train_report_path": str(train_report_file),
            "train_report_sha256": file_sha256(train_report_file),
            "e1_training_source_commit": str(train_report.get("source_commit") or ""),
            "e1_selected_step": int(train_report.get("best_step") or 0),
        },
        "cache": {
            "state": state_cache.report(),
            "action": action_cache.report(),
            "result": result_cache.report(),
            "matched_history": None if history_cache is None else history_cache.report(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    report_path = output / "test_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return report
