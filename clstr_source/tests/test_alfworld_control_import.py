from __future__ import annotations

import json
from pathlib import Path

import pytest

from clstr.alfworld_control_import import (
    import_alfworld_qwen_control,
    load_alfworld_qwen_control_import,
)
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path


QWEN_DIGEST = "c8c9a22462bf04dee2cbce3e8d216819e28b46cbe50b247083d91cdb546acfe5"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_control(tmp_path: Path, *, split: str = "valid_seen") -> tuple[Path, Path]:
    source = tmp_path / "source"
    rows = [
        {
            "episode_index": index,
            "split": split,
            "gamefile": f"/games/{split}/game-{index}.tw-pddl",
            "chosen_action_trace": ["look", f"action-{index}"],
            "success": index in {1, 2},
            "points": float(index in {1, 2}),
            "goal_condition_points": float(index) / 10.0,
            "steps": float(10 + index),
        }
        for index in range(3)
    ]
    run_path = source / "run.jsonl"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = {
        "status": "ok",
        "method": "qwen3_14b_qwen_only_qwen_only",
        "run_name": "qwen3_14b_qwen_only_qwen_only",
        "split": split,
        "success_rate": 2.0 / 3.0,
        "average_reward": 2.0 / 3.0,
        "average_goal_condition_points": 0.1,
        "average_episode_steps": 11.0,
        "average_steps": 11.0,
        "episode_count": 3,
        "episodes": 3,
    }
    _write_json(source / "metrics.json", metrics)
    identity = {
        "schema_version": 1,
        "protocol_version": "method_alfworld_qwen3_14b_executor_v1",
        "method": "qwen_only",
        "split": split,
        "backbone_family": "qwen_only",
        "qwen_model_path": "/models/Qwen3-14B",
        "qwen_model_digest": QWEN_DIGEST,
        "qwen_weight": 1.0,
        "prior_weight": 0.0,
        "checkpoint_binding": None,
        "benchmark_manifest_sha256": "b" * 64,
        "loop_guard": True,
        "max_steps": 50,
    }
    identity["identity_sha256"] = canonical_digest(identity)
    _write_json(source / "method_executor_identity.json", identity)
    report = {
        "schema_version": 1,
        "status": "ok",
        "protocol_version": "method_alfworld_qwen3_14b_executor_v1",
        "method": "qwen_only",
        "split": split,
        "backbone_family": "qwen_only",
        "executor_model_family": "qwen3_14b",
        "executor_scoring_method": "generate",
        "external_llm_executor": True,
        "qwen_model_path": "/models/Qwen3-14B",
        "qwen_model_digest": QWEN_DIGEST,
        "qwen_weight": 1.0,
        "prior_weight": 0.0,
        "checkpoint_binding": None,
        "benchmark_manifest_sha256": "b" * 64,
        "loop_guard": True,
        "max_steps": 50,
        "action_selection": "qwen_executor_only",
        "executor_identity_sha256": identity["identity_sha256"],
        "run_path": str(run_path.resolve()),
        "metrics": metrics,
        "trace_summary": {"episode_count": 3, "total_steps": 33},
    }
    report["report_sha256"] = canonical_digest(report)
    _write_json(source / "method_alfworld_executor_report.json", report)
    prefix_reference = tmp_path / "prefix_reference.jsonl"
    prefix_reference.write_text(run_path.read_text(encoding="utf-8"), encoding="utf-8")
    return source, prefix_reference


def test_import_alfworld_qwen_control_writes_provenance_bound_artifacts(
    tmp_path: Path,
) -> None:
    source, prefix_reference = _source_control(tmp_path)
    output = tmp_path / "output"

    manifest = import_alfworld_qwen_control(
        source_dir=source,
        output_dir=output,
        split="valid_seen",
        prefix_reference_run_path=prefix_reference,
        prefix_rows=2,
    )

    loaded = load_alfworld_qwen_control_import(
        output / "control_import.json",
        expected_split="valid_seen",
    )
    assert loaded == manifest
    assert manifest["status"] == "ok"
    assert manifest["episode_count"] == 3
    assert manifest["prefix_equivalence"]["rows"] == 2
    assert manifest["prefix_equivalence"]["all_equal"] is True
    assert manifest["source"]["run"] == sha256_path(source / "run.jsonl")
    assert manifest["imported"]["run"] == sha256_path(
        output / "qwen_only" / "run.jsonl"
    )
    metrics = json.loads(
        (output / "qwen_only" / "metrics.json").read_text(encoding="utf-8")
    assert metrics["qwen_direct_baseline"] is True
    assert metrics["uses_clstr"] is False
    assert metrics["executor_gate"] is True
    assert metrics["scoring_method"] == "generate"
    assert metrics["qwen_model_digest"] == QWEN_DIGEST
    payload = dict(manifest)
    recorded = payload.pop("manifest_sha256")
    assert recorded == canonical_digest(payload)


def test_import_alfworld_qwen_control_filters_to_declared_smoke_gamefiles(
    tmp_path: Path,
) -> None:
    source, prefix_reference = _source_control(tmp_path)
    split_manifest = {
        "schema_version": 1,
        "benchmark": "alfworld",
        "split": "valid_seen",
        "canonical_source_rows": [
            {"gamefile": "/games/valid_seen/game-0.tw-pddl"},
            {"gamefile": "/games/valid_seen/game-2.tw-pddl"},
        ],
    }
    split_manifest["manifest_sha256"] = canonical_digest(split_manifest)
    split_manifest_path = tmp_path / "split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    output = tmp_path / "smoke"

    manifest = import_alfworld_qwen_control(
        source_dir=source,
        output_dir=output,
        split="valid_seen",
        prefix_reference_run_path=prefix_reference,
        split_manifest_path=split_manifest_path,
        prefix_rows=2,
    )

    assert manifest["episode_count"] == 2
    assert manifest["split_manifest"] == sha256_path(split_manifest_path)
    metrics = json.loads(
        (output / "qwen_only" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["episode_count"] == 2
    assert metrics["success_rate"] == pytest.approx(0.5)


def test_import_alfworld_qwen_control_rejects_prefix_drift(tmp_path: Path) -> None:
    source, prefix_reference = _source_control(tmp_path)
    rows = [json.loads(line) for line in prefix_reference.read_text().splitlines()]
    rows[1]["chosen_action_trace"] = ["different"]
    prefix_reference.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prefix equivalence"):
        import_alfworld_qwen_control(
            source_dir=source,
            output_dir=tmp_path / "output",
            split="valid_seen",
            prefix_reference_run_path=prefix_reference,
            prefix_rows=2,
        )


def test_load_alfworld_qwen_control_import_rejects_tampered_artifact(
    tmp_path: Path,
) -> None:
    source, prefix_reference = _source_control(tmp_path)
    output = tmp_path / "output"
    import_alfworld_qwen_control(
        source_dir=source,
        output_dir=output,
        split="valid_seen",
        prefix_reference_run_path=prefix_reference,
        prefix_rows=2,
    )
    with (output / "qwen_only" / "run.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="imported run identity"):
        load_alfworld_qwen_control_import(
            output / "control_import.json",
            expected_split="valid_seen",
        )
