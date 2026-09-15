from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

from clstr.prior_damage_slices import (
    build_prior_damage_slice_report,
    load_prior_damage_inputs,
    write_prior_damage_slice_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_cli_module():
    path = Path("scripts/audit_clstr_prior_damage_slices.py")
    spec = importlib.util.spec_from_file_location("audit_clstr_prior_damage_slices", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_prior_damage_slices_from_row_records_and_handoff_counts():
    report = build_prior_damage_slice_report(
        row_records=[
            {
                "stage0_gold_rank": 1,
                "stage2_gold_rank": 3,
                "stage2_hit@5": True,
                "benchmark": "toolbench_g3",
                "transition_relation": "switch",
            },
            {
                "stage0_gold_rank": 8,
                "stage2_gold_rank": 2,
                "stage2_hit@5": True,
                "benchmark": "webshop",
                "transition_relation": "self",
            },
            {
                "stage0_gold_rank": 15,
                "stage2_gold_rank": 30,
                "stage2_hit@5": False,
                "benchmark": "webshop",
                "transition_relation": "switch",
            },
        ],
        handoff_report={
            "next_positive_required_rows": 100,
            "next_positive_covered_rows": 90,
            "masked_next_skill_ce_rows": 10,
            "next_positive_rank_bucket_counts": {
                "1": 30,
                "2-5": 20,
                "6-20": 25,
                "21-100": 10,
                ">100": 5,
                "missing": 10,
            },
        },
        metric_rows=[],
        run_id="synthetic",
    )

    assert report["run_id"] == "synthetic"
    assert report["slices"]["gt_missing_from_topM"]["count"] == 10
    assert report["slices"]["stage0_top1_wrong_top5_contains_gt"]["count"] == 20
    assert report["slices"]["stage0_top1_wrong_top20_contains_gt"]["count"] == 45
    assert report["slices"]["clstr_worse_than_stage0_prior"]["count"] == 2
    assert report["slices"]["transition_switch_rows"]["count"] == 2
    assert report["by_benchmark"]["webshop"]["clstr_worse_than_stage0_prior"]["count"] == 1
    assert report["decision"]["primary_bottleneck"] == "mixed_or_unclear"


def test_build_prior_damage_slices_uses_metric_tail_when_no_rows():
    metric_rows = [
        {
            "step": 1,
            "transition_skill_ce_count": 10,
            "transition_worse_than_stage0_prior_fraction": 0.1,
            "transition_delta_vs_stage0_prior_mrr": 0.05,
            "transition_skill_recall@5": 0.7,
            "stage0_prior_transition_skill_recall@5": 0.72,
            "transition_current_skill_switch_rows": 4,
        },
        {
            "step": 2,
            "transition_skill_ce_count": 20,
            "transition_worse_than_stage0_prior_fraction": 0.25,
            "transition_delta_vs_stage0_prior_mrr": -0.03,
            "transition_skill_recall@5": 0.6,
            "stage0_prior_transition_skill_recall@5": 0.75,
            "transition_current_skill_switch_rows": 6,
        },
    ]

    report = build_prior_damage_slice_report(
        row_records=[],
        handoff_report={"next_positive_required_rows": 100, "masked_next_skill_ce_rows": 5},
        metric_rows=metric_rows,
        tail_window=1,
        run_id="metrics-only",
    )

    assert report["slices"]["clstr_worse_than_stage0_prior"]["estimated_count"] == 5.0
    assert report["slices"]["transition_switch_rows"]["estimated_count"] == 6.0
    assert report["metric_windows"]["last"]["transition_skill_recall@5"] == 0.6
    assert report["metric_windows"]["last"]["transition_delta_vs_stage0_prior_mrr"] == -0.03
    assert report["decision"]["primary_bottleneck"] == "prior_damage_or_gate"


def test_load_and_write_prior_damage_report(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage1_heads_init"
    _write_jsonl(stage_dir / "training_metrics.jsonl", [{"step": 1, "transition_skill_ce_count": 3}])
    (stage_dir / "stage0_candidate_handoff.json").write_text(
        json.dumps({"next_positive_required_rows": 3, "masked_next_skill_ce_rows": 1}),
        encoding="utf-8",
    )

    inputs = load_prior_damage_inputs(run_dir)

    assert inputs["metric_rows"][0]["step"] == 1
    assert inputs["handoff_report"]["masked_next_skill_ce_rows"] == 1

    output_path = tmp_path / "out" / "slice_report.json"
    written = write_prior_damage_slice_report(
        output_path,
        build_prior_damage_slice_report(
            row_records=inputs["row_records"],
            handoff_report=inputs["handoff_report"],
            metric_rows=inputs["metric_rows"],
            run_id="load-write",
        ),
    )

    assert written == output_path
    assert json.loads(output_path.read_text(encoding="utf-8"))["run_id"] == "load-write"


def test_cli_writes_slice_report(monkeypatch, tmp_path):
    module = _load_cli_module()
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage1_heads_init"
    _write_jsonl(stage_dir / "training_metrics.jsonl", [{"step": 1, "transition_skill_ce_count": 3}])
    (stage_dir / "stage0_candidate_handoff.json").write_text(
        json.dumps({"next_positive_required_rows": 3, "masked_next_skill_ce_rows": 1}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "slices"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_clstr_prior_damage_slices.py",
            "--run_dir",
            str(run_dir),
            "--output_dir",
            str(output_dir),
            "--run_id",
            "cli-test",
        ],
    )

    assert module.main() == 0

    report_path = output_dir / "slice_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == "cli-test"
    assert report["slices"]["gt_missing_from_topM"]["count"] == 1
