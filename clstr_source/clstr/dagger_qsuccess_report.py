from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.external_data import write_json


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _prefer_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _metric(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    return {
        "status": data.get("status", "missing" if not data else "ok"),
        "success_rate": data.get("success_rate"),
        "average_reward": data.get("average_reward"),
        "average_goal_condition_points": data.get("average_goal_condition_points"),
        "average_episode_steps": data.get("average_episode_steps"),
        "episodes": data.get("episodes", data.get("episode_count", 0)),
        "path": str(path),
        "not_clstr_result": bool(data.get("not_clstr_result", False)),
    }


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _improvement(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cand = _float(candidate.get("success_rate"))
    base = _float(baseline.get("success_rate"))
    if cand is None or base is None:
        return {"improved": False, "delta_success_rate": None}
    return {"improved": cand > base, "delta_success_rate": round(cand - base, 6)}


def _fmt(value: Any) -> str:
    number = _float(value)
    return "N/A" if number is None else f"{number:.6f}"


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | status | CLSTR result | success_rate | avg_reward | avg_gcp | avg_steps | episodes | path |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {is_clstr} | {sr} | {reward} | {gcp} | {steps} | {episodes} | {path} |".format(
                method=row["method"],
                status=row["metrics"]["status"],
                is_clstr=not bool(row["metrics"].get("not_clstr_result", False)),
                sr=_fmt(row["metrics"].get("success_rate")),
                reward=_fmt(row["metrics"].get("average_reward")),
                gcp=_fmt(row["metrics"].get("average_goal_condition_points")),
                steps=_fmt(row["metrics"].get("average_episode_steps")),
                episodes=row["metrics"].get("episodes", 0),
                path=row["metrics"].get("path"),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Qwen direct is a reference baseline, not a CLSTR result.",
            "- DAgger expert correction uses ALFWorld train rollout states and official expert actions.",
            "- Q_success is an estimated action-value head, not an oracle success probability.",
            "- Offline diagnostics are not closed-loop success.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dagger_qsuccess_reports(
    output_root: str | Path = "outputs",
    data_root: str | Path = "data",
    output_table_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    output_paper_md: str | Path | None = None,
    output_paper_json: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    data_root = Path(data_root)
    output_table_path = Path(output_table_path) if output_table_path else output_root / "clstr_dagger_qsuccess_comparison_table.md"
    output_summary_path = Path(output_summary_path) if output_summary_path else output_root / "clstr_dagger_qsuccess_comparison_summary.json"
    output_paper_md = Path(output_paper_md) if output_paper_md else output_root / "clstr_dagger_qsuccess_paper_report.md"
    output_paper_json = Path(output_paper_json) if output_paper_json else output_root / "clstr_dagger_qsuccess_paper_report.json"

    eval_root = output_root / "alfworld_eval"
    rows = [
        {
            "method": "0.6B CLSTR controller best",
            "metrics": _metric(eval_root / "clstr_controller_gate_full_2k" / "metrics.json"),
        },
        {
            "method": "Qwen3-8B direct reference",
            "metrics": _metric(eval_root / "qwen3_8b_chat_available_full" / "metrics.json"),
        },
        {
            "method": "previous structured CLSTR-Qwen gate",
            "metrics": _metric(eval_root / "clstr_qwen3_structured_gate_valid_seen" / "metrics.json"),
        },
        {
            "method": "DAgger + Q_success gate",
            "metrics": _metric(eval_root / "clstr_dagger_qsuccess_gate_valid_seen" / "metrics.json"),
        },
        {
            "method": "DAgger + Q_success full",
            "metrics": _metric(eval_root / "clstr_dagger_qsuccess_full" / "metrics.json"),
        },
    ]
    by_method = {row["method"]: row["metrics"] for row in rows}
    dagger_data_root = _prefer_existing(
        data_root / "clstr_dagger_expert_corrected_train_enriched",
        data_root / "clstr_dagger_expert_corrected_train",
    )
    dagger_preprocess_path = _prefer_existing(
        output_root / "clstr_dagger_success_train_enriched" / "preprocess_report.json",
        output_root / "clstr_dagger_expert_corrected_train_enriched" / "preprocess_report.json",
        output_root / "clstr_dagger_expert_corrected_train" / "preprocess_report.json",
    )
    train_report_path = _prefer_existing(
        output_root / "clstr_dagger_success_train_enriched" / "train_report.json",
        output_root / "clstr_dagger_success_train" / "train_report.json",
    )
    eval_report_path = _prefer_existing(
        output_root / "clstr_dagger_success_train_enriched" / "eval_report.json",
        output_root / "clstr_dagger_success_train" / "eval_report.json",
    )
    ablation_summary_path = _prefer_existing(
        output_root / "clstr_loss_ablation_enriched" / "ablation_summary.json",
        output_root / "clstr_loss_ablation" / "ablation_summary.json",
    )
    ablation_table_path = _prefer_existing(
        output_root / "clstr_loss_ablation_enriched" / "ablation_table.md",
        output_root / "clstr_loss_ablation" / "ablation_table.md",
    )
    rollout_report = _read_json(output_root / "alfworld_qwen3_expert_corrected_rollout" / "extraction_report.json")
    rollout_manifest = _read_json(data_root / "alfworld_qwen3_expert_corrected_rollout" / "manifest.json")
    preprocess_report = _read_json(dagger_preprocess_path)
    train_report = _read_json(train_report_path)
    eval_report = _read_json(eval_report_path)
    ablation_summary = _read_json(ablation_summary_path)
    mc_report = _read_json(output_root / "alfworld_mc_success_labels" / "report.json")
    blocker_path = output_root / "alfworld_eval" / "clstr_dagger_qsuccess_gate_valid_seen" / "blocker_report.json"
    blocker_report = _read_json(blocker_path)

    gate = by_method["DAgger + Q_success gate"]
    improvements = {
        "vs_0_6b_best": _improvement(gate, by_method["0.6B CLSTR controller best"]),
        "vs_previous_structured_clstr_qwen": _improvement(gate, by_method["previous structured CLSTR-Qwen gate"]),
        "vs_qwen_direct_reference": _improvement(gate, by_method["Qwen3-8B direct reference"]),
    }
    recommended = ablation_summary.get("recommended_slim_loss_config") or {}
    best_ablation = ablation_summary.get("best_by_success_rate") or {}
    best_train_report = best_ablation.get("train_report") if isinstance(best_ablation.get("train_report"), dict) else train_report

    summary = {
        "status": "ok",
        "rows": rows,
        "rollout": {
            "data_path": str(data_root / "alfworld_qwen3_expert_corrected_rollout" / "train_rollout.jsonl"),
            "manifest_path": str(data_root / "alfworld_qwen3_expert_corrected_rollout" / "manifest.json"),
            "report_path": str(output_root / "alfworld_qwen3_expert_corrected_rollout" / "extraction_report.json"),
            "manifest": rollout_manifest,
            "stats": rollout_report,
            "train_split_only": rollout_manifest.get("split") == "train" or rollout_manifest.get("train_split_only", True),
        },
        "dagger_train_data": {
            "path": str(dagger_data_root / "train.jsonl"),
            "preprocess_report_path": str(dagger_preprocess_path),
            "preprocess_report": preprocess_report,
        },
        "train_report_path": str(train_report_path),
        "eval_report_path": str(eval_report_path),
        "checkpoint": best_ablation.get("checkpoint") or train_report.get("checkpoint"),
        "train_report": train_report,
        "best_checkpoint_train_report": best_train_report,
        "eval_report": eval_report,
        "ablation_summary_path": str(ablation_summary_path),
        "ablation_table_path": str(ablation_table_path),
        "recommended_slim_loss_config": recommended,
        "best_ablation": best_ablation,
        "improvements": improvements,
        "blocker_report_path": str(blocker_path) if blocker_report else None,
        "blocker_report": blocker_report or None,
        "monte_carlo_success_labeling_executed": bool(mc_report),
        "monte_carlo_success_labeling_report": mc_report or None,
        "paper_safe_caveats": [
            "Qwen direct is a reference baseline, not a CLSTR result.",
            "Q_success is an estimated action-value, not an oracle probability.",
            "offline diagnostics are not closed-loop success.",
            "ALFWorld valid/test are not used for training or rollout.",
        ],
    }

    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text(_table(rows), encoding="utf-8")
    write_json(output_summary_path, summary)

    paper_lines = [
        "# CLSTR-Qwen DAgger + Q_success Report",
        "",
        "## Data",
        "",
        f"- Expert-corrected rollout: {summary['rollout']['data_path']}",
        f"- Train split only: {summary['rollout']['train_split_only']}",
        f"- Extracted steps: {rollout_report.get('extracted_step_count', 'N/A')}",
        f"- expert_action_in_admissible_rate: {_fmt(rollout_report.get('expert_action_in_admissible_rate'))}",
        f"- qwen_action_matches_expert_rate: {_fmt(rollout_report.get('qwen_action_matches_expert_rate'))}",
        "",
        "## Training",
        "",
        f"- Checkpoint: {summary['checkpoint']}",
        f"- Offline policy_expert_recall@1: {_fmt((best_train_report.get('metrics') or {}).get('policy_expert_recall@1'))}",
        f"- Offline Q_success accuracy@0.5: {_fmt((best_train_report.get('metrics') or {}).get('q_success_accuracy@0.5'))}",
        f"- Offline transition_skill_recall@1: {_fmt((best_train_report.get('metrics') or {}).get('transition_skill_recall@1'))}",
        "",
        "## Loss Ablation",
        "",
        f"- Table: {summary['ablation_table_path']}",
        f"- Recommended slim config: {recommended}",
        "",
        "## Closed-Loop Results",
        "",
        _table(rows),
        "",
        "## Interpretation",
        "",
        "- Previous Qwen teacher rollout could not be treated as expert because Qwen actions can diverge from official expert actions.",
        "- DAgger expert correction uses official ALFWorld expert actions at visited train states as L_policy positives.",
        "- Q_success is an estimated action-value head, not an oracle success probability.",
        "- offline diagnostics are not closed-loop success.",
        "- If the gate fails, the blocker report identifies whether expert correction, Q_success, skill grounding, controller calibration, or long-horizon planning is limiting.",
    ]
    output_paper_md.write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    write_json(output_paper_json, {"status": "ok", "summary": summary, "paper_md": str(output_paper_md)})
    return summary
