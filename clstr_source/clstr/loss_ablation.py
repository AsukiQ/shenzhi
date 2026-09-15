from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.external_data import write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _normalise_action(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _weights(**overrides: float) -> dict[str, float]:
    base = {
        "L_policy": 0.0,
        "hard_negative_margin": 0.0,
        "Q_success": 0.0,
        "L_trans": 0.0,
        "L_trans_skill_ce": 0.0,
        "belief": 0.0,
        "STOP": 0.0,
        "routing": 0.0,
    }
    base.update({key: float(value) for key, value in overrides.items()})
    return base


ABLATION_CONFIGS: list[dict[str, Any]] = [
    {"name": "policy_only", "loss_weights": _weights(L_policy=1.0)},
    {"name": "policy_plus_hard_negative", "loss_weights": _weights(L_policy=1.0, hard_negative_margin=0.2)},
    {"name": "policy_plus_q_success", "loss_weights": _weights(L_policy=1.0, Q_success=0.5)},
    {
        "name": "policy_plus_q_success_trans_skill_ce",
        "loss_weights": _weights(L_policy=1.0, Q_success=0.5, L_trans_skill_ce=0.2),
    },
    {"name": "policy_plus_q_success_stop", "loss_weights": _weights(L_policy=1.0, Q_success=0.5, STOP=0.2)},
    {"name": "policy_plus_q_success_belief", "loss_weights": _weights(L_policy=1.0, Q_success=0.5, belief=0.05)},
    {"name": "policy_plus_q_success_routing", "loss_weights": _weights(L_policy=1.0, Q_success=0.5, routing=0.2)},
    {
        "name": "full",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.02,
            L_trans_skill_ce=0.2,
            belief=0.05,
            STOP=0.2,
            routing=0.2,
        ),
    },
    {
        "name": "full_minus_observation_cosine",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.0,
            L_trans_skill_ce=0.2,
            belief=0.05,
            STOP=0.2,
            routing=0.2,
        ),
    },
    {
        "name": "full_minus_STOP",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.02,
            L_trans_skill_ce=0.2,
            belief=0.05,
            STOP=0.0,
            routing=0.2,
        ),
    },
    {
        "name": "full_minus_belief",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.02,
            L_trans_skill_ce=0.2,
            belief=0.0,
            STOP=0.2,
            routing=0.2,
        ),
    },
    {
        "name": "full_minus_routing",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.02,
            L_trans_skill_ce=0.2,
            belief=0.05,
            STOP=0.2,
            routing=0.0,
        ),
    },
    {
        "name": "full_minus_trans_skill_ce",
        "loss_weights": _weights(
            L_policy=1.0,
            hard_negative_margin=0.2,
            Q_success=0.5,
            L_trans=0.02,
            L_trans_skill_ce=0.0,
            belief=0.05,
            STOP=0.2,
            routing=0.2,
        ),
    },
]


def _metric(report: dict[str, Any], key: str) -> float:
    metrics = report.get("eval_metrics") or {}
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _active_loss_count(loss_weights: dict[str, Any]) -> int:
    return sum(1 for value in loss_weights.values() if float(value) > 0.0)


def recommended_slim_config(ablation_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not ablation_reports:
        return {"status": "empty", "source_ablation": None, "loss_weights": {}}
    best = sorted(
        ablation_reports,
        key=lambda row: (
            -_metric(row, "success_rate"),
            -_metric(row, "average_reward"),
            _active_loss_count((row.get("train_report") or {}).get("loss_weights") or {}),
            str(row.get("name") or ""),
        ),
    )[0]
    return {
        "status": "ok",
        "source_ablation": best.get("name"),
        "checkpoint": best.get("checkpoint"),
        "success_rate": _metric(best, "success_rate"),
        "average_reward": _metric(best, "average_reward"),
        "loss_weights": dict((best.get("train_report") or {}).get("loss_weights") or {}),
    }


def _markdown_table(ablation_reports: list[dict[str, Any]]) -> str:
    lines = [
        "| ablation | success_rate | avg_reward | avg_steps | episodes | checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in ablation_reports:
        metrics = row.get("eval_metrics") or {}
        lines.append(
            "| {name} | {sr:.6f} | {reward:.6f} | {steps:.3f} | {episodes} | {ckpt} |".format(
                name=row.get("name"),
                sr=float(metrics.get("success_rate", 0.0) or 0.0),
                reward=float(metrics.get("average_reward", 0.0) or 0.0),
                steps=float(metrics.get("average_episode_steps", 0.0) or 0.0),
                episodes=int(metrics.get("episodes", metrics.get("episode_count", 0)) or 0),
                ckpt=row.get("checkpoint") or "",
            )
        )
    return "\n".join(lines) + "\n"


def build_ablation_summary(ablation_reports: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = max(ablation_reports, key=lambda row: (_metric(row, "success_rate"), _metric(row, "average_reward")), default=None)
    slim = recommended_slim_config(ablation_reports)
    summary = {
        "status": "ok",
        "ablation_count": len(ablation_reports),
        "best_by_success_rate": best,
        "recommended_slim_loss_config": slim,
        "ablation_reports": ablation_reports,
    }
    write_json(output_dir / "ablation_summary.json", summary)
    (output_dir / "ablation_table.md").write_text(_markdown_table(ablation_reports), encoding="utf-8")
    write_json(output_dir / "recommended_slim_loss_config.json", slim)
    return summary


def build_ablation_action_trace_diagnostic(
    eval_output_dir: str | Path,
    train_report: dict[str, Any],
    offline_report: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    eval_output_dir = Path(eval_output_dir)
    controller_diagnostic = _read_json(eval_output_dir / "controller_diagnostic.json")
    trace = controller_diagnostic.get("action_trace_analysis") or {}
    stuck = trace.get("stuck_patterns") or {}
    episodes = int(trace.get("episodes") or 0)
    stuck_episode_count = min(
        episodes,
        int(stuck.get("consecutive_repeat_ge_5_episodes") or 0)
        + int(stuck.get("single_action_only_episodes") or 0)
        + int(stuck.get("tail_two_action_cycle_episodes") or 0),
    )

    accepted = 0
    rejected = 0
    missing = 0
    fallback_count = 0
    parse_status_counts: dict[str, int] = {}
    run_path = eval_output_dir / "run.jsonl"
    if run_path.exists():
        for line in run_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            actions = [str(item) for item in row.get("action_trace") or row.get("chosen_action_trace") or []]
            metadata_trace = row.get("policy_metadata_trace") or []
            for idx, metadata in enumerate(metadata_trace):
                if not isinstance(metadata, dict):
                    continue
                proposed = metadata.get("qwen_proposed_action")
                if not proposed:
                    missing += 1
                    continue
                chosen = actions[idx] if idx < len(actions) else ""
                if _normalise_action(proposed) == _normalise_action(chosen):
                    accepted += 1
                else:
                    rejected += 1
                parse_status = str(metadata.get("qwen_parse_status") or metadata.get("parse_status") or "unknown")
                parse_status_counts[parse_status] = parse_status_counts.get(parse_status, 0) + 1
                if bool(metadata.get("qwen_fallback_used") or metadata.get("fallback_used")):
                    fallback_count += 1

    train_metrics = (train_report.get("metrics") or {}) if isinstance(train_report, dict) else {}
    offline_metrics = (offline_report.get("metrics") or {}) if isinstance(offline_report, dict) else {}
    expert_recall = train_metrics.get("policy_expert_recall@1", offline_metrics.get("policy_expert_recall@1"))
    diagnostic = {
        "status": "ok",
        "eval_output_dir": str(eval_output_dir),
        "episodes": episodes,
        "top_verbs": trace.get("top_verbs") or [],
        "top_actions": trace.get("top_actions") or [],
        "stuck_patterns": stuck,
        "loop_or_repeat_episode_count": stuck_episode_count,
        "loop_or_repeat_episode_rate": round(float(stuck_episode_count / max(1, episodes)), 6),
        "offline_policy_expert_recall@1": expert_recall,
        "offline_policy_ce_loss": offline_metrics.get("policy_ce_loss", train_metrics.get("policy_ce_loss")),
        "q_success_accuracy@0.5": offline_metrics.get("q_success_accuracy@0.5", train_metrics.get("q_success_accuracy@0.5")),
        "qwen_proposal_stats": {
            "proposal_count": accepted + rejected,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "missing_count": missing,
            "accept_rate": round(float(accepted / max(1, accepted + rejected)), 6),
            "parse_status_counts": dict(sorted(parse_status_counts.items())),
            "fallback_count": fallback_count,
        },
        "not_closed_loop_success": True,
        "diagnostic_role": "ablation_trace_diagnostic_only",
    }
    if output_path is not None:
        write_json(Path(output_path), diagnostic)
    return diagnostic
