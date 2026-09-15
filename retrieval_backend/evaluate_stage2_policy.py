"""Evaluate the native Stage2 action policy on frozen executable dev traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

from multistep_search import MultiStepPaperSearch, Stage2ActionPolicy
from paper_search import PaperSearchIndex
from retrieval_pipeline import HybridPaperSearch


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[int(fraction * (len(ordered) - 1))])


def _load_trajectories(path: Path) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["trajectory_id"])].append(row)
    output = []
    for trajectory_id in sorted(grouped):
        rows = sorted(grouped[trajectory_id], key=lambda row: int(row["decision_step_index"]))
        if [int(row["decision_step_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous dev trajectory: {trajectory_id}")
        output.append(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--clstr-source", required=True)
    parser.add_argument("--dev-trajectories", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-trajectories", type=int, default=48)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prototype = Stage2ActionPolicy(
        checkpoint_path=args.checkpoint,
        skills_path=args.skills,
        clstr_source=args.clstr_source,
        device=args.device,
    )
    if not prototype.enabled:
        raise RuntimeError(f"Stage2 policy failed to load: {prototype.load_error}")
    trajectories = _load_trajectories(Path(args.dev_trajectories))[: args.max_trajectories]
    engine = HybridPaperSearch(PaperSearchIndex(args.db))
    executor = MultiStepPaperSearch(engine, candidate_limit=100)

    teacher_total = 0
    teacher_top1 = 0
    teacher_top5 = 0
    teacher_latencies: list[float] = []
    teacher_target_by_step: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1": 0, "top5": 0})
    for rows in trajectories:
        session = prototype.new_episode()
        for row in rows:
            candidates = [str(item) for item in session._selector.skill_ids]
            start = time.perf_counter()
            ranked = session._session.select(
                str(row["state_text_current"]),
                candidate_skill_ids=candidates,
                top_k=5,
                route_mode=session.route_mode,
            )
            teacher_latencies.append((time.perf_counter() - start) * 1000.0)
            ids = [str(item["skill_id"]) for item in ranked]
            target = str(row["target_skill_id"])
            hit1 = bool(ids and ids[0] == target)
            hit5 = target in ids
            teacher_total += 1
            teacher_top1 += int(hit1)
            teacher_top5 += int(hit5)
            bucket = teacher_target_by_step[str(row["decision_step_index"])]
            bucket["count"] += 1
            bucket["top1"] += int(hit1)
            bucket["top5"] += int(hit5)
            session._session.observe(
                state_text_before=str(row["state_text_current"]),
                skill_id=target,
                action_text=str(row["action_text"]),
                result_text=str(row["actual_result_text"]),
            )

    closed_latencies: list[float] = []
    episode_steps: list[int] = []
    stop_count = 0
    failure_count = 0
    illegal_count = 0
    nonempty_result_count = 0
    stage2_decision_count = 0
    for rows in trajectories:
        start = time.perf_counter()
        results, state = executor.run(
            str(rows[0]["goal_text"]),
            policy=prototype.new_episode(),
            max_steps=len(rows),
            top_k=10,
        )
        closed_latencies.append((time.perf_counter() - start) * 1000.0)
        episode_steps.append(len(state.events))
        stop_count += int(state.stopped and bool(state.stop_reason))
        failure_count += len(state.failures)
        nonempty_result_count += int(bool(results))
        for event in state.events:
            illegal_count += int(event.skill_id not in event.legal_actions)
            stage2_decision_count += int(event.policy.get("stage2_enabled") is True)

    report = {
        "status": "ok",
        "protocol": "shenzhi_stage2_frozen_dev_evaluation_v1",
        "evidence_scope": "weak_bootstrap_executable_replay_engineering_evaluation",
        "trajectory_count": len(trajectories),
        "teacher_forced": {
            "decision_count": teacher_total,
            "top1_accuracy": teacher_top1 / teacher_total,
            "top5_accuracy": teacher_top5 / teacher_total,
            "by_step": {
                step: {
                    "count": values["count"],
                    "top1_accuracy": values["top1"] / values["count"],
                    "top5_accuracy": values["top5"] / values["count"],
                }
                for step, values in sorted(teacher_target_by_step.items(), key=lambda item: int(item[0]))
            },
            "latency_ms": {
                "mean": statistics.fmean(teacher_latencies),
                "p50": _percentile(teacher_latencies, 0.50),
                "p95": _percentile(teacher_latencies, 0.95),
                "max": max(teacher_latencies),
            },
        },
        "closed_loop": {
            "episode_count": len(trajectories),
            "stopped_episode_rate": stop_count / len(trajectories),
            "nonempty_result_rate": nonempty_result_count / len(trajectories),
            "illegal_action_count": illegal_count,
            "executor_failure_count": failure_count,
            "stage2_decision_count": stage2_decision_count,
            "mean_steps": statistics.fmean(episode_steps),
            "latency_ms": {
                "mean": statistics.fmean(closed_latencies),
                "p50": _percentile(closed_latencies, 0.50),
                "p95": _percentile(closed_latencies, 0.95),
                "max": max(closed_latencies),
            },
        },
        "interpretation_limit": (
            "These frozen dev traces are generated executable weak supervision, not observed user trajectories; "
            "the metrics establish engineering behavior only."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
