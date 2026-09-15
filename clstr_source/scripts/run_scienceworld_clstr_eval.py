#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.envs.scienceworld_adapter import (
    DEFAULT_SCIENCEWORLD_REPO_PATH,
    ScienceWorldEnvAdapter,
    build_scienceworld_smoke_report,
)
from clstr.harness_controller import build_clstr_text_action_controller


BLOCKED_STATUS = "blocked_no_closed_loop_eval_harness"


class FirstAdmissibleController:
    def choose_action(self, state_text: str, admissible_actions: list[str]) -> str:
        if not admissible_actions:
            return ""
        return admissible_actions[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _choose_action(controller: Any, state_text: str, admissible_actions: list[str]) -> str:
    if hasattr(controller, "choose_action"):
        return str(controller.choose_action(state_text, admissible_actions))
    if hasattr(controller, "select_action"):
        return str(controller.select_action(state_text, admissible_actions))
    if callable(controller):
        return str(controller(state_text, admissible_actions))
    raise TypeError("controller must expose choose_action, select_action, or be callable")


def _write_blocker(output_dir: Path, smoke_report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocker = {
        "status": BLOCKED_STATUS,
        "reason": "ScienceWorld official harness is not available; no offline diagnostic is counted as eval success.",
        "smoke_report": smoke_report,
    }
    metrics = {
        "status": BLOCKED_STATUS,
        "method": "clstr_controller_gate",
        "episodes": 0,
        "episode_count": 0,
        "success_rate": None,
        "average_reward": None,
        "average_episode_steps": None,
        "blocker_report": str(output_dir / "blocker_report.json"),
        "controller_class": None,
    }
    _write_json(output_dir / "blocker_report.json", blocker)
    _write_json(output_dir / "metrics.json", metrics)
    return {"status": BLOCKED_STATUS, "metrics": metrics, "blocker_report": blocker}


def _remove_stale_blocker(output_dir: Path) -> None:
    blocker_path = output_dir / "blocker_report.json"
    if blocker_path.exists():
        blocker_path.unlink()


def run_scienceworld_eval_or_blocker(
    *,
    output_dir: str | Path = "outputs/scienceworld_eval/clstr_controller_gate",
    smoke_report: dict[str, Any] | None = None,
    adapter_factory: Any | None = None,
    controller: Any | None = None,
    controller_report: dict[str, Any] | None = None,
    max_episodes: int = 1,
    max_steps: int = 50,
    task_id: int | str = 13,
    variation: int = 0,
    simplifications: str = "easy",
) -> dict[str, Any]:
    output = Path(output_dir)
    smoke = smoke_report or build_scienceworld_smoke_report(repo_path=None)
    if smoke.get("status") != "ok" or smoke.get("smoke_success") is not True:
        return _write_blocker(output, smoke)

    controller = controller or FirstAdmissibleController()
    adapter_factory = adapter_factory or (
        lambda: ScienceWorldEnvAdapter(
            task_id=task_id,
            variation=variation,
            simplifications=simplifications,
            step_limit=max_steps,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    _remove_stale_blocker(output)

    runs: list[dict[str, Any]] = []
    successes = 0
    rewards: list[float] = []
    step_counts: list[int] = []
    for episode_idx in range(max_episodes):
        adapter = adapter_factory()
        adapter.reset(task_id=str(task_id) if task_id is not None else None)
        steps: list[dict[str, Any]] = []
        total_reward = 0.0
        success = False
        done = False
        try:
            for step_idx in range(max_steps):
                state_text = adapter.state_text() if hasattr(adapter, "state_text") else f"observation: {adapter.observation_text()}"
                admissible_actions = (
                    adapter.admissible_actions()
                    if hasattr(adapter, "admissible_actions")
                    else adapter.candidate_actions()
                )
                action = _choose_action(controller, state_text, admissible_actions)
                decision = getattr(controller, "last_decision", None)
                step_result = adapter.step(action)
                total_reward += float(step_result.reward or 0.0)
                done = bool(step_result.done)
                success = bool(step_result.success)
                step_row = {
                    "step": step_idx,
                    "state_text": state_text,
                    "admissible_actions": list(admissible_actions),
                    "action": action,
                    "observation": step_result.observation_text,
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "success": step_result.success,
                }
                if decision is not None:
                    step_row.update(
                        {
                            "candidate_trace": list(admissible_actions),
                            "chosen_index": decision.chosen_index,
                            "chosen_reason": decision.chosen_reason,
                            "component_scores": decision.component_scores,
                        }
                    )
                steps.append(step_row)
                if done:
                    break
        finally:
            if hasattr(adapter, "close"):
                adapter.close()
        successes += 1 if success else 0
        rewards.append(total_reward)
        step_counts.append(len(steps))
        runs.append(
            {
                "episode": episode_idx,
                "success": success,
                "done": done,
                "reward": total_reward,
                "steps": steps,
            }
        )

    metrics = {
        "status": "ok",
        "method": "clstr_controller_gate",
        "episodes": len(runs),
        "episode_count": len(runs),
        "success_rate": round(successes / len(runs), 6) if runs else 0.0,
        "average_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "average_episode_steps": round(sum(step_counts) / len(step_counts), 6) if step_counts else 0.0,
        "controller_class": (controller_report or {}).get("controller_class", type(controller).__name__),
        "controller_report": controller_report or {},
    }
    _write_json(output / "metrics.json", metrics)
    (output / "run.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in runs) + ("\n" if runs else ""),
        encoding="utf-8",
    )
    return {"status": "ok", "metrics": metrics, "runs": runs}


def _load_smoke_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cli_controller(args: argparse.Namespace, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    return build_clstr_text_action_controller(
        routing_init_manifest=args.routing_init_manifest,
        checkpoint_path=args.checkpoint_path,
        aux_data_root=args.aux_data_root,
        output_dir=output_dir / "model_cache",
        controller_mode=args.controller_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR closed-loop ScienceWorld eval or write a blocker report.")
    parser.add_argument("--output_dir", default="outputs/scienceworld_eval/clstr_controller_gate")
    parser.add_argument("--smoke_report_path", default="outputs/scienceworld_eval/harness_smoke_report.json")
    parser.add_argument("--repo_path", default=str(DEFAULT_SCIENCEWORLD_REPO_PATH))
    parser.add_argument("--task_id", default=13)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--simplifications", default="easy")
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--aux_data_root", default="data/clstr_full_base_train")
    parser.add_argument(
        "--controller_mode",
        default="policy_plus_transition_belief_stop_loop_penalty",
        choices=[
            "policy_only",
            "policy_plus_transition",
            "policy_plus_transition_belief",
            "policy_plus_transition_belief_stop",
            "policy_plus_transition_belief_stop_loop_penalty",
        ],
    )
    args = parser.parse_args()

    smoke_report = _load_smoke_report(Path(args.smoke_report_path))
    if smoke_report is None:
        smoke_report = build_scienceworld_smoke_report(repo_path=Path(args.repo_path) if args.repo_path else None)
    controller = None
    controller_report = None
    if smoke_report.get("status") == "ok" and smoke_report.get("smoke_success") is True:
        try:
            controller, controller_report = _build_cli_controller(args, Path(args.output_dir))
        except Exception as exc:  # noqa: BLE001
            smoke_report = {
                **smoke_report,
                "status": "blocker",
                "smoke_success": False,
                "blockers": {
                    **dict(smoke_report.get("blockers") or {}),
                    "clstr_controller_loading": {
                        "message": "CLSTR controller could not be loaded for ScienceWorld closed-loop eval.",
                        "error": repr(exc),
                        "reproduce_command": (
                            "python scripts/run_scienceworld_clstr_eval.py "
                            f"--output_dir {args.output_dir} "
                            f"--routing_init_manifest {args.routing_init_manifest} "
                            f"--checkpoint_path {args.checkpoint_path or ''} "
                            f"--aux_data_root {args.aux_data_root} "
                            f"--controller_mode {args.controller_mode}"
                        ),
                    },
                },
            }
    report = run_scienceworld_eval_or_blocker(
        output_dir=Path(args.output_dir),
        smoke_report=smoke_report,
        controller=controller,
        controller_report=controller_report,
        task_id=args.task_id,
        variation=args.variation,
        simplifications=args.simplifications,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
