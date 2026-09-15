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

from clstr.envs.scienceworld_adapter import ScienceWorldEnvAdapter
from clstr.policy_replay_verifier import MissingReplayHarness, verify_policy_replay, write_policy_replay_blocker


def _default_source_path(benchmark: str) -> str:
    return f"data/aux_skillnet_rebuilt/{benchmark}/trajectories.jsonl"


def _default_output_path(benchmark: str) -> str:
    return f"data/verified_policy_replay/{benchmark}_train_replay.jsonl"


def _default_manifest_path(benchmark: str) -> str:
    return f"data/verified_policy_replay/{benchmark}_manifest.json"


def _default_report_path(benchmark: str) -> str:
    return f"outputs/policy_replay_verification/{benchmark}/verification_report.json"


def _default_blocker_path(benchmark: str) -> str:
    return f"outputs/policy_replay_verification/{benchmark}/blocker_report.json"


def _scienceworld_harness_factory(trajectory: dict[str, Any]) -> ScienceWorldEnvAdapter:
    task_id = trajectory.get("task_type") or trajectory.get("task_name") or trajectory.get("task_id") or "13"
    return ScienceWorldEnvAdapter(task_id=str(task_id), variation=0, simplifications="easy", step_limit=100)


def _harness_factory_for(benchmark: str):
    if benchmark == "scienceworld":
        return _scienceworld_harness_factory
    if benchmark == "webshop":
        raise MissingReplayHarness(
            "official WebShop text env factory is not available in this repo; run setup_webshop_harness.py "
            "and provide a concrete env factory before upgrading WebShop SFT traces to verified_replay"
        )
    raise MissingReplayHarness(f"no replay harness factory registered for benchmark={benchmark}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify external trajectories against an official harness before upgrading them to CLSTR L_policy replay."
    )
    parser.add_argument("--benchmark", choices=["scienceworld", "webshop"], required=True)
    parser.add_argument("--source_id", default=None)
    parser.add_argument("--source_dataset", default=None)
    parser.add_argument("--source_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--report_path", default=None)
    parser.add_argument("--blocker_path", default=None)
    parser.add_argument("--max_trajectories", type=int)
    parser.add_argument("--max_steps_per_trajectory", type=int)
    args = parser.parse_args()

    benchmark = args.benchmark
    source_path = Path(args.source_path or _default_source_path(benchmark))
    output_path = Path(args.output_path or _default_output_path(benchmark))
    manifest_path = Path(args.manifest_path or _default_manifest_path(benchmark))
    report_path = Path(args.report_path or _default_report_path(benchmark))
    blocker_path = Path(args.blocker_path or _default_blocker_path(benchmark))
    source_id = args.source_id or f"{benchmark}_official_replay_verified"
    source_dataset = args.source_dataset or str(source_path)
    repro_command = "python scripts/verify_policy_replay_candidates.py " + " ".join(sys.argv[1:])

    try:
        if not source_path.exists():
            raise MissingReplayHarness(f"source trajectory path is missing: {source_path}")
        harness_factory = _harness_factory_for(benchmark)
        report = verify_policy_replay(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            benchmark=benchmark,
            source_id=source_id,
            source_dataset=source_dataset,
            harness_factory=harness_factory,
            max_trajectories=args.max_trajectories,
            max_steps_per_trajectory=args.max_steps_per_trajectory,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except MissingReplayHarness as exc:
        report = write_policy_replay_blocker(
            output_path=blocker_path,
            benchmark=benchmark,
            source_id=source_id,
            source_dataset=source_dataset,
            error=exc,
            repro_command=repro_command,
        )
    except Exception as exc:  # noqa: BLE001
        report = write_policy_replay_blocker(
            output_path=blocker_path,
            benchmark=benchmark,
            source_id=source_id,
            source_dataset=source_dataset,
            error=MissingReplayHarness(f"replay verification failed before producing train rows: {exc!r}"),
            repro_command=repro_command,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
