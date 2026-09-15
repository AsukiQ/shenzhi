#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_online import (  # noqa: E402
    CONTROLLED_HISTORY_METHODS,
    ControlledMatchedHistorySelector,
)
from clstr.vnext_tau2_agent import CLSTRVNextTau2Agent  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.run import run_domain  # noqa: E402


AGENT_NAME = "clstr_matched_history_e3_agent"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact_json_credentials(path: Path, secrets: set[str]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    protected = {value for value in secrets if value and value != "EMPTY"}

    def redact(value: Any, key: str | None = None) -> Any:
        if key is not None and key.lower() in {
            "api_key",
            "openai_api_key",
            "user_api_key",
        }:
            return "***REDACTED***"
        if isinstance(value, str) and value in protected:
            return "***REDACTED***"
        if isinstance(value, dict):
            return {str(k): redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    _write_json(path, redact(payload))


def _official_test_ids(manifest_path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = payload.get("tau2_split_manifest") or {}
    if not bool(split.get("official_test_preserved")):
        raise ValueError("matched union does not preserve the official Tau2 test split")
    records = list((split.get("splits") or {}).get("test") or [])
    by_domain = {domain: [] for domain in ("airline", "retail", "telecom")}
    for record in records:
        domain, separator, task_id = str(record).partition("/")
        if not separator or domain not in by_domain or not task_id:
            raise ValueError(f"invalid Tau2 official-test identity: {record}")
        by_domain[domain].append(task_id)
    expected = int((split.get("task_count_by_split") or {}).get("test") or 0)
    if sum(map(len, by_domain.values())) != expected or expected != 100:
        raise ValueError("Tau2 official-test count differs from 100")
    expected_counts = {"airline": 20, "retail": 40, "telecom": 40}
    observed_counts = {key: len(value) for key, value in by_domain.items()}
    if observed_counts != expected_counts:
        raise ValueError(f"Tau2 domain counts differ: {observed_counts}")
    return {domain: sorted(values) for domain, values in by_domain.items()}


def _simulation_infra_error(simulation: Any) -> bool:
    if getattr(simulation, "reward_info", None) is None:
        return True
    reason = str(getattr(simulation, "termination_reason", "") or "").lower()
    if "error" in reason or "exception" in reason or reason in {"failed", "failure"}:
        return True
    info = getattr(simulation, "info", None)
    if info in (None, {}, ""):
        return False
    text = json.dumps(info, ensure_ascii=False, sort_keys=True).lower()
    return "error" in text or "exception" in text or "traceback" in text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-support E1 Static/Transformer/LSTR comparison on "
            "the official 100-task Tau2 test split."
        )
    )
    parser.add_argument("--method", choices=sorted(CONTROLLED_HISTORY_METHODS), required=True)
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--e1_checkpoint_path", default=None)
    parser.add_argument("--e1_report_path", default=None)
    parser.add_argument("--matched_union_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--executor_model", required=True)
    parser.add_argument("--executor_base_url", required=True)
    parser.add_argument("--executor_api_key", default="EMPTY")
    parser.add_argument("--executor_temperature", type=float, default=0.0)
    parser.add_argument("--user_model", default="gpt-4.1")
    parser.add_argument("--user_base_url", default=None)
    parser.add_argument("--user_api_key", default=None)
    parser.add_argument(
        "--domain",
        action="append",
        choices=["airline", "retail", "telecom"],
        default=None,
    )
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=300)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.matched_union_manifest_path).resolve()
    expected_test_ids = _official_test_ids(manifest_path)
    method = str(args.method).strip().lower()
    domains = list(dict.fromkeys(args.domain or ["airline", "retail", "telecom"]))
    strict_full_request = args.num_tasks is None and domains == [
        "airline",
        "retail",
        "telecom",
    ]
    if strict_full_request and (
        int(args.num_trials) != 1
        or int(args.max_steps) != 100
        or int(args.top_k) != 8
        or abs(float(args.executor_temperature)) > 1.0e-12
    ):
        raise ValueError(
            "official E3 requires one trial, 100 steps, Top-8, and temperature 0"
        )
    selector = ControlledMatchedHistorySelector(
        method=method,
        foundation_checkpoint_path=args.foundation_checkpoint_path,
        skills_path=args.skills_path,
        e1_checkpoint_path=args.e1_checkpoint_path,
        e1_report_path=args.e1_report_path,
        device=args.device,
    )
    route_mode = "static" if method == "static" else "dynamic"
    selection_log_path = output_dir / "retrieval_selections.jsonl"
    outcomes_path = output_dir / "task_outcomes.jsonl"
    for path in (selection_log_path, outcomes_path):
        if path.exists():
            path.unlink()

    def create_agent(tools, domain_policy, **kwargs):
        task = kwargs.get("task")
        return CLSTRVNextTau2Agent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
            selector=selector,
            top_k=args.top_k,
            selection_log_path=selection_log_path,
            route_mode=route_mode,
            task_id=getattr(task, "id", None),
        )

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(create_agent, AGENT_NAME)
    executor_api_key = (
        os.environ.get("EXECUTOR_API_KEY", "EMPTY")
        if args.executor_api_key == "EMPTY"
        else args.executor_api_key
    )
    llm_args_agent = {
        "api_base": args.executor_base_url,
        "api_key": executor_api_key,
        "temperature": float(args.executor_temperature),
    }
    litellm_agent_model = (
        args.executor_model
        if "/" in args.executor_model
        else f"openai/{args.executor_model}"
    )
    llm_args_user: dict[str, Any] = {}
    user_api_key = args.user_api_key or os.environ.get("OPENAI_API_KEY")
    if args.user_base_url:
        llm_args_user["api_base"] = args.user_base_url
    if user_api_key:
        llm_args_user["api_key"] = user_api_key

    domain_reports: dict[str, Any] = {}
    outcome_records: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    all_simulation_count = 0
    all_missing_reward_count = 0
    all_infra_error_count = 0
    termination_reasons: Counter[str] = Counter()
    for domain in domains:
        domain_output = (output_dir / domain).resolve()
        config = TextRunConfig(
            domain=domain,
            task_split_name="test",
            num_tasks=args.num_tasks,
            agent=AGENT_NAME,
            user="user_simulator",
            llm_agent=litellm_agent_model,
            llm_args_agent=llm_args_agent,
            llm_user=args.user_model,
            llm_args_user=llm_args_user,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            max_concurrency=max(1, int(args.max_concurrency)),
            seed=args.seed,
            save_to=str(domain_output),
            auto_resume=False,
            auto_review=False,
            hallucination_retries=0,
        )
        results = run_domain(config)
        results_path = domain_output / "results.json"
        _redact_json_credentials(results_path, {executor_api_key, user_api_key or ""})
        observed_task_ids = sorted(str(task.id) for task in results.tasks)
        domain_identity_verified = observed_task_ids == expected_test_ids[domain]
        if strict_full_request and not domain_identity_verified:
            raise ValueError(f"Tau2 {domain} identities differ from the official manifest")

        simulations = list(results.simulations)
        rewards: list[float] = []
        missing_reward_count = 0
        infra_error_count = 0
        domain_reasons: Counter[str] = Counter()
        for simulation in simulations:
            reason = str(simulation.termination_reason or "unknown")
            domain_reasons[reason] += 1
            termination_reasons[reason] += 1
            reward_info = simulation.reward_info
            reward = None if reward_info is None else float(reward_info.reward)
            if reward is None:
                missing_reward_count += 1
            else:
                rewards.append(reward)
                all_rewards.append(reward)
            infra_error = _simulation_infra_error(simulation)
            infra_error_count += int(infra_error)
            outcome_records.append(
                {
                    "domain": domain,
                    "task_id": str(simulation.task_id),
                    "trial": int(simulation.trial),
                    "simulation_id": str(simulation.id),
                    "reward": reward,
                    "success": bool(reward is not None and reward >= 1.0 - 1.0e-12),
                    "termination_reason": reason,
                    "infra_error": infra_error,
                }
            )
        all_simulation_count += len(simulations)
        all_missing_reward_count += missing_reward_count
        all_infra_error_count += infra_error_count
        success_count = sum(reward >= 1.0 - 1.0e-12 for reward in rewards)
        domain_reports[domain] = {
            "task_count": len(results.tasks),
            "simulation_count": len(simulations),
            "reward_count": len(rewards),
            "missing_reward_count": missing_reward_count,
            "infra_error_count": infra_error_count,
            "termination_reason_counts": dict(sorted(domain_reasons.items())),
            "success_count": success_count,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "success_rate": success_count / len(rewards) if rewards else 0.0,
            "official_test_identity_verified": domain_identity_verified,
            "results_path": str(results_path),
        }

    with outcomes_path.open("w", encoding="utf-8") as handle:
        for record in sorted(
            outcome_records,
            key=lambda item: (item["domain"], item["task_id"], item["trial"]),
        ):
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    expected_simulations = sum(len(expected_test_ids[domain]) for domain in domains)
    expected_simulations *= int(args.num_trials)
    if strict_full_request and (
        all_simulation_count != expected_simulations
        or len(all_rewards) != expected_simulations
        or all_missing_reward_count != 0
        or all_infra_error_count != 0
    ):
        raise ValueError(
            "controlled Tau2 full-run audit failed: "
            f"simulations={all_simulation_count}, rewards={len(all_rewards)}, "
            f"missing={all_missing_reward_count}, infra={all_infra_error_count}"
        )
    success_count = sum(reward >= 1.0 - 1.0e-12 for reward in all_rewards)
    report = {
        "status": "ok" if all_rewards and all_infra_error_count == 0 else "action_required",
        "schema_version": "clstr_matched_history_tau2_e3_result_v1",
        "metric_scope": "official Tau2 test environment reward/task success",
        "method": selector.method,
        "controlled_method": method,
        "executor_model": args.executor_model,
        "executor_temperature": float(args.executor_temperature),
        "litellm_agent_model": litellm_agent_model,
        "user_model": args.user_model,
        "split_protocol": "official_test_bound_to_matched_union_v1",
        "evaluation_scope": "full" if strict_full_request else "smoke",
        "matched_union_manifest_path": str(manifest_path),
        "matched_union_manifest_sha256": _file_sha256(manifest_path),
        "domains": domain_reports,
        "simulation_count": all_simulation_count,
        "reward_count": len(all_rewards),
        "missing_reward_count": all_missing_reward_count,
        "infra_error_count": all_infra_error_count,
        "termination_reason_counts": dict(sorted(termination_reasons.items())),
        "success_count": success_count,
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "success_rate": success_count / len(all_rewards) if all_rewards else 0.0,
        "selection_log_path": str(selection_log_path),
        "task_outcomes_path": str(outcomes_path),
        "task_outcomes_sha256": _file_sha256(outcomes_path),
        "route_mode": route_mode,
        "top_k": int(args.top_k),
        "max_steps": int(args.max_steps),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "checkpoint_binding": selector.checkpoint_binding,
        "oracle_or_private_task_fields_used_by_agent": False,
    }
    _write_json(output_dir / "clstr_task_accuracy.json", report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
