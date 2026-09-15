#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from clstr.vnext_online_selector import VNextOnlineSelector
from clstr.vnext_tau2_agent import CLSTRVNextTau2Agent
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.run import run_domain


AGENT_NAME = "clstr_vnext_recurrent_agent"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
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
    if sum(map(len, by_domain.values())) != expected or expected <= 0:
        raise ValueError("Tau2 official-test count differs from the matched manifest")
    return {domain: sorted(values) for domain, values in by_domain.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the release-bound recurrent CLSTR agent through official Tau2 "
            "test simulations and report environment reward/task success."
        )
    )
    parser.add_argument("--matched_release_selection_path", required=True)
    parser.add_argument("--stage2_checkpoint_path", default=None)
    parser.add_argument("--training_skills_path", default=None)
    parser.add_argument("--benchmark_skills_path", required=True)
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
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument(
        "--route_mode",
        choices=["adaptive", "static", "dynamic"],
        default="adaptive",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=300)
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.matched_union_manifest_path).resolve()
    expected_test_ids = _official_test_ids(manifest_path)
    selector = VNextOnlineSelector(
        matched_release_selection_path=args.matched_release_selection_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        training_skills_path=args.training_skills_path,
        benchmark_skills_path=args.benchmark_skills_path,
        device=args.device,
        coarse_k=args.coarse_k,
        dynamic_extra_k=args.dynamic_extra_k,
    )
    release_manifest = selector.checkpoint_binding["release"][
        "matched_union_manifest"
    ]
    if Path(release_manifest["path"]).resolve() != manifest_path or str(
        release_manifest["sha256"]
    ) != _file_sha256(manifest_path):
        raise ValueError("Tau2 split manifest differs from the matched release")
    selection_log_path = output_dir / "retrieval_selections.jsonl"
    if selection_log_path.exists():
        selection_log_path.unlink()

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
            route_mode=args.route_mode,
            task_id=getattr(task, "id", None),
        )

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(create_agent, AGENT_NAME)
    domains = list(dict.fromkeys(args.domain or ["airline", "retail", "telecom"]))
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
    all_rewards: list[float] = []
    strict_full_request = args.num_tasks is None
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
        _redact_json_credentials(
            domain_output / "results.json",
            {executor_api_key, user_api_key or ""},
        )
        observed_task_ids = sorted(str(task.id) for task in results.tasks)
        domain_identity_verified = observed_task_ids == expected_test_ids[domain]
        if strict_full_request and not domain_identity_verified:
            raise ValueError(
                f"Tau2 {domain} official-test identities differ from the matched manifest"
            )
        rewards = [
            float(simulation.reward_info.reward)
            for simulation in results.simulations
            if simulation.reward_info is not None
        ]
        success_count = sum(reward >= 1.0 - 1e-12 for reward in rewards)
        all_rewards.extend(rewards)
        domain_reports[domain] = {
            "task_count": len(results.tasks),
            "simulation_count": len(results.simulations),
            "reward_count": len(rewards),
            "success_count": success_count,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "success_rate": (
                success_count / len(rewards)
                if rewards
                else 0.0
            ),
            "official_test_identity_verified": domain_identity_verified,
            "results_path": str(domain_output / "results.json"),
        }
    full_split = all(
        bool(domain_reports[domain]["official_test_identity_verified"])
        for domain in domains
    )
    expected_simulations = sum(len(expected_test_ids[domain]) for domain in domains)
    expected_simulations *= int(args.num_trials)
    if full_split and len(all_rewards) != expected_simulations:
        raise ValueError(
            "Tau2 official-test reward denominator differs: "
            f"expected {expected_simulations}, observed {len(all_rewards)}"
        )
    report = {
        "status": "ok" if all_rewards else "action_required",
        "metric_scope": "official Tau2 test environment reward/task success",
        "method": selector.method,
        "executor_model": args.executor_model,
        "executor_temperature": float(args.executor_temperature),
        "litellm_agent_model": litellm_agent_model,
        "user_model": args.user_model,
        "split_protocol": "official_test_bound_to_matched_union_v1",
        "evaluation_scope": "full" if full_split else "smoke",
        "matched_union_manifest_path": str(manifest_path),
        "matched_union_manifest_sha256": _file_sha256(manifest_path),
        "domains": domain_reports,
        "simulation_count": len(all_rewards),
        "success_count": sum(
            reward >= 1.0 - 1e-12 for reward in all_rewards
        ),
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "success_rate": (
            sum(reward >= 1.0 - 1e-12 for reward in all_rewards) / len(all_rewards)
            if all_rewards
            else 0.0
        ),
        "selection_log_path": str(selection_log_path),
        "route_mode": args.route_mode,
        "checkpoint_binding": selector.checkpoint_binding,
        "oracle_or_private_task_fields_used_by_agent": False,
    }
    _write_json(output_dir / "clstr_task_accuracy.json", report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
