from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass
class Skill:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    executor_desc: str
    failure_modes: list[str]
    skill_id: str | None = None


@dataclass
class Task:
    task_id: str | int
    query: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    query: str
    history: list[tuple[str, str]]
    observation: str
    artifact: dict[str, Any]
    error: str | None


@dataclass
class TrajectoryStep:
    x: ExecutionState
    m: Tensor
    candidates: list[int]
    action_local: int
    skill_idx: int | None
    log_prob: Tensor
    obs: str
    m_hat: Tensor | None
    m_tilde_next: Tensor | None
    policy_logits: Tensor | None = None
    ref_logits: Tensor | None = None


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    reward: float
    task_id: str | int


@dataclass
class ReplayStep:
    x: ExecutionState
    skill_idx: int
    obs: str


@dataclass
class VerifiedPair:
    state_before: ExecutionState
    action_at_t: int
    obs_at_t: str
    candidates_next: list[int]
    a_next_plus: int
    was_in_raw_topk: bool = True
    m_t_exact: Tensor | None = None
    replay_prefix: list[ReplayStep] | None = None
    replay_prefix_is_essential_subsequence: bool = True
    task_id: str | int | None = None
    step_idx: int | None = None


@dataclass
class RetrievalPositive:
    state: ExecutionState
    positive_skill_idx: int


def serialize_execution_state(state: ExecutionState) -> str:
    history_bits = [f"{name} -> {summary}" for name, summary in state.history]
    history_text = " | ".join(history_bits) if history_bits else "empty"
    error_text = state.error if state.error is not None else "none"
    return "\n".join(
        [
            f"query:{state.query}",
            f"history:{history_text}",
            f"observation:{state.observation}",
            f"artifact:{state.artifact}",
            f"error:{error_text}",
        ]
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def execution_state_to_dict(state: ExecutionState) -> dict[str, Any]:
    return {
        "query": state.query,
        "history": list(state.history),
        "observation": state.observation,
        "artifact": state.artifact,
        "error": state.error,
    }


def execution_state_from_dict(data: dict[str, Any]) -> ExecutionState:
    return ExecutionState(
        query=data["query"],
        history=[tuple(item) for item in data.get("history", [])],
        observation=data.get("observation", ""),
        artifact=data.get("artifact", {}),
        error=data.get("error"),
    )


def replay_step_to_dict(step: ReplayStep) -> dict[str, Any]:
    return {
        "x": execution_state_to_dict(step.x),
        "skill_idx": step.skill_idx,
        "obs": step.obs,
    }


def replay_step_from_dict(data: dict[str, Any]) -> ReplayStep:
    return ReplayStep(
        x=execution_state_from_dict(data["x"]),
        skill_idx=int(data["skill_idx"]),
        obs=data["obs"],
    )


def verified_pair_to_dict(pair: VerifiedPair) -> dict[str, Any]:
    return {
        "task_id": pair.task_id,
        "step_idx": pair.step_idx,
        "state_before": execution_state_to_dict(pair.state_before),
        "action_at_t": pair.action_at_t,
        "obs_at_t": pair.obs_at_t,
        "candidates_next": list(pair.candidates_next),
        "a_next_plus": pair.a_next_plus,
        "was_in_raw_topk": pair.was_in_raw_topk,
        "m_t_exact": pair.m_t_exact.detach().cpu().tolist() if pair.m_t_exact is not None else None,
        "replay_prefix": [replay_step_to_dict(step) for step in pair.replay_prefix or []],
        "replay_prefix_is_essential_subsequence": pair.replay_prefix_is_essential_subsequence,
    }


def verified_pair_from_dict(data: dict[str, Any]) -> VerifiedPair:
    m_t_exact = data.get("m_t_exact")
    replay_prefix = None
    if "replay_prefix" in data:
        replay_prefix = [replay_step_from_dict(item) for item in data.get("replay_prefix", [])]
    return VerifiedPair(
        task_id=data.get("task_id"),
        step_idx=data.get("step_idx"),
        state_before=execution_state_from_dict(data["state_before"]),
        action_at_t=int(data["action_at_t"]),
        obs_at_t=data["obs_at_t"],
        candidates_next=[int(idx) for idx in data.get("candidates_next", [])],
        a_next_plus=int(data["a_next_plus"]),
        was_in_raw_topk=bool(data.get("was_in_raw_topk", True)),
        m_t_exact=torch.tensor(m_t_exact, dtype=torch.float32) if m_t_exact is not None else None,
        replay_prefix=replay_prefix,
        replay_prefix_is_essential_subsequence=bool(data.get("replay_prefix_is_essential_subsequence", True)),
    )


def load_verified_pairs(path: str | Path) -> list[VerifiedPair]:
    return [verified_pair_from_dict(row) for row in read_jsonl(path)]


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def validate_data_roles(data_cfg: dict[str, Any], require_training_data: bool = True) -> dict[str, dict[str, Any]]:
    legacy_replay_key = "tool" + "bench_root"
    if legacy_replay_key in data_cfg:
        raise ValueError("legacy benchmark root is no longer a supported CLSTR experiment data role")

    training_source = str(data_cfg.get("training_data_source", "mock"))
    if training_source in {"skillrouter_eval", "skillrouter", "skillrouter_benchmark"}:
        raise ValueError("SkillRouter benchmark is eval-only and must not be used as CLSTR training data")

    skillrouter_eval_root = Path(data_cfg.get("skillrouter_eval_root", "data/skillrouter_eval_core"))
    skillsbench_root = Path(data_cfg.get("skillsbench_root", "/root/autodl-tmp/skillsbench"))
    alfworld_root = Path(data_cfg.get("alfworld_root", "/root/autodl-tmp/alfworld"))
    leakage_audit_dir = Path(data_cfg.get("leakage_audit_dir", "outputs/leakage_audit"))
    clean_router_data_root = Path(data_cfg.get("clean_router_data_root", "data/clean_router"))
    skillret_root = Path(data_cfg.get("skillret_root", "/root/autodl-tmp/skillret"))
    retrieval_warmup_data_root = Path(data_cfg.get("retrieval_warmup_data_root", "data/skillret"))
    aux_trajectory_root = Path(data_cfg.get("aux_trajectory_root", "/root/autodl-tmp/aux_trajectories"))
    aux_trajectory_data_root = Path(data_cfg.get("aux_trajectory_data_root", "data/aux_trajectories"))
    appworld_root = Path(data_cfg.get("appworld_root", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root"))
    appworld_routing_data_root = Path(data_cfg.get("appworld_routing_data_root", "data/appworld_routing"))

    roles = {
        "skillrouter_eval": {
            "role": "eval-only",
            "path": str(skillrouter_eval_root),
            "status": "ok" if _path_exists(skillrouter_eval_root) else "missing",
        },
        "skillsbench": {
            "role": "train/dev/test",
            "path": str(skillsbench_root),
            "status": "ok" if _path_exists(skillsbench_root) else "missing",
        },
        "alfworld": {
            "role": "downstream-eval",
            "path": str(alfworld_root),
            "status": "ok" if _path_exists(alfworld_root) else "missing",
        },
        "leakage_audit": {
            "role": "audit-output",
            "path": str(leakage_audit_dir),
            "status": "configured",
        },
        "clean_router_data": {
            "role": "training-clean-router-data",
            "path": str(clean_router_data_root),
            "status": "ok" if _path_exists(clean_router_data_root) else "missing",
        },
        "skillret": {
            "role": "retrieval-pretraining-source",
            "path": str(skillret_root),
            "status": "ok" if _path_exists(skillret_root) else "missing",
        },
        "retrieval_warmup_data": {
            "role": "training-retrieval-warmup-data",
            "path": str(retrieval_warmup_data_root),
            "status": "ok" if _path_exists(retrieval_warmup_data_root) else "missing",
        },
        "aux_trajectory": {
            "role": "auxiliary-transition-pretraining-source",
            "path": str(aux_trajectory_root),
            "status": "ok" if _path_exists(aux_trajectory_root) else "missing",
        },
        "aux_trajectory_data": {
            "role": "training-auxiliary-trajectory-data",
            "path": str(aux_trajectory_data_root),
            "status": "ok" if _path_exists(aux_trajectory_data_root) else "missing",
        },
        "appworld": {
            "role": "main-benchmark-runtime",
            "path": str(appworld_root),
            "status": "ok" if _path_exists(appworld_root) else "missing",
        },
        "appworld_routing_data": {
            "role": "training-appworld-routing-data",
            "path": str(appworld_routing_data_root),
            "status": "ok" if _path_exists(appworld_routing_data_root) else "missing",
        },
        "training": {
            "role": "train",
            "source": training_source,
        },
    }

    if require_training_data and training_source == "skillsbench" and not _path_exists(skillsbench_root):
        raise FileNotFoundError(f"SkillsBench training data is missing: {skillsbench_root}")
    if require_training_data and training_source == "alfworld" and not _path_exists(alfworld_root):
        raise FileNotFoundError(f"ALFWorld training data is missing: {alfworld_root}")
    if require_training_data and training_source == "clean_router" and not _path_exists(clean_router_data_root):
        raise FileNotFoundError(f"Clean router training data is missing: {clean_router_data_root}")
    if require_training_data and training_source == "skillret_retrieval":
        missing = [
            path
            for path in (
                retrieval_warmup_data_root / "skills.jsonl",
                retrieval_warmup_data_root / "queries.jsonl",
                retrieval_warmup_data_root / "qrels.jsonl",
            )
            if not _path_exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "SKILLRET retrieval warmup data is missing: "
                + ", ".join(str(path) for path in missing)
            )
    if require_training_data and training_source == "aux_trajectory":
        missing = [
            path
            for path in (
                aux_trajectory_data_root / "trajectories.jsonl",
                aux_trajectory_data_root / "pseudo_skills.jsonl",
                aux_trajectory_data_root / "manifest.json",
            )
            if not _path_exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Auxiliary trajectory data is missing: "
                + ", ".join(str(path) for path in missing)
            )
    if require_training_data and training_source == "appworld_routing":
        missing = [
            path
            for path in (
                appworld_routing_data_root / "train_tasks.jsonl",
                appworld_routing_data_root / "train_replay.jsonl",
                appworld_routing_data_root / "manifest.json",
            )
            if not _path_exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "AppWorld routing data is missing: "
                + ", ".join(str(path) for path in missing)
            )
    return roles
