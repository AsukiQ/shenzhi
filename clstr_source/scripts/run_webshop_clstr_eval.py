#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import (
    _apply_replay_prefix_beliefs,
    _encode_texts,
    _load_clstr_alfworld_model,
    _model_skill_id_to_idx,
)
from clstr.closed_loop_controller import ClosedLoopControllerConfig, ClstrTextActionController
from clstr.envs.webshop_official_adapter import (
    DEFAULT_WEBSHOP_REPO_PATH,
    WebShopOfficialAdapter,
    build_webshop_smoke_report,
)
from clstr.memory_candidate_recall import environment_candidate_recall_metadata

build_clstr_text_action_controller = None
WebShopSkillRouterActionScorer = None


BLOCKED_STATUS = "blocked_no_closed_loop_eval_harness"


class FirstAdmissibleController:
    def choose_action(self, state_text: str, admissible_actions: list[str]) -> str:
        del state_text
        return admissible_actions[0] if admissible_actions else ""


WEBSHOP_SEARCH_SKILL_ID = "webshop/webshop-search-executor"
WEBSHOP_CLICK_SKILL_ID = "webshop/webshop-click-executor"
WEBSHOP_FINISH_SKILL_ID = "webshop/webshop-finish-executor"
WEBSHOP_REASONING_SKILL_ID = "webshop/webshop-reasoning-planner"


def _webshop_action_to_skill_id(action: str) -> str | None:
    normalized = str(action or "").strip().lower()
    if normalized.startswith("search[") or normalized == "search":
        return WEBSHOP_SEARCH_SKILL_ID
    if normalized.startswith("click["):
        return WEBSHOP_CLICK_SKILL_ID
    if normalized.startswith("finish") or normalized in {"done", "stop", "submit"}:
        return WEBSHOP_FINISH_SKILL_ID
    if normalized.startswith("think"):
        return WEBSHOP_REASONING_SKILL_ID
    return None


def _webshop_history_actions_from_state_text(state_text: str) -> list[str]:
    for raw_line in str(state_text).splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("history:"):
            continue
        value = line.split(":", 1)[1].strip()
        if not value or value == "<empty>":
            return []
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


class WebShopUnifiedMemoryActionScorer:
    """Score WebShop admissible actions through current CLSTR unified-memory logits."""

    def __init__(self, model: Any, *, replay_prefix_max_steps: int = 6) -> None:
        self.model = model
        self.replay_prefix_max_steps = max(0, int(replay_prefix_max_steps))
        self.skill_id_to_idx = _model_skill_id_to_idx(model)
        self.last_metadata: list[dict[str, Any]] = []
        if not callable(getattr(model, "initial_belief", None)):
            raise ValueError("WebShop unified memory scorer requires model.initial_belief")
        if callable(getattr(model, "eval", None)):
            model.eval()

    def _history_replay_prefix(self, state_text: str) -> list[dict[str, Any]]:
        actions = _webshop_history_actions_from_state_text(state_text)
        if self.replay_prefix_max_steps > 0:
            actions = actions[-self.replay_prefix_max_steps :]
        prefix: list[dict[str, Any]] = []
        for action in actions:
            skill_id = _webshop_action_to_skill_id(action)
            if not skill_id or skill_id not in self.skill_id_to_idx:
                continue
            prefix.append(
                {
                    "observation_text": str(state_text),
                    "action_text": str(action),
                    "skill_id": skill_id,
                    "skill_idx": int(self.skill_id_to_idx[skill_id]),
                    "mapping_confidence": "rule:webshop_action_type",
                    "mapping_reason": "webshop action prefix mapped to action-type skill",
                }
            )
        return prefix

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            self.last_metadata = []
            return torch.empty(0, 0, dtype=torch.float32)

        max_width = max((len(row) for row in candidate_texts), default=0)
        device = torch.device(getattr(self.model, "device", "cpu"))
        index_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        mapping_rows: list[list[dict[str, Any]]] = []
        for row in candidate_texts:
            indices: list[int] = []
            valid: list[bool] = []
            mappings: list[dict[str, Any]] = []
            padded = list(row) + [""] * max(0, max_width - len(row))
            for col_idx, action in enumerate(padded):
                skill_id = _webshop_action_to_skill_id(action)
                skill_idx = self.skill_id_to_idx.get(skill_id) if skill_id else None
                indices.append(int(skill_idx) if skill_idx is not None else 0)
                valid.append(bool(col_idx < len(row) and skill_idx is not None))
                mappings.append(
                    {
                        "action": str(action),
                        "mapped_skill_id": skill_id,
                        "mapped_skill_idx": skill_idx,
                        "mapping_reason": "webshop_action_type" if skill_id else "unmapped",
                    }
                )
            index_rows.append(indices)
            valid_rows.append(valid)
            mapping_rows.append(mappings)

        with torch.no_grad():
            h = _encode_texts(self.model, [str(item) for item in state_texts], freeze=True).to(device)
            m_obs = self.model.initial_belief(h)
            replay_rows = [{"replay_prefix": self._history_replay_prefix(str(item))} for item in state_texts]
            m_obs, replay_prefix_used_count = _apply_replay_prefix_beliefs(
                self.model,
                replay_rows,
                m_obs,
                self.skill_id_to_idx,
                len(self.skill_id_to_idx),
                device,
                trainable=False,
            )
            logits = self.model.unified_route_logits(h, m_obs, candidate_rows=index_rows).detach().float().cpu()

        valid_mask = torch.tensor(valid_rows, dtype=torch.bool)
        scores = logits.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
        self.last_metadata = [
            {
                "policy_family": "clstr_webshop_unified_memory_action_scorer",
                "route_scorer": "unified_memory",
                "clstr_memory_source": "replay_prefix" if replay_rows[row_idx]["replay_prefix"] else "initial_belief",
                "clstr_replay_prefix_len": len(replay_rows[row_idx]["replay_prefix"]),
                "clstr_replay_prefix_used_count": int(replay_prefix_used_count),
                "uses_recurrent_m_t": bool(replay_rows[row_idx]["replay_prefix"]),
                "candidate_skill_mappings": mapping_rows[row_idx][: len(candidate_texts[row_idx])],
                "candidate_count": len(candidate_texts[row_idx]),
            }
            for row_idx in range(len(candidate_texts))
        ]
        return scores


class WebShopUnifiedMemoryConcreteActionScorer:
    """Score concrete WebShop admissible action text with recurrent CLSTR memory."""

    def __init__(
        self,
        model: Any,
        *,
        replay_prefix_max_steps: int = 6,
        cache_candidate_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.replay_prefix_max_steps = max(0, int(replay_prefix_max_steps))
        self.cache_candidate_embeddings = bool(cache_candidate_embeddings)
        self.skill_id_to_idx = _model_skill_id_to_idx(model)
        self.candidate_embedding_cache: dict[str, torch.Tensor] = {}
        self.last_metadata: list[dict[str, Any]] = []
        if not callable(getattr(model, "initial_belief", None)):
            raise ValueError("WebShop unified memory concrete-action scorer requires model.initial_belief")
        if callable(getattr(model, "eval", None)):
            model.eval()

    def _history_replay_prefix(self, state_text: str) -> list[dict[str, Any]]:
        actions = _webshop_history_actions_from_state_text(state_text)
        if self.replay_prefix_max_steps > 0:
            actions = actions[-self.replay_prefix_max_steps :]
        prefix: list[dict[str, Any]] = []
        for action in actions:
            skill_id = _webshop_action_to_skill_id(action)
            if not skill_id or skill_id not in self.skill_id_to_idx:
                continue
            prefix.append(
                {
                    "observation_text": str(state_text),
                    "action_text": str(action),
                    "skill_id": skill_id,
                    "skill_idx": int(self.skill_id_to_idx[skill_id]),
                    "mapping_confidence": "rule:webshop_action_type",
                    "mapping_reason": "webshop action prefix mapped to action-type skill",
                }
            )
        return prefix

    def _encode_candidates(self, texts: list[str]) -> torch.Tensor:
        device = torch.device(getattr(self.model, "device", "cpu"))
        if not self.cache_candidate_embeddings:
            return _encode_texts(self.model, texts, freeze=True).to(device)
        missing = [text for text in dict.fromkeys(str(item) for item in texts) if text not in self.candidate_embedding_cache]
        if missing:
            encoded = _encode_texts(self.model, missing, freeze=True).detach()
            for text, emb in zip(missing, encoded):
                self.candidate_embedding_cache[text] = emb.detach()
        return torch.stack([self.candidate_embedding_cache[str(text)].to(device) for text in texts])

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            self.last_metadata = []
            return torch.empty(0, 0, dtype=torch.float32)

        max_width = max((len(row) for row in candidate_texts), default=0)
        if max_width <= 0:
            self.last_metadata = [
                {
                    "policy_family": "clstr_webshop_unified_memory_concrete_action_scorer",
                    "route_scorer": "unified_memory_concrete_action",
                    "candidate_count": 0,
                    "uses_recurrent_m_t": False,
                    "concrete_action_text_scoring": True,
                }
                for _ in candidate_texts
            ]
            return torch.empty(len(candidate_texts), 0, dtype=torch.float32)
        padded = [list(row) + [""] * max(0, max_width - len(row)) for row in candidate_texts]
        candidate_mask = torch.tensor(
            [[col_idx < len(row) for col_idx in range(max_width)] for row in candidate_texts],
            dtype=torch.bool,
        )
        flat_candidates = [str(text) for row in padded for text in row]
        device = torch.device(getattr(self.model, "device", "cpu"))
        with torch.no_grad():
            h = _encode_texts(self.model, [str(item) for item in state_texts], freeze=True).to(device)
            m_obs = self.model.initial_belief(h)
            replay_rows = [{"replay_prefix": self._history_replay_prefix(str(item))} for item in state_texts]
            m_obs, replay_prefix_used_count = _apply_replay_prefix_beliefs(
                self.model,
                replay_rows,
                m_obs,
                self.skill_id_to_idx,
                len(self.skill_id_to_idx),
                device,
                trainable=False,
            )
            candidate_embs = self._encode_candidates(flat_candidates).view(len(padded), max_width, -1)
            skill_head = getattr(self.model, "skill_head", None)
            scores = None
            if skill_head is not None:
                try:
                    scores = skill_head(candidate_embs, m_obs)
                except TypeError:
                    scores = None
                if scores is not None and scores.ndim == 3 and scores.size(-1) == 1:
                    scores = scores.squeeze(-1)
            if scores is None or scores.ndim != 2:
                m_norm = F.normalize(m_obs.float(), p=2, dim=-1)
                c_norm = F.normalize(candidate_embs.float(), p=2, dim=-1)
                scores = torch.einsum("bd,bcd->bc", m_norm, c_norm)
            scores = scores.detach().float().masked_fill(~candidate_mask.to(device), torch.finfo(torch.float32).min)

        mapping_rows: list[list[dict[str, Any]]] = []
        for row in candidate_texts:
            mappings = []
            for action in row:
                skill_id = _webshop_action_to_skill_id(action)
                mappings.append(
                    {
                        "action": str(action),
                        "mapped_skill_id": skill_id,
                        "mapped_skill_idx": self.skill_id_to_idx.get(skill_id) if skill_id else None,
                        "mapping_reason": "webshop_action_type" if skill_id else "unmapped",
                    }
                )
            mapping_rows.append(mappings)
        self.last_metadata = [
            {
                "policy_family": "clstr_webshop_unified_memory_concrete_action_scorer",
                "route_scorer": "unified_memory_concrete_action",
                "clstr_memory_source": "replay_prefix" if replay_rows[row_idx]["replay_prefix"] else "initial_belief",
                "clstr_replay_prefix_len": len(replay_rows[row_idx]["replay_prefix"]),
                "clstr_replay_prefix_used_count": int(replay_prefix_used_count),
                "uses_recurrent_m_t": bool(replay_rows[row_idx]["replay_prefix"]),
                "candidate_skill_mappings": mapping_rows[row_idx],
                "candidate_count": len(candidate_texts[row_idx]),
                "concrete_action_text_scoring": True,
            }
            for row_idx in range(len(candidate_texts))
        ]
        return scores.cpu()


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


def _state_with_history(state_text: str, action_history: list[str]) -> str:
    history = " | ".join(str(item) for item in action_history) if action_history else "<empty>"
    return f"{state_text}\nhistory: {history}"


def _controller_policy_metadata(controller: Any) -> dict[str, Any] | None:
    scorer = getattr(controller, "candidate_scorer", None)
    metadata = getattr(scorer, "last_metadata", None)
    if isinstance(metadata, list) and metadata:
        first = metadata[0]
        if isinstance(first, dict):
            return dict(first)
    return None


def _summarize_policy_metadata(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = 0
    recurrent_mt_steps = 0
    unified_memory_steps = 0
    concrete_action_steps = 0
    for run in runs:
        for step in run.get("steps") or []:
            metadata = step.get("policy_metadata")
            if not isinstance(metadata, dict):
                continue
            total_steps += 1
            if metadata.get("route_scorer") in {"unified_memory", "unified_memory_concrete_action"}:
                unified_memory_steps += 1
            if metadata.get("route_scorer") == "unified_memory_concrete_action":
                concrete_action_steps += 1
            if metadata.get("uses_recurrent_m_t"):
                recurrent_mt_steps += 1
    return {
        "policy_metadata_steps": total_steps,
        "clstr_unified_memory_steps": unified_memory_steps,
        "clstr_unified_memory_concrete_action_steps": concrete_action_steps,
        "clstr_recurrent_mt_steps": recurrent_mt_steps,
        "clstr_recurrent_mt_step_rate": round(recurrent_mt_steps / total_steps, 6) if total_steps else 0.0,
    }


def _make_official_webshop_adapter_factory(repo_path: str | Path):
    repo = Path(repo_path)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    conda_java_home = Path(sys.executable).resolve().parent.parent
    fallback_java_home = Path("/usr/lib/jvm/java-17-openjdk-amd64")
    java_home = conda_java_home if (conda_java_home / "bin" / "java").exists() else fallback_java_home
    if (java_home / "bin" / "java").exists():
        os.environ["JAVA_HOME"] = str(java_home)
        libjvm_path = java_home / "lib" / "jvm" / "lib" / "server" / "libjvm.so"
        if libjvm_path.exists():
            os.environ["JVM_PATH"] = str(libjvm_path)
        java_bin = str(java_home / "bin")
        path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
        os.environ["PATH"] = os.pathsep.join([java_bin] + [part for part in path_parts if part != java_bin])
    from web_agent_site.envs import WebAgentTextEnv
    from web_agent_site.utils import DEBUG_PROD_SIZE

    def factory() -> WebShopOfficialAdapter:
        return WebShopOfficialAdapter(
            env_factory=lambda **_kwargs: WebAgentTextEnv(
                observation_mode="text",
                num_products=DEBUG_PROD_SIZE,
            )
        )

    return factory


def _write_blocker(output_dir: Path, smoke_report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_policy = smoke_report.get("leakage_policy") or {
        "webshop_test_split_train_then_eval": "forbidden",
    }
    candidate_recall = environment_candidate_recall_metadata()
    blocker = {
        "status": BLOCKED_STATUS,
        "reason": "WebShop official harness is not available; no offline diagnostic is counted as eval success.",
        "smoke_report": smoke_report,
        "leakage_policy": leakage_policy,
        "not_closed_loop_success": True,
        "candidate_recall": candidate_recall,
    }
    metrics = {
        "status": BLOCKED_STATUS,
        "method": "webshop_clstr_controller_gate",
        "closed_loop_evaluated": False,
        "episodes": 0,
        "episode_count": 0,
        "success_rate": None,
        "average_reward": None,
        "average_episode_steps": None,
        "blocker_report": str(output_dir / "blocker_report.json"),
        "webshop_test_split_train_then_eval": "forbidden",
        "not_closed_loop_success": True,
        "controller_class": None,
        "candidate_recall": candidate_recall,
    }
    _write_json(output_dir / "blocker_report.json", blocker)
    _write_json(output_dir / "metrics.json", metrics)
    return {
        "status": BLOCKED_STATUS,
        "metrics": metrics,
        "blocker_report": blocker,
        "candidate_recall": candidate_recall,
    }


def _remove_stale_blocker(output_dir: Path) -> None:
    blocker_path = output_dir / "blocker_report.json"
    if blocker_path.exists():
        blocker_path.unlink()


def run_webshop_eval_or_blocker(
    *,
    output_dir: str | Path = "outputs/webshop_eval/clstr_controller_gate",
    smoke_report: dict[str, Any] | None = None,
    adapter_factory: Any | None = None,
    controller: Any | None = None,
    controller_report: dict[str, Any] | None = None,
    max_episodes: int = 1,
    max_steps: int = 50,
    repo_path: str | Path | None = DEFAULT_WEBSHOP_REPO_PATH,
) -> dict[str, Any]:
    output = Path(output_dir)
    smoke = smoke_report or build_webshop_smoke_report(repo_path=repo_path)
    if smoke.get("status") != "ok" or smoke.get("smoke_success") is not True:
        return _write_blocker(output, smoke)

    controller = controller or FirstAdmissibleController()
    if adapter_factory is None:
        try:
            adapter_factory = _make_official_webshop_adapter_factory(repo_path or DEFAULT_WEBSHOP_REPO_PATH)
        except Exception as exc:  # noqa: BLE001
            return _write_blocker(
                output,
                {
                    **smoke,
                    "status": "blocker",
                    "smoke_success": False,
                    "blockers": {
                        **dict(smoke.get("blockers") or {}),
                        "adapter_factory": {
                            "message": "Official WebShop env factory could not be configured for this workspace.",
                            "error": repr(exc),
                            "install_command": f"cd {repo_path or DEFAULT_WEBSHOP_REPO_PATH} && ./setup.sh -d small",
                        },
                    },
                },
            )
    output.mkdir(parents=True, exist_ok=True)
    _remove_stale_blocker(output)
    for stale_name in ("run.jsonl", "progress.json"):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()
    runs: list[dict[str, Any]] = []
    successes = 0
    rewards: list[float] = []
    step_counts: list[int] = []

    for episode_idx in range(max_episodes):
        if hasattr(controller, "reset"):
            controller.reset()
        adapter = adapter_factory()
        adapter.reset(task_id=str(episode_idx))
        steps: list[dict[str, Any]] = []
        episode_history: list[str] = []
        total_reward = 0.0
        success = False
        done = False
        try:
            for step_idx in range(max_steps):
                raw_state_text = adapter.state_text() if hasattr(adapter, "state_text") else f"observation: {adapter.observation_text()}"
                state_text = _state_with_history(str(raw_state_text), episode_history)
                admissible_actions = adapter.valid_actions() if hasattr(adapter, "valid_actions") else adapter.candidate_actions()
                action = _choose_action(controller, state_text, admissible_actions)
                decision = getattr(controller, "last_decision", None)
                policy_metadata = _controller_policy_metadata(controller)
                step_result = adapter.step(action)
                episode_history.append(str(action))
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
                if policy_metadata is not None:
                    step_row["policy_metadata"] = policy_metadata
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
        runs.append({"episode": episode_idx, "success": success, "done": done, "reward": total_reward, "steps": steps})
        _write_json(
            output / "progress.json",
            {
                "status": "running" if episode_idx + 1 < max_episodes else "complete",
                "completed_episodes": episode_idx + 1,
                "target_episodes": max_episodes,
                "successes": successes,
            },
        )
        with (output / "run.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(runs[-1], ensure_ascii=False) + "\n")

    trace_summary = _summarize_policy_metadata(runs)
    candidate_recall = environment_candidate_recall_metadata()
    metrics = {
        "status": "ok",
        "method": (controller_report or {}).get("method", "webshop_clstr_controller_gate"),
        "closed_loop_evaluated": True,
        "episodes": len(runs),
        "episode_count": len(runs),
        "success_rate": round(successes / len(runs), 6) if runs else 0.0,
        "average_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "average_episode_steps": round(sum(step_counts) / len(step_counts), 6) if step_counts else 0.0,
        "webshop_test_split_train_then_eval": "forbidden",
        "controller_class": (controller_report or {}).get("controller_class", type(controller).__name__),
        "controller_report": controller_report or {},
        "trace_summary": trace_summary,
        "candidate_recall": candidate_recall,
    }
    _write_json(output / "metrics.json", metrics)
    return {
        "status": "ok",
        "metrics": metrics,
        "runs": runs,
        "candidate_recall": candidate_recall,
    }


def _load_smoke_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_clstr_controller_builder():
    global build_clstr_text_action_controller
    if build_clstr_text_action_controller is None:
        from clstr.harness_controller import build_clstr_text_action_controller as builder

        build_clstr_text_action_controller = builder
    return build_clstr_text_action_controller


def _get_webshop_skillrouter_scorer_cls():
    global WebShopSkillRouterActionScorer
    if WebShopSkillRouterActionScorer is not None:
        return WebShopSkillRouterActionScorer
    from clstr.alfworld_eval import SkillRouterAdmissibleActionScorer

    class _WebShopSkillRouterActionScorer(SkillRouterAdmissibleActionScorer):
        def _query_text(self, state_text: str) -> str:
            return (
                "Instruct: Given a WebShop shopping task, page observation, and action history, "
                "retrieve the next admissible web action that best advances the task\n"
                f"Query:{str(state_text)[:1800]}"
            )

        def _action_text(self, action: str) -> str:
            return f"WebShop admissible action: {str(action)}"

    WebShopSkillRouterActionScorer = _WebShopSkillRouterActionScorer
    return WebShopSkillRouterActionScorer


def _build_skillrouter_text_action_controller(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    scorer_cls = _get_webshop_skillrouter_scorer_cls()
    scorer = scorer_cls(
        model_name_or_path=args.skillrouter_model_name_or_path,
        batch_size=args.skillrouter_batch_size,
        max_length=args.skillrouter_max_length,
        adapter_checkpoint_path=args.skillrouter_adapter_checkpoint_path,
    )
    controller = ClstrTextActionController(
        candidate_scorer=scorer,
        component_scorer=None,
        config=ClosedLoopControllerConfig(mode="policy_only"),
    )
    return controller, {
        "controller_class": "ClstrTextActionController",
        "method": "webshop_skillrouter_controller_gate",
        "controller_mode": "policy_only",
        "skillrouter_model_name_or_path": str(args.skillrouter_model_name_or_path),
        "skillrouter_adapter_checkpoint_path": (
            str(args.skillrouter_adapter_checkpoint_path)
            if args.skillrouter_adapter_checkpoint_path
            else None
        ),
        "skillrouter_batch_size": int(args.skillrouter_batch_size),
        "skillrouter_max_length": int(args.skillrouter_max_length),
        "component_scorer_enabled": False,
    }


def _build_webshop_unified_memory_controller(args: argparse.Namespace, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    model, _action_adapter, routing_report = _load_clstr_alfworld_model(
        routing_init_manifest=Path(args.routing_init_manifest),
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        stage4_checkpoint_path=Path(args.stage4_checkpoint_path) if args.stage4_checkpoint_path else None,
        skill_rows_path_override=Path(args.skill_rows_path_override) if args.skill_rows_path_override else None,
        output_dir=output_dir / "model_cache",
        data_root=Path(args.aux_data_root),
    )
    concrete_action = getattr(args, "method", "clstr_unified_memory") == "clstr_unified_memory_concrete_action"
    scorer_cls = WebShopUnifiedMemoryConcreteActionScorer if concrete_action else WebShopUnifiedMemoryActionScorer
    scorer = scorer_cls(model, replay_prefix_max_steps=int(args.replay_prefix_max_steps))
    controller = ClstrTextActionController(
        candidate_scorer=scorer,
        component_scorer=None,
        config=ClosedLoopControllerConfig(mode="policy_only"),
    )
    return controller, {
        "controller_class": "ClstrTextActionController",
        "method": (
            "webshop_clstr_unified_memory_concrete_action_controller_gate"
            if concrete_action
            else "webshop_clstr_unified_memory_controller_gate"
        ),
        "controller_mode": "policy_only",
        "route_scorer": "unified_memory_concrete_action" if concrete_action else "unified_memory",
        "routing_init_manifest": str(args.routing_init_manifest),
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "stage4_checkpoint_path": str(args.stage4_checkpoint_path) if args.stage4_checkpoint_path else None,
        "skill_rows_path_override": str(args.skill_rows_path_override) if args.skill_rows_path_override else None,
        "stage4_overlay_loaded": bool(routing_report.get("stage4_checkpoint")),
        "replay_prefix_max_steps": int(args.replay_prefix_max_steps),
        "routing_report": routing_report,
    }


def _build_cli_controller(args: argparse.Namespace, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    if getattr(args, "method", "clstr") == "skillrouter":
        return _build_skillrouter_text_action_controller(args)
    if getattr(args, "method", "clstr") in {"clstr_unified_memory", "clstr_unified_memory_concrete_action"}:
        return _build_webshop_unified_memory_controller(args, output_dir)
    builder = _get_clstr_controller_builder()
    return builder(
        routing_init_manifest=args.routing_init_manifest,
        checkpoint_path=args.checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        skill_rows_path_override=args.skill_rows_path_override,
        aux_data_root=args.aux_data_root,
        output_dir=output_dir / "model_cache",
        controller_mode=args.controller_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR closed-loop WebShop eval or write a blocker report.")
    parser.add_argument(
        "--method",
        default="clstr",
        choices=["clstr", "clstr_unified_memory", "clstr_unified_memory_concrete_action", "skillrouter"],
    )
    parser.add_argument("--output_dir", default="outputs/webshop_eval/clstr_controller_gate")
    parser.add_argument("--smoke_report_path", default="outputs/webshop_eval/harness_smoke_report.json")
    parser.add_argument("--repo_path", default=str(DEFAULT_WEBSHOP_REPO_PATH))
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--stage4_checkpoint_path", default=None)
    parser.add_argument("--skill_rows_path_override", default=None)
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
    parser.add_argument("--skillrouter_model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--skillrouter_adapter_checkpoint_path", default=None)
    parser.add_argument("--skillrouter_batch_size", type=int, default=16)
    parser.add_argument("--skillrouter_max_length", type=int, default=2048)
    parser.add_argument("--replay_prefix_max_steps", type=int, default=6)
    args = parser.parse_args()

    smoke_report = _load_smoke_report(Path(args.smoke_report_path))
    if smoke_report is None:
        smoke_report = build_webshop_smoke_report(repo_path=Path(args.repo_path) if args.repo_path else None)
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
                        "message": "CLSTR controller could not be loaded for WebShop closed-loop eval.",
                        "error": repr(exc),
                        "reproduce_command": (
                            "python scripts/run_webshop_clstr_eval.py "
                            f"--output_dir {args.output_dir} "
                            f"--routing_init_manifest {args.routing_init_manifest} "
                            f"--checkpoint_path {args.checkpoint_path or ''} "
                            f"--aux_data_root {args.aux_data_root} "
                            f"--controller_mode {args.controller_mode}"
                        ),
                    },
                },
            }
    report = run_webshop_eval_or_blocker(
        output_dir=Path(args.output_dir),
        smoke_report=smoke_report,
        repo_path=Path(args.repo_path) if args.repo_path else None,
        controller=controller,
        controller_report=controller_report,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
