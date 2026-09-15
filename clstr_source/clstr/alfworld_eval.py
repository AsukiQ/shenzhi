from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from clstr.candidate_admission_residual import score_candidate_admission_residual
from clstr.counterfactual_memory_calibration import score_cmc_candidates
from clstr.action_adapter import UniversalActionAdapter
from clstr.alfworld_action_skills import alfworld_action_to_skill_id, map_alfworld_action_to_skill_id
from clstr.alfworld_skillrouter_finetune import SkillRouterActionProjectionAdapter
from clstr.aux_trajectories import load_aux_training_rows
from clstr.aux_pretrain import _build_model_from_routing_init
from clstr.belief import subspace_obs
from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.closed_loop_controller import (
    ClosedLoopControllerConfig,
    ClosedLoopControllerState,
    score_candidates_with_components,
)
from clstr.external_data import write_json, write_jsonl
from clstr.full_base_train import _apply_replay_prefix_beliefs, _encode_state_queries, _post_action_memory
from clstr.history_channel import strip_history_sections
from clstr.memory_utility_gate import (
    RELIABILITY_MODES,
    effective_memory_alpha,
    fuse_route_scores,
    heuristic_memory_alpha,
    memory_utility_features,
)
from clstr.memory_utility_gate_train import resolve_reliability_gate
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.memory_candidate_recall import environment_candidate_recall_metadata
from clstr.native_rerank import _filter_state_dict
from clstr.native_benchmark_checkpoint_adapter import (
    restore_native_benchmark_checkpoint_chain,
)
from clstr.qwen_direct_policy import (
    QwenDirectAdmissibleActionScorer,
    QwenDirectLikelihoodActionScorer,
    QwenDirectPolicyConfig,
)
from clstr.safe_memory_ranking import (
    bounded_memory_fusion,
    candidate_provenance_positive_residual_fusion,
)
from clstr.success_value import q_success_scores
from clstr.qwen_external_encoder import build_qwen_external_clstr_model


_GOAL_RE = re.compile(r"your task is to:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _validate_alfworld_resume_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    run_name: str,
) -> None:
    for index, row in enumerate(rows):
        if (
            int(row.get("episode_index", -1)) != index
            or str(row.get("split") or "") != str(split)
            or str(row.get("method") or "") != str(run_name)
        ):
            raise ValueError("ALFWorld resume identity does not match requested run")
        if not str(row.get("gamefile") or "").strip():
            raise ValueError("ALFWorld resume identity requires a gamefile for every episode")


def _resolve_existing_path(path_value: str | Path | None, *bases: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, _REPO_ROOT / candidate]
    candidates.extend(base / candidate for base in bases if not candidate.is_absolute())
    for item in candidates:
        if item.exists():
            return item
    return candidate if candidate.is_absolute() and candidate.exists() else None


def _build_model_from_checkpoint_config(
    config: dict[str, Any],
    skill_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_skills_path = output_dir / "selected_checkpoint_skills.jsonl"
    selected_skills_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skill_rows),
        encoding="utf-8",
    )
    allowed = {field.name for field in fields(CLSTRConfig)}
    model_config = {key: value for key, value in dict(config).items() if key in allowed}
    cfg = CLSTRConfig(**model_config)
    skills = load_eval_pool(selected_skills_path)
    model = CLSTRModel(cfg, skills)
    report = {
        "checkpoint_config_init": True,
        "selected_skills_path": str(selected_skills_path),
        "qdoc_adapter_used": False,
        "base_model_name": cfg.base_model_name,
        "skill_count": len(skill_rows),
    }
    return model, dict(config), report


def _extract_goal_from_initial_observation(observation: str) -> str:
    match = _GOAL_RE.search(observation or "")
    if match:
        return " ".join(match.group(1).strip().split())
    return ""


def build_alfworld_policy_state_text(
    goal_text: str,
    task_type: str,
    observation: str,
    action_history: list[str],
    history_window: int = 6,
    progress_memory: str | None = None,
) -> str:
    compact_history = " | ".join(action_history[-history_window:]) if action_history else "<empty>"
    lines = [
        f"goal: {goal_text}",
        f"task_type: {task_type}",
        f"observation: {observation}",
        f"history: {compact_history}",
    ]
    if progress_memory:
        lines.append(f"progress_memory: {progress_memory}")
    return "\n".join(lines)


def build_available_actions_planner_state_text(state_text: str, admissible_actions: list[str]) -> str:
    action_lines = "\n".join(f"{idx}. {action}" for idx, action in enumerate(admissible_actions, start=1))
    return "\n".join(
        [
            str(state_text),
            "",
            "AVAILABLE ACTIONS:",
            action_lines,
            "",
            "CLSTR skill-routing query:",
            "Infer the high-level skill intent and target objects/receptacles needed for the next step.",
            "Do not directly execute or copy an action; CLSTR will rank the admissible actions.",
        ]
    )


def _alfworld_router_state_text(state_text: str) -> str:
    return strip_history_sections(str(state_text or ""))


def _task_type_from_gamefile(gamefile: str | None) -> str:
    if not gamefile:
        return ""
    try:
        return Path(str(gamefile)).parents[1].name.split("-", 1)[0]
    except IndexError:
        return ""


def _read_yaml_like_eval_paths(config_text: str) -> list[str]:
    paths: list[str] = []
    in_eval_paths = False
    for raw in config_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("eval_paths:"):
            in_eval_paths = True
            continue
        if in_eval_paths:
            if line.startswith("- "):
                value = line[2:].strip().strip("'\"")
                if value:
                    paths.append(value)
                continue
            if not raw.startswith(" ") and not raw.startswith("\t"):
                break
    return paths


def _dependency_status(module_name: str) -> dict[str, Any]:
    return {
        "module": module_name,
        "available": importlib.util.find_spec(module_name) is not None,
    }


def build_alfworld_env_report(
    official_repo: str | Path,
    data_dir: str | Path | None,
    use_network_turbo: bool = False,
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir_path = Path(data_dir) if data_dir is not None else None
    expected_dirs = [
        "json_2.1.1/train",
        "json_2.1.1/valid_seen",
        "json_2.1.1/valid_unseen",
        "logic",
        "detectors",
    ]
    if data_dir_path is not None:
        expected = {entry: (data_dir_path / entry).exists() for entry in expected_dirs}
    else:
        expected = {entry: False for entry in expected_dirs}
    report = {
        "status": "ok" if official_repo.exists() else "missing",
        "official_repo": str(official_repo),
        "data_dir": str(data_dir_path) if data_dir_path is not None else None,
        "network_turbo_used": use_network_turbo,
        "dependencies": {
            name: _dependency_status(name)
            for name in ("alfworld", "textworld", "jericho", "gym", "termcolor")
        },
        "expected_paths": expected,
        "alfworld_data_env": os.environ.get("ALFWORLD_DATA"),
    }
    if data_dir_path is not None:
        report["available_paths"] = {
            "json_2.1.1": (data_dir_path / "json_2.1.1").exists(),
            "logic": (data_dir_path / "logic").exists(),
            "detectors": (data_dir_path / "detectors").exists(),
        }
    return report


def build_alfworld_protocol_report(
    official_repo: str | Path,
    output_path: str | Path,
    data_dir: str | Path | None = None,
    use_network_turbo: bool = False,
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    readme = _read_text(official_repo / "README.md")
    eval_config_text = _read_text(official_repo / "configs" / "eval_config.yaml")
    run_eval_path = official_repo / "scripts" / "run_eval.py"
    report = {
        "status": "ok" if official_repo.exists() and run_eval_path.exists() else "missing",
        "official_repo": str(official_repo),
        "official_eval_entrypoint": str(run_eval_path),
        "official_download_entrypoint": str(official_repo / "scripts" / "alfworld-download"),
        "official_eval_config": str(official_repo / "configs" / "eval_config.yaml"),
        "official_metrics": [
            "average_points",
            "average_goal_condition_points",
            "average_steps",
        ],
        "metrics": [
            "average_points",
            "average_goal_condition_points",
            "average_steps",
        ],
        "eval_splits": _read_yaml_like_eval_paths(eval_config_text),
        "supported_env_types": [
            "AlfredTWEnv",
            "AlfredThorEnv",
            "AlfredHybrid",
        ],
        "baseline_methods": ["dagger", "dqn"],
        "required_commands": [
            "alfworld-download",
            "python scripts/run_eval.py configs/eval_config.yaml",
        ],
        "install_notes": [
            "pip install alfworld[full]",
            "pip install textworld[pddl] jericho gym termcolor",
        ],
        "network_turbo_used": use_network_turbo,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "dependencies": {
            name: _dependency_status(name)
            for name in ("alfworld", "textworld", "jericho", "gym", "termcolor")
        },
        "readme_mentions": {
            "download": "alfworld-download" in readme,
            "run_eval": "run_eval.py" in readme,
            "install_full": "alfworld[full]" in readme,
        },
    }
    if data_dir is not None:
        report["env_report"] = build_alfworld_env_report(
            official_repo=official_repo,
            data_dir=data_dir,
            use_network_turbo=use_network_turbo,
        )
    write_json(Path(output_path), report)
    return report


def _pad_candidate_rows(candidate_rows: list[list[str]]) -> tuple[list[list[str]], torch.Tensor]:
    max_width = max(len(row) for row in candidate_rows)
    padded: list[list[str]] = []
    mask_rows: list[list[bool]] = []
    for row in candidate_rows:
        if len(row) < max_width:
            pad_value = row[-1] if row else ""
            padded.append(row + [pad_value] * (max_width - len(row)))
            mask_rows.append([True] * len(row) + [False] * (max_width - len(row)))
        else:
            padded.append(row)
            mask_rows.append([True] * len(row))
    return padded, torch.tensor(mask_rows, dtype=torch.bool)


def _encode_texts(
    model: CLSTRModel,
    texts: list[str],
    freeze: bool = True,
) -> torch.Tensor:
    if freeze:
        with torch.no_grad():
            return model.encode_observations(texts)
    return model.encode_observations(texts)


def _encode_state_texts(
    model: CLSTRModel,
    texts: list[str],
    freeze: bool = True,
) -> torch.Tensor:
    if freeze:
        with torch.no_grad():
            return _encode_state_queries(model, texts)
    return _encode_state_queries(model, texts)


def _encode_candidate_texts_cached(
    model: CLSTRModel,
    texts: list[str],
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    missing = [text for text in dict.fromkeys(str(item) for item in texts) if text not in cache]
    if missing:
        encoded = _encode_texts(model, missing, freeze=True).detach()
        for text, emb in zip(missing, encoded):
            cache[text] = emb.detach()
    return torch.stack([cache[str(text)].to(model.device) for text in texts])


class SkillRouterAdmissibleActionScorer:
    """Frozen SkillRouter-style scorer over the official admissible action set."""

    def __init__(
        self,
        model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
        batch_size: int = 16,
        max_length: int = 2048,
        adapter_checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.adapter_checkpoint_path = str(adapter_checkpoint_path) if adapter_checkpoint_path else None
        self._candidate_cache: dict[str, torch.Tensor] = {}
        self.last_metadata: list[dict[str, Any]] = []
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None
        self._adapter: SkillRouterActionProjectionAdapter | None = None
        self._adapter_payload: dict[str, Any] | None = None

    def _query_text(self, state_text: str) -> str:
        return (
            "Instruct: Given an ALFWorld goal, observation, and action history, "
            "retrieve the next admissible text action that best advances the task\n"
            f"Query:{str(state_text)[:1800]}"
        )

    def _action_text(self, action: str) -> str:
        return f"ALFWorld admissible action: {str(action)}"

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        tokenizer, model, device = self._load_encoder()
        encoded: list[torch.Tensor] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            tok = tokenizer(
                texts[start : start + max(1, self.batch_size)],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {key: value.to(device) for key, value in tok.items()}
            with torch.no_grad():
                out = model(**tok)
                pooled = self._last_token_pool(out.last_hidden_state, tok["attention_mask"])
                pooled = F.normalize(pooled.float(), p=2, dim=-1)
            encoded.append(pooled.cpu())
        return torch.cat(encoded, dim=0)

    def _load_encoder(self):
        if self._tokenizer is not None and self._model is not None and self._device is not None:
            return self._tokenizer, self._model, self._device
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return tokenizer, model, device

    def _load_adapter(self, hidden_size: int) -> SkillRouterActionProjectionAdapter | None:
        if not self.adapter_checkpoint_path:
            return None
        if self._adapter is not None:
            return self._adapter
        payload = torch.load(self.adapter_checkpoint_path, map_location="cpu")
        checkpoint_hidden = int(payload.get("hidden_size") or hidden_size)
        if checkpoint_hidden != int(hidden_size):
            raise ValueError(
                f"SkillRouter adapter hidden size mismatch: checkpoint={checkpoint_hidden}, encoder={hidden_size}"
            )
        adapter = SkillRouterActionProjectionAdapter(hidden_size=checkpoint_hidden, projection_init="identity")
        adapter.load_state_dict(payload["adapter_state_dict"])
        adapter.eval()
        self._adapter = adapter
        self._adapter_payload = payload
        return adapter

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, lengths]

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            self.last_metadata = []
            return torch.empty(0, 0)
        padded, candidate_mask = _pad_candidate_rows(candidate_texts)
        flat_candidates = [self._action_text(text) for row in padded for text in row]
        missing = [text for text in dict.fromkeys(flat_candidates) if text not in self._candidate_cache]
        if missing:
            encoded_missing = self._encode(missing).detach().float().cpu()
            for text, emb in zip(missing, encoded_missing):
                self._candidate_cache[text] = emb.detach()
        query_embs = self._encode([self._query_text(text) for text in state_texts]).detach().float()
        candidate_embs = torch.stack([self._candidate_cache[text] for text in flat_candidates]).float()
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=-1)
        adapter = self._load_adapter(int(query_embs.size(-1)))
        if adapter is not None:
            with torch.no_grad():
                query_embs = adapter.encode_queries(query_embs)
                candidate_embs = adapter.encode_docs(candidate_embs)
        candidate_embs = candidate_embs.view(len(padded), len(padded[0]), -1)
        scores = torch.einsum("bd,bcd->bc", query_embs, candidate_embs)
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        policy_family = (
            "skillrouter_finetuned_admissible_action"
            if self.adapter_checkpoint_path
            else "skillrouter_frozen_admissible_action"
        )
        self.last_metadata = [
            {
                "policy_family": policy_family,
                "model_name_or_path": self.model_name_or_path,
                "adapter_checkpoint_path": self.adapter_checkpoint_path,
                "available_action_count": len(candidates),
                "direct_action_generation": False,
            }
            for candidates in candidate_texts
        ]
        return scores


def make_candidate_scorer(
    model: CLSTRModel,
    action_adapter: UniversalActionAdapter | None = None,
    include_available_actions_in_state: bool = False,
    cache_candidate_embeddings: bool = True,
) -> Callable[[list[str], list[list[str]]], torch.Tensor]:
    device = model.device
    candidate_embedding_cache: dict[str, torch.Tensor] = {}

    def scorer(state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            scorer.last_metadata = []
            return torch.empty(0, 0, device=device)
        padded, candidate_mask = _pad_candidate_rows(candidate_texts)
        flat_candidates = [text for row in padded for text in row]
        scoring_state_texts = (
            [
                build_available_actions_planner_state_text(
                    _alfworld_router_state_text(state_text),
                    candidates,
                )
                for state_text, candidates in zip(state_texts, candidate_texts)
            ]
            if include_available_actions_in_state
            else [_alfworld_router_state_text(state_text) for state_text in state_texts]
        )
        scorer.last_metadata = [
            {
                "policy_family": "clstr_qwen_available_actions_planner"
                if include_available_actions_in_state
                else "clstr_native_action_scorer",
                "planner_intent_mode": "latent_available_actions_query"
                if include_available_actions_in_state
                else "none",
                "available_action_count": len(candidates),
                "qwen_direct_generator": False,
                "direct_action_generation": False,
            }
            for candidates in candidate_texts
        ]
        state_embs = _encode_state_texts(model, scoring_state_texts, freeze=True).to(device)
        if cache_candidate_embeddings:
            candidate_embs = _encode_candidate_texts_cached(model, flat_candidates, candidate_embedding_cache).to(device)
        else:
            candidate_embs = _encode_texts(model, flat_candidates, freeze=True).to(device)
        candidate_embs = candidate_embs.view(len(padded), len(padded[0]), -1)
        skill_head = getattr(model, "skill_head", None)
        if skill_head is not None:
            with torch.no_grad():
                belief_embs = _skill_memory(model, state_embs)
                try:
                    scores = skill_head(candidate_embs, belief_embs)
                except TypeError:
                    scores = None
                if scores is not None:
                    if scores.ndim == 3 and scores.size(-1) == 1:
                        scores = scores.squeeze(-1)
                    if scores.ndim == 2:
                        return scores.masked_fill(
                            ~candidate_mask.to(device),
                            torch.finfo(scores.dtype).min,
                        )
        if action_adapter is not None:
            action_adapter.eval()
            with torch.no_grad():
                scores = action_adapter(state_embs, candidate_embs, candidate_mask.to(device))
            return scores
        state_embs = F.normalize(state_embs.float(), p=2, dim=-1)
        candidate_embs = F.normalize(candidate_embs.float(), p=2, dim=-1)
        scores = torch.einsum("bd,bcd->bc", state_embs, candidate_embs)
        scores = scores.masked_fill(~candidate_mask.to(device), torch.finfo(scores.dtype).min)
        return scores

    return scorer


def _model_skill_id_to_idx(model: CLSTRModel) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, skill in enumerate(getattr(model, "skills", []) or []):
        if isinstance(skill, dict):
            skill_id = str(skill.get("skill_id") or "").strip()
        else:
            skill_id = str(getattr(skill, "skill_id", "") or "").strip()
        if skill_id:
            mapping[skill_id] = idx
    return mapping


def _skill_memory(model: CLSTRModel, h: torch.Tensor) -> torch.Tensor:
    skill_table = getattr(model, "skill_table", None)
    if skill_table is None or not hasattr(skill_table, "E"):
        return h
    if hasattr(skill_table, "belief_logits"):
        return subspace_obs(skill_table, h)
    if not hasattr(skill_table, "retrieval_logits"):
        return h
    logits = skill_table.retrieval_logits(h)
    return torch.softmax(logits, dim=-1) @ skill_table.E.to(h.device)


def _alfworld_history_actions_from_state_text(state_text: str) -> list[str]:
    for raw_line in str(state_text).splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("history:"):
            continue
        value = line.split(":", 1)[1].strip()
        if not value or value == "<empty>":
            return []
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def _alfworld_replay_prefix_from_state_text(
    state_text: str,
    skill_id_to_idx: dict[str, int],
    replay_prefix_max_steps: int,
) -> list[dict[str, Any]]:
    actions = _alfworld_history_actions_from_state_text(state_text)
    if replay_prefix_max_steps > 0:
        actions = actions[-replay_prefix_max_steps:]
    prefix: list[dict[str, Any]] = []
    known_skill_ids = set(skill_id_to_idx)
    for action in actions:
        mapping = map_alfworld_action_to_skill_id(action, known_skill_ids)
        if not mapping.skill_id or mapping.skill_id not in skill_id_to_idx:
            continue
        prefix.append(
            {
                "observation_text": _alfworld_router_state_text(state_text),
                "action_text": str(action),
                "next_observation_text": "",
                "skill_id": str(mapping.skill_id),
                "skill_idx": int(skill_id_to_idx[str(mapping.skill_id)]),
                "mapping_confidence": mapping.confidence,
                "mapping_reason": mapping.reason,
                "observation_source": "action_only_no_tool_result",
            }
        )
    return prefix


class ClstrUnifiedMemoryAdmissibleActionScorer:
    """Stateful causal CLSTR scorer over ALFWorld admissible actions."""

    def __init__(
        self,
        model: CLSTRModel,
        *,
        replay_prefix_max_steps: int = 6,
        reliability_mode: str = "dynamic",
        fixed_alpha: float = 1.0,
        safe_memory_residual_bound: float = 2.0,
        memory_utility_gate: torch.nn.Module | None = None,
        feature_update_count_cap: float = 1.0,
        feature_candidate_count_cap: float = 1.0,
    ) -> None:
        self.model = model
        self.replay_prefix_max_steps = max(0, int(replay_prefix_max_steps))
        self.reliability_mode = str(reliability_mode or "dynamic")
        self.fixed_alpha = float(fixed_alpha)
        self.safe_memory_residual_bound = float(safe_memory_residual_bound)
        self.memory_utility_gate = memory_utility_gate
        self.feature_update_count_cap = float(feature_update_count_cap)
        self.feature_candidate_count_cap = float(feature_candidate_count_cap)
        self.skill_id_to_idx = _model_skill_id_to_idx(model)
        self.last_metadata: list[dict[str, Any]] = []
        self.last_transition_inputs: dict[str, str] | None = None
        self.dynamic_memory: torch.Tensor | None = None
        self.causal_update_count: torch.Tensor | None = None
        if not callable(getattr(model, "initial_belief", None)):
            raise ValueError("ALFWorld unified memory scorer requires model.initial_belief")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"unsupported reliability mode: {self.reliability_mode}")
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be in [0, 1]")
        if self.reliability_mode == "learned" and not callable(memory_utility_gate):
            raise ValueError("learned reliability requires a memory utility gate")
        if callable(getattr(model, "eval", None)):
            model.eval()

    def reset_episode_batch(self, state_texts: list[str]) -> None:
        if not state_texts:
            raise ValueError("stateful ALFWorld memory requires a nonempty episode batch")
        device = torch.device(getattr(self.model, "device", "cpu"))
        with torch.no_grad():
            h = _encode_state_texts(
                self.model,
                [_alfworld_router_state_text(item) for item in state_texts],
                freeze=True,
            ).to(device)
            self.dynamic_memory = self.model.initial_belief(h).detach()
        self.causal_update_count = torch.zeros(len(state_texts), device=device, dtype=torch.float32)
        self.last_transition_inputs = None

    def observe_transitions(
        self,
        *,
        chosen_actions: list[str],
        next_observation_texts: list[str],
        next_state_texts: list[str],
        active_mask: list[bool],
    ) -> None:
        if self.dynamic_memory is None or self.causal_update_count is None:
            raise ValueError("observe_transitions requires reset_episode_batch first")
        batch_size = int(self.dynamic_memory.size(0))
        if not all(len(items) == batch_size for items in (chosen_actions, next_observation_texts, next_state_texts, active_mask)):
            raise ValueError("stateful ALFWorld transition batch size drift")
        active_indices = [idx for idx, active in enumerate(active_mask) if bool(active)]
        if not active_indices:
            return
        known_skill_ids = set(self.skill_id_to_idx)
        skill_indices: list[int] = []
        for idx in active_indices:
            if not str(next_observation_texts[idx]).strip() or not str(next_state_texts[idx]).strip():
                raise ValueError("active ALFWorld transition requires next observation and state")
            mapping = map_alfworld_action_to_skill_id(chosen_actions[idx], known_skill_ids)
            if not mapping.skill_id or mapping.skill_id not in self.skill_id_to_idx:
                raise ValueError(f"active ALFWorld action has no mapped skill: {chosen_actions[idx]}")
            skill_indices.append(int(self.skill_id_to_idx[mapping.skill_id]))
        device = self.dynamic_memory.device
        with torch.no_grad():
            action_embeddings = _encode_texts(
                self.model, [str(chosen_actions[idx]) for idx in active_indices], freeze=True
            ).to(device)
            observation_embeddings = _encode_texts(
                self.model, [str(next_observation_texts[idx]) for idx in active_indices], freeze=True
            ).to(device)
            h_next = _encode_state_texts(
                self.model,
                [_alfworld_router_state_text(next_state_texts[idx]) for idx in active_indices],
                freeze=True,
            ).to(device)
            active_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)
            _predicted, _observed, next_memory = _post_action_memory(
                self.model,
                m_t=self.dynamic_memory.index_select(0, active_tensor),
                current_skill_labels=torch.tensor(skill_indices, device=device, dtype=torch.long),
                action_embeddings=action_embeddings,
                observation_embeddings=observation_embeddings,
                h_next=h_next,
            )
            updated = self.dynamic_memory.clone()
            updated.index_copy_(0, active_tensor, next_memory.detach())
            self.dynamic_memory = updated
            self.causal_update_count.index_add_(
                0, active_tensor, torch.ones(len(active_indices), device=device)
            )
        first = active_indices[0]
        self.last_transition_inputs = {
            "action_text": str(chosen_actions[first]),
            "next_observation_text": str(next_observation_texts[first]),
            "next_state_text": str(next_state_texts[first]),
        }

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            self.last_metadata = []
            return torch.empty(0, 0, dtype=torch.float32)
        if self.dynamic_memory is None or self.causal_update_count is None:
            raise ValueError("stateful ALFWorld scorer requires reset_episode_batch first")
        if len(state_texts) != int(self.dynamic_memory.size(0)) or len(candidate_texts) != len(state_texts):
            raise ValueError("stateful ALFWorld scorer batch size drift")
        padded, candidate_mask = _pad_candidate_rows(candidate_texts)
        device = torch.device(getattr(self.model, "device", "cpu"))
        known_skill_ids = set(self.skill_id_to_idx)
        index_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        mapping_rows: list[list[dict[str, Any]]] = []
        for row_idx, row in enumerate(padded):
            indices: list[int] = []
            valid: list[bool] = []
            mappings: list[dict[str, Any]] = []
            for col_idx, action in enumerate(row):
                is_real_candidate = bool(candidate_mask[row_idx, col_idx].item())
                mapping = map_alfworld_action_to_skill_id(action, known_skill_ids)
                skill_idx = self.skill_id_to_idx.get(str(mapping.skill_id)) if mapping.skill_id else None
                indices.append(int(skill_idx) if skill_idx is not None else 0)
                valid.append(bool(is_real_candidate and skill_idx is not None))
                mappings.append(
                    {
                        "action": str(action),
                        "mapped_skill_id": mapping.skill_id,
                        "mapped_skill_idx": skill_idx,
                        "mapping_confidence": mapping.confidence,
                        "mapping_reason": mapping.reason,
                    }
                )
            index_rows.append(indices)
            valid_rows.append(valid)
            mapping_rows.append(mappings)

        with torch.no_grad():
            h = _encode_state_texts(
                self.model,
                [_alfworld_router_state_text(item) for item in state_texts],
                freeze=True,
            ).to(device)
            static_memory = self.model.initial_belief(h)
            dynamic_memory = self.dynamic_memory.to(device=device, dtype=static_memory.dtype)
            static_logits = self.model.unified_route_logits(h, static_memory, candidate_rows=index_rows).detach().float()
            raw_dynamic_logits = self.model.unified_route_logits(h, dynamic_memory, candidate_rows=index_rows).detach().float()
            valid_mask = torch.tensor(valid_rows, device=device, dtype=torch.bool)
            floor = torch.finfo(static_logits.dtype).min
            static_masked = static_logits.masked_fill(~valid_mask, floor)
            cmc_scoring = None
            candidate_embeddings = None
            if self.reliability_mode in {
                "cmc",
                "cmc_candidate_provenance",
                "candidate_admission_residual",
            }:
                candidate_indices = torch.tensor(
                    index_rows,
                    dtype=torch.long,
                    device=device,
                )
                candidate_embeddings = self.model.skill_table.E.index_select(
                    0,
                    candidate_indices.reshape(-1),
                ).view(len(index_rows), len(index_rows[0]), -1)
                if self.reliability_mode == "candidate_admission_residual":
                    adapter = getattr(
                        self.model,
                        "route_memory_residual_adapter",
                        None,
                    )
                    if not callable(adapter):
                        raise ValueError(
                            "candidate admission requires frozen CMC adapter"
                        )
                    route_residual = adapter(
                        h,
                        dynamic_memory - static_memory,
                    )
                    correction = torch.einsum(
                        "bd,bcd->bc",
                        route_residual,
                        candidate_embeddings.to(
                            device=route_residual.device,
                            dtype=route_residual.dtype,
                        ),
                    )
                    dynamic_masked = (
                        raw_dynamic_logits
                        + correction.to(dtype=raw_dynamic_logits.dtype)
                    ).masked_fill(~valid_mask, floor)
                    features = memory_utility_features(
                        static_masked,
                        dynamic_masked,
                        valid_mask,
                        static_memory.float(),
                        dynamic_memory.float(),
                        self.causal_update_count,
                        update_count_cap=self.feature_update_count_cap,
                        candidate_count_cap=self.feature_candidate_count_cap,
                    )
                else:
                    cmc_scoring = score_cmc_candidates(
                        self.model,
                        h=h,
                        static_memory=static_memory,
                        dynamic_memory=dynamic_memory,
                        static_logits=static_logits,
                        raw_dynamic_logits=raw_dynamic_logits,
                        candidate_embeddings=candidate_embeddings,
                        valid_mask=valid_mask,
                        causal_update_count=self.causal_update_count,
                        feature_update_count_cap=self.feature_update_count_cap,
                        feature_candidate_count_cap=self.feature_candidate_count_cap,
                    )
                    dynamic_masked = cmc_scoring.dynamic_logits
                    features = cmc_scoring.features
            else:
                dynamic_masked = raw_dynamic_logits.masked_fill(~valid_mask, floor)
                features = memory_utility_features(
                    static_masked,
                    dynamic_masked,
                    valid_mask,
                    static_memory.float(),
                    dynamic_memory.float(),
                    self.causal_update_count,
                    update_count_cap=self.feature_update_count_cap,
                    candidate_count_cap=self.feature_candidate_count_cap,
                )
            candidate_admission_scoring = None
            if self.reliability_mode == "candidate_admission_residual":
                head = getattr(
                    self.model,
                    "route_memory_candidate_admission_residual",
                    None,
                )
                if not callable(head) or candidate_embeddings is None:
                    raise ValueError("candidate admission head is missing")
                candidate_admission_scoring = score_candidate_admission_residual(
                    head,
                    h=h,
                    memory_delta=dynamic_memory - static_memory,
                    candidate_embeddings=candidate_embeddings,
                    static_logits=static_masked,
                    dynamic_logits=dynamic_masked,
                    dynamic_extra_mask=torch.zeros_like(valid_mask),
                    valid_mask=valid_mask,
                    causal_update_count=self.causal_update_count,
                )
                raw_alpha = (
                    candidate_admission_scoring.admission_probability
                    .masked_fill(~valid_mask, 0.0)
                    .sum(dim=-1)
                    / valid_mask.sum(dim=-1).clamp_min(1)
                )
            elif self.reliability_mode == "cmc":
                if cmc_scoring is None:
                    raise AssertionError("CMC scoring output is missing")
                raw_alpha = (
                    self.memory_utility_gate(features)
                    if self.memory_utility_gate is not None
                    else cmc_scoring.raw_alpha
                )
                if not isinstance(raw_alpha, torch.Tensor):
                    raise ValueError("CMC memory utility gate must return a tensor")
                raw_alpha = raw_alpha.to(
                    device=device,
                    dtype=static_logits.dtype,
                )
            elif self.reliability_mode == "cmc_candidate_provenance":
                if cmc_scoring is None:
                    raise AssertionError("CMC scoring output is missing")
                raw_alpha = static_logits.new_ones(len(state_texts))
            elif self.reliability_mode == "static":
                raw_alpha = static_logits.new_zeros(len(state_texts))
            elif self.reliability_mode == "dynamic":
                raw_alpha = static_logits.new_ones(len(state_texts))
            elif self.reliability_mode == "fixed_alpha":
                raw_alpha = static_logits.new_full((len(state_texts),), self.fixed_alpha)
            elif self.reliability_mode == "heuristic":
                raw_alpha = heuristic_memory_alpha(features).to(device)
            elif self.reliability_mode == "learned":
                raw_alpha = self.memory_utility_gate(features).to(device=device, dtype=static_logits.dtype)
            else:
                route_alpha_fn = getattr(self.model, "route_memory_alpha", None)
                if not callable(route_alpha_fn):
                    raise ValueError(
                        "causal_gate reliability mode requires model.route_memory_alpha"
                    )
                raw_alpha = route_alpha_fn(
                    h,
                    static_memory,
                    dynamic_memory,
                    self.causal_update_count,
                ).to(device=device, dtype=static_logits.dtype)
            effective_alpha = effective_memory_alpha(
                raw_alpha,
                self.causal_update_count,
            )
            if self.reliability_mode == "candidate_admission_residual":
                if candidate_admission_scoring is None:
                    raise AssertionError(
                        "candidate admission scoring output is missing"
                    )
                scores = candidate_admission_scoring.final_logits.cpu()
            elif self.reliability_mode == "cmc_candidate_provenance":
                if cmc_scoring is None:
                    raise AssertionError("CMC scoring output is missing")
                scores = candidate_provenance_positive_residual_fusion(
                    static_masked,
                    dynamic_masked,
                    torch.zeros_like(valid_mask),
                    valid_mask,
                    self.causal_update_count,
                    residual_bound=self.safe_memory_residual_bound,
                ).cpu()
            elif cmc_scoring is not None:
                scores = fuse_route_scores(
                    static_masked,
                    dynamic_masked,
                    effective_alpha,
                    valid_mask,
                ).cpu()
            elif self.reliability_mode == "causal_gate":
                scores = bounded_memory_fusion(
                    static_masked,
                    dynamic_masked,
                    effective_alpha,
                    valid_mask,
                    residual_bound=self.safe_memory_residual_bound,
                ).cpu()
            else:
                scores = fuse_route_scores(
                    static_masked,
                    dynamic_masked,
                    effective_alpha,
                    valid_mask,
                ).cpu()
        self.last_metadata = [
            {
                "policy_family": "clstr_unified_memory_admissible_action_scorer",
                "candidate_skill_mappings": mapping_rows[row_idx][: len(candidate_texts[row_idx])],
                "candidate_count": len(candidate_texts[row_idx]),
                "route_scorer": "unified_memory",
                "memory_utility_reliability_mode": self.reliability_mode,
                "memory_utility_alpha": float(effective_alpha[row_idx].item()),
                "memory_utility_external_gate_loaded": self.memory_utility_gate
                is not None,
                "safe_memory_residual_bound": self.safe_memory_residual_bound,
                "zero_history_fallback": "exact_static",
                "uses_recurrent_m_t": bool(self.causal_update_count[row_idx].item() > 0),
                "causal_update_count": int(self.causal_update_count[row_idx].item()),
                "memory_protocol": "stateful_post_action_v1",
            }
            for row_idx in range(len(candidate_texts))
        ]
        return scores


class ClstrUnifiedMemoryConcreteActionScorer:
    """CLSTR recurrent-memory scorer over concrete ALFWorld action text."""

    def __init__(
        self,
        model: CLSTRModel,
        *,
        replay_prefix_max_steps: int = 6,
        cache_candidate_embeddings: bool = True,
        reliability_mode: str = "dynamic",
        fixed_alpha: float = 1.0,
        safe_memory_residual_bound: float = 2.0,
        memory_utility_gate: torch.nn.Module | None = None,
        feature_update_count_cap: float = 1.0,
        feature_candidate_count_cap: float = 1.0,
    ) -> None:
        self.model = model
        self.replay_prefix_max_steps = max(0, int(replay_prefix_max_steps))
        self.cache_candidate_embeddings = bool(cache_candidate_embeddings)
        self.reliability_mode = str(reliability_mode or "dynamic")
        self.fixed_alpha = float(fixed_alpha)
        self.safe_memory_residual_bound = float(safe_memory_residual_bound)
        self.memory_utility_gate = memory_utility_gate
        self.feature_update_count_cap = float(feature_update_count_cap)
        self.feature_candidate_count_cap = float(feature_candidate_count_cap)
        self.skill_id_to_idx = _model_skill_id_to_idx(model)
        self.candidate_embedding_cache: dict[str, torch.Tensor] = {}
        self.last_metadata: list[dict[str, Any]] = []
        if not callable(getattr(model, "initial_belief", None)):
            raise ValueError("ALFWorld unified memory concrete-action scorer requires model.initial_belief")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"unsupported reliability mode: {self.reliability_mode}")
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be in [0, 1]")
        if self.reliability_mode == "learned" and not callable(memory_utility_gate):
            raise ValueError("learned reliability requires a memory utility gate")
        if callable(getattr(model, "eval", None)):
            model.eval()

    def _history_replay_prefix(self, state_text: str) -> list[dict[str, Any]]:
        return _alfworld_replay_prefix_from_state_text(
            state_text,
            self.skill_id_to_idx,
            self.replay_prefix_max_steps,
        )

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        if not candidate_texts:
            self.last_metadata = []
            return torch.empty(0, 0, dtype=torch.float32)
        padded, candidate_mask = _pad_candidate_rows(candidate_texts)
        flat_candidates = [str(text) for row in padded for text in row]
        device = torch.device(getattr(self.model, "device", "cpu"))
        with torch.no_grad():
            h = _encode_state_texts(
                self.model,
                [_alfworld_router_state_text(item) for item in state_texts],
                freeze=True,
            ).to(device)
            static_memory = self.model.initial_belief(h)
            replay_rows = [{"replay_prefix": self._history_replay_prefix(str(item))} for item in state_texts]
            dynamic_memory, replay_prefix_used_count = _apply_replay_prefix_beliefs(
                self.model,
                replay_rows,
                static_memory,
                self.skill_id_to_idx,
                len(self.skill_id_to_idx),
                device,
                trainable=False,
            )
            if self.cache_candidate_embeddings:
                candidate_embs = _encode_candidate_texts_cached(
                    self.model,
                    flat_candidates,
                    self.candidate_embedding_cache,
                ).to(device)
            else:
                candidate_embs = _encode_texts(self.model, flat_candidates, freeze=True).to(device)
            candidate_embs = candidate_embs.view(len(padded), len(padded[0]), -1)
            skill_head = getattr(self.model, "skill_head", None)
            def candidate_scores(memory: torch.Tensor) -> torch.Tensor:
                output = None
                if skill_head is not None:
                    try:
                        output = skill_head(candidate_embs, memory)
                    except TypeError:
                        output = None
                    if output is not None and output.ndim == 3 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                if output is None or output.ndim != 2:
                    memory_norm = F.normalize(memory.float(), p=2, dim=-1)
                    candidate_norm = F.normalize(candidate_embs.float(), p=2, dim=-1)
                    output = torch.einsum("bd,bcd->bc", memory_norm, candidate_norm)
                return output.detach().float()

            valid_mask = candidate_mask.to(device=device, dtype=torch.bool)
            floor = torch.finfo(torch.float32).min
            static_logits = candidate_scores(static_memory).masked_fill(~valid_mask, floor)
            raw_dynamic_logits = candidate_scores(dynamic_memory).masked_fill(~valid_mask, floor)
            causal_update_count = torch.tensor(
                [
                    float(
                        sum(
                            1
                            for step in row["replay_prefix"]
                            if str(step.get("action_text") or "").strip()
                        )
                    )
                    for row in replay_rows
                ],
                dtype=static_logits.dtype,
                device=device,
            )
            cmc_scoring = None
            if self.reliability_mode in {"cmc", "cmc_candidate_provenance"}:
                cmc_scoring = score_cmc_candidates(
                    self.model,
                    h=h,
                    static_memory=static_memory,
                    dynamic_memory=dynamic_memory,
                    static_logits=static_logits,
                    raw_dynamic_logits=raw_dynamic_logits,
                    candidate_embeddings=candidate_embs,
                    valid_mask=valid_mask,
                    causal_update_count=causal_update_count,
                    feature_update_count_cap=self.feature_update_count_cap,
                    feature_candidate_count_cap=self.feature_candidate_count_cap,
                )
                features = cmc_scoring.features
                if self.reliability_mode == "cmc":
                    raw_alpha = (
                        self.memory_utility_gate(features)
                        if self.memory_utility_gate is not None
                        else cmc_scoring.raw_alpha
                    )
                    if not isinstance(raw_alpha, torch.Tensor):
                        raise ValueError("CMC memory utility gate must return a tensor")
                    raw_alpha = raw_alpha.to(
                        device=device,
                        dtype=static_logits.dtype,
                    )
                else:
                    raw_alpha = static_logits.new_ones(len(state_texts))
                effective_alpha = effective_memory_alpha(
                    raw_alpha,
                    causal_update_count,
                )
                if self.reliability_mode == "cmc_candidate_provenance":
                    scores = candidate_provenance_positive_residual_fusion(
                        static_logits,
                        cmc_scoring.dynamic_logits,
                        torch.zeros_like(valid_mask),
                        valid_mask,
                        causal_update_count,
                        residual_bound=self.safe_memory_residual_bound,
                    )
                else:
                    scores = fuse_route_scores(
                        static_logits,
                        cmc_scoring.dynamic_logits,
                        effective_alpha,
                        valid_mask,
                    )
            else:
                features = memory_utility_features(
                    static_logits,
                    raw_dynamic_logits,
                    valid_mask,
                    static_memory.float(),
                    dynamic_memory.float(),
                    causal_update_count,
                    update_count_cap=self.feature_update_count_cap,
                    candidate_count_cap=self.feature_candidate_count_cap,
                )
                if self.reliability_mode == "static":
                    raw_alpha = static_logits.new_zeros(len(state_texts))
                elif self.reliability_mode == "dynamic":
                    raw_alpha = static_logits.new_ones(len(state_texts))
                elif self.reliability_mode == "fixed_alpha":
                    raw_alpha = static_logits.new_full((len(state_texts),), self.fixed_alpha)
                elif self.reliability_mode == "heuristic":
                    raw_alpha = heuristic_memory_alpha(features).to(device)
                elif self.reliability_mode == "learned":
                    raw_alpha = self.memory_utility_gate(features).to(
                        device=device,
                        dtype=static_logits.dtype,
                    )
                else:
                    route_alpha_fn = getattr(self.model, "route_memory_alpha", None)
                    if not callable(route_alpha_fn):
                        raise ValueError(
                            "causal_gate reliability mode requires model.route_memory_alpha"
                        )
                    raw_alpha = route_alpha_fn(
                        h,
                        static_memory,
                        dynamic_memory,
                        causal_update_count,
                    ).to(device=device, dtype=static_logits.dtype)
                effective_alpha = effective_memory_alpha(
                    raw_alpha,
                    causal_update_count,
                )
                if self.reliability_mode == "causal_gate":
                    scores = bounded_memory_fusion(
                        static_logits,
                        raw_dynamic_logits,
                        effective_alpha,
                        valid_mask,
                        residual_bound=self.safe_memory_residual_bound,
                    )
                else:
                    scores = fuse_route_scores(
                        static_logits,
                        raw_dynamic_logits,
                        effective_alpha,
                        valid_mask,
                    )
        known_skill_ids = set(self.skill_id_to_idx)
        mapping_rows: list[list[dict[str, Any]]] = []
        for row in candidate_texts:
            mappings = []
            for action in row:
                mapping = map_alfworld_action_to_skill_id(action, known_skill_ids)
                mappings.append(
                    {
                        "action": str(action),
                        "mapped_skill_id": mapping.skill_id,
                        "mapping_confidence": mapping.confidence,
                        "mapping_reason": mapping.reason,
                    }
                )
            mapping_rows.append(mappings)
        self.last_metadata = [
            {
                "policy_family": "clstr_unified_memory_concrete_action_scorer",
                "clstr_memory_source": "replay_prefix" if replay_rows[row_idx]["replay_prefix"] else "initial_belief",
                "clstr_replay_prefix_len": len(replay_rows[row_idx]["replay_prefix"]),
                "clstr_replay_prefix_used_count": int(replay_prefix_used_count),
                "candidate_skill_mappings": mapping_rows[row_idx],
                "candidate_count": len(candidate_texts[row_idx]),
                "route_scorer": "unified_memory_concrete_action",
                "uses_recurrent_m_t": bool(replay_rows[row_idx]["replay_prefix"]),
                "concrete_action_text_scoring": True,
                "memory_utility_reliability_mode": self.reliability_mode,
                "memory_utility_alpha": float(effective_alpha[row_idx].item()),
                "memory_utility_external_gate_loaded": self.memory_utility_gate
                is not None,
                "zero_history_fallback": "exact_static",
            }
            for row_idx in range(len(candidate_texts))
        ]
        return scores.cpu()


def make_alfworld_clstr_candidate_scorer(
    model: CLSTRModel,
    action_adapter: UniversalActionAdapter | None,
    *,
    scorer_mode: str = "legacy_concrete_action_head",
    include_available_actions_in_state: bool = False,
    replay_prefix_max_steps: int = 6,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    safe_memory_residual_bound: float = 2.0,
    memory_utility_gate: torch.nn.Module | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
) -> Callable[[list[str], list[list[str]]], torch.Tensor]:
    mode = str(scorer_mode).strip()
    if mode == "legacy_concrete_action_head":
        if include_available_actions_in_state:
            return make_candidate_scorer(
                model,
                action_adapter,
                include_available_actions_in_state=True,
            )
        return make_candidate_scorer(model, action_adapter)
    if include_available_actions_in_state:
        raise ValueError("unified-memory ALFWorld scorers do not use the planner-state prompt")
    if mode == "unified_memory_admissible_action":
        return ClstrUnifiedMemoryAdmissibleActionScorer(
            model,
            replay_prefix_max_steps=replay_prefix_max_steps,
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            safe_memory_residual_bound=safe_memory_residual_bound,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
        )
    if mode == "unified_memory_concrete_action":
        return ClstrUnifiedMemoryConcreteActionScorer(
            model,
            replay_prefix_max_steps=replay_prefix_max_steps,
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            safe_memory_residual_bound=safe_memory_residual_bound,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
        )
    raise ValueError(f"unsupported ALFWorld CLSTR scorer mode: {scorer_mode}")


def _stop_logits_for_state(model: CLSTRModel, h: torch.Tensor, m_obs: torch.Tensor) -> torch.Tensor:
    stop_head = getattr(model, "stop_head", None)
    if stop_head is None:
        return torch.zeros(h.size(0), device=h.device)
    try:
        out = stop_head(h, m_obs)
    except TypeError:
        out = stop_head(torch.cat([h, m_obs], dim=-1))
    return out.squeeze(-1)


def make_controller_component_scorer(
    model: CLSTRModel,
) -> Callable[[list[str], list[list[str]], torch.Tensor, list[list[str]]], dict[str, torch.Tensor]]:
    device = model.device
    skill_ids = set(_model_skill_id_to_idx(model))

    def scorer(
        state_texts: list[str],
        candidate_texts: list[list[str]],
        policy_scores: torch.Tensor,
        action_histories: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        del action_histories
        if not candidate_texts:
            return {
                "transition_scores": torch.empty(0, 0, device=device),
                "belief_scores": torch.empty(0, 0, device=device),
                "stop_logits": torch.empty(0, 0, device=device),
            }
        padded, candidate_mask = _pad_candidate_rows(candidate_texts)
        flat_candidates = [text for row in padded for text in row]
        with torch.no_grad():
            h = _encode_state_texts(
                model,
                [_alfworld_router_state_text(item) for item in state_texts],
                freeze=True,
            ).to(device)
            action_embs = _encode_texts(model, flat_candidates, freeze=True).to(device)
            action_embs = action_embs.view(len(padded), len(padded[0]), -1)
            m_obs = _skill_memory(model, h)
            mapping_rows: list[list[dict[str, Any]]] = []
            flat_mappings: list[dict[str, Any]] = []
            for action in flat_candidates:
                mapping = map_alfworld_action_to_skill_id(action, skill_ids)
                flat_mappings.append(
                    {
                        "action": str(action),
                        "mapped_skill_id": mapping.skill_id,
                        "mapping_confidence": mapping.confidence,
                        "mapping_reason": mapping.reason,
                    }
                )
            for row_idx in range(len(padded)):
                start = row_idx * len(padded[0])
                mapping_rows.append(flat_mappings[start : start + len(padded[0])])
            flat_m_obs = m_obs.unsqueeze(1).expand(-1, len(padded[0]), -1).reshape(-1, m_obs.size(-1))
            transition_source = "disabled_no_post_action_observation"
            transition_candidate_embedding_source = "not_used"
            belief_source = "disabled_no_post_action_observation"
            transition_scores = torch.zeros_like(policy_scores)
            belief_scores = torch.zeros_like(policy_scores)
            flat_h = h.unsqueeze(1).expand(-1, len(padded[0]), -1).reshape(-1, h.size(-1))
            stop_state = _stop_logits_for_state(model, flat_h, flat_m_obs).view(len(padded), len(padded[0]))
            q_success_head = getattr(model, "q_success_head", None)
            if q_success_head is not None:
                q_success_state = q_success_scores(q_success_head, h, m_obs, action_embs, candidate_mask.to(device))
            else:
                q_success_state = torch.zeros_like(policy_scores)
            output_device = policy_scores.device
            mask = candidate_mask.to(output_device)
            min_value = torch.finfo(policy_scores.dtype).min
            transition_scores = transition_scores.to(
                device=output_device, dtype=policy_scores.dtype
            ).masked_fill(~mask, min_value)
            belief_scores = belief_scores.to(
                device=output_device, dtype=policy_scores.dtype
            ).masked_fill(~mask, min_value)
            stop_state = stop_state.to(
                device=output_device, dtype=policy_scores.dtype
            ).masked_fill(~mask, min_value)
            q_success_state = q_success_state.to(
                device=output_device, dtype=policy_scores.dtype
            ).masked_fill(~mask, min_value)
            scorer.last_metadata = [
                {
                    "component_score_source": "policy_q_success_stop_only_no_proxy_observation",
                    "transition_score_source": transition_source,
                    "transition_candidate_embedding_source": transition_candidate_embedding_source,
                    "belief_score_source": belief_source,
                    "candidate_skill_mappings": mapping_rows[row_idx][: len(candidate_texts[row_idx])],
                }
                for row_idx in range(len(candidate_texts))
            ]
        return {
            "transition_scores": transition_scores,
            "belief_scores": belief_scores,
            "stop_logits": stop_state,
            "q_success_scores": q_success_state,
        }

    scorer.supports_transition_scores = False
    scorer.supports_belief_scores = False
    scorer.supports_stop_scores = True
    scorer.supports_q_success_scores = True
    return scorer


def _make_env_config(
    official_repo: Path,
    data_dir: Path,
    split: str,
) -> tuple[dict[str, Any], str]:
    import yaml

    config_path = official_repo / "configs" / "eval_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    os.environ["ALFWORLD_DATA"] = str(data_dir)
    env_var_config = json.loads(json.dumps(config))
    env_var_config["general"]["use_cuda"] = torch.cuda.is_available()
    env_var_config["general"]["evaluate"]["run_eval"] = True
    env_var_config["general"]["evaluate"]["env"]["type"] = "AlfredTWEnv"
    env_var_config["env"]["type"] = "AlfredTWEnv"
    env_var_config["controller"]["type"] = "tw"
    if split == "valid_seen":
        env_var_config["dataset"]["eval_id_data_path"] = str(data_dir / "json_2.1.1" / "valid_seen")
        env_var_config["dataset"]["eval_ood_data_path"] = None
        train_eval = "eval_in_distribution"
    elif split == "valid_unseen":
        env_var_config["dataset"]["eval_id_data_path"] = None
        env_var_config["dataset"]["eval_ood_data_path"] = str(data_dir / "json_2.1.1" / "valid_unseen")
        train_eval = "eval_out_of_distribution"
    else:
        raise ValueError(f"unsupported ALFWorld split: {split}")
    env_var_config["dataset"]["num_eval_games"] = -1
    env_var_config["general"]["evaluate"]["eval_paths"] = [
        env_var_config["dataset"]["eval_id_data_path"] or env_var_config["dataset"]["eval_ood_data_path"]
    ]
    return env_var_config, train_eval


def run_alfworld_closed_loop_eval(
    env_factory,
    config: dict[str, Any],
    split: str,
    candidate_scorer: Callable[[list[str], list[list[str]]], torch.Tensor],
    output_dir: str | Path,
    max_episodes: int | None = None,
    max_steps: int = 50,
    batch_size: int = 1,
    run_name: str = "alfworld_clstr",
    component_scorer: Callable[[list[str], list[list[str]], torch.Tensor, list[list[str]]], dict[str, torch.Tensor]] | None = None,
    controller_config: ClosedLoopControllerConfig | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.jsonl"
    progress_path = output_dir / "progress.json"
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("ALFWorld batch_size must be a positive integer")
    batch_size = int(batch_size)
    run_rows = (
        _read_jsonl(run_path)
        if run_path.exists() and run_path.stat().st_size > 0
        else []
    )
    env_wrapper = env_factory(config, train_eval=split)
    eval_env = env_wrapper.init_env(batch_size=batch_size)
    if hasattr(eval_env, "seed"):
        eval_env.seed(42)
    target_episodes = max_episodes if max_episodes is not None else int(getattr(env_wrapper, "num_games", 1))
    run_rows = run_rows[:target_episodes]
    _validate_alfworld_resume_rows(run_rows, split=split, run_name=run_name)
    complete_resume = len(run_rows) >= target_episodes
    if run_rows and not complete_resume and len(run_rows) % batch_size != 0:
        raise ValueError(
            "ALFWorld batched resume requires previously committed complete batches"
        )
    res_points: list[float] = [float(row.get("points", 0.0)) for row in run_rows]
    res_steps: list[float] = [float(row.get("steps", 0.0)) for row in run_rows]
    res_gcs: list[float] = [float(row.get("goal_condition_points", 0.0)) for row in run_rows]
    skipped_episodes = 0
    while not complete_resume and skipped_episodes < len(run_rows):
        obs_skip, infos_skip = eval_env.reset()
        stored_chunk = run_rows[skipped_episodes : skipped_episodes + len(obs_skip)]
        skipped_gamefiles = list(
            infos_skip.get(
                "extra.gamefile",
                [f"{split}/episode-{skipped_episodes + index}" for index in range(len(obs_skip))],
            )
        )
        if len(stored_chunk) != len(obs_skip) or [
            str(row.get("gamefile") or "") for row in stored_chunk
        ] != [str(item) for item in skipped_gamefiles]:
            raise ValueError("ALFWorld resume gamefile order does not match environment")
        skipped_episodes += len(obs_skip)
    episode_no = len(run_rows)
    while episode_no < target_episodes:
        obs, infos = eval_env.reset()
        game_names = list(infos.get("extra.gamefile", [f"{split}/episode-{episode_no}"] * len(obs)))
        batch_n = len(obs)
        execute_actions = ["restart"] * batch_n
        prev_rewards = [0.0] * batch_n
        prev_dones = [0.0] * batch_n
        still_running_mask: list[list[float]] = []
        seq_points: list[list[float]] = []
        seq_gcs: list[list[float]] = []
        action_trace: list[list[str]] = [[] for _ in range(batch_n)]
        action_histories: list[list[str]] = [[] for _ in range(batch_n)]
        candidate_trace: list[list[list[str]]] = [[] for _ in range(batch_n)]
        chosen_index_trace: list[list[int]] = [[] for _ in range(batch_n)]
        chosen_action_trace: list[list[str]] = [[] for _ in range(batch_n)]
        chosen_reason_trace: list[list[str]] = [[] for _ in range(batch_n)]
        component_score_trace: list[list[list[dict[str, Any]]]] = [[] for _ in range(batch_n)]
        score_trace_top_actions: list[list[list[dict[str, Any]]]] = [[] for _ in range(batch_n)]
        policy_metadata_trace: list[list[dict[str, Any]]] = [[] for _ in range(batch_n)]
        raw_model_response_trace: list[list[str]] = [[] for _ in range(batch_n)]
        parsed_action_trace: list[list[str]] = [[] for _ in range(batch_n)]
        parse_status_trace: list[list[str]] = [[] for _ in range(batch_n)]
        fallback_trace: list[list[bool]] = [[] for _ in range(batch_n)]
        policy_family_trace: list[list[str]] = [[] for _ in range(batch_n)]
        observation_strings = list(obs)
        goal_texts = [_extract_goal_from_initial_observation(item) for item in observation_strings]
        task_types = [_task_type_from_gamefile(item) for item in game_names]
        if callable(getattr(candidate_scorer, "reset_episode_batch", None)):
            initial_state_texts = [
                build_alfworld_policy_state_text(goal, task_type, observation, history)
                for goal, task_type, observation, history in zip(
                    goal_texts, task_types, observation_strings, action_histories
                )
            ]
            candidate_scorer.reset_episode_batch(initial_state_texts)
        for step_no in range(max_steps):
            candidate_texts = list(infos.get("admissible_commands", [[] for _ in range(batch_n)]))
            state_texts = [
                build_alfworld_policy_state_text(goal, task_type, observation, history)
                for goal, task_type, observation, history in zip(goal_texts, task_types, observation_strings, action_histories)
            ]
            scores = candidate_scorer(state_texts, candidate_texts)
            scorer_step_metadata = getattr(candidate_scorer, "last_metadata", None)
            if scores.ndim != 2:
                raise ValueError("candidate_scorer must return [batch, candidates] scores")
            if component_scorer is not None:
                component_tensors = component_scorer(state_texts, candidate_texts, scores, action_histories)
                decisions = score_candidates_with_components(
                    candidate_rows=candidate_texts,
                    policy_scores=scores,
                    transition_scores=component_tensors.get("transition_scores"),
                    belief_scores=component_tensors.get("belief_scores"),
                    stop_logits=component_tensors.get("stop_logits"),
                    state=ClosedLoopControllerState(action_history=action_histories),
                    config=controller_config or ClosedLoopControllerConfig(),
                    planner_scores=component_tensors.get("planner_scores"),
                    q_success_scores=component_tensors.get("q_success_scores"),
                )
                chosen_indices = [decision.chosen_index for decision in decisions]
                execute_actions = [decision.chosen_action for decision in decisions]
            else:
                decisions = []
                chosen_indices = torch.argmax(scores, dim=-1).tolist()
                execute_actions = [row[idx] for row, idx in zip(candidate_texts, chosen_indices)]
            for idx, action in enumerate(execute_actions):
                candidates = [str(item) for item in candidate_texts[idx]]
                candidate_trace[idx].append(candidates)
                chosen_index_trace[idx].append(int(chosen_indices[idx]))
                chosen_action_trace[idx].append(str(action))
                metadata_item: dict[str, Any] = {}
                if isinstance(scorer_step_metadata, list) and idx < len(scorer_step_metadata) and isinstance(scorer_step_metadata[idx], dict):
                    metadata_item = dict(scorer_step_metadata[idx])
                component_step_metadata = getattr(component_scorer, "last_metadata", None) if component_scorer is not None else None
                if (
                    isinstance(component_step_metadata, list)
                    and idx < len(component_step_metadata)
                    and isinstance(component_step_metadata[idx], dict)
                ):
                    metadata_item.update(component_step_metadata[idx])
                policy_metadata_trace[idx].append(metadata_item)
                raw_model_response_trace[idx].append(str(metadata_item.get("raw_model_response", "")))
                parsed_action_trace[idx].append(str(metadata_item.get("parsed_action", "")))
                parse_status_trace[idx].append(str(metadata_item.get("parse_status", "")))
                fallback_trace[idx].append(bool(metadata_item.get("fallback_used", False)))
                policy_family_trace[idx].append(str(metadata_item.get("policy_family", "")))
                if decisions:
                    chosen_reason_trace[idx].append(decisions[idx].chosen_reason)
                    component_score_trace[idx].append(decisions[idx].component_scores)
                else:
                    chosen_reason_trace[idx].append("policy_argmax")
                    component_score_trace[idx].append([])
                valid_scores = scores[idx, : len(candidates)].detach().float().cpu()
                if candidates:
                    top_indices = torch.topk(valid_scores, k=min(5, len(candidates))).indices.tolist()
                    score_trace_top_actions[idx].append(
                        [
                            {
                                "action": candidates[top_idx],
                                "score": round(float(valid_scores[top_idx].item()), 6),
                            }
                            for top_idx in top_indices
                        ]
                    )
                else:
                    score_trace_top_actions[idx].append([])
                action_trace[idx].append(action)
                action_histories[idx].append(action)
            active_before_step = [not bool(item) for item in prev_dones]
            obs, _, dones, infos = eval_env.step(execute_actions)
            scores_now = [float(item) for item in infos["won"]]
            gcs = (
                [float(item) for item in infos["goal_condition_success_rate"]]
                if "goal_condition_success_rate" in infos
                else [0.0] * batch_n
            )
            dones = [float(item) for item in dones]
            observation_strings = list(obs)
            if callable(getattr(candidate_scorer, "observe_transitions", None)):
                next_state_texts = [
                    build_alfworld_policy_state_text(goal, task_type, observation, history)
                    for goal, task_type, observation, history in zip(
                        goal_texts, task_types, observation_strings, action_histories
                    )
                ]
                candidate_scorer.observe_transitions(
                    chosen_actions=[str(item) for item in execute_actions],
                    next_observation_texts=[str(item) for item in observation_strings],
                    next_state_texts=next_state_texts,
                    active_mask=active_before_step,
                )
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores_now, prev_rewards)]
            prev_rewards = scores_now
            seq_points.append(step_rewards)
            seq_gcs.append(gcs)
            still_running_mask.append([1.0 - float(item) for item in prev_dones])
            prev_dones = dones
            if sum(1.0 - float(item) for item in dones) == 0:
                break
        game_steps = torch.tensor(still_running_mask).sum(dim=0).tolist() if still_running_mask else [0.0] * batch_n
        game_points = torch.tensor(seq_points).max(dim=0).values.tolist() if seq_points else [0.0] * batch_n
        game_gcs = torch.tensor(seq_gcs).max(dim=0).values.tolist() if seq_gcs else [0.0] * batch_n
        batch_rows: list[dict[str, Any]] = []
        batch_points: list[float] = []
        batch_gcs: list[float] = []
        batch_steps: list[float] = []
        for i in range(batch_n):
            if len(run_rows) + len(batch_rows) >= target_episodes:
                break
            scorer_update_counts = getattr(candidate_scorer, "causal_update_count", None)
            scorer_update_count = (
                int(scorer_update_counts[i].item())
                if isinstance(scorer_update_counts, torch.Tensor)
                and i < int(scorer_update_counts.numel())
                else 0
            )
            replay_update_count = max(
                (
                    int(item.get("clstr_replay_prefix_len") or 0)
                    for item in policy_metadata_trace[i]
                    if isinstance(item, Mapping)
                    and bool(item.get("uses_recurrent_m_t"))
                ),
                default=0,
            )
            final_update_count = max(scorer_update_count, replay_update_count)
            row = {
                    "episode_index": len(run_rows) + len(batch_rows),
                    "split": split,
                    "gamefile": game_names[i],
                    "success": bool(game_points[i] > 0.0),
                    "points": float(game_points[i]),
                    "goal_condition_points": float(game_gcs[i]),
                    "steps": float(game_steps[i]),
                    "action_trace": action_trace[i],
                    "candidate_trace": candidate_trace[i],
                    "chosen_indices": chosen_index_trace[i],
                    "chosen_action_trace": chosen_action_trace[i],
                    "chosen_reason_trace": chosen_reason_trace[i],
                    "component_score_trace": component_score_trace[i],
                    "score_trace_top_actions": score_trace_top_actions[i],
                    "policy_metadata_trace": policy_metadata_trace[i],
                    "policy_family_trace": policy_family_trace[i],
                    "raw_model_response_trace": raw_model_response_trace[i],
                    "parsed_action_trace": parsed_action_trace[i],
                    "parse_status_trace": parse_status_trace[i],
                    "fallback_trace": fallback_trace[i],
                    "causal_update_count_final": final_update_count,
                    "memory_protocol": (
                        "stateful_post_action_v1" if final_update_count > 0 else None
                    ),
                    "method": run_name,
                }
            batch_rows.append(row)
            batch_points.append(float(game_points[i]))
            batch_gcs.append(float(game_gcs[i]))
            batch_steps.append(float(game_steps[i]))
        if batch_rows:
            run_rows.extend(batch_rows)
            res_points.extend(batch_points)
            res_gcs.extend(batch_gcs)
            res_steps.extend(batch_steps)
            _atomic_write_jsonl(run_path, run_rows)
            write_json(
                progress_path,
                {
                    "status": "running",
                    "method": run_name,
                    "split": split,
                    "completed_episodes": len(res_points),
                    "target_episodes": target_episodes,
                },
            )
        episode_no += batch_n
    metrics = {
        "status": "ok",
        "method": run_name,
        "split": split,
        "success_rate": round(float(sum(res_points) / max(1, len(res_points))), 6),
        "average_reward": round(float(sum(res_points) / max(1, len(res_points))), 6),
        "average_goal_condition_points": round(float(sum(res_gcs) / max(1, len(res_gcs))), 6),
        "average_episode_steps": round(float(sum(res_steps) / max(1, len(res_steps))), 6),
        "average_steps": round(float(sum(res_steps) / max(1, len(res_steps))), 6),
        "episode_count": len(res_points),
        "episodes": len(res_points),
        "run_name": run_name,
    }
    _atomic_write_jsonl(run_path, run_rows)
    write_json(output_dir / "metrics.json", metrics)
    if component_scorer is not None:
        write_json(output_dir / "controller_diagnostic.json", _controller_diagnostic(run_rows, metrics))
    return {"status": "ok", "metrics": metrics, "run_path": str(output_dir / "run.jsonl")}


def _controller_diagnostic(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    mode_counter: Counter[str] = Counter()
    top_action_counter: Counter[str] = Counter()
    component_trace_logged = False
    final_minus_policy: list[float] = []
    for row in rows:
        for step_components in row.get("component_score_trace", []) or []:
            if step_components:
                component_trace_logged = True
            for item in step_components:
                mode_counter[str(item.get("controller_mode", "unknown"))] += 1
                top_action_counter[str(item.get("action", ""))] += 1
                final_minus_policy.append(float(item.get("final_score", 0.0)) - float(item.get("policy_score", 0.0)))
    trace = _trace_stats(rows)
    return {
        "status": "ok",
        "method": metrics.get("method"),
        "split": metrics.get("split"),
        "episodes": metrics.get("episodes", metrics.get("episode_count")),
        "success_rate": metrics.get("success_rate"),
        "average_reward": metrics.get("average_reward"),
        "average_goal_condition_points": metrics.get("average_goal_condition_points"),
        "component_trace_logged": component_trace_logged,
        "controller_mode_counts": dict(sorted(mode_counter.items())),
        "component_action_counts": [[action, count] for action, count in top_action_counter.most_common(20)],
        "mean_final_minus_policy_score": round(sum(final_minus_policy) / max(1, len(final_minus_policy)), 6),
        "action_trace_analysis": trace,
        "not_offline_diagnostic": True,
    }


def _qwen_direct_diagnostic(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    parse_counter: Counter[str] = Counter()
    fallback_count = 0
    raw_count = 0
    for row in rows:
        parse_counter.update(str(item) for item in row.get("parse_status_trace", []) if str(item))
        fallback_count += sum(1 for item in row.get("fallback_trace", []) if bool(item))
        raw_count += len(row.get("raw_model_response_trace", []) or [])
    parse_success = raw_count - fallback_count
    return {
        "status": "ok",
        "method": metrics.get("method"),
        "policy_family": "qwen_direct_admissible",
        "uses_clstr": False,
        "episodes": metrics.get("episodes", metrics.get("episode_count")),
        "success_rate": metrics.get("success_rate"),
        "average_reward": metrics.get("average_reward"),
        "parse_status_counts": dict(sorted(parse_counter.items())),
        "parse_success_rate": round(float(parse_success / max(1, raw_count)), 6),
        "fallback_count": int(fallback_count),
        "raw_response_count": int(raw_count),
        "action_trace_analysis": _trace_stats(rows),
        "not_clstr_result": True,
        "not_offline_diagnostic": True,
    }


def evaluate_alfworld_qwen_direct(
    official_repo: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = "Qwen/Qwen3-8B",
    split: str = "valid_seen",
    run_name: str = "qwen3_8b_direct",
    max_episodes: int | None = None,
    max_steps: int = 50,
    batch_size: int = 1,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 16,
    fallback_strategy: str = "look_if_available_else_first",
    scoring_method: str = "generate",
    likelihood_batch_size: int = 4,
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir = Path(data_dir)
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer_cls = QwenDirectLikelihoodActionScorer if scoring_method == "likelihood" else QwenDirectAdmissibleActionScorer
    if scoring_method not in {"generate", "likelihood"}:
        raise ValueError(f"unsupported Qwen direct scoring_method: {scoring_method}")
    scorer = scorer_cls(
        QwenDirectPolicyConfig(
            model_name_or_path=str(model_name_or_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            fallback_strategy=fallback_strategy,
            likelihood_batch_size=likelihood_batch_size,
        )
    )
    config, train_eval_name = _make_env_config(official_repo, data_dir, split)

    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(cfg, train_eval=train_eval_name)

    result = run_alfworld_closed_loop_eval(
        env_factory=env_factory,
        config=config,
        split=split,
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=max_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        run_name=run_name,
    )
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(output_dir / "run.jsonl")
    diagnostic = _qwen_direct_diagnostic(rows, metrics)
    metrics.update(
        {
            "policy_family": "qwen_direct_admissible",
            "uses_clstr": False,
            "is_clstr_result": False,
            "controller_mode": "policy_only",
            "controller_complete_decision": False,
            "model_name_or_path": str(model_name_or_path),
            "inference_backend": "transformers",
            "prompt_template_id": scorer.config.prompt_template_id,
            "action_selection": (
                "direct_loglikelihood_ranking_over_admissible_actions"
                if scoring_method == "likelihood"
                else "direct_generation_over_admissible_actions"
            ),
            "scoring_method": scoring_method,
            "routing_init_manifest": None,
            "checkpoint_path": None,
            "trajectory_trained": False,
            "training_data": "none",
            "training_objective": "none",
            "policy_head_type": "qwen_direct",
            "qdoc_adapter_used": False,
            "native_clstr_heads_used": False,
            "qwen_direct_baseline": True,
            "qwen_direct_generator": scoring_method == "generate",
            "qwen_direct_likelihood_ranker": scoring_method == "likelihood",
            "parse_success_rate": diagnostic["parse_success_rate"],
            "fallback_count": diagnostic["fallback_count"],
            "not_clstr_result": True,
        }
    )
    write_json(metrics_path, metrics)
    diagnostic.update({"metrics": metrics})
    write_json(output_dir / "diagnostic.json", diagnostic)
    return {"status": "ok", "metrics": metrics, "result": result, "diagnostic": diagnostic}


def evaluate_alfworld_skillrouter(
    official_repo: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    split: str = "valid_seen",
    run_name: str = "skillrouter_frozen_admissible",
    max_episodes: int | None = None,
    max_steps: int = 50,
    batch_size: int = 1,
    encode_batch_size: int = 16,
    max_length: int = 2048,
    adapter_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir = Path(data_dir)
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = SkillRouterAdmissibleActionScorer(
        model_name_or_path=model_name_or_path,
        batch_size=encode_batch_size,
        max_length=max_length,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    config, train_eval_name = _make_env_config(official_repo, data_dir, split)

    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(cfg, train_eval=train_eval_name)

    result = run_alfworld_closed_loop_eval(
        env_factory=env_factory,
        config=config,
        split=split,
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=max_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        run_name=run_name,
    )
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "policy_family": "skillrouter_frozen_admissible_action",
            "uses_clstr": False,
            "is_clstr_result": False,
            "model_name_or_path": str(model_name_or_path),
            "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
            "action_selection": "embedding_ranking_over_official_admissible_actions",
            "routing_init_manifest": None,
            "checkpoint_path": None,
            "trajectory_trained": False,
            "training_data": "none",
            "training_objective": "none",
            "policy_head_type": "skillrouter_embedding_similarity",
            "controller_mode": "policy_only",
            "official_repo": str(official_repo),
            "data_dir": str(data_dir),
            "qwen_direct_baseline": False,
            "skillrouter_baseline": True,
            "not_clstr_result": True,
            "encode_batch_size": int(encode_batch_size),
            "max_length": int(max_length),
        }
    )
    write_json(metrics_path, metrics)
    return {"status": "ok", "metrics": metrics, "result": result}


def _load_clstr_alfworld_model(
    routing_init_manifest: str | Path,
    checkpoint_path: str | Path | None,
    output_dir: str | Path,
    data_root: str | Path = "data/aux_trajectories",
    stage4_checkpoint_path: str | Path | None = None,
    skill_rows_path_override: str | Path | None = None,
    stage0_checkpoint_path: str | Path | None = None,
    benchmark_skill_rows_path: str | Path | None = None,
) -> tuple[CLSTRModel, UniversalActionAdapter | None, dict[str, Any]]:
    data_root = Path(data_root)
    checkpoint_payload: dict[str, Any] | None = None
    checkpoint_skill_rows_path: Path | None = None
    override_skill_rows_path = _resolve_existing_path(skill_rows_path_override) if skill_rows_path_override else None
    if stage0_checkpoint_path is not None:
        if checkpoint_path is None or stage4_checkpoint_path is None:
            raise ValueError("final-chain ALFWorld loading requires Stage0, Stage2, and Stage4 checkpoints")
        if override_skill_rows_path is None:
            raise ValueError("final-chain ALFWorld loading requires the exact training skill pool")
        benchmark_skill_rows_path = _resolve_existing_path(benchmark_skill_rows_path)
        if benchmark_skill_rows_path is None:
            raise ValueError("final-chain ALFWorld loading requires benchmark skill rows")
        benchmark_skill_rows = _read_jsonl(benchmark_skill_rows_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_config, skill_id_to_idx, adapter_report = (
            restore_native_benchmark_checkpoint_chain(
                stage0_checkpoint_path=stage0_checkpoint_path,
                stage2_checkpoint_path=checkpoint_path,
                stage4_checkpoint_path=stage4_checkpoint_path,
                training_skills_path=override_skill_rows_path,
                benchmark_skills=benchmark_skill_rows,
                model_cache_dir=Path(output_dir),
                device=device,
                require_safe_memory_delta=True,
            )
        )
        return model, None, {
            "routing_init_manifest": str(routing_init_manifest),
            "skill_source_path": str(override_skill_rows_path),
            "skill_source_override": str(override_skill_rows_path),
            "skill_source_from_checkpoint": False,
            "benchmark_skill_source_path": str(benchmark_skill_rows_path),
            "benchmark_skill_count": len(benchmark_skill_rows),
            "skill_count": len(skill_id_to_idx),
            "model_config": model_config,
            "final_chain_adapter": adapter_report,
            "policy_checkpoint": adapter_report.get("stage2") or {},
            "stage4_checkpoint": adapter_report.get("stage4") or {},
            "checkpoint_skill_table_embeddings_loaded": True,
            "skill_table_rebuilt_after_load": False,
        }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_routing_init = (
            checkpoint_payload.get("routing_init")
            if isinstance(checkpoint_payload.get("routing_init"), dict)
            else {}
        )
        checkpoint_policy = (
            checkpoint_routing_init.get("policy_checkpoint")
            if isinstance(checkpoint_routing_init.get("policy_checkpoint"), dict)
            else {}
        )
        checkpoint_skill_rows_path = _resolve_existing_path(
            checkpoint_payload.get("skills_path")
            or checkpoint_payload.get("resolved_skills_path")
            or checkpoint_payload.get("skill_source_path")
            or checkpoint_routing_init.get("skill_source_path")
            or checkpoint_routing_init.get("resolved_skills_path")
            or checkpoint_policy.get("resolved_skills_path")
            or checkpoint_policy.get("skills_path")
            or checkpoint_policy.get("skill_source_path"),
            checkpoint_path.parent,
        )

    if override_skill_rows_path is not None:
        skill_rows_path = override_skill_rows_path
        skill_rows = _read_jsonl(skill_rows_path)
    elif checkpoint_skill_rows_path is not None:
        skill_rows_path = checkpoint_skill_rows_path
        skill_rows = _read_jsonl(skill_rows_path)
    else:
        skill_rows_path = data_root / "pseudo_skills.jsonl"
        if skill_rows_path.exists():
            skill_rows = _read_jsonl(skill_rows_path)
        else:
            skill_rows_path = data_root / "skills.jsonl"
            if not skill_rows_path.exists():
                skill_rows = load_aux_training_rows(data_root)[0]
                skill_rows_path = data_root / "pseudo_skills.jsonl"
            else:
                skill_rows = _read_jsonl(skill_rows_path)
    if checkpoint_payload is not None and bool(checkpoint_payload.get("qwen_external_encoder", False)):
        qwen_model_path = (
            checkpoint_payload.get("qwen_model_name_or_path")
            or (checkpoint_payload.get("config") or {}).get("base_model_name")
            or "Qwen/Qwen3-8B"
        )
        model, model_config, routing_report = build_qwen_external_clstr_model(
            skill_rows,
            model_name_or_path=str(qwen_model_path),
            model_dim=int((checkpoint_payload.get("config") or {}).get("d", checkpoint_payload.get("encoder_output_dim", 4096))),
            max_length=(checkpoint_payload.get("config") or {}).get("max_length", 4096),
            torch_dtype=str((checkpoint_payload.get("config") or {}).get("torch_dtype", "bfloat16")),
            cache_dir=(checkpoint_payload.get("config") or {}).get("hf_cache_dir"),
            local_files_only=bool(checkpoint_payload.get("qwen_local_files_only", False)),
        )
    elif checkpoint_payload is not None and isinstance(checkpoint_payload.get("config"), dict):
        model, model_config, routing_report = _build_model_from_checkpoint_config(
            checkpoint_payload["config"],
            skill_rows,
            Path(output_dir),
        )
    else:
        model, model_config, routing_report = _build_model_from_routing_init(
            routing_init_manifest,
            skill_rows,
            Path(output_dir),
        )
    routing_report["skill_source_path"] = str(skill_rows_path)
    routing_report["skill_source_from_checkpoint"] = checkpoint_skill_rows_path is not None
    routing_report["skill_source_override"] = str(override_skill_rows_path) if override_skill_rows_path is not None else None
    routing_report["skill_count"] = len(skill_rows)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    action_adapter = None
    checkpoint_report: dict[str, Any] = {}
    checkpoint_skill_table_embeddings_loaded = False
    if checkpoint_path is not None:
        payload = checkpoint_payload or torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        if hasattr(model, "state_dict"):
            state_dict = _filter_state_dict(model, state_dict)
        checkpoint_skill_table_embeddings_loaded = "skill_table.E" in state_dict
        incompatible = model.load_state_dict(state_dict, strict=False)
        adapter_state = payload.get("universal_action_adapter_state_dict")
        if adapter_state is not None:
            action_adapter = UniversalActionAdapter(int(model_config["d"]), hidden_dim=int(model_config["d"]))
            action_adapter.load_state_dict(adapter_state)
            action_adapter.to(model.device)
            action_adapter.eval()
        checkpoint_report = {
            "checkpoint_path": str(checkpoint_path),
            "stage": payload.get("stage"),
            "training_objective": payload.get("training_objective"),
            "training_data": payload.get("training_data"),
            "policy_head_type": payload.get("policy_head_type"),
            "native_clstr_heads_used": bool(payload.get("native_clstr_heads_used", False)),
            "clstr_native_act_trained": bool(payload.get("clstr_native_act_trained", False)),
            "legacy_universal_action_adapter_role": payload.get("legacy_universal_action_adapter_role"),
            "qdoc_adapter_used": bool(payload.get("qdoc_adapter_used", False)),
            "qwen_external_encoder": bool(payload.get("qwen_external_encoder", False)),
            "qwen_model_name_or_path": payload.get("qwen_model_name_or_path"),
            "qwen_frozen": bool(payload.get("qwen_frozen", False)),
            "qwen_quantization_mode": payload.get("qwen_quantization_mode"),
            "qwen_direct_generator": bool(payload.get("qwen_direct_generator", False)),
            "frozen_routing_foundation": bool(payload.get("frozen_routing_foundation", False)),
            "has_universal_action_adapter": adapter_state is not None,
            "skills_path": payload.get("skills_path"),
            "resolved_skills_path": str(checkpoint_skill_rows_path) if checkpoint_skill_rows_path is not None else None,
            "skill_rows_path_override": str(override_skill_rows_path) if override_skill_rows_path is not None else None,
            "checkpoint_skill_table_embeddings_loaded": checkpoint_skill_table_embeddings_loaded,
            "loaded_key_count": len(state_dict),
            "missing_keys": list(getattr(incompatible, "missing_keys", []) or []),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", []) or []),
        }
        routing_report["policy_checkpoint"] = checkpoint_report
    if stage4_checkpoint_path is not None:
        stage4_checkpoint_path = Path(stage4_checkpoint_path)
        stage4_payload = torch.load(stage4_checkpoint_path, map_location="cpu")
        if not isinstance(stage4_payload, dict):
            raise ValueError(f"Stage4 checkpoint must be a dict: {stage4_checkpoint_path}")
        stage4_state_dict = stage4_payload.get("model_state_dict", stage4_payload)
        if hasattr(model, "state_dict"):
            stage4_state_dict = _filter_state_dict(model, stage4_state_dict)
        if "skill_table.E" in stage4_state_dict:
            checkpoint_skill_table_embeddings_loaded = True
        incompatible = model.load_state_dict(stage4_state_dict, strict=False)
        stage4_adapter_state = stage4_payload.get("universal_action_adapter_state_dict")
        if stage4_adapter_state is not None:
            if action_adapter is None:
                action_adapter = UniversalActionAdapter(int(model_config["d"]), hidden_dim=int(model_config["d"]))
            action_adapter.load_state_dict(stage4_adapter_state)
            action_adapter.to(model.device)
            action_adapter.eval()
        routing_report["stage4_checkpoint"] = {
            "checkpoint_path": str(stage4_checkpoint_path),
            "stage": stage4_payload.get("stage"),
            "training_objective": stage4_payload.get("training_objective"),
            "training_data": stage4_payload.get("training_data"),
            "loaded_key_count": len(stage4_state_dict),
            "missing_keys": list(getattr(incompatible, "missing_keys", []) or []),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", []) or []),
            "has_universal_action_adapter": stage4_adapter_state is not None,
        }
    if (
        bool(getattr(getattr(model, "config", None), "defer_skill_table_init", False))
        and hasattr(model, "rebuild_skill_table")
        and not checkpoint_skill_table_embeddings_loaded
    ):
        model.rebuild_skill_table()
        routing_report["checkpoint_skill_table_embeddings_loaded"] = False
        routing_report["skill_table_rebuilt_after_load"] = True
    else:
        routing_report["checkpoint_skill_table_embeddings_loaded"] = bool(checkpoint_skill_table_embeddings_loaded)
        routing_report["skill_table_rebuilt_after_load"] = False
    model.eval()
    return model, action_adapter, routing_report


def evaluate_alfworld_clstr(
    official_repo: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    routing_init_manifest: str | Path,
    checkpoint_path: str | Path | None,
    split: str,
    run_name: str,
    stage4_checkpoint_path: str | Path | None = None,
    skill_rows_path_override: str | Path | None = None,
    stage0_checkpoint_path: str | Path | None = None,
    benchmark_skill_rows_path: str | Path | None = None,
    scorer_mode: str = "legacy_concrete_action_head",
    replay_prefix_max_steps: int = 6,
    max_episodes: int | None = None,
    max_steps: int = 50,
    batch_size: int = 1,
    aux_data_root: str | Path = "data/aux_trajectories",
    controller_mode: str = "policy_only",
    include_available_actions_in_state: bool = False,
    q_success_weight: float = 0.0,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    safe_memory_residual_bound: float = 2.0,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    expected_memory_utility_gate_checkpoint_sha256: str | None = None,
    expected_memory_utility_gate_audit_sha256: str | None = None,
    memory_protocol: str = "stateful_post_action_v1",
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir = Path(data_dir)
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    output_dir = Path(output_dir)
    loader_kwargs: dict[str, Any] = {
        "routing_init_manifest": routing_init_manifest,
        "checkpoint_path": checkpoint_path,
        "stage4_checkpoint_path": stage4_checkpoint_path,
        "skill_rows_path_override": skill_rows_path_override,
        "output_dir": output_dir / "model_cache",
        "data_root": aux_data_root,
    }
    if stage0_checkpoint_path is not None:
        loader_kwargs["stage0_checkpoint_path"] = stage0_checkpoint_path
    if benchmark_skill_rows_path is not None:
        loader_kwargs["benchmark_skill_rows_path"] = benchmark_skill_rows_path
    model, action_adapter, routing_report = _load_clstr_alfworld_model(**loader_kwargs)
    if scorer_mode == "unified_memory_admissible_action" and memory_protocol != "stateful_post_action_v1":
        raise ValueError("final ALFWorld unified memory requires stateful_post_action_v1")
    memory_utility_gate, reliability_gate_report = resolve_reliability_gate(
        reliability_mode=reliability_mode,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
        expected_gate_sha256=expected_memory_utility_gate_checkpoint_sha256,
        expected_audit_sha256=expected_memory_utility_gate_audit_sha256,
        device=torch.device(getattr(model, "device", "cpu")),
    )
    if reliability_mode == "learned":
        feature_update_count_cap = float(
            reliability_gate_report["feature_update_count_cap"]
        )
        feature_candidate_count_cap = float(
            reliability_gate_report["feature_candidate_count_cap"]
        )
    scorer = make_alfworld_clstr_candidate_scorer(
        model,
        action_adapter,
        scorer_mode=scorer_mode,
        include_available_actions_in_state=include_available_actions_in_state,
        replay_prefix_max_steps=replay_prefix_max_steps,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        safe_memory_residual_bound=safe_memory_residual_bound,
        memory_utility_gate=memory_utility_gate,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
    )
    controller_config = ClosedLoopControllerConfig(mode=controller_mode, q_success_weight=float(q_success_weight))
    component_scorer = None
    if controller_mode != "policy_only":
        component_scorer = make_controller_component_scorer(model)
    transition_supported = bool(getattr(component_scorer, "supports_transition_scores", component_scorer is not None))
    belief_supported = bool(getattr(component_scorer, "supports_belief_scores", component_scorer is not None))
    stop_supported = bool(getattr(component_scorer, "supports_stop_scores", component_scorer is not None))
    config, train_eval_name = _make_env_config(official_repo, data_dir, split)

    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(cfg, train_eval=train_eval_name)

    result = run_alfworld_closed_loop_eval(
        env_factory=env_factory,
        config=config,
        split=split,
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=max_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        run_name=run_name,
        component_scorer=component_scorer,
        controller_config=controller_config,
    )
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_path = output_dir / "run.jsonl"
    run_rows = _read_jsonl(run_path) if run_path.exists() else []
    causal_update_count = sum(
        int(row.get("causal_update_count_final") or 0) for row in run_rows
    )
    metrics.update(
        {
            "official_repo": str(official_repo),
            "data_dir": str(data_dir),
            "routing_init_manifest": str(routing_init_manifest),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "stage4_checkpoint_path": str(stage4_checkpoint_path) if stage4_checkpoint_path is not None else None,
            "stage0_checkpoint_path": str(stage0_checkpoint_path) if stage0_checkpoint_path is not None else None,
            "skill_rows_path_override": str(skill_rows_path_override) if skill_rows_path_override is not None else None,
            "benchmark_skill_rows_path": str(benchmark_skill_rows_path) if benchmark_skill_rows_path is not None else None,
            "candidate_scorer_mode": str(scorer_mode),
            "replay_prefix_max_steps": int(replay_prefix_max_steps),
            "memory_active_protocol": scorer_mode == "unified_memory_admissible_action",
            "memory_protocol": memory_protocol if scorer_mode == "unified_memory_admissible_action" else None,
            "causal_update_count": int(causal_update_count),
            "uses_recurrent_m_t": bool(causal_update_count > 0),
            "memory_utility_reliability_mode": reliability_mode,
            "memory_utility_fixed_alpha": float(fixed_alpha),
            "safe_memory_residual_bound": float(safe_memory_residual_bound),
            "feature_update_count_cap": float(feature_update_count_cap),
            "feature_candidate_count_cap": float(feature_candidate_count_cap),
            "memory_utility_gate_checkpoint_path": reliability_gate_report.get("checkpoint_path"),
            "memory_utility_gate_checkpoint_sha256": reliability_gate_report.get("checkpoint_sha256"),
            "memory_utility_gate_audit_sha256": reliability_gate_report.get("audit_manifest_sha256"),
            "stage4_overlay_loaded": bool(routing_report.get("stage4_checkpoint")),
            "stage4_stage": routing_report.get("stage4_checkpoint", {}).get("stage"),
            "trajectory_trained": checkpoint_path is not None,
            "training_data": (
                routing_report.get("policy_checkpoint", {}).get("training_data")
                if checkpoint_path is not None
                else "none"
            ),
            "training_objective": routing_report.get("policy_checkpoint", {}).get("training_objective"),
            "policy_head_type": routing_report.get("policy_checkpoint", {}).get("policy_head_type")
            or ("native_skill_head" if getattr(model, "skill_head", None) is not None else "cosine"),
            "native_clstr_heads_used": bool(routing_report.get("policy_checkpoint", {}).get("native_clstr_heads_used", False)),
            "clstr_native_act": bool(routing_report.get("policy_checkpoint", {}).get("clstr_native_act_trained", False)),
            "clstr_native_act_trained": bool(routing_report.get("policy_checkpoint", {}).get("clstr_native_act_trained", False)),
            "legacy_universal_action_adapter_role": routing_report.get("policy_checkpoint", {}).get("legacy_universal_action_adapter_role"),
            "qdoc_adapter_used": bool(routing_report.get("policy_checkpoint", {}).get("qdoc_adapter_used", False)),
            "qwen_external_encoder": bool(routing_report.get("policy_checkpoint", {}).get("qwen_external_encoder", False)),
            "qwen_model_name_or_path": routing_report.get("policy_checkpoint", {}).get("qwen_model_name_or_path"),
            "qwen_frozen": bool(routing_report.get("policy_checkpoint", {}).get("qwen_frozen", False)),
            "qwen_quantization_mode": routing_report.get("policy_checkpoint", {}).get("qwen_quantization_mode"),
            "qwen_direct_generator": bool(routing_report.get("policy_checkpoint", {}).get("qwen_direct_generator", False)),
            "aux_data_root": str(aux_data_root),
            "skill_source_path": routing_report.get("skill_source_path"),
            "skill_count": routing_report.get("skill_count"),
            "controller_mode": controller_mode,
            "qwen_planner_available_actions": bool(include_available_actions_in_state),
            "planner_intent_mode": "latent_available_actions_query" if include_available_actions_in_state else "none",
            "qwen_direct_generator_in_clstr": False,
            "controller_complete_decision": controller_mode != "policy_only",
            "transition_belief_stop_participate_in_action_selection": controller_mode
            in {
                "policy_plus_transition_belief_stop",
                "policy_plus_transition_belief_stop_loop_penalty",
            }
            and transition_supported
            and belief_supported
            and stop_supported,
            "transition_participates_in_action_selection": controller_mode
            in {
                "policy_plus_transition",
                "policy_plus_transition_belief",
                "policy_plus_transition_belief_stop",
                "policy_plus_transition_belief_stop_loop_penalty",
            }
            and transition_supported,
            "belief_participates_in_action_selection": controller_mode
            in {
                "policy_plus_transition_belief",
                "policy_plus_transition_belief_stop",
                "policy_plus_transition_belief_stop_loop_penalty",
            }
            and belief_supported,
            "stop_participates_in_action_selection": controller_mode
            in {
                "policy_plus_transition_belief_stop",
                "policy_plus_transition_belief_stop_loop_penalty",
            }
            and stop_supported,
            "loop_penalty_participates_in_action_selection": controller_mode
            == "policy_plus_transition_belief_stop_loop_penalty",
            "q_success_weight": float(q_success_weight),
            "q_success_participates_in_action_selection": bool(float(q_success_weight) != 0.0 and component_scorer is not None),
            "routing_init_report": routing_report,
        }
    )
    write_json(metrics_path, metrics)
    return {"status": "ok", "metrics": metrics, "result": result}


def build_alfworld_comparison_table(
    eval_root: str | Path,
    run_names: list[str],
    output_table_path: str | Path,
    output_summary_path: str | Path,
) -> dict[str, Any]:
    eval_root = Path(eval_root)
    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        metrics_path = eval_root / run_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training_data = metrics.get("training_data")
        l_policy_source = metrics.get("training_objective")
        if training_data is None:
            if "aux" in run_name:
                training_data = "auxiliary trajectory action-pool pretraining"
            elif metrics.get("trajectory_trained", False):
                training_data = "unknown"
            else:
                training_data = "none"
        if l_policy_source is None:
            if "aux" in run_name:
                l_policy_source = "auxiliary_action_pool_proxy_not_full_L_policy"
            elif metrics.get("trajectory_trained", False):
                l_policy_source = "unknown"
            else:
                l_policy_source = "none"
        rows.append(
            {
                "method": metrics.get("method", run_name),
                "training data": training_data,
                "trajectory_trained": bool(metrics.get("trajectory_trained", False)),
                "L_policy source": l_policy_source,
                "success_rate": metrics.get("success_rate", 0.0),
                "average_reward": metrics.get("average_reward", 0.0),
                "average_goal_condition_points": metrics.get("average_goal_condition_points", 0.0),
                "average_episode_steps": metrics.get("average_episode_steps", 0.0),
                "episodes": metrics.get("episodes", metrics.get("episode_count", 0)),
                "metrics_note": metrics.get("caveat", metrics.get("note", "")),
            }
        )
    header = [
        "| method | training data | trajectory/policy trained | L_policy source | success_rate | average_reward | average_goal_condition_points | average_episode_steps | episodes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines = header[:]
    for row in rows:
        training_data = str(row["training data"]).replace("|", "\\|")
        l_policy_source = str(row["L_policy source"]).replace("|", "\\|")
        lines.append(
            f"| {row['method']} | {training_data} | {row['trajectory_trained']} | {l_policy_source} | "
            f"{float(row['success_rate']):.4f} | {float(row['average_reward']):.4f} | "
            f"{float(row['average_goal_condition_points']):.4f} | "
            f"{float(row['average_episode_steps']):.2f} | {int(row['episodes'])} |"
        )
    caveat = (
        "ALFWorld is a closed-loop benchmark; baseline is CLSTR routing init without trajectory training; "
        "SKILLRET is used only for routing initialization/static retrieval, not ALFWorld training data; "
        "ALFWorld train replay is used for supervised policy pretraining, not RL fine-tuning; "
        "valid_seen/valid_unseen are used only for eval, not training."
    )
    lines.extend(
        [
            "",
            "Notes:",
            "- ALFWorld is a closed-loop benchmark.",
            "- Baseline is CLSTR routing init without trajectory training.",
            "- SKILLRET is used only for routing initialization/static retrieval, not ALFWorld training data.",
            "- ALFWorld train replay is used for supervised policy pretraining, not RL fine-tuning.",
            "- valid_seen/valid_unseen are used only for eval, not training.",
        ]
    )
    output_table_path = Path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "ok",
        "row_count": len(rows),
        "rows": rows,
        "caveat": caveat,
    }
    write_json(Path(output_summary_path), summary)
    return summary


def _load_run_rows_for_method(eval_root: Path, method: str) -> list[dict[str, Any]]:
    method_dir = eval_root / method
    split_paths = sorted(path for path in method_dir.glob("*/run.jsonl") if path.is_file())
    paths = split_paths or ([method_dir / "run.jsonl"] if (method_dir / "run.jsonl").exists() else [])
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _has_consecutive_repeat(actions: list[str], threshold: int = 5) -> bool:
    if not actions:
        return False
    run = 1
    previous = actions[0]
    for action in actions[1:]:
        if action == previous:
            run += 1
            if run >= threshold:
                return True
        else:
            previous = action
            run = 1
    return False


def _has_tail_two_action_cycle(actions: list[str], tail_len: int = 10) -> bool:
    tail = actions[-tail_len:]
    if len(tail) < 6:
        return False
    a, b = tail[-2], tail[-1]
    if a == b:
        return False
    return all(action == (a if idx % 2 == 0 else b) for idx, action in enumerate(tail[-6:]))


def _trace_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counter: Counter[str] = Counter()
    verb_counter: Counter[str] = Counter()
    consecutive_repeat = 0
    single_action_only = 0
    tail_cycle = 0
    for row in rows:
        actions = [str(action) for action in row.get("action_trace", [])]
        action_counter.update(actions)
        verb_counter.update(action.split(maxsplit=1)[0] if action else "" for action in actions)
        if _has_consecutive_repeat(actions):
            consecutive_repeat += 1
        if actions and len(set(actions)) == 1:
            single_action_only += 1
        if _has_tail_two_action_cycle(actions):
            tail_cycle += 1
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    success_count = sum(1 for row in rows if bool(row.get("success", False)))
    points_sum = sum(float(row.get("points", 0.0) or 0.0) for row in rows)
    gcp_sum = sum(float(row.get("goal_condition_points", 0.0) or 0.0) for row in rows)
    steps = [float(row.get("steps", 0.0) or 0.0) for row in rows]
    return {
        "episodes": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "success_count": success_count,
        "success_rate": round(float(success_count / max(1, len(rows))), 6),
        "points_sum": round(points_sum, 6),
        "goal_condition_points_sum": round(gcp_sum, 6),
        "all_episodes_timeout_50_steps": bool(rows) and all(step >= 50.0 for step in steps),
        "top_actions": [[action, count] for action, count in action_counter.most_common(10)],
        "top_verbs": [[verb, count] for verb, count in verb_counter.most_common(10)],
        "stuck_patterns": {
            "consecutive_repeat_ge_5_episodes": consecutive_repeat,
            "single_action_only_episodes": single_action_only,
            "tail_two_action_cycle_episodes": tail_cycle,
        },
    }


def build_alfworld_policy_diagnostic_report(
    eval_root: str | Path,
    output_path: str | Path,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    eval_root = Path(eval_root)
    methods = methods or ["clstr_routing_init_baseline", "clstr_aux_full_obs_cosine"]
    method_reports: dict[str, Any] = {}
    field_union: set[str] = set()
    for method in methods:
        rows = _load_run_rows_for_method(eval_root, method)
        for row in rows:
            field_union.update(row)
        method_reports[method] = _trace_stats(rows)
    available_fields = {
        "admissible_commands": "admissible_commands" in field_union,
        "scores": "scores" in field_union or "score" in field_union,
        "chosen_action": "chosen_action" in field_union,
        "reward": "reward" in field_union,
        "done": "done" in field_union,
        "action_trace": "action_trace" in field_union,
        "points": "points" in field_union,
        "goal_condition_points": "goal_condition_points" in field_union,
        "success": "success" in field_union,
    }
    report = {
        "status": "ok",
        "current_policy": {
            "uses_action_scorer_only": True,
            "action_selection": "argmax over candidate_scorer(state_text, admissible_commands_t)",
            "transition_participates_in_action_selection": False,
            "belief_participates_in_action_selection": False,
            "stop_participates_in_action_selection": False,
            "transition_belief_stop_participate_in_action_selection": False,
        },
        "action_trace_diagnostic": {
            "eval_root": str(eval_root),
            "methods": method_reports,
            "available_trace_fields": available_fields,
            "counting_policy": "split run.jsonl files are preferred over aggregate root run.jsonl to avoid double counting",
        },
        "auxiliary_action_pool_ce_vs_l_policy": {
            "conclusion": "auxiliary action pool CE cannot be equivalent to complete ALFWorld L_policy",
            "reason": (
                "the old auxiliary objective ranks positives against sampled historical action-pool negatives, "
                "while ALFWorld closed-loop policy must choose the expert action from the exact "
                "admissible_commands_t distribution produced by the environment at each state"
            ),
            "old_auxiliary_candidate_source": "sampled action pool negatives",
            "required_l_policy_candidate_source": "official ALFWorld admissible_commands_t",
        },
        "replay_extraction_necessity": {
            "required": True,
            "reason": (
                "training data must align state_t, goal, action history, admissible_commands_t, and expert_action_t "
                "from official ALFWorld train episodes; valid_seen/valid_unseen remain eval-only"
            ),
        },
        "candidate_recall": environment_candidate_recall_metadata(),
    }
    write_json(Path(output_path), report)
    return report
