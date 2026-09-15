from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.action_adapter import UniversalActionAdapter
from clstr.aux_trajectories import load_aux_training_rows
from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.native_rerank import CLSTRNativeResidualRerankHead, _config_from_checkpoint, _filter_state_dict
from clstr.transition_utils import transition_action_input


DEFAULT_INCLUDE_SPLITS = ("train", "unknown", "valid_seen", "valid_unseen")
DEFAULT_EXCLUDE_SPLITS = ("test",)
RERANK_SIDEcar_REASON = "auxiliary pretrain optimizes L_trans/L_act_proxy/STOP proxy over pseudo-skills"
RETENTION_BASELINE = {"NDCG@10": 0.70584, "Recall@10": 0.75669, "MAP@10": 0.63390}
RETENTION_THRESHOLDS = {"NDCG@10": 0.005, "Recall@10": 0.005, "MAP@10": 0.005}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_existing_path(path_value: str | Path, anchor: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    if anchor is not None:
        anchored = anchor / path
        if anchored.exists():
            return anchored
    return path


def _flatten_step_rows(
    trajectories: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    max_trajectories: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = trajectories[:max_trajectories] if max_trajectories is not None else trajectories
    for trajectory in selected:
        steps = trajectory.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            pseudo_skill_id = str(step.get("pseudo_skill_id") or "")
            if pseudo_skill_id not in skill_id_to_idx:
                continue
            next_pseudo_skill_id = None
            next_label = None
            if step_idx + 1 < len(steps):
                next_step = steps[step_idx + 1]
                if isinstance(next_step, dict):
                    next_pseudo_skill_id = str(next_step.get("pseudo_skill_id") or "")
                    if next_pseudo_skill_id in skill_id_to_idx:
                        next_label = skill_id_to_idx[next_pseudo_skill_id]
            observation = str(step.get("observation_text") or "")
            action = str(step.get("action_text") or "")
            state_text = "\n".join(
                [
                    f"environment:{trajectory.get('environment', '')}",
                    f"task_type:{trajectory.get('task_type', '')}",
                    f"observation:{observation}",
                    f"action_prefix:{action[:128]}",
                ]
            )
            rows.append(
                {
                    "row_index": len(rows),
                    "state_text": state_text,
                    "action_text": action,
                    "label": skill_id_to_idx[pseudo_skill_id],
                    "pseudo_skill_id": pseudo_skill_id,
                    "done": bool(step.get("done")) if step.get("done") is not None else False,
                    "next_observation_text": step.get("next_observation_text"),
                    "reward": step.get("reward"),
                    "split": trajectory.get("split", "unknown"),
                    "environment": trajectory.get("environment", "unknown"),
                    "task_id": trajectory.get("task_id"),
                    "step_index": step_idx,
                    "next_pseudo_skill_id": next_pseudo_skill_id,
                    "next_label": next_label,
                }
            )
    return rows


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _build_action_pools(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_env: dict[str, list[str]] = defaultdict(list)
    by_env_skill: dict[tuple[str, str], list[str]] = defaultdict(list)
    global_actions: list[str] = []
    for row in rows:
        action = str(row.get("action_text") or "").strip()
        if not action:
            continue
        env = str(row.get("environment", "unknown"))
        pseudo_skill = str(row.get("pseudo_skill_id", "unknown"))
        by_env[env].append(action)
        by_env_skill[(env, pseudo_skill)].append(action)
        global_actions.append(action)
    return {
        "by_env": {key: _unique_texts(values) for key, values in by_env.items()},
        "by_env_skill": {key: _unique_texts(values) for key, values in by_env_skill.items()},
        "global": _unique_texts(global_actions),
    }


def _pool_action_texts(action_pools: dict[str, Any]) -> list[str]:
    return _unique_texts(
        [
            *sum((list(values) for values in action_pools.get("by_env", {}).values()), []),
            *sum((list(values) for values in action_pools.get("by_env_skill", {}).values()), []),
            *list(action_pools.get("global", [])),
        ]
    )


def _encode_text_batches(
    encoder: Any,
    texts: list[str],
    batch_size: int,
    label: str = "texts",
    stats: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0, dtype=torch.float32)
    batch_size = max(1, int(batch_size))
    current_batch_size = batch_size
    encoded_batches: list[torch.Tensor] = []
    start = 0
    batch_no = 0
    while start < len(texts):
        end = min(start + current_batch_size, len(texts))
        batch = texts[start:end]
        try:
            with torch.inference_mode():
                encoded = encoder.encode_observations(batch)
        except torch.OutOfMemoryError:
            if current_batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_batch_size = max(1, current_batch_size // 2)
            if stats is not None:
                stats.setdefault("oom_backoffs", []).append(
                    {
                        "label": label,
                        "start": start,
                        "attempted_batch_size": current_batch_size,
                        "next_batch_size": new_batch_size,
                    }
                )
            print(
                f"[aux_pretrain] OOM while caching {label}; reducing batch size from {current_batch_size} to {new_batch_size}",
                file=sys.stderr,
                flush=True,
            )
            current_batch_size = new_batch_size
            continue
        encoded_batches.append(encoded.detach().cpu().float())
        start = end
        batch_no += 1
        if len(texts) >= 10000 and (batch_no == 1 or batch_no % 100 == 0):
            print(
                f"[aux_pretrain] cached {min(start, len(texts))}/{len(texts)} {label}",
                file=sys.stderr,
                flush=True,
            )
    if stats is not None:
        stats.setdefault("effective_batch_sizes", {})[label] = current_batch_size
        stats.setdefault("encoded_items", {})[label] = len(texts)
        stats["oom_backoff_count"] = len(stats.get("oom_backoffs", []))
    return torch.cat(encoded_batches, dim=0)


@dataclass
class FrozenAuxiliaryFeatureCache:
    state_embeddings: torch.Tensor
    action_embeddings: torch.Tensor
    next_observation_embeddings: torch.Tensor
    row_action_indices: torch.Tensor
    row_candidate_indices: torch.Tensor
    row_candidate_mask: torch.Tensor
    action_text_to_index: dict[str, int]
    action_texts: list[str]
    cache_batch_size: int
    cache_stats: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "used": True,
            "state_rows": int(self.state_embeddings.size(0)),
            "action_vocab_size": int(self.action_embeddings.size(0)),
            "candidate_rows": int(self.row_candidate_indices.size(0)),
            "candidate_width": int(self.row_candidate_indices.size(1)) if self.row_candidate_indices.ndim == 2 else 0,
            "cache_batch_size": int(self.cache_batch_size),
            "oom_backoff_count": len(self.cache_stats.get("oom_backoffs", [])),
        }


def _build_frozen_auxiliary_feature_cache(
    encoder: Any,
    rows: list[dict[str, Any]],
    action_pools: dict[str, Any],
    action_negative_k: int,
    candidate_strategy: str,
    encode_batch_size: int = 256,
) -> FrozenAuxiliaryFeatureCache:
    row_action_texts = [
        str(row.get("action_text") or "").strip() or "<empty action>"
        for row in rows
    ]
    action_texts = _unique_texts([*_pool_action_texts(action_pools), *row_action_texts])
    action_text_to_index = {text: idx for idx, text in enumerate(action_texts)}
    cache_stats: dict[str, Any] = {}
    state_embeddings = _encode_text_batches(
        encoder,
        [row["state_text"] for row in rows],
        encode_batch_size,
        "state embeddings",
        cache_stats,
    )
    action_embeddings = _encode_text_batches(
        encoder,
        action_texts,
        encode_batch_size,
        "action embeddings",
        cache_stats,
    )
    next_obs_texts = [str(row["next_observation_text"]) if row.get("next_observation_text") is not None else "" for row in rows]
    next_observation_embeddings = torch.zeros_like(state_embeddings)
    next_obs_indices = [idx for idx, row in enumerate(rows) if row.get("next_observation_text") is not None]
    if next_obs_indices:
        encoded_next_obs = _encode_text_batches(
            encoder,
            [next_obs_texts[idx] for idx in next_obs_indices],
            encode_batch_size,
            "next-observation embeddings",
            cache_stats,
        )
        next_observation_embeddings.index_copy_(
            0,
            torch.tensor(next_obs_indices, dtype=torch.long),
            encoded_next_obs.to(next_observation_embeddings.dtype),
        )
    row_action_indices = torch.tensor(
        [action_text_to_index[str(row.get("action_text") or "").strip() or "<empty action>"] for row in rows],
        dtype=torch.long,
    )
    candidate_indices_rows: list[list[int]] = []
    candidate_mask_rows: list[list[bool]] = []
    for row in rows:
        candidate_texts = _candidate_actions_for_row(row, action_pools, action_negative_k, candidate_strategy)
        candidate_indices = [action_text_to_index[text] for text in candidate_texts]
        width = action_negative_k + 1
        if len(candidate_indices) < width:
            pad_value = candidate_indices[-1]
            candidate_mask = [True] * len(candidate_indices) + [False] * (width - len(candidate_indices))
            candidate_indices = candidate_indices + [pad_value] * (width - len(candidate_indices))
        else:
            candidate_mask = [True] * width
            candidate_indices = candidate_indices[:width]
        candidate_indices_rows.append(candidate_indices)
        candidate_mask_rows.append(candidate_mask)
    row_candidate_indices = torch.tensor(candidate_indices_rows, dtype=torch.long)
    row_candidate_mask = torch.tensor(candidate_mask_rows, dtype=torch.bool)
    return FrozenAuxiliaryFeatureCache(
        state_embeddings=state_embeddings,
        action_embeddings=action_embeddings,
        next_observation_embeddings=next_observation_embeddings,
        row_action_indices=row_action_indices,
        row_candidate_indices=row_candidate_indices,
        row_candidate_mask=row_candidate_mask,
        action_text_to_index=action_text_to_index,
        action_texts=action_texts,
        cache_batch_size=encode_batch_size,
        cache_stats=cache_stats,
    )


def _candidate_actions_for_row(
    row: dict[str, Any],
    pools: dict[str, Any],
    negative_k: int,
    candidate_strategy: str,
) -> list[str]:
    positive = str(row.get("action_text") or "").strip()
    if not positive:
        positive = "<empty action>"
    env = str(row.get("environment", "unknown"))
    pseudo_skill = str(row.get("pseudo_skill_id", "unknown"))
    sources: list[list[str]]
    if candidate_strategy == "environment_action_pool":
        sources = [
            pools.get("by_env_skill", {}).get((env, pseudo_skill), []),
            pools.get("by_env", {}).get(env, []),
            pools.get("global", []),
        ]
    elif candidate_strategy == "global_action_pool":
        sources = [pools.get("global", [])]
    else:
        raise ValueError(f"unsupported candidate_strategy: {candidate_strategy}")
    selected: list[str] = []
    seen = {positive}
    for source in sources:
        for candidate in source:
            text = str(candidate or "").strip()
            if not text or text in seen:
                continue
            selected.append(text)
            seen.add(text)
            if len(selected) >= negative_k:
                break
        if len(selected) >= negative_k:
            break
    if selected and len(selected) < negative_k:
        cursor = 0
        seed = list(selected)
        while len(selected) < negative_k:
            selected.append(seed[cursor % len(seed)])
            cursor += 1
    return [positive] + selected


def _build_candidate_action_batch(
    batch: list[dict[str, Any]],
    pools: dict[str, Any],
    negative_k: int,
    candidate_strategy: str,
) -> tuple[list[list[str]], torch.Tensor]:
    candidates = [
        _candidate_actions_for_row(row, pools, negative_k, candidate_strategy)
        for row in batch
    ]
    max_width = max(len(row) for row in candidates)
    mask_rows: list[list[bool]] = []
    padded: list[list[str]] = []
    for row in candidates:
        if len(row) < max_width:
            pad_value = row[-1]
            padded.append(row + [pad_value] * (max_width - len(row)))
            mask_rows.append([True] * len(row) + [False] * (max_width - len(row)))
        else:
            padded.append(row)
            mask_rows.append([True] * len(row))
    return padded, torch.tensor(mask_rows, dtype=torch.bool)


def _batch_rows_for_step(rows: list[dict[str, Any]], step_idx: int, batch_size: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    batch_size = max(1, int(batch_size))
    start = ((max(1, int(step_idx)) - 1) * batch_size) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def _encode_candidate_actions(
    model: CLSTRModel,
    candidate_action_texts: list[list[str]],
    freeze_encoder: bool,
) -> torch.Tensor:
    width = len(candidate_action_texts[0])
    flat = [text for row in candidate_action_texts for text in row]
    if freeze_encoder:
        with torch.no_grad():
            encoded = model.encode_observations(flat)
    else:
        encoded = model.encode_observations(flat)
    return encoded.view(len(candidate_action_texts), width, -1)


def _recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(-1))
    top = torch.topk(logits, k=k, dim=-1).indices
    return float((top == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())


def _normalize_split_list(values: list[str] | tuple[str, ...] | None, default: tuple[str, ...]) -> list[str]:
    if values is None:
        return list(default)
    return [str(value) for value in values]


def _filter_trajectories_by_split(
    trajectories: list[dict[str, Any]],
    include_splits: list[str],
    exclude_splits: list[str],
    max_trajectories: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    include_set = set(include_splits)
    exclude_set = set(exclude_splits)
    included_candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    raw_split_counts = Counter(str(row.get("split", "unknown")) for row in trajectories)
    for trajectory in trajectories:
        split = str(trajectory.get("split", "unknown"))
        if split in exclude_set or (include_set and split not in include_set):
            excluded.append(trajectory)
        else:
            included_candidates.append(trajectory)
    selection_strategy = "all"
    if max_trajectories is None:
        selected = included_candidates
    else:
        selection_strategy = "environment_round_robin"
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trajectory in included_candidates:
            groups[str(trajectory.get("environment", "unknown"))].append(trajectory)
        selected = []
        env_order = sorted(groups)
        cursor = 0
        while len(selected) < max_trajectories and any(groups.values()):
            env = env_order[cursor % len(env_order)]
            if groups[env]:
                selected.append(groups[env].pop(0))
            cursor += 1
    overflow = included_candidates[len(selected) :]
    if max_trajectories is not None:
        selected_ids = {id(row) for row in selected}
        overflow = [row for row in included_candidates if id(row) not in selected_ids]
    excluded_for_report = excluded + overflow
    filter_report = {
        "include_splits": include_splits,
        "exclude_splits": exclude_splits,
        "selection_strategy": selection_strategy,
        "raw_split_counts": dict(sorted(raw_split_counts.items())),
        "included_split_counts": dict(sorted(Counter(str(row.get("split", "unknown")) for row in selected).items())),
        "included_environment_counts": dict(
            sorted(Counter(str(row.get("environment", "unknown")) for row in selected).items())
        ),
        "excluded_split_counts": dict(
            sorted(Counter(str(row.get("split", "unknown")) for row in excluded_for_report).items())
        ),
        "test_split_excluded_from_training": "test" not in {
            str(row.get("split", "unknown")) for row in selected
        },
        "max_trajectories": max_trajectories,
    }
    return selected, excluded_for_report, filter_report


def _coverage_counts(values: list[Any]) -> dict[str, int]:
    present = sum(1 for value in values if value is not None)
    return {"present": present, "missing": len(values) - present}


def _data_readiness(
    selected_trajectories: list[dict[str, Any]],
    excluded_trajectories: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    step_values = [
        step
        for trajectory in selected_trajectories
        for step in trajectory.get("steps", [])
        if isinstance(step, dict)
    ]
    return {
        "trajectory_count": len(selected_trajectories),
        "flattened_step_count": len(rows),
        "raw_step_count": len(step_values),
        "pseudo_skill_count": len(skill_rows),
        "low_confidence_mapping_count": sum(
            1 for step in step_values if bool(step.get("pseudo_skill_low_confidence"))
        ),
        "included_split_counts": dict(
            sorted(Counter(str(row.get("split", "unknown")) for row in selected_trajectories).items())
        ),
        "excluded_split_counts": dict(
            sorted(Counter(str(row.get("split", "unknown")) for row in excluded_trajectories).items())
        ),
        "environment_counts": dict(
            sorted(Counter(str(row.get("environment", "unknown")) for row in selected_trajectories).items())
        ),
        "coverage": {
            "success": _coverage_counts([row.get("success") for row in selected_trajectories]),
            "reward": _coverage_counts([row.get("reward") for row in selected_trajectories]),
        },
        "missing_fields": {
            "next_observation_text": _coverage_counts([step.get("next_observation_text") for step in step_values]),
            "step_reward": _coverage_counts([step.get("reward") for step in step_values]),
        },
    }


def _gpu_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": None,
            "total_memory_gb": None,
            "peak_memory_allocated_gb": None,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device": str(device),
        "gpu_name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 3),
        "peak_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
    }


def write_aux_pretrain_design_report(output_path: str | Path) -> dict[str, Any]:
    report = {
        "status": "ok",
        "stage": "auxiliary_trajectory_pretrain_gate",
        "not_rl_fine_tuning": True,
        "retention_failure_analysis": {
            "old_checkpoint": "outputs/aux_trajectory_pretrain_full/checkpoints/aux_trajectory_pretrain-step512.pt",
            "old_retention_metrics": {
                "NDCG@10": 0.24290,
                "Recall@10": 0.29501,
                "MAP@10": 0.20377,
            },
            "failure_reason": (
                "old auxiliary action imitation used skill_table.logits pseudo-skill CE and optimized "
                "encoder.proj plus skill_table.W, damaging the SKILLRET routing geometry"
            ),
            "old_checkpoint_role": "debug_only_not_full_init",
        },
        "routing_foundation_components": [
            "encoder.backbone",
            "encoder.proj",
            "skill_table.W",
            "skill_table.E",
            "native_rerank_sidecar",
        ],
        "action_imitation_policy": {
            "disallowed_old_path": "skill_table.logits pseudo-skill CE",
            "new_path": "UniversalActionAdapter over candidate action_text ranking",
            "does_not_train_skillret_routing": True,
            "pseudo_skill_role": "transition action id only; not the imitation class target",
        },
        "data_boundary": {
            "training_data_source": "auxiliary_trajectory",
            "not_skillsbench_clean_router": True,
            "not_skillsbench_harness_results": True,
            "not_successful_trajectories": True,
        },
        "loss_taxonomy": {
            "action_imitation": "L_act_pretraining_proxy",
            "transition_consistency_proxy": "L_trans",
            "belief_update_proxy": "L_trans",
            "stop_done_prediction": "STOP_policy_auxiliary_proxy",
            "retrieval_listwise_sidecar": "L_retr",
            "new_core_loss_added": False,
        },
        "frozen_gate_policy": {
            "freeze_encoder_backbone": True,
            "freeze_encoder_proj": True,
            "freeze_skill_table_W": True,
            "freeze_skill_table_E": True,
            "native_rerank_loaded_for_provenance_only": True,
        },
        "routing_init_policy": {
            "uses_clstr_native_routing_init": True,
            "qdoc_adapter_used": False,
            "native_rerank_sidecar_role": "loaded for routing provenance; not optimized by auxiliary losses",
        },
        "eval_positioning": {
            "auxiliary_eval_diagnostic": True,
            "not_skillsbench_pass_rate": True,
            "not_closed_loop_harness_success": True,
        },
    }
    _write_json(Path(output_path), report)
    return report


def _config_to_model(config: dict[str, Any], skills: list[Any]) -> CLSTRModel:
    return CLSTRModel(
        CLSTRConfig(
            base_model_name=str(config["base_model_name"]),
            d=int(config["d"]),
            d_a=int(config["d_a"]),
            top_k=int(config["top_k"]),
            encoder_pooling=str(config["encoder_pooling"]),
            cross_encoder_pooling=str(config["cross_encoder_pooling"]),
            tokenizer_padding_side=config["tokenizer_padding_side"],
            torch_dtype=config["torch_dtype"],
            freeze_backbone=bool(config["freeze_backbone"]),
            max_length=config["max_length"],
            projection_init=str(config["projection_init"]),
            normalize_embeddings=bool(config["normalize_embeddings"]),
            hf_cache_dir=config.get("hf_cache_dir"),
            local_files_only=bool(config.get("local_files_only", False)),
            defer_skill_table_init=bool(config["defer_skill_table_init"]),
            skill_text_format=str(config["skill_text_format"]),
            skill_table_batch_size=int(config["skill_table_batch_size"]),
            skill_table_adapter_init=str(config["skill_table_adapter_init"]),
            use_cross_encoder=bool(config["use_cross_encoder"]),
        ),
        skills,
    )


def _build_model_from_routing_init(
    routing_init_manifest: str | Path,
    skill_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any]]:
    manifest_path = Path(routing_init_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("qdoc_adapter_used") is not False:
        raise ValueError("routing init manifest must have qdoc_adapter_used=false")
    base_checkpoint = _resolve_existing_path(str(manifest["base_clstr_checkpoint"]), Path.cwd())
    payload = torch.load(base_checkpoint, map_location="cpu")
    config = _config_from_checkpoint(payload)
    selected_skills_path = output_dir / "selected_pseudo_skills.jsonl"
    _write_jsonl(selected_skills_path, skill_rows)
    skills = load_eval_pool(selected_skills_path)
    model = _config_to_model(config, skills)
    state_dict = payload.get("model_state_dict", payload)
    filtered = _filter_state_dict(model, state_dict)
    model.load_state_dict(filtered, strict=False)
    native_rerank_loaded = False
    native_rerank_checkpoint = manifest.get("native_rerank_checkpoint")
    if manifest.get("native_rerank_adopted") and native_rerank_checkpoint:
        rerank_path = _resolve_existing_path(str(native_rerank_checkpoint), Path.cwd())
        rerank_payload = torch.load(rerank_path, map_location="cpu")
        rerank_config = dict(rerank_payload.get("config") or {})
        reranker = CLSTRNativeResidualRerankHead(int(rerank_config.get("d", config["d"])))
        reranker.load_state_dict(rerank_payload["reranker_state_dict"])
        reranker.eval()
        native_rerank_loaded = True
    report = {
        "manifest": str(manifest_path),
        "base_clstr_checkpoint": str(base_checkpoint),
        "base_clstr_loaded": True,
        "base_loaded_state_keys": len(filtered),
        "native_rerank_checkpoint": str(native_rerank_checkpoint) if native_rerank_checkpoint else None,
        "native_rerank_loaded": native_rerank_loaded,
        "native_rerank_used_for_loss": False,
        "native_rerank_loss_reason": RERANK_SIDEcar_REASON,
        "native_rerank_adopted": bool(manifest.get("native_rerank_adopted")),
        "qdoc_adapter_used": False,
        "downstream_init_for": manifest.get("downstream_init_for"),
    }
    return model, config, report


def _build_model_from_base_name(
    base_model_name: str,
    skill_rows: list[dict[str, Any]],
    output_dir: Path,
    model_dim: int,
    top_k: int,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any]]:
    selected_skills_path = output_dir / "selected_pseudo_skills.jsonl"
    _write_jsonl(selected_skills_path, skill_rows)
    skills = load_eval_pool(selected_skills_path)
    config = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "d_a": max(4, model_dim // 4),
        "top_k": top_k,
        "encoder_pooling": "masked_mean",
        "cross_encoder_pooling": "masked_mean",
        "tokenizer_padding_side": None,
        "torch_dtype": None,
        "freeze_backbone": False,
        "max_length": None,
        "projection_init": "default",
        "normalize_embeddings": False,
        "defer_skill_table_init": False,
        "skill_text_format": "clstr",
        "skill_table_batch_size": 32,
        "skill_table_adapter_init": "default",
        "use_cross_encoder": True,
    }
    model = _config_to_model(config, skills)
    report = {
        "manifest": None,
        "base_clstr_checkpoint": None,
        "base_clstr_loaded": False,
        "base_model_name": base_model_name,
        "native_rerank_checkpoint": None,
        "native_rerank_loaded": False,
        "native_rerank_used_for_loss": False,
        "native_rerank_loss_reason": "no routing init manifest supplied",
        "qdoc_adapter_used": False,
    }
    return model, config, report


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(trainable)


def _freeze_routing_foundation(model: CLSTRModel) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad_(False)
    _set_module_trainable(model.transition, True)
    _set_module_trainable(model.gate, True)
    _set_module_trainable(model.stop_head, True)
    frozen_modules = [
        "encoder.backbone",
        "encoder.proj",
        "skill_table.W",
        "skill_table.E",
        "native_rerank_sidecar",
    ]
    if model.cross_encoder is not None:
        frozen_modules.append("cross_encoder")
    return {
        "frozen_routing_foundation": True,
        "frozen_modules": frozen_modules,
        "trainable_modules": ["transition", "gate", "stop_head"],
    }


def _parameter_audit(
    model: CLSTRModel,
    action_adapter: UniversalActionAdapter | None,
    frozen_routing_foundation: bool,
) -> dict[str, Any]:
    trainable_modules: list[str] = []
    module_map: list[tuple[str, torch.nn.Module | None]] = [
        ("encoder.proj", model.encoder.proj),
        ("skill_table.W", model.skill_table.W),
        ("transition", model.transition),
        ("gate", model.gate),
        ("stop_head", model.stop_head),
        ("cross_encoder", model.cross_encoder),
    ]
    for module_name, module in module_map:
        if module is not None and any(param.requires_grad for param in module.parameters()):
            trainable_modules.append(module_name)
    if action_adapter is not None and any(param.requires_grad for param in action_adapter.parameters()):
        trainable_modules.append("universal_action_adapter")
    frozen_modules = []
    if frozen_routing_foundation:
        frozen_modules = [
            "encoder.backbone",
            "encoder.proj",
            "skill_table.W",
            "skill_table.E",
            "native_rerank_sidecar",
        ]
        if model.cross_encoder is not None:
            frozen_modules.append("cross_encoder")
    trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if action_adapter is not None:
        trainable_param_count += sum(param.numel() for param in action_adapter.parameters() if param.requires_grad)
    frozen_param_count = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {
        "trainable_modules": trainable_modules,
        "frozen_modules": frozen_modules,
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": frozen_param_count,
    }


def _optimizer_groups(
    model: CLSTRModel,
    action_adapter: UniversalActionAdapter | None,
    learning_rate: float,
    use_universal_action_adapter: bool,
    freeze_routing_foundation: bool,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if use_universal_action_adapter and action_adapter is not None:
        for module in (model.transition, model.gate, model.stop_head, action_adapter):
            params = [param for param in module.parameters() if param.requires_grad]
            if params:
                groups.append({"params": params, "lr": learning_rate})
        return groups
    legacy_modules = [model.encoder.proj, model.skill_table.W, model.transition, model.gate, model.stop_head]
    if freeze_routing_foundation:
        legacy_modules = [model.transition, model.gate, model.stop_head]
    for module in legacy_modules:
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            groups.append({"params": params, "lr": learning_rate})
    return groups


def _compute_batch_losses(
    model: CLSTRModel,
    batch: list[dict[str, Any]],
    device: torch.device,
    top_k: int,
    action_adapter: UniversalActionAdapter | None = None,
    action_pools: dict[str, Any] | None = None,
    action_negative_k: int = 15,
    candidate_strategy: str = "environment_action_pool",
    freeze_routing_foundation: bool = False,
    transition_objective: str = "observation_cosine",
    feature_cache: FrozenAuxiliaryFeatureCache | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    if transition_objective not in {"observation_cosine", "next_action_ce"}:
        raise ValueError(f"unsupported transition_objective: {transition_objective}")
    labels = torch.tensor([row["label"] for row in batch], device=device, dtype=torch.long)
    row_indices_cpu = (
        torch.tensor([int(row["row_index"]) for row in batch], dtype=torch.long)
        if feature_cache is not None
        else None
    )
    if feature_cache is not None:
        h = feature_cache.state_embeddings.index_select(0, row_indices_cpu).to(device)
    else:
        states = [row["state_text"] for row in batch]
        h = model.encode_states(states)
    logits = model.skill_table.retrieval_logits(h)
    action_scores = logits
    if action_adapter is None:
        action_loss = F.cross_entropy(logits, labels)
        action_loss_key = "action_imitation_loss"
    else:
        if action_pools is None:
            raise ValueError("action_pools are required when action_adapter is enabled")
        if feature_cache is not None:
            assert row_indices_cpu is not None
            candidate_indices = feature_cache.row_candidate_indices.index_select(0, row_indices_cpu)
            candidate_mask = feature_cache.row_candidate_mask.index_select(0, row_indices_cpu).to(device)
            flat_candidate_indices = candidate_indices.reshape(-1)
            action_embs = feature_cache.action_embeddings.index_select(0, flat_candidate_indices)
            action_embs = action_embs.view(candidate_indices.size(0), candidate_indices.size(1), -1).to(device)
        else:
            candidate_texts, candidate_mask = _build_candidate_action_batch(
                batch,
                action_pools,
                action_negative_k,
                candidate_strategy,
            )
            candidate_mask = candidate_mask.to(device)
            action_embs = _encode_candidate_actions(model, candidate_texts, freeze_routing_foundation)
        action_scores = action_adapter(h, action_embs, candidate_mask)
        action_loss = action_adapter.ranking_loss(action_scores)
        action_loss_key = "universal_action_ranking_loss"
    done_labels = torch.tensor([1.0 if row["done"] else 0.0 for row in batch], device=device)
    m_obs = torch.softmax(logits, dim=-1) @ model.skill_table.E
    stop_logits = model.stop_head(h, m_obs).squeeze(-1)
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, done_labels)
    transition_loss = torch.tensor(0.0, device=device)
    transition_cosine = torch.tensor(0.0, device=device)
    belief_loss = torch.tensor(0.0, device=device)
    belief_cosine = torch.tensor(0.0, device=device)
    transition_metrics: dict[str, float] = {}
    if transition_objective == "observation_cosine":
        next_items = [
            (idx, row)
            for idx, row in enumerate(batch)
            if row["next_observation_text"] is not None
        ]
        if next_items:
            next_texts = [str(row["next_observation_text"]) for _, row in next_items]
            subset_idx = torch.tensor([idx for idx, _ in next_items], device=device)
            m_t = m_obs.index_select(0, subset_idx)
            a_t = labels.index_select(0, subset_idx)
            if feature_cache is not None:
                assert row_indices_cpu is not None
                global_idx = row_indices_cpu.index_select(0, subset_idx.cpu())
                action_idx = feature_cache.row_action_indices.index_select(0, global_idx)
                o_t = feature_cache.action_embeddings.index_select(0, action_idx).to(device)
                h_next = feature_cache.next_observation_embeddings.index_select(0, global_idx).to(device)
                next_logits = model.skill_table.belief_logits(h_next)
                m_tilde_next = torch.softmax(next_logits, dim=-1) @ model.skill_table.E
                action_input = transition_action_input(model, a_t, like=o_t)
                m_hat = model.transition(m_t, action_input, o_t)
                gamma = model.gate(m_hat, m_tilde_next, o_t)
                m_next = gamma * m_tilde_next + (1.0 - gamma) * m_hat
            else:
                o_t = model.encode_observations([row["action_text"] for _, row in next_items])
                m_next, m_hat, m_tilde_next = model.step_update(m_t, a_t, o_t, next_texts)
            transition_cosine = F.cosine_similarity(m_hat, m_tilde_next.detach(), dim=-1).mean()
            transition_loss = 1.0 - transition_cosine
            belief_cosine = F.cosine_similarity(m_next, m_tilde_next.detach(), dim=-1).mean()
            belief_loss = 1.0 - belief_cosine
        transition_metrics = {
            "transition_consistency_proxy": float(transition_loss.detach().cpu().item()),
            "transition_proxy_cosine": float(transition_cosine.detach().cpu().item()),
            "belief_update_proxy": float(belief_loss.detach().cpu().item()),
            "belief_update_cosine": float(belief_cosine.detach().cpu().item()),
        }
    else:
        next_items = [
            (idx, row)
            for idx, row in enumerate(batch)
            if row.get("next_label") is not None
        ]
        if next_items:
            subset_idx = torch.tensor([idx for idx, _ in next_items], device=device)
            next_labels = torch.tensor(
                [int(row["next_label"]) for _, row in next_items],
                device=device,
                dtype=torch.long,
            )
            m_t = m_obs.index_select(0, subset_idx)
            a_t = labels.index_select(0, subset_idx)
            if feature_cache is not None:
                assert row_indices_cpu is not None
                global_idx = row_indices_cpu.index_select(0, subset_idx.cpu())
                action_idx = feature_cache.row_action_indices.index_select(0, global_idx)
                o_t = feature_cache.action_embeddings.index_select(0, action_idx).to(device)
            else:
                o_t = model.encode_observations([row["action_text"] for _, row in next_items])
            action_input = transition_action_input(model, a_t, like=o_t)
            m_hat = model.transition(m_t, action_input, o_t)
            next_logits = model.skill_table.retrieval_logits(m_hat)
            transition_loss = F.cross_entropy(next_logits, next_labels)
            transition_metrics = {
                "next_action_prediction_loss": float(transition_loss.detach().cpu().item()),
                "next_action_recall_at_1": _recall_at_k(next_logits.detach(), next_labels, 1),
                f"next_action_recall_at_{top_k}": _recall_at_k(next_logits.detach(), next_labels, top_k),
            }
        else:
            transition_metrics = {
                "next_action_prediction_loss": 0.0,
                "next_action_recall_at_1": 0.0,
                f"next_action_recall_at_{top_k}": 0.0,
            }
    loss = action_loss + stop_loss + transition_loss + belief_loss
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "action_imitation_loss": float(action_loss.detach().cpu().item()),
        action_loss_key: float(action_loss.detach().cpu().item()),
        "stop_loss": float(stop_loss.detach().cpu().item()),
        **transition_metrics,
    }
    if action_adapter is None:
        metrics["recall_at_1"] = _recall_at_k(logits.detach(), labels, 1)
        metrics[f"recall_at_{top_k}"] = _recall_at_k(logits.detach(), labels, top_k)
    else:
        action_labels = torch.zeros(action_scores.size(0), device=device, dtype=torch.long)
        action_k = min(action_scores.size(1), max(1, top_k))
        metrics["action_recall_at_1"] = _recall_at_k(action_scores.detach(), action_labels, 1)
        metrics[f"action_recall_at_{action_k}"] = _recall_at_k(action_scores.detach(), action_labels, action_k)
    return loss, metrics, logits, labels


def _evaluate_aux_diagnostic(
    model: CLSTRModel,
    eval_rows: list[dict[str, Any]],
    output_path: Path,
    device: torch.device,
    batch_size: int,
    top_k: int,
    max_eval_steps: int | None,
    action_adapter: UniversalActionAdapter | None = None,
    action_pools: dict[str, Any] | None = None,
    action_negative_k: int = 15,
    candidate_strategy: str = "environment_action_pool",
    freeze_routing_foundation: bool = False,
    transition_objective: str = "observation_cosine",
) -> dict[str, Any]:
    selected_rows = eval_rows[:max_eval_steps] if max_eval_steps is not None else eval_rows
    if not selected_rows:
        report = {
            "status": "skipped",
            "reason": "no held-out auxiliary rows available for diagnostic eval",
            "not_skillsbench_pass_rate": True,
            "not_closed_loop_harness_success": True,
        }
        _write_json(output_path, report)
        return report
    totals: Counter[str] = Counter()
    metric_sums: Counter[str] = Counter()
    split_counts = Counter(str(row.get("split", "unknown")) for row in selected_rows)
    model.eval()
    if action_adapter is not None:
        action_adapter.eval()
    with torch.no_grad():
        for start in range(0, len(selected_rows), max(1, batch_size)):
            batch = selected_rows[start : start + max(1, batch_size)]
            _loss, metrics, logits, labels = _compute_batch_losses(
                model,
                batch,
                device,
                top_k,
                action_adapter=action_adapter,
                action_pools=action_pools,
                action_negative_k=action_negative_k,
                candidate_strategy=candidate_strategy,
                freeze_routing_foundation=freeze_routing_foundation,
                transition_objective=transition_objective,
            )
            batch_n = len(batch)
            for key, value in metrics.items():
                metric_sums[key] += float(value) * batch_n
            totals["rows"] += batch_n
            done_labels = torch.tensor([1.0 if row["done"] else 0.0 for row in batch], device=device)
            h = model.encode_states([row["state_text"] for row in batch])
            m_obs = torch.softmax(logits, dim=-1) @ model.skill_table.E
            stop_logits = model.stop_head(h, m_obs).squeeze(-1)
            stop_pred = (torch.sigmoid(stop_logits) >= 0.5).float()
            metric_sums["stop_accuracy"] += float((stop_pred == done_labels).float().mean().cpu().item()) * batch_n
            if action_adapter is None:
                metric_sums["action_recall_at_1"] += _recall_at_k(logits, labels, 1) * batch_n
                metric_sums[f"action_recall_at_{top_k}"] += _recall_at_k(logits, labels, top_k) * batch_n
    denom = max(1, totals["rows"])
    metrics_out = {key: round(value / denom, 6) for key, value in sorted(metric_sums.items())}
    report = {
        "status": "ok",
        "row_count": len(selected_rows),
        "available_row_count": len(eval_rows),
        "max_eval_steps": max_eval_steps,
        "split_counts": dict(sorted(split_counts.items())),
        "metrics": metrics_out,
        "action_loss_source": "universal_action_adapter" if action_adapter is not None else "skill_table",
        "transition_objective": transition_objective,
        "not_skillsbench_pass_rate": True,
        "not_closed_loop_harness_success": True,
        "note": "Auxiliary diagnostic measures imitation/STOP/transition proxy quality only.",
    }
    _write_json(output_path, report)
    model.train()
    if action_adapter is not None:
        action_adapter.train()
    return report


def _aux_loss_taxonomy(transition_objective: str) -> dict[str, Any]:
    taxonomy: dict[str, Any] = {
        "action_imitation": "L_act_pretraining_proxy",
        "stop_done_prediction": "STOP_policy_auxiliary_proxy",
        "retrieval_listwise_sidecar": "L_retr",
        "new_core_loss_added": False,
    }
    if transition_objective == "next_action_ce":
        taxonomy.update(
            {
                "next_action_prediction": "supervised_transition_prior_action_ce",
                "cosine_transition_alignment": "disabled_ablation_only",
            }
        )
    else:
        taxonomy.update(
            {
                "transition_consistency_proxy": "L_trans",
                "belief_update_proxy": "L_trans",
                "cosine_transition_alignment": "enabled_observation_alignment",
            }
        )
    return taxonomy


def run_aux_trajectory_pretrain(
    data_root: str | Path,
    base_model_name: str | None,
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 2,
    model_dim: int = 16,
    top_k: int = 8,
    max_trajectories: int | None = 512,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
    routing_init_manifest: str | Path | None = None,
    eval_splits: list[str] | None = None,
    max_eval_steps: int | None = 2048,
    freeze_routing_foundation: bool = False,
    use_universal_action_adapter: bool = False,
    action_negative_k: int = 15,
    candidate_strategy: str = "environment_action_pool",
    transition_objective: str = "observation_cosine",
    run_role: str | None = None,
) -> dict[str, Any]:
    if transition_objective not in {"observation_cosine", "next_action_ce"}:
        raise ValueError(f"unsupported transition_objective: {transition_objective}")
    if run_role is None:
        run_role = "debug_gate_not_final" if freeze_routing_foundation else "auxiliary_pretrain_full_candidate"
    if run_role not in {"debug_gate_not_final", "auxiliary_pretrain_full_candidate"}:
        raise ValueError(f"unsupported run_role: {run_role}")
    not_final = run_role == "debug_gate_not_final"
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    design_report_path = output_dir / "design_report.json"
    write_aux_pretrain_design_report(design_report_path)

    skill_rows, trajectories = load_aux_training_rows(data_root)
    if not skill_rows:
        raise ValueError(f"no auxiliary pseudo skills found under {data_root}")
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skill_rows)}
    include_splits_normalized = _normalize_split_list(include_splits, DEFAULT_INCLUDE_SPLITS)
    exclude_splits_normalized = _normalize_split_list(exclude_splits, DEFAULT_EXCLUDE_SPLITS)
    selected_trajectories, excluded_trajectories, split_filter = _filter_trajectories_by_split(
        trajectories,
        include_splits_normalized,
        exclude_splits_normalized,
        max_trajectories,
    )
    rows = _flatten_step_rows(selected_trajectories, skill_id_to_idx, None)
    if not rows:
        raise ValueError("no auxiliary trajectory steps map to pseudo skills")

    if routing_init_manifest is not None:
        model, model_config, routing_report = _build_model_from_routing_init(
            routing_init_manifest,
            skill_rows,
            output_dir,
        )
        top_k = min(top_k, int(model_config.get("top_k", top_k)), len(skill_rows))
    else:
        if base_model_name is None:
            raise ValueError("base_model_name is required when routing_init_manifest is not supplied")
        model, model_config, routing_report = _build_model_from_base_name(
            base_model_name,
            skill_rows,
            output_dir,
            model_dim,
            top_k,
        )
        top_k = min(top_k, len(skill_rows))
    action_adapter = None
    if use_universal_action_adapter:
        action_adapter = UniversalActionAdapter(int(model_config["d"]), hidden_dim=max(16, int(model_config["d"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    if action_adapter is not None:
        action_adapter.to(device)
    model.rebuild_skill_table()
    freeze_audit = {"frozen_routing_foundation": False, "frozen_modules": [], "trainable_modules": []}
    if freeze_routing_foundation:
        freeze_audit = _freeze_routing_foundation(model)
    if action_adapter is not None:
        freeze_audit["trainable_modules"] = list(
            dict.fromkeys([*freeze_audit.get("trainable_modules", []), "universal_action_adapter"])
        )
    optimizer_groups = _optimizer_groups(
        model,
        action_adapter,
        learning_rate,
        use_universal_action_adapter,
        freeze_routing_foundation,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)

    next_obs_missing = any(row["next_observation_text"] is None for row in rows)
    next_label_missing = any(row.get("next_label") is None for row in rows)
    skipped_targets = []
    if next_obs_missing:
        skipped_targets.append("next_observation_text_missing")
    if transition_objective == "next_action_ce" and next_label_missing:
        skipped_targets.append("next_action_label_missing")
    if all(row["reward"] is None for row in rows):
        skipped_targets.append("reward_missing")
    if any(row["reward"] is None for row in rows):
        skipped_targets.append("reward/outcome_policy")
    trained_targets = ["stop_done_prediction"]
    action_loss_source = "universal_action_adapter" if action_adapter is not None else "skill_table"
    if action_adapter is None:
        trained_targets.append("action_imitation_loss")
    else:
        trained_targets.append("universal_action_ranking_loss")
    if transition_objective == "next_action_ce":
        if any(row.get("next_label") is not None for row in rows):
            trained_targets.append("next_action_prediction_loss")
    elif any(row["next_observation_text"] is not None for row in rows):
        trained_targets.append("transition_consistency_proxy")
        trained_targets.append("belief_update_proxy")

    metrics: dict[str, float] = {}
    model.train()
    if action_adapter is not None:
        action_adapter.train()
    action_pools = _build_action_pools(rows) if action_adapter is not None else None
    feature_cache: FrozenAuxiliaryFeatureCache | None = None
    feature_cache_report: dict[str, Any] = {"used": False, "reason": "disabled"}
    if freeze_routing_foundation and action_adapter is not None and action_pools is not None:
        cache_batch_size = min(128, max(32, batch_size * 32))
        feature_cache = _build_frozen_auxiliary_feature_cache(
            model,
            rows,
            action_pools,
            action_negative_k=action_negative_k,
            candidate_strategy=candidate_strategy,
            encode_batch_size=cache_batch_size,
        )
        feature_cache_report = {
            **feature_cache.report(),
            "reason": "routing foundation is frozen; fixed encoder features are precomputed for full auxiliary training",
            "cache_persisted": False,
            "dtype": str(feature_cache.state_embeddings.dtype),
            "candidate_strategy": candidate_strategy,
            "action_negative_k": action_negative_k,
            "effective_batch_sizes": feature_cache.cache_stats.get("effective_batch_sizes", {}),
            "encoded_items": feature_cache.cache_stats.get("encoded_items", {}),
            "oom_backoffs": feature_cache.cache_stats.get("oom_backoffs", []),
        }
    for step_idx in range(1, max_steps + 1):
        batch = _batch_rows_for_step(rows, step_idx, batch_size)
        loss, metrics, _logits, _labels = _compute_batch_losses(
            model,
            batch,
            device,
            top_k,
            action_adapter=action_adapter,
            action_pools=action_pools,
            action_negative_k=action_negative_k,
            candidate_strategy=candidate_strategy,
            freeze_routing_foundation=freeze_routing_foundation,
            transition_objective=transition_objective,
            feature_cache=feature_cache,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if action_adapter is not None:
            torch.nn.utils.clip_grad_norm_(action_adapter.parameters(), 1.0)
        optimizer.step()
        if max_steps >= 1000 and (step_idx == 1 or step_idx % max(1000, max_steps // 20) == 0):
            print(
                f"[aux_pretrain] train step {step_idx}/{max_steps} loss={metrics.get('loss', 0.0):.4f}",
                file=sys.stderr,
                flush=True,
            )

    checkpoint_path = checkpoint_dir / f"aux_trajectory_pretrain-step{max_steps}.pt"
    loss_taxonomy = _aux_loss_taxonomy(transition_objective)
    torch.save(
        {
            "stage": "aux_trajectory_pretrain",
            "step": max_steps,
            "config": model_config,
            "transition_objective": transition_objective,
            "cosine_transition_alignment_enabled": transition_objective == "observation_cosine",
            "paper_role": run_role,
            "not_for_paper_table": not_final,
            "not_full_training_init": not_final,
            "frozen_routing_foundation": freeze_routing_foundation,
            "qdoc_adapter_used": False,
            "action_loss_source": action_loss_source,
            "candidate_strategy": candidate_strategy,
            "action_negative_k": action_negative_k,
            "sampling_strategy": "batch_stride",
            "feature_cache": feature_cache_report,
            "trainable_modules": freeze_audit["trainable_modules"] if freeze_routing_foundation else _parameter_audit(model, action_adapter, False)["trainable_modules"],
            "frozen_modules": freeze_audit["frozen_modules"] if freeze_routing_foundation else _parameter_audit(model, action_adapter, False)["frozen_modules"],
            "model_state_dict": model.state_dict(),
            "universal_action_adapter_state_dict": None if action_adapter is None else action_adapter.state_dict(),
            "metrics": metrics,
            "pseudo_skill_count": len(skill_rows),
            "trajectory_count": len(selected_trajectories),
            "step_count": len(rows),
            "routing_init": routing_report,
            "loss_taxonomy": loss_taxonomy,
        },
        checkpoint_path,
    )
    eval_splits_normalized = _normalize_split_list(eval_splits, ("test",))
    eval_trajectories, _eval_excluded, eval_filter = _filter_trajectories_by_split(
        trajectories,
        eval_splits_normalized,
        [],
        None,
    )
    eval_rows = _flatten_step_rows(eval_trajectories, skill_id_to_idx, None)
    eval_report_path = output_dir / "eval_report.json"
    eval_report = _evaluate_aux_diagnostic(
        model,
        eval_rows,
        eval_report_path,
        device,
        batch_size,
        top_k,
        max_eval_steps,
        action_adapter=action_adapter,
        action_pools=_build_action_pools(eval_rows) if action_adapter is not None else None,
        action_negative_k=action_negative_k,
        candidate_strategy=candidate_strategy,
        freeze_routing_foundation=freeze_routing_foundation,
        transition_objective=transition_objective,
    )
    readiness = _data_readiness(selected_trajectories, excluded_trajectories, skill_rows, rows)
    report = {
        "status": "ok",
        "training_data_source": "aux_trajectory",
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "design_report": str(design_report_path),
        "eval_report": str(eval_report_path),
        "steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pseudo_skill_count": len(skill_rows),
        "trajectory_count": len(selected_trajectories),
        "step_count": len(rows),
        "split_filter": split_filter,
        "eval_filter": eval_filter,
        "data_readiness": readiness,
        "routing_init": routing_report,
        "frozen_routing_foundation": freeze_routing_foundation,
        "qdoc_adapter_used": False,
        "action_loss_source": action_loss_source,
        "candidate_strategy": candidate_strategy,
        "action_negative_k": action_negative_k,
        "transition_objective": transition_objective,
        "cosine_transition_alignment_enabled": transition_objective == "observation_cosine",
        "sampling_strategy": "batch_stride",
        "feature_cache": feature_cache_report,
        "trainable_modules": freeze_audit["trainable_modules"] if freeze_routing_foundation else _parameter_audit(model, action_adapter, False)["trainable_modules"],
        "frozen_modules": freeze_audit["frozen_modules"] if freeze_routing_foundation else _parameter_audit(model, action_adapter, False)["frozen_modules"],
        "trained_targets": trained_targets,
        "skipped_targets": skipped_targets,
        "loss_taxonomy": loss_taxonomy,
        "metrics": metrics,
        "eval_metrics": eval_report.get("metrics", {}),
        "gpu": _gpu_report(device),
        "paper_role": run_role,
        "not_for_paper_table": not_final,
        "not_full_training_init": not_final,
        "model_config": {
            "base_model_name": model_config.get("base_model_name"),
            "d": model_config.get("d"),
            "d_a": model_config.get("d_a"),
            "top_k": model_config.get("top_k"),
            "freeze_backbone": model_config.get("freeze_backbone"),
            "encoder_pooling": model_config.get("encoder_pooling"),
            "tokenizer_padding_side": model_config.get("tokenizer_padding_side"),
            "torch_dtype": model_config.get("torch_dtype"),
            "qdoc_adapter_used": False,
        },
        "parameter_audit": _parameter_audit(model, action_adapter, freeze_routing_foundation),
        "note": "auxiliary trajectories train transition/belief/STOP/imitation pretraining signals; they are not SkillsBench clean_router_data.",
        "not_rl_fine_tuning": True,
        "forbidden_outputs": {
            "skillsbench_successful_trajectories": "not_written",
            "skillsbench_harness_results": "not_written",
            "clean_router_train_jsonl": "not_written",
        },
    }
    if action_adapter is not None:
        report["action_adapter"] = {
            "class": "UniversalActionAdapter",
            "hidden_dim": next(action_adapter.scorer.parameters()).shape[0] if hasattr(action_adapter, "scorer") else None,
        }
    _write_json(output_dir / "train_report.json", report)
    return report


def build_aux_pretrain_gate_report(
    train_report_path: str | Path,
    eval_report_path: str | Path,
    retention_metrics_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    train_report = _read_json(train_report_path)
    eval_report = _read_json(eval_report_path)
    retention_metrics = _read_json(retention_metrics_path)
    deltas = {
        key: round(float(retention_metrics.get(key, 0.0)) - baseline, 5)
        for key, baseline in RETENTION_BASELINE.items()
    }
    pass_fail = all(
        abs(deltas[key]) <= RETENTION_THRESHOLDS[key]
        for key in RETENTION_BASELINE
    )
    report = {
        "status": "pass" if pass_fail else "fail",
        "paper_role": "gate_debug_not_paper_result",
        "allow_full_non_test_auxiliary_training": bool(pass_fail and train_report.get("frozen_routing_foundation")),
        "retention_baseline": RETENTION_BASELINE,
        "retention_thresholds": RETENTION_THRESHOLDS,
        "retention_metrics": retention_metrics,
        "retention_deltas": deltas,
        "train_report_path": str(train_report_path),
        "eval_report_path": str(eval_report_path),
        "train_metrics": train_report.get("metrics", {}),
        "eval_metrics": eval_report.get("metrics", {}),
        "frozen_routing_foundation": train_report.get("frozen_routing_foundation", False),
        "frozen_modules": train_report.get("parameter_audit", {}).get("frozen_modules", []),
        "trainable_modules": train_report.get("parameter_audit", {}).get("trainable_modules", []),
        "action_loss_source": train_report.get("action_loss_source"),
        "candidate_strategy": train_report.get("candidate_strategy"),
        "action_negative_k": train_report.get("action_negative_k"),
        "retention_check": True,
    }
    _write_json(Path(output_path), report)
    return report
