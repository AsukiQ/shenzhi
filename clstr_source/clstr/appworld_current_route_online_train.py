from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from clstr.appworld_current_route_preference import CurrentRoutePreferenceDataset, build_current_route_preference_dataset
from clstr.appworld_current_route_preference_train import (
    _model_config_payload,
    compute_current_route_preference_batch_loss,
    configure_current_route_preference_trainable,
)
from clstr.appworld_current_route_rollout import build_current_route_rollout_trace
from clstr.full_base_train import V4_1B_TRANSITION_SCORING_MODE


RolloutRunner = Callable[..., dict[str, Any]]
FailureCorrectionBuilder = Callable[[dict[str, Any]], dict[str, Any]]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_trace(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("schema_version") or "") == "current_route_rollout.v1":
        return row
    return build_current_route_rollout_trace(row)


def _skill_id_to_idx(model: Any) -> dict[str, int]:
    skill_ids = [
        str(skill.get("skill_id") if isinstance(skill, dict) else getattr(skill, "skill_id", idx))
        for idx, skill in enumerate(getattr(model, "skills", []))
    ]
    return {skill_id: idx for idx, skill_id in enumerate(skill_ids)}


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    params = list(module.parameters())
    for param in params:
        param.requires_grad_(bool(trainable))
    return params if trainable else []


def _set_tensor_trainable(value: Any, trainable: bool) -> list[torch.nn.Parameter]:
    if isinstance(value, torch.nn.Parameter):
        value.requires_grad_(bool(trainable))
        return [value] if trainable else []
    if isinstance(value, torch.Tensor) and value.is_leaf:
        value.requires_grad_(bool(trainable))
    return []


def configure_online_stage0_correction_trainable(
    model: Any,
    *,
    train_skill_adapter: bool = True,
    train_retrieval_scale: bool = True,
    train_skill_bias: bool = True,
    train_encoder_projection: bool = False,
) -> dict[str, Any]:
    """Enable only Stage0 routing parameters needed for online miss correction.

    This is called after the Stage4 trainability setup. It deliberately avoids
    transition/belief modules so candidate-miss failures do not silently become
    Stage1/2 retraining.
    """

    trainable_params: list[torch.nn.Parameter] = []
    trainable_modules: list[str] = []
    skill_table = getattr(model, "skill_table", None)
    if skill_table is not None:
        params = _set_module_trainable(getattr(skill_table, "W", None), bool(train_skill_adapter))
        if params:
            trainable_params.extend(params)
            trainable_modules.append("skill_table.W")
        params = _set_tensor_trainable(
            getattr(skill_table, "logit_scale_retr", None),
            bool(train_retrieval_scale),
        )
        if params:
            trainable_params.extend(params)
            trainable_modules.append("skill_table.logit_scale_retr")
        params = _set_tensor_trainable(
            getattr(skill_table, "skill_bias_retr", None),
            bool(train_skill_bias),
        )
        if params:
            trainable_params.extend(params)
            trainable_modules.append("skill_table.skill_bias_retr")
    encoder = getattr(model, "encoder", None)
    params = _set_module_trainable(getattr(encoder, "proj", None), bool(train_encoder_projection))
    if params:
        trainable_params.extend(params)
        trainable_modules.append("encoder.proj")
    return {
        "enabled": bool(trainable_params),
        "trainable_modules": trainable_modules,
        "train_skill_adapter": bool(train_skill_adapter),
        "train_retrieval_scale": bool(train_retrieval_scale),
        "train_skill_bias": bool(train_skill_bias),
        "train_encoder_projection": bool(train_encoder_projection),
        "parameter_count": int(sum(param.numel() for param in trainable_params)),
    }


def _stage0_trainable_params(model: Any) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    skill_table = getattr(model, "skill_table", None)
    if isinstance(getattr(skill_table, "W", None), torch.nn.Module):
        params.extend(param for param in skill_table.W.parameters() if param.requires_grad)
    for name in ("logit_scale_retr", "skill_bias_retr"):
        value = getattr(skill_table, name, None)
        if isinstance(value, torch.nn.Parameter) and value.requires_grad:
            params.append(value)
    encoder = getattr(model, "encoder", None)
    proj = getattr(encoder, "proj", None)
    if isinstance(proj, torch.nn.Module):
        params.extend(param for param in proj.parameters() if param.requires_grad)
    unique: dict[int, torch.nn.Parameter] = {}
    for param in params:
        unique[id(param)] = param
    return list(unique.values())


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    values.extend(_as_list(row.get("positive_skill_id")))
    values.extend(_as_list(row.get("positive_skill_ids")))
    output: list[str] = []
    for value in values:
        skill_id = str(value)
        if skill_id and skill_id not in output:
            output.append(skill_id)
    return output


def compute_online_stage0_correction_loss(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = torch.device(getattr(model, "device", "cpu"))
    losses: list[torch.Tensor] = []
    usable_row_count = 0
    target_counts: list[int] = []
    missing_positive_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_text = str(row.get("query_text") or row.get("state_text") or "")
        targets = [skill_id_to_idx[skill_id] for skill_id in _positive_skill_ids(row) if skill_id in skill_id_to_idx]
        if not query_text or not targets:
            missing_positive_count += 1
            continue
        h_t = model.encode_states([query_text])
        logits = model.skill_table.retrieval_logits(h_t).reshape(-1).float()
        log_probs = F.log_softmax(logits, dim=-1)
        target_tensor = torch.tensor(targets, device=log_probs.device, dtype=torch.long)
        losses.append(-torch.logsumexp(log_probs.index_select(0, target_tensor), dim=0))
        usable_row_count += 1
        target_counts.append(len(targets))
    loss = torch.stack(losses).mean() if losses else torch.zeros((), device=device, requires_grad=True)
    return loss.to(device), {
        "online_stage0_loss": float(loss.detach().cpu().item()),
        "online_stage0_usable_row_count": int(usable_row_count),
        "online_stage0_missing_positive_count": int(missing_positive_count),
        "online_stage0_mean_target_count": float(sum(target_counts) / max(1, len(target_counts))),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fp.flush()


def _append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        _append_jsonl(path, row)
        count += 1
    return count


def _read_jsonl_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not str(path):
        return []
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _replay_dataset_from_row(row: dict[str, Any]) -> tuple[dict[str, Any], CurrentRoutePreferenceDataset]:
    if str(row.get("schema_version") or "") == "current_route_preference.v1":
        report = {
            "schema_version": "current_route_preference_report.v1",
            "rollout_count": 0,
            "sample_count": 1,
            "positive_rollout_count": 0,
            "positive_sample_count": 1,
            "negative_rollout_count": 0,
            "skipped_ambiguous_rollout_count": 0,
            "skipped_blocked_decision_count": 0,
            "skipped_unaligned_decision_count": 0,
            "skipped_by_outcome": {},
            "source": "prebuilt_current_route_preference",
        }
        trace = {
            "task_id": row.get("task_id"),
            "query_id": row.get("query_id"),
            "schema_version": "current_route_preference.v1",
        }
        return trace, CurrentRoutePreferenceDataset(samples=[row], report=report)
    trace = _as_trace(row)
    return trace, build_current_route_preference_dataset([trace])


def _save_latest_checkpoint(
    *,
    model: Any,
    output_path: Path,
    report_payload: dict[str, Any],
    checkpoint_metadata: dict[str, Any] | None = None,
) -> str:
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    torch.save(
        {
            "stage": "appworld_current_route_online_stage4",
            "config": _model_config_payload(model),
            "model_state_dict": model.state_dict(),
            "train_report": {
                **report_payload,
                "checkpoint_metadata": dict(checkpoint_metadata or {}),
            },
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def run_current_route_online_stage4_train_with_model(
    *,
    model: Any,
    rollout_runner: RolloutRunner,
    tasks: Iterable[dict[str, Any]],
    output_dir: str | Path,
    max_online_updates: int = 1,
    online_update_epochs: int = 1,
    replay_success_rollouts_path: str | Path | None = None,
    max_replay_updates: int = 0,
    batch_size: int = 1,
    learning_rate: float = 5.0e-5,
    train_transition: bool = False,
    loss_score_mode: str = "policy_head",
    policy_blend_alpha: float = 0.5,
    transition_residual_lambda: float = 0.0,
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE,
    learned_component_min_range: float = 0.1,
    learned_component_trust_top_k: int | None = 80,
    stability_kl_weight: float = 0.1,
    checkpoint_metadata: dict[str, Any] | None = None,
    failure_correction_builder: FailureCorrectionBuilder | None = None,
    enable_failure_stage4_correction: bool = False,
    enable_online_stage0_correction_update: bool = False,
    online_stage0_learning_rate: float | None = None,
    train_online_stage0_skill_adapter: bool = True,
    train_online_stage0_retrieval_scale: bool = True,
    train_online_stage0_skill_bias: bool = True,
    train_online_stage0_encoder_projection: bool = False,
) -> dict[str, Any]:
    """Run a minimal current-route online Stage4 loop.

    The runner executes one task with the current in-memory model, returns an
    official current-route rollout row/trace, and this function immediately
    updates trainable Stage4 parameters from clean positive decisions before
    moving to the next task.
    """

    task_rows = list(tasks)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_path / "online_rollouts.jsonl"
    metrics_path = output_path / "online_metrics.jsonl"
    stage0_corrections_path = output_path / "stage0_retrieval_corrections.jsonl"

    freeze_report = configure_current_route_preference_trainable(model, train_transition=train_transition)
    stage0_freeze_report: dict[str, Any] = {"enabled": False, "trainable_modules": []}
    if bool(enable_online_stage0_correction_update):
        stage0_freeze_report = configure_online_stage0_correction_trainable(
            model,
            train_skill_adapter=bool(train_online_stage0_skill_adapter),
            train_retrieval_scale=bool(train_online_stage0_retrieval_scale),
            train_skill_bias=bool(train_online_stage0_skill_bias),
            train_encoder_projection=bool(train_online_stage0_encoder_projection),
        )
    stage0_params = _stage0_trainable_params(model) if bool(enable_online_stage0_correction_update) else []
    stage0_param_ids = {id(param) for param in stage0_params}
    stage4_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in stage0_param_ids
    ]
    if not stage4_params and not stage0_params:
        return {
            "status": "blocked",
            "blocker": "no_trainable_parameters",
            "on_policy_rollout_used": False,
            "freeze_report": freeze_report,
            "stage0_freeze_report": stage0_freeze_report,
        }
    skill_id_to_idx = _skill_id_to_idx(model)
    if not skill_id_to_idx:
        return {
            "status": "blocked",
            "blocker": "empty_model_skill_table",
            "on_policy_rollout_used": False,
            "freeze_report": freeze_report,
        }

    optimizer = torch.optim.AdamW(stage4_params, lr=float(learning_rate)) if stage4_params else None
    stage0_optimizer = (
        torch.optim.AdamW(stage0_params, lr=float(online_stage0_learning_rate or learning_rate))
        if stage0_params
        else None
    )
    online_update_count = 0
    replay_update_count = 0
    optimizer_step_count = 0
    online_stage0_update_count = 0
    online_stage0_sample_count = 0
    checkpoint_save_count = 0
    rollout_count = 0
    sample_count = 0
    replay_sample_count = 0
    skipped_no_sample_count = 0
    stage0_correction_row_count = 0
    last_metrics: dict[str, Any] = {}
    latest_checkpoint = ""
    replay_load_report = {
        "path": str(replay_success_rollouts_path or ""),
        "row_count": 0,
        "max_replay_updates": int(max(0, int(max_replay_updates))),
    }

    replay_rows = _read_jsonl_rows(replay_success_rollouts_path)
    replay_load_report["row_count"] = len(replay_rows)
    for replay_index, replay_row in enumerate(replay_rows, start=1):
        if replay_update_count >= int(max(0, int(max_replay_updates))):
            break
        trace, dataset = _replay_dataset_from_row(replay_row)
        samples = dataset.samples
        if not samples:
            _append_jsonl(
                metrics_path,
                {
                    "task_index": 0,
                    "task_id": trace.get("task_id"),
                    "replay_index": replay_index,
                    "online_update": online_update_count,
                    "replay_update": replay_update_count,
                    "update_source": "replay_success",
                    "updated": False,
                    "reason": "no_clean_preference_samples",
                    "dataset_report": dataset.report,
                },
            )
            continue
        if optimizer is None:
            _append_jsonl(
                metrics_path,
                {
                    "task_index": 0,
                    "task_id": trace.get("task_id"),
                    "replay_index": replay_index,
                    "online_update": online_update_count,
                    "replay_update": replay_update_count,
                    "update_source": "replay_success",
                    "updated": False,
                    "reason": "no_stage4_trainable_parameters",
                    "dataset_report": dataset.report,
                },
            )
            continue
        batch = samples[: max(1, int(batch_size))]
        if callable(getattr(model, "train", None)):
            model.train()
        loss, metrics = compute_current_route_preference_batch_loss(
            model,
            batch,
            skill_id_to_idx=skill_id_to_idx,
            loss_score_mode=str(loss_score_mode),
            policy_blend_alpha=float(policy_blend_alpha),
            transition_residual_lambda=float(transition_residual_lambda),
            transition_scoring_mode=str(transition_scoring_mode),
            learned_component_min_range=float(learned_component_min_range),
            learned_component_trust_top_k=learned_component_trust_top_k,
            stability_kl_weight=float(stability_kl_weight),
        )
        online_batch_retry_all_samples = False
        if int(metrics.get("decision_count", 0) or 0) <= 0 and len(samples) > len(batch):
            retry_batch = list(samples)
            retry_loss, retry_metrics = compute_current_route_preference_batch_loss(
                model,
                retry_batch,
                skill_id_to_idx=skill_id_to_idx,
                loss_score_mode=str(loss_score_mode),
                policy_blend_alpha=float(policy_blend_alpha),
                transition_residual_lambda=float(transition_residual_lambda),
                transition_scoring_mode=str(transition_scoring_mode),
                learned_component_min_range=float(learned_component_min_range),
                learned_component_trust_top_k=learned_component_trust_top_k,
                stability_kl_weight=float(stability_kl_weight),
            )
            online_batch_retry_all_samples = True
            batch = retry_batch
            loss = retry_loss
            metrics = retry_metrics
        metrics = {"online_batch_retry_all_samples": bool(online_batch_retry_all_samples), **metrics}
        if (not torch.isfinite(loss.detach())) or int(metrics.get("decision_count", 0) or 0) <= 0:
            _append_jsonl(
                metrics_path,
                {
                    "task_index": 0,
                    "task_id": trace.get("task_id"),
                    "replay_index": replay_index,
                    "online_update": online_update_count,
                    "replay_update": replay_update_count,
                    "update_source": "replay_success",
                    "updated": False,
                    "reason": "no_trainable_stage4_preference_signal",
                    "dataset_report": dataset.report,
                    **metrics,
                },
            )
            continue
        epoch_losses: list[float] = []
        actual_online_update_epochs = 0
        requested_online_update_epochs = int(max(1, int(online_update_epochs)))
        epoch_metrics = metrics
        epoch_loss = loss
        for epoch_idx in range(requested_online_update_epochs):
            if epoch_idx > 0:
                epoch_loss, epoch_metrics = compute_current_route_preference_batch_loss(
                    model,
                    batch,
                    skill_id_to_idx=skill_id_to_idx,
                    loss_score_mode=str(loss_score_mode),
                    policy_blend_alpha=float(policy_blend_alpha),
                    transition_residual_lambda=float(transition_residual_lambda),
                    transition_scoring_mode=str(transition_scoring_mode),
                    learned_component_min_range=float(learned_component_min_range),
                    learned_component_trust_top_k=learned_component_trust_top_k,
                    stability_kl_weight=float(stability_kl_weight),
                )
                epoch_metrics = {
                    "online_batch_retry_all_samples": bool(online_batch_retry_all_samples),
                    **epoch_metrics,
                }
                if (not torch.isfinite(epoch_loss.detach())) or int(epoch_metrics.get("decision_count", 0) or 0) <= 0:
                    break
            optimizer.zero_grad(set_to_none=True)
            epoch_loss.backward()
            optimizer.step()
            optimizer_step_count += 1
            actual_online_update_epochs += 1
            epoch_losses.append(float(epoch_loss.detach().cpu()))
            metrics = epoch_metrics
        replay_update_count += 1
        replay_sample_count += len(batch)
        last_metrics = {
            "task_index": 0,
            "task_id": trace.get("task_id"),
            "replay_index": replay_index,
            "online_update": online_update_count,
            "replay_update": replay_update_count,
            "update_source": "replay_success",
            "updated": True,
            "dataset_report": dataset.report,
            "online_update_epochs": requested_online_update_epochs,
            "actual_online_update_epochs": actual_online_update_epochs,
            "optimizer_step_count": actual_online_update_epochs,
            "total_optimizer_step_count": optimizer_step_count,
            "epoch_losses": epoch_losses,
            **metrics,
        }
        _append_jsonl(metrics_path, last_metrics)
        latest_checkpoint = _save_latest_checkpoint(
            model=model,
            output_path=output_path,
            report_payload={
                "online_update_count": online_update_count,
                "replay_update_count": replay_update_count,
                "total_stage4_update_count": online_update_count + replay_update_count,
                "optimizer_step_count": optimizer_step_count,
                "rollout_count": rollout_count,
                "sample_count": sample_count,
                "replay_sample_count": replay_sample_count,
                "batch_size": int(batch_size),
                "online_update_epochs": requested_online_update_epochs,
                "learning_rate": float(learning_rate),
                "freeze_report": freeze_report,
                "loss_score_mode": str(loss_score_mode),
                "policy_blend_alpha": float(policy_blend_alpha),
                "transition_residual_lambda": float(transition_residual_lambda),
                "transition_scoring_mode": str(transition_scoring_mode),
                "learned_component_min_range": float(learned_component_min_range),
                "stability_kl_weight": float(stability_kl_weight),
                "learned_component_trust_top_k": (
                    int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
                ),
                "replay_load_report": replay_load_report,
                "on_policy_rollout_used": bool(rollout_count > 0),
            },
            checkpoint_metadata=checkpoint_metadata,
        )
        checkpoint_save_count += 1

    for task_index, task in enumerate(task_rows, start=1):
        if online_update_count >= int(max_online_updates):
            break
        if callable(getattr(model, "eval", None)):
            model.eval()
        row = rollout_runner(task=task, task_index=task_index, model=model)
        trace = _as_trace(row)
        rollout_count += 1
        _append_jsonl(rollouts_path, trace)

        dataset = build_current_route_preference_dataset([trace])
        samples = dataset.samples
        dataset_report: dict[str, Any] = dict(dataset.report)
        update_source = "live_online"
        failure_correction_report: dict[str, Any] = {}
        appended_stage0_rows = 0
        stage0_metrics: dict[str, Any] = {}
        stage0_updated = False
        if not samples:
            correction_payload: dict[str, Any] = {}
            if failure_correction_builder is not None:
                correction_payload = failure_correction_builder(trace)
                if isinstance(correction_payload, dict):
                    failure_correction_report = dict(correction_payload.get("report") or {})
                    appended_stage0_rows = _append_jsonl_rows(
                        stage0_corrections_path,
                        correction_payload.get("stage0_retrieval_rows") or [],
                    )
                    stage0_correction_row_count += appended_stage0_rows
                    if bool(enable_failure_stage4_correction):
                        correction_samples = [
                            dict(item)
                            for item in (correction_payload.get("samples") or [])
                            if isinstance(item, dict)
                        ]
                        if correction_samples:
                            samples = correction_samples
                            dataset_report = failure_correction_report or {
                                "schema_version": "api_correction_preference_report.v1",
                                "sample_count": len(correction_samples),
                            }
                            update_source = "failure_stage4_correction"
            if appended_stage0_rows and stage0_optimizer is not None:
                correction_rows = (correction_payload or {}).get("stage0_retrieval_rows") if isinstance(correction_payload, dict) else []
                loss, stage0_metrics = compute_online_stage0_correction_loss(
                    model,
                    list(correction_rows or []),
                    skill_id_to_idx=skill_id_to_idx,
                )
                if torch.isfinite(loss.detach()) and int(stage0_metrics.get("online_stage0_usable_row_count", 0)) > 0:
                    stage0_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    stage0_optimizer.step()
                    online_stage0_update_count += 1
                    online_stage0_sample_count += int(stage0_metrics.get("online_stage0_usable_row_count", 0))
                    stage0_updated = True
                    latest_checkpoint = _save_latest_checkpoint(
                        model=model,
                        output_path=output_path,
                        report_payload={
                            "online_update_count": online_update_count,
                            "replay_update_count": replay_update_count,
                            "total_stage4_update_count": online_update_count + replay_update_count,
                            "optimizer_step_count": optimizer_step_count,
                            "online_stage0_update_count": online_stage0_update_count,
                            "rollout_count": rollout_count,
                            "sample_count": sample_count,
                            "replay_sample_count": replay_sample_count,
                            "online_stage0_sample_count": online_stage0_sample_count,
                            "batch_size": int(batch_size),
                            "online_update_epochs": int(max(1, int(online_update_epochs))),
                            "learning_rate": float(learning_rate),
                            "online_stage0_learning_rate": float(online_stage0_learning_rate or learning_rate),
                            "freeze_report": freeze_report,
                            "stage0_freeze_report": stage0_freeze_report,
                            "loss_score_mode": str(loss_score_mode),
                            "policy_blend_alpha": float(policy_blend_alpha),
                            "transition_residual_lambda": float(transition_residual_lambda),
                            "transition_scoring_mode": str(transition_scoring_mode),
                            "learned_component_min_range": float(learned_component_min_range),
                            "stability_kl_weight": float(stability_kl_weight),
                            "learned_component_trust_top_k": (
                                int(learned_component_trust_top_k)
                                if learned_component_trust_top_k is not None
                                else None
                            ),
                            "replay_load_report": replay_load_report,
                            "on_policy_rollout_used": True,
                        },
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    checkpoint_save_count += 1
            if not samples:
                skipped_no_sample_count += 1
                _append_jsonl(
                    metrics_path,
                    {
                        "task_index": task_index,
                        "task_id": trace.get("task_id"),
                        "online_update": online_update_count,
                        "updated": False,
                        "reason": "no_clean_preference_samples",
                        "dataset_report": dataset_report,
                        "stage0_correction_row_count": appended_stage0_rows,
                        "failure_correction_report": failure_correction_report,
                        "stage0_updated": bool(stage0_updated),
                        "module_update_target": "stage0_retrieval" if stage0_updated else "none",
                        **stage0_metrics,
                    },
                )
                continue

        batch = samples[: max(1, int(batch_size))]
        enable_pairwise_for_batch = bool(update_source == "failure_stage4_correction")
        if callable(getattr(model, "train", None)):
            model.train()
        loss, metrics = compute_current_route_preference_batch_loss(
            model,
            batch,
            skill_id_to_idx=skill_id_to_idx,
            loss_score_mode=str(loss_score_mode),
            policy_blend_alpha=float(policy_blend_alpha),
            transition_residual_lambda=float(transition_residual_lambda),
            transition_scoring_mode=str(transition_scoring_mode),
            learned_component_min_range=float(learned_component_min_range),
            learned_component_trust_top_k=learned_component_trust_top_k,
            stability_kl_weight=float(stability_kl_weight),
            enable_pairwise_correction_loss=enable_pairwise_for_batch,
        )
        online_batch_retry_all_samples = False
        if int(metrics.get("decision_count", 0) or 0) <= 0 and len(samples) > len(batch):
            retry_batch = list(samples)
            retry_loss, retry_metrics = compute_current_route_preference_batch_loss(
                model,
                retry_batch,
                skill_id_to_idx=skill_id_to_idx,
                loss_score_mode=str(loss_score_mode),
                policy_blend_alpha=float(policy_blend_alpha),
                transition_residual_lambda=float(transition_residual_lambda),
                transition_scoring_mode=str(transition_scoring_mode),
                learned_component_min_range=float(learned_component_min_range),
                learned_component_trust_top_k=learned_component_trust_top_k,
                stability_kl_weight=float(stability_kl_weight),
                enable_pairwise_correction_loss=enable_pairwise_for_batch,
            )
            online_batch_retry_all_samples = True
            batch = retry_batch
            loss = retry_loss
            metrics = retry_metrics
        metrics = {"online_batch_retry_all_samples": bool(online_batch_retry_all_samples), **metrics}
        if not torch.isfinite(loss.detach()):
            _append_jsonl(
                metrics_path,
                {
                    "task_index": task_index,
                    "task_id": trace.get("task_id"),
                    "online_update": online_update_count,
                    "updated": False,
                    "reason": "non_finite_loss",
                    "dataset_report": dataset_report,
                    "failure_correction_report": failure_correction_report,
                    **metrics,
                },
            )
            continue
        if int(metrics.get("decision_count", 0) or 0) <= 0:
            skipped_no_sample_count += 1
            _append_jsonl(
                metrics_path,
                {
                    "task_index": task_index,
                    "task_id": trace.get("task_id"),
                    "online_update": online_update_count,
                    "updated": False,
                    "reason": "no_trainable_stage4_preference_signal",
                    "dataset_report": dataset_report,
                    "failure_correction_report": failure_correction_report,
                    **metrics,
                },
            )
            continue
        if optimizer is None:
            skipped_no_sample_count += 1
            _append_jsonl(
                metrics_path,
                {
                    "task_index": task_index,
                    "task_id": trace.get("task_id"),
                    "online_update": online_update_count,
                    "updated": False,
                    "reason": "no_stage4_trainable_parameters",
                    "dataset_report": dataset_report,
                    "failure_correction_report": failure_correction_report,
                    **metrics,
                },
            )
            continue
        epoch_losses: list[float] = []
        actual_online_update_epochs = 0
        requested_online_update_epochs = int(max(1, int(online_update_epochs)))
        epoch_metrics = metrics
        epoch_loss = loss
        for epoch_idx in range(requested_online_update_epochs):
            if epoch_idx > 0:
                epoch_loss, epoch_metrics = compute_current_route_preference_batch_loss(
                    model,
                    batch,
                    skill_id_to_idx=skill_id_to_idx,
                    loss_score_mode=str(loss_score_mode),
                    policy_blend_alpha=float(policy_blend_alpha),
                    transition_residual_lambda=float(transition_residual_lambda),
                    transition_scoring_mode=str(transition_scoring_mode),
                    learned_component_min_range=float(learned_component_min_range),
                    learned_component_trust_top_k=learned_component_trust_top_k,
                    stability_kl_weight=float(stability_kl_weight),
                    enable_pairwise_correction_loss=enable_pairwise_for_batch,
                )
                epoch_metrics = {
                    "online_batch_retry_all_samples": bool(online_batch_retry_all_samples),
                    **epoch_metrics,
                }
                if (not torch.isfinite(epoch_loss.detach())) or int(epoch_metrics.get("decision_count", 0) or 0) <= 0:
                    break
            optimizer.zero_grad(set_to_none=True)
            epoch_loss.backward()
            optimizer.step()
            optimizer_step_count += 1
            actual_online_update_epochs += 1
            epoch_losses.append(float(epoch_loss.detach().cpu()))
            metrics = epoch_metrics
        online_update_count += 1
        sample_count += len(batch)
        last_metrics = {
            "task_index": task_index,
            "task_id": trace.get("task_id"),
                "online_update": online_update_count,
                "replay_update": replay_update_count,
                "update_source": update_source,
                "updated": True,
            "dataset_report": dataset_report,
            "failure_correction_report": failure_correction_report,
            "stage0_correction_row_count": appended_stage0_rows,
            "stage0_updated": bool(stage0_updated),
            "module_update_target": "stage4_correction" if update_source == "failure_stage4_correction" else "stage4_online",
            "online_update_epochs": requested_online_update_epochs,
            "actual_online_update_epochs": actual_online_update_epochs,
            "optimizer_step_count": actual_online_update_epochs,
            "total_optimizer_step_count": optimizer_step_count,
            "epoch_losses": epoch_losses,
            **metrics,
        }
        _append_jsonl(metrics_path, last_metrics)
        latest_checkpoint = _save_latest_checkpoint(
            model=model,
            output_path=output_path,
            report_payload={
                "online_update_count": online_update_count,
                "replay_update_count": replay_update_count,
                "total_stage4_update_count": online_update_count + replay_update_count,
                "optimizer_step_count": optimizer_step_count,
                "rollout_count": rollout_count,
                "sample_count": sample_count,
                "replay_sample_count": replay_sample_count,
                "batch_size": int(batch_size),
                "online_update_epochs": requested_online_update_epochs,
                "learning_rate": float(learning_rate),
                "freeze_report": freeze_report,
                "loss_score_mode": str(loss_score_mode),
                "policy_blend_alpha": float(policy_blend_alpha),
                "transition_residual_lambda": float(transition_residual_lambda),
                "transition_scoring_mode": str(transition_scoring_mode),
                "learned_component_min_range": float(learned_component_min_range),
                "stability_kl_weight": float(stability_kl_weight),
                "learned_component_trust_top_k": (
                    int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
                ),
                "replay_load_report": replay_load_report,
                "on_policy_rollout_used": True,
            },
            checkpoint_metadata=checkpoint_metadata,
        )
        checkpoint_save_count += 1

    status = "ok" if (online_update_count > 0 or replay_update_count > 0 or online_stage0_update_count > 0) else "blocked"
    report = {
        "status": status,
        "blocker": "" if status == "ok" else "no_online_updates",
        "on_policy_rollout_used": bool(rollout_count > 0),
        "rollout_count": rollout_count,
        "online_update_count": online_update_count,
        "replay_update_count": replay_update_count,
        "total_stage4_update_count": online_update_count + replay_update_count,
        "optimizer_step_count": optimizer_step_count,
        "online_stage0_update_count": online_stage0_update_count,
        "sample_count": sample_count,
        "replay_sample_count": replay_sample_count,
        "online_stage0_sample_count": online_stage0_sample_count,
        "skipped_no_sample_count": skipped_no_sample_count,
        "stage0_correction_row_count": stage0_correction_row_count,
        "stage0_retrieval_corrections_path": str(stage0_corrections_path),
        "checkpoint_save_count": checkpoint_save_count,
        "latest_checkpoint": latest_checkpoint,
        "online_rollouts_path": str(rollouts_path),
        "training_metrics_path": str(metrics_path),
        "max_online_updates": int(max_online_updates),
        "online_update_epochs": int(max(1, int(online_update_epochs))),
        "max_replay_updates": int(max(0, int(max_replay_updates))),
        "replay_load_report": replay_load_report,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "online_stage0_learning_rate": float(online_stage0_learning_rate or learning_rate),
        "freeze_report": freeze_report,
        "stage0_freeze_report": stage0_freeze_report,
        "loss_score_mode": str(loss_score_mode),
        "policy_blend_alpha": float(policy_blend_alpha),
        "transition_residual_lambda": float(transition_residual_lambda),
        "transition_scoring_mode": str(transition_scoring_mode),
        "learned_component_min_range": float(learned_component_min_range),
        "stability_kl_weight": float(stability_kl_weight),
        "learned_component_trust_top_k": (
            int(learned_component_trust_top_k) if learned_component_trust_top_k is not None else None
        ),
        "last_metrics": last_metrics,
    }
    (output_path / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
