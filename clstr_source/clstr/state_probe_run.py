from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import torch

from clstr.matched_history_data import serialize_factual_history
from clstr.matched_history_encoders import build_matched_history_encoder
from clstr.state_probe import (
    CANONICAL_SPLIT_SEED,
    ProbeExample,
    STATE_PROBE_SCHEMA,
    examples_as_dicts,
    fit_regularized_linear_probe,
    load_toolbench_probe_trajectories,
    majority_baseline,
    post_transition_events,
    require_canonical_split,
    split_report,
    trajectory_stratified_split,
)
from clstr.vnext_eval import load_vnext_stage2_for_evaluation
from clstr.vnext_training import file_sha256, load_or_build_frozen_text_cache


E2_RUN_SCHEMA = "clstr_state_probe_e2_run_v1"
E1_CHECKPOINT_SCHEMA = "clstr_matched_history_e1_train_v1"
E2_METHODS = ("current_only", "serialized", "gru", "transformer", "lstr")
E2_HISTORY_METHODS = ("serialized", "gru", "transformer", "lstr")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise RuntimeError("formal E2 execution requires a clean source worktree")
    return {"source_commit": commit, "source_worktree_clean": True}


def _load_e1_encoder(
    *,
    method: str,
    checkpoint_path: Path,
    expected_seed: int,
    d: int,
    max_horizon: int,
    foundation_checkpoint_path: Path,
    skills_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if method not in E2_HISTORY_METHODS:
        raise ValueError(f"unsupported E1 encoder for probing: {method}")
    checkpoint = checkpoint_path.resolve()
    report_path = checkpoint.parent / "train_report.json"
    if not checkpoint.is_file() or not report_path.is_file():
        raise ValueError(f"E2 is missing an E1 checkpoint/report pair: {checkpoint}")
    report = _read_json(report_path)
    if report.get("schema_version") != E1_CHECKPOINT_SCHEMA or report.get("status") != "ok":
        raise ValueError("E2 received an invalid E1 train report")
    if str(report.get("encoder_kind") or "") != method:
        raise ValueError("E2 checkpoint method differs from its E1 report")
    if int(report.get("seed") or -1) != int(expected_seed):
        raise ValueError("E2 checkpoint seed differs from the requested E1 seed")
    if Path(str(report.get("best_checkpoint_path") or "")).resolve() != checkpoint:
        raise ValueError("E2 checkpoint path differs from the E1 selected checkpoint")
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != str(report.get("best_checkpoint_sha256") or ""):
        raise ValueError("E2 checkpoint digest differs from the E1 train report")
    inputs = report.get("inputs") or {}
    foundation = inputs.get("foundation") or {}
    if Path(str(foundation.get("checkpoint_path") or "")).resolve() != foundation_checkpoint_path:
        raise ValueError("E2 foundation checkpoint differs from E1")
    if str(foundation.get("checkpoint_sha256") or "") != file_sha256(
        foundation_checkpoint_path
    ):
        raise ValueError("E2 foundation checkpoint digest differs from E1")
    if Path(str(foundation.get("training_skills_path") or "")).resolve() != skills_path:
        raise ValueError("E2 skill inventory differs from E1")
    if str(foundation.get("training_skills_sha256") or "") != file_sha256(skills_path):
        raise ValueError("E2 skill inventory digest differs from E1")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != E1_CHECKPOINT_SCHEMA:
        raise ValueError("E2 checkpoint has the wrong schema")
    if str(payload.get("encoder_kind") or "") != method:
        raise ValueError("E2 checkpoint payload has the wrong encoder kind")
    if int(payload.get("seed") or -1) != int(expected_seed):
        raise ValueError("E2 checkpoint payload has the wrong seed")
    state = payload.get("encoder_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("E2 checkpoint lacks encoder weights")
    encoder = build_matched_history_encoder(
        method,
        int(d),
        max_horizon=int(max_horizon),
    ).to(device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, {
        "method": method,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "train_report_path": str(report_path.resolve()),
        "train_report_sha256": file_sha256(report_path),
        "e1_source_commit": str(report.get("source_commit") or ""),
        "e1_seed": int(expected_seed),
        "e1_best_step": int(report.get("best_step") or 0),
        "e1_best_macro_mrr": float(report.get("best_macro_mrr") or 0.0),
    }


def _flatten_examples(
    trajectories: Sequence[Sequence[ProbeExample]],
) -> tuple[list[ProbeExample], list[list[dict[str, Any]]], list[str]]:
    examples: list[ProbeExample] = []
    event_prefixes: list[list[dict[str, Any]]] = []
    serialized_histories: list[str] = []
    for trajectory in trajectories:
        for row_index, example in enumerate(trajectory):
            events = post_transition_events(trajectory, row_index, max_horizon=16)
            examples.append(example)
            event_prefixes.append(events)
            serialized_histories.append(serialize_factual_history(events))
    if len(examples) != len(event_prefixes) or len(examples) != len(serialized_histories):
        raise RuntimeError("E2 causal-prefix materialization is misaligned")
    return examples, event_prefixes, serialized_histories


def _build_initial_memories(
    *,
    trajectories: Sequence[Sequence[ProbeExample]],
    foundation: Any,
    state_cache: Any,
    skill_count: int,
    device: torch.device,
    batch_size: int,
    belief_top_k: int,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for start in range(0, len(trajectories), int(batch_size)):
        batch = trajectories[start : start + int(batch_size)]
        initial_states = state_cache.batch(
            [trajectory[0].state_text for trajectory in batch],
            device=device,
        )
        legal = torch.ones(
            (len(batch), int(skill_count)),
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            memories = foundation.vnext_initial_belief(
                initial_states,
                legal,
                top_k=int(belief_top_k),
            )
        for trajectory, memory in zip(batch, memories):
            output[trajectory[0].trajectory_id] = memory.detach().cpu().to(torch.bfloat16)
    if len(output) != len(trajectories):
        raise RuntimeError("E2 initial-memory cache omitted trajectories")
    return output


def _extract_representations(
    *,
    examples: Sequence[ProbeExample],
    event_prefixes: Sequence[Sequence[dict[str, Any]]],
    serialized_histories: Sequence[str],
    encoders: dict[str, torch.nn.Module],
    foundation: Any,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any,
    history_cache: Any,
    initial_memories: dict[str, torch.Tensor],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if set(encoders) != set(E2_HISTORY_METHODS):
        raise ValueError("E2 representation extraction requires all four E1 encoders")
    if not (len(examples) == len(event_prefixes) == len(serialized_histories)):
        raise ValueError("E2 representation inputs are misaligned")
    d = int(foundation.vnext.d)
    output = {
        method: torch.empty((len(examples), d), dtype=torch.float32)
        for method in E2_METHODS
    }
    skill_embeddings: torch.Tensor | None = None
    for start in range(0, len(examples), int(batch_size)):
        stop = min(len(examples), start + int(batch_size))
        batch_examples = examples[start:stop]
        batch_events = event_prefixes[start:stop]
        current = state_cache.batch(
            [example.state_text for example in batch_examples],
            device=device,
        )
        initial = torch.stack(
            [initial_memories[example.trajectory_id] for example in batch_examples]
        ).to(device=device, dtype=current.dtype)
        horizon = max(len(events) for events in batch_events)
        shape = (len(batch_examples), horizon, d)
        event_states = torch.zeros(shape, device=device, dtype=current.dtype)
        event_skills = torch.zeros(shape, device=device, dtype=current.dtype)
        event_actions = torch.zeros(shape, device=device, dtype=current.dtype)
        event_results = torch.zeros(shape, device=device, dtype=current.dtype)
        event_mask = torch.zeros(
            (len(batch_examples), horizon),
            device=device,
            dtype=torch.bool,
        )
        result_mask = torch.zeros_like(event_mask)
        positions: list[tuple[int, int]] = []
        flat_states: list[str] = []
        flat_actions: list[str] = []
        flat_results: list[str] = []
        flat_skill_indices: list[int] = []
        for batch_index, events in enumerate(batch_events):
            for event_index, event in enumerate(events):
                skill_id = str(event.get("skill_id") or "")
                if skill_id not in skill_id_to_idx:
                    raise ValueError(f"E2 executed skill is absent from the E1 inventory: {skill_id}")
                positions.append((batch_index, event_index))
                flat_states.append(str(event["state_text"]))
                flat_actions.append(str(event["action_text"]))
                flat_results.append(str(event["result_text"]))
                flat_skill_indices.append(skill_id_to_idx[skill_id])
                event_mask[batch_index, event_index] = True
                result_mask[batch_index, event_index] = True
        state_values = state_cache.batch(flat_states, device=device)
        action_values = action_cache.batch(flat_actions, device=device)
        result_values = result_cache.batch(flat_results, device=device)
        if skill_embeddings is None or skill_embeddings.dtype != current.dtype:
            skill_embeddings = foundation.vnext_normalized_skill_embeddings(
                dtype=current.dtype
            )
        skill_values = skill_embeddings.index_select(
            0,
            torch.tensor(flat_skill_indices, device=device, dtype=torch.long),
        )
        for flat_index, (batch_index, event_index) in enumerate(positions):
            event_states[batch_index, event_index] = state_values[flat_index]
            event_actions[batch_index, event_index] = action_values[flat_index]
            event_results[batch_index, event_index] = result_values[flat_index]
            event_skills[batch_index, event_index] = skill_values[flat_index]
        serialized = history_cache.batch(
            serialized_histories[start:stop],
            device=device,
        )
        output["current_only"][start:stop] = current.detach().float().cpu()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for method in E2_HISTORY_METHODS:
                memory = encoders[method](
                    initial,
                    event_states,
                    event_skills,
                    event_actions,
                    event_results,
                    event_mask,
                    result_mask,
                    serialized_history_embedding=(serialized if method == "serialized" else None),
                )
                output[method][start:stop] = memory.detach().float().cpu()
    if any(not bool(torch.isfinite(value).all().item()) for value in output.values()):
        raise RuntimeError("E2 extracted a non-finite representation")
    return output


def run_state_probe_e2(
    *,
    foundation_checkpoint_path: str | Path,
    skills_path: str | Path,
    aligned_rows_path: str | Path,
    serialized_checkpoint_path: str | Path,
    gru_checkpoint_path: str | Path,
    transformer_checkpoint_path: str | Path,
    lstr_checkpoint_path: str | Path,
    expected_e1_seed: int,
    output_dir: str | Path,
    frozen_cache_dir: str | Path,
    batch_size: int = 32,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    max_horizon: int = 16,
    belief_top_k: int = 64,
    probe_max_iter: int = 100,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 29,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("E2 state probing requires CUDA for frozen representation extraction")
    if int(max_horizon) != 16 or int(belief_top_k) != 64:
        raise ValueError("canonical E2 requires horizon 16 and belief Top-64")
    if min(int(batch_size), int(cache_batch_size), int(probe_max_iter)) <= 0:
        raise ValueError("E2 batch sizes and probe iterations must be positive")
    device = torch.device("cuda")
    torch.manual_seed(int(expected_e1_seed))
    torch.cuda.manual_seed_all(int(expected_e1_seed))
    torch.set_float32_matmul_precision("high")

    root = Path(__file__).resolve().parents[1]
    git_identity = _git_identity(root)
    foundation_path = Path(foundation_checkpoint_path).resolve()
    skills = Path(skills_path).resolve()
    aligned = Path(aligned_rows_path).resolve()
    output = Path(output_dir).resolve()
    cache_root = Path(frozen_cache_dir).resolve()
    required = [foundation_path, skills, aligned]
    if any(not path.is_file() for path in required):
        raise ValueError("E2 is missing an immutable foundation/data input")
    if output.exists() and any(output.iterdir()):
        raise ValueError("E2 output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    trajectories = load_toolbench_probe_trajectories(aligned)
    assignment = trajectory_stratified_split(
        trajectories,
        seed=CANONICAL_SPLIT_SEED,
    )
    data_split_report = split_report(trajectories, assignment)
    require_canonical_split(data_split_report)
    examples, event_prefixes, serialized_histories = _flatten_examples(trajectories)
    if len(examples) != 1362:
        raise RuntimeError("E2 canonical example count changed after flattening")

    foundation, loaded_skills, skill_id_to_idx, foundation_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=foundation_path,
            training_skills_path=skills,
            benchmark_skills=[],
            device=device,
        )
    )
    foundation.eval()
    for parameter in foundation.parameters():
        parameter.requires_grad_(False)
    d = int(foundation.vnext.d)
    checkpoint_paths = {
        "serialized": Path(serialized_checkpoint_path),
        "gru": Path(gru_checkpoint_path),
        "transformer": Path(transformer_checkpoint_path),
        "lstr": Path(lstr_checkpoint_path),
    }
    encoders: dict[str, torch.nn.Module] = {}
    checkpoint_reports: dict[str, Any] = {}
    for method in E2_HISTORY_METHODS:
        encoder, checkpoint_report = _load_e1_encoder(
            method=method,
            checkpoint_path=checkpoint_paths[method],
            expected_seed=int(expected_e1_seed),
            d=d,
            max_horizon=max_horizon,
            foundation_checkpoint_path=foundation_path,
            skills_path=skills,
            device=device,
        )
        encoders[method] = encoder
        checkpoint_reports[method] = checkpoint_report
    e1_source_commits = {
        str(report.get("e1_source_commit") or "") for report in checkpoint_reports.values()
    }
    if len(e1_source_commits) != 1 or "" in e1_source_commits:
        raise ValueError("E2 E1 checkpoints do not share one source commit")

    state_texts = [example.state_text for example in examples]
    action_texts = [example.action_text for example in examples]
    result_texts = [example.result_text for example in examples]
    state_cache = load_or_build_frozen_text_cache(
        foundation,
        state_texts,
        role="state",
        batch_size=int(cache_batch_size),
        cache_root=cache_root,
        cache_shard_size=int(cache_shard_size),
    )
    action_cache = load_or_build_frozen_text_cache(
        foundation,
        action_texts,
        role="action",
        batch_size=int(cache_batch_size),
        cache_root=cache_root,
        cache_shard_size=int(cache_shard_size),
    )
    result_cache = load_or_build_frozen_text_cache(
        foundation,
        result_texts,
        role="result",
        batch_size=int(cache_batch_size),
        cache_root=cache_root,
        cache_shard_size=int(cache_shard_size),
    )
    history_cache = load_or_build_frozen_text_cache(
        foundation,
        serialized_histories,
        role="matched_history",
        batch_size=int(cache_batch_size),
        cache_root=cache_root,
        cache_shard_size=int(cache_shard_size),
    )
    initial_memories = _build_initial_memories(
        trajectories=trajectories,
        foundation=foundation,
        state_cache=state_cache,
        skill_count=len(loaded_skills),
        device=device,
        batch_size=batch_size,
        belief_top_k=belief_top_k,
    )
    representations = _extract_representations(
        examples=examples,
        event_prefixes=event_prefixes,
        serialized_histories=serialized_histories,
        encoders=encoders,
        foundation=foundation,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        history_cache=history_cache,
        initial_memories=initial_memories,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        batch_size=batch_size,
    )

    split_names = [assignment[example.trajectory_id] for example in examples]
    trajectory_ids = [example.trajectory_id for example in examples]
    latest_labels = torch.tensor(
        [example.latest_result_label for example in examples],
        dtype=torch.long,
    )
    cumulative_labels = torch.tensor(
        [example.cumulative_error_label for example in examples],
        dtype=torch.long,
    )
    representation_path = output / "representations.pt"
    temporary_representation_path = representation_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": STATE_PROBE_SCHEMA,
            "e1_seed": int(expected_e1_seed),
            "example_ids": [example.example_id for example in examples],
            "trajectory_ids": trajectory_ids,
            "split_names": split_names,
            "latest_result_labels": latest_labels,
            "cumulative_error_labels": cumulative_labels,
            "representations": {
                method: value.to(torch.bfloat16) for method, value in representations.items()
            },
        },
        temporary_representation_path,
    )
    temporary_representation_path.replace(representation_path)

    method_metrics: dict[str, Any] = {}
    test_predictions: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for method in E2_METHODS:
        latest_metrics, latest_predictions = fit_regularized_linear_probe(
            features=representations[method],
            labels=latest_labels,
            split_names=split_names,
            trajectory_ids=trajectory_ids,
            num_classes=2,
            device=device,
            max_iter=int(probe_max_iter),
            bootstrap_draws=int(bootstrap_draws),
            bootstrap_seed=int(bootstrap_seed),
        )
        cumulative_metrics, cumulative_predictions = fit_regularized_linear_probe(
            features=representations[method],
            labels=cumulative_labels,
            split_names=split_names,
            trajectory_ids=trajectory_ids,
            num_classes=3,
            device=device,
            max_iter=int(probe_max_iter),
            bootstrap_draws=int(bootstrap_draws),
            bootstrap_seed=int(bootstrap_seed),
        )
        method_metrics[method] = {
            "latest_result": latest_metrics,
            "cumulative_error_count": cumulative_metrics,
        }
        test_predictions[method] = {
            "latest_result": latest_predictions,
            "cumulative_error_count": cumulative_predictions,
        }
    method_metrics["majority"] = {
        "latest_result": majority_baseline(
            labels=latest_labels,
            split_names=split_names,
            trajectory_ids=trajectory_ids,
            num_classes=2,
            bootstrap_draws=int(bootstrap_draws),
            bootstrap_seed=int(bootstrap_seed),
        ),
        "cumulative_error_count": majority_baseline(
            labels=cumulative_labels,
            split_names=split_names,
            trajectory_ids=trajectory_ids,
            num_classes=3,
            bootstrap_draws=int(bootstrap_draws),
            bootstrap_seed=int(bootstrap_seed),
        ),
    }

    test_indices = [index for index, name in enumerate(split_names) if name == "test"]
    prediction_path = output / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for local_index, global_index in enumerate(test_indices):
            record: dict[str, Any] = {
                "example_id": examples[global_index].example_id,
                "trajectory_id": examples[global_index].trajectory_id,
                "step_index": examples[global_index].step_index,
                "latest_result_label": int(latest_labels[global_index].item()),
                "cumulative_error_label": int(cumulative_labels[global_index].item()),
                "methods": {},
            }
            for method in E2_METHODS:
                record["methods"][method] = {
                    "latest_result_probabilities": test_predictions[method][
                        "latest_result"
                    ]["probabilities"][local_index].tolist(),
                    "cumulative_error_probabilities": test_predictions[method][
                        "cumulative_error_count"
                    ]["probabilities"][local_index].tolist(),
                }
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    report = {
        "schema_version": E2_RUN_SCHEMA,
        "status": "ok",
        "e1_seed": int(expected_e1_seed),
        **git_identity,
        "contract": {
            "probe_timing": "post_transition_after_aligned_action_and_result",
            "current_only_input": "pre_action_current_state_without_current_action_or_result",
            "history_horizon": int(max_horizon),
            "frozen_foundation": True,
            "frozen_history_encoders": True,
            "shared_route_scorer_used": False,
            "probe_model": "class_balanced_l2_regularized_linear_softmax",
            "probe_l2_grid": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0],
            "probe_max_iter": int(probe_max_iter),
            "l2_selection": "dev_macro_f1_then_dev_nll",
            "final_fit": "train_plus_dev_after_l2_selection",
            "test_access": "single_final_evaluation",
            "split_unit": "trajectory_id",
            "split_seed": CANONICAL_SPLIT_SEED,
            "bootstrap_unit": "trajectory_id",
            "bootstrap_draws": int(bootstrap_draws),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "data": {
            "aligned_rows_path": str(aligned),
            "aligned_rows_sha256": file_sha256(aligned),
            "row_count": len(examples),
            "trajectory_count": len(trajectories),
            "latest_result_counts": dict(sorted(Counter(latest_labels.tolist()).items())),
            "cumulative_error_counts": dict(
                sorted(Counter(cumulative_labels.tolist()).items())
            ),
            "split": data_split_report,
        },
        "inputs": {
            "foundation": foundation_report,
            "foundation_checkpoint_path": str(foundation_path),
            "foundation_checkpoint_sha256": file_sha256(foundation_path),
            "skills_path": str(skills),
            "skills_sha256": file_sha256(skills),
            "e1_checkpoints": checkpoint_reports,
        },
        "cache": {
            "state": state_cache.report(),
            "action": action_cache.report(),
            "result": result_cache.report(),
            "matched_history": history_cache.report(),
        },
        "artifacts": {
            "representations_path": str(representation_path),
            "representations_sha256": file_sha256(representation_path),
            "test_predictions_path": str(prediction_path),
            "test_predictions_sha256": file_sha256(prediction_path),
        },
        "metrics": method_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    report_path = output / "state_probe_report.json"
    temporary_report_path = report_path.with_suffix(".json.tmp")
    temporary_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report_path.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return report
