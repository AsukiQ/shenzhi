from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from clstr.external_data import write_json
from clstr.full_base_train import _read_jsonl, train_clstr_full_base_with_model
from clstr.qwen_external_encoder import build_qwen_external_clstr_model, write_qwen_init_manifest
from clstr.training_monitor import append_setup_status, reset_setup_status


def _qwen_hardware_report(device: torch.device, model_name_or_path: str, cache_dir: str | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "model_name_or_path": str(model_name_or_path),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "gpu_name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 3),
            "current_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
            "current_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
        }
    return report


def write_qwen_blocker_report(
    output_dir: str | Path,
    stage: str,
    error: BaseException | str,
    command: str,
    model_name_or_path: str = "Qwen/Qwen3-8B",
    cache_dir: str | Path | None = None,
    setup_status_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "blocked",
        "stage": str(stage),
        "model_name_or_path": str(model_name_or_path),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "Blocked",
        "error": str(error),
        "repro_command": command,
        "network_turbo_reference": "https://www.autodl.com/docs/network_turbo/",
        "valid_or_test_used_for_training": False,
        "qwen_frozen": True,
        "qwen_direct_generator": False,
    }
    if setup_status_path is not None:
        report["setup_status_path"] = str(setup_status_path)
    write_json(output_dir / "blocker_report.json", report)
    return report


def run_clstr_qwen3_full_base_train(
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    qwen_model_path: str | Path = "Qwen/Qwen3-8B",
    cache_dir: str | Path | None = None,
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    max_length: int | None = 4096,
    torch_dtype: str = "bfloat16",
    local_files_only: bool = False,
    model_dim: int | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    include_available_actions_in_state: bool = False,
    routing_checkpoint_path: str | Path | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    stage0_top_m: int | None = None,
    stage0_positive_missing_policy: str = "skip",
    allow_full_pool_stage2_debug: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_status_path = output_dir / "setup_status.jsonl"
    reset_setup_status(setup_status_path)
    append_setup_status(
        setup_status_path,
        "qwen_stage2_started",
        train_path=str(train_path),
        skills_path=str(skills_path),
        qwen_model_path=str(qwen_model_path),
    )
    skills = _read_jsonl(skills_path)
    command = (
        "python scripts/run_clstr_qwen3_full_base_train.py "
        f"--train_path {train_path} --skills_path {skills_path} --output_dir {output_dir} "
        f"--qwen_model_path {qwen_model_path} --max_steps {max_steps} --batch_size {batch_size}"
    )
    if routing_checkpoint_path is not None:
        command += f" --routing_checkpoint_path {routing_checkpoint_path}"
    command += f" --embedding_cache_mode {embedding_cache_mode} --embedding_cache_max_rows {embedding_cache_max_rows}"
    if stage0_top_m is not None:
        command += f" --stage0_top_m {stage0_top_m} --stage0_positive_missing_policy {stage0_positive_missing_policy}"
    if allow_full_pool_stage2_debug:
        command += " --allow_full_pool_stage2_debug"
    if allowed_benchmarks:
        command += f" --allowed_benchmarks {','.join(sorted(str(item) for item in allowed_benchmarks))}"
    try:
        if routing_checkpoint_path is not None and not Path(routing_checkpoint_path).exists():
            raise FileNotFoundError(f"routing checkpoint not found: {routing_checkpoint_path}")
        append_setup_status(
            setup_status_path,
            "qwen_stage2_checkpoint_preflight_ok",
            routing_checkpoint_path=str(routing_checkpoint_path) if routing_checkpoint_path is not None else None,
        )
        append_setup_status(setup_status_path, "qwen_model_load_started", qwen_model_path=str(qwen_model_path))
        model, model_config, routing_report = build_qwen_external_clstr_model(
            skills,
            model_name_or_path=str(qwen_model_path),
            model_dim=model_dim,
            max_length=max_length,
            torch_dtype=torch_dtype,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        append_setup_status(
            setup_status_path,
            "qwen_model_loaded",
            model_dim=model_config.get("d"),
            qwen_external_encoder=bool(routing_report.get("qwen_external_encoder", False)),
        )
        write_qwen_init_manifest(
            output_dir=output_dir.parent / "clstr_qwen3_8b_init",
            model_name_or_path=str(qwen_model_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            quantization_mode="none",
        )
        write_json(
            output_dir.parent / "clstr_qwen3_8b_init" / "hardware_report.json",
            _qwen_hardware_report(torch.device("cuda" if torch.cuda.is_available() else "cpu"), str(qwen_model_path), str(cache_dir) if cache_dir is not None else None),
        )
        return train_clstr_full_base_with_model(
            model=model,
            model_config=model_config,
            routing_report=routing_report,
            train_path=train_path,
            skills_path=skills_path,
            output_dir=output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            loss_weights=loss_weights,
            retrieval_num_negatives=retrieval_num_negatives,
            retrieval_hard_ratio=retrieval_hard_ratio,
            include_available_actions_in_state=include_available_actions_in_state,
            routing_checkpoint_path=routing_checkpoint_path,
            embedding_cache_mode=embedding_cache_mode,
            embedding_cache_max_rows=embedding_cache_max_rows,
            allowed_benchmarks=allowed_benchmarks,
            stage0_top_m=stage0_top_m,
            stage0_positive_missing_policy=stage0_positive_missing_policy,
            allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
            setup_status_path=setup_status_path,
        )
    except BaseException as exc:
        append_setup_status(setup_status_path, "qwen_stage2_blocked", error=str(exc), error_type=type(exc).__name__)
        blocker = write_qwen_blocker_report(
            output_dir=output_dir,
            stage="clstr_qwen3_full_base_train",
            error=exc,
            command=command,
            model_name_or_path=str(qwen_model_path),
            cache_dir=cache_dir,
            setup_status_path=setup_status_path,
        )
        write_qwen_init_manifest(
            output_dir=output_dir.parent / "clstr_qwen3_8b_init",
            model_name_or_path=str(qwen_model_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            status="blocked",
            error=str(exc),
        )
        return blocker


def write_qwen_comparison_reports(
    output_root: str | Path = "outputs",
    output_table_path: str | Path = "outputs/clstr_qwen3_8b_comparison_table.md",
    output_summary_path: str | Path = "outputs/clstr_qwen3_8b_comparison_summary.json",
    output_paper_md: str | Path = "outputs/clstr_qwen3_8b_paper_report.md",
    output_paper_json: str | Path = "outputs/clstr_qwen3_8b_paper_report.json",
) -> dict[str, Any]:
    output_root = Path(output_root)
    eval_root = output_root / "alfworld_eval"
    methods = [
        "clstr_routing_init_baseline",
        "clstr_controller_gate_valid_seen",
        "clstr_controller_gate_full_2k",
        "qwen3_8b_direct_valid_seen",
        "qwen3_8b_chat_available_valid_seen_20",
        "qwen3_8b_chat_available_valid_seen_50",
        "qwen3_8b_chat_available_full",
        "qwen3_8b_likelihood_valid_seen",
        "qwen3_8b_likelihood_full",
        "clstr_qwen3_8b_frozen_smoke_valid_seen",
        "clstr_qwen3_8b_gate_valid_seen",
        "clstr_qwen3_8b_available_actions_gate_valid_seen_20",
        "clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20",
        "clstr_qwen3_8b_full",
    ]
    rows: list[dict[str, Any]] = []
    for method in methods:
        metrics_path = eval_root / method / "metrics.json"
        blocker_path = eval_root / method / "blocker_report.json"
        is_direct_qwen_method = method.startswith("qwen3_8b_")
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            blocker_data = json.loads(blocker_path.read_text(encoding="utf-8")) if blocker_path.exists() else {}
            qwen_direct_baseline = bool(data.get("qwen_direct_baseline", False)) or is_direct_qwen_method
            rows.append(
                {
                    "method": method,
                    "status": blocker_data.get("status", data.get("status", "ok")),
                    "is_clstr": not bool(data.get("not_clstr_result", False)) and not qwen_direct_baseline,
                    "success_rate": data.get("success_rate", 0.0),
                    "average_reward": data.get("average_reward", 0.0),
                    "average_goal_condition_points": data.get("average_goal_condition_points", 0.0),
                    "average_episode_steps": data.get("average_episode_steps", 0.0),
                    "episodes": data.get("episodes", data.get("episode_count", 0)),
                    "qwen_external_encoder": bool(data.get("qwen_external_encoder", False)),
                    "qwen_direct_baseline": qwen_direct_baseline,
                    "clstr_native_act": bool(data.get("clstr_native_act", False)),
                    "path": str(metrics_path),
                    "blocker_path": str(blocker_path) if blocker_path.exists() else None,
                }
            )
        elif blocker_path.exists():
            data = json.loads(blocker_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "method": method,
                    "status": "blocked",
                    "is_clstr": not is_direct_qwen_method,
                    "success_rate": None,
                    "average_reward": None,
                    "average_goal_condition_points": None,
                    "average_episode_steps": None,
                    "episodes": 0,
                    "qwen_external_encoder": "qwen3_8b" in method,
                    "qwen_direct_baseline": is_direct_qwen_method,
                    "clstr_native_act": method.startswith("clstr_qwen3_8b"),
                    "path": str(blocker_path),
                    "blocker_path": str(blocker_path),
                    "error": data.get("error"),
                }
            )
    lines = [
        "| method | status | CLSTR result | Qwen external encoder | direct Qwen baseline | CLSTR-native ACT | success_rate | avg_reward | avg_gcp | avg_steps | episodes | path | blocker |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value is None:
                return "blocked"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            f"| {row['method']} | {row['status']} | {row['is_clstr']} | {row['qwen_external_encoder']} | "
            f"{row['qwen_direct_baseline']} | {row['clstr_native_act']} | {fmt(row['success_rate'])} | "
            f"{fmt(row['average_reward'])} | {fmt(row['average_goal_condition_points'])} | "
            f"{fmt(row['average_episode_steps'])} | {row['episodes']} | {row['path']} | {row.get('blocker_path') or ''} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Qwen3-8B direct baseline is a reference line, not a CLSTR result.",
            "- CLSTR-Qwen3-8B keeps Qwen frozen as an external grounding encoder.",
            "- CLSTR-native ACT means admissible action ranking is trained through CLSTR heads/controller.",
            "- Offline diagnostics are not closed-loop success.",
            "- ALFWorld valid_seen/valid_unseen are evaluation only.",
        ]
    )
    Path(output_table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_table_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "ok",
        "rows": rows,
        "caveats": [
            "Qwen3-8B direct baseline is not a CLSTR result.",
            "Qwen3-8B is frozen in CLSTR-Qwen3-8B main experiments.",
            "No ALFWorld valid/test data is used for training.",
        ],
    }
    write_json(Path(output_summary_path), summary)
    paper = {
        "status": "ok",
        "summary": summary,
        "method_notes": {
            "qwen_direct_baseline": "reference only, not CLSTR",
            "clstr_qwen3_8b": "frozen external encoder plus trainable CLSTR-native ACT/transition/belief/STOP/controller",
            "l_trans_skill_ce": "supervised transition-prior next-skill CE; retained as CLSTR-act implementation detail",
        },
    }
    write_json(Path(output_paper_json), paper)
    Path(output_paper_md).write_text(
        "# CLSTR-Qwen3-8B Report\n\n"
        + "\n".join(lines)
        + "\n\nQwen3-8B direct baseline is reported only as a reference. "
        "The CLSTR result keeps Qwen frozen and uses CLSTR-native heads/controller for admissible action ranking.\n",
        encoding="utf-8",
    )
    return summary
