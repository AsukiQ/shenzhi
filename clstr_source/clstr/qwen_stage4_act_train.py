from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from clstr.external_data import write_json
from clstr.full_base_train import UNIFIED_MEMORY_ROUTE_SCORER
from clstr.qwen_checkpoint_init import load_routing_and_head_checkpoints
from clstr.qwen_external_encoder import build_qwen_external_clstr_model, write_qwen_init_manifest
from clstr.qwen_full_base_train import _qwen_hardware_report, write_qwen_blocker_report
from clstr.stage4_act_train import _read_jsonl, train_stage4_act_with_model
from clstr.training_monitor import append_setup_status, reset_setup_status


def run_clstr_qwen3_stage4_act_train(
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    qwen_model_path: str | Path = "Qwen/Qwen3-8B",
    cache_dir: str | Path | None = None,
    routing_checkpoint_path: str | Path | None = None,
    head_checkpoint_path: str | Path | None = None,
    max_steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    candidate_count: int | None = 64,
    train_transition: bool = False,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    max_length: int | None = 4096,
    torch_dtype: str = "bfloat16",
    local_files_only: bool = False,
    model_dim: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_status_path = output_dir / "setup_status.jsonl"
    reset_setup_status(setup_status_path)
    append_setup_status(
        setup_status_path,
        "qwen_stage4_started",
        trajectories_path=str(trajectories_path),
        skills_path=str(skills_path),
        qwen_model_path=str(qwen_model_path),
    )
    command = (
        "python scripts/run_clstr_qwen3_stage4_act_train.py "
        f"--trajectories_path {trajectories_path} --skills_path {skills_path} --output_dir {output_dir} "
        f"--qwen_model_path {qwen_model_path} --max_steps {max_steps} --batch_size {batch_size}"
    )
    if routing_checkpoint_path is not None:
        command += f" --routing_checkpoint_path {routing_checkpoint_path}"
    if head_checkpoint_path is not None:
        command += f" --head_checkpoint_path {head_checkpoint_path}"
    if allowed_benchmarks:
        command += f" --allowed_benchmarks {','.join(sorted(str(item) for item in allowed_benchmarks))}"

    try:
        if routing_checkpoint_path is None:
            raise FileNotFoundError("routing checkpoint not found: <not provided>")
        if not Path(routing_checkpoint_path).exists():
            raise FileNotFoundError(f"routing checkpoint not found: {routing_checkpoint_path}")
        if head_checkpoint_path is None:
            raise FileNotFoundError("head checkpoint not found: <not provided>")
        if not Path(head_checkpoint_path).exists():
            raise FileNotFoundError(f"head checkpoint not found: {head_checkpoint_path}")
        append_setup_status(
            setup_status_path,
            "qwen_stage4_checkpoint_preflight_ok",
            routing_checkpoint_path=str(routing_checkpoint_path),
            head_checkpoint_path=str(head_checkpoint_path),
        )

        skills = _read_jsonl(skills_path)
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
        checkpoint_init_report = load_routing_and_head_checkpoints(
            model,
            routing_checkpoint_path=routing_checkpoint_path,
            head_checkpoint_path=head_checkpoint_path,
            partial_load_mode="stage1_routing_plus_stage2_heads_strict_false",
            protect_routing_foundation=True,
        )
        append_setup_status(
            setup_status_path,
            "stage4_checkpoint_init_loaded",
            checkpoint_init_report=checkpoint_init_report,
        )
        init_dir = output_dir.parent / "clstr_qwen3_8b_init"
        write_qwen_init_manifest(
            output_dir=init_dir,
            model_name_or_path=str(qwen_model_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            quantization_mode="none",
        )
        write_json(
            init_dir / "stage4_checkpoint_init.json",
            checkpoint_init_report,
        )
        write_json(
            init_dir / "hardware_report.json",
            _qwen_hardware_report(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                str(qwen_model_path),
                str(cache_dir) if cache_dir is not None else None,
            ),
        )
        report = train_stage4_act_with_model(
            model=model,
            trajectories_path=trajectories_path,
            skills_path=skills_path,
            output_dir=output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            candidate_count=candidate_count,
            train_transition=train_transition,
            max_rows=max_rows,
            allowed_benchmarks=allowed_benchmarks,
            route_scorer=UNIFIED_MEMORY_ROUTE_SCORER,
            checkpoint_init_report=checkpoint_init_report,
            setup_status_path=setup_status_path,
        )
        report["checkpoint_init"] = checkpoint_init_report
        report["routing_init"] = routing_report
        report["model_config"] = model_config
        write_json(output_dir / "stage4_qwen_train_report.json", report)
        return report
    except BaseException as exc:
        append_setup_status(setup_status_path, "qwen_stage4_blocked", error=str(exc), error_type=type(exc).__name__)
        blocker = write_qwen_blocker_report(
            output_dir=output_dir,
            stage="clstr_qwen3_stage4_act_train",
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
