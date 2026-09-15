from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clstr.history_channel import compact_causal_state_from_row
from clstr.vnext_stage0_train import _build_model
from clstr.vnext_training import (
    capture_vnext_source_manifest,
    load_or_capture_frozen_backbone_snapshot,
    load_or_build_frozen_text_cache,
    load_or_build_skill_embedding_cache,
    persist_vnext_source_manifest,
    read_jsonl,
    require_verified_data_contract,
    seed_vnext_run,
    write_json,
)


def precompute_vnext_assets(
    *,
    skills_path: str | Path,
    retrieval_rows_path: str | Path,
    retrieval_dev_rows_path: str | Path,
    static_route_rows_path: str | Path,
    static_route_dev_rows_path: str | Path,
    trajectory_rows_path: str | Path,
    trajectory_dev_rows_path: str | Path,
    causal_pair_support_rows_path: str | Path,
    causal_pair_support_dev_rows_path: str | Path,
    data_contract_path: str | Path,
    output_dir: str | Path,
    cache_root: str | Path,
    model_name_or_path: str,
    model_dim: int = 1024,
    max_length: int = 2048,
    torch_dtype: str = "bfloat16",
    skill_table_batch_size: int = 32,
    belief_top_k: int = 64,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    skill_cache_shard_size: int = 2048,
    seed: int = 17,
    local_files_only: bool = True,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reproducibility = seed_vnext_run(seed)
    source_manifest = capture_vnext_source_manifest(
        Path(__file__).resolve().parents[1],
        (
            "clstr/model.py",
            "clstr/belief.py",
            "clstr/history_channel.py",
            "clstr/vnext_core.py",
            "clstr/vnext_training.py",
            "clstr/vnext_stage0_train.py",
            "clstr/vnext_precompute.py",
            "scripts/run_clstr_vnext_precompute.py",
            "scripts/resolve_clstr_vnext_full_inputs.py",
            "scripts/sbatch/run_clstr_vnext_full_precompute.sh",
        ),
        require_clean=bool(require_clean_source),
    )
    source_manifest_record = persist_vnext_source_manifest(output_dir, source_manifest)
    data_contract = require_verified_data_contract(
        data_contract_path,
        {
            "training_skills": skills_path,
            "retrieval_rows": retrieval_rows_path,
            "retrieval_dev_rows": retrieval_dev_rows_path,
            "static_route_rows": static_route_rows_path,
            "static_route_dev_rows": static_route_dev_rows_path,
            "trajectory_rows": trajectory_rows_path,
            "trajectory_dev_rows": trajectory_dev_rows_path,
            "causal_pair_support_rows": causal_pair_support_rows_path,
            "causal_pair_support_dev_rows": causal_pair_support_dev_rows_path,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("vNext precompute requires a CUDA Slurm node")
    backbone_snapshot = load_or_capture_frozen_backbone_snapshot(
        model_name_or_path,
        manifest_path=output_dir / "backbone_snapshot.json",
    )
    skills = read_jsonl(skills_path)
    model, model_config = _build_model(
        skills,
        model_name_or_path=model_name_or_path,
        model_dim=model_dim,
        max_length=max_length,
        torch_dtype=torch_dtype,
        skill_table_batch_size=skill_table_batch_size,
        belief_top_k=belief_top_k,
        local_files_only=local_files_only,
        frozen_backbone_snapshot_digest=backbone_snapshot["contract_digest"],
        frozen_backbone_snapshot_manifest_path=backbone_snapshot["path"],
    )
    model.to(device)
    model.eval()
    skill_cache = load_or_build_skill_embedding_cache(
        model,
        skills,
        cache_root=cache_root,
        batch_size=skill_table_batch_size,
        cache_shard_size=skill_cache_shard_size,
    )
    state_texts: list[str] = []
    action_texts: list[str] = []
    result_texts: list[str] = []
    for path in (
        retrieval_rows_path,
        retrieval_dev_rows_path,
    ):
        for row in read_jsonl(path):
            causal = compact_causal_state_from_row(row)
            current = str(row.get("state_text_current") or "").strip()
            if not causal or not current:
                raise ValueError("vNext retrieval precompute requires causal/current channels")
            state_texts.extend((causal, current))
    for path in (static_route_rows_path, static_route_dev_rows_path):
        for row in read_jsonl(path):
            current = str(row.get("state_text_current") or "").strip()
            causal = compact_causal_state_from_row(row)
            if not current:
                raise ValueError("vNext static-route precompute requires current state")
            state_texts.extend((current, causal))
    for path in (
        trajectory_rows_path,
        trajectory_dev_rows_path,
        causal_pair_support_rows_path,
        causal_pair_support_dev_rows_path,
    ):
        rows = read_jsonl(path)
        for row in rows:
            current = str(row.get("state_text_current") or "").strip()
            causal = compact_causal_state_from_row(row)
            if not current or not causal:
                raise ValueError("vNext trajectory precompute requires causal/current channels")
            state_texts.extend((current, causal))
        action_texts.extend(str(row.get("action_text") or "") for row in rows)
        result_texts.extend(
            str(row.get("actual_result_text") or "")
            for row in rows
            if str(row.get("actual_result_text") or "")
        )
    state_cache = load_or_build_frozen_text_cache(
        model,
        state_texts,
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    action_cache = load_or_build_frozen_text_cache(
        model,
        action_texts,
        role="action",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    result_cache = (
        load_or_build_frozen_text_cache(
            model,
            result_texts,
            role="result",
            batch_size=cache_batch_size,
            cache_root=cache_root,
            cache_shard_size=cache_shard_size,
        )
        if result_texts
        else None
    )
    report = {
        "status": "ok",
        "stage": "clstr_vnext_precompute",
        "model_config": model_config,
        "data_contract": data_contract,
        "source_manifest": source_manifest_record,
        "frozen_backbone_snapshot": backbone_snapshot,
        "reproducibility": reproducibility,
        "skill_cache": skill_cache,
        "state_cache": state_cache.report(),
        "action_cache": action_cache.report(),
        "result_cache": None if result_cache is None else result_cache.report(),
    }
    write_json(output_dir / "precompute_report.json", report)
    return report
