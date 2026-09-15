from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from clstr.full_base_train import STATE_QUERY_ROLE, _batch_cached_or_encode
from clstr.memory_utility_records import canonical_digest, checkpoint_chain_digest
from clstr.qwen_clstr_final_chain import (
    FINAL_CHAIN_SCHEMA_VERSION,
    _validate_reliability,
)
from clstr.qwen_clstr_lineage import QWEN_MODEL_BASENAME, sha256_path
from clstr.stage4_safe_memory import (
    router_state_digest,
    validate_stage4_checkpoint_payload,
)
from clstr.stage_checkpoint_init import (
    build_clstr_model_from_stage0_checkpoint,
    load_head_checkpoint_into_model,
)


FROZEN_MULTIBENCH_PROTOCOL_VERSION = "qwen06_method_multibench_v1"
FROZEN_MULTIBENCH_FORMATTING_VERSION = "qwen06_method_multibench_format_v1"
FROZEN_ROUTE_IDENTITY_SCHEMA_VERSION = "qwen06_clstr_frozen_route_identity_v1"


@dataclass(frozen=True)
class FrozenRouteCorpus:
    benchmark: str
    source_rows: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    manifest: dict[str, Any]
    manifest_sha256: str
    input_identities: dict[str, dict[str, Any]]


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{label} line {line_number} must be a JSON object")
            rows.append(value)
    return rows


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _canonical_route_row(
    row: dict[str, Any],
    *,
    benchmark: str,
    candidate_source: str,
) -> dict[str, Any]:
    row_id = str(row.get("row_id") or "").strip()
    raw_state = str(row.get("raw_state") or "")
    positive_skill_id = str(row.get("positive_skill_id") or "").strip()
    if not row_id or not raw_state.strip() or not positive_skill_id:
        raise ValueError("frozen source rows require row_id, raw_state, and positive_skill_id")
    canonical = {
        "row_id": row_id,
        "raw_state": raw_state,
        "positive_skill_id": positive_skill_id,
    }
    raw_candidates = [str(item) for item in row.get("candidate_skill_ids") or []]
    if len(raw_candidates) != len(set(raw_candidates)):
        raise ValueError("frozen row candidate IDs must be unique")
    if candidate_source != "full_pool":
        canonical["candidate_skill_ids"] = sorted(raw_candidates)
    if "domain" in row or benchmark == "tau2":
        domain = str(row.get("domain") or "").strip()
        if not domain:
            raise ValueError("frozen tau2 rows require a non-empty domain")
        canonical["domain"] = domain
    return canonical


def load_frozen_route_corpus(
    *,
    manifest_path: str | Path,
    source_rows_path: str | Path,
    skills_path: str | Path,
    expected_manifest_sha256: str | None = None,
) -> FrozenRouteCorpus:
    manifest_path = Path(manifest_path).resolve()
    source_rows_path = Path(source_rows_path).resolve()
    skills_path = Path(skills_path).resolve()
    manifest = _read_json_object(manifest_path, label="frozen benchmark manifest")
    recorded_sha256 = str(manifest.get("manifest_sha256") or "")
    if not recorded_sha256:
        raise ValueError("frozen benchmark manifest lacks manifest_sha256")
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_sha256", None)
    actual_manifest_sha256 = canonical_digest(digest_payload)
    if actual_manifest_sha256 != recorded_sha256:
        raise ValueError("frozen benchmark manifest self-hash mismatch")
    if expected_manifest_sha256 is not None and recorded_sha256 != str(expected_manifest_sha256):
        raise ValueError("frozen benchmark manifest identity mismatch")
    if manifest.get("multibench_protocol_version") != FROZEN_MULTIBENCH_PROTOCOL_VERSION:
        raise ValueError("unsupported frozen multibench protocol version")
    if manifest.get("formatting_version") != FROZEN_MULTIBENCH_FORMATTING_VERSION:
        raise ValueError("unsupported frozen multibench formatting version")

    source_rows = _read_jsonl(source_rows_path, label="frozen source rows")
    skills = _read_jsonl(skills_path, label="frozen skills")
    if skills != manifest.get("canonical_skills"):
        raise ValueError("skills do not match frozen manifest")
    if len(source_rows) != int(manifest.get("source_row_count", -1)):
        raise ValueError("frozen source row count mismatch")
    if len(skills) != int(manifest.get("skill_count", -1)):
        raise ValueError("frozen skill count mismatch")
    coverage = manifest.get("positive_coverage") or {}
    if not isinstance(coverage, dict) or coverage.get("status") != "ok":
        raise ValueError("frozen positive coverage is not ok")
    if int(coverage.get("covered_row_count", -1)) != len(source_rows):
        raise ValueError("frozen positive coverage count mismatch")
    if int(coverage.get("source_row_count", -1)) != len(source_rows):
        raise ValueError("frozen positive coverage source count mismatch")
    if coverage.get("missing_positive_row_ids") not in ([], None):
        raise ValueError("frozen corpus contains missing positive rows")

    benchmark = str(manifest.get("benchmark") or "").strip()
    candidate_source = str(
        manifest.get("candidate_source")
        or ("full_pool" if benchmark == "toolbench_g3" else "row_candidates")
    )
    if candidate_source not in {"full_pool", "row_candidates"}:
        raise ValueError("unsupported frozen candidate source")
    canonical_source_rows = manifest.get("canonical_source_rows")
    if not isinstance(canonical_source_rows, list):
        raise ValueError("frozen manifest canonical_source_rows must be a list")
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        row_id = str(row.get("row_id") or "").strip()
        if not row_id or row_id in source_by_id:
            raise ValueError("frozen source rows require unique non-empty row IDs")
        source_by_id[row_id] = row
    validated_source_rows: list[dict[str, Any]] = []
    canonical_ids: set[str] = set()
    for canonical_row in canonical_source_rows:
        if not isinstance(canonical_row, dict):
            raise TypeError("frozen manifest canonical rows must be JSON objects")
        expected = _canonical_route_row(
            canonical_row,
            benchmark=benchmark,
            candidate_source=candidate_source,
        )
        row_id = str(expected["row_id"])
        if row_id in canonical_ids:
            raise ValueError("frozen manifest rows require unique non-empty row IDs")
        canonical_ids.add(row_id)
        actual_row = source_by_id.get(row_id)
        if actual_row is None:
            raise ValueError("source rows do not match frozen manifest")
        actual = _canonical_route_row(
            actual_row,
            benchmark=benchmark,
            candidate_source=candidate_source,
        )
        if actual != expected:
            raise ValueError("source rows do not match frozen manifest")
        validated = {**actual_row, **expected}
        if candidate_source == "full_pool":
            validated.pop("candidate_skill_ids", None)
        validated_source_rows.append(validated)
    if set(source_by_id) != canonical_ids:
        raise ValueError("source rows do not match frozen manifest")
    source_rows = validated_source_rows

    skill_ids = [_skill_id(skill) for skill in skills]
    if any(not skill_id for skill_id in skill_ids) or len(skill_ids) != len(set(skill_ids)):
        raise ValueError("frozen skills must have unique non-empty skill IDs")
    declared = set(skill_ids)
    for row in source_rows:
        target = str(row.get("positive_skill_id") or "").strip()
        if not target or target not in declared:
            raise ValueError("frozen row positive is outside declared skills")
        candidates = [str(item) for item in row.get("candidate_skill_ids") or []]
        if candidates and target not in candidates:
            raise ValueError("frozen row positive is outside declared candidates")
        if any(candidate not in declared for candidate in candidates):
            raise ValueError("frozen row candidate is outside declared skills")

    return FrozenRouteCorpus(
        benchmark=benchmark,
        source_rows=source_rows,
        skills=skills,
        manifest=manifest,
        manifest_sha256=recorded_sha256,
        input_identities={
            "manifest": sha256_path(manifest_path),
            "source_rows": sha256_path(source_rows_path),
            "skills": sha256_path(skills_path),
        },
    )


def adapt_frozen_route_rows(
    *,
    benchmark: str,
    source_rows: list[dict[str, Any]],
    benchmark_skill_ids: list[str],
) -> list[dict[str, Any]]:
    benchmark = str(benchmark or "").strip()
    global_candidates = [str(skill_id) for skill_id in benchmark_skill_ids if str(skill_id)]
    if not benchmark or not global_candidates or len(global_candidates) != len(set(global_candidates)):
        raise ValueError("frozen route adaptation requires a benchmark and unique skill pool")
    declared = set(global_candidates)
    adapted: list[dict[str, Any]] = []
    for row_index, row in enumerate(source_rows):
        row_id = str(row.get("row_id") or f"{benchmark}:{row_index}")
        state_text = str(row.get("raw_state") or "")
        target = str(row.get("positive_skill_id") or "").strip()
        if not state_text.strip():
            raise ValueError("frozen route row is missing raw_state")
        if target not in declared:
            raise ValueError("frozen route target is outside benchmark skills")
        raw_candidates = row.get("candidate_skill_ids")
        if raw_candidates:
            candidates = list(dict.fromkeys(str(item) for item in raw_candidates if str(item)))
        else:
            candidates = global_candidates
        if not candidates or any(candidate not in declared for candidate in candidates):
            raise ValueError("frozen route candidates are outside benchmark skills")
        if target not in candidates:
            raise ValueError("frozen route target is outside legal candidates")
        adapted_row = {
            "row_id": row_id,
            "task_id": str(row.get("task_id") or row_id),
            "trajectory_id": str(
                row.get("trajectory_id") or row.get("gamefile") or row_id
            ),
            "step_index": int(row.get("step_index") or 0),
            "benchmark": benchmark,
            "source_benchmark": benchmark,
            "state_text": state_text,
            "next_skill_id": target,
            "visible_inventory_skill_ids": candidates,
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            "provenance": {
                "source": "qwen06_frozen_multibench_route",
                "original_row": row,
            },
        }
        for field in ("domain", "split", "task_type", "gamefile"):
            if field in row:
                adapted_row[field] = row[field]
        adapted.append(adapted_row)
    return adapted


def merge_frozen_benchmark_skills(
    base_skills: list[dict[str, Any]],
    benchmark_skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill in base_skills:
        skill_id = _skill_id(skill)
        if not skill_id or skill_id in seen:
            raise ValueError("base skills must have unique non-empty skill IDs")
        seen.add(skill_id)
        merged.append(dict(skill))
    known = 0
    appended = 0
    benchmark_seen: set[str] = set()
    for skill in benchmark_skills:
        skill_id = _skill_id(skill)
        if not skill_id or skill_id in benchmark_seen:
            raise ValueError("benchmark skills must have unique non-empty skill IDs")
        benchmark_seen.add(skill_id)
        if skill_id in seen:
            known += 1
            continue
        seen.add(skill_id)
        appended += 1
        merged.append(dict(skill))
    return merged, {
        "base_skill_count": len(base_skills),
        "benchmark_skill_count": len(benchmark_skills),
        "known_benchmark_skill_count": known,
        "appended_benchmark_skill_count": appended,
        "merged_skill_count": len(merged),
    }


def score_frozen_route_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_size = int(batch_size)
    top_k = int(top_k)
    if batch_size <= 0 or top_k <= 0:
        raise ValueError("frozen route batch_size and top_k must be positive")
    if sorted(int(index) for index in skill_id_to_idx.values()) != list(
        range(len(skill_id_to_idx))
    ):
        raise ValueError("skill_id_to_idx must be a contiguous complete mapping")
    predictions: list[dict[str, Any]] = []
    recall1 = 0
    recall5 = 0
    reciprocal_rank = 0.0
    declared_candidate_total = 0
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                if any(row.get("replay_prefix") for row in batch):
                    raise ValueError("frozen comparable routing must not use replay prefixes")
                candidate_ids_by_row: list[list[str]] = []
                candidate_rows: list[list[int]] = []
                for row in batch:
                    candidate_ids = list(
                        dict.fromkeys(
                            str(item)
                            for item in row.get("visible_inventory_skill_ids") or []
                            if str(item)
                        )
                    )
                    target = str(row.get("next_skill_id") or "")
                    if not candidate_ids or target not in candidate_ids:
                        raise ValueError("frozen route row lacks its positive in declared candidates")
                    if any(candidate not in skill_id_to_idx for candidate in candidate_ids):
                        raise ValueError("frozen route candidate is outside the merged skill table")
                    candidate_ids_by_row.append(candidate_ids)
                    candidate_rows.append([skill_id_to_idx[candidate] for candidate in candidate_ids])
                h = _batch_cached_or_encode(
                    model,
                    batch,
                    "_state_embedding",
                    "state_text",
                    device,
                    text_role=STATE_QUERY_ROLE,
                )
                memory = model.initial_belief(h)
                logits = model.unified_route_logits(h, memory, candidate_rows=candidate_rows)
                if logits.ndim != 2 or int(logits.size(0)) != len(batch):
                    raise ValueError("frozen route logits must cover every row")
                for local_idx, (row, candidate_ids) in enumerate(
                    zip(batch, candidate_ids_by_row, strict=True)
                ):
                    width = len(candidate_ids)
                    scores = [
                        float(value)
                        for value in logits[local_idx, :width]
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    ]
                    if any(not math.isfinite(score) for score in scores):
                        raise ValueError("frozen route scores must be finite")
                    order = sorted(range(width), key=lambda idx: (-scores[idx], idx))
                    target = str(row["next_skill_id"])
                    positive_position = candidate_ids.index(target)
                    positive_rank = order.index(positive_position) + 1
                    recall1 += int(positive_rank <= 1)
                    recall5 += int(positive_rank <= 5)
                    reciprocal_rank += 1.0 / positive_rank
                    declared_candidate_total += width
                    selected = order[: min(top_k, width)]
                    prediction = {
                        "row_id": str(row.get("row_id") or ""),
                        "benchmark": str(row.get("benchmark") or ""),
                        "positive_skill_id": target,
                        "positive_rank": positive_rank,
                        "declared_candidate_count": width,
                        "ranked_skill_ids": [candidate_ids[idx] for idx in selected],
                        "ranked_scores": [scores[idx] for idx in selected],
                    }
                    for field in ("domain", "split", "task_type", "gamefile", "step_index"):
                        if field in row:
                            prediction[field] = row[field]
                    predictions.append(prediction)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    denominator = len(rows)
    return predictions, {
        "status": "ok" if len(predictions) == denominator else "action_required",
        "source_rows": denominator,
        "prediction_rows": len(predictions),
        "recall@1": recall1 / max(denominator, 1),
        "recall@5": recall5 / max(denominator, 1),
        "mrr": reciprocal_rank / max(denominator, 1),
        "mean_declared_candidate_count": declared_candidate_total / max(denominator, 1),
        "memory_active_rows": 0,
        "zero_history_fallback": "exact_static",
        "route_scorer": "unified_memory_initial_belief",
        "top_k": top_k,
    }


def _validate_recorded_identity(recorded: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(recorded, dict) or not recorded.get("path") or not recorded.get("sha256"):
        raise ValueError(f"{label} identity is missing")
    current = sha256_path(recorded["path"])
    for key in ("sha256", "size", "file_count"):
        if str(current[key]) != str(recorded.get(key)):
            raise ValueError(f"{label} {key} mismatch")
    return current


def load_final_chain_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _read_json_object(path, label="Qwen CLSTR final-chain manifest")
    recorded_sha256 = str(manifest.get("manifest_sha256") or "")
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_sha256", None)
    if not recorded_sha256 or canonical_digest(digest_payload) != recorded_sha256:
        raise ValueError("Qwen CLSTR final-chain manifest self-hash mismatch")
    if manifest.get("schema_version") != FINAL_CHAIN_SCHEMA_VERSION:
        raise ValueError("unsupported Qwen CLSTR final-chain schema")
    if manifest.get("status") != "ok" or manifest.get("final_checkpoint_role") != "stage4":
        raise ValueError("Qwen CLSTR final chain is not evaluation-ready")
    if manifest.get("route_scorer") != "unified_memory":
        raise ValueError("Qwen CLSTR final chain must use unified_memory")
    backbone = manifest.get("backbone") or {}
    if not isinstance(backbone, dict) or backbone.get("frozen") is not True:
        raise ValueError("Qwen backbone must remain frozen")
    if Path(str(backbone.get("path") or "")).resolve().name != QWEN_MODEL_BASENAME:
        raise ValueError("Qwen backbone identity mismatch")
    _validate_recorded_identity(backbone, label="Qwen backbone")
    checkpoints = manifest.get("checkpoints") or {}
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"stage0", "stage1", "stage2", "stage4"}:
        raise ValueError("Qwen CLSTR final chain checkpoint roles mismatch")
    for role in ("stage0", "stage1", "stage2", "stage4"):
        _validate_recorded_identity(checkpoints[role], label=f"{role} checkpoint")
    actual_chain_digest = checkpoint_chain_digest(
        {
            role: Path(str(checkpoints[role]["path"]))
            for role in ("stage0", "stage1", "stage2", "stage4")
        }
    )
    if actual_chain_digest != str(manifest.get("checkpoint_chain_digest") or ""):
        raise ValueError("Qwen CLSTR checkpoint-chain digest mismatch")
    _validate_recorded_identity(manifest.get("skill_pool"), label="training skill pool")
    _validate_recorded_identity(
        manifest.get("stage4_selection"),
        label="Stage4 selection",
    )
    _validate_reliability(manifest.get("reliability"))
    return manifest


def load_final_chain_model(
    *,
    final_chain: dict[str, Any],
    base_skills_path: str | Path,
    appended_skills: list[dict[str, Any]],
    model_cache_dir: str | Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    checkpoints = final_chain.get("checkpoints") or {}
    stage4_payload = torch.load(
        Path(str(checkpoints["stage4"]["path"])),
        map_location="cpu",
    )
    if not isinstance(stage4_payload, dict):
        raise TypeError("Stage4 checkpoint payload must be a dict")
    stage4_delta_validation = validate_stage4_checkpoint_payload(stage4_payload)
    model, model_config, stage0_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(str(checkpoints["stage0"]["path"])),
        skills_path=Path(base_skills_path),
        model_cache_dir=Path(model_cache_dir),
        allow_skill_table_prefix_expansion=False,
    )
    if model_config.get("freeze_backbone") is not True:
        raise ValueError("loaded Qwen CLSTR model does not freeze its backbone")
    if stage0_report.get("stage0_loaded") is not True:
        raise ValueError("Stage0 checkpoint did not load")
    backbone = getattr(getattr(model, "encoder", None), "backbone", None)
    trainable_backbone_parameters = (
        sum(int(parameter.numel()) for parameter in backbone.parameters() if parameter.requires_grad)
        if backbone is not None
        else 0
    )
    if trainable_backbone_parameters:
        raise ValueError("loaded Qwen CLSTR backbone has trainable parameters")
    stage2_report = load_head_checkpoint_into_model(
        model,
        Path(str(checkpoints["stage2"]["path"])),
        partial_load_mode="stage2_checkpoint_compatible_state_for_frozen_route_eval",
        protect_routing_foundation=True,
        allow_skill_table_prefix_expansion=False,
    )
    if stage2_report.get("loaded") is not True:
        raise ValueError("Stage2 checkpoint overlay did not load")
    if stage2_report.get("shape_mismatched"):
        raise ValueError("Stage2 checkpoint overlay has shape mismatches")
    stage2_router_digest = router_state_digest(model, scope="full")
    stage4_report = load_head_checkpoint_into_model(
        model,
        Path(str(checkpoints["stage4"]["path"])),
        partial_load_mode="stage4_checkpoint_compatible_state_for_frozen_route_eval",
        protect_routing_foundation=True,
        allow_skill_table_prefix_expansion=False,
    )
    if stage4_report.get("loaded") is not True:
        raise ValueError("Stage4 checkpoint overlay did not load")
    if stage4_report.get("shape_mismatched"):
        raise ValueError("Stage4 checkpoint overlay has shape mismatches")
    stage4_router_digest = router_state_digest(model, scope="full")
    if stage4_router_digest != stage2_router_digest:
        raise ValueError("Stage4 overlay changed the immutable Stage2 router")
    if hasattr(model, "to"):
        model.to(device)
    append_fn = getattr(model, "append_skills", None)
    if not callable(append_fn):
        raise ValueError("loaded Qwen CLSTR model cannot append unseen benchmark skills")
    skill_append_report = append_fn(appended_skills)
    if not isinstance(skill_append_report, dict):
        raise TypeError("unseen benchmark skill append report must be a dict")
    expected_appended_ids = [_skill_id(row) for row in appended_skills]
    if list(skill_append_report.get("appended_skill_ids") or []) != expected_appended_ids:
        raise ValueError("unseen benchmark skill append order mismatch")
    if int(skill_append_report.get("appended_count", -1)) != len(expected_appended_ids):
        raise ValueError("unseen benchmark skill append count mismatch")
    if hasattr(model, "eval"):
        model.eval()
    return model, {
        "checkpoint_load_order": ["stage0", "stage2", "stage4"],
        "freeze_backbone": True,
        "trainable_backbone_parameters": trainable_backbone_parameters,
        "stage0": stage0_report,
        "stage2": stage2_report,
        "stage4": stage4_report,
        "stage4_delta_validation": stage4_delta_validation,
        "stage2_router_digest": stage2_router_digest,
        "stage4_router_digest": stage4_router_digest,
        "skill_append": skill_append_report,
    }


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _self_hashed(payload: dict[str, Any], *, field: str) -> dict[str, Any]:
    output = dict(payload)
    output[field] = canonical_digest(output)
    return output


def _load_existing_frozen_route_artifacts(
    *,
    identity_path: Path,
    predictions_path: Path,
    report_path: Path,
    expected_identity: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    paths = (identity_path, predictions_path, report_path)
    existing = [path.is_file() for path in paths]
    if not any(existing):
        return None
    if not all(existing):
        raise ValueError("existing frozen route artifacts are incomplete")
    identity = _read_json_object(identity_path, label="existing evaluation identity")
    recorded_identity_sha256 = str(identity.get("identity_sha256") or "")
    identity_payload = dict(identity)
    identity_payload.pop("identity_sha256", None)
    if not recorded_identity_sha256 or canonical_digest(identity_payload) != recorded_identity_sha256:
        raise ValueError("existing evaluation identity self-hash mismatch")
    if identity != expected_identity:
        raise ValueError("existing evaluation identity does not match this evaluation")

    predictions = _read_jsonl(predictions_path, label="existing frozen route predictions")
    if len(predictions) != len(expected_rows):
        raise ValueError("existing frozen route prediction count mismatch")
    for prediction, row in zip(predictions, expected_rows, strict=True):
        prediction_sha256 = str(prediction.get("prediction_sha256") or "")
        prediction_payload = dict(prediction)
        prediction_payload.pop("prediction_sha256", None)
        if not prediction_sha256 or canonical_digest(prediction_payload) != prediction_sha256:
            raise ValueError("existing frozen route prediction self-hash mismatch")
        if prediction.get("evaluation_identity_sha256") != recorded_identity_sha256:
            raise ValueError("existing prediction identity does not match this evaluation")
        if prediction.get("row_id") != row.get("row_id"):
            raise ValueError("existing prediction row order does not match frozen rows")
        if prediction.get("positive_skill_id") != row.get("next_skill_id"):
            raise ValueError("existing prediction positive does not match frozen rows")

    report = _read_json_object(report_path, label="existing frozen route report")
    report_sha256 = str(report.get("report_sha256") or "")
    report_payload = dict(report)
    report_payload.pop("report_sha256", None)
    if not report_sha256 or canonical_digest(report_payload) != report_sha256:
        raise ValueError("existing frozen route report self-hash mismatch")
    if report.get("evaluation_identity_sha256") != recorded_identity_sha256:
        raise ValueError("existing frozen route report identity mismatch")
    if int(report.get("prediction_rows", -1)) != len(predictions):
        raise ValueError("existing frozen route report prediction count mismatch")
    if report.get("predictions_identity") != sha256_path(predictions_path):
        raise ValueError("existing frozen route predictions file identity mismatch")
    resumed = dict(report)
    resumed["resume"] = {"used": True, "prediction_rows": len(predictions)}
    return resumed


def run_frozen_clstr_route_eval(
    *,
    final_chain_manifest_path: str | Path,
    benchmark_manifest_path: str | Path,
    source_rows_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    top_k: int = 100,
    max_eval_rows: int | None = None,
    device: str | torch.device | None = None,
    expected_benchmark_manifest_sha256: str | None = None,
    expected_final_chain_manifest_sha256: str | None = None,
    expected_checkpoint_chain_digest: str | None = None,
) -> dict[str, Any]:
    final_chain = load_final_chain_manifest(final_chain_manifest_path)
    if (
        expected_final_chain_manifest_sha256 is not None
        and final_chain["manifest_sha256"] != str(expected_final_chain_manifest_sha256)
    ):
        raise ValueError("final-chain manifest identity mismatch")
    if (
        expected_checkpoint_chain_digest is not None
        and final_chain["checkpoint_chain_digest"] != str(expected_checkpoint_chain_digest)
    ):
        raise ValueError("checkpoint-chain identity mismatch")
    corpus = load_frozen_route_corpus(
        manifest_path=benchmark_manifest_path,
        source_rows_path=source_rows_path,
        skills_path=skills_path,
        expected_manifest_sha256=expected_benchmark_manifest_sha256,
    )
    if max_eval_rows is not None and (isinstance(max_eval_rows, bool) or int(max_eval_rows) <= 0):
        raise ValueError("max_eval_rows must be a positive integer")
    source_rows = list(corpus.source_rows)
    evaluated_source_rows = source_rows if max_eval_rows is None else source_rows[: int(max_eval_rows)]
    benchmark_skill_ids = [_skill_id(row) for row in corpus.skills]
    adapted_rows = adapt_frozen_route_rows(
        benchmark=corpus.benchmark,
        source_rows=evaluated_source_rows,
        benchmark_skill_ids=benchmark_skill_ids,
    )

    base_skills_path = Path(str(final_chain["skill_pool"]["path"])).resolve()
    base_skills = _read_jsonl(base_skills_path, label="training skill pool")
    merged_skills, skill_merge = merge_frozen_benchmark_skills(base_skills, corpus.skills)
    base_skill_ids = {_skill_id(row) for row in base_skills}
    appended_skills = [dict(row) for row in corpus.skills if _skill_id(row) not in base_skill_ids]
    merged_skill_ids = [_skill_id(row) for row in merged_skills]
    skill_id_to_idx = {skill_id: index for index, skill_id in enumerate(merged_skill_ids)}

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    identity_payload = {
        "schema_version": FROZEN_ROUTE_IDENTITY_SCHEMA_VERSION,
        "protocol_version": FROZEN_MULTIBENCH_PROTOCOL_VERSION,
        "formatting_version": FROZEN_MULTIBENCH_FORMATTING_VERSION,
        "benchmark": corpus.benchmark,
        "split": corpus.manifest.get("split"),
        "final_chain_manifest_sha256": final_chain["manifest_sha256"],
        "checkpoint_chain_digest": final_chain["checkpoint_chain_digest"],
        "benchmark_manifest_sha256": corpus.manifest_sha256,
        "frozen_inputs": corpus.input_identities,
        "training_skill_pool": final_chain["skill_pool"],
        "merged_skill_ids_sha256": canonical_digest(merged_skill_ids),
        "source_row_ids_sha256": canonical_digest([row["row_id"] for row in evaluated_source_rows]),
        "source_rows": len(source_rows),
        "evaluated_rows": len(evaluated_source_rows),
        "max_eval_rows": max_eval_rows,
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "device": str(resolved_device),
        "route_scorer": "unified_memory_initial_belief",
        "memory_active_rows": 0,
    }
    evaluation_identity = _self_hashed(identity_payload, field="identity_sha256")

    output_dir = Path(output_dir).resolve()
    identity_path = output_dir / "evaluation_identity.json"
    predictions_path = output_dir / "frozen_route_predictions.jsonl"
    report_path = output_dir / "frozen_route_eval_report.json"
    resumed = _load_existing_frozen_route_artifacts(
        identity_path=identity_path,
        predictions_path=predictions_path,
        report_path=report_path,
        expected_identity=evaluation_identity,
        expected_rows=adapted_rows,
    )
    if resumed is not None:
        return resumed

    model_cache_dir = output_dir / "model_cache"
    merged_skills_path = model_cache_dir / "merged_skills.jsonl"
    _atomic_write_jsonl(merged_skills_path, merged_skills)
    model, checkpoint_load = load_final_chain_model(
        final_chain=final_chain,
        base_skills_path=base_skills_path,
        appended_skills=appended_skills,
        model_cache_dir=model_cache_dir,
        device=resolved_device,
    )
    predictions, routing_metrics = score_frozen_route_rows(
        model,
        adapted_rows,
        skill_id_to_idx=skill_id_to_idx,
        batch_size=batch_size,
        top_k=top_k,
        device=resolved_device,
    )
    identity_sha256 = str(evaluation_identity["identity_sha256"])
    complete_predictions: list[dict[str, Any]] = []
    for prediction, row in zip(predictions, adapted_rows, strict=True):
        raw_state = str((row.get("provenance") or {}).get("original_row", {}).get("raw_state") or "")
        payload = {
            **prediction,
            "row_key": canonical_digest(
                {
                    "evaluation_identity_sha256": identity_sha256,
                    "row_id": row["row_id"],
                    "raw_state": raw_state,
                    "positive_skill_id": row["next_skill_id"],
                    "candidate_skill_ids": row["visible_inventory_skill_ids"],
                }
            ),
            "evaluation_identity_sha256": identity_sha256,
            "benchmark_manifest_sha256": corpus.manifest_sha256,
        }
        complete_predictions.append(_self_hashed(payload, field="prediction_sha256"))
    _atomic_write_jsonl(predictions_path, complete_predictions)
    _atomic_write_json(identity_path, evaluation_identity)
    report_payload = {
        "status": "ok" if len(complete_predictions) == len(adapted_rows) else "action_required",
        "blockers": [] if len(complete_predictions) == len(adapted_rows) else ["incomplete_predictions"],
        "benchmark": corpus.benchmark,
        "result_label": "frozen",
        "metric_scope": "next_tool_action_routing",
        "task_success": False,
        "source_rows": len(source_rows),
        "evaluated_rows": len(adapted_rows),
        "prediction_rows": len(complete_predictions),
        "memory_active_rows": 0,
        "zero_history_fallback": "exact_static",
        "evaluation_identity_sha256": identity_sha256,
        "checkpoint_chain_digest": final_chain["checkpoint_chain_digest"],
        "benchmark_manifest_sha256": corpus.manifest_sha256,
        "skill_merge": skill_merge,
        "checkpoint_load": checkpoint_load,
        "routing_metrics": routing_metrics,
        "identity_path": str(identity_path),
        "predictions_path": str(predictions_path),
        "predictions_identity": sha256_path(predictions_path),
        "resume": {"used": False, "prediction_rows": 0},
        "config": {
            "batch_size": int(batch_size),
            "top_k": int(top_k),
            "max_eval_rows": max_eval_rows,
            "device": str(resolved_device),
        },
    }
    report = _self_hashed(report_payload, field="report_sha256")
    _atomic_write_json(report_path, report)
    return report
