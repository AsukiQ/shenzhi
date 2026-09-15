from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.memory_utility_gate import (
    AnchoredMemoryUtilityGate,
    MEMORY_UTILITY_FEATURE_NAMES,
    RELIABILITY_MODES,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    MemoryUtilityGate,
    effective_memory_alpha,
    fuse_route_scores,
    positive_rank_and_utility,
)
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        temporary.unlink(missing_ok=True)


def _require_self_hash(payload: dict[str, Any], *, label: str) -> None:
    copy = dict(payload)
    recorded = str(copy.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(copy):
        raise ValueError(f"{label} self-hash mismatch")


def _record_digest(rows: list[dict[str, Any]]) -> str:
    return canonical_digest([str(row.get("row_digest") or "") for row in rows])


def _trajectory_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark')}::{row.get('trajectory_id')}"


def _gate_eligible_record(row: dict[str, Any]) -> bool:
    valid = [bool(value) for value in row.get("valid_mask") or []]
    positive = [bool(value) for value in row.get("positive_mask") or []]
    if len(valid) != len(positive):
        raise ValueError("route-record valid and positive masks differ in width")
    has_positive = any(keep and label for keep, label in zip(valid, positive))
    has_negative = any(keep and not label for keep, label in zip(valid, positive))
    return (
        has_positive
        and has_negative
        and float(row.get("causal_update_count") or 0.0) > 0.0
    )


def _transformed_features(
    rows: list[dict[str, Any]],
    *,
    update_cap: float,
    candidate_cap: float,
) -> torch.Tensor:
    features = []
    for row in rows:
        values = [float(value) for value in row.get("features") or []]
        if len(values) != len(MEMORY_UTILITY_FEATURE_NAMES):
            raise ValueError("route-record features do not match feature schema")
        count = max(0.0, float(row.get("causal_update_count") or 0.0))
        valid_count = sum(bool(value) for value in row.get("valid_mask") or [])
        values[MEMORY_UTILITY_FEATURE_NAMES.index("log_normalized_causal_update_count")] = (
            math.log1p(min(count, update_cap)) / math.log1p(update_cap)
        )
        values[MEMORY_UTILITY_FEATURE_NAMES.index("log_normalized_valid_candidate_count")] = (
            math.log1p(min(valid_count, candidate_cap)) / math.log1p(candidate_cap)
        )
        features.append(values)
    output = torch.tensor(features, dtype=torch.float32)
    if not torch.isfinite(output).all():
        raise ValueError("memory utility gate features must be finite")
    return output


def _route_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    width = max(len(row["static_logits"]) for row in rows)
    floor = torch.finfo(torch.float32).min
    static = torch.full((len(rows), width), floor)
    dynamic = torch.full_like(static, floor)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    positive = torch.zeros_like(valid)
    counts = torch.zeros(len(rows), dtype=torch.float32)
    for index, row in enumerate(rows):
        row_width = len(row["static_logits"])
        if len(row["dynamic_logits"]) != row_width:
            raise ValueError("static and dynamic route-record widths differ")
        static[index, :row_width] = torch.tensor(row["static_logits"])
        dynamic[index, :row_width] = torch.tensor(row["dynamic_logits"])
        valid[index, :row_width] = torch.tensor(row["valid_mask"], dtype=torch.bool)
        positive[index, :row_width] = torch.tensor(
            row["positive_mask"], dtype=torch.bool
        )
        counts[index] = float(row.get("causal_update_count") or 0.0)
    return static, dynamic, valid, positive, counts


def _positive_log_probability(
    logits: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
) -> torch.Tensor:
    _ranks, utility, eligible = positive_rank_and_utility(
        logits,
        positive,
        valid,
    )
    if not bool(eligible.all()):
        raise ValueError("direct utility calibration requires a positive on every row")
    return utility


def direct_utility_targets(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("direct utility temperature must be finite and positive")
    static_utility = _positive_log_probability(static, valid, positive)
    dynamic_utility = _positive_log_probability(dynamic, valid, positive)
    delta = (dynamic_utility - static_utility).detach()
    target = torch.sigmoid(delta / scale)
    weight = (delta.abs() / scale).clamp(0.0, 1.0)
    return target.detach(), weight.detach(), delta


def direct_harm_targets(
    static: torch.Tensor,
    fixed: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("direct harm temperature must be finite and positive")
    static_utility = _positive_log_probability(static, valid, positive)
    fixed_utility = _positive_log_probability(fixed, valid, positive)
    delta = (static_utility - fixed_utility).detach()
    target = torch.sigmoid(delta / scale)
    weight = (delta.abs() / scale).clamp(0.0, 1.0)
    return target.detach(), weight.detach(), delta


def _source_balanced_mean(
    values: torch.Tensor,
    sources: list[str],
) -> torch.Tensor:
    if values.ndim != 1 or int(values.numel()) != len(sources) or not sources:
        raise ValueError("source-balanced values must align with nonempty sources")
    means = []
    for source in sorted(set(sources)):
        mask = torch.tensor(
            [row_source == source for row_source in sources],
            dtype=torch.bool,
            device=values.device,
        )
        means.append(values[mask].mean())
    return torch.stack(means).mean()


def _source_sign_balanced_mean(
    values: torch.Tensor,
    sources: list[str],
    harm_sign: torch.Tensor,
    informative: torch.Tensor,
) -> torch.Tensor:
    if (
        values.ndim != 1
        or int(values.numel()) != len(sources)
        or harm_sign.shape != values.shape
        or informative.shape != values.shape
        or not sources
    ):
        raise ValueError("source/sign-balanced values must align with rows")
    means = []
    for source in sorted(set(sources)):
        source_mask = torch.tensor(
            [row_source == source for row_source in sources],
            dtype=torch.bool,
            device=values.device,
        )
        for sign in (False, True):
            mask = source_mask & informative & (harm_sign == sign)
            if bool(mask.any()):
                means.append(values[mask].mean())
    if not means:
        return values.sum() * 0.0
    return torch.stack(means).mean()


def _ranking_summary(
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
) -> dict[str, Any]:
    ranks, _utility, eligible = positive_rank_and_utility(logits, positive, valid)
    by_benchmark: dict[str, list[float]] = {}
    for row, rank, is_eligible in zip(rows, ranks.tolist(), eligible.tolist()):
        value = 1.0 / int(rank) if is_eligible and int(rank) > 0 else 0.0
        by_benchmark.setdefault(str(row["benchmark"]), []).append(value)
    benchmark_summary = {
        benchmark: {"mrr": sum(values) / len(values), "row_count": len(values)}
        for benchmark, values in sorted(by_benchmark.items())
    }
    return {
        "balanced_macro_mrr": sum(
            item["mrr"] for item in benchmark_summary.values()
        )
        / max(1, len(benchmark_summary)),
        "by_benchmark": benchmark_summary,
    }


def _summary(
    harm_gate: MemoryUtilityGate,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    temperature: float,
    alpha_base: float,
) -> dict[str, Any]:
    static, dynamic, valid, positive, counts = _route_tensors(rows)
    with torch.no_grad():
        harm_probability = harm_gate(features)
        raw_alpha = float(alpha_base) * (1.0 - harm_probability)
        alpha = effective_memory_alpha(raw_alpha, counts)
        fused = fuse_route_scores(static, dynamic, alpha, valid)
        static_endpoint = fuse_route_scores(
            static,
            dynamic,
            torch.zeros_like(alpha),
            valid,
        )
        dynamic_endpoint = fuse_route_scores(
            static,
            dynamic,
            torch.ones_like(alpha),
            valid,
        )
        fixed_endpoint = fuse_route_scores(
            static,
            dynamic,
            torch.full_like(alpha, float(alpha_base)),
            valid,
        )
        target_harm, row_weight, delta = direct_harm_targets(
            static,
            fixed_endpoint,
            valid,
            positive,
            temperature=temperature,
        )
    static_summary = _ranking_summary(rows, static_endpoint, valid, positive)
    dynamic_summary = _ranking_summary(rows, dynamic_endpoint, valid, positive)
    fixed_summary = _ranking_summary(rows, fixed_endpoint, valid, positive)
    learned_summary = _ranking_summary(rows, fused, valid, positive)
    source_regret = {
        benchmark: float(fixed_summary["by_benchmark"][benchmark]["mrr"])
        - float(learned_summary["by_benchmark"][benchmark]["mrr"])
        for benchmark in fixed_summary["by_benchmark"]
    }
    informative = (counts > 0) & (row_weight > 0)
    informative_alpha = raw_alpha[informative]
    if int(informative_alpha.numel()) > 0:
        quantiles = torch.quantile(
            informative_alpha,
            torch.tensor(
                [0.25, 0.5, 0.75],
                dtype=informative_alpha.dtype,
                device=informative_alpha.device,
            ),
        )
        alpha_quantiles = {
            "q25": float(quantiles[0].item()),
            "q50": float(quantiles[1].item()),
            "q75": float(quantiles[2].item()),
        }
        alpha_min = float(informative_alpha.min().item())
        alpha_max = float(informative_alpha.max().item())
    else:
        alpha_quantiles = {"q25": 0.0, "q50": 0.0, "q75": 0.0}
        alpha_min = 0.0
        alpha_max = 0.0
    harmful = informative & (delta > 0)
    helpful = informative & (delta < 0)

    def subset_summary(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        indices = mask.nonzero(as_tuple=False).view(-1)
        if int(indices.numel()) == 0:
            return {"balanced_macro_mrr": 0.0, "by_benchmark": {}}
        selected_rows = [rows[int(index)] for index in indices.tolist()]
        return _ranking_summary(
            selected_rows,
            logits[indices],
            valid[indices],
            positive[indices],
        )

    harmful_fixed = subset_summary(fixed_endpoint, harmful)
    harmful_learned = subset_summary(fused, harmful)
    helpful_fixed = subset_summary(fixed_endpoint, helpful)
    helpful_learned = subset_summary(fused, helpful)
    harmful_alpha_mean = (
        float(raw_alpha[harmful].mean().item()) if bool(harmful.any()) else 0.0
    )
    helpful_alpha_mean = (
        float(raw_alpha[helpful].mean().item()) if bool(helpful.any()) else 0.0
    )
    return {
        "balanced_macro_mrr": float(learned_summary["balanced_macro_mrr"]),
        "by_benchmark": learned_summary["by_benchmark"],
        "static": static_summary,
        "dynamic": dynamic_summary,
        "fixed": fixed_summary,
        "learned": learned_summary,
        "alpha_base": float(alpha_base),
        "balanced_macro_mrr_improvement_vs_fixed": float(
            learned_summary["balanced_macro_mrr"]
            - fixed_summary["balanced_macro_mrr"]
        ),
        "source_regret_vs_fixed": source_regret,
        "worst_source_regret_vs_fixed": max(source_regret.values(), default=1.0),
        "harmful_rows": {
            "row_count": int(harmful.sum().item()),
            "fixed": harmful_fixed,
            "learned": harmful_learned,
            "mrr_improvement_vs_fixed": float(
                harmful_learned["balanced_macro_mrr"]
                - harmful_fixed["balanced_macro_mrr"]
            ),
            "alpha_mean": harmful_alpha_mean,
        },
        "helpful_rows": {
            "row_count": int(helpful.sum().item()),
            "fixed": helpful_fixed,
            "learned": helpful_learned,
            "mrr_regret_vs_fixed": float(
                helpful_fixed["balanced_macro_mrr"]
                - helpful_learned["balanced_macro_mrr"]
            ),
            "alpha_mean": helpful_alpha_mean,
        },
        "alpha": {
            "mean": float(raw_alpha.mean().item()),
            "min": alpha_min,
            "max": alpha_max,
            "quantiles": alpha_quantiles,
            "low_count": int((informative_alpha <= 0.25).sum().item()),
            "high_count": int((informative_alpha >= 0.75).sum().item()),
            "informative_row_count": int(informative.sum().item()),
        },
        "direct_harm": {
            "delta_mean": float(delta.mean().item()),
            "delta_abs_mean": float(delta.abs().mean().item()),
            "target_mean": float(target_harm.mean().item()),
            "row_weight_mean": float(row_weight.mean().item()),
        },
        "alpha_zero_exact": bool(
            torch.equal(
                static_endpoint,
                static.masked_fill(~valid, torch.finfo(static.dtype).min),
            )
        ),
        "alpha_one_exact": bool(
            torch.equal(
                dynamic_endpoint,
                dynamic.masked_fill(~valid, torch.finfo(dynamic.dtype).min),
            )
        ),
        "zero_history_exact": bool(
            torch.equal(
                fused[counts <= 0],
                static.masked_fill(~valid, torch.finfo(static.dtype).min)[counts <= 0],
            )
        ),
    }


def load_memory_utility_gate_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_audit_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    identity = sha256_path(path)
    if expected_sha256 is not None and identity["sha256"] != str(expected_sha256):
        raise ValueError("memory utility gate checkpoint SHA-256 mismatch")
    payload = torch.load(identity["path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("memory utility gate checkpoint must be a dict")
    if payload.get("stage") != "clstr_memory_utility_gate":
        raise ValueError("memory utility gate checkpoint stage mismatch")
    schema = str(payload.get("schema_version") or "")
    if schema not in {
        "memory_utility_gate_checkpoint_v1",
        "memory_utility_gate_checkpoint_v2",
    }:
        raise ValueError("unsupported memory utility gate checkpoint schema")
    if payload.get("feature_schema") != RELIABILITY_FEATURE_SCHEMA_VERSION:
        raise ValueError("memory utility gate feature schema mismatch")
    if payload.get("zero_history_fallback") != "exact_static":
        raise ValueError("memory utility gate must preserve exact static fallback")
    if payload.get("reliability_changes_memory_state") is not False:
        raise ValueError("memory utility gate cannot change memory state")
    audit_sha = str(payload.get("audit_manifest_sha256") or "")
    if expected_audit_sha256 is not None and audit_sha != str(
        expected_audit_sha256
    ):
        raise ValueError("memory utility gate audit identity mismatch")
    update_cap = float(payload.get("feature_update_count_cap", 0.0))
    candidate_cap = float(payload.get("feature_candidate_count_cap", 0.0))
    if not all(math.isfinite(value) and value > 0 for value in (update_cap, candidate_cap)):
        raise ValueError("memory utility gate feature caps must be finite and positive")
    state = (
        payload.get("harm_probability_gate_state_dict")
        if schema == "memory_utility_gate_checkpoint_v2"
        else payload.get("memory_utility_gate_state_dict")
    ) or {}
    if not isinstance(state, dict) or not state:
        raise ValueError("memory utility gate state dict is missing")
    mean = state.get("feature_mean")
    scale = state.get("feature_scale")
    if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("memory utility gate normalization is missing")
    core_gate = MemoryUtilityGate(feature_mean=mean, feature_scale=scale)
    core_gate.load_state_dict(state, strict=True)
    if schema == "memory_utility_gate_checkpoint_v2":
        if payload.get("gate_output_semantics") != "anchored_harm_suppression_alpha":
            raise ValueError("anchored gate output semantics mismatch")
        alpha_base = float(payload.get("alpha_base", 0.0))
        gate: torch.nn.Module = AnchoredMemoryUtilityGate(
            core_gate,
            alpha_base=alpha_base,
        )
        output_semantics = "anchored_harm_suppression_alpha"
    else:
        alpha_base = None
        gate = core_gate
        output_semantics = "free_alpha_v1"
    gate.eval()
    return gate, {
        "checkpoint_path": identity["path"],
        "checkpoint_sha256": identity["sha256"],
        "audit_manifest_sha256": audit_sha,
        "feature_update_count_cap": update_cap,
        "feature_candidate_count_cap": candidate_cap,
        "alpha_base": alpha_base,
        "gate_output_semantics": output_semantics,
    }


def resolve_reliability_gate(
    *,
    reliability_mode: str,
    gate_checkpoint_path: str | Path | None,
    expected_gate_sha256: str | None = None,
    expected_audit_sha256: str | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    mode = str(reliability_mode or "dynamic")
    if mode not in RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability mode: {mode}")
    normalized_path = (
        None
        if gate_checkpoint_path is None or not str(gate_checkpoint_path).strip()
        else Path(gate_checkpoint_path)
    )
    if mode not in {"learned", "cmc"}:
        if normalized_path is not None:
            raise ValueError("this reliability mode must not load a gate checkpoint")
        return None, {"mode": mode, "gate_loaded": False}
    if normalized_path is None:
        if mode == "learned":
            raise ValueError("learned reliability requires a gate checkpoint")
        return None, {
            "mode": mode,
            "gate_loaded": False,
            "gate_source": "stage4_checkpoint",
        }
    gate, report = load_memory_utility_gate_checkpoint(
        normalized_path,
        expected_sha256=(
            None
            if expected_gate_sha256 is None
            or not str(expected_gate_sha256).strip()
            else str(expected_gate_sha256).strip()
        ),
        expected_audit_sha256=(
            None
            if expected_audit_sha256 is None
            or not str(expected_audit_sha256).strip()
            else str(expected_audit_sha256).strip()
        ),
    )
    if device is not None:
        gate = gate.to(device)
    gate.eval()
    return gate, {
        "mode": mode,
        "gate_loaded": True,
        "gate_source": (
            "direct_utility_overlay" if mode == "cmc" else "standalone_learned"
        ),
        **report,
    }


def train_memory_utility_gate(
    *,
    route_records: list[dict[str, Any]],
    audit_report: dict[str, Any],
    dynamic_selection: dict[str, Any],
    output_dir: str | Path,
    temperature: float = 0.5,
    fused_rank_weight: float = 1.0,
    static_no_regret_weight: float = 1.0,
    direct_gate_bce_weight: float = 1.0,
    max_steps: int = 300,
    learning_rate: float = 1.0e-2,
    seed: int = 17,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "gate_report.json"
    _require_self_hash(audit_report, label="memory utility oracle audit")
    if audit_report.get("status") != "ok":
        raise ValueError("memory utility oracle audit is not ok")
    _require_self_hash(dynamic_selection, label="Stage4 dynamic selection")
    if dynamic_selection.get("status") != "ok" or dynamic_selection.get(
        "release_status"
    ) != "ok":
        raise ValueError("Stage4 dynamic selection is not release-safe")
    if dynamic_selection.get("stage4_method") != "counterfactual_memory_calibration_v1":
        raise ValueError("direct utility gate requires a selected CMC Stage4 endpoint")
    if not isinstance(dynamic_selection.get("cmc_fused_selection"), dict) or not isinstance(
        dynamic_selection.get("stage2_baseline"), dict
    ):
        raise ValueError("CMC Stage4 selection lacks endpoint validation summaries")
    expected_validation = dict(dynamic_selection.get("validation_route_records") or {})
    if len(route_records) != int(
        expected_validation.get("record_count", -1)
    ) or _record_digest(route_records) != str(
        expected_validation.get("row_digest_sha256") or ""
    ):
        raise ValueError("selected validation route-record identity mismatch")
    source = dict(audit_report.get("route_record_source") or {})
    if len(route_records) != int(source.get("record_count", -1)) or _record_digest(
        route_records
    ) != str(source.get("row_digest_sha256") or ""):
        raise ValueError("oracle audit route-record source mismatch")
    selected_records_path = Path(str(expected_validation.get("path") or "")).resolve()
    selected_records_sha = str(expected_validation.get("sha256") or "")
    selected_records_identity = sha256_path(selected_records_path)
    if (
        selected_records_identity["sha256"] != selected_records_sha
        or int(selected_records_identity["size"])
        != int(expected_validation.get("size", -1))
        or int(selected_records_identity["file_count"])
        != int(expected_validation.get("file_count", -1))
    ):
        raise ValueError("selected validation route-record file identity mismatch")
    manifest_matches = False
    for manifest_record in list(audit_report.get("route_manifests") or []):
        manifest_path = Path(str(manifest_record.get("manifest_path") or "")).resolve()
        if not manifest_path.is_file() or sha256_path(manifest_path) != manifest_record.get(
            "manifest_file_identity"
        ):
            raise ValueError("oracle audit route-manifest identity mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records_path = Path(str(manifest.get("records_path") or ""))
        if not records_path.is_absolute():
            records_path = manifest_path.parent / records_path
        if (
            records_path.resolve() == selected_records_path
            and str(manifest.get("records_sha256") or "")
            == hashlib.sha256(records_path.read_bytes()).hexdigest()
            and int(manifest.get("record_count", -1)) == len(route_records)
            and str(manifest.get("source_rows_digest") or "")
            == _record_digest(route_records)
        ):
            manifest_matches = True
    if not manifest_matches:
        raise ValueError("oracle audit does not bind the selected validation route manifest")
    objective = {
        "temperature": float(temperature),
        "fused_rank_weight": float(fused_rank_weight),
        "static_no_regret_weight": float(static_no_regret_weight),
        "direct_gate_bce_weight": float(direct_gate_bce_weight),
        "source_balanced": True,
        "harm_sign_balanced": True,
    }
    if not math.isfinite(objective["temperature"]) or objective["temperature"] <= 0:
        raise ValueError("direct utility temperature must be finite and positive")
    if any(
        not math.isfinite(objective[key]) or objective[key] < 0
        for key in (
            "fused_rank_weight",
            "static_no_regret_weight",
            "direct_gate_bce_weight",
        )
    ):
        raise ValueError("direct utility loss weights must be finite and nonnegative")
    if sum(
        objective[key]
        for key in (
            "fused_rank_weight",
            "static_no_regret_weight",
            "direct_gate_bce_weight",
        )
    ) <= 0:
        raise ValueError("direct utility objective must have a positive loss weight")
    base_report: dict[str, Any] = {
        "schema_version": "memory_utility_gate_report_v2",
        "audit_manifest_sha256": audit_report["manifest_sha256"],
        "selected_stage4_checkpoint_sha256": dynamic_selection[
            "selected_checkpoint_sha256"
        ],
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "promoted": False,
        "objective": objective,
    }
    if audit_report.get("learned_gate_recommended") is not True:
        result = {**base_report, "status": "not_recommended"}
        result["manifest_sha256"] = canonical_digest(result)
        _atomic_json(report_path, result)
        return result
    anchored_audit = dict(audit_report.get("anchored_harm_eligibility") or {})
    alpha_base = float(anchored_audit.get("alpha_base", 0.0))
    if (
        anchored_audit.get("eligible") is not True
        or not math.isfinite(alpha_base)
        or not 0.0 < alpha_base <= 1.0
        or alpha_base != float(audit_report.get("best_fixed_alpha", -1.0))
    ):
        raise ValueError("oracle audit lacks an eligible anchored harm endpoint")
    base_report["alpha_base"] = alpha_base
    base_report["gate_output_semantics"] = "anchored_harm_suppression_alpha"
    caps = dict(audit_report.get("count_feature_caps") or {})
    update_cap = float(caps.get("causal_update_count", 0.0))
    candidate_cap = float(caps.get("valid_candidate_count", 0.0))
    if caps.get("derived_from") != "train_trajectories_only" or not all(
        math.isfinite(value) and value > 0 for value in (update_cap, candidate_cap)
    ):
        raise ValueError("oracle audit count feature caps are invalid")
    split = dict(audit_report.get("split") or {})
    train_keys = set(split.get("train_trajectory_ids") or [])
    dev_keys = set(split.get("dev_trajectory_ids") or [])
    selected_train_rows = [
        row for row in route_records if _trajectory_key(row) in train_keys
    ]
    selected_dev_rows = [
        row for row in route_records if _trajectory_key(row) in dev_keys
    ]
    train_rows = [row for row in selected_train_rows if _gate_eligible_record(row)]
    dev_rows = [row for row in selected_dev_rows if _gate_eligible_record(row)]
    if not train_rows or not dev_rows:
        raise ValueError("oracle audit train/dev split does not cover gate records")
    calibration_rows = {
        "train": len(train_rows),
        "dev": len(dev_rows),
        "excluded_from_selected_trajectories": (
            len(selected_train_rows)
            + len(selected_dev_rows)
            - len(train_rows)
            - len(dev_rows)
        ),
    }
    train_features = _transformed_features(
        train_rows,
        update_cap=update_cap,
        candidate_cap=candidate_cap,
    )
    dev_features = _transformed_features(
        dev_rows,
        update_cap=update_cap,
        candidate_cap=candidate_cap,
    )
    feature_mean = train_features.mean(dim=0)
    feature_scale = train_features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    torch.manual_seed(int(seed))
    gate = MemoryUtilityGate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    torch.nn.init.zeros_(gate.net[0].weight)
    torch.nn.init.constant_(gate.net[0].bias, math.log(0.01 / 0.99))
    trainable_parameter_names = [
        name for name, parameter in gate.named_parameters() if parameter.requires_grad
    ]
    if set(trainable_parameter_names) != {"net.0.weight", "net.0.bias"}:
        raise ValueError("memory utility gate trainable scope is invalid")
    optimizer = torch.optim.Adam(gate.parameters(), lr=float(learning_rate))
    static, dynamic, valid, positive, counts = _route_tensors(train_rows)
    fixed_alpha = torch.full_like(counts, alpha_base)
    fixed = fuse_route_scores(static, dynamic, fixed_alpha, valid)
    fixed_utility = _positive_log_probability(fixed, valid, positive).detach()
    target_harm, row_weight, harm_delta = direct_harm_targets(
        static,
        fixed,
        valid,
        positive,
        temperature=objective["temperature"],
    )
    harm_sign = harm_delta > 0
    informative_harm = row_weight > 0
    train_sources = [str(row["benchmark"]) for row in train_rows]
    for _step in range(max(1, int(max_steps))):
        harm_probability = gate(train_features)
        raw_alpha = alpha_base * (1.0 - harm_probability)
        alpha = effective_memory_alpha(raw_alpha, counts)
        fused = fuse_route_scores(static, dynamic, alpha, valid)
        fused_utility = _positive_log_probability(fused, valid, positive)
        fused_rank_loss = _source_balanced_mean(-fused_utility, train_sources)
        no_regret_loss = _source_balanced_mean(
            torch.relu(fixed_utility - fused_utility),
            train_sources,
        )
        bce_rows = F.binary_cross_entropy(
            harm_probability,
            target_harm,
            reduction="none",
        ) * row_weight
        direct_bce_loss = _source_sign_balanced_mean(
            bce_rows,
            train_sources,
            harm_sign,
            informative_harm,
        )
        loss = (
            objective["fused_rank_weight"] * fused_rank_loss
            + objective["static_no_regret_weight"] * no_regret_loss
            + objective["direct_gate_bce_weight"] * direct_bce_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    dev_summary = _summary(
        gate,
        dev_rows,
        dev_features,
        temperature=objective["temperature"],
        alpha_base=alpha_base,
    )
    promotion_checks = {
        "alpha_zero_exact": dev_summary["alpha_zero_exact"] is True,
        "alpha_one_exact": dev_summary["alpha_one_exact"] is True,
        "zero_history_exact": dev_summary["zero_history_exact"] is True,
        "positive_source_balanced_fused_mrr_delta_vs_fixed": float(
            dev_summary["balanced_macro_mrr_improvement_vs_fixed"]
        )
        > 0.0,
        "worst_source_regret_vs_fixed_within_tolerance": float(
            dev_summary["worst_source_regret_vs_fixed"]
        )
        <= 0.005,
        "harmful_rows_improve": float(
            dev_summary["harmful_rows"]["mrr_improvement_vs_fixed"]
        )
        > 0.0,
        "helpful_rows_preserved": float(
            dev_summary["helpful_rows"]["mrr_regret_vs_fixed"]
        )
        <= 0.005,
        "harmful_helpful_alpha_separated": float(
            dev_summary["helpful_rows"]["alpha_mean"]
            - dev_summary["harmful_rows"]["alpha_mean"]
        )
        >= 0.10,
    }
    promoted = all(promotion_checks.values())
    checkpoint_path: Path | None = None
    checkpoint_sha: str | None = None
    if promoted:
        checkpoint_path = output / "memory_utility_gate.pt"
        torch.save(
            {
                "stage": "clstr_memory_utility_gate",
                "schema_version": "memory_utility_gate_checkpoint_v2",
                "feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
                "feature_names": list(MEMORY_UTILITY_FEATURE_NAMES),
                "gate_output_semantics": "anchored_harm_suppression_alpha",
                "alpha_base": alpha_base,
                "audit_manifest_sha256": audit_report["manifest_sha256"],
                "zero_history_fallback": "exact_static",
                "reliability_changes_memory_state": False,
                "selected_stage4_checkpoint_sha256": dynamic_selection[
                    "selected_checkpoint_sha256"
                ],
                "training_seed": int(seed),
                "trainable_parameter_names": trainable_parameter_names,
                "calibration_rows": calibration_rows,
                "objective": objective,
                "feature_update_count_cap": update_cap,
                "feature_candidate_count_cap": candidate_cap,
                "harm_probability_gate_state_dict": gate.state_dict(),
                "validation": {
                    "dev": dev_summary,
                    "promotion_checks": promotion_checks,
                },
            },
            checkpoint_path,
        )
        checkpoint_sha = sha256_path(checkpoint_path)["sha256"]
    result = {
        **base_report,
        "status": "ok",
        "promoted": promoted,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "training_seed": int(seed),
        "alpha_base": alpha_base,
        "gate_output_semantics": "anchored_harm_suppression_alpha",
        "trainable_parameter_names": trainable_parameter_names,
        "calibration_rows": calibration_rows,
        "feature_update_count_cap": update_cap,
        "feature_candidate_count_cap": candidate_cap,
        "dev_summary": dev_summary,
        "validation": {
            "dev": dev_summary,
            "promotion_checks": promotion_checks,
        },
    }
    result["manifest_sha256"] = canonical_digest(result)
    _atomic_json(report_path, result)
    return result
