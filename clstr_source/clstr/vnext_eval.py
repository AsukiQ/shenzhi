from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from clstr.history_channel import (
    audit_history_channel_rows,
    serialize_compact_causal_state,
    materialize_history_free_state,
    router_state_text,
    serialize_causal_state,
)
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.stage_checkpoint_init import load_compatible_state_dict
from clstr.vnext_candidates import (
    direct_natural_support,
    masked_topk_tensor,
)
from clstr.vnext_matched_release import (
    MATCHED_RELEASE_SELECTION_SCHEMA,
    validate_matched_release_selection,
)
from clstr.vnext_training import (
    candidate_foundation_digest,
    file_sha256,
    is_vnext_checkpoint_state_key,
    load_or_build_frozen_text_cache,
    require_canonical_vnext_checkpoint_state,
    static_foundation_digest,
    verify_frozen_backbone_contract,
)
from clstr.vnext_unified_router import (
    SUPPORT_AWARE_ANCHOR_PROTOCOL,
    SUPPORT_AWARE_FOUNDATION_PREFIX_K,
    UNIFIED_EXPERT_NAMES,
    UNIFIED_ROUTER_FEATURE_NAMES,
    UNIFIED_ROUTER_SCHEMA,
    UNIFIED_ROUTING_MODES,
    UNIFIED_ROUTING_MODE_SPARSE,
    UnifiedThreeExpertRouter,
    route_with_support_aware_anchor,
    route_with_unified_three_experts,
)


VNEXT_EVAL_SCHEMA = "clstr_vnext_checkpoint_native_eval_v20"
VNEXT_BENCHMARKS = frozenset(
    {"toolbench_g3", "toolsandbox", "tau2", "trajectbench"}
)
TOOLBENCH_G3_NATIVE_SKILL_COUNT = 27_486
CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE = 8
OPEN_POOL_SELECTION_SCHEMA = "clstr_vnext_stage2_open_pool_selection_v1"
UNIFIED_ROUTER_RELEASE_WRAPPER_SCHEMA = (
    "clstr_vnext_unified_router_stage2_release_wrapper_v1"
)
TOOLBENCH_EXECUTED_RESULT_SOURCES = frozenset(
    {
        "actual_tool_result",
        "executed_trace:toolbench_g3",
    }
)


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_unified_router_contract(
    router_path: str | Path,
    *,
    stage2_checkpoint_path: str | Path,
    foundation_checkpoint_path: str | Path,
    training_skills_path: str | Path,
    stage2_release_selection_contract: dict[str, Any],
    device: torch.device,
) -> tuple[UnifiedThreeExpertRouter, dict[str, Any]]:
    path = Path(router_path).resolve()
    if not path.is_file():
        raise ValueError(f"unified-router artifact is missing: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != UNIFIED_ROUTER_SCHEMA:
        raise ValueError("unsupported unified-router artifact schema")
    if payload.get("status") != "ok" or payload.get("blockers"):
        raise ValueError("unified-router artifact is not calibration-approved")
    if list(payload.get("feature_names") or []) != list(UNIFIED_ROUTER_FEATURE_NAMES):
        raise ValueError("unified-router feature schema differs")
    if list(payload.get("expert_names") or []) != list(UNIFIED_EXPERT_NAMES):
        raise ValueError("unified-router expert schema differs")
    routing_mode = str(payload.get("routing_mode") or "")
    if routing_mode not in UNIFIED_ROUTING_MODES:
        raise ValueError("unified-router routing mode differs")
    if routing_mode != UNIFIED_ROUTING_MODE_SPARSE:
        raise ValueError("unified-router artifact is not sparse-deployment approved")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("unified-router artifact lacks its selection contract")
    if bool(selection.get("uses_benchmark_eval_rows")) or bool(
        selection.get("uses_benchmark_identity_feature")
    ):
        raise ValueError("unified router may not use benchmark evaluation identity")
    training_contract = payload.get("training_contract")
    expected_training_contract = {
        "default_expert_prior": "foundation",
        "observation_grounding_feature": "observation_correction_fraction",
        "paired_calibration_protocol": (
            "same_prefix_factual_and_action_only_result_suppressed_v1"
        ),
        "paired_views_share_trajectory_split": True,
        "foundation_expert_protocol": (
            "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
        ),
        "expert_supervision_protocol": (
            "class_balanced_sparse_minimum_regret_top1_v1"
        ),
        "expert_tie_break_order": list(UNIFIED_EXPERT_NAMES),
        "expert_utility": "reciprocal_rank_plus_0.5_top5",
        "oracle_expert_recall_min_rows": 20,
        "oracle_expert_recall_floor": 0.20,
        "calibration_no_regret_tolerances": {
            "global_mrr": 0.005,
            "global_recall_at_5": 0.01,
            "family_mrr": 0.02,
            "family_recall_at_5": 0.03,
            "memory_evidence_mrr": 0.005,
            "memory_evidence_recall_at_5": 0.01,
        },
    }
    if not isinstance(training_contract, dict) or any(
        training_contract.get(key) != value
        for key, value in expected_training_contract.items()
    ):
        raise ValueError(
            "unified-router observation-grounded calibration contract differs"
        )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if str(payload.get("source_commit") or "") != source_commit:
        raise ValueError("unified-router artifact was trained from a different source commit")
    expected = {
        "stage2_checkpoint_sha256": file_sha256(stage2_checkpoint_path),
        "foundation_checkpoint_sha256": file_sha256(foundation_checkpoint_path),
        "training_skills_sha256": file_sha256(training_skills_path),
    }
    for key, value in expected.items():
        if str(payload.get(key) or "") != value:
            raise ValueError(f"unified-router {key} differs from evaluation lineage")
    release_wrapper = payload.get("stage2_release_wrapper")
    expected_release_wrapper = _unified_router_release_wrapper(
        stage2_release_selection_contract,
        router_source_commit=source_commit,
        foundation_checkpoint_path=foundation_checkpoint_path,
        foundation_checkpoint_sha256=expected["foundation_checkpoint_sha256"],
        training_skills_path=training_skills_path,
        training_skills_sha256=expected["training_skills_sha256"],
    )
    if release_wrapper != expected_release_wrapper:
        raise ValueError(
            "unified-router Stage2 release wrapper differs from evaluation lineage"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("unified-router artifact lacks a state dictionary")
    router = UnifiedThreeExpertRouter().to(device)
    missing, unexpected = router.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "unified-router state mismatch: "
            f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"
        )
    router.eval()
    return router, {
        "schema_version": UNIFIED_ROUTER_SCHEMA,
        "path": str(path),
        "sha256": file_sha256(path),
        "stage2_checkpoint_sha256": expected["stage2_checkpoint_sha256"],
        "foundation_checkpoint_sha256": expected[
            "foundation_checkpoint_sha256"
        ],
        "training_skills_sha256": expected["training_skills_sha256"],
        "feature_names": list(UNIFIED_ROUTER_FEATURE_NAMES),
        "expert_names": list(UNIFIED_EXPERT_NAMES),
        "routing_mode": routing_mode,
        "uses_benchmark_identity": False,
        "uses_source_identity": False,
        "source_commit": source_commit,
        "stage2_release_wrapper": expected_release_wrapper,
        "training_contract": expected_training_contract,
    }


def _stage2_release_selection_contract(
    selection_path: str | Path,
    *,
    stage2_checkpoint_path: str | Path,
) -> dict[str, Any]:
    path = Path(selection_path).resolve()
    if not path.is_file():
        raise ValueError(f"Stage2 release selection is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage2 release selection must be a JSON object")
    if payload.get("schema_version") == MATCHED_RELEASE_SELECTION_SCHEMA:
        return validate_matched_release_selection(
            path,
            stage2_checkpoint_path=stage2_checkpoint_path,
        )
    if payload.get("schema_version") != OPEN_POOL_SELECTION_SCHEMA:
        raise ValueError("unsupported Stage2 release-selection schema")
    if payload.get("status") != "ok":
        raise ValueError("Stage2 release selection is not approved")
    if payload.get("selection_mode") != "open_pool_full_pool":
        raise ValueError("Stage2 release selection has the wrong deployment mode")
    if bool(payload.get("uses_benchmark_eval_rows")) or bool(
        payload.get("uses_test_metrics")
    ):
        raise ValueError("Stage2 release selection may not use benchmark evaluation rows")
    if not bool(payload.get("closed_set_dispatch_required")):
        raise ValueError("open-pool release must preserve the closed-set dispatch")
    requested = Path(stage2_checkpoint_path).resolve()
    selected = Path(str(payload.get("selected_checkpoint_path") or "")).resolve()
    if selected != requested:
        raise ValueError("Stage2 checkpoint differs from the release selection")
    if not selected.is_file():
        raise ValueError("release-selected Stage2 checkpoint is missing")
    expected_checkpoint_sha = str(payload.get("selected_checkpoint_sha256") or "")
    if not expected_checkpoint_sha or file_sha256(selected) != expected_checkpoint_sha:
        raise ValueError("release-selected Stage2 checkpoint digest differs")
    output = Path(str(payload.get("stage2_output_dir") or "")).resolve()
    try:
        selected.relative_to((output / "checkpoints").resolve())
    except ValueError as error:
        raise ValueError("release-selected checkpoint escapes Stage2 output") from error
    original_path = Path(
        str(payload.get("original_stage2_selection_path") or "")
    ).resolve()
    if original_path != (output / "stage2_selection.json").resolve():
        raise ValueError("release selection names the wrong original Stage2 selection")
    if not original_path.is_file():
        raise ValueError("original Stage2 selection is missing")
    original_sha = str(payload.get("original_stage2_selection_sha256") or "")
    if not original_sha or file_sha256(original_path) != original_sha:
        raise ValueError("original Stage2 selection digest differs")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original_status = str(
        original.get("status") if isinstance(original, dict) else ""
    )
    if original_status not in {"ok", "action_required"}:
        raise ValueError("original Stage2 training selection is not complete")
    artifact_digests = (
        ("stage2_quality_gate.json", "original_stage2_quality_gate_sha256"),
        ("train_report.json", "original_train_report_sha256"),
    )
    for filename, field in artifact_digests:
        artifact = output / filename
        expected = str(payload.get(field) or "")
        if not artifact.is_file() or not expected or file_sha256(artifact) != expected:
            raise ValueError(f"Stage2 release source artifact differs: {filename}")
    train_report = json.loads((output / "train_report.json").read_text(encoding="utf-8"))
    interface_scoped = bool(payload.get("interface_scoped_training_approval"))
    interface_contract = payload.get("interface_scoped_training_contract") or {}
    if original_status == "ok":
        if interface_scoped:
            raise ValueError(
                "universally approved Stage2 selection may not claim scoped approval"
            )
    else:
        warm_start = train_report.get("objective_warm_start") or {}
        topk = train_report.get("route_topk") or {}
        if not (
            interface_scoped
            and bool(interface_contract.get("enabled"))
            and str(payload.get("original_training_status") or "")
            == original_status
            and str(train_report.get("status") or "") == original_status
            and train_report.get("selection") == original
            and bool(train_report.get("finite_loss"))
            and bool(warm_start.get("enabled"))
            and bool(warm_start.get("model_state_exact"))
            and not bool(warm_start.get("optimizer_restored"))
            and not bool(warm_start.get("trainer_progress_restored"))
            and bool(topk.get("enabled"))
            and int(topk.get("k") or 0) == 5
            and float(topk.get("lambda") or 0.0) > 0.0
            and int(topk.get("eligible_rows") or 0) > 0
            and int(topk.get("positive_injection_count") or 0) == 0
            and str(topk.get("candidate_membership_protocol") or "")
            == "immutable_natural_support"
            and int(train_report.get("positive_injection_count") or 0) == 0
            and int(train_report.get("training_teacher_retained_rows") or 0) == 0
            and interface_contract.get("objective_warm_start") == warm_start
            and interface_contract.get("route_topk") == topk
            and not bool(
                interface_contract.get(
                    "closed_set_deployment_executes_stage2",
                    True,
                )
            )
            and bool(interface_contract.get("open_pool_candidate_gates_remain_required"))
        ):
            raise ValueError(
                "action-required Stage2 selection lacks verified scoped refinement approval"
            )
    source_manifest_sha = payload.get("source_manifest_sha256")
    if source_manifest_sha is not None:
        source_manifest = output / "source_manifest.json"
        if (
            not source_manifest.is_file()
            or file_sha256(source_manifest) != str(source_manifest_sha)
        ):
            raise ValueError("Stage2 source-manifest digest differs")
    records = list(original.get("validation_records") or [])
    if _json_digest(records) != str(payload.get("validation_records_sha256") or ""):
        raise ValueError("Stage2 validation-record digest differs")
    if int(payload.get("positive_injection_count") or 0) != 0:
        raise ValueError("release selection declares positive candidate injection")
    if int(payload.get("selected_step") or 0) <= 0:
        raise ValueError("release selection lacks a positive checkpoint step")
    selected_summary = payload.get("selected_candidate_summary")
    selected_validation = payload.get("selected_validation")
    if (
        not isinstance(selected_summary, dict)
        or not bool(selected_summary.get("eligible"))
        or int(selected_summary.get("step") or 0)
        != int(payload.get("selected_step") or 0)
        or not isinstance(selected_validation, dict)
        or int(selected_validation.get("step") or 0)
        != int(payload.get("selected_step") or 0)
    ):
        raise ValueError("release selection has inconsistent selected evidence")
    if original_status == "action_required":
        required_gate_values = dict(
            selected_summary.get("required_gate_values") or {}
        )
        if not required_gate_values or not all(
            bool(value) for value in required_gate_values.values()
        ):
            raise ValueError(
                "scoped refinement release does not pass every open-pool candidate gate"
            )
    if not str(payload.get("source_commit") or ""):
        raise ValueError("release selection lacks source provenance")
    return {
        "schema_version": OPEN_POOL_SELECTION_SCHEMA,
        "selection_path": str(path),
        "selection_sha256": file_sha256(path),
        "selection_mode": str(payload["selection_mode"]),
        "selection_scope": str(payload.get("selection_scope") or ""),
        "selection_reason": str(payload.get("selection_reason") or ""),
        "selection_source": str(payload.get("selection_source") or ""),
        "source_commit": str(payload["source_commit"]),
        "selected_step": int(payload["selected_step"]),
        "selected_checkpoint_path": str(selected),
        "selected_checkpoint_sha256": expected_checkpoint_sha,
        "original_selected_step": int(payload.get("original_selected_step") or 0),
        "original_training_status": original_status,
        "interface_scoped_training_approval": interface_scoped,
        "original_stage2_selection_sha256": original_sha,
        "original_stage2_selection_path": str(original_path),
        "validation_records_sha256": str(payload["validation_records_sha256"]),
        "heldout_family": str(payload.get("heldout_family") or ""),
        "heldout_source": str(payload.get("heldout_source") or ""),
        "heldout_metric": str(payload.get("heldout_metric") or ""),
        "selected_candidate_summary": selected_summary,
        "closed_set_dispatch_required": True,
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
    }


def _unified_router_release_wrapper(
    stage2_release_selection_contract: dict[str, Any],
    *,
    router_source_commit: str,
    foundation_checkpoint_path: str | Path,
    foundation_checkpoint_sha256: str,
    training_skills_path: str | Path,
    training_skills_sha256: str,
) -> dict[str, Any]:
    if not router_source_commit:
        raise ValueError("unified-router wrapper requires router source provenance")
    if stage2_release_selection_contract.get("schema_version") != OPEN_POOL_SELECTION_SCHEMA:
        raise ValueError("unified-router wrapper requires an approved Stage2 release")
    if bool(stage2_release_selection_contract.get("uses_benchmark_eval_rows")) or bool(
        stage2_release_selection_contract.get("uses_test_metrics")
    ):
        raise ValueError("unified-router wrapper may not bind benchmark test evidence")
    required = (
        "selection_path",
        "selection_sha256",
        "source_commit",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "original_stage2_selection_path",
        "original_stage2_selection_sha256",
    )
    missing = [
        key for key in required if not str(stage2_release_selection_contract.get(key) or "")
    ]
    if missing:
        raise ValueError(
            "unified-router wrapper lacks Stage2 release fields: "
            + ", ".join(missing)
        )
    return {
        "schema_version": UNIFIED_ROUTER_RELEASE_WRAPPER_SCHEMA,
        "router_source_commit": str(router_source_commit),
        "stage2_release_selection_path": str(
            Path(stage2_release_selection_contract["selection_path"]).resolve()
        ),
        "stage2_release_selection_sha256": str(
            stage2_release_selection_contract["selection_sha256"]
        ),
        "stage2_release_source_commit": str(
            stage2_release_selection_contract["source_commit"]
        ),
        "stage2_selection_mode": str(
            stage2_release_selection_contract.get("selection_mode") or ""
        ),
        "stage2_checkpoint_path": str(
            Path(stage2_release_selection_contract["selected_checkpoint_path"]).resolve()
        ),
        "stage2_checkpoint_sha256": str(
            stage2_release_selection_contract["selected_checkpoint_sha256"]
        ),
        "original_stage2_selection_path": str(
            Path(
                stage2_release_selection_contract["original_stage2_selection_path"]
            ).resolve()
        ),
        "original_stage2_selection_sha256": str(
            stage2_release_selection_contract["original_stage2_selection_sha256"]
        ),
        "foundation_checkpoint_path": str(Path(foundation_checkpoint_path).resolve()),
        "foundation_checkpoint_sha256": str(foundation_checkpoint_sha256),
        "training_skills_path": str(Path(training_skills_path).resolve()),
        "training_skills_sha256": str(training_skills_sha256),
        "uses_benchmark_eval_rows": False,
        "uses_test_metrics": False,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _source_file_contract(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    unique = sorted({path.resolve() for path in paths})
    rows: list[dict[str, Any]] = []
    for path in unique:
        if not path.is_file():
            raise ValueError(f"evaluation source artifact is missing: {path}")
        label = (
            str(path.relative_to(root.resolve()))
            if root is not None and path.is_relative_to(root.resolve())
            else str(path)
        )
        rows.append(
            {
                "path": label,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    if not rows:
        raise ValueError("evaluation source contract has no files")
    return {
        "file_count": len(rows),
        "files": rows,
        "contract_sha256": _json_digest(rows),
    }


def _corpus_source_contract(benchmark: str, corpus: Any) -> dict[str, Any]:
    report = dict(corpus.report)
    benchmark = str(benchmark).strip().lower()
    if report.get("matched_prebuilt_manifest_path"):
        return {
            "protocol": "matched_prebuilt_route_corpus_v1",
            "benchmark": benchmark,
            "task_split": str(report.get("task_split") or ""),
            "matched_union_manifest_sha256": str(
                report.get("matched_union_manifest_sha256") or ""
            ),
            **_source_file_contract(
                [
                    Path(str(report["matched_prebuilt_manifest_path"])),
                    Path(str(report["matched_prebuilt_source_rows_path"])),
                    Path(str(report["matched_prebuilt_skills_path"])),
                ]
            ),
        }
    if benchmark == "toolbench_g3":
        return {
            "protocol": "toolbench_exact_files_v1",
            **_source_file_contract(
                [
                    Path(str(report.get("eval_trajectories_path") or "")),
                    Path(str(report.get("skills_path") or "")),
                ]
            ),
        }
    if benchmark == "toolsandbox":
        scenarios_root = Path(str(report.get("scenarios_root") or ""))
        tools_value = report.get("tools_root")
        paths = list(sorted(scenarios_root.glob("*_scenarios.py")))
        if tools_value:
            tools_root = Path(str(tools_value))
            paths.extend(sorted(tools_root.rglob("*.py")))
        return {
            "protocol": "toolsandbox_source_tree_v1",
            "scenarios_root": str(scenarios_root.resolve()),
            "tools_root": None if not tools_value else str(Path(str(tools_value)).resolve()),
            **_source_file_contract(paths),
        }
    if benchmark == "tau2":
        data_root = Path(str(report.get("data_root") or ""))
        paths: list[Path] = []
        for domain in report.get("domains") or []:
            domain_root = data_root / "domains" / str(domain)
            paths.append(domain_root / "tasks.json")
            split_path = domain_root / "split_tasks.json"
            if split_path.exists():
                paths.append(split_path)
            for name in (
                "policy.md",
                "main_policy.md",
                "policy_solo.md",
                "main_policy_solo.md",
            ):
                policy_path = domain_root / name
                if policy_path.exists():
                    paths.append(policy_path)
                    break
        return {
            "protocol": "tau2_domain_source_files_v1",
            "data_root": str(data_root.resolve()),
            "task_split": str(report.get("task_split") or ""),
            "split_membership_sha256": str(
                report.get("split_membership_sha256") or ""
            ),
            **_source_file_contract(paths, root=data_root),
        }
    if benchmark == "trajectbench":
        public_data = Path(str(report.get("public_data") or ""))
        paths = [Path(str(value)) for value in report.get("source_files") or []]
        return {
            "protocol": "trajectbench_deterministic_heldout_global_inventory_v1",
            "public_data": str(public_data.resolve()),
            "split_partition": str(report.get("split_partition") or ""),
            "split_policy": str(report.get("split_policy") or ""),
            **_source_file_contract(paths, root=public_data),
        }
    raise ValueError(f"unsupported source contract benchmark: {benchmark}")


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _unique_skills(rows: list[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        skill_id = _skill_id(row)
        if not skill_id:
            raise ValueError(f"{label} contains an empty skill identity")
        if skill_id in seen:
            raise ValueError(f"{label} contains a duplicate skill identity: {skill_id}")
        seen.add(skill_id)
        output.append(dict(row))
    return output


def _partition_benchmark_skill_append(
    training_skills: list[dict[str, Any]],
    benchmark_skills: list[dict[str, Any]],
    *,
    serializer: Callable[[dict[str, Any]], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fail closed when one skill ID has two different serialized definitions."""

    training_by_id = {_skill_id(row): row for row in training_skills}
    appended: list[dict[str, Any]] = []
    verified_overlap: list[str] = []
    for row in _unique_skills(benchmark_skills, label="benchmark skill pool"):
        skill_id = _skill_id(row)
        existing = training_by_id.get(skill_id)
        if existing is None:
            appended.append(dict(row))
            continue
        if serializer(existing) != serializer(row):
            raise ValueError(
                f"benchmark skill definition conflicts with checkpoint prefix: {skill_id}"
            )
        verified_overlap.append(skill_id)
    return appended, sorted(verified_overlap)


def _evaluation_restore_contract(
    *,
    checkpoint_stage: str,
    checkpoint_state_keys: Iterable[str],
    model_state_keys: Iterable[str],
    loaded_keys: Iterable[str],
) -> dict[str, Any]:
    """Describe and enforce the stage-specific native-eval restore scope.

    A release Stage2 checkpoint is a complete deployable model and therefore
    must restore every canonical vNext tensor instantiated by its config.  A
    Stage0 or candidate-compressor checkpoint is explicitly admitted only as a
    static-foundation diagnostic.  Such a checkpoint must restore every
    canonical tensor that it declares, but it cannot be required to contain
    later-stage recurrent heads that were not part of that checkpoint.
    """

    checkpoint_keys = {
        str(name)
        for name in checkpoint_state_keys
        if is_vnext_checkpoint_state_key(str(name))
    }
    model_keys = {
        str(name)
        for name in model_state_keys
        if is_vnext_checkpoint_state_key(str(name))
    }
    loaded = {str(name) for name in loaded_keys}
    if checkpoint_stage == "clstr_vnext_stage2":
        required_scope = "complete_stage2_model"
        required = model_keys
    else:
        required_scope = "checkpoint_declared_static_diagnostic"
        required = checkpoint_keys
    return {
        "required_scope": required_scope,
        "required_key_count": len(required),
        "declared_checkpoint_key_count": len(checkpoint_keys),
        "loaded_canonical_key_count": len(model_keys & loaded),
        "missing_required_keys": sorted(required - loaded),
        "unrestored_model_canonical_keys": sorted(model_keys - loaded),
    }


def _interface_expert_decision(
    source_rows: Iterable[dict[str, Any]],
    *,
    coarse_k: int,
    declared_open_pool_size: int | None = None,
    declared_pool_protocol: str | None = None,
) -> dict[str, Any]:
    """Select a preserved closed-set foundation only when recall is unnecessary."""

    routing_rows = 0
    enumerated_rows = 0
    candidate_counts: list[int] = []
    for row in source_rows:
        if str(row.get("route_target") or "").upper() in {"STOP", "NO_CALL"}:
            continue
        routing_rows += 1
        raw_candidates = row.get("candidate_next_skill_ids")
        if not isinstance(raw_candidates, list):
            continue
        candidates = list(
            dict.fromkeys(str(item) for item in raw_candidates if str(item))
        )
        target = str(row.get("next_skill_id") or "").strip()
        if not candidates or not target or target not in set(candidates):
            continue
        enumerated_rows += 1
        candidate_counts.append(len(candidates))
    declared_size = (
        None if declared_open_pool_size is None else int(declared_open_pool_size)
    )
    fully_enumerated = routing_rows > 0 and enumerated_rows == routing_rows
    maximum_row_pool_size = max(candidate_counts, default=0)
    closed_set = (
        declared_size is None
        and fully_enumerated
        and maximum_row_pool_size <= int(coarse_k)
    )
    return {
        "protocol": "retrieval_requirement_aware_foundation_dispatch_v1",
        "selected_expert": (
            "preserved_closed_set_foundation"
            if closed_set
            else "adapted_static_plus_recurrent_dynamic"
        ),
        "reason": (
            "all_decision_time_legal_pools_fit_natural_candidate_budget"
            if closed_set
            else (
                "declared_open_pool_exceeds_natural_candidate_budget"
                if declared_size is not None and declared_size > int(coarse_k)
                else "decision_time_legal_pool_not_fully_enumerated_within_budget"
            )
        ),
        "coarse_k": int(coarse_k),
        "routing_row_count": int(routing_rows),
        "fully_enumerated_row_count": int(enumerated_rows),
        "all_routing_rows_fully_enumerated": bool(fully_enumerated),
        "maximum_row_pool_size": int(maximum_row_pool_size),
        "declared_open_pool_size": declared_size,
        "declared_pool_protocol": declared_pool_protocol,
        "uses_benchmark_identity": False,
    }


def _foundation_preservation_contract(
    *,
    stage2_checkpoint_path: str | Path,
    foundation_checkpoint_path: str | Path,
    training_skills_path: str | Path,
) -> dict[str, Any]:
    """Verify that two checkpoints form one frozen-backbone deployment lineage."""

    stage2_path = Path(stage2_checkpoint_path)
    foundation_path = Path(foundation_checkpoint_path)
    skills_path = Path(training_skills_path)
    stage2 = torch.load(stage2_path, map_location="cpu")
    foundation = torch.load(foundation_path, map_location="cpu")
    if not isinstance(stage2, dict) or stage2.get("stage") != "clstr_vnext_stage2":
        raise ValueError(
            "foundation-preserving deployment requires a Stage2 checkpoint"
        )
    if (
        not isinstance(foundation, dict)
        or foundation.get("stage") != "clstr_vnext_stage0"
        or int(foundation.get("step", -1)) != 0
    ):
        raise ValueError(
            "preserved closed-set foundation must be the stable Stage0 step-0 checkpoint"
        )
    stage2_state = stage2.get("model_state_dict")
    foundation_state = foundation.get("model_state_dict")
    require_canonical_vnext_checkpoint_state(stage2_state)
    require_canonical_vnext_checkpoint_state(foundation_state)
    if not isinstance(stage2_state, dict) or not isinstance(foundation_state, dict):
        raise ValueError("foundation-preserving checkpoints lack model state")
    for prefix in (
        "vnext.memory_recall_query.",
        "vnext.route_expert_mixture.",
        "vnext.unified_route_query.",
    ):
        if not any(str(name).startswith(prefix) for name in stage2_state):
            raise ValueError(
                f"Stage2 checkpoint lacks required deployment head: {prefix}"
            )
    stage2_table = stage2_state.get("skill_table.E")
    foundation_table = foundation_state.get("skill_table.E")
    if not isinstance(stage2_table, torch.Tensor) or not isinstance(
        foundation_table, torch.Tensor
    ):
        raise ValueError("foundation-preserving checkpoints lack skill-table tensors")
    if tuple(stage2_table.shape) != tuple(foundation_table.shape):
        raise ValueError("foundation and Stage2 skill-table prefixes differ in shape")
    observed_skills_sha = file_sha256(skills_path)

    def _inputs(payload: dict[str, Any]) -> dict[str, Any]:
        return ((payload.get("run_contract") or {}).get("inputs") or {})

    stage2_inputs = _inputs(stage2)
    foundation_inputs = _inputs(foundation)
    if (
        str(stage2_inputs.get("skills_sha256") or "") != observed_skills_sha
        or str(foundation_inputs.get("skills_sha256") or "") != observed_skills_sha
    ):
        raise ValueError("foundation-preserving checkpoints do not bind the skill prefix")
    stage2_config = dict(stage2.get("config") or {})
    foundation_config = dict(foundation.get("config") or {})
    identity_fields = (
        "base_model_name",
        "frozen_backbone_snapshot_digest",
        "skill_text_format",
        "state_query_prompt_version",
    )
    mismatched_identity = [
        field
        for field in identity_fields
        if stage2_config.get(field) != foundation_config.get(field)
    ]
    if mismatched_identity:
        raise ValueError(
            "foundation and Stage2 checkpoint identities differ: "
            f"{mismatched_identity}"
        )
    if not bool(stage2_config.get("freeze_backbone")) or not bool(
        foundation_config.get("freeze_backbone")
    ):
        raise ValueError("foundation-preserving deployment requires frozen backbones")
    return {
        "protocol": "lineage_bound_stage0_step0_plus_release_stage2_v1",
        "stage2_checkpoint_path": str(stage2_path.resolve()),
        "stage2_checkpoint_sha256": file_sha256(stage2_path),
        "stage2_checkpoint_step": int(stage2.get("step") or 0),
        "foundation_checkpoint_path": str(foundation_path.resolve()),
        "foundation_checkpoint_sha256": file_sha256(foundation_path),
        "foundation_checkpoint_step": 0,
        "training_skills_path": str(skills_path.resolve()),
        "training_skills_sha256": observed_skills_sha,
        "checkpoint_skill_count": int(stage2_table.size(0)),
        "frozen_backbone_snapshot_digest": str(
            stage2_config.get("frozen_backbone_snapshot_digest") or ""
        ),
        "identity_fields_verified": list(identity_fields),
    }


def load_vnext_stage2_for_evaluation(
    *,
    checkpoint_path: str | Path,
    training_skills_path: str | Path,
    benchmark_skills: list[dict[str, Any]],
    device: torch.device,
    allow_stage0_static_diagnostic: bool = False,
    allow_static_reranker_diagnostic: bool = False,
) -> tuple[CLSTRModel, list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    training_skills_path = Path(training_skills_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_stage = payload.get("stage") if isinstance(payload, dict) else None
    allowed_stages = {"clstr_vnext_stage2"}
    if allow_stage0_static_diagnostic:
        allowed_stages.add("clstr_vnext_stage0")
    if allow_static_reranker_diagnostic:
        allowed_stages.add("clstr_vnext_candidate_compressor")
    if not isinstance(payload, dict) or checkpoint_stage not in allowed_stages:
        raise ValueError(
            "vNext evaluation requires Stage2 or an explicitly enabled static "
            "diagnostic checkpoint"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or "skill_table.E" not in state:
        raise ValueError("vNext Stage2 checkpoint lacks its frozen skill-table prefix")
    checkpoint_state_report = require_canonical_vnext_checkpoint_state(state)
    training_skills = _unique_skills(
        _read_jsonl(training_skills_path),
        label="training skill prefix",
    )
    checkpoint_skill_count = int(state["skill_table.E"].shape[0])
    if checkpoint_skill_count != len(training_skills):
        raise ValueError("checkpoint skill-table prefix length differs from training skills")
    expected_skills_digest = str(
        (((payload.get("run_contract") or {}).get("inputs") or {}).get("skills_sha256"))
        or ""
    )
    observed_skills_digest = file_sha256(training_skills_path)
    if expected_skills_digest and expected_skills_digest != observed_skills_digest:
        raise ValueError("training skill file digest differs from the Stage2 run contract")

    config_values = dict(payload.get("config") or {})
    config_values["defer_skill_table_init"] = True
    config = CLSTRConfig(**config_values)
    if not bool(config.freeze_backbone):
        raise ValueError("canonical vNext evaluation requires its frozen backbone contract")
    snapshot_contract = (
        ((payload.get("run_contract") or {}).get("inputs") or {}).get(
            "frozen_backbone_snapshot_contract"
        )
    )
    if not isinstance(snapshot_contract, dict):
        raise ValueError("vNext checkpoint lacks its frozen-backbone snapshot contract")
    backbone_snapshot = verify_frozen_backbone_contract(
        snapshot_contract,
        expected_model_name_or_path=config.base_model_name,
    )
    if (
        backbone_snapshot["contract"] != snapshot_contract
        or str(backbone_snapshot["contract_digest"])
        != str(config.frozen_backbone_snapshot_digest or "")
    ):
        raise ValueError("vNext evaluation backbone snapshot identity changed")
    model = CLSTRModel(config, training_skills).to(device)
    load_report = load_compatible_state_dict(
        model,
        state,
        partial_load_mode="vnext_checkpoint_native_eval",
    )
    restore_contract = _evaluation_restore_contract(
        checkpoint_stage=str(checkpoint_stage),
        checkpoint_state_keys=state,
        model_state_keys=model.state_dict(),
        loaded_keys=load_report.get("loaded_keys") or [],
    )
    missing_required = restore_contract["missing_required_keys"]
    if missing_required:
        raise ValueError(
            "vNext evaluator did not restore required canonical parameters: "
            f"{missing_required[:8]}"
        )
    checkpoint_state_report["evaluation_restore_contract"] = restore_contract
    expected_static_digest = str(payload.get("static_foundation_digest") or "")
    observed_static_digest = static_foundation_digest(model)
    if not expected_static_digest or observed_static_digest != expected_static_digest:
        raise ValueError("vNext static foundation digest changed during evaluation restore")
    expected_candidate_digest = str(payload.get("candidate_foundation_digest") or "")
    observed_candidate_digest = candidate_foundation_digest(model)
    if checkpoint_stage in {
        "clstr_vnext_stage2",
        "clstr_vnext_candidate_compressor",
    } and (
        not expected_candidate_digest
        or observed_candidate_digest != expected_candidate_digest
    ):
        raise ValueError("vNext candidate foundation changed during evaluation restore")
    if checkpoint_stage == "clstr_vnext_stage0" and expected_candidate_digest:
        raise ValueError("Stage0 static diagnostic unexpectedly declares a candidate foundation")

    appended, verified_overlap_ids = _partition_benchmark_skill_append(
        training_skills,
        benchmark_skills,
        serializer=model.skill_table.skill_text_fn,
    )
    prefix_before = model.skill_table.E.detach().cpu().clone()
    with torch.no_grad():
        append_report = model.append_skills(appended)
    prefix_after = model.skill_table.E[: len(training_skills)].detach().cpu()
    if not torch.equal(prefix_before, prefix_after):
        raise RuntimeError("dynamic skill append changed the checkpoint skill prefix")
    appended_tensor = model.skill_table.E[len(training_skills) :].detach().float()
    if appended and (
        int(appended_tensor.size(0)) != len(appended)
        or not bool(torch.isfinite(appended_tensor).all())
        or bool((appended_tensor.norm(dim=-1) <= 0).any())
    ):
        raise RuntimeError("dynamic skill append produced invalid frozen embeddings")
    model._vnext_skill_embedding_cache.clear()
    model.eval()
    final_skills = [*training_skills, *appended]
    skill_id_to_idx = {
        _skill_id(row): index for index, row in enumerate(final_skills)
    }
    return model, final_skills, skill_id_to_idx, {
        "checkpoint_stage": checkpoint_stage,
        "stage0_static_diagnostic": checkpoint_stage == "clstr_vnext_stage0",
        "static_reranker_diagnostic": (
            checkpoint_stage == "clstr_vnext_candidate_compressor"
        ),
        "static_foundation_diagnostic": checkpoint_stage != "clstr_vnext_stage2",
        "expected_candidate_foundation_digest": expected_candidate_digest or None,
        "observed_candidate_foundation_digest": observed_candidate_digest,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(payload.get("step") or 0),
        "training_skills_path": str(training_skills_path.resolve()),
        "training_skills_sha256": observed_skills_digest,
        "checkpoint_skill_count": checkpoint_skill_count,
        "benchmark_skill_count": len(benchmark_skills),
        "final_skill_count": len(final_skills),
        "append": append_report,
        "verified_overlapping_skill_ids": verified_overlap_ids,
        "verified_overlapping_skill_count": len(verified_overlap_ids),
        "checkpoint_prefix_bit_exact_after_append": True,
        "static_foundation_digest": observed_static_digest,
        "candidate_foundation_digest": observed_candidate_digest,
        "checkpoint_state": checkpoint_state_report,
        "frozen_backbone_snapshot": backbone_snapshot,
        "load_report": load_report,
    }


def _executed_action_text(row: dict[str, Any]) -> str:
    target = str(row.get("next_skill_id") or "").strip()
    explicit = str(
        row.get("executed_action_text")
        or row.get("next_action_text")
        or ""
    ).strip()
    if explicit:
        return explicit
    observation_source = str(row.get("observation_source") or "").strip()
    arguments = (
        str(row.get("next_observation_text") or "").strip()
        if observation_source.startswith("oracle_next_")
        else ""
    )
    return (
        f"executed_skill: {target}\nexecuted_arguments: {arguments}"
        if arguments
        else f"executed_skill: {target}"
    )


def _aligned_actual_result(
    row: dict[str, Any],
    *,
    executed_skill_id: str | None = None,
) -> str:
    if not bool(row.get("actual_result_executed")):
        return ""
    target = str(executed_skill_id or row.get("next_skill_id") or "").strip()
    result_skill = str(
        row.get("actual_result_skill_id")
        or row.get("result_event_skill_id")
        or ""
    ).strip()
    if not result_skill or result_skill != target:
        return ""
    return str(row.get("actual_result_text") or "").strip()


def _benchmark_event_contract(
    benchmark: str,
    row: dict[str, Any],
) -> dict[str, str]:
    """Map one benchmark row to the event around its declared route decision.

    ToolBench transition rows describe an action/result in ``skill_id`` and ask
    for its successor in ``next_skill_id``.  That event must be replayed before
    the successor is scored.  ToolSandbox and Tau2 rows instead ask for the
    action that is about to execute, so their target event is replayed only
    after the current decision is scored.
    """

    benchmark = str(benchmark).strip().lower()
    if benchmark == "toolbench_g3":
        skill_id = str(row.get("skill_id") or "").strip()
        action_text = str(
            row.get("action_text")
            or row.get("expert_action")
            or ""
        ).strip()
        if not skill_id or not action_text:
            raise ValueError(
                "ToolBench successor routing requires its aligned executed skill/action"
            )
        result_text = _aligned_actual_result(
            row,
            executed_skill_id=skill_id,
        )
        alignment = "explicit_result_skill_id" if result_text else ""
        if not result_text:
            source = str(row.get("observation_source") or "").strip().lower()
            candidate = str(row.get("next_observation_text") or "").strip()
            if source in TOOLBENCH_EXECUTED_RESULT_SOURCES and candidate:
                result_text = candidate
                alignment = "toolbench_executed_row_schema"
        return {
            "timing": "before_decision",
            "skill_id": skill_id,
            "action_text": action_text,
            "result_text": result_text,
            "result_alignment": alignment,
        }

    target = str(row.get("next_skill_id") or "").strip()
    result_text = _aligned_actual_result(
        row,
        executed_skill_id=target,
    )
    return {
        "timing": "after_decision",
        "skill_id": target,
        "action_text": _executed_action_text(row),
        "result_text": result_text,
        "result_alignment": (
            "explicit_result_skill_id" if result_text else ""
        ),
    }


def _preflight_verified_toolbench_results(
    source_rows: list[dict[str, Any]],
    *,
    max_eval_rows: int | None,
) -> dict[str, Any]:
    """Verify exported ToolBench result identities before any model work."""

    selected = (
        source_rows[: max(0, int(max_eval_rows))]
        if max_eval_rows is not None
        else list(source_rows)
    )
    verified_rows = 0
    verified_nonempty_rows = 0
    for raw in selected:
        if str(raw.get("route_target") or "").upper() in {"STOP", "NO_CALL"}:
            continue
        skill_id = str(raw.get("skill_id") or "").strip()
        action_text = str(raw.get("action_text") or "").strip()
        result_text = str(raw.get("next_observation_text") or "")
        actual_text = str(raw.get("actual_result_text") or "")
        actual_skill = str(raw.get("actual_result_skill_id") or "").strip()
        actual_event_id = str(raw.get("actual_result_event_id") or "").strip()
        provenance = raw.get("provenance")
        answer_path_text = str(
            provenance.get("answer_path")
            if isinstance(provenance, dict)
            else ""
        ).strip()
        answer_path = (
            str(Path(answer_path_text).resolve())
            if answer_path_text
            else ""
        )
        expected_event_id = _json_digest(
            {
                "answer_path": answer_path,
                "action_text": action_text,
                "result_text": result_text,
                "skill_id": skill_id,
            }
        )
        if (
            not skill_id
            or not action_text
            or not answer_path
            or actual_skill != skill_id
            or actual_text != result_text
            or bool(raw.get("actual_result_executed")) != bool(result_text)
            or actual_event_id != expected_event_id
            or str(raw.get("observation_source") or "").strip().lower()
            != "executed_trace:toolbench_g3"
        ):
            raise ValueError(
                "ToolBench vNext evaluation requires exact verified executed-result "
                f"provenance for every row: task={raw.get('task_id')}"
            )
        verified_rows += 1
        verified_nonempty_rows += int(bool(result_text))
    return {
        "verified_row_count": int(verified_rows),
        "verified_nonempty_result_row_count": int(verified_nonempty_rows),
    }


def _prepare_eval_rows(
    *,
    benchmark: str,
    source_rows: list[dict[str, Any]],
    final_skill_ids: list[str],
    max_eval_rows: int | None,
    require_verified_toolbench_results: bool = False,
    toolbench_legal_skill_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    benchmark = str(benchmark).strip().lower()
    if benchmark not in VNEXT_BENCHMARKS:
        raise ValueError(f"unsupported vNext benchmark: {benchmark}")
    selected = (
        source_rows[: max(0, int(max_eval_rows))]
        if max_eval_rows is not None
        else list(source_rows)
    )
    final_set = set(final_skill_ids)
    routing_rows: list[dict[str, Any]] = []
    no_call_rows: list[dict[str, Any]] = []
    verified_result_report = (
        _preflight_verified_toolbench_results(
            selected,
            max_eval_rows=None,
        )
        if benchmark == "toolbench_g3" and require_verified_toolbench_results
        else {
            "verified_row_count": 0,
            "verified_nonempty_result_row_count": 0,
        }
    )
    for source_index, raw in enumerate(selected):
        if benchmark != "toolbench_g3" and not str(
            raw.get("state_text_current") or ""
        ).strip():
            raise ValueError(
                "vNext benchmark row lacks an explicit history-free state_text_current"
            )
        row = materialize_history_free_state(raw, replace_state_text=True)
        if not str(row.get("state_text_current") or "").strip():
            raise ValueError(
                "vNext ToolBench row could not materialize a history-free current state"
            )
        if str(row.get("route_target") or "").upper() in {"STOP", "NO_CALL"}:
            no_call_rows.append(row)
            continue
        target = str(row.get("next_skill_id") or "").strip()
        if not target:
            raise ValueError("routing evaluation row lacks next_skill_id")
        positive_ids = list(
            dict.fromkeys(
                [target]
                + [
                    str(item).strip()
                    for item in row.get("equivalent_next_skill_ids") or []
                    if str(item).strip()
                ]
            )
        )
        if benchmark == "toolbench_g3":
            if toolbench_legal_skill_ids is None:
                legal_ids = list(final_skill_ids)
                pool_protocol = "checkpoint_plus_benchmark_full_pool"
            else:
                legal_ids = list(dict.fromkeys(toolbench_legal_skill_ids))
                pool_protocol = "benchmark_declared_native_pool"
        else:
            legal_ids = [
                str(item)
                for item in row.get("candidate_next_skill_ids") or []
                if str(item)
            ]
            pool_protocol = "benchmark_row_local_pool"
        unknown = sorted(set(legal_ids) - final_set)
        if unknown:
            raise ValueError(f"evaluation legal pool contains unknown skills: {unknown[:4]}")
        legal_set = set(legal_ids)
        if not legal_ids or target not in legal_set:
            raise ValueError("evaluation target is absent from its decision-time legal pool")
        unknown_positive_ids = sorted(set(positive_ids) - final_set)
        if unknown_positive_ids:
            raise ValueError(
                "evaluation positive set contains unknown skills: "
                f"{unknown_positive_ids[:4]}"
            )
        illegal_positive_ids = sorted(set(positive_ids) - legal_set)
        if illegal_positive_ids:
            raise ValueError(
                "evaluation positive set escapes its decision-time legal pool: "
                f"{illegal_positive_ids[:4]}"
            )
        current = router_state_text(row).strip()
        if not current:
            raise ValueError("evaluation row has an empty current-state channel")
        copied = dict(row)
        copied["_vnext_source_index"] = source_index
        copied["_vnext_state_text"] = current
        copied["_vnext_target_skill_id"] = target
        copied["_vnext_positive_skill_ids"] = positive_ids
        copied["_vnext_legal_skill_ids"] = list(dict.fromkeys(legal_ids))
        copied["_vnext_pool_protocol"] = pool_protocol
        event = _benchmark_event_contract(benchmark, copied)
        if event["skill_id"] not in final_set:
            raise ValueError("evaluation event references an unknown executed skill")
        copied["_vnext_event_timing"] = event["timing"]
        copied["_vnext_executed_skill_id"] = event["skill_id"]
        copied["_vnext_executed_action_text"] = event["action_text"]
        copied["_vnext_actual_result_text"] = event["result_text"]
        copied["_vnext_result_alignment"] = event["result_alignment"]
        if benchmark == "toolbench_g3" and require_verified_toolbench_results:
            result_text = str(raw.get("next_observation_text") or "")
            if result_text and event["result_text"] != result_text:
                raise ValueError(
                    "ToolBench verified result was not attached to its executed event: "
                    f"task={raw.get('task_id')}"
                )
        routing_rows.append(copied)
    history_report = audit_history_channel_rows(
        routing_rows,
        require_explicit_current=True,
    )
    if history_report.get("status") != "ok":
        raise ValueError("benchmark current-state channel failed its history audit")
    return routing_rows, no_call_rows, {
        "source_row_count": len(selected),
        "routing_row_count": len(routing_rows),
        "no_call_row_count": len(no_call_rows),
        "history_channel": history_report,
        "pool_protocols": dict(
            sorted(Counter(row["_vnext_pool_protocol"] for row in routing_rows).items())
        ),
        "toolbench_verified_result_contract": {
            "required": bool(require_verified_toolbench_results),
            **verified_result_report,
        },
    }


def _legal_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        len(rows),
        len(skill_id_to_idx),
        dtype=torch.bool,
        device=device,
    )
    for row_index, row in enumerate(rows):
        indices = [skill_id_to_idx[item] for item in row["_vnext_legal_skill_ids"]]
        mask[row_index, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def _positive_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        len(rows),
        len(skill_id_to_idx),
        dtype=torch.bool,
        device=device,
    )
    for row_index, row in enumerate(rows):
        positive_ids = list(row.get("_vnext_positive_skill_ids") or [])
        if not positive_ids:
            raise ValueError("evaluation row lacks its positive skill set")
        indices = [skill_id_to_idx[item] for item in positive_ids]
        mask[row_index, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def _initial_memory(
    model: CLSTRModel,
    row: dict[str, Any],
    *,
    state_cache: Any,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
) -> torch.Tensor:
    state = state_cache.batch([row["_vnext_state_text"]], device=device)
    legal = _legal_mask([row], skill_id_to_idx, device=device)
    return model.vnext_initial_belief(state, legal, top_k=belief_top_k)


def _native_synchronization_enabled(model: Any) -> bool:
    return bool(
        getattr(getattr(model, "vnext", None), "synchronization_enabled", False)
    )


def _native_trace_length(model: Any) -> int:
    length = int(
        getattr(getattr(model, "vnext", None), "synchronization_trace_length", 0)
    )
    if length <= 0:
        raise ValueError("native synchronization trace length must be positive")
    return length


def _batch_factual_latent_trace(
    model: Any,
    batch_records: list[dict[str, Any]],
    *,
    foundation_selected: bool,
) -> torch.Tensor | None:
    """Prepare recurrent trace inputs, excluding static foundation scoring."""

    if foundation_selected or not _native_synchronization_enabled(model):
        return None
    factual_traces = [record.get("factual_latent_trace") for record in batch_records]
    if any(trace is None for trace in factual_traces):
        raise RuntimeError(
            "native synchronization evaluation record lacks its factual trace"
        )
    return torch.cat(factual_traces, dim=0)


def _apply_event(
    model: CLSTRModel,
    memory: torch.Tensor,
    event: dict[str, Any],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    enable_result_correction: bool,
    latent_trace: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    state = state_cache.batch([event["state_text_current"]], device=device)
    skill_index = skill_id_to_idx[event["skill_id"]]
    skill = model.vnext_normalized_skill_embeddings(dtype=state.dtype)[
        skill_index
    ].unsqueeze(0)
    action = action_cache.batch([event["action_text"]], device=device)
    result_text = (
        str(event.get("result_text") or "")
        if enable_result_correction
        else ""
    )
    result = (
        result_cache.batch([result_text], device=device)
        if result_text and result_cache is not None
        else None
    )
    update = model.vnext_update_memory(
        memory,
        state,
        skill,
        action,
        result_embedding=result,
        latent_trace=latent_trace,
    )
    # Keep the legacy tensor return when synchronization is disabled.  Native
    # evaluation opts into the trace explicitly and receives the exact same
    # post-action memory together with its fixed-width latent trace.
    if not _native_synchronization_enabled(model):
        return update.effective_memory
    if update.latent_trace is None:
        raise RuntimeError("native synchronization update did not return its trace")
    return update.effective_memory, update.latent_trace


def _replay_events(
    model: CLSTRModel,
    initial_row: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    enable_result_correction: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    memory = _initial_memory(
        model,
        initial_row,
        state_cache=state_cache,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
    )
    latent_trace = (
        memory.new_zeros(
            (
                1,
                _native_trace_length(model),
                int(memory.size(-1)),
            )
        )
        if _native_synchronization_enabled(model)
        else None
    )
    replayed_events: list[dict[str, Any]] = []
    for raw_event in events:
        event = dict(raw_event)
        current_state_text = str(event.get("state_text_current") or "").strip()
        if not current_state_text:
            raise ValueError("replayed evaluation event lacks its current-state channel")
        event["state_text"], _canonical_prefix = serialize_compact_causal_state(
            current_state_text,
            replayed_events,
        )
        updated = _apply_event(
            model,
            memory,
            event,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            enable_result_correction=enable_result_correction,
            latent_trace=latent_trace,
        )
        if latent_trace is None:
            memory = updated
        else:
            memory, latent_trace = updated
        replayed_events.append(event)
    if latent_trace is not None:
        return memory, latent_trace
    return memory


def _materialize_causal_eval_timeline(
    rows: list[dict[str, Any]],
    *,
    include_counterfactual_diagnostics: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "").strip()
        if not trajectory_id:
            raise ValueError("evaluation row lacks trajectory identity")
        grouped[trajectory_id].append(row)
    state_texts: list[str] = []
    decision_prefix_event_count = 0
    counterfactual_state_count = 0
    for trajectory_id, trajectory_rows in sorted(grouped.items()):
        ordered = sorted(
            trajectory_rows,
            key=lambda row: (
                int(row.get("step_index") or 0),
                int(row["_vnext_source_index"]),
            ),
        )
        prior_events: list[dict[str, Any]] = []
        previous_step: int | None = None
        for row in ordered:
            step = int(row.get("step_index") or 0)
            if previous_step is not None and step != previous_step + 1:
                prior_events = []
            current_state_text = str(row["_vnext_state_text"])
            pre_event_causal_state, _pre_event_prefix = serialize_compact_causal_state(
                current_state_text,
                prior_events,
            )
            event = {
                "state_text_current": current_state_text,
                "state_text": pre_event_causal_state,
                "skill_id": row["_vnext_executed_skill_id"],
                "action_text": row["_vnext_executed_action_text"],
                "result_text": row["_vnext_actual_result_text"],
                "result_executed": bool(row["_vnext_actual_result_text"]),
                "trajectory_id": trajectory_id,
                "step_index": step,
            }
            decision_prefix = list(prior_events)
            if row["_vnext_event_timing"] == "before_decision":
                decision_prefix.append(event)
            causal_state_text, causal_events = serialize_compact_causal_state(
                current_state_text,
                decision_prefix,
            )
            row["_vnext_event"] = event
            row["_vnext_decision_prefix_events"] = [
                dict(item) for item in decision_prefix
            ]
            row["_vnext_causal_state_text"] = causal_state_text
            row["_vnext_causal_prefix_event_count"] = len(causal_events)
            decision_prefix_event_count += len(causal_events)
            state_texts.extend(
                (current_state_text, pre_event_causal_state, causal_state_text)
            )
            if include_counterfactual_diagnostics and len(decision_prefix) >= 2:
                replayed_events: list[dict[str, Any]] = []
                for raw_event in reversed(decision_prefix):
                    replay_state, _replay_prefix = serialize_compact_causal_state(
                        str(raw_event["state_text_current"]),
                        replayed_events,
                    )
                    replay_event = dict(raw_event)
                    replay_event["state_text"] = replay_state
                    replayed_events.append(replay_event)
                    state_texts.append(replay_state)
                    counterfactual_state_count += 1
            if row["_vnext_event_timing"] == "before_decision":
                prior_events = decision_prefix
            else:
                prior_events.append(event)
            previous_step = step
    unique_state_texts = list(dict.fromkeys(state_texts))
    return unique_state_texts, {
        "trajectory_count": len(grouped),
        "routing_row_count": len(rows),
        "decision_prefix_event_count": decision_prefix_event_count,
        "counterfactual_state_count": counterfactual_state_count,
        "counterfactual_diagnostics_materialized": bool(
            include_counterfactual_diagnostics
        ),
        "unique_state_text_count": len(unique_state_texts),
        "protocol": "causal_state_before_executed_event_v2",
    }


def _materialize_memories(
    model: CLSTRModel,
    rows: list[dict[str, Any]],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    enable_result_correction: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "").strip()
        if not trajectory_id:
            raise ValueError("evaluation row lacks trajectory identity")
        grouped[trajectory_id].append(row)
    if any("_vnext_event" not in row for row in rows):
        _materialize_causal_eval_timeline(rows)
    records: list[dict[str, Any]] = []
    correction_count = 0
    recurrent_count = 0
    order_count = 0
    with torch.no_grad():
        for trajectory_id, trajectory_rows in sorted(grouped.items()):
            ordered = sorted(
                trajectory_rows,
                key=lambda row: (
                    int(row.get("step_index") or 0),
                    int(row["_vnext_source_index"]),
                ),
            )
            initial_row = ordered[0]
            memory = _initial_memory(
                model,
                initial_row,
                state_cache=state_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
            )
            latent_trace = (
                memory.new_zeros(
                    (
                        1,
                        _native_trace_length(model),
                        int(memory.size(-1)),
                    )
                )
                if _native_synchronization_enabled(model)
                else None
            )
            previous_step: int | None = None
            observed_result_correction_count = 0
            for row in ordered:
                step = int(row.get("step_index") or 0)
                if previous_step is not None and step != previous_step + 1:
                    memory = _initial_memory(
                        model,
                        row,
                        state_cache=state_cache,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        belief_top_k=belief_top_k,
                    )
                    latent_trace = (
                        memory.new_zeros(
                            (
                                1,
                                _native_trace_length(model),
                                int(memory.size(-1)),
                            )
                        )
                        if _native_synchronization_enabled(model)
                        else None
                    )
                    initial_row = row
                    observed_result_correction_count = 0
                event = dict(row["_vnext_event"])
                decision_prefix = [
                    dict(item) for item in row["_vnext_decision_prefix_events"]
                ]
                if row["_vnext_event_timing"] == "before_decision":
                    updated = _apply_event(
                        model,
                        memory,
                        event,
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        enable_result_correction=enable_result_correction,
                        latent_trace=latent_trace,
                    )
                    if latent_trace is None:
                        memory = updated
                    else:
                        memory, latent_trace = updated
                    applied_correction = int(
                        enable_result_correction and bool(event["result_text"])
                    )
                    correction_count += applied_correction
                    observed_result_correction_count += applied_correction
                factual = memory.detach().cpu()
                order_memory = None
                event_digests = [_json_digest(item) for item in decision_prefix]
                if len(decision_prefix) >= 2 and event_digests != list(
                    reversed(event_digests)
                ):
                    replayed_order = _replay_events(
                        model,
                        initial_row,
                        list(reversed(decision_prefix)),
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        belief_top_k=belief_top_k,
                        enable_result_correction=enable_result_correction,
                    )
                    order_trace = None
                    if _native_synchronization_enabled(model):
                        order_memory, order_trace = replayed_order
                        order_memory = order_memory.detach().cpu()
                        order_trace = order_trace.detach().cpu()
                    else:
                        order_memory = replayed_order.detach().cpu()
                    order_count += 1
                else:
                    order_trace = None
                records.append(
                    {
                        "row": row,
                        "factual_memory": factual,
                        "factual_latent_trace": (
                            None
                            if latent_trace is None
                            else latent_trace.detach().cpu()
                        ),
                        "order_memory": order_memory,
                        "order_latent_trace": order_trace,
                        "replay_prefix_length": len(decision_prefix),
                        "uses_recurrent_m_t": bool(decision_prefix),
                        "observed_result_correction_count": int(
                            observed_result_correction_count
                        ),
                        "uses_observation_supported_m_t": bool(
                            decision_prefix and observed_result_correction_count > 0
                        ),
                        "event": event,
                    }
                )
                recurrent_count += int(bool(decision_prefix))
                if row["_vnext_event_timing"] == "after_decision":
                    applied_correction = int(
                        enable_result_correction and bool(event["result_text"])
                    )
                    correction_count += applied_correction
                    updated = _apply_event(
                        model,
                        memory,
                        event,
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        enable_result_correction=enable_result_correction,
                        latent_trace=latent_trace,
                    )
                    if latent_trace is None:
                        memory = updated
                    else:
                        memory, latent_trace = updated
                    observed_result_correction_count += applied_correction
                previous_step = step

    donor_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        row = record["row"]
        key = (
            str(row.get("domain") or row.get("scenario_group") or ""),
            int(record["replay_prefix_length"]),
            _json_digest(sorted(row["_vnext_legal_skill_ids"])),
        )
        donor_groups[key].append(record)
    mismatch_count = 0
    for record in records:
        row = record["row"]
        key = (
            str(row.get("domain") or row.get("scenario_group") or ""),
            int(record["replay_prefix_length"]),
            _json_digest(sorted(row["_vnext_legal_skill_ids"])),
        )
        donors = sorted(
            (
                candidate
                for candidate in donor_groups[key]
                if str(candidate["row"].get("trajectory_id") or "")
                != str(row.get("trajectory_id") or "")
                and set(candidate["row"]["_vnext_positive_skill_ids"]).isdisjoint(
                    row["_vnext_positive_skill_ids"]
                )
            ),
            key=lambda candidate: int(candidate["row"]["_vnext_source_index"]),
        )
        record["mismatch_memory"] = (
            donors[0]["factual_memory"] if donors else None
        )
        record["mismatch_latent_trace"] = (
            donors[0]["factual_latent_trace"] if donors else None
        )
        record["mismatch_donor_source_index"] = (
            int(donors[0]["row"]["_vnext_source_index"]) if donors else None
        )
        record["mismatch_observed_result_correction_count"] = (
            int(donors[0]["observed_result_correction_count"])
            if donors
            else 0
        )
        mismatch_count += int(bool(donors))
    records.sort(key=lambda record: int(record["row"]["_vnext_source_index"]))
    return records, {
        "trajectory_count": len(grouped),
        "routing_row_count": len(records),
        "native_synchronization_enabled": _native_synchronization_enabled(model),
        "native_synchronization_trace_length": (
            _native_trace_length(model)
            if _native_synchronization_enabled(model)
            else None
        ),
        "verified_result_correction_enabled": bool(
            enable_result_correction
        ),
        "uses_recurrent_m_t_count": recurrent_count,
        "observation_supported_route_count": sum(
            int(record["uses_observation_supported_m_t"]) for record in records
        ),
        "actual_result_correction_count": correction_count,
        "order_shuffle_eligible_count": order_count,
        "mismatch_eligible_count": mismatch_count,
    }


def _materialize_closed_set_foundation_records(
    model: CLSTRModel,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create shape-compatible records without executing recurrent Stage2 heads."""

    if any("_vnext_event" not in row for row in rows):
        _materialize_causal_eval_timeline(
            rows,
            include_counterfactual_diagnostics=False,
        )
    memory_width = int(model.vnext.d)
    zero_memory = torch.zeros((1, memory_width), dtype=torch.float32)
    records: list[dict[str, Any]] = []
    for row in rows:
        decision_prefix = row["_vnext_decision_prefix_events"]
        records.append(
            {
                "row": row,
                "factual_memory": zero_memory.clone(),
                "factual_latent_trace": None,
                "order_memory": None,
                "order_latent_trace": None,
                "replay_prefix_length": len(decision_prefix),
                "uses_recurrent_m_t": False,
                "observed_result_correction_count": 0,
                "uses_observation_supported_m_t": False,
                "event": dict(row["_vnext_event"]),
                "mismatch_memory": None,
                "mismatch_latent_trace": None,
                "mismatch_donor_source_index": None,
                "mismatch_observed_result_correction_count": 0,
            }
        )
    records.sort(key=lambda record: int(record["row"]["_vnext_source_index"]))
    return records, {
        "trajectory_count": len(
            {
                str(row.get("trajectory_id") or row.get("task_id") or "")
                for row in rows
            }
        ),
        "routing_row_count": len(records),
        "source_replay_prefix_row_count": sum(
            int(bool(record["replay_prefix_length"])) for record in records
        ),
        "uses_recurrent_m_t_count": 0,
        "observation_supported_route_count": 0,
        "actual_result_correction_count": 0,
        "order_shuffle_eligible_count": 0,
        "mismatch_eligible_count": 0,
        "closed_set_recurrent_path_executed": False,
    }


def _full_pool_rank(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    target_scores = logits.gather(1, target_indices.unsqueeze(-1)).squeeze(-1)
    skill_ids = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)
    return 1 + (
        legal
        & (
            (logits > target_scores.unsqueeze(-1))
            | (
                logits.eq(target_scores.unsqueeze(-1))
                & skill_ids.lt(target_indices.unsqueeze(-1))
            )
        )
    ).sum(dim=-1)


def _candidate_rank(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    valid: torch.Tensor,
    target_indices: torch.Tensor,
) -> list[int | None]:
    matches = valid & candidate_ids.eq(target_indices.unsqueeze(-1))
    output: list[int | None] = []
    for row_index in range(int(logits.size(0))):
        positions = matches[row_index].nonzero(as_tuple=False).view(-1)
        if int(positions.numel()) != 1:
            output.append(None)
            continue
        position = int(positions[0])
        target_score = logits[row_index, position]
        columns = torch.arange(logits.size(1), device=logits.device)
        rank = 1 + int(
            (
                valid[row_index]
                & (
                    (logits[row_index] > target_score)
                    | (
                        logits[row_index].eq(target_score)
                        & columns.lt(position)
                    )
                )
            )
            .sum()
            .cpu()
            .item()
        )
        output.append(rank)
    return output


def _candidate_multi_positive_rank(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
) -> list[int | None]:
    if (
        logits.ndim != 2
        or candidate_ids.shape != logits.shape
        or valid.shape != logits.shape
        or positive.ndim != 2
        or positive.size(0) != logits.size(0)
    ):
        raise ValueError(
            "candidate multi-positive ranking tensors have incompatible shapes"
        )
    matches = valid & positive.gather(1, candidate_ids)
    output: list[int | None] = []
    columns = torch.arange(logits.size(1), device=logits.device)
    for row_index in range(int(logits.size(0))):
        positions = matches[row_index].nonzero(as_tuple=False).view(-1)
        if not int(positions.numel()):
            output.append(None)
            continue
        scores = logits[row_index].index_select(0, positions)
        best_score = scores.max()
        best_position = int(positions[scores.eq(best_score)].min().item())
        rank = 1 + int(
            (
                valid[row_index]
                & (
                    (logits[row_index] > best_score)
                    | (
                        logits[row_index].eq(best_score)
                        & columns.lt(best_position)
                    )
                )
            )
            .sum()
            .cpu()
            .item()
        )
        output.append(rank)
    return output


def _metrics(ranks: list[int | None]) -> dict[str, float | int]:
    count = len(ranks)
    return {
        "count": count,
        "recall@1": (
            sum(int(rank is not None and rank <= 1) for rank in ranks) / count
            if count
            else 0.0
        ),
        "recall@5": (
            sum(int(rank is not None and rank <= 5) for rank in ranks) / count
            if count
            else 0.0
        ),
        "recall@10": (
            sum(int(rank is not None and rank <= 10) for rank in ranks) / count
            if count
            else 0.0
        ),
        "recall@100": (
            sum(int(rank is not None and rank <= 100) for rank in ranks) / count
            if count
            else 0.0
        ),
        "recall@500": (
            sum(int(rank is not None and rank <= 500) for rank in ranks) / count
            if count
            else 0.0
        ),
        "mrr": (
            sum(1.0 / rank for rank in ranks if rank is not None) / count
            if count
            else 0.0
        ),
    }


def _truncate_ranks(
    ranks: list[int | None],
    *,
    final_k: int,
) -> list[int | None]:
    if int(final_k) <= 0:
        raise ValueError("final_k must be positive")
    return [
        rank if rank is not None and int(rank) <= int(final_k) else None
        for rank in ranks
    ]


def _candidate_recall_query(
    model: CLSTRModel,
    states: torch.Tensor,
    causal_states: torch.Tensor,
    causal_memory: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "raw_current":
        return F.normalize(states.float(), p=2, dim=-1).to(states.dtype)
    if mode == "raw_causal":
        return F.normalize(causal_states.float(), p=2, dim=-1).to(
            causal_states.dtype
        )
    if mode == "learned_causal":
        query, _unused_route = model.vnext.static_queries(
            causal_states,
            causal_memory,
        )
        return query
    raise ValueError(f"unsupported candidate query mode: {mode}")


def _equal_reciprocal_rank_fusion_logits(
    static_logits: torch.Tensor,
    semantic_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse two independent rankings without score calibration or weights."""

    if static_logits.ndim != 2 or semantic_logits.shape != static_logits.shape:
        raise ValueError("closed-set RRF logits must match [batch, candidates]")
    if valid_mask.shape != static_logits.shape:
        raise ValueError("closed-set RRF validity must match candidate logits")
    if apply_mask.ndim != 1 or int(apply_mask.numel()) != int(static_logits.size(0)):
        raise ValueError("closed-set RRF apply mask must have one value per row")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    if bool((~valid.any(dim=-1)).any().item()):
        raise ValueError("closed-set RRF requires at least one valid candidate per row")

    def ranks(logits: torch.Tensor) -> torch.Tensor:
        values = logits.float().masked_fill(~valid, float("-inf"))
        order = torch.argsort(
            values,
            dim=-1,
            descending=True,
            stable=True,
        )
        positions = torch.arange(
            1,
            int(logits.size(1)) + 1,
            device=logits.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand_as(order)
        output = torch.empty_like(positions)
        output.scatter_(1, order, positions)
        return output

    static_ranks = ranks(static_logits)
    semantic_ranks = ranks(semantic_logits)
    fused = static_ranks.reciprocal() + semantic_ranks.reciprocal()
    selected = torch.where(
        apply_mask.to(device=static_logits.device, dtype=torch.bool).unsqueeze(-1),
        fused,
        static_logits.float(),
    )
    return selected.masked_fill(~valid, torch.finfo(torch.float32).min)


def _preserved_foundation_candidate_logits(
    model: CLSTRModel,
    states: torch.Tensor,
    legal: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    belief_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the immutable foundation under its held-out-approved protocol."""

    belief = model.vnext_initial_belief(
        states,
        legal,
        top_k=int(belief_top_k),
    )
    static_recall, static_route_delta, _static_route = (
        model.vnext.static_query_components(states, belief)
    )
    static_logits = model.vnext_candidate_logits(
        static_recall,
        candidate_ids,
        candidate_valid,
        head="route",
    ) + model.vnext_candidate_logits(
        static_route_delta,
        candidate_ids,
        candidate_valid,
        head="route",
    )
    semantic_logits = model.vnext_candidate_logits(
        F.normalize(states.float(), p=2, dim=-1).to(states.dtype),
        candidate_ids,
        candidate_valid,
        head="route",
    )
    fusion_applied = candidate_valid.sum(dim=-1).gt(
        CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE
    )
    return (
        _equal_reciprocal_rank_fusion_logits(
            static_logits,
            semantic_logits,
            candidate_valid,
            fusion_applied,
        ),
        fusion_applied,
    )


def _score_alternative(
    model: CLSTRModel,
    rows: list[dict[str, Any]],
    memories: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    latent_trace: torch.Tensor | None = None,
    history_depth: torch.Tensor,
    observation_correction_count: torch.Tensor,
    state_cache: Any,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    dynamic_extra_k: int,
    use_static_query: bool,
    use_h_only: bool = False,
    candidate_query_mode: str = "learned_causal",
    candidate_compression_mode: str = "learned",
    candidate_foundation: dict[str, torch.Tensor] | None = None,
    final_k: int = 100,
    static_only_execution: bool = False,
    foundation_model: CLSTRModel | None = None,
    unified_router: UnifiedThreeExpertRouter | None = None,
    unified_routing_mode: str | None = None,
    support_aware_anchor: bool = False,
) -> dict[str, Any]:
    if static_only_execution and not use_static_query:
        raise ValueError("static-only execution requires the static query path")
    if not (
        history_depth.ndim == observation_correction_count.ndim == 1
        and history_depth.shape == observation_correction_count.shape
        and int(history_depth.numel()) == len(rows)
    ):
        raise ValueError(
            "history depth and observation correction count must match eval rows"
        )
    if (
        bool((history_depth < 0).any().item())
        or bool((observation_correction_count < 0).any().item())
        or bool((observation_correction_count > history_depth).any().item())
    ):
        raise ValueError("invalid causal observation-correction metadata")
    if candidate_foundation is None:
        current_states = state_cache.batch(
            [row["_vnext_state_text"] for row in rows],
            device=device,
        )
        causal_states = state_cache.batch(
            [row["_vnext_causal_state_text"] for row in rows],
            device=device,
        )
        legal = _legal_mask(rows, skill_id_to_idx, device=device)
        positive = _positive_mask(rows, skill_id_to_idx, device=device)
        target_indices = torch.tensor(
            [skill_id_to_idx[row["_vnext_target_skill_id"]] for row in rows],
            dtype=torch.long,
            device=device,
        )
    else:
        current_states = candidate_foundation["current_states"]
        causal_states = candidate_foundation["causal_states"]
        legal = candidate_foundation["legal"]
        positive = candidate_foundation["positive"]
        target_indices = candidate_foundation["target_indices"]
        if int(causal_states.size(0)) != len(rows):
            raise ValueError("candidate foundation batch does not match eval rows")
    route_states = causal_states
    if latent_trace is not None:
        latent_trace = latent_trace.to(device=device, dtype=route_states.dtype)
    route_history_mask = (
        torch.zeros_like(history_mask, device=device, dtype=torch.bool)
        if use_static_query
        else history_mask.to(device=device, dtype=torch.bool)
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if candidate_foundation is None:
            static_memory = model.vnext_initial_belief(
                route_states,
                legal,
                top_k=belief_top_k,
            )
        else:
            static_memory = candidate_foundation["static_memory"]
        queries = None
        static_route_components = None
        if candidate_query_mode == "learned_causal":
            if static_only_execution:
                static_route_components = model.vnext.static_query_components(
                    route_states,
                    static_memory,
                )
                recall_query = static_route_components[0]
            else:
                queries = model.vnext_queries(
                    route_states,
                    memories.to(device=device, dtype=route_states.dtype),
                    static_memory,
                    route_history_mask,
                    memory_state=current_states,
                    latent_trace=latent_trace,
                )
                recall_query = (
                    queries.static_recall
                    if use_static_query
                    else queries.dynamic_recall
                )
        else:
            recall_query = _candidate_recall_query(
                model,
                current_states,
                causal_states,
                static_memory,
                mode=candidate_query_mode,
            )
        if candidate_compression_mode == "direct":
            if int(compressed_m) != int(coarse_k):
                raise ValueError(
                    "direct candidate support requires compressed_m == coarse_k"
                )
            recall_logits = model.vnext_full_pool_logits(recall_query, head="recall")
            coarse_ids, coarse_valid = masked_topk_tensor(
                recall_logits,
                legal,
                k=coarse_k,
            )
            direct = direct_natural_support(
                coarse_ids,
                coarse_valid,
                skill_count=int(recall_logits.size(1)),
            )
            candidate_ids = direct.candidate_ids
            candidate_valid = direct.valid_mask
            static_candidate_ids = direct.coarse_candidate_ids
            static_candidate_valid = direct.coarse_valid_mask
        elif candidate_compression_mode == "learned":
            if static_only_execution and candidate_query_mode == "learned_causal":
                recall_logits = model.vnext_full_pool_logits(
                    recall_query,
                    head="recall",
                )
                coarse_ids, coarse_valid = masked_topk_tensor(
                    recall_logits,
                    legal,
                    k=coarse_k,
                )
                direct = direct_natural_support(
                    coarse_ids,
                    coarse_valid,
                    skill_count=int(recall_logits.size(1)),
                )
                candidate_ids = direct.candidate_ids
                candidate_valid = direct.valid_mask
                static_candidate_ids = direct.coarse_candidate_ids
                static_candidate_valid = direct.coarse_valid_mask
            elif queries is not None:
                path = model.vnext_natural_candidate_union_path(
                    queries.static_recall,
                    recall_query,
                    current_states,
                    legal,
                    route_history_mask,
                    coarse_k=coarse_k,
                    dynamic_extra_k=dynamic_extra_k,
                )
            else:
                path = model.vnext_natural_candidate_path(
                    recall_query,
                    current_states,
                    legal,
                    coarse_k=coarse_k,
                    compressed_m=compressed_m,
                )
            if not static_only_execution or candidate_query_mode != "learned_causal":
                recall_logits = path.recall_logits
                candidate_ids = path.support.candidate_ids
                candidate_valid = path.support.valid_mask
                static_candidate_ids = path.coarse_candidate_ids
                static_candidate_valid = path.coarse_valid_mask
        else:
            raise ValueError(
                "unsupported candidate compression mode: "
                f"{candidate_compression_mode}"
            )
        route_scores = None
        closed_set_semantic_fusion_applied = torch.zeros(
            len(rows),
            dtype=torch.bool,
            device=device,
        )
        if static_only_execution:
            if static_route_components is None:
                static_route_components = model.vnext.static_query_components(
                    route_states,
                    static_memory,
                )
            static_recall, static_route_delta, _static_route = (
                static_route_components
            )
            route_logits = model.vnext_candidate_logits(
                static_recall,
                candidate_ids,
                candidate_valid,
                head="route",
            ) + model.vnext_candidate_logits(
                static_route_delta,
                candidate_ids,
                candidate_valid,
                head="route",
            )
            semantic_route_logits = model.vnext_candidate_logits(
                F.normalize(route_states.float(), p=2, dim=-1).to(
                    route_states.dtype
                ),
                candidate_ids,
                candidate_valid,
                head="route",
            )
            closed_set_semantic_fusion_applied = candidate_valid.sum(dim=-1).gt(
                CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE
            )
            route_logits = _equal_reciprocal_rank_fusion_logits(
                route_logits,
                semantic_route_logits,
                candidate_valid,
                closed_set_semantic_fusion_applied,
            )
        elif use_h_only:
            route_logits = model.vnext_candidate_logits(
                F.normalize(route_states.float(), p=2, dim=-1).to(
                    route_states.dtype
                ),
                candidate_ids,
                candidate_valid,
                head="route",
            )
        else:
            route_scores = model.vnext_safe_candidate_route_scores(
                route_states,
                memories.to(device=device, dtype=route_states.dtype),
                static_memory,
                route_history_mask,
                candidate_ids,
                candidate_valid,
                static_candidate_ids=static_candidate_ids,
                static_valid_mask=static_candidate_valid,
                history_depth=history_depth,
                memory_state=current_states,
                latent_trace=latent_trace,
                hard_fallback=True,
            )
            route_logits = route_scores.mixed_logits
        raw_dynamic_route_logits = (
            route_logits
            if route_scores is None
            else route_scores.raw_dynamic_logits
        )
        unified_expert_weights = torch.zeros(
            (len(rows), len(UNIFIED_EXPERT_NAMES)),
            dtype=torch.float32,
            device=device,
        )
        unified_router_features = torch.zeros(
            (len(rows), len(UNIFIED_ROUTER_FEATURE_NAMES)),
            dtype=torch.float32,
            device=device,
        )
        foundation_route_logits = route_logits
        unified_foundation_semantic_fusion_applied = torch.zeros(
            len(rows),
            dtype=torch.bool,
            device=device,
        )
        support_aware_effective_history_zero = not support_aware_anchor
        support_aware_zero_history_memory_weight_zero = not support_aware_anchor
        support_aware_incomplete_route_exact = torch.ones(
            len(rows),
            dtype=torch.bool,
            device=device,
        )
        if (
            foundation_model is not None
            or unified_router is not None
            or support_aware_anchor
        ):
            if foundation_model is None:
                raise ValueError("multi-expert routing requires its foundation model")
            if support_aware_anchor and unified_router is not None:
                raise ValueError(
                    "support-aware anchoring may not use the learned selector"
                )
            if not support_aware_anchor and unified_router is None:
                raise ValueError(
                    "learned unified routing requires both foundation model and router"
                )
            if route_scores is None:
                raise ValueError(
                    "multi-expert routing requires adapted-static and recurrent scores"
                )
            if (
                not support_aware_anchor
                and unified_routing_mode not in UNIFIED_ROUTING_MODES
            ):
                raise ValueError("unified routing requires a declared routing mode")
            foundation_states = state_cache.batch_with_projection(
                [row["_vnext_causal_state_text"] for row in rows],
                device=device,
                projection_fn=foundation_model.encoder.project_pooled,
            )
            (
                foundation_route_logits,
                unified_foundation_semantic_fusion_applied,
            ) = _preserved_foundation_candidate_logits(
                foundation_model,
                foundation_states,
                legal,
                candidate_ids,
                candidate_valid,
                belief_top_k=belief_top_k,
            )
            effective_history_depth = torch.where(
                route_history_mask,
                history_depth.to(device=device, dtype=torch.float32),
                torch.zeros_like(
                    history_depth.to(device=device, dtype=torch.float32)
                ),
            )
            effective_observation_correction_count = torch.where(
                route_history_mask,
                observation_correction_count.to(
                    device=device,
                    dtype=torch.float32,
                ),
                torch.zeros_like(
                    observation_correction_count.to(
                        device=device,
                        dtype=torch.float32,
                    )
                ),
            )
            if support_aware_anchor:
                support_aware_effective_history_zero = bool(
                    effective_history_depth.eq(0).all().item()
                )
            if support_aware_anchor:
                unified = route_with_support_aware_anchor(
                    foundation_route_logits,
                    route_scores.static_logits,
                    route_scores.raw_dynamic_logits,
                    candidate_valid,
                    static_support_mask=route_scores.static_support_mask,
                    history_depth=effective_history_depth,
                    observation_correction_count=(
                        effective_observation_correction_count
                    ),
                    legal_pool_size=legal.sum(dim=-1).float(),
                )
            else:
                unified = route_with_unified_three_experts(
                    unified_router,
                    foundation_route_logits,
                    route_scores.static_logits,
                    route_scores.raw_dynamic_logits,
                    candidate_valid,
                    static_support_mask=route_scores.static_support_mask,
                    history_depth=effective_history_depth,
                    observation_correction_count=(
                        effective_observation_correction_count
                    ),
                    legal_pool_size=legal.sum(dim=-1).float(),
                    routing_mode=unified_routing_mode,
                )
            route_logits = unified.logits
            unified_expert_weights = unified.expert_weights.float()
            unified_router_features = unified.features.float()
            if support_aware_anchor:
                effective_history = effective_history_depth.gt(0)
                complete_support = candidate_valid.sum(dim=-1).eq(
                    legal.sum(dim=-1)
                )
                score_floor = torch.finfo(torch.float32).min
                expected_static = route_scores.static_logits.float().masked_fill(
                    ~route_scores.static_support_mask,
                    score_floor,
                )
                expected_recurrent = (
                    route_scores.raw_dynamic_logits.float().masked_fill(
                        ~candidate_valid,
                        score_floor,
                    )
                )
                expected_incomplete = torch.where(
                    effective_history.unsqueeze(-1),
                    expected_recurrent,
                    expected_static,
                )
                support_aware_incomplete_route_exact = complete_support | route_logits.eq(
                    expected_incomplete
                ).all(dim=-1)
                zero_history = ~effective_history
                support_aware_zero_history_memory_weight_zero = bool(
                    unified_expert_weights[zero_history, 2].eq(0.0).all().item()
                )
    recall_ranks = _multi_positive_rank(recall_logits, positive, legal)
    union_route_ranks = _candidate_multi_positive_rank(
        route_logits,
        candidate_ids,
        candidate_valid,
        positive,
    )
    union_raw_dynamic_route_ranks = _candidate_multi_positive_rank(
        raw_dynamic_route_logits,
        candidate_ids,
        candidate_valid,
        positive,
    )
    union_foundation_route_ranks = _candidate_multi_positive_rank(
        foundation_route_logits,
        candidate_ids,
        candidate_valid,
        positive,
    )
    route_ranks_at_64 = _truncate_ranks(union_route_ranks, final_k=64)
    route_ranks_at_100 = _truncate_ranks(union_route_ranks, final_k=100)
    route_ranks = _truncate_ranks(union_route_ranks, final_k=final_k)
    raw_dynamic_route_ranks = _truncate_ranks(
        union_raw_dynamic_route_ranks,
        final_k=final_k,
    )
    foundation_route_ranks = _truncate_ranks(
        union_foundation_route_ranks,
        final_k=final_k,
    )
    def _top_skill_indices(logits: torch.Tensor) -> list[list[int]]:
        ranked: list[list[int]] = []
        for row_index in range(len(rows)):
            valid_positions = candidate_valid[row_index].nonzero(
                as_tuple=False
            ).view(-1)
            if not int(valid_positions.numel()):
                ranked.append([])
                continue
            scores = logits[row_index].index_select(0, valid_positions)
            width = min(10, int(scores.numel()))
            order = torch.argsort(
                scores,
                descending=True,
                stable=True,
            )[:width]
            positions = valid_positions.index_select(0, order)
            ranked.append(
                [
                    int(candidate_ids[row_index, position].cpu().item())
                    for position in positions
                ]
            )
        return ranked

    top_ids = _top_skill_indices(route_logits)
    foundation_top_ids = _top_skill_indices(foundation_route_logits)
    adapted_static_top_ids = _top_skill_indices(
        route_logits if route_scores is None else route_scores.static_logits
    )
    recurrent_memory_top_ids = _top_skill_indices(raw_dynamic_route_logits)
    return {
        "candidate_foundation": {
            "current_states": current_states,
            "causal_states": causal_states,
            "legal": legal,
            "positive": positive,
            "target_indices": target_indices,
            "static_memory": static_memory,
            "recall_logits": recall_logits,
            "candidate_ids": candidate_ids,
            "candidate_valid": candidate_valid,
        },
        "recall_ranks": recall_ranks,
        "route_ranks": route_ranks,
        "route_ranks_at_64": route_ranks_at_64,
        "route_ranks_at_100": route_ranks_at_100,
        "union_route_ranks": union_route_ranks,
        "raw_dynamic_route_ranks": raw_dynamic_route_ranks,
        "foundation_route_ranks": foundation_route_ranks,
        "candidate_recalled": [rank is not None for rank in union_route_ranks],
        "coarse_recalled_at_100": [
            rank is not None and rank <= 100 for rank in recall_ranks
        ],
        "coarse_recalled_at_500": [
            rank is not None and rank <= int(coarse_k) for rank in recall_ranks
        ],
        "candidate_skill_indices": [
            candidate_ids[row_index]
            .masked_select(candidate_valid[row_index])
            .cpu()
            .tolist()
            for row_index in range(len(rows))
        ],
        "top_skill_indices": top_ids,
        "foundation_top_skill_indices": foundation_top_ids,
        "adapted_static_top_skill_indices": adapted_static_top_ids,
        "recurrent_memory_top_skill_indices": recurrent_memory_top_ids,
        "recall_delta_rms": (
            torch.zeros(len(rows), dtype=torch.float32).tolist()
            if queries is None
            else queries.recall_delta.float()
            .pow(2)
            .mean(dim=-1)
            .sqrt()
            .cpu()
            .tolist()
        ),
        "route_delta_rms": (
            torch.zeros(len(rows), dtype=torch.float32).tolist()
            if route_scores is None
            else route_scores.raw_residual.float()
            .pow(2)
            .mean(dim=-1)
            .sqrt()
            .cpu()
            .tolist()
        ),
        "masked_equals_static": bool(
            static_only_execution
            or (
                route_scores is not None
                and torch.equal(
                    route_scores.mixed_logits,
                    route_scores.static_logits,
                )
                and torch.equal(
                    route_scores.raw_dynamic_logits,
                    route_scores.static_logits,
                )
            )
        )
        if not bool(history_mask.any())
        else None,
        "mixture_probability": (
            torch.zeros(len(rows), dtype=torch.float32).tolist()
            if route_scores is None
            else route_scores.mixture_probability.float().cpu().tolist()
        ),
        "unified_expert_weights": unified_expert_weights.cpu().tolist(),
        "unified_router_features": unified_router_features.cpu().tolist(),
        "foundation_route_logits_used": bool(foundation_model is not None),
        "unified_foundation_semantic_fusion_applied": (
            unified_foundation_semantic_fusion_applied.cpu().tolist()
        ),
        "unified_memory_weight_zero": bool(
            unified_expert_weights[:, 2].eq(0.0).all().item()
        ),
        "support_aware_effective_history_zero": bool(
            support_aware_effective_history_zero
        ),
        "support_aware_zero_history_memory_weight_zero": bool(
            support_aware_zero_history_memory_weight_zero
        ),
        "support_aware_incomplete_route_exact": (
            support_aware_incomplete_route_exact.cpu().tolist()
        ),
        "closed_set_semantic_fusion_applied": (
            closed_set_semantic_fusion_applied.cpu().tolist()
        ),
    }


def run_vnext_benchmark_eval(
    *,
    benchmark: str,
    corpus: Any,
    checkpoint_path: str | Path,
    training_skills_path: str | Path,
    output_dir: str | Path,
    max_eval_rows: int | None = None,
    coarse_k: int = 500,
    compressed_m: int = 500,
    dynamic_extra_k: int | None = None,
    final_k: int = 100,
    candidate_query_mode: str = "learned_causal",
    candidate_compression_mode: str = "direct",
    score_batch_size: int = 32,
    cache_batch_size: int = 128,
    belief_top_k: int = 64,
    frozen_cache_dir: str | Path | None = None,
    require_verified_toolbench_results: bool = False,
    allow_stage0_static_diagnostic: bool = False,
    allow_static_reranker_diagnostic: bool = False,
    closed_set_foundation_checkpoint_path: str | Path | None = None,
    stage2_release_selection_path: str | Path | None = None,
    unified_router_path: str | Path | None = None,
    support_aware_anchor: bool = False,
    disable_verified_result_correction: bool = False,
    toolbench_pool_scope: str = "global",
    toolbench_native_skills_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark = str(benchmark).strip().lower()
    if benchmark == "toolbench":
        benchmark = "toolbench_g3"
    if benchmark not in VNEXT_BENCHMARKS:
        raise ValueError(f"unsupported vNext benchmark: {benchmark}")
    toolbench_pool_scope = str(toolbench_pool_scope).strip().lower()
    if toolbench_pool_scope not in {"global", "native"}:
        raise ValueError(
            f"unsupported ToolBench pool scope: {toolbench_pool_scope}"
        )
    if benchmark != "toolbench_g3" and toolbench_pool_scope != "global":
        raise ValueError(
            "ToolBench native pool scope is valid only for ToolBench evaluation"
        )
    if toolbench_pool_scope == "global" and toolbench_native_skills_path is not None:
        raise ValueError(
            "ToolBench native skill IDs are valid only for native-pool evaluation"
        )
    candidate_query_mode = str(candidate_query_mode).strip().lower()
    candidate_compression_mode = str(candidate_compression_mode).strip().lower()
    resolved_dynamic_extra_k = (
        int(compressed_m)
        if dynamic_extra_k is None
        else int(dynamic_extra_k)
    )
    support_aware_anchor = bool(support_aware_anchor)
    disable_verified_result_correction = bool(
        disable_verified_result_correction
    )
    if support_aware_anchor and unified_router_path is not None:
        raise ValueError(
            "support-aware anchoring and the learned unified router are mutually exclusive"
        )
    if candidate_query_mode not in {"learned_causal", "raw_current", "raw_causal"}:
        raise ValueError(f"unsupported candidate query mode: {candidate_query_mode}")
    if candidate_compression_mode not in {"learned", "direct"}:
        raise ValueError(
            f"unsupported candidate compression mode: {candidate_compression_mode}"
        )
    if int(coarse_k) != 500 or int(score_batch_size) <= 0:
        raise ValueError(
            "vNext support evaluation requires coarse_k=500 and a positive "
            "score batch size"
        )
    if candidate_compression_mode == "learned" and int(compressed_m) != 64:
        raise ValueError("learned compression requires compressed_m=64")
    if candidate_compression_mode == "direct" and int(compressed_m) != int(coarse_k):
        raise ValueError("direct support requires compressed_m=coarse_k=500")
    if candidate_query_mode == "learned_causal" and candidate_compression_mode == "learned":
        if not 0 <= resolved_dynamic_extra_k <= int(compressed_m):
            raise ValueError(
                "learned causal union requires 0 <= dynamic_extra_k <= compressed_m"
            )
    elif resolved_dynamic_extra_k != int(compressed_m):
        raise ValueError(
            "dynamic_extra_k ablation is valid only for the learned causal union"
        )
    if resolved_dynamic_extra_k not in {0, int(compressed_m)}:
        raise ValueError(
            "paper evaluation supports only the full union or K_e=0 ablation"
        )
    fixed_support_ablation = bool(
        candidate_query_mode == "learned_causal"
        and candidate_compression_mode == "learned"
        and resolved_dynamic_extra_k == 0
    )
    if fixed_support_ablation and disable_verified_result_correction:
        raise ValueError("component ablations must remove exactly one mechanism")
    component_ablation = (
        "without_dynamic_only_candidates_ke0"
        if fixed_support_ablation
        else (
            "without_verified_result_correction"
            if disable_verified_result_correction
            else None
        )
    )
    if not 0 < int(final_k) <= int(coarse_k) + max(
        0,
        resolved_dynamic_extra_k,
    ):
        raise ValueError("final_k must fit within the natural candidate union")
    evaluator_root = Path(__file__).resolve().parents[1]
    evaluator_source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=evaluator_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if support_aware_anchor:
        evaluator_source_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=evaluator_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if evaluator_source_status:
            raise ValueError(
                "support-aware evaluation requires a clean immutable source"
            )
    source_rows = list(corpus.source_rows)
    benchmark_skills = list(corpus.skills)
    requested_stage2_checkpoint_path = Path(checkpoint_path)
    release_selection_contract = (
        _stage2_release_selection_contract(
            stage2_release_selection_path,
            stage2_checkpoint_path=requested_stage2_checkpoint_path,
        )
        if stage2_release_selection_path is not None
        else None
    )
    matched_release = bool(
        release_selection_contract is not None
        and release_selection_contract.get("schema_version")
        == MATCHED_RELEASE_SELECTION_SCHEMA
    )
    if matched_release:
        if str(release_selection_contract.get("source_commit") or "") != str(
            evaluator_source_commit
        ):
            raise ValueError(
                "matched release selection was finalized from a different source commit"
            )
        evaluator_source_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=evaluator_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if evaluator_source_status:
            raise ValueError("matched release evaluation requires a clean immutable source")
        if (
            closed_set_foundation_checkpoint_path is not None
            or unified_router_path is not None
            or support_aware_anchor
        ):
            raise ValueError(
                "matched unified recurrent release may not use deployment dispatch"
            )
    if (
        release_selection_contract is not None
        and bool(release_selection_contract.get("closed_set_dispatch_required"))
        and closed_set_foundation_checkpoint_path is None
    ):
        raise ValueError(
            "open-pool release selection requires the preserved closed-set foundation"
        )
    if unified_router_path is not None and closed_set_foundation_checkpoint_path is None:
        raise ValueError(
            "unified three-expert routing requires its foundation checkpoint"
        )
    if unified_router_path is not None and release_selection_contract is None:
        raise ValueError(
            "unified three-expert routing requires its Stage2 release selection"
        )
    if support_aware_anchor and closed_set_foundation_checkpoint_path is None:
        raise ValueError(
            "support-aware anchoring requires its foundation checkpoint"
        )
    if support_aware_anchor and release_selection_contract is None:
        raise ValueError(
            "support-aware anchoring requires its Stage2 release selection"
        )
    foundation_contract: dict[str, Any] | None = None
    interface_decision: dict[str, Any] | None = None
    foundation_selected = False
    if closed_set_foundation_checkpoint_path is not None:
        foundation_contract = _foundation_preservation_contract(
            stage2_checkpoint_path=requested_stage2_checkpoint_path,
            foundation_checkpoint_path=closed_set_foundation_checkpoint_path,
            training_skills_path=training_skills_path,
        )
        declared_open_pool_size = (
            int(foundation_contract["checkpoint_skill_count"])
            if benchmark == "toolbench_g3"
            else None
        )
        interface_decision = (
            {
                "protocol": SUPPORT_AWARE_ANCHOR_PROTOCOL,
                "selected_expert": "sample_level_support_aware_anchor_residual",
                "reason": (
                    "natural_support_completeness_selects_dynamic_or_semantic_anchor"
                ),
                "coarse_k": int(coarse_k),
                "foundation_protected_top_k": (
                    SUPPORT_AWARE_FOUNDATION_PREFIX_K
                ),
                "uses_benchmark_identity": False,
                "uses_interface_dispatch": False,
            }
            if support_aware_anchor
            else (
                {
                    "protocol": "sample_level_unified_sparse_three_expert_route_v6",
                    "selected_expert": "learned_per_sample_sparse_top1",
                    "reason": "foundation_static_memory_scores_available_on_every_row",
                    "coarse_k": int(coarse_k),
                    "uses_benchmark_identity": False,
                    "uses_interface_dispatch": False,
                }
                if unified_router_path is not None
                else _interface_expert_decision(
                    source_rows,
                    coarse_k=coarse_k,
                    declared_open_pool_size=declared_open_pool_size,
                    declared_pool_protocol=(
                        f"toolbench_{toolbench_pool_scope}_pool"
                        if benchmark == "toolbench_g3"
                        else "row_declared_decision_time_legal_pool"
                    ),
                )
            )
        )
        foundation_selected = bool(
            not support_aware_anchor
            and unified_router_path is None
            and interface_decision["selected_expert"]
            == "preserved_closed_set_foundation"
        )
        if foundation_selected:
            checkpoint_path = Path(closed_set_foundation_checkpoint_path)
            allow_stage0_static_diagnostic = True
        else:
            checkpoint_path = requested_stage2_checkpoint_path
    native_pool_contract: dict[str, Any] | None = None
    toolbench_legal_skill_ids: list[str] | None = None
    if benchmark == "toolbench_g3" and toolbench_pool_scope == "native":
        if toolbench_native_skills_path is None:
            raise ValueError(
                "ToolBench native evaluation requires its declared skill-ID file"
            )
        native_path = Path(toolbench_native_skills_path)
        native_rows = _read_jsonl(native_path)
        toolbench_legal_skill_ids = [_skill_id(row) for row in native_rows]
        if (
            len(toolbench_legal_skill_ids) != TOOLBENCH_G3_NATIVE_SKILL_COUNT
            or len(set(toolbench_legal_skill_ids))
            != TOOLBENCH_G3_NATIVE_SKILL_COUNT
            or any(not skill_id for skill_id in toolbench_legal_skill_ids)
        ):
            raise ValueError(
                "ToolBench native evaluation requires the exact declared "
                f"{TOOLBENCH_G3_NATIVE_SKILL_COUNT}-skill ID pool; observed "
                f"{len(toolbench_legal_skill_ids)} rows and "
                f"{len(set(toolbench_legal_skill_ids))} unique IDs"
            )
        native_pool_contract = {
            "path": str(native_path.resolve()),
            "sha256": file_sha256(native_path),
            "skill_count": len(toolbench_legal_skill_ids),
            "skill_ids_sha256": _json_digest(sorted(toolbench_legal_skill_ids)),
            "role": "legal_candidate_ids_only",
        }
    if benchmark == "toolbench_g3" and require_verified_toolbench_results:
        _preflight_verified_toolbench_results(
            source_rows,
            max_eval_rows=max_eval_rows,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("vNext benchmark evaluation requires a CUDA Slurm node")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, skills, skill_id_to_idx, checkpoint_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=checkpoint_path,
            training_skills_path=training_skills_path,
            benchmark_skills=benchmark_skills,
            device=device,
            allow_stage0_static_diagnostic=allow_stage0_static_diagnostic,
            allow_static_reranker_diagnostic=allow_static_reranker_diagnostic,
        )
    )
    foundation_model: CLSTRModel | None = None
    foundation_checkpoint_report: dict[str, Any] | None = None
    unified_router: UnifiedThreeExpertRouter | None = None
    unified_router_contract: dict[str, Any] | None = None
    unified_routing_mode: str | None = None
    if unified_router_path is not None or support_aware_anchor:
        (
            foundation_model,
            foundation_skills,
            foundation_skill_id_to_idx,
            foundation_checkpoint_report,
        ) = load_vnext_stage2_for_evaluation(
            checkpoint_path=closed_set_foundation_checkpoint_path,
            training_skills_path=training_skills_path,
            benchmark_skills=benchmark_skills,
            device=device,
            allow_stage0_static_diagnostic=True,
        )
        if foundation_skill_id_to_idx != skill_id_to_idx or [
            _skill_id(row) for row in foundation_skills
        ] != [_skill_id(row) for row in skills]:
            raise ValueError("unified expert skill indices differ")
        foundation_model.eval()
        # Appended skill embeddings have already been materialized. Runtime
        # states reuse the Stage2 model's pre-projection pooled cache, so the
        # second identical frozen backbone is no longer needed on the GPU.
        foundation_model.encoder.backbone = torch.nn.Identity()
        torch.cuda.empty_cache()
        if unified_router_path is not None:
            unified_router, unified_router_contract = _load_unified_router_contract(
                unified_router_path,
                stage2_checkpoint_path=requested_stage2_checkpoint_path,
                foundation_checkpoint_path=closed_set_foundation_checkpoint_path,
                training_skills_path=training_skills_path,
                stage2_release_selection_contract=release_selection_contract,
                device=device,
            )
            unified_routing_mode = str(unified_router_contract["routing_mode"])
    skill_ids = [_skill_id(row) for row in skills]
    rows, no_call_rows, row_report = _prepare_eval_rows(
        benchmark=benchmark,
        source_rows=source_rows,
        final_skill_ids=skill_ids,
        max_eval_rows=max_eval_rows,
        require_verified_toolbench_results=require_verified_toolbench_results,
        toolbench_legal_skill_ids=toolbench_legal_skill_ids,
    )
    if not rows:
        raise ValueError("vNext benchmark evaluation has no routing rows")
    closed_set_semantic_fusion_expected_count = (
        sum(
            int(
                len(row["_vnext_legal_skill_ids"])
                > CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE
            )
            for row in rows
        )
        if foundation_selected
        else 0
    )
    corpus_source_contract = _corpus_source_contract(benchmark, corpus)
    cache_root = (
        Path(frozen_cache_dir)
        if frozen_cache_dir is not None
        else output_dir / "frozen_qwen_cache"
    )
    causal_state_texts, causal_timeline_report = _materialize_causal_eval_timeline(
        rows,
        include_counterfactual_diagnostics=not foundation_selected,
    )
    state_cache = load_or_build_frozen_text_cache(
        model,
        causal_state_texts,
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
    )
    action_cache = (
        None
        if foundation_selected
        else load_or_build_frozen_text_cache(
            model,
            (row["_vnext_executed_action_text"] for row in rows),
            role="action",
            batch_size=cache_batch_size,
            cache_root=cache_root,
        )
    )
    result_texts = [row["_vnext_actual_result_text"] for row in rows if row["_vnext_actual_result_text"]]
    result_cache = (
        load_or_build_frozen_text_cache(
            model,
            result_texts,
            role="result",
            batch_size=cache_batch_size,
            cache_root=cache_root,
        )
        if (
            result_texts
            and not foundation_selected
            and not disable_verified_result_correction
        )
        else None
    )
    if foundation_selected:
        memory_records, memory_report = _materialize_closed_set_foundation_records(
            model,
            rows,
        )
    else:
        memory_records, memory_report = _materialize_memories(
            model,
            rows,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            enable_result_correction=(
                not disable_verified_result_correction
            ),
        )

    scored_records: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[Any]]] = {
        name: {
            "recall": [],
            "route": [],
            "raw_dynamic_route": [],
            "foundation_route": [],
            "route_at_64": [],
            "route_at_100": [],
            "union_route": [],
            "candidate": [],
            "mixture_probability": [],
            "unified_expert_weights": [],
        }
        for name in (
            "h_only",
            "static",
            "factual",
            "masked",
            "mismatch",
            "order_shuffle",
        )
    }
    diagnostic_eligibility: dict[str, list[bool]] = {
        "mismatch": [],
        "order_shuffle": [],
    }
    exact_masked_fallback = True
    exact_masked_candidate_support = True
    fixed_support_all_exact = True
    memory_changed_candidate_support = False
    closed_set_semantic_fusion_applied_count = 0
    support_aware_complete_support_count = 0
    support_aware_incomplete_support_count = 0
    support_aware_foundation_semantic_fusion_count = 0
    support_aware_foundation_semantic_fusion_expected_count = 0
    support_aware_foundation_prefix_preserved_count = 0
    support_aware_foundation_prefix_expected_count = 0
    support_aware_incomplete_route_exact_count = 0
    support_aware_incomplete_route_expected_count = 0
    for start in range(0, len(memory_records), int(score_batch_size)):
        batch_records = memory_records[start : start + int(score_batch_size)]
        batch_rows = [record["row"] for record in batch_records]
        factual_memories = torch.cat(
            [record["factual_memory"] for record in batch_records],
            dim=0,
        )
        factual_latent_trace = _batch_factual_latent_trace(
            model,
            batch_records,
            foundation_selected=foundation_selected,
        )
        history_mask = torch.tensor(
            [bool(record["uses_recurrent_m_t"]) for record in batch_records],
            dtype=torch.bool,
        )
        deployment_history_mask = (
            torch.zeros_like(history_mask)
            if foundation_selected
            else history_mask
        )
        history_depth = torch.tensor(
            [float(record["replay_prefix_length"]) for record in batch_records],
            dtype=torch.float32,
        )
        observation_correction_count = torch.tensor(
            [
                float(record["observed_result_correction_count"])
                for record in batch_records
            ],
            dtype=torch.float32,
        )
        zero_observation_correction_count = torch.zeros_like(
            observation_correction_count
        )
        h_only_scored = _score_alternative(
            model,
            batch_rows,
            factual_memories,
            torch.zeros_like(history_mask),
            history_depth=history_depth,
            observation_correction_count=zero_observation_correction_count,
            state_cache=state_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            coarse_k=coarse_k,
            compressed_m=(coarse_k if foundation_selected else compressed_m),
            dynamic_extra_k=(
                coarse_k if foundation_selected else resolved_dynamic_extra_k
            ),
            use_static_query=True,
            use_h_only=True,
            candidate_query_mode="raw_causal",
            candidate_compression_mode=(
                "direct" if foundation_selected else candidate_compression_mode
            ),
            final_k=final_k,
        )
        static_scored = _score_alternative(
            model,
            batch_rows,
            factual_memories,
            torch.zeros_like(history_mask),
            history_depth=history_depth,
            observation_correction_count=zero_observation_correction_count,
            state_cache=state_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            coarse_k=coarse_k,
            compressed_m=compressed_m,
            dynamic_extra_k=resolved_dynamic_extra_k,
            use_static_query=True,
            candidate_query_mode=candidate_query_mode,
            candidate_compression_mode=candidate_compression_mode,
            final_k=final_k,
            static_only_execution=foundation_selected,
        )
        closed_set_semantic_fusion_applied_count += sum(
            int(value)
            for value in static_scored["closed_set_semantic_fusion_applied"]
        )
        if foundation_selected:
            factual_scored = static_scored
            masked_scored = static_scored
        else:
            factual_scored = _score_alternative(
                model,
                batch_rows,
                factual_memories,
                deployment_history_mask,
                latent_trace=factual_latent_trace,
                history_depth=history_depth,
                observation_correction_count=observation_correction_count,
                state_cache=state_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
                dynamic_extra_k=resolved_dynamic_extra_k,
                use_static_query=False,
                candidate_foundation=static_scored["candidate_foundation"],
                final_k=final_k,
                foundation_model=foundation_model,
                unified_router=unified_router,
                unified_routing_mode=unified_routing_mode,
                support_aware_anchor=support_aware_anchor,
            )
            masked_scored = _score_alternative(
                model,
                batch_rows,
                factual_memories,
                torch.zeros_like(deployment_history_mask),
                latent_trace=factual_latent_trace,
                history_depth=history_depth,
                observation_correction_count=(
                    zero_observation_correction_count
                ),
                state_cache=state_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
                dynamic_extra_k=resolved_dynamic_extra_k,
                use_static_query=False,
                candidate_foundation=static_scored["candidate_foundation"],
                final_k=final_k,
                foundation_model=foundation_model,
                unified_router=unified_router,
                unified_routing_mode=unified_routing_mode,
                support_aware_anchor=support_aware_anchor,
            )
        if support_aware_anchor:
            complete_support_flags = [
                len(factual_scored["candidate_skill_indices"][index])
                == len(row["_vnext_legal_skill_ids"])
                for index, row in enumerate(batch_rows)
            ]
            support_aware_complete_support_count += sum(
                int(value) for value in complete_support_flags
            )
            support_aware_incomplete_support_count += sum(
                int(not value) for value in complete_support_flags
            )
            foundation_prefix_preserved_flags = []
            for index, row in enumerate(batch_rows):
                if not complete_support_flags[index]:
                    foundation_prefix_preserved_flags.append(False)
                    continue
                prefix_width = min(
                    SUPPORT_AWARE_FOUNDATION_PREFIX_K,
                    len(row["_vnext_legal_skill_ids"]),
                )
                foundation_prefix_preserved_flags.append(
                    factual_scored["top_skill_indices"][index][:prefix_width]
                    == factual_scored["foundation_top_skill_indices"][index][
                        :prefix_width
                    ]
                )
            support_aware_foundation_prefix_preserved_count += sum(
                int(value) for value in foundation_prefix_preserved_flags
            )
            support_aware_foundation_prefix_expected_count += sum(
                int(value) for value in complete_support_flags
            )
            support_aware_incomplete_route_exact_count += sum(
                int(not complete_support_flags[index]) and int(value)
                for index, value in enumerate(
                    factual_scored["support_aware_incomplete_route_exact"]
                )
            )
            support_aware_incomplete_route_expected_count += sum(
                int(not value) for value in complete_support_flags
            )
            support_aware_foundation_semantic_fusion_count += sum(
                int(complete_support_flags[index])
                and int(value)
                for index, value in enumerate(
                    factual_scored[
                        "unified_foundation_semantic_fusion_applied"
                    ]
                )
            )
            support_aware_foundation_semantic_fusion_expected_count += sum(
                int(complete_support_flags[index])
                and int(
                    len(row["_vnext_legal_skill_ids"])
                    > CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE
                )
                for index, row in enumerate(batch_rows)
            )
        exact_masked_fallback = exact_masked_fallback and bool(
            (
                masked_scored["support_aware_effective_history_zero"]
                and masked_scored[
                    "support_aware_zero_history_memory_weight_zero"
                ]
            )
            if support_aware_anchor
            else (
                masked_scored["unified_memory_weight_zero"]
                if unified_router is not None
                else masked_scored["masked_equals_static"]
            )
        )
        exact_masked_candidate_support = exact_masked_candidate_support and bool(
            masked_scored["candidate_skill_indices"]
            == static_scored["candidate_skill_indices"]
        )
        if fixed_support_ablation:
            fixed_support_all_exact = fixed_support_all_exact and bool(
                factual_scored["candidate_skill_indices"]
                == static_scored["candidate_skill_indices"]
            )
        if foundation_selected:
            mismatch_valid = [False] * len(batch_records)
            mismatch_scored = static_scored
            order_valid = [False] * len(batch_records)
            order_scored = static_scored
        else:
            mismatch_valid = [
                record["mismatch_memory"] is not None for record in batch_records
            ]
            mismatch_memories = torch.cat(
                [
                    record["mismatch_memory"]
                    if record["mismatch_memory"] is not None
                    else record["factual_memory"]
                    for record in batch_records
                ],
                dim=0,
            )
            mismatch_latent_trace = None
            if _native_synchronization_enabled(model):
                mismatch_traces = [
                    record.get("mismatch_latent_trace")
                    if record.get("mismatch_latent_trace") is not None
                    else record.get("factual_latent_trace")
                    for record in batch_records
                ]
                if any(trace is None for trace in mismatch_traces):
                    raise RuntimeError(
                        "native synchronization evaluation record lacks its mismatch trace"
                    )
                mismatch_latent_trace = torch.cat(mismatch_traces, dim=0)
            mismatch_observation_correction_count = torch.tensor(
                [
                    float(record["mismatch_observed_result_correction_count"])
                    if valid
                    else 0.0
                    for record, valid in zip(batch_records, mismatch_valid)
                ],
                dtype=torch.float32,
            )
            mismatch_scored = _score_alternative(
                model,
                batch_rows,
                mismatch_memories,
                torch.tensor(mismatch_valid, dtype=torch.bool),
                latent_trace=mismatch_latent_trace,
                history_depth=history_depth,
                observation_correction_count=(
                    mismatch_observation_correction_count
                ),
                state_cache=state_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
                dynamic_extra_k=resolved_dynamic_extra_k,
                use_static_query=False,
                candidate_foundation=static_scored["candidate_foundation"],
                final_k=final_k,
                foundation_model=foundation_model,
                unified_router=unified_router,
                unified_routing_mode=unified_routing_mode,
                support_aware_anchor=support_aware_anchor,
            )
            order_valid = [
                record["order_memory"] is not None for record in batch_records
            ]
            order_memories = torch.cat(
                [
                    record["order_memory"]
                    if record["order_memory"] is not None
                    else record["factual_memory"]
                    for record in batch_records
                ],
                dim=0,
            )
            order_latent_trace = None
            if _native_synchronization_enabled(model):
                order_traces = [
                    record.get("order_latent_trace")
                    if record.get("order_latent_trace") is not None
                    else record.get("factual_latent_trace")
                    for record in batch_records
                ]
                if any(trace is None for trace in order_traces):
                    raise RuntimeError(
                        "native synchronization evaluation record lacks its order trace"
                    )
                order_latent_trace = torch.cat(order_traces, dim=0)
            order_observation_correction_count = torch.where(
                torch.tensor(order_valid, dtype=torch.bool),
                observation_correction_count,
                zero_observation_correction_count,
            )
            order_scored = _score_alternative(
                model,
                batch_rows,
                order_memories,
                torch.tensor(order_valid, dtype=torch.bool),
                latent_trace=order_latent_trace,
                history_depth=history_depth,
                observation_correction_count=(
                    order_observation_correction_count
                ),
                state_cache=state_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
                dynamic_extra_k=resolved_dynamic_extra_k,
                use_static_query=False,
                candidate_foundation=static_scored["candidate_foundation"],
                final_k=final_k,
                foundation_model=foundation_model,
                unified_router=unified_router,
                unified_routing_mode=unified_routing_mode,
                support_aware_anchor=support_aware_anchor,
            )
        alternatives = {
            "h_only": h_only_scored,
            "static": static_scored,
            "factual": factual_scored,
            "masked": masked_scored,
            "mismatch": mismatch_scored,
            "order_shuffle": order_scored,
        }
        memory_changed_candidate_support = memory_changed_candidate_support or any(
            bool(deployment_history_mask[index].item())
            and set(factual_scored["candidate_skill_indices"][index])
            != set(static_scored["candidate_skill_indices"][index])
            for index in range(len(batch_rows))
        )
        diagnostic_eligibility["mismatch"].extend(mismatch_valid)
        diagnostic_eligibility["order_shuffle"].extend(order_valid)
        for name, scored in alternatives.items():
            aggregate[name]["recall"].extend(scored["recall_ranks"])
            aggregate[name]["route"].extend(scored["route_ranks"])
            aggregate[name]["raw_dynamic_route"].extend(
                scored["raw_dynamic_route_ranks"]
            )
            aggregate[name]["foundation_route"].extend(
                scored["foundation_route_ranks"]
            )
            aggregate[name]["route_at_64"].extend(scored["route_ranks_at_64"])
            aggregate[name]["route_at_100"].extend(scored["route_ranks_at_100"])
            aggregate[name]["union_route"].extend(scored["union_route_ranks"])
            aggregate[name]["candidate"].extend(scored["candidate_recalled"])
            aggregate[name]["mixture_probability"].extend(
                scored["mixture_probability"]
            )
            aggregate[name]["unified_expert_weights"].extend(
                scored["unified_expert_weights"]
            )
        for local_index, record in enumerate(batch_records):
            row = record["row"]
            output = {
                "schema_version": VNEXT_EVAL_SCHEMA,
                "benchmark": benchmark,
                "source_index": int(row["_vnext_source_index"]),
                "task_id": str(row.get("task_id") or ""),
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "step_index": int(row.get("step_index") or 0),
                "domain": str(row.get("domain") or row.get("scenario_group") or ""),
                "target_skill_id": row["_vnext_target_skill_id"],
                "positive_skill_ids": row["_vnext_positive_skill_ids"],
                "legal_pool_size": len(row["_vnext_legal_skill_ids"]),
                "pool_protocol": row["_vnext_pool_protocol"],
                "replay_prefix_length": int(record["replay_prefix_length"]),
                "uses_recurrent_m_t": bool(record["uses_recurrent_m_t"]),
                "deployment_uses_recurrent_m_t": bool(
                    deployment_history_mask[local_index].item()
                ),
                "deployment_uses_closed_set_semantic_fusion": bool(
                    factual_scored[
                        "unified_foundation_semantic_fusion_applied"
                    ][local_index]
                    if support_aware_anchor
                    else static_scored["closed_set_semantic_fusion_applied"][
                        local_index
                    ]
                ),
                "uses_observation_supported_m_t": bool(
                    record["uses_observation_supported_m_t"]
                ),
                "observed_result_correction_count": int(
                    record["observed_result_correction_count"]
                ),
                "uses_actual_result_correction": bool(
                    record["uses_observation_supported_m_t"]
                ),
                "current_event_has_actual_result": bool(
                    row["_vnext_actual_result_text"]
                ),
                "event_timing": row["_vnext_event_timing"],
                "executed_skill_id": row["_vnext_executed_skill_id"],
                "result_alignment": row["_vnext_result_alignment"],
                "mismatch_donor_source_index": record["mismatch_donor_source_index"],
                "mismatch_observed_result_correction_count": int(
                    record["mismatch_observed_result_correction_count"]
                ),
                "order_shuffle_eligible": bool(order_valid[local_index]),
            }
            for name, scored in alternatives.items():
                output[name] = {
                    "full_pool_recall_rank": scored["recall_ranks"][local_index],
                    "end_to_end_route_rank": scored["route_ranks"][local_index],
                    "end_to_end_route_rank_at_64": scored["route_ranks_at_64"][local_index],
                    "end_to_end_route_rank_at_100": scored["route_ranks_at_100"][local_index],
                    "candidate_union_route_rank": scored["union_route_ranks"][local_index],
                    "raw_dynamic_end_to_end_route_rank": scored[
                        "raw_dynamic_route_ranks"
                    ][local_index],
                    "foundation_end_to_end_route_rank": scored[
                        "foundation_route_ranks"
                    ][local_index],
                    "unified_foundation_semantic_fusion_applied": bool(
                        scored["unified_foundation_semantic_fusion_applied"][
                            local_index
                        ]
                    ),
                    "candidate_recalled_at_m": bool(
                        scored["candidate_recalled"][local_index]
                    ),
                    "compressed_candidate_recalled_at_64": bool(
                        scored["route_ranks_at_64"][local_index] is not None
                    ),
                    "candidate_recalled_at_final_k": bool(
                        scored["route_ranks"][local_index] is not None
                    ),
                    "unified_expert_weights": scored[
                        "unified_expert_weights"
                    ][local_index],
                    "unified_router_features": scored[
                        "unified_router_features"
                    ][local_index],
                    "top_skill_ids": [
                        skill_ids[index]
                        for index in scored["top_skill_indices"][local_index]
                    ],
                    "foundation_top_skill_ids": [
                        skill_ids[index]
                        for index in scored["foundation_top_skill_indices"][
                            local_index
                        ]
                    ],
                    "adapted_static_top_skill_ids": [
                        skill_ids[index]
                        for index in scored["adapted_static_top_skill_indices"][
                            local_index
                        ]
                    ],
                    "recurrent_memory_top_skill_ids": [
                        skill_ids[index]
                        for index in scored[
                            "recurrent_memory_top_skill_indices"
                        ][local_index]
                    ],
                    "support_aware_incomplete_route_exact": bool(
                        scored["support_aware_incomplete_route_exact"][
                            local_index
                        ]
                    ),
                    "recall_delta_rms": float(scored["recall_delta_rms"][local_index]),
                    "route_delta_rms": float(scored["route_delta_rms"][local_index]),
                    "mixture_probability": float(
                        scored["mixture_probability"][local_index]
                    ),
                }
            scored_records.append(output)

    metrics: dict[str, Any] = {}
    for name in (
        "h_only",
        "static",
        "factual",
        "masked",
        "mismatch",
        "order_shuffle",
    ):
        eligibility = (
            diagnostic_eligibility[name]
            if name in diagnostic_eligibility
            else [True] * len(rows)
        )
        selected = [index for index, eligible in enumerate(eligibility) if eligible]
        metrics[name] = {
            "eligible_row_count": len(selected),
            "full_pool_recall": _metrics(
                [aggregate[name]["recall"][index] for index in selected]
            ),
            "end_to_end_route": _metrics(
                [aggregate[name]["route"][index] for index in selected]
            ),
            "end_to_end_route_final64": _metrics(
                [aggregate[name]["route_at_64"][index] for index in selected]
            ),
            "end_to_end_route_final100": _metrics(
                [aggregate[name]["route_at_100"][index] for index in selected]
            ),
            "candidate_union_route": _metrics(
                [aggregate[name]["union_route"][index] for index in selected]
            ),
            "raw_dynamic_end_to_end_route": _metrics(
                [aggregate[name]["raw_dynamic_route"][index] for index in selected]
            ),
            "foundation_end_to_end_route": _metrics(
                [aggregate[name]["foundation_route"][index] for index in selected]
            ),
            "candidate_union_recall": (
                sum(int(aggregate[name]["candidate"][index]) for index in selected)
                / len(selected)
                if selected
                else 0.0
            ),
            "mixture_probability": {
                "mean": (
                    sum(
                        float(aggregate[name]["mixture_probability"][index])
                        for index in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "nonzero_rate": (
                    sum(
                        int(
                            float(aggregate[name]["mixture_probability"][index])
                            > 0.0
                        )
                        for index in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "at_least_half_rate": (
                    sum(
                        int(
                            float(aggregate[name]["mixture_probability"][index])
                            >= 0.5
                        )
                        for index in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
            },
            "unified_expert_weights": {
                expert_name: (
                    sum(
                        float(
                            aggregate[name]["unified_expert_weights"][index][
                                expert_index
                            ]
                        )
                        for index in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                )
                for expert_index, expert_name in enumerate(UNIFIED_EXPERT_NAMES)
            },
        }
    metrics["factual_minus_static"] = {
        "full_pool_recall_mrr": float(metrics["factual"]["full_pool_recall"]["mrr"])
        - float(metrics["static"]["full_pool_recall"]["mrr"]),
        "end_to_end_route_mrr": float(metrics["factual"]["end_to_end_route"]["mrr"])
        - float(metrics["static"]["end_to_end_route"]["mrr"]),
        "end_to_end_route_final64_mrr": float(
            metrics["factual"]["end_to_end_route_final64"]["mrr"]
        ) - float(metrics["static"]["end_to_end_route_final64"]["mrr"]),
        "end_to_end_route_final100_mrr": float(
            metrics["factual"]["end_to_end_route_final100"]["mrr"]
        ) - float(metrics["static"]["end_to_end_route_final100"]["mrr"]),
        "raw_dynamic_end_to_end_route_mrr": float(
            metrics["factual"]["raw_dynamic_end_to_end_route"]["mrr"]
        )
        - float(metrics["static"]["raw_dynamic_end_to_end_route"]["mrr"]),
        "candidate_union_recall": float(
            metrics["factual"]["candidate_union_recall"]
        )
        - float(metrics["static"]["candidate_union_recall"]),
    }
    deployment_metric_source = "static" if foundation_selected else "factual"
    metrics["deployment"] = metrics[deployment_metric_source]

    records_path = output_dir / f"{benchmark}_vnext_route_records.jsonl"
    _write_jsonl(records_path, scored_records)
    checkpoint_is_static_foundation = bool(
        checkpoint_report.get("static_foundation_diagnostic")
    )
    foundation_preserving_deployment = foundation_contract is not None
    static_foundation_diagnostic = bool(
        checkpoint_is_static_foundation and not foundation_preserving_deployment
    )
    stage0_static_diagnostic = bool(checkpoint_report.get("stage0_static_diagnostic"))
    static_reranker_diagnostic = bool(
        checkpoint_report.get("static_reranker_diagnostic")
    )
    protocol_manifest = {
        "schema_version": VNEXT_EVAL_SCHEMA,
        "benchmark": benchmark,
        "checkpoint": checkpoint_report,
        "foundation_checkpoint": foundation_checkpoint_report,
        "unified_router_contract": unified_router_contract,
        "stage2_release_selection_contract": release_selection_contract,
        "corpus_report": dict(corpus.report),
        "corpus_report_sha256": _json_digest(dict(corpus.report)),
        "corpus_source_contract": corpus_source_contract,
        "row_protocol_sha256": _json_digest(
            [
                {
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "step_index": int(row.get("step_index") or 0),
                    "target_skill_id": row["_vnext_target_skill_id"],
                    "positive_skill_ids": row["_vnext_positive_skill_ids"],
                    "legal_skill_ids": row["_vnext_legal_skill_ids"],
                    "event_timing": row["_vnext_event_timing"],
                    "executed_skill_id": row["_vnext_executed_skill_id"],
                }
                for row in rows
            ]
        ),
        "coarse_k": int(coarse_k),
        "compressed_m": int(compressed_m),
        "dynamic_extra_k": int(resolved_dynamic_extra_k),
        "final_k": int(final_k),
        "component_ablation": component_ablation,
        "verified_result_correction_enabled": bool(
            not disable_verified_result_correction
        ),
        "candidate_query_mode": candidate_query_mode,
        "candidate_compression_mode": candidate_compression_mode,
        "candidate_protocol": (
            "preserved_closed_set_foundation_static_semantic_rrf_top500_v1"
            if foundation_selected
            else (
                (
                    "static_top500_fixed_support_ke0_v1"
                    if fixed_support_ablation
                    else "static_top500_plus_dynamic_top500_diff64_union_v2"
                )
                if candidate_query_mode == "learned_causal"
                and candidate_compression_mode == "learned"
                else (
                    f"{candidate_query_mode}_top{int(coarse_k)}_"
                    f"{candidate_compression_mode}_v1"
                )
            )
        ),
        "score_batch_size": int(score_batch_size),
        "belief_top_k": int(belief_top_k),
        "require_verified_toolbench_results": bool(
            require_verified_toolbench_results
        ),
        "toolbench_pool_scope": (
            toolbench_pool_scope if benchmark == "toolbench_g3" else None
        ),
        "toolbench_native_pool_contract": native_pool_contract,
        "foundation_preserving_deployment": bool(
            foundation_preserving_deployment
        ),
        "foundation_preservation_contract": foundation_contract,
        "interface_expert_decision": interface_decision,
        "evaluator_source_commit": evaluator_source_commit,
        "support_aware_anchor_contract": (
            {
                "protocol": SUPPORT_AWARE_ANCHOR_PROTOCOL,
                "foundation_protected_top_k": (
                    SUPPORT_AWARE_FOUNDATION_PREFIX_K
                ),
                "complete_support_tail_fusion": (
                    "equal_reciprocal_rank_fusion_k0_three_experts"
                ),
                "incomplete_support_route": (
                    "recurrent_with_history_else_adapted_static"
                ),
                "uses_benchmark_identity": False,
                "uses_source_identity": False,
                "uses_interface_dispatch": False,
                "candidate_membership_changed": False,
                "positive_injection_count": 0,
            }
            if support_aware_anchor
            else None
        ),
        "support_aware_complete_support_count": (
            int(support_aware_complete_support_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_incomplete_support_count": (
            int(support_aware_incomplete_support_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_support_partition_exact": (
            bool(
                support_aware_complete_support_count
                + support_aware_incomplete_support_count
                == len(rows)
            )
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_semantic_fusion_count": (
            int(support_aware_foundation_semantic_fusion_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_semantic_fusion_expected_count": (
            int(support_aware_foundation_semantic_fusion_expected_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_semantic_fusion_count_exact": (
            bool(
                support_aware_foundation_semantic_fusion_count
                == support_aware_foundation_semantic_fusion_expected_count
            )
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_prefix_preserved_count": (
            int(support_aware_foundation_prefix_preserved_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_prefix_expected_count": (
            int(support_aware_foundation_prefix_expected_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_foundation_prefix_preserved_exact": (
            bool(
                support_aware_foundation_prefix_preserved_count
                == support_aware_foundation_prefix_expected_count
            )
            if support_aware_anchor
            else None
        ),
        "support_aware_incomplete_route_exact_count": (
            int(support_aware_incomplete_route_exact_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_incomplete_route_expected_count": (
            int(support_aware_incomplete_route_expected_count)
            if support_aware_anchor
            else None
        ),
        "support_aware_incomplete_route_all_exact": (
            bool(
                support_aware_incomplete_route_exact_count
                == support_aware_incomplete_route_expected_count
            )
            if support_aware_anchor
            else None
        ),
        "deployment_metric_source": deployment_metric_source,
        "evaluation_scope": (
            "stage2_complete_method_component_ablation"
            if component_ablation is not None
            else (
                "sample_level_support_aware_anchor_residual_complete_method"
                if support_aware_anchor
                else (
                    "sample_level_unified_sparse_three_expert_complete_method"
                    if unified_router is not None
                    else (
                        "foundation_preserving_complete_method"
                        if foundation_preserving_deployment
                        else (
                            "stage0_static_foundation_diagnostic"
                            if stage0_static_diagnostic
                            else (
                                "successor_static_reranker_diagnostic"
                                if static_reranker_diagnostic
                                else "stage2_complete_method"
                            )
                        )
                    )
                )
            )
        ),
        "release_eligible_metrics": (
            []
            if component_ablation is not None
            else (
                ["deployment.end_to_end_route", "deployment.full_pool_recall"]
                if foundation_preserving_deployment
                else (
                    ["static.full_pool_recall", "static.end_to_end_route"]
                    if static_foundation_diagnostic
                    else ["factual.end_to_end_route", "factual.full_pool_recall"]
                )
            )
        ),
        "non_static_memory_metrics_release_eligible": bool(
            component_ablation is None
            and not static_foundation_diagnostic
            and not foundation_selected
        ),
        "closed_set_recurrent_path_executed": (
            False if foundation_selected else None
        ),
        "deployment_uses_random_unrestored_stage2_heads": False,
        "closed_set_semantic_fusion_protocol": (
            "equal_reciprocal_rank_fusion_k0_above_heldout_selected_pool8_v1"
            if foundation_selected
            else None
        ),
        "closed_set_static_only_max_pool_size": (
            CLOSED_SET_STATIC_ONLY_MAX_POOL_SIZE if foundation_selected else None
        ),
        "closed_set_semantic_fusion_applied_count": int(
            closed_set_semantic_fusion_applied_count
        ),
        "closed_set_semantic_fusion_expected_count": int(
            closed_set_semantic_fusion_expected_count
        ),
        "closed_set_semantic_fusion_count_exact": bool(
            closed_set_semantic_fusion_applied_count
            == closed_set_semantic_fusion_expected_count
        ),
        "no_positive_injection": True,
        "zero_history_static_reuse": bool(
            unified_router is None and not support_aware_anchor
        ),
        "zero_history_recurrent_weight_is_zero": (
            None if support_aware_anchor else bool(exact_masked_fallback)
        ),
        "support_aware_masked_effective_history_is_zero": (
            bool(exact_masked_fallback) if support_aware_anchor else None
        ),
        "exact_masked_fallback": bool(exact_masked_fallback),
        "fixed_causal_candidate_support": bool(fixed_support_ablation),
        "fixed_candidate_support": bool(fixed_support_ablation),
        "fixed_support_all_exact": (
            bool(fixed_support_all_exact) if fixed_support_ablation else None
        ),
        "exact_masked_candidate_support": bool(exact_masked_candidate_support),
        "memory_changes_candidate_support": bool(memory_changed_candidate_support),
        "candidate_query_is_route_input": False,
        "candidate_recall_scores_are_route_inputs": False,
        "candidate_positions_are_route_inputs": False,
        "primary_candidate_recall_metric": "candidate_union_recall",
        "candidate_support_order": (
            "canonical_skill_index"
            if foundation_selected or candidate_compression_mode == "direct"
            else "static_then_dynamic_top500_difference_union"
        ),
        "candidate_foundation_reused_across_memory_alternatives": (
            None if foundation_selected else False
        ),
        "primary_route_scoring": (
            SUPPORT_AWARE_ANCHOR_PROTOCOL
            if support_aware_anchor
            else (
                "sample_level_unified_foundation_static_memory_sparse_top1_v6"
                if unified_router is not None
                else (
                    "preserved_closed_set_static_semantic_rrf_route_v1"
                    if foundation_selected
                    else "history_depth_aware_soft_train_confidence_abstaining_hard_expert_route_final100_v7"
                )
            )
        ),
        "route_reliability_features": (
            list(UNIFIED_ROUTER_FEATURE_NAMES)
            if unified_router is not None
            else (
                None
                if foundation_selected or support_aware_anchor
                else "static_dynamic_margin_confidence_residual_rms_and_log_normalized_replay_prefix_length"
            )
        ),
        "support_aware_route_features": (
            ["natural_support_fraction", "history_depth"]
            if support_aware_anchor
            else None
        ),
        "route_expert_selection_threshold": (
            None
            if foundation_selected or unified_router is not None or support_aware_anchor
            else 0.55
        ),
        "unified_expert_weights_semantics": (
            "complete_support_equal_rrf_tail_else_static_or_recurrent_anchor"
            if support_aware_anchor
            else (
                "one_hot_sparse_top1_expert_selection"
                if unified_router is not None
                else None
            )
        ),
        "raw_route_diagnostic": (
            "ineligible_not_executed"
            if foundation_selected
            else "unbounded_dynamic_expert_reported_separately"
        ),
        "static_route_context": "compact_causal_skill_action_history",
        "memory_query_state": "structured_current_only",
        "route_fallback_scope": (
            "sample_level_support_completeness_without_benchmark_dispatch"
            if support_aware_anchor
            else (
                "sample_level_sparse_foundation_static_memory_route_with_exact_zero_recurrent_weight_without_history"
                if unified_router is not None
                else (
                    "closed_set_preserved_foundation_else_exact_static_or_recurrent_dynamic"
                    if foundation_preserving_deployment
                    else "exact_static_for_zero_history_and_binary_static_or_dynamic_for_history"
                )
            )
        ),
        "observation_support_gate_is_benchmark_agnostic": bool(
            unified_router is not None
        ),
        "support_completeness_rule_is_benchmark_agnostic": bool(
            support_aware_anchor
        ),
        "observation_grounding_feature": (
            "observation_correction_fraction"
            if unified_router is not None
            else None
        ),
        "unified_router_calibration_views": (
            "same_prefix_factual_and_action_only_result_suppressed_v1"
            if unified_router is not None
            else None
        ),
        "unified_foundation_expert_protocol": (
            "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
            if unified_router is not None or support_aware_anchor
            else None
        ),
        "reported_ablations": (
            ["foundation", "adapted_static", "raw_recurrent_memory"]
            if unified_router is not None or support_aware_anchor
            else ([
                "direct_causal_semantic",
                "preserved_foundation_plus_equal_semantic_rrf",
            ]
            if foundation_selected
            else [
                "h_only",
                "h_plus_deterministic_init",
                "recurrent_memory",
            ])
        ),
        "result_correction_requires_explicit_aligned_executed_result": True,
        "decision_event_timing_is_explicit": True,
        "transition_state_protocol": "structured_current_state_v1",
        "causal_timeline": causal_timeline_report,
        "records_path": str(records_path.resolve()),
        "records_sha256": file_sha256(records_path),
    }
    manifest_path = output_dir / f"{benchmark}_vnext_protocol_manifest.json"
    _write_json(manifest_path, protocol_manifest)
    blockers: list[str] = []
    if len(scored_records) != len(rows):
        blockers.append("route_record_count_mismatch")
    if not exact_masked_fallback:
        blockers.append(
            "support_aware_masked_history_is_not_zero"
            if support_aware_anchor
            else (
                "masked_history_has_nonzero_recurrent_weight"
                if unified_router is not None
                else "masked_history_is_not_exact_static"
            )
        )
    if not exact_masked_candidate_support:
        blockers.append("zero_history_changed_candidate_support")
    if fixed_support_ablation and not fixed_support_all_exact:
        blockers.append("ke0_candidate_support_differs_from_static")
    if (
        support_aware_anchor
        and support_aware_complete_support_count
        + support_aware_incomplete_support_count
        != len(rows)
    ):
        blockers.append("support_aware_support_partition_mismatch")
    if (
        support_aware_anchor
        and support_aware_foundation_semantic_fusion_count
        != support_aware_foundation_semantic_fusion_expected_count
    ):
        blockers.append("support_aware_foundation_fusion_count_mismatch")
    if (
        support_aware_anchor
        and support_aware_foundation_prefix_preserved_count
        != support_aware_foundation_prefix_expected_count
    ):
        blockers.append("support_aware_foundation_prefix_not_preserved")
    if (
        support_aware_anchor
        and support_aware_incomplete_route_exact_count
        != support_aware_incomplete_route_expected_count
    ):
        blockers.append("support_aware_incomplete_route_not_exact")
    if not foundation_selected and int(memory_report["uses_recurrent_m_t_count"]) <= 0:
        blockers.append("no_recurrent_memory_rows")
    if (
        foundation_selected
        and closed_set_semantic_fusion_applied_count
        != closed_set_semantic_fusion_expected_count
    ):
        blockers.append("closed_set_semantic_fusion_count_mismatch")
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "schema_version": VNEXT_EVAL_SCHEMA,
        "benchmark": benchmark,
        "method": (
            "clstr_vnext_support_aware_anchor_residual_route"
            if support_aware_anchor
            else (
                "clstr_vnext_unified_sparse_three_expert_route"
                if unified_router is not None
                else (
                    "clstr_vnext_foundation_preserving_route"
                    if foundation_preserving_deployment
                    else (
                        "clstr_vnext_stage0_static_diagnostic"
                        if stage0_static_diagnostic
                        else (
                            "clstr_vnext_successor_static_reranker_diagnostic"
                            if static_reranker_diagnostic
                            else "clstr_vnext_stage2_checkpoint_native"
                        )
                    )
                )
            )
        ),
        "evaluation_scope": protocol_manifest["evaluation_scope"],
        "component_ablation": component_ablation,
        "dynamic_extra_k": int(resolved_dynamic_extra_k),
        "verified_result_correction_enabled": bool(
            not disable_verified_result_correction
        ),
        "fixed_support_all_exact": protocol_manifest[
            "fixed_support_all_exact"
        ],
        "release_eligible_metrics": protocol_manifest["release_eligible_metrics"],
        "non_static_memory_metrics_release_eligible": protocol_manifest[
            "non_static_memory_metrics_release_eligible"
        ],
        "checkpoint": checkpoint_report,
        "evaluator_source_commit": evaluator_source_commit,
        "foundation_checkpoint": foundation_checkpoint_report,
        "unified_router_contract": unified_router_contract,
        "support_aware_anchor_contract": protocol_manifest[
            "support_aware_anchor_contract"
        ],
        "support_aware_complete_support_count": protocol_manifest[
            "support_aware_complete_support_count"
        ],
        "support_aware_incomplete_support_count": protocol_manifest[
            "support_aware_incomplete_support_count"
        ],
        "support_aware_support_partition_exact": protocol_manifest[
            "support_aware_support_partition_exact"
        ],
        "support_aware_foundation_semantic_fusion_count": protocol_manifest[
            "support_aware_foundation_semantic_fusion_count"
        ],
        "support_aware_foundation_semantic_fusion_expected_count": (
            protocol_manifest[
                "support_aware_foundation_semantic_fusion_expected_count"
            ]
        ),
        "support_aware_foundation_semantic_fusion_count_exact": (
            protocol_manifest[
                "support_aware_foundation_semantic_fusion_count_exact"
            ]
        ),
        "support_aware_foundation_prefix_preserved_count": protocol_manifest[
            "support_aware_foundation_prefix_preserved_count"
        ],
        "support_aware_foundation_prefix_expected_count": protocol_manifest[
            "support_aware_foundation_prefix_expected_count"
        ],
        "support_aware_foundation_prefix_preserved_exact": protocol_manifest[
            "support_aware_foundation_prefix_preserved_exact"
        ],
        "support_aware_incomplete_route_exact_count": protocol_manifest[
            "support_aware_incomplete_route_exact_count"
        ],
        "support_aware_incomplete_route_expected_count": protocol_manifest[
            "support_aware_incomplete_route_expected_count"
        ],
        "support_aware_incomplete_route_all_exact": protocol_manifest[
            "support_aware_incomplete_route_all_exact"
        ],
        "stage2_release_selection_contract": release_selection_contract,
        "requested_stage2_checkpoint_path": str(
            requested_stage2_checkpoint_path.resolve()
        ),
        "foundation_preservation_contract": foundation_contract,
        "interface_expert_decision": interface_decision,
        "deployment_metric_source": deployment_metric_source,
        "closed_set_recurrent_path_executed": (
            False if foundation_selected else None
        ),
        "deployment_uses_random_unrestored_stage2_heads": False,
        "closed_set_semantic_fusion_protocol": protocol_manifest[
            "closed_set_semantic_fusion_protocol"
        ],
        "closed_set_static_only_max_pool_size": protocol_manifest[
            "closed_set_static_only_max_pool_size"
        ],
        "closed_set_semantic_fusion_applied_count": int(
            closed_set_semantic_fusion_applied_count
        ),
        "closed_set_semantic_fusion_expected_count": int(
            closed_set_semantic_fusion_expected_count
        ),
        "closed_set_semantic_fusion_count_exact": bool(
            closed_set_semantic_fusion_applied_count
            == closed_set_semantic_fusion_expected_count
        ),
        "corpus": dict(corpus.report),
        "corpus_source_contract": corpus_source_contract,
        "rows": row_report,
        "causal_timeline": causal_timeline_report,
        "memory": memory_report,
        "state_cache": state_cache.report(),
        "action_cache": None if action_cache is None else action_cache.report(),
        "result_cache": None if result_cache is None else result_cache.report(),
        "metrics": metrics,
        "route_records_path": str(records_path.resolve()),
        "protocol_manifest_path": str(manifest_path.resolve()),
        "no_call_rows_excluded_from_routing_denominator": len(no_call_rows),
    }
    report_path = output_dir / f"{benchmark}_vnext_eval_report.json"
    _write_json(report_path, report)
    return report


def _multi_positive_rank(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> list[int | None]:
    if logits.ndim != 2 or positive.shape != logits.shape or legal.shape != logits.shape:
        raise ValueError("multi-positive ranking tensors must share [batch, skills]")
    output: list[int | None] = []
    skill_ids = torch.arange(logits.size(1), device=logits.device)
    for row_index in range(int(logits.size(0))):
        indices = (positive[row_index] & legal[row_index]).nonzero(
            as_tuple=False
        ).view(-1)
        if not int(indices.numel()):
            output.append(None)
            continue
        scores = logits[row_index].index_select(0, indices)
        best_score = scores.max()
        best_index = int(indices[scores.eq(best_score)].min().item())
        rank = 1 + int(
            (
                legal[row_index]
                & (
                    (logits[row_index] > best_score)
                    | (
                        logits[row_index].eq(best_score)
                        & skill_ids.lt(best_index)
                    )
                )
            )
            .sum()
            .item()
        )
        output.append(rank)
    return output


def _exact_id_occurrence_count(value: Any, identities: set[str]) -> int:
    if isinstance(value, dict):
        return sum(
            _exact_id_occurrence_count(item, identities)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_exact_id_occurrence_count(item, identities) for item in value)
    return int(isinstance(value, str) and value in identities)


def _manifest_jsonl(
    manifest: dict[str, Any],
    key: str,
) -> tuple[Path, list[dict[str, Any]]]:
    entry = (manifest.get("files") or {}).get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"vNext data manifest lacks file entry: {key}")
    path = Path(str(entry.get("path") or ""))
    expected = str(entry.get("sha256") or "")
    if not path.is_file() or not expected:
        raise ValueError(f"vNext data manifest has an invalid file contract: {key}")
    if file_sha256(path) != expected:
        raise ValueError(f"vNext data artifact digest changed: {key}")
    return path, _read_jsonl(path)


def run_vnext_unseen_skill_eval(
    *,
    checkpoint_path: str | Path,
    data_manifest_path: str | Path,
    output_dir: str | Path,
    max_eval_rows: int | None = None,
    top_m: int = 100,
    score_batch_size: int = 32,
    cache_batch_size: int = 128,
    belief_top_k: int = 64,
    frozen_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate append-only retrieval of skills absent from every train view."""

    if min(int(top_m), int(score_batch_size), int(cache_batch_size)) <= 0:
        raise ValueError("unseen-skill evaluation budgets must be positive")
    data_manifest_path = Path(data_manifest_path)
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "ok":
        raise ValueError("unseen-skill evaluation requires an approved vNext data release")
    holdout_report = manifest.get("dynamic_skill_holdout_report") or {}
    if holdout_report.get("protocol") != "unseen_skill_append_v2":
        raise ValueError("vNext data release lacks the canonical unseen-skill protocol")
    if int(((holdout_report.get("training_antijoin") or {}).get("leak_count")) or 0):
        raise ValueError("vNext data release reports a held-out skill training leak")

    training_skills_path, training_skills = _manifest_jsonl(
        manifest,
        "training_skills",
    )
    holdout_skills_path, holdout_skills = _manifest_jsonl(
        manifest,
        "dynamic_skill_holdout_skills",
    )
    holdout_queries_path, holdout_queries = _manifest_jsonl(
        manifest,
        "dynamic_skill_holdout_queries",
    )
    append_catalogs_path, append_catalog_rows = _manifest_jsonl(
        manifest,
        "dynamic_skill_holdout_catalogs",
    )
    training_skills = _unique_skills(training_skills, label="training skill prefix")
    holdout_skills = _unique_skills(holdout_skills, label="dynamic holdout skills")
    training_ids = {_skill_id(row) for row in training_skills}
    holdout_ids = {_skill_id(row) for row in holdout_skills}
    if not holdout_ids or training_ids & holdout_ids:
        raise ValueError("unseen-skill partition is empty or overlaps the training prefix")
    training_prefix = manifest.get("training_prefix") or {}
    if (
        str(training_prefix.get("skills_sha256") or "")
        != file_sha256(training_skills_path)
        or int(training_prefix.get("skill_count") or -1) != len(training_skills)
    ):
        raise ValueError("data manifest training-prefix contract is inconsistent")

    leak_counts: dict[str, int] = {}
    for key in sorted((manifest.get("files") or {})):
        if key == "data_exclusions" or key.startswith("dynamic_skill_holdout_"):
            continue
        _path, artifact_rows = _manifest_jsonl(manifest, key)
        count = sum(
            _exact_id_occurrence_count(row, holdout_ids)
            for row in artifact_rows
        )
        if count:
            leak_counts[key] = count
    if leak_counts:
        raise ValueError(f"held-out skill IDs occur in training artifacts: {leak_counts}")

    catalogs: dict[str, dict[str, Any]] = {}
    for row in append_catalog_rows:
        catalog_id = str(row.get("inventory_catalog_id") or "").strip()
        if not catalog_id or catalog_id in catalogs:
            raise ValueError("dynamic append catalogs contain an invalid identity")
        catalogs[catalog_id] = row
    selected_queries = (
        holdout_queries[: max(0, int(max_eval_rows))]
        if max_eval_rows is not None
        else holdout_queries
    )
    prepared_rows: list[dict[str, Any]] = []
    for source_index, raw in enumerate(selected_queries):
        if raw.get("dynamic_holdout_protocol") != "unseen_skill_append_v2" or not bool(
            raw.get("excluded_from_dynamic_ablation_training")
        ):
            raise ValueError("unseen-skill query lacks its exclusion provenance")
        if not str(raw.get("state_text_current") or "").strip():
            raise ValueError(
                "unseen-skill query lacks an explicit history-free state_text_current"
            )
        row = materialize_history_free_state(raw, replace_state_text=True)
        catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
        catalog = catalogs.get(catalog_id)
        if catalog is None:
            raise ValueError("unseen-skill query references an unknown append catalog")
        expected_digest = str(row.get("inventory_catalog_digest") or "")
        if expected_digest != str(catalog.get("inventory_catalog_digest") or ""):
            raise ValueError("unseen-skill query/catalog digest mismatch")
        legal_ids = [
            str(item)
            for item in catalog.get("runtime_visible_skill_ids") or []
            if str(item)
        ]
        heldout_positive_ids = [
            str(item)
            for item in row.get("dynamic_holdout_positive_skill_ids") or []
            if str(item)
        ]
        all_positive_ids = [
            str(item)
            for item in row.get("dynamic_holdout_all_positive_skill_ids") or []
            if str(item)
        ]
        if (
            not heldout_positive_ids
            or not set(heldout_positive_ids).issubset(holdout_ids)
            or not set(heldout_positive_ids).issubset(legal_ids)
            or not set(all_positive_ids).issubset(legal_ids)
        ):
            raise ValueError("unseen-skill query has invalid positive/legal identities")
        current = router_state_text(row).strip()
        if not current:
            raise ValueError("unseen-skill query has an empty current-state channel")
        copied = dict(row)
        copied["_vnext_source_index"] = source_index
        copied["_vnext_state_text"] = current
        copied["_vnext_legal_skill_ids"] = list(dict.fromkeys(legal_ids))
        copied["_vnext_holdout_positive_skill_ids"] = sorted(
            set(heldout_positive_ids)
        )
        copied["_vnext_all_positive_skill_ids"] = sorted(set(all_positive_ids))
        copied["_vnext_seen_positive_skill_ids"] = sorted(
            set(all_positive_ids) - holdout_ids
        )
        prepared_rows.append(copied)
    if not prepared_rows:
        raise ValueError("unseen-skill evaluation has no held-out queries")
    history_report = audit_history_channel_rows(
        prepared_rows,
        require_explicit_current=True,
    )
    if history_report.get("status") != "ok":
        raise ValueError("unseen-skill current-state channel failed its history audit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("unseen-skill evaluation requires a CUDA Slurm node")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, final_skills, skill_id_to_idx, checkpoint_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=checkpoint_path,
            training_skills_path=training_skills_path,
            benchmark_skills=holdout_skills,
            device=device,
        )
    )
    if int((checkpoint_report.get("append") or {}).get("appended_count") or 0) != len(
        holdout_skills
    ):
        raise ValueError("unseen-skill evaluator did not append the complete holdout")
    final_ids = {_skill_id(row) for row in final_skills}
    if final_ids != training_ids | holdout_ids:
        raise ValueError("unseen-skill final table differs from the declared partition")
    for row in prepared_rows:
        if not set(row["_vnext_legal_skill_ids"]).issubset(final_ids):
            raise ValueError("append catalog references a skill absent from the final table")

    cache_root = (
        Path(frozen_cache_dir)
        if frozen_cache_dir is not None
        else output_dir / "frozen_qwen_cache"
    )
    state_cache = load_or_build_frozen_text_cache(
        model,
        (row["_vnext_state_text"] for row in prepared_rows),
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
    )
    aggregate: dict[str, list[int | None]] = {
        "holdout": [],
        "all": [],
        "seen": [],
    }
    per_skill_ranks: dict[str, list[int | None]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for start in range(0, len(prepared_rows), int(score_batch_size)):
        batch = prepared_rows[start : start + int(score_batch_size)]
        states = state_cache.batch(
            [row["_vnext_state_text"] for row in batch],
            device=device,
        )
        legal = _legal_mask(batch, skill_id_to_idx, device=device)
        holdout_positive = torch.zeros_like(legal)
        all_positive = torch.zeros_like(legal)
        seen_positive = torch.zeros_like(legal)
        for row_index, row in enumerate(batch):
            for field, mask in (
                ("_vnext_holdout_positive_skill_ids", holdout_positive),
                ("_vnext_all_positive_skill_ids", all_positive),
                ("_vnext_seen_positive_skill_ids", seen_positive),
            ):
                indices = [skill_id_to_idx[item] for item in row[field]]
                if indices:
                    mask[row_index, torch.tensor(indices, device=device)] = True
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            belief = model.vnext_initial_belief(states, legal, top_k=belief_top_k)
            queries = model.vnext_queries(
                states,
                belief,
                belief,
                torch.zeros(len(batch), dtype=torch.bool, device=device),
            )
            logits = model.vnext_full_pool_logits(
                queries.static_recall,
                head="recall",
            )
        holdout_ranks = _multi_positive_rank(logits, holdout_positive, legal)
        all_ranks = _multi_positive_rank(logits, all_positive, legal)
        seen_ranks = _multi_positive_rank(logits, seen_positive, legal)
        aggregate["holdout"].extend(holdout_ranks)
        aggregate["all"].extend(all_ranks)
        aggregate["seen"].extend(seen_ranks)
        for row_index, row in enumerate(batch):
            individual: dict[str, int] = {}
            for skill_id in row["_vnext_holdout_positive_skill_ids"]:
                target = torch.tensor(
                    [skill_id_to_idx[skill_id]],
                    dtype=torch.long,
                    device=device,
                )
                rank = int(
                    _full_pool_rank(
                        logits[row_index : row_index + 1],
                        target,
                        legal[row_index : row_index + 1],
                    )[0].item()
                )
                individual[skill_id] = rank
                per_skill_ranks[skill_id].append(rank)
            valid_ids = legal[row_index].nonzero(as_tuple=False).view(-1)
            width = min(10, int(valid_ids.numel()))
            scores = logits[row_index].index_select(0, valid_ids)
            order = scores.topk(width, largest=True, sorted=True).indices
            top_ids = [
                _skill_id(final_skills[int(index)])
                for index in valid_ids.index_select(0, order).tolist()
            ]
            records.append(
                {
                    "schema_version": VNEXT_EVAL_SCHEMA,
                    "source_index": int(row["_vnext_source_index"]),
                    "query_id": str(row.get("query_id") or ""),
                    "legal_pool_size": len(row["_vnext_legal_skill_ids"]),
                    "holdout_positive_skill_ids": row[
                        "_vnext_holdout_positive_skill_ids"
                    ],
                    "holdout_best_rank": holdout_ranks[row_index],
                    "holdout_individual_ranks": individual,
                    "all_positive_best_rank": all_ranks[row_index],
                    "seen_positive_best_rank": seen_ranks[row_index],
                    "top_skill_ids": top_ids,
                }
            )

    seen_eligible = [rank for rank in aggregate["seen"] if rank is not None]
    metrics = {
        "holdout_appended": _metrics(aggregate["holdout"]),
        "all_required": _metrics(aggregate["all"]),
        "seen_positive_when_present": _metrics(seen_eligible),
        "holdout_candidate_recall_at_top_m": sum(
            int(rank is not None and rank <= int(top_m))
            for rank in aggregate["holdout"]
        )
        / len(aggregate["holdout"]),
        "per_holdout_skill": {
            skill_id: _metrics(ranks)
            for skill_id, ranks in sorted(per_skill_ranks.items())
        },
    }
    records_path = output_dir / "unseen_skill_vnext_records.jsonl"
    _write_jsonl(records_path, records)
    report = {
        "status": "ok",
        "schema_version": VNEXT_EVAL_SCHEMA,
        "method": "clstr_vnext_stage2_static_unseen_append",
        "checkpoint": checkpoint_report,
        "data_manifest_path": str(data_manifest_path.resolve()),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "artifacts": {
            "training_skills": str(training_skills_path.resolve()),
            "holdout_skills": str(holdout_skills_path.resolve()),
            "holdout_queries": str(holdout_queries_path.resolve()),
            "append_catalogs": str(append_catalogs_path.resolve()),
        },
        "zero_holdout_id_occurrences_in_training_artifacts": True,
        "history_channel": history_report,
        "query_count": len(prepared_rows),
        "training_skill_count": len(training_skills),
        "appended_holdout_skill_count": len(holdout_skills),
        "metrics": metrics,
        "state_cache": state_cache.report(),
        "records_path": str(records_path.resolve()),
        "records_sha256": file_sha256(records_path),
    }
    _write_json(output_dir / "unseen_skill_vnext_eval_report.json", report)
    return report
