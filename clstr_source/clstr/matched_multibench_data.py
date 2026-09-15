from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any, Iterable

from clstr.history_channel import (
    actual_causal_observation,
    materialize_structured_current_state,
)


TOOLSANDBOX_SPLIT_SCHEMA = "clstr_toolsandbox_template_family_split_v1"
TAU2_SPLIT_SCHEMA = "clstr_tau2_official_test_train_dev_split_v1"
LOCKED_SPLIT_SCHEMA = "clstr_locked_semantic_split_v1"
MATCHED_SPLITS = frozenset({"train", "dev", "test"})


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_toolsandbox_family_name(scenario_name: str) -> str:
    value = re.sub(
        r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(scenario_name).lower())
    ).strip("_")
    suffixes = (
        "alt",
        "implicit",
        "multiple_user_turn",
        "insufficient_information",
        "low_battery_mode",
        "wifi_off",
        "cellular_off",
        "ambiguous",
        "canonicalize",
        "twice",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            marker = f"_{suffix}"
            if value.endswith(marker):
                value = value[: -len(marker)]
                changed = True
    for marker in (
        "_low_battery_mode",
        "_wifi_off",
        "_cellular_off",
        "_insufficient_information",
        "_multiple_user_turn",
    ):
        value = value.replace(marker, "")
    value = re.sub(r"_no_(?:remove_contact|search_contacts)", "", value)
    value = re.sub(
        r"_recency_(?:latest|oldest|upcoming|yesterday)", "_recency", value
    )
    value = value.replace("_creation_recency", "_recency")
    value = value.replace("_and_time_diff", "")
    if value.startswith("add_reminder_content_and_"):
        value = "add_reminder_content_schedule"
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("ToolSandbox scenario has no canonical family")
    return value


def _hash_fraction(seed: str, identity: str) -> float:
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _signed_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["manifest_sha256"] = canonical_digest(payload)
    return output


def _verify_manifest(manifest: dict[str, Any], schema: str) -> None:
    if str(manifest.get("schema_version") or "") != schema:
        raise ValueError(f"unsupported split manifest schema: {schema}")
    observed = str(manifest.get("manifest_sha256") or "")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if observed != canonical_digest(unsigned):
        raise ValueError("split manifest digest mismatch")


def _toolsandbox_scenarios(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        name = str(provenance.get("scenario_name") or "").strip()
        if not name:
            raise ValueError("ToolSandbox row lacks provenance.scenario_name")
        grouped[name].append(row)
    records: dict[str, dict[str, Any]] = {}
    for name, scenario_rows in sorted(grouped.items()):
        by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scenario_rows:
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            if not trajectory_id:
                raise ValueError(f"ToolSandbox scenario row lacks trajectory_id: {name}")
            by_trajectory[trajectory_id].append(row)
        for trajectory_id, trajectory_rows in sorted(by_trajectory.items()):
            ordered = sorted(
                trajectory_rows,
                key=lambda row: int(row.get("step_index") or 0),
            )
            if [int(row.get("step_index") or 0) for row in ordered] != list(
                range(len(ordered))
            ):
                raise ValueError(
                    f"ToolSandbox route steps are not contiguous: {trajectory_id}"
                )
        provenance = scenario_rows[0].get("provenance") or {}
        family_name = canonical_toolsandbox_family_name(name)
        records[name] = {
            "scenario_name": name,
            "scenario_group": str(provenance.get("scenario_group") or ""),
            "family_name": family_name,
            "family_id": canonical_digest(
                {"schema": TOOLSANDBOX_SPLIT_SCHEMA, "family": family_name}
            ),
            "route_variant_count": len(by_trajectory),
            "route_row_count": len(scenario_rows),
        }
    return records


def build_toolsandbox_split_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    seed: str = "clstr-matched-v1",
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> dict[str, Any]:
    if not (0.0 < train_fraction < 1.0 and 0.0 < dev_fraction < 1.0):
        raise ValueError("ToolSandbox train/dev fractions must be in (0, 1)")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("ToolSandbox train+dev fractions must be below 1")
    records = _toolsandbox_scenarios(rows)
    family_to_split: dict[str, str] = {}
    for record in records.values():
        family_id = str(record["family_id"])
        if family_id in family_to_split:
            continue
        value = _hash_fraction(seed, str(record["family_name"]))
        family_to_split[family_id] = (
            "train"
            if value < train_fraction
            else "dev"
            if value < train_fraction + dev_fraction
            else "test"
        )
    scenario_to_split = {
        name: family_to_split[str(record["family_id"])]
        for name, record in records.items()
    }
    splits = {
        split: sorted(name for name, value in scenario_to_split.items() if value == split)
        for split in sorted(MATCHED_SPLITS)
    }
    if any(not splits[split] for split in MATCHED_SPLITS):
        raise ValueError("ToolSandbox split produced an empty partition")
    family_counts = Counter(family_to_split.values())
    scenario_counts = Counter(scenario_to_split.values())
    row_counts: Counter[str] = Counter()
    for name, record in records.items():
        row_counts[scenario_to_split[name]] += int(record["route_row_count"])
    manifest = _signed_manifest(
        {
            "schema_version": TOOLSANDBOX_SPLIT_SCHEMA,
            "seed": seed,
            "train_fraction": train_fraction,
            "dev_fraction": dev_fraction,
            "test_fraction": 1.0 - train_fraction - dev_fraction,
            "scenario_records": records,
            "family_to_split": dict(sorted(family_to_split.items())),
            "scenario_to_split": dict(sorted(scenario_to_split.items())),
            "splits": splits,
            "family_count_by_split": dict(sorted(family_counts.items())),
            "scenario_count_by_split": dict(sorted(scenario_counts.items())),
            "route_row_count_by_split": dict(sorted(row_counts.items())),
        }
    )
    validate_toolsandbox_split_manifest(manifest)
    return manifest


def validate_toolsandbox_split_manifest(manifest: dict[str, Any]) -> None:
    _verify_manifest(manifest, TOOLSANDBOX_SPLIT_SCHEMA)
    records = manifest.get("scenario_records") or {}
    assignments = manifest.get("scenario_to_split") or {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    for name, split in assignments.items():
        if split not in MATCHED_SPLITS or name not in records:
            raise ValueError("invalid ToolSandbox split assignment")
        family_splits[str(records[name].get("family_id") or "")].add(str(split))
    if any(not family or len(splits) != 1 for family, splits in family_splits.items()):
        raise ValueError("ToolSandbox family crosses split boundaries")
    for split in MATCHED_SPLITS:
        expected = sorted(name for name, value in assignments.items() if value == split)
        if sorted((manifest.get("splits") or {}).get(split) or []) != expected:
            raise ValueError("ToolSandbox split membership does not reproduce")


def toolsandbox_split_scenario_names(manifest: dict[str, Any], split: str) -> list[str]:
    validate_toolsandbox_split_manifest(manifest)
    if split not in MATCHED_SPLITS:
        raise ValueError("ToolSandbox split must be train, dev, or test")
    return [str(item) for item in (manifest.get("splits") or {}).get(split) or []]


def _tau2_task_key(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    domain = str(row.get("domain") or provenance.get("domain") or "").strip().lower()
    task_id = str(provenance.get("raw_task_id") or "").strip()
    if not domain or not task_id:
        raise ValueError("Tau2 row lacks domain/raw_task_id")
    return f"{domain}/{task_id}"


def build_tau2_split_manifest(
    *,
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    train_task_keys: Iterable[str] | None = None,
    test_task_keys: Iterable[str] | None = None,
    seed: str = "clstr-matched-v1",
    dev_fraction: float = 0.2,
) -> dict[str, Any]:
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("Tau2 dev_fraction must be in (0, 0.5)")
    train_tasks = sorted(
        {
            *(_tau2_task_key(row) for row in train_rows),
            *(str(item).strip() for item in train_task_keys or [] if str(item).strip()),
        }
    )
    test_tasks = sorted(
        {
            *(_tau2_task_key(row) for row in test_rows),
            *(str(item).strip() for item in test_task_keys or [] if str(item).strip()),
        }
    )
    if set(train_tasks) & set(test_tasks):
        raise ValueError("Tau2 official train/test task identities overlap")
    task_to_split = {
        task: "dev" if _hash_fraction(seed, task) < dev_fraction else "train"
        for task in train_tasks
    }
    task_to_split.update({task: "test" for task in test_tasks})
    splits = {
        split: sorted(task for task, value in task_to_split.items() if value == split)
        for split in sorted(MATCHED_SPLITS)
    }
    domains = sorted({task.split("/", 1)[0] for task in task_to_split})
    for domain in domains:
        for split in MATCHED_SPLITS:
            if not any(task.startswith(f"{domain}/") for task in splits[split]):
                raise ValueError(f"Tau2 {split} is empty for domain {domain}")
    manifest = _signed_manifest(
        {
            "schema_version": TAU2_SPLIT_SCHEMA,
            "seed": seed,
            "dev_fraction_within_official_train": dev_fraction,
            "official_test_preserved": True,
            "domains": domains,
            "task_to_split": dict(sorted(task_to_split.items())),
            "splits": splits,
            "task_count_by_split": {key: len(value) for key, value in splits.items()},
        }
    )
    validate_tau2_split_manifest(manifest)
    return manifest


def validate_tau2_split_manifest(manifest: dict[str, Any]) -> None:
    _verify_manifest(manifest, TAU2_SPLIT_SCHEMA)
    assignments = manifest.get("task_to_split") or {}
    for split in MATCHED_SPLITS:
        expected = sorted(task for task, value in assignments.items() if value == split)
        if sorted((manifest.get("splits") or {}).get(split) or []) != expected:
            raise ValueError("Tau2 split membership does not reproduce")


def tau2_row_split(manifest: dict[str, Any], row: dict[str, Any]) -> str:
    validate_tau2_split_manifest(manifest)
    split = str((manifest.get("task_to_split") or {}).get(_tau2_task_key(row)) or "")
    if split not in MATCHED_SPLITS:
        raise ValueError("Tau2 row is absent from its split manifest")
    return split


def locked_split_fields(
    *, split: str, group_identity: str, manifest_sha256: str
) -> dict[str, str]:
    if split not in {"train", "dev"}:
        raise ValueError("training rows may lock only train/dev")
    if not re.fullmatch(r"[0-9a-f]{64}", group_identity):
        raise ValueError("locked group identity must be SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("locked manifest identity must be SHA256")
    return {
        "locked_data_split": split,
        "locked_split_group_identity": group_identity,
        "locked_split_manifest_sha256": manifest_sha256,
        "locked_split_schema_version": LOCKED_SPLIT_SCHEMA,
    }


def canonicalize_matched_route_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split_by_trajectory: dict[str, str],
    group_identity_by_trajectory: dict[str, str],
    split_manifest_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        split = str(split_by_trajectory.get(trajectory_id) or "")
        group_identity = str(group_identity_by_trajectory.get(trajectory_id) or "")
        if split not in MATCHED_SPLITS:
            raise ValueError("matched route row lacks a valid split")
        if not re.fullmatch(r"[0-9a-f]{64}", group_identity):
            raise ValueError("matched route row lacks a valid group identity")
        current = str(row.get("state_text_current") or row.get("state_text") or "").strip()
        if not current:
            raise ValueError("matched route row lacks current state")
        structured = materialize_structured_current_state(
            row
            if isinstance(row.get("current_state_components"), dict)
            else {
                "current_state_components": {
                    "goal_text": current,
                    "task_text": "",
                    "current_observation_text": "",
                },
                "state_text": current,
            },
            replace_state_text=True,
        )
        copied = dict(row)
        copied.update(
            {
                "state_text": str(structured["state_text_current"]),
                "state_text_current": str(structured["state_text_current"]),
                "state_text_full": str(structured["state_text_current"]),
                "current_state_components": dict(structured["current_state_components"]),
                "history_text": "",
                "matched_data_split": split,
                "matched_split_group_identity": group_identity,
                "matched_split_manifest_sha256": split_manifest_sha256,
                "matched_route_protocol": "history_free_current_state_v1",
            }
        )
        output.append(copied)
    return output


def route_rows_to_training_trajectories(
    rows: Iterable[dict[str, Any]],
    *,
    benchmark: str,
    split_by_trajectory: dict[str, str],
    group_identity_by_trajectory: dict[str, str],
    split_manifest_sha256: str,
    evaluation_split: str | None = None,
) -> list[dict[str, Any]]:
    """Shift evaluator rows from previous-tool to current-action semantics.

    Training callers bind every trajectory to an immutable train/dev manifest.
    A disjoint evaluator may set ``evaluation_split='test'``; this preserves the
    same temporal shift while deliberately omitting train/dev lock fields.
    """

    if evaluation_split not in {None, "test"}:
        raise ValueError("matched route evaluation split must be test")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("route_target") or "TOOL").upper() in {"STOP", "NO_CALL"}:
            continue
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError("matched route row lacks trajectory_id")
        grouped[trajectory_id].append(row)
    output: list[dict[str, Any]] = []
    for trajectory_id, trajectory_rows in sorted(grouped.items()):
        if evaluation_split is None:
            split = str(split_by_trajectory.get(trajectory_id) or "")
            locked = locked_split_fields(
                split=split,
                group_identity=str(group_identity_by_trajectory.get(trajectory_id) or ""),
                manifest_sha256=split_manifest_sha256,
            )
        else:
            split = evaluation_split
            locked = {}
        ordered = sorted(trajectory_rows, key=lambda row: int(row.get("step_index") or 0))
        if [int(row.get("step_index") or 0) for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"matched trajectory is not contiguous: {trajectory_id}")
        for index, row in enumerate(ordered):
            target = str(row.get("next_skill_id") or "").strip()
            action = str(row.get("target_action_text") or "").strip()
            current = str(row.get("state_text_current") or row.get("state_text") or "").strip()
            if not target or not action or not current:
                raise ValueError("matched route row lacks target/action/current state")
            structured = materialize_structured_current_state(
                row
                if isinstance(row.get("current_state_components"), dict)
                else {
                    "current_state_components": {
                        "goal_text": current,
                        "task_text": "",
                        "current_observation_text": "",
                    },
                    "state_text": current,
                },
                replace_state_text=True,
            )
            next_target = (
                str(ordered[index + 1].get("next_skill_id") or "").strip()
                if index + 1 < len(ordered)
                else ""
            )
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            causal_result = str(row.get("next_observation_text") or "").strip()
            if not actual_causal_observation(row):
                causal_result = ""
            observation_source = (
                str(row.get("observation_source") or "").strip()
                if causal_result
                else "action_only_no_tool_result"
            )
            source_id = str(provenance.get("source_id") or "").strip()
            if not source_id:
                source_id = (
                    f"matched_{benchmark}_actual_result_v1"
                    if causal_result
                    else f"matched_{benchmark}_action_only_v1"
                )
            output.append(
                {
                    "benchmark": benchmark,
                    "source_benchmark": benchmark,
                    "task_id": str(row.get("task_id") or f"{trajectory_id}::{index}"),
                    "trajectory_id": trajectory_id,
                    "step_index": index,
                    "goal_text": str(
                        structured["current_state_components"].get("goal_text")
                        or current
                    ),
                    "task_text": str(
                        structured["current_state_components"].get("task_text")
                        or ""
                    ),
                    "state_text": str(structured["state_text_current"]),
                    "state_text_current": str(structured["state_text_current"]),
                    "state_text_full": str(structured["state_text_current"]),
                    "current_state_components": dict(structured["current_state_components"]),
                    "history_text": "",
                    "action_text": action,
                    "next_observation_text": causal_result,
                    "observation_source": observation_source,
                    "actual_result_event_id": str(
                        row.get("actual_result_event_id") or ""
                    ),
                    "state_event_id": str(row.get("state_event_id") or ""),
                    "skill_id": target,
                    "next_skill_id": next_target,
                    "equivalent_next_skill_ids": [
                        str(item)
                        for item in row.get("equivalent_next_skill_ids") or []
                        if str(item)
                    ],
                    "route_target": "TOOL",
                    "tool_inventory_skill_ids": [
                        str(item)
                        for item in row.get("candidate_next_skill_ids") or []
                        if str(item)
                    ],
                    "split_semantic_text": str(
                        row.get("split_semantic_text") or current
                    ),
                    "matched_data_protocol": (
                        "current_action_actual_result_v1"
                        if causal_result
                        else "current_action_action_only_v1"
                    ),
                    "training_target_semantics": (
                        "current_skill_before_memory_update_v1"
                    ),
                    "equivalent_positive_semantics": (
                        "current_target_equivalents_v1"
                    ),
                    **locked,
                    "provenance": {
                        **provenance,
                        "source_id": source_id,
                        "matched_source_previous_skill_id": str(row.get("skill_id") or ""),
                        "matched_split": split,
                        "matched_split_manifest_sha256": split_manifest_sha256,
                    },
                }
            )
    return output
