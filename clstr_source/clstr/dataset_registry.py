from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clstr.external_data import write_json, write_jsonl


VALID_BUCKETS = {"official_replay", "verified_replay", "weak_policy", "aux_only", "eval_only"}
VALID_LOSSES = {"L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"}
REQUIRED_SOURCE_FIELDS = (
    "source_id",
    "benchmark",
    "source_dataset",
    "split",
    "bucket",
    "allowed_losses",
    "path",
    "has_admissible_actions",
    "has_expert_action",
    "has_next_observation",
    "has_done",
    "has_reward",
    "leakage_risk",
    "train_allowed",
    "reason_if_excluded",
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    benchmark: str
    source_dataset: str
    split: str
    bucket: str
    allowed_losses: list[str]
    path: str | None
    has_admissible_actions: bool
    has_expert_action: bool
    has_next_observation: bool
    has_done: bool
    has_reward: bool
    leakage_risk: str
    train_allowed: bool
    reason_if_excluded: str | None
    notes: str | None = None
    remote_url: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "benchmark": self.benchmark,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "bucket": self.bucket,
            "allowed_losses": list(self.allowed_losses),
            "path": self.path,
            "has_admissible_actions": self.has_admissible_actions,
            "has_expert_action": self.has_expert_action,
            "has_next_observation": self.has_next_observation,
            "has_done": self.has_done,
            "has_reward": self.has_reward,
            "leakage_risk": self.leakage_risk,
            "train_allowed": self.train_allowed,
            "reason_if_excluded": self.reason_if_excluded,
            "notes": self.notes,
            "remote_url": self.remote_url,
        }


def validate_source_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"source record missing fields: {missing}")
    if record["bucket"] not in VALID_BUCKETS:
        raise ValueError(f"unsupported bucket: {record['bucket']}")
    losses = record.get("allowed_losses") or []
    invalid_losses = [loss for loss in losses if loss not in VALID_LOSSES]
    if invalid_losses:
        raise ValueError(f"unsupported allowed_losses: {invalid_losses}")
    if bool(record.get("train_allowed")) and record.get("bucket") == "eval_only":
        raise ValueError("eval_only sources cannot be train_allowed")
    if bool(record.get("train_allowed")) and str(record.get("split", "")).lower() in {"valid", "valid_seen", "valid_unseen", "test"}:
        raise ValueError("valid/test sources cannot be train_allowed")
    if bool(record.get("train_allowed")) and record.get("leakage_risk") in {"target_test_split", "target_valid_split"}:
        raise ValueError("leaky sources cannot be train_allowed")


def _hf_url(dataset_id: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}"


def default_source_specs(repo_root: str | Path = ".") -> list[SourceSpec]:
    repo_root = Path(repo_root)
    return [
        SourceSpec(
            "alfworld_official_train_replay",
            "alfworld",
            "data/alfworld_policy_replay/train_replay.jsonl",
            "train",
            "official_replay",
            ["L_policy", "L_trans", "L_trans_skill_ce", "STOP", "routing"],
            "data/alfworld_policy_replay/train_replay.jsonl",
            True,
            True,
            True,
            True,
            False,
            "none",
            True,
            None,
            "Locally extracted from official ALFWorld train env.",
        ),
        SourceSpec(
            "aux_skillnet_rebuilt",
            "multi",
            "data/aux_skillnet_rebuilt",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            "data/aux_skillnet_rebuilt",
            False,
            True,
            True,
            True,
            True,
            "none",
            True,
            None,
            "SkillNet-remapped auxiliary trajectories; train split only during preprocessing.",
        ),
        SourceSpec(
            "auto_dreamer",
            "multi",
            "uyffg/auto-dreamer",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "uyffg_auto-dreamer"),
            False,
            True,
            True,
            True,
            True,
            "needs_replay_verification",
            True,
            None,
            "ALFWorld/ScienceWorld/Crafter rollouts; must pass official replay verification before L_policy use.",
            _hf_url("uyffg/auto-dreamer"),
        ),
        SourceSpec(
            "agent_eto_sft",
            "multi",
            "agent-eto/eto-sft-trajectory",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "agent-eto_eto-sft-trajectory"),
            False,
            True,
            True,
            False,
            True,
            "needs_source_split_filter",
            True,
            None,
            "Expert SFT conversations for ALFWorld/ScienceWorld/WebShop; no official per-step admissible field.",
            _hf_url("agent-eto/eto-sft-trajectory"),
        ),
        *[
            SourceSpec(
                f"u10bei_alfworld_v{suffix or '1'}",
                "alfworld",
                f"u-10bei/sft_alfworld_trajectory_dataset{suffix}",
                "train",
                "weak_policy" if suffix in {"_v3", "_v4"} else "aux_only",
                ["L_trans", "L_trans_skill_ce", "STOP", "routing"],
                str(repo_root / ".cache" / "hf" / f"u-10bei_sft_alfworld_trajectory_dataset{suffix}"),
                False,
                True,
                True,
                False,
                False,
                "synthetic_template",
                True,
                None,
                "Synthetic/template ALFWorld SFT; never treated as official replay.",
                _hf_url(f"u-10bei/sft_alfworld_trajectory_dataset{suffix}"),
            )
            for suffix in ("", "_v2", "_v3", "_v4")
        ],
        SourceSpec(
            "af_rl_alfworld_gamefiles",
            "alfworld",
            "af-rl/alfworld",
            "train",
            "eval_only",
            [],
            str(repo_root / ".cache" / "hf" / "af-rl_alfworld"),
            False,
            False,
            False,
            False,
            False,
            "requires_replay_extraction",
            False,
            "gamefile_pool_requires_official_replay_extraction",
            "TW-PDDL game files, not trajectory labels.",
            _hf_url("af-rl/alfworld"),
        ),
        SourceSpec(
            "multi_square",
            "multi",
            "sangeun-park/Multi-Square",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "sangeun-park_Multi-Square"),
            False,
            True,
            True,
            True,
            True,
            "needs_source_split_filter",
            True,
            None,
            "High/low-level planner/executor data for ALFWorld/ScienceWorld/TextCraft.",
            _hf_url("sangeun-park/Multi-Square"),
        ),
        SourceSpec(
            "talesuite_downsampled",
            "multi",
            "talesuite/tale_suite_trajectories_downsampled",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "talesuite_tale_suite_trajectories_downsampled"),
            False,
            True,
            True,
            True,
            True,
            "model_trajectory_not_expert",
            True,
            None,
            "Top-model text adventure trajectories; filter by score before training.",
            _hf_url("talesuite/tale_suite_trajectories_downsampled"),
        ),
        SourceSpec(
            "anon_retrieval_demo_pool",
            "multi",
            "anon123312/retrieval-conditional-neurips2026",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "anon123312_retrieval-conditional-neurips2026"),
            False,
            True,
            True,
            True,
            True,
            "mixed_eval_and_demo_archive",
            True,
            None,
            "Use demonstration_pools only; eval logs remain eval_only.",
            _hf_url("anon123312/retrieval-conditional-neurips2026"),
        ),
        SourceSpec(
            "webshop_step60_test_real_env",
            "webshop",
            "Ricardo-H/ws-step60-webshop-wm-w2r-qwen3-8b-40k",
            "test",
            "eval_only",
            [],
            str(repo_root / ".cache" / "hf" / "Ricardo-H_ws-step60-webshop-wm-w2r-qwen3-8b-40k"),
            True,
            True,
            True,
            True,
            True,
            "target_test_split",
            False,
            "webshop_test_split_not_allowed_for_training",
            "Real WebShop replay but README identifies WebShop test split.",
            _hf_url("Ricardo-H/ws-step60-webshop-wm-w2r-qwen3-8b-40k"),
        ),
        SourceSpec(
            "webshop_step80_test_real_env",
            "webshop",
            "Ricardo-H/ws-step80-webshop-wm-w2r-qwen3-32b-32k",
            "test",
            "eval_only",
            [],
            str(repo_root / ".cache" / "hf" / "Ricardo-H_ws-step80-webshop-wm-w2r-qwen3-32b-32k"),
            True,
            True,
            True,
            True,
            True,
            "target_test_split",
            False,
            "webshop_test_split_not_allowed_for_training",
            "Real WebShop replay but README identifies WebShop test split.",
            _hf_url("Ricardo-H/ws-step80-webshop-wm-w2r-qwen3-32b-32k"),
        ),
        SourceSpec(
            "scienceworld_verified_policy_replay",
            "scienceworld",
            "data/verified_policy_replay/scienceworld_train_replay.jsonl",
            "train",
            "verified_replay",
            ["L_policy", "L_trans", "belief", "STOP", "routing"],
            "data/verified_policy_replay/scienceworld_train_replay.jsonl",
            True,
            True,
            True,
            True,
            True,
            "none",
            True,
            None,
            "Generated only by scripts/verify_policy_replay_candidates.py after official ScienceWorld harness verification.",
        ),
        SourceSpec(
            "webshop_verified_policy_replay",
            "webshop",
            "data/verified_policy_replay/webshop_train_replay.jsonl",
            "train",
            "verified_replay",
            ["L_policy", "L_trans", "belief", "STOP", "routing"],
            "data/verified_policy_replay/webshop_train_replay.jsonl",
            True,
            True,
            True,
            True,
            True,
            "none",
            True,
            None,
            "Generated only by scripts/verify_policy_replay_candidates.py after official WebShop harness verification.",
        ),
        SourceSpec(
            "dbbench_sft_react_v4",
            "dbbench",
            "u-10bei/dbbench_sft_dataset_react_v4",
            "train",
            "aux_only",
            ["L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
            str(repo_root / ".cache" / "hf" / "u-10bei_dbbench_sft_dataset_react_v4"),
            False,
            True,
            True,
            True,
            True,
            "synthetic_sft",
            True,
            None,
            "Synthetic DBBench ReAct SFT; evaluate with separate DBBench harness.",
            _hf_url("u-10bei/dbbench_sft_dataset_react_v4"),
        ),
        SourceSpec(
            "skillnet_skill_pools",
            "multi",
            "SkillNet skill pools",
            "train",
            "aux_only",
            ["routing"],
            "/tmp/SkillNet/experiments/src/skills",
            False,
            False,
            False,
            False,
            False,
            "none",
            True,
            None,
            "Skill pool only, not trajectory labels.",
            "https://github.com/zjunlp/SkillNet",
        ),
    ]


def _resolve_path(repo_root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _augment_availability(repo_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    resolved = _resolve_path(repo_root, record.get("path"))
    output["resolved_path"] = str(resolved) if resolved is not None else None
    output["available"] = bool(resolved is not None and resolved.exists())
    if not output["available"] and output.get("train_allowed"):
        output["train_allowed"] = False
        output["reason_if_excluded"] = output.get("reason_if_excluded") or "source_path_missing"
    return output


def build_clstr_full_base_registry(
    repo_root: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    source_specs: Iterable[SourceSpec] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    specs = list(source_specs) if source_specs is not None else default_source_specs(repo_root)
    records = []
    for spec in specs:
        record = _augment_availability(repo_root, spec.to_record())
        validate_source_record(record)
        records.append(record)

    bucket_counts = Counter(str(record["bucket"]) for record in records)
    benchmark_counts = Counter(str(record["benchmark"]) for record in records)
    train_allowed_counts = Counter(str(record["benchmark"]) for record in records if record["train_allowed"])
    leakage_counts = Counter(str(record["leakage_risk"]) for record in records)
    manifest = {
        "status": "ok" if records else "missing",
        "source_count": len(records),
        "available_source_count": sum(1 for record in records if record["available"]),
        "train_allowed_source_count": sum(1 for record in records if record["train_allowed"]),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "train_allowed_counts_by_benchmark": dict(sorted(train_allowed_counts.items())),
        "leakage_risk_counts": dict(sorted(leakage_counts.items())),
        "sources_path": str(Path(output_dir) / "sources.jsonl"),
        "valid_or_test_train_allowed": any(
            record["train_allowed"] and str(record["split"]).lower() in {"valid", "valid_seen", "valid_unseen", "test"}
            for record in records
        ),
        "forbidden_training_policy": {
            "alfworld_valid_or_test": "not_allowed",
            "webshop_test_split_for_webshop_eval": "not_allowed",
            "synthetic_as_official_replay": "not_allowed",
        },
    }
    audit = {
        **manifest,
        "sources": records,
        "bucket_definitions": {
            "official_replay": "official environment extraction with admissible/expert evidence",
            "verified_replay": "external trajectory verified or replay-verifiable with environment evidence",
            "weak_policy": "synthetic or text-derived candidate/action labels",
            "aux_only": "trajectory or skill data without official policy candidate evidence",
            "eval_only": "not allowed for training",
        },
    }
    write_jsonl(Path(output_dir) / "sources.jsonl", records)
    write_json(Path(output_dir) / "manifest.json", manifest)
    write_json(Path(report_dir) / "data_audit_report.json", audit)
    return manifest
