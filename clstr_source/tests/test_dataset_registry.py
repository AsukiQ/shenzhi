import json
from pathlib import Path

from clstr.dataset_registry import (
    REQUIRED_SOURCE_FIELDS,
    SourceSpec,
    build_clstr_full_base_registry,
    default_source_specs,
    validate_source_record,
)


def test_default_source_specs_cover_required_goal_sources():
    specs = default_source_specs(repo_root=Path("/tmp/clstr"))
    names = {spec.source_dataset for spec in specs}

    assert "data/alfworld_policy_replay/train_replay.jsonl" in names
    assert "data/aux_skillnet_rebuilt" in names
    assert "uyffg/auto-dreamer" in names
    assert "agent-eto/eto-sft-trajectory" in names
    assert "u-10bei/sft_alfworld_trajectory_dataset_v4" in names
    assert "af-rl/alfworld" in names
    assert "sangeun-park/Multi-Square" in names
    assert "talesuite/tale_suite_trajectories_downsampled" in names
    assert "anon123312/retrieval-conditional-neurips2026" in names
    assert "Ricardo-H/ws-step60-webshop-wm-w2r-qwen3-8b-40k" in names
    assert "Ricardo-H/ws-step80-webshop-wm-w2r-qwen3-32b-32k" in names
    assert "u-10bei/dbbench_sft_dataset_react_v4" in names
    assert "SkillNet skill pools" in names
    assert "data/verified_policy_replay/scienceworld_train_replay.jsonl" in names
    assert "data/verified_policy_replay/webshop_train_replay.jsonl" in names


def test_default_registry_does_not_assign_full_l_policy_to_unverified_external_sources():
    specs = default_source_specs(repo_root=Path("/tmp/clstr"))
    for spec in specs:
        if "L_policy" not in spec.allowed_losses:
            continue
        assert spec.bucket in {"official_replay", "verified_replay"}
        assert spec.source_dataset.startswith("data/")
        assert spec.source_id in {
            "alfworld_official_train_replay",
            "scienceworld_verified_policy_replay",
            "webshop_verified_policy_replay",
        }


def test_source_record_validation_requires_bucket_losses_and_leakage_flags():
    record = SourceSpec(
        source_id="alfworld_official_train_replay",
        benchmark="alfworld",
        source_dataset="data/alfworld_policy_replay/train_replay.jsonl",
        split="train",
        bucket="official_replay",
        allowed_losses=["L_policy", "L_trans", "STOP", "routing"],
        path="data/alfworld_policy_replay/train_replay.jsonl",
        has_admissible_actions=True,
        has_expert_action=True,
        has_next_observation=True,
        has_done=True,
        has_reward=False,
        leakage_risk="none",
        train_allowed=True,
        reason_if_excluded=None,
    ).to_record()

    validate_source_record(record)

    assert set(REQUIRED_SOURCE_FIELDS) <= set(record)
    assert record["bucket"] == "official_replay"
    assert record["train_allowed"] is True
    assert "L_policy" in record["allowed_losses"]


def test_build_registry_writes_sources_manifest_and_audit(tmp_path):
    local_replay = tmp_path / "data" / "alfworld_policy_replay" / "train_replay.jsonl"
    local_replay.parent.mkdir(parents=True)
    local_replay.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "registry"
    report_dir = tmp_path / "reports"

    manifest = build_clstr_full_base_registry(
        repo_root=tmp_path,
        output_dir=output_dir,
        report_dir=report_dir,
        source_specs=[
            SourceSpec(
                source_id="alfworld_official_train_replay",
                benchmark="alfworld",
                source_dataset="data/alfworld_policy_replay/train_replay.jsonl",
                split="train",
                bucket="official_replay",
                allowed_losses=["L_policy", "L_trans", "STOP", "routing"],
                path="data/alfworld_policy_replay/train_replay.jsonl",
                has_admissible_actions=True,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=False,
                leakage_risk="none",
                train_allowed=True,
                reason_if_excluded=None,
            ),
            SourceSpec(
                source_id="webshop_test_replay",
                benchmark="webshop",
                source_dataset="Ricardo-H/ws-step60-webshop-wm-w2r-qwen3-8b-40k",
                split="test",
                bucket="eval_only",
                allowed_losses=[],
                path=None,
                has_admissible_actions=True,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=True,
                leakage_risk="target_test_split",
                train_allowed=False,
                reason_if_excluded="webshop_test_split_not_allowed_for_training",
            ),
        ],
    )

    assert manifest["status"] == "ok"
    assert manifest["source_count"] == 2
    assert manifest["train_allowed_source_count"] == 1
    assert manifest["bucket_counts"] == {"eval_only": 1, "official_replay": 1}
    assert manifest["benchmark_counts"] == {"alfworld": 1, "webshop": 1}
    assert (output_dir / "sources.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (report_dir / "data_audit_report.json").exists()

    rows = [json.loads(line) for line in (output_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["available"] is True
    assert rows[1]["available"] is False
    assert rows[1]["train_allowed"] is False
