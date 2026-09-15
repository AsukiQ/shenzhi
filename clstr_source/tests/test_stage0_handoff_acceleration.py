import json

import pytest
import torch


def test_raw_candidate_record_rejects_mismatched_indices_and_scores():
    from clstr.stage0_handoff_acceleration import RawStage0Candidates

    with pytest.raises(ValueError, match="current candidate indices and scores must match"):
        RawStage0Candidates(
            current_indices=(0, 1),
            current_scores=(1.0,),
            next_indices=(2,),
            next_scores=(0.5,),
        )


def _cache_identity(state_query_prompt_version=None, encode_batch_size=None):
    from clstr.stage0_handoff_acceleration import stage0_handoff_global_identity

    kwargs = {
        "checkpoint_digest": {"sha256": "checkpoint-a"},
        "skills_digest": "skills-a",
        "top_m": 2,
        "candidate_count": 2,
        "query_mode": "checkpoint_state_query",
        "inventory_min_candidates": 0,
        "initial_belief_top_k": 64,
        "state_text_format": "default",
        "skill_text_format": "default",
        "skill_embedding_digest": "embedding-a",
        "declared_pool_order_digest": "order-a",
        "candidate_selection_version": "stable_declared_pool_v1",
        "tie_break_policy": "declared_pool_index_ascending",
        "unified_static_scorer_digest": "scorer-a",
    }
    if state_query_prompt_version is not None:
        kwargs["state_query_prompt_version"] = state_query_prompt_version
    if encode_batch_size is not None:
        kwargs["encode_batch_size"] = encode_batch_size
    return stage0_handoff_global_identity(**kwargs)


def _raw_candidates(offset):
    from clstr.stage0_handoff_acceleration import RawStage0Candidates

    return RawStage0Candidates(
        current_indices=(offset, offset + 1),
        current_scores=(2.0, 1.0),
        next_indices=(offset + 1, offset),
        next_scores=(3.0, 0.5),
    )


def test_global_identity_is_raw_candidate_only_and_tracks_static_scorer():
    identity = _cache_identity()

    assert identity["static_candidate_scorer"] == "unified_static"
    assert "rows_digest" not in identity
    assert "positive_missing_policy" not in identity
    assert "next_skill_pool_mode" not in identity


def test_global_identity_tracks_checkpoint_state_query_prompt_version():
    raw_prompt = _cache_identity("raw_state_v1")
    retrieval_prompt = _cache_identity("retrieval_query_v1")

    assert raw_prompt["global_key"] != retrieval_prompt["global_key"]


def test_global_identity_tracks_legacy_execution_batch_size():
    batch16 = _cache_identity(encode_batch_size=16)
    batch64 = _cache_identity(encode_batch_size=64)

    assert batch16["execution_schedule_version"] == "legacy_split_batches_v1"
    assert batch16["global_key"] != batch64["global_key"]


def test_scheduled_row_keys_track_batch_context_and_duplicate_position():
    from clstr.stage0_handoff_acceleration import build_legacy_schedule_row_key_plan

    plan = build_legacy_schedule_row_key_plan(
        ["current-a", "current-a", "current-c"],
        ["next-a", "next-a", "next-c"],
        [(), (), ()],
        batch_size=2,
    )
    changed_context = build_legacy_schedule_row_key_plan(
        ["current-a", "current-x", "current-c"],
        ["next-a", "next-x", "next-c"],
        [(), (), ()],
        batch_size=2,
    )
    batch1 = build_legacy_schedule_row_key_plan(
        ["current-a", "current-a", "current-c"],
        ["next-a", "next-a", "next-c"],
        [(), (), ()],
        batch_size=1,
    )

    assert plan.row_keys[0] != plan.row_keys[1]
    assert plan.row_keys[0] != changed_context.row_keys[0]
    assert plan.row_keys[0] != batch1.row_keys[0]
    assert plan.batch_ranges == ((0, 2), (2, 3))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"current_indices": (-1,), "current_scores": (1.0,)}, "nonnegative"),
        ({"current_indices": (0, 0), "current_scores": (2.0, 1.0)}, "unique"),
        ({"current_indices": (0,), "current_scores": (float("nan"),)}, "finite"),
    ],
)
def test_raw_candidate_record_rejects_values_that_cannot_be_cached(kwargs, message):
    from clstr.stage0_handoff_acceleration import RawStage0Candidates

    values = {
        "current_indices": (0,),
        "current_scores": (1.0,),
        "next_indices": (1,),
        "next_scores": (0.5,),
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        RawStage0Candidates(**values)


def test_row_sharded_cache_reuses_rows_across_different_subsets(tmp_path):
    from clstr.stage0_handoff_acceleration import (
        append_row_sharded_cache,
        load_row_sharded_cache,
    )

    identity = _cache_identity()
    first = {"row-a": _raw_candidates(0), "row-b": _raw_candidates(1)}

    write_report = append_row_sharded_cache(tmp_path, identity, first, shard_size=1)
    hits, load_report = load_row_sharded_cache(
        tmp_path,
        identity,
        ["row-b", "row-c"],
    )

    assert write_report["written_rows"] == 2
    assert hits == {"row-b": first["row-b"]}
    assert load_report["hit_rows"] == 1
    assert load_report["miss_rows"] == 1
    assert load_report["cached_rows"] == 2


def _manifest_and_first_shard(cache_root, identity):
    from clstr.stage0_handoff_acceleration import row_sharded_cache_entry_dir

    manifest_path = row_sharded_cache_entry_dir(cache_root, identity) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / manifest["shards"][0]["path"]
    return manifest_path, manifest, shard_path


@pytest.mark.parametrize(
    "mutation",
    ["missing_shard", "bad_digest", "wrong_dtype", "bad_shape", "extra_payload_key"],
)
def test_row_sharded_cache_fails_closed_on_corrupt_artifacts(tmp_path, mutation):
    from clstr.stage0_handoff_acceleration import (
        append_row_sharded_cache,
        file_sha256,
        load_row_sharded_cache,
    )

    identity = _cache_identity()
    append_row_sharded_cache(tmp_path, identity, {"row-a": _raw_candidates(0)})
    manifest_path, manifest, shard_path = _manifest_and_first_shard(tmp_path, identity)
    if mutation == "missing_shard":
        shard_path.unlink()
    elif mutation == "bad_digest":
        with shard_path.open("ab") as handle:
            handle.write(b"corrupt")
    else:
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        if mutation == "wrong_dtype":
            payload["current_indices"] = payload["current_indices"].to(torch.int64)
        elif mutation == "bad_shape":
            payload["next_scores"] = payload["next_scores"][:, :-1]
        elif mutation == "extra_payload_key":
            payload["loss_mask"] = torch.ones(1, dtype=torch.bool)
        torch.save(payload, shard_path)
        manifest["shards"][0]["sha256"] = file_sha256(shard_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="row_sharded_v1"):
        load_row_sharded_cache(tmp_path, identity, ["row-a"])


def test_row_sharded_cache_skips_identical_duplicate_and_rejects_conflict(tmp_path):
    from clstr.stage0_handoff_acceleration import append_row_sharded_cache

    identity = _cache_identity()
    original = _raw_candidates(0)
    append_row_sharded_cache(tmp_path, identity, {"row-a": original})

    duplicate_report = append_row_sharded_cache(tmp_path, identity, {"row-a": original})
    assert duplicate_report["written_rows"] == 0
    assert duplicate_report["existing_rows"] == 1

    with pytest.raises(ValueError, match="conflicting duplicate row key"):
        append_row_sharded_cache(tmp_path, identity, {"row-a": _raw_candidates(1)})


def test_row_sharded_cache_refresh_reset_starts_empty(tmp_path):
    from clstr.stage0_handoff_acceleration import (
        append_row_sharded_cache,
        load_row_sharded_cache,
        reset_row_sharded_cache,
    )

    identity = _cache_identity()
    append_row_sharded_cache(tmp_path, identity, {"row-a": _raw_candidates(0)})

    reset_report = reset_row_sharded_cache(tmp_path, identity)
    hits, load_report = load_row_sharded_cache(tmp_path, identity, ["row-a"])

    assert reset_report["previous_rows"] == 1
    assert hits == {}
    assert load_report["cached_rows"] == 0


def _handoff_output(candidate_indices=(0, 1), candidate_scores=(2.0, 1.0)):
    return [
        {
            "row_id": "row-a",
            "trajectory_id": "trajectory-a",
            "step_index": 0,
            "benchmark": "traject_bench",
            "state_text": "state-a",
            "next_state_text": "state-b",
            "stage0_candidate_skill_indices": list(candidate_indices),
            "stage0_candidate_skill_scores": list(candidate_scores),
            "stage0_next_candidate_skill_indices": list(candidate_indices),
            "stage0_next_candidate_skill_scores": list(candidate_scores),
            "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            "stage0_current_positive_hit": True,
            "stage0_next_positive_hit": True,
            "stage0_positive_injected": False,
            "stage0_current_skill_candidate_added": False,
            "stage0_current_skill_candidate_role": "none",
        }
    ]


def test_handoff_gate_rejects_candidate_order_mismatch():
    from scripts.audit_qwen06_stage0_handoff_acceleration import compare_handoff_outputs

    result = compare_handoff_outputs(
        _handoff_output(candidate_indices=(0, 1)),
        _handoff_output(candidate_indices=(1, 0)),
        score_atol=1.0e-5,
    )

    assert result["candidate_ids_equal"] is False
    assert result["status"] == "action_required"


def test_handoff_gate_accepts_threshold_boundaries():
    from scripts.audit_qwen06_stage0_handoff_acceleration import evaluate_gate

    result = evaluate_gate(
        parity_report={"status": "ok"},
        backbone_trainable_parameters=0,
        cold_speedup=1.25,
        warm_speedup=5.0,
        compact_to_legacy_ratio=0.25,
        errors=[],
    )

    assert result["status"] == "ok"
    assert result["blockers"] == []


@pytest.mark.parametrize(
    "overrides, blocker",
    [
        ({"backbone_trainable_parameters": 1}, "backbone_trainable_parameters"),
        ({"cold_speedup": 1.249}, "cold_speedup"),
        ({"warm_speedup": 4.999}, "warm_speedup"),
        ({"compact_to_legacy_ratio": 0.251}, "compact_to_legacy_ratio"),
    ],
)
def test_handoff_gate_rejects_each_required_threshold(overrides, blocker):
    from scripts.audit_qwen06_stage0_handoff_acceleration import evaluate_gate

    kwargs = {
        "parity_report": {"status": "ok"},
        "backbone_trainable_parameters": 0,
        "cold_speedup": 1.25,
        "warm_speedup": 5.0,
        "compact_to_legacy_ratio": 0.25,
        "errors": [],
        **overrides,
    }
    result = evaluate_gate(**kwargs)

    assert result["status"] == "action_required"
    assert blocker in result["blockers"]


def test_handoff_gate_json_writer_handles_paths_and_tensors(tmp_path):
    from scripts.audit_qwen06_stage0_handoff_acceleration import _write_json

    output_path = tmp_path / "report.json"
    _write_json(
        output_path,
        {
            "path": tmp_path / "artifact",
            "tensor": torch.tensor([1.0, 2.0]),
        },
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["path"] == str(tmp_path / "artifact")
    assert payload["tensor"] == [1.0, 2.0]
