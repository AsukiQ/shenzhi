import gzip
import json
from pathlib import Path

from clstr.data import (
    ExecutionState,
    VerifiedPair,
    validate_data_roles,
    load_verified_pairs,
    read_jsonl,
    serialize_execution_state,
    verified_pair_to_dict,
)


def test_serialize_execution_state_contains_all_fields():
    state = ExecutionState(
        query="book a train",
        history=[("search_train", "found trains")],
        observation="found 3 results",
        artifact={"ticket": "draft"},
        error=None,
    )

    text = serialize_execution_state(state)
    assert "query:book a train" in text
    assert "history:search_train -> found trains" in text
    assert "observation:found 3 results" in text
    assert "artifact:{'ticket': 'draft'}" in text
    assert "error:none" in text


def test_serialize_execution_state_uses_empty_history_and_emits_error():
    state = ExecutionState(
        query="recover booking",
        history=[],
        observation="failed to submit",
        artifact={},
        error="network timeout",
    )

    text = serialize_execution_state(state)
    assert "history:empty" in text
    assert "error:network timeout" in text


def test_verified_pair_defaults():
    pair = VerifiedPair(
        state_before=ExecutionState("q", [], "", {}, None),
        action_at_t=3,
        obs_at_t="ok",
        candidates_next=[3, 4, 5],
        a_next_plus=4,
    )
    assert pair.was_in_raw_topk is True
    assert pair.m_t_exact is None
    assert pair.replay_prefix is None


def test_read_jsonl_reads_plain_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [{"task_id": 1, "query": "book train"}, {"task_id": 2, "query": "cancel train"}]

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert read_jsonl(path) == rows


def test_read_jsonl_reads_gzipped_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl.gz"
    rows = [{"skill": "search_train"}, {"skill": "book_ticket", "ok": True}]

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert read_jsonl(path) == rows


def test_verified_pair_roundtrip_jsonl(tmp_path):
    pair = VerifiedPair(
        task_id="task-1",
        step_idx=0,
        state_before=ExecutionState("q", [], "", {}, None),
        action_at_t=3,
        obs_at_t="ok",
        candidates_next=[3, 4, 5],
        a_next_plus=4,
    )
    path = tmp_path / "verified.jsonl"
    path.write_text(json.dumps(verified_pair_to_dict(pair)) + "\n", encoding="utf-8")

    loaded = load_verified_pairs(path)

    assert len(loaded) == 1
    assert loaded[0].task_id == "task-1"
    assert loaded[0].step_idx == 0
    assert loaded[0].candidates_next == [3, 4, 5]


def test_validate_data_roles_marks_skillrouter_benchmark_eval_only(tmp_path):
    cfg = {
        "skillrouter_eval_root": str(tmp_path / "skillrouter_eval_core"),
        "skillsbench_root": str(tmp_path / "missing_skillsbench"),
        "alfworld_root": str(tmp_path / "missing_alfworld"),
        "leakage_audit_dir": str(tmp_path / "audit"),
        "clean_router_data_root": str(tmp_path / "clean_router"),
        "training_data_source": "mock",
    }

    roles = validate_data_roles(cfg, require_training_data=False)

    assert roles["skillrouter_eval"]["role"] == "eval-only"
    assert roles["skillrouter_eval"]["path"] == cfg["skillrouter_eval_root"]
    assert roles["skillsbench"]["status"] == "missing"
    assert roles["alfworld"]["status"] == "missing"


def test_validate_data_roles_treats_inaccessible_optional_roots_as_missing(tmp_path, monkeypatch):
    inaccessible = tmp_path / "inaccessible_skillret"
    cfg = {
        "skillrouter_eval_root": str(tmp_path / "skillrouter_eval_core"),
        "skillsbench_root": str(tmp_path / "missing_skillsbench"),
        "alfworld_root": str(tmp_path / "missing_alfworld"),
        "leakage_audit_dir": str(tmp_path / "audit"),
        "clean_router_data_root": str(tmp_path / "clean_router"),
        "skillret_root": str(inaccessible),
        "training_data_source": "mock",
    }
    original_exists = Path.exists

    def fake_exists(path):
        if path == inaccessible:
            raise PermissionError("denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    roles = validate_data_roles(cfg, require_training_data=False)

    assert roles["skillret"]["status"] == "missing"


def test_validate_data_roles_rejects_skillrouter_eval_as_training_source(tmp_path):
    cfg = {
        "skillrouter_eval_root": str(tmp_path / "skillrouter_eval_core"),
        "skillsbench_root": str(tmp_path / "skillsbench"),
        "alfworld_root": str(tmp_path / "alfworld"),
        "leakage_audit_dir": str(tmp_path / "audit"),
        "clean_router_data_root": str(tmp_path / "clean_router"),
        "training_data_source": "skillrouter_eval",
    }

    try:
        validate_data_roles(cfg, require_training_data=False)
    except ValueError as exc:
        assert "SkillRouter benchmark is eval-only" in str(exc)
    else:
        raise AssertionError("SkillRouter eval data must not be accepted as training data")
