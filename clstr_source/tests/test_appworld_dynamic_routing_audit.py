import json
from pathlib import Path

import torch

from clstr.appworld_dynamic_routing_audit import (
    audit_appworld_dynamic_routing,
    rank_positive_pairs,
    sample_positive_pairs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_rank_positive_pairs_reports_global_and_appworld_masked_metrics():
    skill_ids = ["base/strong", "skillx/appworld/a", "skillx/appworld/b", "base/also-strong"]
    logits = torch.tensor(
        [
            [0.9, 0.5, 0.2, 0.8],
            [0.9, 0.5, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    report = rank_positive_pairs(
        logits=logits,
        skill_ids=skill_ids,
        positive_skill_ids=["skillx/appworld/a", "skillx/appworld/b"],
        appworld_compatible_mask=[False, True, True, False],
        k_values=[1, 2, 3, 4],
    )

    assert report["positive_pair_count"] == 2
    assert report["global"]["positive_ranks"] == [3, 4]
    assert report["global"]["recall@1"] == 0.0
    assert report["global"]["recall@3"] == 0.5
    assert report["global"]["mrr"] == 0.291667
    assert report["appworld_compatible"]["positive_ranks"] == [1, 2]
    assert report["appworld_compatible"]["recall@1"] == 0.5
    assert report["appworld_compatible"]["recall@2"] == 1.0
    assert report["appworld_compatible"]["mrr"] == 0.75


def test_sample_positive_pairs_stride_spreads_budget_across_file():
    pairs = [{"query_id": f"q{idx}", "query_text": f"text {idx}", "positive_skill_id": "s"} for idx in range(10)]

    sampled, report = sample_positive_pairs(pairs, max_pairs=4, sampling_strategy="stride")

    assert [row["query_id"] for row in sampled] == ["q0", "q2", "q5", "q7"]
    assert report["sampling_strategy"] == "stride"
    assert report["positive_pair_count_before_cap"] == 10
    assert report["positive_pair_count_after_cap"] == 4


class _FakeSkillTable:
    def __init__(self):
        self.calls = 0

    def retrieval_logits(self, h):
        del h
        self.calls += 1
        return torch.tensor(
            [
                [0.9, 0.4, 0.8],
                [0.1, 0.7, 0.2],
            ],
            dtype=torch.float32,
        )


class _FakeModel:
    def __init__(self):
        self.skills = [{"skill_id": "base/a"}]
        self.skill_table = _FakeSkillTable()
        self.appended_rows = None
        self.events = []

    def append_skills(self, rows):
        self.events.append("append_skills")
        self.appended_rows = list(rows)
        self.skills.extend(self.appended_rows)
        return {
            "old_count": 1,
            "new_count": len(self.skills),
            "appended_count": len(self.appended_rows),
            "appended_skill_ids": [row["skill_id"] for row in self.appended_rows],
            "skipped_duplicate_skill_ids": [],
        }

    def to(self, device):
        self.events.append(f"to:{device}")
        self.device = device
        return self

    def eval(self):
        self.events.append("eval")
        return self

    def encode_states(self, texts):
        self.events.append("encode_states")
        self.encoded_texts = list(texts)
        return torch.zeros(len(texts), 2)


def test_audit_appworld_dynamic_routing_appends_after_checkpoint_and_writes_report(tmp_path, monkeypatch):
    base_pool = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool = tmp_path / "dynamic_skill_pool.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    output = tmp_path / "audit.json"
    checkpoint = tmp_path / "stage0.pt"
    checkpoint.write_bytes(b"fake")
    _write_jsonl(base_pool, [{"skill_id": "base/a", "appworld_executor_compatible": False}])
    _write_jsonl(
        dynamic_pool,
        [
            {"skill_id": "base/a", "skill_pool_role": "checkpoint_base", "appworld_executor_compatible": False},
            {
                "skill_id": "skillx/appworld/a",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
            {
                "skill_id": "skillx/appworld/b",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
        ],
    )
    _write_jsonl(
        retrieval,
        [
            {"query_id": "q1", "query_text": "first", "positive_skill_id": "skillx/appworld/a", "split": "train"},
            {"query_id": "q2", "query_text": "second", "positive_skill_id": "skillx/appworld/a", "split": "train"},
        ],
    )
    fake_model = _FakeModel()

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        assert checkpoint_path == checkpoint
        assert skills_path == base_pool
        assert model_cache_dir is None
        return fake_model, {"d": 2}, {"stage0_loaded": True, "skill_count": 1}

    monkeypatch.setattr("clstr.appworld_dynamic_routing_audit.build_clstr_model_from_stage0_checkpoint", fake_build)

    report = audit_appworld_dynamic_routing(
        checkpoint_path=checkpoint,
        base_skill_pool_path=base_pool,
        dynamic_skill_pool_path=dynamic_pool,
        retrieval_path=retrieval,
        output_path=output,
        batch_size=2,
        k_values=[1, 2, 3],
        device="cpu",
    )

    assert fake_model.events.index("to:cpu") < fake_model.events.index("append_skills")
    assert [row["skill_id"] for row in fake_model.appended_rows] == ["skillx/appworld/a", "skillx/appworld/b"]
    assert fake_model.encoded_texts == ["first", "second"]
    assert report["status"] == "ok"
    assert report["append_report"]["appended_count"] == 2
    assert report["rank_metrics"]["global"]["positive_ranks"] == [3, 1]
    assert report["rank_metrics"]["appworld_compatible"]["positive_ranks"] == [2, 1]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_audit_appworld_dynamic_routing_uses_checkpoint_skillrouter_query_format(tmp_path, monkeypatch):
    base_pool = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool = tmp_path / "dynamic_skill_pool.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    checkpoint = tmp_path / "stage0.pt"
    checkpoint.write_bytes(b"fake")
    _write_jsonl(base_pool, [{"skill_id": "base/a", "appworld_executor_compatible": False}])
    _write_jsonl(
        dynamic_pool,
        [
            {"skill_id": "base/a", "skill_pool_role": "checkpoint_base", "appworld_executor_compatible": False},
            {
                "skill_id": "skillx/appworld/a",
                "skill_pool_role": "appworld_appended_after_checkpoint",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
        ],
    )
    _write_jsonl(
        retrieval,
        [{"query_id": "q1", "query_text": "raw appworld task", "positive_skill_id": "skillx/appworld/a"}],
    )
    fake_model = _FakeModel()
    fake_model.skill_table.retrieval_logits = lambda h: torch.tensor([[0.1, 0.9]], dtype=torch.float32)

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        del checkpoint_path, skills_path, model_cache_dir
        return fake_model, {"query_text_format": "skillrouter"}, {"stage0_loaded": True}

    monkeypatch.setattr("clstr.appworld_dynamic_routing_audit.build_clstr_model_from_stage0_checkpoint", fake_build)

    report = audit_appworld_dynamic_routing(
        checkpoint_path=checkpoint,
        base_skill_pool_path=base_pool,
        dynamic_skill_pool_path=dynamic_pool,
        retrieval_path=retrieval,
        batch_size=1,
    )

    assert fake_model.encoded_texts == [
        "Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:raw appworld task"
    ]
    assert report["query_text_format"] == "skillrouter"
