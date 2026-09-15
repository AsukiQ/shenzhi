import json
from pathlib import Path

import torch

from clstr.appworld_dynamic_controller_candidate_audit import audit_appworld_dynamic_controller_candidates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class _FourSkillTable:
    def __init__(self):
        self.E = torch.eye(2)

    def retrieval_logits(self, h):
        return torch.tensor([[100.0, 90.0, 10.0, 9.0]], dtype=torch.float32)

    def belief_logits(self, h):
        return self.retrieval_logits(h)

    def append_skills(self, new_skills, **kwargs):
        del kwargs
        self.E = torch.eye(4)
        return {
            "old_count": 2,
            "new_count": 4,
            "appended_count": len(list(new_skills)),
            "appended_skill_ids": [row["skill_id"] for row in new_skills],
            "skipped_duplicate_skill_ids": [],
        }


class _FourSkillModel:
    def __init__(self):
        self.skills = [
            {"skill_id": "toolbench/high-score-a", "appworld_executor_compatible": False},
            {"skill_id": "toolbench/high-score-b", "appworld_executor_compatible": False},
        ]
        self.skill_table = _FourSkillTable()
        self.K = 2
        self.device = torch.device("cpu")
        self.events = []

    def to(self, device):
        self.events.append(f"to:{device}")
        self.device = torch.device(device)
        return self

    def eval(self):
        self.events.append("eval")
        return self

    def append_skills(self, rows):
        self.events.append("append_skills")
        rows = [dict(row) for row in rows]
        self.skills.extend(rows)
        return self.skill_table.append_skills(rows)

    def encode_states(self, states):
        self.encoded_states = list(states)
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    def batch_cross_encode(self, states, candidate_rows):
        del states
        return torch.eye(4).view(1, 4, 4)[:, : len(candidate_rows[0]), :]

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, routing_logits
        candidate_count = int(candidate_embs.shape[1])
        return torch.cat(
            [
                torch.zeros((1, candidate_count), dtype=torch.float32),
                torch.tensor([[-100.0]], dtype=torch.float32),
            ],
            dim=1,
        )


def test_dynamic_controller_candidate_audit_uses_available_mask_before_topk(tmp_path, monkeypatch):
    base_pool = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool = tmp_path / "dynamic_skill_pool.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    output = tmp_path / "controller_audit.json"
    checkpoint = tmp_path / "stage0.pt"
    checkpoint.write_bytes(b"fake")
    _write_jsonl(
        base_pool,
        [
            {"skill_id": "toolbench/high-score-a", "appworld_executor_compatible": False},
            {"skill_id": "toolbench/high-score-b", "appworld_executor_compatible": False},
        ],
    )
    _write_jsonl(
        dynamic_pool,
        [
            {"skill_id": "toolbench/high-score-a", "appworld_executor_compatible": False},
            {"skill_id": "toolbench/high-score-b", "appworld_executor_compatible": False},
            {
                "skill_id": "skillx/appworld/lower-score-a",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
            {
                "skill_id": "skillx/appworld/lower-score-b",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
        ],
    )
    _write_jsonl(
        retrieval,
        [
            {
                "query_id": "q1",
                "query_text": "playlist task",
                "positive_skill_id": "skillx/appworld/lower-score-a",
            }
        ],
    )
    fake_model = _FourSkillModel()

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        assert checkpoint_path == checkpoint
        assert skills_path == base_pool
        assert model_cache_dir is None
        return fake_model, {"query_text_format": "raw"}, {"stage0_loaded": True, "skill_count": 2}

    monkeypatch.setattr("clstr.appworld_dynamic_controller_candidate_audit.build_clstr_model_from_stage0_checkpoint", fake_build)

    report = audit_appworld_dynamic_controller_candidates(
        checkpoint_path=checkpoint,
        base_skill_pool_path=base_pool,
        dynamic_skill_pool_path=dynamic_pool,
        retrieval_path=retrieval,
        output_path=output,
        candidate_top_k=2,
        top_k=2,
        device="cpu",
    )

    assert fake_model.events.index("to:cpu") < fake_model.events.index("append_skills")
    assert report["status"] == "ok"
    assert report["candidate_positive_hit_rate"] == 1.0
    assert report["selected_positive_hit_rate"] == 1.0
    assert report["rows"][0]["candidate_positive_hit"] is True
    assert report["rows"][0]["candidate_skill_ids"] == [
        "skillx/appworld/lower-score-a",
        "skillx/appworld/lower-score-b",
    ]
    assert report["rows"][0]["diagnostics"]["available_candidate_mask_enabled"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_positive_hit_rate"] == 1.0


def test_dynamic_controller_candidate_audit_can_use_official_schema_state(tmp_path, monkeypatch):
    base_pool = tmp_path / "base_skill_pool.jsonl"
    dynamic_pool = tmp_path / "dynamic_skill_pool.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    output = tmp_path / "controller_audit.json"
    checkpoint = tmp_path / "stage0.pt"
    checkpoint.write_bytes(b"fake")
    docs_dir = tmp_path / "appworld_root" / "data" / "api_docs" / "standard"
    docs_dir.mkdir(parents=True)
    (docs_dir / "spotify.json").write_text(
        """
{
  "create_playlist": {"description": "Create a new playlist.", "parameters": [{"name": "title", "type": "string", "required": true}], "response_schemas": {"success": {"playlist_id": 1}}},
  "show_song_library": {"description": "Get songs.", "parameters": [], "response_schemas": {"success": [{"song_id": 1}]}}
}
""",
        encoding="utf-8",
    )
    _write_jsonl(
        base_pool,
        [
            {"skill_id": "toolbench/high-score-a", "appworld_executor_compatible": False},
            {"skill_id": "toolbench/high-score-b", "appworld_executor_compatible": False},
        ],
    )
    _write_jsonl(
        dynamic_pool,
        [
            {"skill_id": "toolbench/high-score-a", "appworld_executor_compatible": False},
            {"skill_id": "toolbench/high-score-b", "appworld_executor_compatible": False},
            {
                "skill_id": "skillx/appworld/lower-score-a",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
            {
                "skill_id": "skillx/appworld/lower-score-b",
                "appworld_executor_compatible": True,
                "is_appended_after_checkpoint": True,
            },
        ],
    )
    _write_jsonl(
        retrieval,
        [
            {
                "query_id": "task_1::correction::0",
                "query_text": "diagnostic correction query",
                "positive_skill_id": "skillx/appworld/lower-score-a",
            }
        ],
    )
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Make a Spotify playlist.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.create_playlist"],
            }
        ],
    )
    fake_model = _FourSkillModel()

    def fake_build(*, checkpoint_path, skills_path, model_cache_dir=None):
        del model_cache_dir
        assert checkpoint_path == checkpoint
        assert skills_path == base_pool
        return fake_model, {"query_text_format": "raw"}, {"stage0_loaded": True, "skill_count": 2}

    monkeypatch.setattr("clstr.appworld_dynamic_controller_candidate_audit.build_clstr_model_from_stage0_checkpoint", fake_build)

    report = audit_appworld_dynamic_controller_candidates(
        checkpoint_path=checkpoint,
        base_skill_pool_path=base_pool,
        dynamic_skill_pool_path=dynamic_pool,
        retrieval_path=retrieval,
        output_path=output,
        tasks_path=tasks_path,
        appworld_root=tmp_path / "appworld_root",
        state_text_source="official_schema_state",
        candidate_top_k=2,
        top_k=2,
        device="cpu",
    )

    encoded_state = fake_model.encoded_states[0]
    assert "[Available API Inventory]" in encoded_state
    assert "apis.spotify.create_playlist" in encoded_state
    assert "diagnostic correction query" not in encoded_state
    assert report["state_text_source"] == "official_schema_state"
    assert report["rows"][0]["state_text_chars"] == len(encoded_state)
