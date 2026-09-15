import json
from pathlib import Path

import torch

from clstr.appworld_corrective_preference import (
    build_api_correction_dataset,
    extract_appworld_api_refs,
    match_solution_api_refs_to_candidate_skills,
    write_api_correction_dataset,
)
from clstr.appworld_current_route_preference_train import (
    compute_current_route_preference_batch_loss,
    configure_current_route_preference_trainable,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_extract_appworld_api_refs_from_solution_code_handles_function_objects():
    code = """
transactions = find_all_from_pages(
    apis.venmo.show_transactions,
    access_token=token,
)
apis.venmo.like_transaction(transaction_id=1, access_token=token)
apis.supervisor.complete_task()
"""

    refs = extract_appworld_api_refs(code)

    assert "apis.venmo.show_transactions" in refs
    assert "apis.venmo.like_transaction" in refs
    assert "apis.supervisor.complete_task" not in refs


def test_match_solution_api_refs_to_candidate_skills_prefers_exact_api_overlap():
    candidates = [
        "skill/bad",
        "skill/good",
        "skill/weak",
    ]
    skills = {
        "skill/bad": {"skill_id": "skill/bad", "executor_desc": "apis.venmo.create_transaction"},
        "skill/good": {
            "skill_id": "skill/good",
            "executor_desc": "apis.venmo.show_transactions, apis.venmo.like_transaction",
        },
        "skill/weak": {"skill_id": "skill/weak", "executor_desc": "apis.venmo.show_transactions"},
    }

    matches = match_solution_api_refs_to_candidate_skills(
        solution_api_refs=["apis.venmo.show_transactions", "apis.venmo.like_transaction"],
        candidate_skill_ids=candidates,
        skill_by_id=skills,
        max_targets=2,
    )

    assert [item["skill_id"] for item in matches] == ["skill/good"]
    assert matches[0]["local_index"] == 1
    assert matches[0]["exact_api_overlap"] == ["apis.venmo.like_transaction", "apis.venmo.show_transactions"]


def test_match_solution_api_refs_requires_write_api_when_solution_has_write_action():
    candidates = ["skill/read-only", "skill/write"]
    skills = {
        "skill/read-only": {"skill_id": "skill/read-only", "executor_desc": "apis.spotify.show_song"},
        "skill/write": {
            "skill_id": "skill/write",
            "executor_desc": "apis.spotify.create_playlist, apis.spotify.add_song_to_playlist, apis.spotify.show_song",
        },
    }

    matches = match_solution_api_refs_to_candidate_skills(
        solution_api_refs=[
            "apis.spotify.create_playlist",
            "apis.spotify.add_song_to_playlist",
            "apis.spotify.show_song",
        ],
        candidate_skill_ids=candidates,
        skill_by_id=skills,
    )

    assert [item["skill_id"] for item in matches] == ["skill/write"]


def test_match_solution_api_refs_preserves_multiple_required_write_apis():
    candidates = ["skill/add-a", "skill/add-b", "skill/create"]
    skills = {
        "skill/add-a": {"skill_id": "skill/add-a", "executor_desc": "apis.spotify.add_song_to_playlist"},
        "skill/add-b": {"skill_id": "skill/add-b", "executor_desc": "apis.spotify.add_song_to_playlist"},
        "skill/create": {"skill_id": "skill/create", "executor_desc": "apis.spotify.create_playlist"},
    }

    matches = match_solution_api_refs_to_candidate_skills(
        solution_api_refs=["apis.spotify.create_playlist", "apis.spotify.add_song_to_playlist"],
        candidate_skill_ids=candidates,
        skill_by_id=skills,
        max_targets=2,
    )

    assert {item["skill_id"] for item in matches} == {"skill/add-a", "skill/create"}


def test_build_api_correction_dataset_adds_target_and_rejected_indices(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    solution_root = tmp_path / "tasks"
    solution_path = solution_root / "task_1" / "ground_truth" / "solution.py"
    solution_path.parent.mkdir(parents=True)
    solution_path.write_text(
        "def _solution(main_user, apis, requester, public_data):\n"
        "    rows = find_all_from_pages(apis.venmo.show_transactions)\n"
        "    apis.venmo.like_transaction(transaction_id=rows[0].transaction_id)\n",
        encoding="utf-8",
    )
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/bad", "executor_desc": "apis.venmo.create_transaction"},
            {
                "skill_id": "skill/good",
                "executor_desc": "apis.venmo.show_transactions, apis.venmo.like_transaction",
            },
        ],
    )
    _write_jsonl(
        rollouts_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "success": False,
                "outcome": {"label": "wrong_completion"},
                "user_goal": "Like relevant Venmo transactions.",
                "decisions": [
                    {
                        "step_idx": 0,
                        "training_ready": True,
                        "state_text": "[User Goal]\nLike relevant Venmo transactions.",
                        "candidate_skill_ids": ["skill/bad", "skill/good"],
                        "selected_skill_ids": ["skill/bad"],
                        "selected_candidate_local_indices": [0],
                        "policy_log_probs": [-0.1, -3.0],
                    }
                ],
            }
        ],
    )

    dataset = build_api_correction_dataset(
        rollouts_path=rollouts_path,
        skill_pool_path=skills_path,
        appworld_tasks_root=solution_root,
    )

    assert dataset.report["sample_count"] == 1
    sample = dataset.samples[0]
    assert sample["sample_type"] == "counterfactual_api_correction"
    assert sample["target_skill_ids"] == ["skill/good"]
    assert sample["target_local_indices"] == [1]
    assert sample["rejected_skill_ids"] == ["skill/bad"]
    assert sample["rejected_local_indices"] == [0]


def test_missing_candidate_api_match_becomes_stage0_retrieval_correction_only(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    solution_root = tmp_path / "tasks"
    solution_path = solution_root / "task_1" / "ground_truth" / "solution.py"
    solution_path.parent.mkdir(parents=True)
    solution_path.write_text(
        "def _solution(main_user, apis, requester, public_data):\n"
        "    playlist = apis.spotify.create_playlist(name='Road trip')\n"
        "    apis.spotify.add_song_to_playlist(playlist_id=playlist.playlist_id, song_id='s1')\n",
        encoding="utf-8",
    )
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/wrong", "executor_desc": "apis.spotify.show_song"},
            {
                "skill_id": "skill/create-playlist",
                "executor_desc": "apis.spotify.create_playlist, apis.spotify.add_song_to_playlist",
            },
        ],
    )
    _write_jsonl(
        rollouts_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "success": False,
                "outcome": {"label": "wrong_completion"},
                "user_goal": "Create a Spotify playlist and add a song.",
                "decisions": [
                    {
                        "step_idx": 0,
                        "training_ready": True,
                        "state_text": "[User Goal]\nCreate a Spotify playlist and add a song.",
                        "candidate_skill_ids": ["skill/wrong"],
                        "selected_skill_ids": ["skill/wrong"],
                        "selected_candidate_local_indices": [0],
                    }
                ],
            }
        ],
    )

    dataset = build_api_correction_dataset(
        rollouts_path=rollouts_path,
        skill_pool_path=skills_path,
        appworld_tasks_root=solution_root,
    )

    assert dataset.samples == []
    assert dataset.report["sample_count"] == 0
    assert dataset.report["stage0_retrieval_correction_count"] == 1
    assert dataset.report["skipped"]["no_candidate_api_match"] == 1
    row = dataset.stage0_retrieval_rows[0]
    assert row["source"] == "appworld_stage0_api_correction"
    assert row["positive_skill_id"] == "skill/create-playlist"
    assert row["negative_skill_ids"] == ["skill/wrong"]
    assert "Required API refs: apis.spotify.add_song_to_playlist, apis.spotify.create_playlist" in row["query_text"]


def test_write_api_correction_dataset_can_write_stage0_retrieval_rows(tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    solution_root = tmp_path / "tasks"
    solution_path = solution_root / "task_1" / "ground_truth" / "solution.py"
    solution_path.parent.mkdir(parents=True)
    solution_path.write_text("apis.spotify.create_playlist(name='x')\n", encoding="utf-8")
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/wrong", "executor_desc": "apis.spotify.show_song"},
            {"skill_id": "skill/create-playlist", "executor_desc": "apis.spotify.create_playlist"},
        ],
    )
    _write_jsonl(
        rollouts_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "success": False,
                "decisions": [
                    {
                        "step_idx": 0,
                        "training_ready": True,
                        "state_text": "Need a playlist.",
                        "candidate_skill_ids": ["skill/wrong"],
                        "selected_skill_ids": ["skill/wrong"],
                    }
                ],
            }
        ],
    )

    stage0_path = tmp_path / "stage0.jsonl"
    dataset = write_api_correction_dataset(
        rollouts_path=rollouts_path,
        skill_pool_path=skills_path,
        appworld_tasks_root=solution_root,
        output_jsonl_path=tmp_path / "stage4.jsonl",
        report_json_path=tmp_path / "report.json",
        stage0_retrieval_jsonl_path=stage0_path,
    )

    assert dataset.stage0_retrieval_rows
    written_rows = [json.loads(line) for line in stage0_path.read_text(encoding="utf-8").splitlines()]
    assert [row["positive_skill_id"] for row in written_rows] == ["skill/create-playlist"]


class _TinySkillTable:
    def __init__(self):
        self.E = torch.eye(2)

    def retrieval_logits(self, h):
        del h
        return torch.zeros(1, 2)


class _TinyPreferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skills = [{"skill_id": "skill/bad"}, {"skill_id": "skill/good"}]
        self.skill_table = _TinySkillTable()
        self.skill_head = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.skill_head.weight)

    @property
    def device(self):
        return torch.device("cpu")

    def encode_states(self, states):
        return torch.zeros(len(states), 2)

    def batch_cross_encode(self, states, candidate_rows):
        del states
        return torch.stack(
            [self.skill_table.E.index_select(0, torch.tensor(candidates, dtype=torch.long)) for candidates in candidate_rows]
        )

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, routing_logits
        skill_logits = self.skill_head(candidate_embs).squeeze(-1)
        stop = torch.full((skill_logits.size(0), 1), -100.0)
        return torch.cat([skill_logits, stop], dim=-1)


def test_pairwise_correction_loss_pushes_good_skill_above_rejected_skill():
    model = _TinyPreferenceModel()
    configure_current_route_preference_trainable(model)
    sample = {
        "sample_type": "counterfactual_api_correction",
        "state_text": "[User Goal]\nLike transactions.",
        "candidate_skill_ids": ["skill/bad", "skill/good"],
        "target_local_indices": [1],
        "rejected_local_indices": [0],
        "previous_selected_skill_ids": [],
        "previous_execute_output": "",
        "weight": 1.0,
    }

    loss, metrics = compute_current_route_preference_batch_loss(
        model,
        [sample],
        skill_id_to_idx={"skill/bad": 0, "skill/good": 1},
        enable_pairwise_correction_loss=True,
        pairwise_correction_margin=0.5,
    )
    loss.backward()

    assert metrics["pairwise_correction_sample_count"] == 1
    assert metrics["pairwise_correction_loss"] > 0.0
    assert model.skill_head.weight.grad is not None
    assert float(model.skill_head.weight.grad[0, 1].item()) < float(model.skill_head.weight.grad[0, 0].item())
