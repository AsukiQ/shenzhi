import json
from pathlib import Path

import pytest
import torch

from clstr.appworld_executor import NullSkillProvider
from clstr.appworld_multistep import (
    CLSTRMultiStepController,
    HybridCLSTRSkillRouterController,
    LiveEmbeddingStepController,
    StaticSkillProviderStepController,
    build_appworld_eval_step_label_tasks,
    build_appworld_oracle_multistep_trajectories,
    build_multistep_executor_prompt,
    build_multistep_executor_comparison,
    build_multistep_failure_diagnostics,
    build_multistep_state_text,
    build_train_multistep_trajectories_from_runs,
    run_appworld_multistep_executor_eval,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fake_appworld_root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld_root"
    _write_json(
        root / "data" / "api_docs" / "standard" / "spotify.json",
        {
            "show_song_library": {
                "description": "List songs.",
                "parameters": [{"name": "page_index", "type": "integer", "required": False}],
                "response_schemas": {"success": [{"song_id": 1}]},
            },
            "show_song": {
                "description": "Show song.",
                "parameters": [{"name": "song_id", "type": "integer", "required": True}],
                "response_schemas": {"success": {"title": "string"}},
            },
            "search_songs": {
                "description": "Search songs.",
                "parameters": [
                    {"name": "artist_id", "type": "integer", "required": False},
                    {"name": "min_play_count", "type": "integer", "required": False},
                    {"name": "page_limit", "type": "integer", "required": False},
                ],
                "response_schemas": {
                    "success": [
                        {
                            "song_id": 1,
                            "artists": [{"name": "string"}],
                            "play_count": 1,
                        }
                    ]
                },
            },
            "add_to_queue": {
                "description": "Add a song to the queue.",
                "parameters": [
                    {"name": "access_token", "type": "string", "required": True},
                    {"name": "song_id", "type": "integer", "required": True},
                ],
                "response_schemas": {"success": {}},
            },
        },
    )
    return root


class _TwoStepGenerator:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            assert "[Execution History]" in prompt
            assert "No previous execution steps." in prompt
            return "song_ids = [1]\nprint('loaded ids')"
        assert "library returned song id 1" in prompt
        return "apis.supervisor.complete_task(answer='Song A', status='success')"


class _TwoStepWorld:
    def __init__(self, task_id, experiment_name, **kwargs):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self.executed: list[str] = []
        self.closed = False

    def execute(self, code):
        self.executed.append(code)
        if len(self.executed) == 1:
            return "Execution successful.\nlibrary returned song id 1"
        return "Execution successful.\ncompleted"

    def task_completed(self):
        return len(self.executed) >= 2

    def evaluate(self, suppress_errors=True):
        return {"success": len(self.executed) >= 2, "suppress_errors": suppress_errors}

    def close(self):
        self.closed = True


class _FakeMultiStepSkillTable:
    def __init__(self):
        self.E = torch.eye(3)

    def logits(self, h):
        return torch.tensor([[3.0, 2.0, 1.0]])

    def retrieval_logits(self, h):
        return self.logits(h)

    def belief_logits(self, h):
        return self.logits(h)


class _FakeMultiStepModel:
    def __init__(self):
        self.skill_table = _FakeMultiStepSkillTable()
        self.K = 3
        self.device = torch.device("cpu")
        self.stop_idx = 3
        self.updated = False
        self.observations: list[str] = []
        self.next_texts: list[str] = []

    def eval(self):
        return self

    def encode_states(self, states):
        return torch.tensor([[1.0, 0.0, 0.0]])

    def batch_cross_encode(self, states, candidate_rows):
        return torch.eye(3).view(1, 3, 3)[:, : len(candidate_rows[0]), :]

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        if self.updated:
            return torch.tensor([[0.0, 5.0, 1.0, -100.0]])
        return torch.tensor([[4.0, 1.0, 0.0, -100.0]])

    def action_embeddings(self, action_ids):
        flat = action_ids.reshape(-1).to(dtype=torch.long)
        embs = self.skill_table.E.index_select(0, flat)[:, :2]
        return embs.view(*action_ids.shape, -1)

    def encode_observations(self, observations):
        self.observations.extend(str(item) for item in observations)
        return torch.tensor([[0.0, 1.0, 0.0]])

    def step_update(self, m_t, a_t, o_t_emb, x_next_text):
        self.updated = True
        self.next_texts.extend(str(item) for item in x_next_text)
        return torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[0.0, 1.0, 0.0]])


class _FourSkillTable:
    def __init__(self):
        self.E = torch.eye(4)

    def retrieval_logits(self, h):
        return torch.tensor([[100.0, 90.0, 10.0, 9.0]])

    def belief_logits(self, h):
        return self.retrieval_logits(h)


class _FourSkillModel:
    def __init__(self):
        self.skill_table = _FourSkillTable()
        self.K = 2
        self.device = torch.device("cpu")
        self.stop_idx = 4

    def eval(self):
        return self

    def encode_states(self, states):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def batch_cross_encode(self, states, candidate_rows):
        return torch.eye(4).view(1, 4, 4)[:, : len(candidate_rows[0]), :]

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        candidate_count = int(candidate_embs.shape[1])
        return torch.cat(
            [
                torch.zeros((1, candidate_count), dtype=torch.float32),
                torch.tensor([[-100.0]], dtype=torch.float32),
            ],
            dim=1,
        )


class _InitialBeliefPointsToSkillBModel(_FakeMultiStepModel):
    def encode_states(self, states):
        return torch.tensor([[0.0, 1.0, 0.0]])


class _FakeHybridRetriever:
    def __init__(self, skills):
        self.skill_ids = [skill["skill_id"] for skill in skills]
        self.skill_payloads = list(skills)
        self.reset_called = False

    def reset(self, task):
        self.reset_called = True

    def rank_state(self, state_text: str, top_k: int):
        return [0, 1, 2][:top_k], torch.tensor([3.0, 2.0, 1.0])[:top_k]


def test_build_multistep_state_text_compacts_goal_and_history():
    state_text = build_multistep_state_text(
        task={"instruction_text": "Find my Spotify song.", "required_apps": ["spotify"]},
        steps=[
            {
                "step_idx": 0,
                "selected_skill_ids": ["skill-a"],
                "code": "print('hello')",
                "execute_output": "Execution successful.",
                "execution_ok": True,
                "task_completed": False,
            }
        ],
    )

    assert "Find my Spotify song." in state_text
    assert "Required apps: spotify" in state_text
    assert "Step 0" in state_text
    assert "skill-a" in state_text
    assert "Execution successful." in state_text
    assert "Need choose the next useful SkillX skill/context" in state_text

    executor_state_text = build_multistep_state_text(
        task={"instruction_text": "Find my Spotify song.", "required_apps": ["spotify"]},
        steps=[
            {
                "step_idx": 0,
                "selected_skill_ids": ["skill-a"],
                "code": "print('hello')",
                "execute_output": "Execution successful.",
                "execution_ok": True,
                "task_completed": False,
            }
        ],
        include_selected_skill_ids=False,
    )
    assert "- selected_skill_ids:" not in executor_state_text
    assert "Execution successful." in executor_state_text


def test_multistep_executor_prompt_tells_model_to_continue_after_successful_history():
    task = {
        "task_id": "task_1",
        "instruction_text": "Send Joseph the owed grocery money and text them.",
        "required_apps": ["phone", "venmo"],
    }
    state_text = build_multistep_state_text(
        task=task,
        steps=[
            {
                "step_idx": 0,
                "selected_skill_ids": ["skillx/appworld/login-to-venmo-and-phone-12"],
                "code": "phone_access_token = 'token'\nvenmo_access_token = 'token'",
                "execute_output": "Execution successful.",
                "execution_ok": True,
                "task_completed": False,
            }
        ],
    )

    prompt = build_multistep_executor_prompt(
        task=task,
        state_text=state_text,
        skills=[],
        api_docs_context="[AppWorld API Docs]",
        skill_context_mode="safe_metadata",
    )

    assert "When [Execution History] contains successful code, assume prior variables and app state persist across steps." in prompt
    assert "Only variables explicitly assigned with `=` in previous successful code persist across steps." in prompt
    assert "Bare calls such as `apis.supervisor.show_profile()` do not create `supervisor_profile`." in prompt
    assert "Do not repeat successful login-only code; continue with the next missing API calls." in prompt
    assert "If a previous successful step already created the requested files, transactions, messages, or records but `task_completed` is false, call `complete_task` instead of repeating the same loop." in prompt
    assert "If evaluation history says records such as `venmo.Transaction`, `phone.GlobalTextMessage`, or `phone.UserTextMessage` are missing, create those records before `complete_task`." in prompt


def test_multistep_executor_prompt_warns_failed_preflight_has_no_variables():
    task = {
        "task_id": "task_1",
        "instruction_text": "Organize my files.",
        "required_apps": ["file_system"],
    }
    state_text = build_multistep_state_text(
        task=task,
        steps=[
            {
                "step_idx": 0,
                "selected_skill_ids": ["skillx/appworld/file-system-organize-files-by-date-9"],
                "code": "file_system_access_token = 'token'",
                "execute_output": "Preflight failed before execution: Python syntax error",
                "execution_ok": False,
                "task_completed": False,
            }
        ],
    )

    prompt = build_multistep_executor_prompt(
        task=task,
        state_text=state_text,
        skills=[],
        api_docs_context="[AppWorld API Docs]",
        skill_context_mode="safe_metadata",
    )

    assert "If a previous step says `Preflight failed before execution`, none of its variables or side effects exist." in prompt
    assert "Regenerate a complete self-contained program including login/setup after a preflight failure." in prompt


def test_multistep_schema_plan_uses_full_schema_refs():
    task = {
        "task_id": "task_1",
        "instruction_text": "Inspect my Spotify song.",
        "required_apps": ["spotify"],
    }
    state_text = build_multistep_state_text(task=task, steps=[])

    prompt = build_multistep_executor_prompt(
        task=task,
        state_text=state_text,
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-song-detail-1",
                "name": "spotify find song detail",
                "description": "Inspect a song.",
                "executor_desc": "apis.spotify.show_song",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(access_token: string required): List songs.\n",
        skill_context_mode="schema_plan",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "show_song"),
            ("supervisor", "complete_task"),
        },
    )

    assert "schema-grounded constraints" in prompt
    assert "valid_appworld_apis: apis.spotify.show_song" in prompt
    assert "skillx/appworld/spotify-find-song-detail-1" not in prompt
    assert "spotify find song detail" not in prompt


def test_multistep_executor_records_gated_schema_handoff_diagnostics(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "How many unique songs are there across my Spotify song library?",
                "required_apps": ["spotify"],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "query_id": "task_1",
                "ranked_skill_ids": ["skillx/appworld/spotify-find-songs-based-on-play-count-59"],
            }
        ],
    )

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_gated_schema",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="skillrouter_multistep",
        controller=StaticSkillProviderStepController.from_prediction_file(
            skill_pool_path=skill_pool_path,
            predictions_path=predictions_path,
        ),
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        skill_context_mode="gated_schema",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    handoff = runs[0]["steps"][0]["skill_handoff"]
    assert handoff["skill_context_mode"] == "gated_schema"
    assert handoff["schema_plan_skill_count"] == 1
    assert handoff["safe_metadata_skill_count"] == 0
    assert handoff["handoff_decisions"][0]["decision"] == "schema_plan"
    assert "pollution_pattern" in handoff["handoff_decisions"][0]["reasons"]


def test_multistep_executor_records_api_evidence_handoff_diagnostics(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "How many unique songs are there across my Spotify song library?",
                "required_apps": ["spotify"],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song, apis.spotify.fake_api",
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "query_id": "task_1",
                "ranked_skill_ids": ["skillx/appworld/spotify-find-songs-based-on-play-count-59"],
            }
        ],
    )

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_api_evidence",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="skillrouter_multistep",
        controller=StaticSkillProviderStepController.from_prediction_file(
            skill_pool_path=skill_pool_path,
            predictions_path=predictions_path,
        ),
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        skill_context_mode="api_evidence",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    handoff = runs[0]["steps"][0]["skill_handoff"]
    assert handoff["skill_context_mode"] == "api_evidence"
    assert handoff["selected_skill_ids"] == ["skillx/appworld/spotify-find-songs-based-on-play-count-59"]
    assert handoff["api_evidence_refs"] == ["apis.spotify.show_song", "apis.spotify.show_song_library"]
    assert handoff["read_support_apis"] == ["apis.spotify.show_song", "apis.spotify.show_song_library"]
    assert handoff["invalid_skill_api_refs"] == ["apis.spotify.fake_api"]
    assert handoff["hidden_skill_text_fields"] == ["skill_id", "name", "description", "body", "skill_md"]


def test_multistep_executor_records_verified_hints_handoff_diagnostics(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "How many unique songs are there across my Spotify song library?",
                "required_apps": ["spotify"],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "query_id": "task_1",
                "ranked_skill_ids": ["skillx/appworld/spotify-find-songs-based-on-play-count-59"],
            }
        ],
    )

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_verified_hints",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="skillrouter_multistep",
        controller=StaticSkillProviderStepController.from_prediction_file(
            skill_pool_path=skill_pool_path,
            predictions_path=predictions_path,
        ),
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        skill_context_mode="verified_hints",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    handoff = runs[0]["steps"][0]["skill_handoff"]
    assert handoff["skill_context_mode"] == "verified_hints"
    assert handoff["handoff_decisions"][0]["decision"] == "suppress"
    assert "unique_count_goal_vs_ranking_skill" in handoff["handoff_decisions"][0]["reasons"]
    assert handoff["handoff_decision_counts"]["suppress"] == 1
    assert handoff["hidden_skill_text_fields"] == ["skill_id", "name", "description", "body", "skill_md"]


def test_clstr_multistep_controller_updates_belief_between_steps():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        allow_legacy_policy_skill_router=True,
    )
    task = {"task_id": "task_1", "instruction_text": "Find song."}

    controller.reset(task)
    first = controller.select(task=task, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(
        task=task,
        state_text="state 2 with environment output",
        steps=[{"step_idx": 0}],
        top_k=1,
    )

    assert first.selected_skill_ids == ["skill-a"]
    assert second.selected_skill_ids == ["skill-b"]
    assert model.observations == ["Execution successful. ids loaded"]
    assert model.next_texts == ["Execution successful. ids loaded"]
    assert second.diagnostics["belief_updates"] == 1
    assert second.diagnostics["candidate_skill_ids"] == ["skill-a", "skill-b", "skill-c"]


def test_clstr_multistep_policy_ranking_requires_legacy_opt_in():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]

    with pytest.raises(ValueError, match="legacy_policy_skill_router"):
        CLSTRMultiStepController(
            model=model,
            skills=skills,
            ranking_mode="policy_head",
            candidate_top_k=3,
        )


def test_clstr_multistep_controller_records_policy_logprobs_for_current_route_training():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        allow_legacy_policy_skill_router=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=2)

    expected = torch.log_softmax(torch.tensor([4.0, 1.0, 0.0]), dim=-1)
    assert selection.selected_skill_ids == ["skill-a", "skill-b"]
    assert selection.diagnostics["selected_candidate_local_indices"] == [0, 1]
    assert selection.diagnostics["selected_log_probs"] == [
        float(expected[0].item()),
        float(expected[1].item()),
    ]
    assert selection.diagnostics["policy_log_probs"] == [float(item) for item in expected.tolist()]
    assert selection.diagnostics["candidate_policy_logits"] == [4.0, 1.0, 0.0]


def test_clstr_multistep_controller_can_filter_to_appworld_executor_compatible_skills():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A", "appworld_executor_compatible": True},
        {"skill_id": "toolbench/not-executable", "name": "ToolBench", "appworld_executor_compatible": False},
        {"skill_id": "skill-c", "name": "C", "appworld_executor_compatible": True},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="skill_table",
        candidate_top_k=3,
        appworld_executor_compatible_only=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state", steps=[], top_k=2)

    assert selection.selected_skill_ids == ["skill-a", "skill-c"]
    assert selection.diagnostics["candidate_skill_ids"] == ["skill-a", "skill-c"]
    assert selection.diagnostics["available_candidate_mask_enabled"] is True
    assert selection.diagnostics["available_candidate_count"] == 2
    assert selection.diagnostics["candidate_pool_before_available_filter"] == 3
    assert selection.diagnostics["available_filtered_executor_incompatible_skills"] == 1
    assert selection.diagnostics["filtered_executor_incompatible_skills"] == 0
    assert selection.diagnostics["executor_compatible_only"] is True


def test_clstr_multistep_controller_executor_compatible_filter_requires_positive_evidence():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "toolret/missing-compat-flag", "name": "ToolRet"},
        {"skill_id": "skillx/appworld/by-id", "name": "SkillX by id"},
        {"skill_id": "domain/appworld", "name": "SkillX by domain", "executor_domain": "appworld"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="skill_table",
        candidate_top_k=3,
        appworld_executor_compatible_only=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state", steps=[], top_k=2)

    assert selection.selected_skill_ids == ["skillx/appworld/by-id", "domain/appworld"]
    assert selection.diagnostics["available_candidate_mask_enabled"] is True
    assert selection.diagnostics["available_candidate_count"] == 2
    assert selection.diagnostics["available_filtered_executor_incompatible_skills"] == 1
    assert selection.diagnostics["filtered_executor_incompatible_skills"] == 0


def test_clstr_multistep_controller_applies_executor_compatible_filter_before_candidate_truncation():
    model = _FourSkillModel()
    skills = [
        {"skill_id": "toolbench/high-score-a", "name": "ToolBench A", "appworld_executor_compatible": False},
        {"skill_id": "toolbench/high-score-b", "name": "ToolBench B", "appworld_executor_compatible": False},
        {"skill_id": "skillx/appworld/lower-score-a", "name": "AppWorld A", "appworld_executor_compatible": True},
        {"skill_id": "skillx/appworld/lower-score-b", "name": "AppWorld B", "appworld_executor_compatible": True},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="skill_table",
        candidate_top_k=2,
        appworld_executor_compatible_only=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state", steps=[], top_k=2)

    assert selection.selected_skill_ids == ["skillx/appworld/lower-score-a", "skillx/appworld/lower-score-b"]
    assert selection.diagnostics["available_candidate_mask_enabled"] is True
    assert selection.diagnostics["available_candidate_count"] == 2
    assert selection.diagnostics["candidate_skill_ids"] == [
        "skillx/appworld/lower-score-a",
        "skillx/appworld/lower-score-b",
    ]
    assert selection.diagnostics["routing_candidate_skill_ids"] == [
        "skillx/appworld/lower-score-a",
        "skillx/appworld/lower-score-b",
    ]


def test_clstr_multistep_controller_updates_belief_from_next_state_text_when_available():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        allow_legacy_policy_skill_router=True,
    )

    controller.reset({"task_id": "task_1"})
    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={
            "step_idx": 0,
            "next_state_text": "[User Goal]\nFind song.\n[Execution History]\nStep 0 environment_output: ids loaded",
        },
    )

    assert model.observations == ["Execution successful. ids loaded"]
    assert model.next_texts == ["[User Goal]\nFind song.\n[Execution History]\nStep 0 environment_output: ids loaded"]


def test_clstr_multistep_controller_can_disable_recurrent_belief_updates():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        recurrent_belief=False,
        allow_legacy_policy_skill_router=True,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert second.diagnostics["recurrent_belief"] is False


def test_clstr_multistep_controller_can_union_belief_candidates_after_update():
    model = _FakeMultiStepModel()
    model.K = 1
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=1,
        candidate_source="routing_belief_union",
        allow_legacy_policy_skill_router=True,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert first.diagnostics["candidate_skill_ids"] == ["skill-a"]
    assert first.diagnostics["belief_candidate_enabled"] is True
    assert first.diagnostics["routing_candidate_skill_ids"] == ["skill-a"]
    assert first.diagnostics["belief_candidate_skill_ids"] == ["skill-a"]
    assert "skill-b" in second.diagnostics["candidate_skill_ids"]
    assert second.diagnostics["belief_candidate_enabled"] is True
    assert second.diagnostics["routing_candidate_skill_ids"] == ["skill-a"]
    assert second.diagnostics["belief_candidate_skill_ids"] == ["skill-b"]
    assert second.selected_skill_ids == ["skill-b"]
    assert second.diagnostics["belief_updates"] == 1


def test_clstr_multistep_after_update_candidate_source_does_not_union_initial_belief():
    model = _InitialBeliefPointsToSkillBModel()
    model.K = 1
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=1,
        candidate_source="routing_belief_union_after_update",
        allow_legacy_policy_skill_router=True,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert first.diagnostics["candidate_skill_ids"] == ["skill-a"]
    assert first.diagnostics["belief_candidate_enabled"] is False
    assert first.diagnostics["routing_candidate_skill_ids"] == ["skill-a"]
    assert first.diagnostics["belief_candidate_skill_ids"] == []
    assert "skill-b" in second.diagnostics["candidate_skill_ids"]
    assert second.diagnostics["belief_candidate_enabled"] is True
    assert second.diagnostics["routing_candidate_skill_ids"] == ["skill-a"]
    assert second.diagnostics["belief_candidate_skill_ids"] == ["skill-b"]
    assert second.selected_skill_ids == ["skill-b"]


def test_clstr_multistep_transition_blend_uses_stage4_transition_after_observe(monkeypatch):
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    captured = {}

    def fake_transition_candidate_logits_for_mode(
        model_arg,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb,
        skill_count,
        *,
        candidate_ids=None,
        candidate_valid_mask=None,
        residual_lambda=0.0,
        scoring_mode="v4_1b_action_observation",
    ):
        captured["current_labels"] = current_labels.detach().cpu().tolist()
        captured["candidate_ids"] = candidate_ids.detach().cpu().tolist()
        captured["action_emb_shape"] = list(action_emb.shape)
        captured["residual_lambda"] = residual_lambda
        captured["scoring_mode"] = scoring_mode
        logits = torch.tensor([[0.0, 6.0, 1.0]], device=h.device)
        return logits, "fake_stage4_transition", logits, logits

    monkeypatch.setattr(
        "clstr.appworld_multistep._transition_candidate_logits_for_mode",
        fake_transition_candidate_logits_for_mode,
    )
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="transition_blend",
        candidate_top_k=3,
        policy_blend_alpha=1.0,
        transition_scoring_mode="v4_1b_action_observation",
        transition_residual_lambda=0.0,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert first.selected_skill_ids == ["skill-a"]
    assert first.diagnostics["transition_scores_available"] is False
    assert first.diagnostics["transition_fallback_reason"] == "missing_previous_action"
    assert second.selected_skill_ids == ["skill-b"]
    assert second.diagnostics["ranking_mode"] == "transition_blend"
    assert second.diagnostics["transition_scores_available"] is True
    assert second.diagnostics["transition_head_type"] == "fake_stage4_transition"
    assert captured == {
        "current_labels": [0],
        "candidate_ids": [[0, 1, 2]],
        "action_emb_shape": [1, 3],
        "residual_lambda": 0.0,
        "scoring_mode": "v4_1b_action_observation",
    }


class _PolicyTransitionBlendModel(_FakeMultiStepModel):
    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, candidate_embs, routing_logits
        if self.updated:
            return torch.tensor([[0.0, 0.0, 0.0, -100.0]])
        return torch.tensor([[0.0, 6.0, 0.0, -100.0]])


def test_clstr_multistep_policy_transition_blend_uses_policy_then_transition(monkeypatch):
    model = _PolicyTransitionBlendModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]

    def fake_transition_candidate_logits_for_mode(
        model_arg,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb,
        skill_count,
        *,
        candidate_ids=None,
        candidate_valid_mask=None,
        residual_lambda=0.0,
        scoring_mode="v4_1b_action_observation",
    ):
        del model_arg, h, m_obs, current_labels, obs_emb, action_emb, skill_count
        del candidate_ids, candidate_valid_mask, residual_lambda, scoring_mode
        logits = torch.tensor([[0.0, 0.0, 6.0]])
        return logits, "fake_stage4_transition", logits, logits

    monkeypatch.setattr(
        "clstr.appworld_multistep._transition_candidate_logits_for_mode",
        fake_transition_candidate_logits_for_mode,
    )
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_transition_blend",
        candidate_top_k=3,
        policy_blend_alpha=1.0,
        allow_legacy_policy_skill_router=True,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert first.selected_skill_ids == ["skill-b"]
    assert first.diagnostics["ranking_mode"] == "policy_transition_blend"
    assert first.diagnostics["transition_scores_available"] is False
    assert first.diagnostics["transition_fallback_reason"] == "missing_previous_action"
    assert second.selected_skill_ids == ["skill-c"]
    assert second.diagnostics["transition_scores_available"] is True
    assert second.diagnostics["transition_head_type"] == "fake_stage4_transition"


class _SixSkillTable:
    def __init__(self):
        self.E = torch.eye(6)

    def logits(self, h):
        return torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])

    def retrieval_logits(self, h):
        return self.logits(h)

    def belief_logits(self, h):
        return self.logits(h)


class _SixSkillTransitionBlendModel(_FakeMultiStepModel):
    def __init__(self):
        super().__init__()
        self.skill_table = _SixSkillTable()
        self.K = 6
        self.stop_idx = 6

    def encode_states(self, states):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def batch_cross_encode(self, states, candidate_rows):
        return torch.eye(6).view(1, 6, 6)[:, : len(candidate_rows[0]), :]

    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, candidate_embs, routing_logits
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0]])

    def encode_observations(self, observations):
        self.observations.extend(str(item) for item in observations)
        return torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def step_update(self, m_t, a_t, o_t_emb, x_next_text):
        self.updated = True
        self.next_texts.extend(str(item) for item in x_next_text)
        return (
            torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        )


def test_clstr_multistep_policy_transition_blend_keeps_learned_rerank_inside_routing_trust_region(
    monkeypatch,
):
    model = _SixSkillTransitionBlendModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
        {"skill_id": "skill-d", "name": "D"},
        {"skill_id": "skill-e", "name": "E"},
        {"skill_id": "skill-f", "name": "F"},
    ]

    def fake_transition_candidate_logits_for_mode(
        model_arg,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb,
        skill_count,
        *,
        candidate_ids=None,
        candidate_valid_mask=None,
        residual_lambda=0.0,
        scoring_mode="v4_1b_action_observation",
    ):
        del model_arg, h, m_obs, current_labels, obs_emb, action_emb, skill_count
        del candidate_ids, candidate_valid_mask, residual_lambda, scoring_mode
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 100.0]])
        return logits, "fake_stage4_transition", logits, logits

    monkeypatch.setattr(
        "clstr.appworld_multistep._transition_candidate_logits_for_mode",
        fake_transition_candidate_logits_for_mode,
    )
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_transition_blend",
        candidate_top_k=6,
        policy_blend_alpha=0.5,
        learned_component_trust_top_k=3,
        allow_legacy_policy_skill_router=True,
    )

    first = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)
    controller.observe(
        selection=first,
        code="print('ids')",
        execute_output="Execution successful. ids loaded",
        step={"step_idx": 0},
    )
    second = controller.select(task={"task_id": "task_1"}, state_text="state 2", steps=[{"step_idx": 0}], top_k=1)

    assert first.selected_skill_ids == ["skill-a"]
    assert second.selected_skill_ids == ["skill-a"]
    assert second.diagnostics["learned_component_enabled"] is True
    assert second.diagnostics["learned_component_trust_top_k"] == 3
    assert second.diagnostics["learned_component_trusted_candidate_count"] == 3
    assert second.diagnostics["learned_component_outside_trust_candidate_count"] == 3
    assert second.diagnostics["selected_candidate_local_indices"] == [0]


class _LowConfidencePolicyTransitionBlendModel(_FakeMultiStepModel):
    def policy_forward(self, h_t, m_t, candidate_embs, routing_logits=None):
        del h_t, m_t, candidate_embs, routing_logits
        return torch.tensor([[0.0, 0.001, 0.0, -100.0]])


def test_clstr_multistep_policy_transition_blend_gates_low_confidence_policy():
    model = _LowConfidencePolicyTransitionBlendModel()
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_transition_blend",
        candidate_top_k=3,
        policy_blend_alpha=0.5,
        learned_component_min_range=0.1,
        allow_legacy_policy_skill_router=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state 1", steps=[], top_k=1)

    assert selection.selected_skill_ids == ["skill-a"]
    assert selection.diagnostics["policy_component_enabled"] is False
    assert selection.diagnostics["policy_component_raw_range"] == 0.0010000000474974513
    assert selection.diagnostics["learned_component_enabled"] is False
    assert selection.diagnostics["selected_scores"] == [3.0]


def test_clstr_multistep_controller_can_limit_auth_like_contexts():
    model = _FakeMultiStepModel()
    skills = [
        {"skill_id": "skill-a", "name": "login auth primary", "description": "Authenticate."},
        {"skill_id": "skill-b", "name": "login auth duplicate", "description": "Authenticate again."},
        {"skill_id": "skill-c", "name": "spotify analyze songs", "description": "Analyze songs."},
    ]
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        max_auth_like_skills=1,
        allow_legacy_policy_skill_router=True,
    )

    selection = controller.select(task={"task_id": "task_1"}, state_text="state", steps=[], top_k=2)

    assert selection.selected_skill_ids == ["skill-a", "skill-c"]
    assert selection.diagnostics["max_auth_like_skills"] == 1
    assert selection.diagnostics["filtered_auth_like_skills"] == 1


def test_hybrid_controller_uses_skillrouter_candidates_and_clstr_policy_scores():
    model = _FakeMultiStepModel()
    model.updated = True
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    retriever = _FakeHybridRetriever(skills)
    controller = HybridCLSTRSkillRouterController(
        model=model,
        skills=skills,
        retriever=retriever,
        clstr_alpha=1.0,
        candidate_top_k=3,
    )

    controller.reset({"task_id": "task_1"})
    selection = controller.select(task={"task_id": "task_1"}, state_text="state", steps=[], top_k=1)

    assert retriever.reset_called is True
    assert selection.selected_skill_ids == ["skill-b"]
    assert selection.diagnostics["controller"] == "hybrid_clstr_skillrouter"
    assert selection.diagnostics["candidate_source"] == "skillrouter_live"
    assert selection.diagnostics["clstr_alpha"] == 1.0


def test_live_embedding_step_controller_reranks_from_current_state_text():
    skills = [
        {"skill_id": "skill-a", "name": "Skill A", "description": "first route"},
        {"skill_id": "skill-b", "name": "Skill B", "description": "second route"},
    ]

    def fake_encode(texts):
        rows = []
        for text in texts:
            if "second" in text.lower() or "skill b" in text.lower():
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    controller = LiveEmbeddingStepController(
        skills=skills,
        encode_texts=fake_encode,
        query_text_fn=lambda state_text: state_text,
        skill_text_fn=lambda skill: f"{skill['name']} {skill.get('description', '')}",
    )
    task = {"task_id": "task_1"}

    first = controller.select(task=task, state_text="first observation", steps=[], top_k=1)
    second = controller.select(task=task, state_text="second observation", steps=[], top_k=1)

    assert first.selected_skill_ids == ["skill-a"]
    assert second.selected_skill_ids == ["skill-b"]
    assert second.diagnostics["controller"] == "live_embedding_step"
    assert second.diagnostics["candidate_source"] == "current_state_text"


def test_multistep_executor_records_step_trajectory_and_metrics(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Find my Spotify song.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.show_song_library", "spotify.show_song"],
                "positive_skill_ids": ["skillx/appworld/spotify-song"],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-song",
                "name": "spotify inspect song",
                "description": "Inspect Spotify songs.",
                "executor_desc": "spotify.show_song_library spotify.show_song",
                "body": "Find song ids and inspect them.",
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, [{"query_id": "task_1", "ranked_skill_ids": ["skillx/appworld/spotify-song"]}])

    generator = _TwoStepGenerator()
    controller = StaticSkillProviderStepController.from_prediction_file(
        skill_pool_path=skill_pool_path,
        predictions_path=predictions_path,
    )
    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="skillrouter_multistep",
        controller=controller,
        generator=generator,
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=3,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    assert report["status"] == "ok"
    assert report["success_count"] == 1
    assert report["average_steps"] == 2.0
    assert report["task_completed_count"] == 1
    assert report["evaluate_success_count"] == 1
    assert report["step_positive_skill_hit_rate"] == 1.0
    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert runs[0]["success"] is True
    assert runs[0]["final_success"] is True
    assert len(runs[0]["steps"]) == 2
    assert runs[0]["steps"][0]["state_text"]
    assert runs[0]["steps"][0]["selected_skill_ids"] == ["skillx/appworld/spotify-song"]
    assert runs[0]["steps"][0]["positive_skill_ids"] == ["skillx/appworld/spotify-song"]
    assert runs[0]["steps"][0]["qwen_code"] == "song_ids = [1]\nprint('loaded ids')"
    assert "library returned song id 1" in runs[0]["steps"][0]["execute_output"]
    assert runs[0]["steps"][1]["task_completed"] is True
    assert "library returned song id 1" in generator.prompts[1]
    assert "- selected_skill_ids:" not in generator.prompts[1]


def test_multistep_executor_prefers_step_positive_labels_over_task_level_labels(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Find my Spotify song.",
                "required_apps": ["spotify"],
                "positive_skill_ids": ["skillx/appworld/task-level-wrong"],
                "step_positive_skill_ids": [
                    ["skillx/appworld/spotify-song"],
                    ["skillx/appworld/spotify-song"],
                ],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-song",
                "name": "spotify inspect song",
                "description": "Inspect Spotify songs.",
                "executor_desc": "spotify.show_song_library spotify.show_song",
                "body": "Find song ids and inspect them.",
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, [{"query_id": "task_1", "ranked_skill_ids": ["skillx/appworld/spotify-song"]}])

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_step_labels",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="skillrouter_multistep",
        controller=StaticSkillProviderStepController.from_prediction_file(
            skill_pool_path=skill_pool_path,
            predictions_path=predictions_path,
        ),
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=3,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert report["step_positive_skill_hit_rate"] == 1.0
    assert report["step_positive_labeled_count"] == 2
    assert runs[0]["steps"][0]["positive_skill_ids"] == ["skillx/appworld/spotify-song"]


def test_multistep_executor_stops_after_task_completed_even_if_evaluation_fails(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Send the requested payment.",
                "required_apps": ["phone", "venmo"],
                "positive_skill_ids": [],
            }
        ],
    )
    _write_jsonl(skill_pool_path, [])

    class _CompleteButIncorrectGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            return "apis.supervisor.complete_task(status='success')"

    class _CompleteButIncorrectWorld:
        def __init__(self, task_id, experiment_name, **kwargs):
            self.executed: list[str] = []
            self.closed = False

        def execute(self, code):
            self.executed.append(code)
            return "Execution successful."

        def task_completed(self):
            return bool(self.executed)

        def evaluate(self, suppress_errors=True):
            return {"success": False, "suppress_errors": suppress_errors, "failures": ["wrong state"]}

        def close(self):
            self.closed = True

    generator = _CompleteButIncorrectGenerator()
    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_incorrect_stop",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only_multistep",
        controller=StaticSkillProviderStepController(NullSkillProvider()),
        generator=generator,
        world_factory=_CompleteButIncorrectWorld,
        max_tasks=1,
        max_steps=3,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert generator.calls == 1
    assert len(runs[0]["steps"]) == 1
    assert runs[0]["task_completed"] is True
    assert runs[0]["evaluation_success"] is False
    assert runs[0]["success"] is False


def test_multistep_executor_repairs_invalid_api_before_execution(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Find my Spotify song.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.show_song"],
                "positive_skill_ids": [],
            }
        ],
    )
    _write_jsonl(skill_pool_path, [])

    class _RepairGenerator:
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                assert "[Preflight Error]" not in prompt
                return "apis.spotify.categorize_files_by_creation_date()"
            assert "[Preflight Error]" in prompt
            assert "apis.spotify.categorize_files_by_creation_date" in prompt
            return "apis.supervisor.complete_task(answer='Song A', status='success')"

    class _RepairWorld:
        def __init__(self, task_id, experiment_name, **kwargs):
            self.executed: list[str] = []
            self.closed = False

        def execute(self, code):
            self.executed.append(code)
            assert "categorize_files_by_creation_date" not in code
            return "Execution successful."

        def task_completed(self):
            return bool(self.executed)

        def evaluate(self, suppress_errors=True):
            return {"success": bool(self.executed), "suppress_errors": suppress_errors}

        def close(self):
            self.closed = True

    generator = _RepairGenerator()
    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_preflight_repair",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only_multistep",
        controller=StaticSkillProviderStepController(NullSkillProvider()),
        generator=generator,
        world_factory=_RepairWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert len(generator.prompts) == 2
    assert report["success_count"] == 1
    assert report["preflight_repair_count"] == 1
    assert runs[0]["steps"][0]["preflight_repair_count"] == 1
    assert runs[0]["steps"][0]["preflight_ok"] is True
    assert "complete_task" in runs[0]["steps"][0]["code"]


def test_multistep_executor_sanitizes_literal_id_filter_before_preflight(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Add songs from Aria Sterling with play_count over 990 to my Spotify queue.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.search_songs", "spotify.add_to_queue"],
                "positive_skill_ids": [],
            }
        ],
    )
    _write_jsonl(skill_pool_path, [])

    class _LiteralIdGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            assert "[Preflight Error]" not in prompt
            return (
                "songs = apis.spotify.search_songs(artist_id=1, min_play_count=990)\n"
                "for song in songs:\n"
                "    if any(artist['name'] == 'Aria Sterling' for artist in song['artists']):\n"
                "        apis.spotify.add_to_queue(access_token='token', song_id=song['song_id'])\n"
                "apis.supervisor.complete_task(status='success')"
            )

    class _SanitizerWorld:
        def __init__(self, task_id, experiment_name, **kwargs):
            self.executed: list[str] = []

        def execute(self, code):
            self.executed.append(code)
            assert "artist_id" not in code
            assert "min_play_count=990" in code
            assert "add_to_queue" in code
            return "Execution successful."

        def task_completed(self):
            return bool(self.executed)

        def evaluate(self, suppress_errors=True):
            return {"success": bool(self.executed), "suppress_errors": suppress_errors}

        def close(self):
            return None

    generator = _LiteralIdGenerator()
    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_preflight_sanitizer",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only_multistep",
        controller=StaticSkillProviderStepController(NullSkillProvider()),
        generator=generator,
        world_factory=_SanitizerWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    step = runs[0]["steps"][0]
    assert generator.calls == 1
    assert report["success_count"] == 1
    assert report["preflight_repair_count"] == 0
    assert step["preflight_ok"] is True
    assert step["code_sanitizer"]["changed"] is True
    assert "artist_id=1" in step["qwen_code"]
    assert "artist_id" not in step["code"]


def test_multistep_preflight_uses_full_schema_not_truncated_prompt_docs(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Inspect my Spotify song.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.show_song_library"],
                "positive_skill_ids": [],
            }
        ],
    )
    _write_jsonl(skill_pool_path, [])

    class _ShowSongGenerator:
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            assert "[Preflight Error]" not in prompt
            assert "apis.spotify.show_song_library" in prompt
            assert "- apis.spotify.show_song(" not in prompt
            return "song = apis.spotify.show_song(song_id=1)\napis.supervisor.complete_task(answer=song['title'], status='success')"

    class _ShowSongWorld:
        def __init__(self, task_id, experiment_name, **kwargs):
            self.executed: list[str] = []

        def execute(self, code):
            self.executed.append(code)
            assert "apis.spotify.show_song" in code
            return "Execution successful."

        def task_completed(self):
            return bool(self.executed)

        def evaluate(self, suppress_errors=True):
            return {"success": bool(self.executed)}

        def close(self):
            return None

    generator = _ShowSongGenerator()
    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_full_schema_preflight",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only_multistep",
        controller=StaticSkillProviderStepController(NullSkillProvider()),
        generator=generator,
        world_factory=_ShowSongWorld,
        max_tasks=1,
        max_steps=1,
        top_k=1,
        max_apis_per_app=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert len(generator.prompts) == 1
    assert report["success_count"] == 1
    assert report["preflight_repair_count"] == 0
    assert runs[0]["steps"][0]["preflight_ok"] is True


def test_multistep_executor_passes_next_state_text_to_recurrent_controller(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Find my Spotify song.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.show_song_library", "spotify.show_song"],
                "positive_skill_ids": ["skill-a"],
            }
        ],
    )
    skills = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
        {"skill_id": "skill-c", "name": "C"},
    ]
    _write_jsonl(skill_pool_path, skills)
    model = _FakeMultiStepModel()
    controller = CLSTRMultiStepController(
        model=model,
        skills=skills,
        ranking_mode="policy_head",
        candidate_top_k=3,
        allow_legacy_policy_skill_router=True,
    )

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out_next_state",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="clstr_multistep",
        controller=controller,
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=2,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    assert report["success_count"] == 1
    assert model.observations[0] == "Execution successful.\nlibrary returned song id 1"
    assert "[User Goal]" in model.next_texts[0]
    assert "Find my Spotify song." in model.next_texts[0]
    assert "library returned song id 1" in model.next_texts[0]


def test_multistep_qwen_only_uses_null_controller(tmp_path):
    root = _fake_appworld_root(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [{"task_id": "task_1", "query_id": "task_1", "instruction_text": "Find my Spotify song.", "required_apps": ["spotify"]}],
    )
    _write_jsonl(skill_pool_path, [])

    report = run_appworld_multistep_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_dir=tmp_path / "out",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only_multistep",
        controller=StaticSkillProviderStepController(NullSkillProvider()),
        generator=_TwoStepGenerator(),
        world_factory=_TwoStepWorld,
        max_tasks=1,
        max_steps=2,
        top_k=1,
        skill_context_mode="safe_metadata",
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text(encoding="utf-8").splitlines()]
    assert runs[0]["steps"][0]["selected_skill_ids"] == []


def test_build_multistep_executor_comparison_includes_step_metrics(tmp_path):
    report_a = tmp_path / "qwen" / "report.json"
    report_b = tmp_path / "clstr" / "report.json"
    _write_json(
        report_a,
        {
            "status": "ok",
            "method": "qwen_only",
            "task_count": 3,
            "success_count": 1,
            "success_rate": 0.333333,
            "execution_failures": 2,
            "task_completed_count": 1,
            "evaluate_success_count": 1,
            "average_steps": 2.0,
            "step_positive_skill_hit_rate": None,
            "stop_accuracy": None,
        },
    )
    _write_json(
        report_b,
        {
            "status": "ok",
            "method": "clstr_multistep",
            "task_count": 3,
            "success_count": 2,
            "success_rate": 0.666667,
            "execution_failures": 1,
            "task_completed_count": 2,
            "evaluate_success_count": 2,
            "average_steps": 1.666667,
            "step_positive_skill_hit_rate": 0.75,
            "stop_accuracy": 0.5,
        },
    )

    comparison = build_multistep_executor_comparison(
        [report_a, report_b],
        tmp_path / "comparison",
    )

    assert comparison["status"] == "ok"
    assert comparison["rows"][1]["method"] == "clstr_multistep"
    assert comparison["rows"][1]["average_steps"] == 1.666667
    assert comparison["rows"][1]["step_positive_skill_hit_rate"] == 0.75
    assert comparison["rows"][1]["task_completed_count"] == 2
    assert (tmp_path / "comparison" / "comparison.md").exists()


def test_build_train_multistep_trajectories_filters_non_train_and_failed_runs(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    output_path = tmp_path / "train_trajectories.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(
        runs_path,
        [
            {
                "task_id": "train_success",
                "split": "train",
                "user_goal": "Do train task.",
                "final_success": True,
                "steps": [
                    {
                        "step_idx": 0,
                        "state_text": "goal",
                        "selected_skill_ids": ["skill-a"],
                        "positive_skill_ids": ["skill-a"],
                        "qwen_code": "print('ok')",
                        "execute_output": "Execution successful.",
                        "execution_ok": True,
                        "task_completed": True,
                        "evaluate": {"success": True},
                    }
                ],
            },
            {"task_id": "dev_success", "split": "dev", "user_goal": "Do dev task.", "final_success": True, "steps": []},
            {"task_id": "train_failed", "split": "train", "user_goal": "Do failed task.", "final_success": False, "steps": []},
        ],
    )

    manifest = build_train_multistep_trajectories_from_runs(
        runs_path=runs_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert manifest["trajectory_count"] == 1
    assert manifest["skipped_by_split"] == 1
    assert manifest["skipped_unsuccessful"] == 1
    assert rows[0]["task_id"] == "train_success"
    assert rows[0]["split"] == "train"
    assert rows[0]["supervision_type"] == "train_success_rollout"
    assert rows[0]["steps"][0]["state_text"] == "goal"
    assert rows[0]["leakage_guard"] == "train_split_only"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["trajectory_count"] == 1


def test_build_appworld_oracle_multistep_trajectories_uses_train_api_trace(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    task_dir = appworld_root / "data" / "tasks" / "task_1"
    _write_json(task_dir / "specs.json", {"instruction": "Use foo then bar."})
    _write_json(task_dir / "ground_truth" / "required_apps.json", ["demo"])
    _write_json(
        task_dir / "ground_truth" / "api_calls.json",
        [
            {"method": "get", "url": "/demo/foo", "data": {}},
            {"method": "post", "url": "/demo/bar", "data": {"value": 1}},
        ],
    )
    _write_json(
        appworld_root / "data" / "api_docs" / "standard" / "demo.json",
        {
            "foo": {"app_name": "demo", "api_name": "foo", "method": "GET", "path": "/demo/foo"},
            "bar": {"app_name": "demo", "api_name": "bar", "method": "POST", "path": "/demo/bar"},
        },
    )
    tasks_path = tmp_path / "train_tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "split": "train",
                "instruction_text": "Use foo then bar.",
                "required_apps": ["demo"],
                "positive_skill_ids": ["skill/foo", "skill/bar"],
            },
            {
                "task_id": "task_dev",
                "query_id": "task_dev",
                "split": "dev",
                "instruction_text": "Do not use for train.",
            },
        ],
    )
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "skill/foo",
                "name": "demo foo",
                "description": "Call foo.",
                "body": "apis.demo.foo()",
                "executor_desc": "apis.demo.foo",
            },
            {
                "skill_id": "skill/bar",
                "name": "demo bar",
                "description": "Call bar.",
                "body": "apis.demo.bar(value=1)",
                "executor_desc": "apis.demo.bar",
            },
        ],
    )

    manifest = build_appworld_oracle_multistep_trajectories(
        appworld_root=appworld_root,
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_path=tmp_path / "oracle_trajectories.jsonl",
        manifest_path=tmp_path / "oracle_manifest.json",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "oracle_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["trajectory_count"] == 1
    assert manifest["skipped_by_split"] == 1
    assert rows[0]["task_id"] == "task_1"
    assert rows[0]["split"] == "train"
    assert rows[0]["supervision_type"] == "train_oracle_api_trace"
    assert rows[0]["execution_source"] == "ground_truth_api_calls_not_qwen_execution"
    assert [step["selected_skill_ids"] for step in rows[0]["steps"]] == [["skill/foo"], ["skill/bar"]]
    assert "No previous execution steps." in rows[0]["steps"][0]["state_text"]
    assert "skill/foo" in rows[0]["steps"][1]["state_text"]


def test_build_appworld_eval_step_label_tasks_marks_dev_labels_eval_only(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    task_dir = appworld_root / "data" / "tasks" / "task_dev"
    _write_json(
        task_dir / "ground_truth" / "api_calls.json",
        [
            {"method": "get", "url": "/demo/foo", "data": {}},
            {"method": "post", "url": "/demo/bar", "data": {"value": 1}},
        ],
    )
    _write_json(
        appworld_root / "data" / "api_docs" / "standard" / "demo.json",
        {
            "foo": {"app_name": "demo", "api_name": "foo", "method": "GET", "path": "/demo/foo"},
            "bar": {"app_name": "demo", "api_name": "bar", "method": "POST", "path": "/demo/bar"},
        },
    )
    tasks_path = tmp_path / "dev_tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_train",
                "query_id": "task_train",
                "split": "train",
                "instruction_text": "Do not export as eval-only dev labels.",
            },
            {
                "task_id": "task_dev",
                "query_id": "task_dev",
                "split": "dev",
                "instruction_text": "Use foo then bar.",
                "required_apps": ["demo"],
                "positive_skill_ids": ["skill/noisy-task-level"],
            },
        ],
    )
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "checkpoint-base/foo",
                "name": "demo foo exact",
                "description": "Call foo.",
                "body": "apis.demo.foo()",
                "executor_desc": "apis.demo.foo",
                "appworld_executor_compatible": False,
            },
            {
                "skill_id": "skillx/appworld/foo",
                "name": "demo foo",
                "description": "Call foo.",
                "body": "apis.demo.foo()\u2028extra body line",
                "executor_desc": "apis.demo.foo",
                "appworld_executor_compatible": True,
            },
            {
                "skill_id": "skillx/appworld/foo-alt",
                "name": "demo foo alternative",
                "description": "Call foo in another reusable route.",
                "body": "apis.demo.foo()",
                "executor_desc": "apis.demo.foo",
                "appworld_executor_compatible": True,
            },
            {
                "skill_id": "skillx/appworld/bar",
                "name": "demo bar",
                "description": "Call bar.",
                "body": "apis.demo.bar(value=1)",
                "executor_desc": "apis.demo.bar",
                "appworld_executor_compatible": True,
            },
        ],
    )

    manifest = build_appworld_eval_step_label_tasks(
        appworld_root=appworld_root,
        tasks_path=tasks_path,
        skill_pool_path=skill_pool_path,
        output_path=tmp_path / "dev_tasks_with_step_labels.jsonl",
        manifest_path=tmp_path / "dev_step_labels_manifest.json",
        allowed_split="dev",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "dev_tasks_with_step_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["task_count"] == 2
    assert manifest["labeled_task_count"] == 1
    assert manifest["skipped_by_split"] == 1
    assert manifest["eval_only"] is True
    assert manifest["dev_test_used_for_training"] is False
    assert manifest["leakage_guard"] == "dev_eval_only_no_training"
    assert rows[0]["task_id"] == "task_dev"
    assert rows[0]["step_positive_skill_ids"] == [
        ["skillx/appworld/foo", "skillx/appworld/foo-alt"],
        ["skillx/appworld/bar"],
    ]
    assert rows[0]["step_label_source"] == "appworld_dev_ground_truth_api_calls_mapped_to_skillx_eval_only"
    assert rows[0]["leakage_guard"] == "dev_eval_only_no_training"


def test_build_multistep_failure_diagnostics_compares_reference_successes(tmp_path):
    focus_runs = tmp_path / "clstr_runs.jsonl"
    reference_runs = tmp_path / "skillrouter_runs.jsonl"
    _write_jsonl(
        focus_runs,
        [
            {
                "task_id": "task_1",
                "success": False,
                "steps": [
                    {"execution_ok": False, "selected_skill_ids": ["auth-skill"], "execute_output": "Execution failed."}
                ],
            },
            {"task_id": "task_2", "success": True, "steps": []},
        ],
    )
    _write_jsonl(
        reference_runs,
        [
            {"task_id": "task_1", "success": True, "steps": []},
            {"task_id": "task_2", "success": True, "steps": []},
        ],
    )

    report = build_multistep_failure_diagnostics(
        focus_runs_path=focus_runs,
        reference_runs_path=reference_runs,
        output_dir=tmp_path / "diag",
        focus_name="clstr",
        reference_name="skillrouter",
    )

    assert report["reference_success_focus_failure_count"] == 1
    assert report["rows"][0]["task_id"] == "task_1"
    assert report["rows"][0]["focus_execution_failures"] == 1
    assert report["rows"][0]["first_focus_selected_skill_ids"] == ["auth-skill"]
    assert (tmp_path / "diag" / "failure_diagnostics.json").exists()
