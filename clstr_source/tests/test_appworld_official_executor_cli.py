import json

from scripts import run_appworld_official_executor_eval as cli


class _FakeSelection:
    def __init__(self, skills, diagnostics=None):
        self.skills = skills
        self.diagnostics = diagnostics or {}

    @property
    def selected_skill_ids(self):
        return [skill["skill_id"] for skill in self.skills]


class _FakeController:
    def __init__(self):
        self.select_calls = []
        self.observe_calls = []
        self.reset_calls = []

    def reset(self, task):
        self.reset_calls.append(task["task_id"])

    def select(self, *, task, state_text, steps, top_k):
        self.select_calls.append({"task": task, "state_text": state_text, "steps": steps, "top_k": top_k})
        return _FakeSelection(
            [
                {
                    "skill_id": "skillx/appworld/spotify-read-library",
                    "name": "spotify read library",
                    "executor_desc": "Use apis.spotify.show_song_library(access_token=...) to inspect songs.",
                }
            ],
            diagnostics={"controller": "fake"},
        )

    def observe(self, *, selection, code, execute_output, step):
        self.observe_calls.append((selection.selected_skill_ids, code, execute_output, step["step_idx"]))


def test_cli_accepts_qwen3_14b_and_official_react_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            "data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl",
            "--output_dir",
            "outputs/tmp",
            "--model_name_or_path",
            "models/Qwen3-14B",
        ]
    )
    assert args.method == "qwen_only"
    assert args.model_name_or_path == "models/Qwen3-14B"
    assert args.max_interactions == 40
    assert args.completion_precheck_mode == "off"


def test_cli_accepts_clstr_evidence_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "clstr_multistep",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--skill_pool_path",
            "data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl",
            "--base_skill_pool_path",
            "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
            "--clstr_checkpoint_path",
            "outputs/checkpoints/model.pt",
            "--ranking_mode",
            "policy_transition_blend",
            "--candidate_top_k",
            "350",
        ]
    )
    assert args.method == "clstr_multistep"
    assert args.ranking_mode == "policy_transition_blend"
    assert args.candidate_top_k == 350


def test_cli_accepts_task_id_subset_filter():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--task_ids",
            "a,b,c",
        ]
    )
    assert args.task_ids == "a,b,c"


def test_cli_accepts_completion_precheck_mode():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--completion_precheck_mode",
            "constraint_tokens",
        ]
    )
    assert args.completion_precheck_mode == "constraint_tokens"


def test_cli_accepts_current_route_rollouts_path():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--current_route_rollouts_path",
            "out/current_route_rollouts.jsonl",
        ]
    )
    assert args.current_route_rollouts_path == "out/current_route_rollouts.jsonl"


def test_cli_accepts_clstr_state_context():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "clstr_multistep",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--clstr_state_context",
            "api_schema",
        ]
    )
    assert args.clstr_state_context == "api_schema"


def test_cli_accepts_handoff_visible_skill_limit():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--method",
            "clstr_multistep",
            "--tasks_path",
            "tasks.jsonl",
            "--output_dir",
            "out",
            "--handoff_visible_skill_limit",
            "5",
        ]
    )
    assert args.handoff_visible_skill_limit == 5


def test_clstr_evidence_builder_uses_verified_handoff_and_hides_raw_skill_text():
    controller = _FakeController()
    builder = cli.CLSTREvidenceBuilder(
        controller=controller,
        appworld_root="unused",
        top_k=3,
        valid_api_refs_loader=lambda **_kwargs: {
            ("spotify", "show_song_library"),
            ("supervisor", "complete_task"),
        },
    )

    evidence = builder(
        task={"task_id": "task_1", "instruction_text": "Inspect my Spotify library.", "required_apps": ["spotify"]},
        history=[],
    )

    assert "apis.spotify.show_song_library" in evidence.prompt_text
    assert "skillx/appworld" not in evidence.prompt_text.lower()
    assert evidence.diagnostics["selected_skill_ids"] == ["skillx/appworld/spotify-read-library"]
    assert controller.select_calls[0]["top_k"] == 3


def test_clstr_evidence_builder_can_use_legacy_hint_prompt_style():
    controller = _FakeController()
    builder = cli.CLSTREvidenceBuilder(
        controller=controller,
        appworld_root="unused",
        top_k=3,
        valid_api_refs_loader=lambda **_kwargs: {
            ("spotify", "show_song_library"),
            ("supervisor", "complete_task"),
        },
        handoff_prompt_style="legacy_hints",
    )

    evidence = builder(
        task={"task_id": "task_1", "instruction_text": "Inspect my Spotify library.", "required_apps": ["spotify"]},
        history=[],
    )

    assert "[Optional Retrieved Hints]" in evidence.prompt_text
    assert "[Optional Retrieved Evidence]" not in evidence.prompt_text
    assert "Potentially useful APIs:" in evidence.prompt_text
    assert evidence.diagnostics["prompt_style"] == "legacy_hints"


def test_clstr_evidence_builder_passes_visible_skill_limit_to_handoff():
    class ManySkillController(_FakeController):
        def select(self, *, task, state_text, steps, top_k):
            self.select_calls.append({"task": task, "state_text": state_text, "steps": steps, "top_k": top_k})
            return _FakeSelection(
                [
                    {
                        "skill_id": "s1",
                        "name": "show song",
                        "description": "Show songs.",
                        "executor_desc": "apis.spotify.show_song",
                    },
                    {
                        "skill_id": "s2",
                        "name": "search songs",
                        "description": "Search songs.",
                        "executor_desc": "apis.spotify.search_songs",
                    },
                    {
                        "skill_id": "s3",
                        "name": "show albums",
                        "description": "Show albums.",
                        "executor_desc": "apis.spotify.show_album_library",
                    },
                ],
                diagnostics={"controller": "fake"},
            )

    builder = cli.CLSTREvidenceBuilder(
        controller=ManySkillController(),
        appworld_root="unused",
        top_k=20,
        handoff_visible_skill_limit=2,
        valid_api_refs_loader=lambda **_kwargs: {
            ("spotify", "show_song"),
            ("spotify", "search_songs"),
            ("spotify", "show_album_library"),
        },
    )

    evidence = builder(
        task={"task_id": "task_1", "instruction_text": "List Spotify songs.", "required_apps": ["spotify"]},
        history=[],
    )

    assert evidence.diagnostics["selected_skill_ids"] == ["s1", "s2", "s3"]
    assert evidence.diagnostics["visible_skill_ids"] == ["s1", "s2"]
    assert "apis.spotify.show_album_library" not in evidence.prompt_text


def test_clstr_evidence_builder_can_route_with_schema_aware_state():
    controller = _FakeController()
    builder = cli.CLSTREvidenceBuilder(
        controller=controller,
        appworld_root="unused",
        top_k=3,
        valid_api_refs_loader=lambda **_kwargs: {
            ("spotify", "create_playlist"),
            ("supervisor", "complete_task"),
        },
        api_docs_context_builder=lambda **_kwargs: (
            "[Task-Relevant AppWorld API Docs]\n"
            "- apis.spotify.create_playlist(title: string required): Create playlist."
        ),
        clstr_state_context="api_schema",
    )

    evidence = builder(
        task={
            "task_id": "task_1",
            "instruction_text": "Make a Spotify playlist.",
            "required_apps": ["spotify"],
            "api_refs": ["spotify.create_playlist"],
        },
        history=[],
    )

    state_text = controller.select_calls[0]["state_text"]
    assert "[Available API Inventory]" in state_text
    assert "apis.spotify.create_playlist" in state_text
    assert evidence.diagnostics["state_context"] == "api_schema"
    assert evidence.diagnostics["state_api_docs_chars"] > 0


def test_clstr_evidence_builder_passes_controller_diagnostics_to_gate():
    class SuppressedController(_FakeController):
        def select(self, *, task, state_text, steps, top_k):
            self.select_calls.append({"task": task, "state_text": state_text, "steps": steps, "top_k": top_k})
            return _FakeSelection(
                [
                    {
                        "skill_id": "skillx/appworld/spotify-rank-play-count",
                        "name": "spotify rank play count",
                        "description": "Rank songs by play count.",
                        "executor_desc": "apis.spotify.show_song_library",
                    }
                ],
                diagnostics={"selected_scores": [0.2], "transition_scores_available": False},
            )

    builder = cli.CLSTREvidenceBuilder(
        controller=SuppressedController(),
        appworld_root="unused",
        top_k=1,
        valid_api_refs_loader=lambda **_kwargs: {("spotify", "show_song_library")},
    )

    evidence = builder(
        task={
            "task_id": "task_1",
            "instruction_text": "How many unique songs are in my Spotify library?",
            "required_apps": ["spotify"],
        },
        history=[],
    )

    assert evidence.prompt_text == ""
    assert evidence.diagnostics["gate_decision"] == "no_evidence_fallback"
    assert evidence.diagnostics["exact_base_executor_fallback"] is True
    assert evidence.diagnostics["controller"]["selected_scores"] == [0.2]
    assert evidence.diagnostics["evidence_gate"]["selected_scores"] == [0.2]


def test_official_prompt_omits_retrieved_evidence_guard_when_evidence_is_empty():
    prompt = cli.run_official_react_task.__globals__["_build_user_prompt"](
        {"task_id": "task_1", "instruction_text": "Give me the answer.", "required_apps": []},
        [],
        "",
        api_docs_context=(
            "[Task-Relevant AppWorld API Docs]\n"
            "- apis.supervisor.complete_task(answer: string optional): Complete task."
        ),
    )

    assert "[Retrieved Evidence Guard]" not in prompt
    assert "[Optional Retrieved Evidence]" not in prompt
    assert "Write the next Python code block." in prompt


class _FakeGenerator:
    def __init__(self):
        self.user_prompts = []

    def metadata(self):
        return {"backend": "fake"}

    def generate_text(self, system_prompt, user_prompt):
        del system_prompt
        self.user_prompts.append(user_prompt)
        if len(self.user_prompts) == 1:
            return "```python\nprint(apis.api_docs.show_app_descriptions())\n```"
        return "```python\napis.supervisor.complete_task()\n```"


class _FakeWorld:
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.completed = False
        self.closed = False

    def execute(self, code):
        if "complete_task" in code:
            self.completed = True
            return "completed"
        return "apps"

    def task_completed(self):
        return self.completed

    def evaluate(self, suppress_errors=False):
        del suppress_errors
        return {"success": self.completed}

    def close(self):
        self.closed = True


def test_run_official_executor_eval_writes_report_and_uses_instruction_text(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        (
            '{"task_id": "task_1", "query_id": "query_1", '
            '"instruction_text": "Give me the answer.", "required_apps": ["spotify"]}\n'
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    generator = _FakeGenerator()
    worlds = []

    def world_factory(task_id, **kwargs):
        world = _FakeWorld(task_id, **kwargs)
        worlds.append(world)
        return world

    args = cli.build_parser().parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(output_dir),
            "--max_tasks",
            "1",
            "--max_interactions",
            "3",
        ]
    )

    report = cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: generator,
        world_factory=world_factory,
    )

    assert report["status"] == "ok"
    assert report["task_count"] == 1
    assert report["success_count"] == 1
    assert report["average_steps"] == 2.0
    assert (output_dir / "runs.jsonl").exists()
    assert (output_dir / "report.json").exists()
    assert "Give me the answer." in generator.user_prompts[0]
    assert worlds[0].kwargs["max_interactions"] == 3
    assert worlds[0].closed is True
    captured = capsys.readouterr().out
    assert "official_executor_start" in captured
    assert "official_executor_task_done" in captured
    assert "official_executor_world_start" in captured
    assert "official_executor_world_done" in captured
    assert "official_executor_api_docs_done" in captured
    assert "task_1" in captured


def test_run_official_executor_eval_writes_optional_current_route_rollouts(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        (
            '{"task_id": "task_1", "query_id": "query_1", '
            '"instruction_text": "Give me the answer.", "required_apps": ["spotify"]}\n'
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    rollouts_path = output_dir / "current_route_rollouts.jsonl"
    args = cli.build_parser().parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(output_dir),
            "--max_tasks",
            "1",
            "--max_interactions",
            "3",
            "--current_route_rollouts_path",
            str(rollouts_path),
        ]
    )

    report = cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: _FakeGenerator(),
        world_factory=lambda task_id, **kwargs: _FakeWorld(task_id, **kwargs),
    )

    assert report["current_route_rollouts_path"] == str(rollouts_path)
    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "current_route_rollout.v1"
    assert rows[0]["task_id"] == "task_1"
    assert rows[0]["outcome"]["label"] == "success"


def test_official_executor_updates_execution_ok_before_next_clstr_state_text(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        (
            '{"task_id": "task_1", "query_id": "query_1", '
            '"instruction_text": "Give me the answer.", "required_apps": ["spotify"]}\n'
        ),
        encoding="utf-8",
    )
    controller = _FakeController()

    def evidence_builder_factory(_args):
        return cli.CLSTREvidenceBuilder(
            controller=controller,
            appworld_root="unused",
            top_k=1,
            valid_api_refs_loader=lambda **_kwargs: {
                ("spotify", "show_song_library"),
                ("supervisor", "complete_task"),
            },
            state_text_builder=cli._legacy_state_text,
        )

    args = cli.build_parser().parse_args(
        [
            "--method",
            "clstr_multistep",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_tasks",
            "1",
            "--max_interactions",
            "3",
        ]
    )

    cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: _FakeGenerator(),
        world_factory=lambda task_id, **kwargs: _FakeWorld(task_id, **kwargs),
        evidence_builder_factory=evidence_builder_factory,
    )

    assert len(controller.select_calls) == 2
    assert "- execution_ok: True" in controller.select_calls[1]["state_text"]
    assert "- execution_ok: False" not in controller.select_calls[1]["state_text"]


def test_run_official_executor_eval_flushes_runs_after_each_task(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            [
                '{"task_id": "task_1", "instruction_text": "Give me the first answer.", "required_apps": ["spotify"]}',
                '{"task_id": "task_2", "instruction_text": "Give me the second answer.", "required_apps": ["spotify"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    seen_task_ids = []

    def world_factory(task_id, **kwargs):
        del kwargs
        if task_id == "task_2":
            runs_path = output_dir / "runs.jsonl"
            assert runs_path.exists()
            rows = [line for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(rows) == 1
            assert '"task_id": "task_1"' in rows[0]
        seen_task_ids.append(task_id)
        return _FakeWorld(task_id)

    args = cli.build_parser().parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(output_dir),
            "--max_tasks",
            "2",
            "--max_interactions",
            "2",
        ]
    )

    report = cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: _FakeGenerator(),
        world_factory=world_factory,
    )

    assert report["task_count"] == 2
    assert seen_task_ids == ["task_1", "task_2"]


def test_run_official_executor_eval_counts_preflight_blocks_as_failures(tmp_path):
    class MutationGenerator:
        def metadata(self):
            return {"backend": "mutation_fake"}

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\napis.spotify.signup(username='new_user', password='pw', email='x@y.com')\n```"

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        '{"task_id": "task_2", "instruction_text": "Inspect my library.", "required_apps": ["spotify"]}\n',
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_tasks",
            "1",
            "--max_interactions",
            "1",
        ]
    )

    report = cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: MutationGenerator(),
        world_factory=lambda task_id, **kwargs: _FakeWorld(task_id, **kwargs),
    )

    assert report["execution_failures"] == 1
    assert report["success_count"] == 0


def test_run_official_executor_eval_filters_task_ids(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            [
                '{"task_id": "keep_1", "instruction_text": "A", "required_apps": ["spotify"]}',
                '{"task_id": "drop_1", "instruction_text": "B", "required_apps": ["spotify"]}',
                '{"task_id": "keep_2", "instruction_text": "C", "required_apps": ["spotify"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(
        [
            "--method",
            "qwen_only",
            "--tasks_path",
            str(tasks_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--task_ids",
            "keep_2,keep_1",
            "--max_tasks",
            "10",
        ]
    )
    seen = []

    def world_factory(task_id, **kwargs):
        seen.append(task_id)
        return _FakeWorld(task_id, **kwargs)

    report = cli.run_official_executor_eval(
        args,
        generator_factory=lambda _args: _FakeGenerator(),
        world_factory=world_factory,
    )

    assert seen == ["keep_1", "keep_2"]
    assert report["task_count"] == 2
