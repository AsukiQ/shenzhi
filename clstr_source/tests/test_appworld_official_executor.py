from clstr.appworld_official_executor import (
    EvidenceResult,
    OfficialReActExecutorConfig,
    _completion_precheck,
    build_official_clstr_state_text,
    build_official_api_docs_context,
    build_official_react_system_prompt,
    run_official_react_task,
)


def test_official_react_prompt_uses_api_doc_discovery_not_static_api_dump():
    prompt = build_official_react_system_prompt(max_interactions=40)
    assert "apis.api_docs.show_app_descriptions()" in prompt
    assert "apis.api_docs.show_api_descriptions(app_name=" in prompt
    assert "apis.api_docs.show_api_doc(app_name=" in prompt
    assert "multi-step conversation" in prompt
    assert "Do not treat retrieved skill evidence as executable APIs" in prompt
    assert "page_index" in prompt
    assert "process the current page before checking whether its length is smaller than page_limit" in prompt
    assert "inspect every named source" in prompt
    assert "Keep full record dictionaries" in prompt
    assert "Never invent credentials" in prompt
    assert "Do not sign up" in prompt
    assert "complete_task(answer=result)" in prompt


def test_completion_precheck_blocks_subtracting_user_share_when_records_omit_user():
    result = _completion_precheck(
        code="""
total_amount = sum(transaction["amount"] for transaction in dinner_transactions)
answer = total_amount - 38
apis.supervisor.complete_task(answer=answer)
""",
        task={
            "instruction_text": (
                "Everyones' transactions except mine should be on my social feed. "
                "My share was $38. How much did my manager pay for the others, including me?"
            ),
            "required_apps": ["venmo"],
        },
        history=[{"code": "social_feed = apis.venmo.show_social_feed(access_token=token)"}],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "For aggregate answers where records omit the user's item but the final answer includes the user, add the explicitly stated user amount/share to the aggregate; do not subtract it from records that already omit it."
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "violates_inclusion_exclusion_constraint"
    assert "add the stated user amount/share" in result["error"]


def test_completion_precheck_blocks_archive_move_without_prefixed_basename():
    result = _completion_precheck(
        code="""
files = apis.file_system.show_directory(access_token=token, directory_path="~/my_downloads/", entry_type="files")
for file in files:
    file_details = apis.file_system.show_file(file_path=file, access_token=token)
    created_at = file_details["created_at"].split("T")[0]
    if created_at.split("-")[0] == "2023":
        new_file_name = f"{created_at}_{file.split('/')[-1]}"
        apis.file_system.move_file(source_file_path=file, destination_file_path=f"~/my_downloads/{new_file_name}", access_token=token)
    else:
        new_archive_path = f"~/archive/{file.split('/')[-1]}"
        apis.file_system.move_file(source_file_path=file, destination_file_path=new_archive_path, access_token=token)
apis.supervisor.complete_task()
""",
        task={
            "instruction_text": (
                'Add the prefix "YYYY_MM_DD_" to all file names in ~/my_downloads/ based on creation dates, '
                "and then move all files not from this year to ~/archive/."
            ),
            "required_apps": ["file_system"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "When a file must be both renamed/prefixed and moved, combine the final directory with the renamed/prefixed basename in one destination path."
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "violates_file_move_prefix_constraint"
    assert "prefixed basename" in result["error"]


def test_completion_precheck_expands_archive_destination_helper_variables():
    result = _completion_precheck(
        code="""
for file in files:
    file_details = apis.file_system.show_file(file_path=file, access_token=token)
    created_at = file_details["created_at"].split("T")[0]
    archive_directory = "~/archive/"
    original_file_name = file.split("/")[-1]
    new_archive_file_path = f"{archive_directory}{original_file_name}"
    apis.file_system.move_file(source_file_path=file, destination_file_path=new_archive_file_path, access_token=token)
apis.supervisor.complete_task()
""",
        task={
            "instruction_text": (
                'Add the prefix "YYYY_MM_DD_" to all file names in ~/my_downloads/ based on creation dates, '
                "and then move all files not from this year to ~/archive/."
            ),
            "required_apps": ["file_system"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "When a file must be both renamed/prefixed and moved, combine the final directory with the renamed/prefixed basename in one destination path."
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "violates_file_move_prefix_constraint"


def test_completion_precheck_blocks_hyphenated_date_prefix_when_goal_requires_underscore_format():
    result = _completion_precheck(
        code="""
file = "~/my_downloads/report.txt"
file_details = apis.file_system.show_file(file_path=file, access_token=token)
created_at = file_details["created_at"].split("T")[0]
new_basename = f"{created_at}_{file.split('/')[-1]}"
apis.file_system.move_file(source_file_path=file, destination_file_path=f"~/archive/{new_basename}", access_token=token)
apis.supervisor.complete_task()
""",
        task={
            "instruction_text": (
                'Add the prefix "YYYY_MM_DD_" to all file names in ~/my_downloads/ '
                "based on their creation dates, and then move old files to ~/archive/."
            ),
            "required_apps": ["file_system"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "When adding a quoted date prefix like YYYY_MM_DD_, format the date with underscores before concatenating it with the basename."
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "violates_date_prefix_format_constraint"


def test_completion_precheck_allows_underscore_date_prefix_reformatting():
    result = _completion_precheck(
        code="""
file = "~/my_downloads/report.txt"
file_details = apis.file_system.show_file(file_path=file, access_token=token)
created_at = file_details["created_at"].split("T")[0]
date_prefix = created_at.replace("-", "_")
new_basename = f"{date_prefix}_{file.split('/')[-1]}"
apis.file_system.move_file(source_file_path=file, destination_file_path=f"~/archive/{new_basename}", access_token=token)
apis.supervisor.complete_task()
""",
        task={
            "instruction_text": (
                'Add the prefix "YYYY_MM_DD_" to all file names in ~/my_downloads/ '
                "based on their creation dates, and then move old files to ~/archive/."
            ),
            "required_apps": ["file_system"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "When adding a quoted date prefix like YYYY_MM_DD_, format the date with underscores before concatenating it with the basename."
            ]
        },
    )

    assert result["ok"] is True


def test_completion_precheck_blocks_state_change_completion_on_missing_write_fallback_path():
    result = _completion_precheck(
        code="""
amount = None
for message in messages:
    if "grocery" in message["message"].lower():
        amount = float(message["message"].split("$")[1])
        break
if amount is not None:
    apis.venmo.create_transaction(receiver_email=friend_email, amount=amount, access_token=venmo_access_token)
    apis.phone.send_text_message(phone_number=friend_phone, message="Done.", access_token=phone_access_token)
    apis.supervisor.complete_task()
else:
    apis.supervisor.complete_task()
""",
        task={
            "instruction_text": "Send my friend the money I owe from our phone conversation and text them Done.",
            "required_apps": ["phone", "venmo"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "For payment tasks based on a phone conversation, read the conversation before sending money; derive the amount and recipient from API records, not guesses.",
                "After the payment action, send the requested phone text message to the same resolved contact.",
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "state_change_completion_has_missing_write_fallback"


def test_completion_precheck_blocks_state_change_completion_after_guarded_write_without_success_gate():
    result = _completion_precheck(
        code="""
amount = None
if amount is not None and friend_email is not None:
    apis.venmo.create_transaction(receiver_email=friend_email, amount=amount, access_token=venmo_access_token)
apis.phone.send_text_message(phone_number=friend_phone, message="Done.", access_token=phone_access_token)
apis.supervisor.complete_task()
""",
        task={
            "instruction_text": "Pay my friend what I owe from the text conversation and send them a text.",
            "required_apps": ["phone", "venmo"],
        },
        history=[],
        mode="constraint_tokens",
        evidence_diagnostics={
            "constraint_hints": [
                "For payment tasks based on a phone conversation, read the conversation before sending money; derive the amount and recipient from API records, not guesses.",
                "After the payment action, send the requested phone text message to the same resolved contact.",
            ]
        },
    )

    assert result["ok"] is False
    assert result["reason"] == "state_change_completion_after_guarded_write"


def test_official_api_docs_context_prioritizes_task_relevant_library_apis(tmp_path):
    docs_dir = tmp_path / "data" / "api_docs" / "standard"
    docs_dir.mkdir(parents=True)
    (docs_dir / "spotify.json").write_text(
        """
{
  "signup": {"description": "Sign up to create account.", "parameters": [], "response_schemas": {"success": {}}},
  "search_songs": {"description": "Search for songs with a query.", "parameters": [{"name": "query", "type": "string", "required": false}], "response_schemas": {"success": [{"title": "Song"}]}},
  "show_song_library": {"description": "Get a list of songs in the user's song library.", "parameters": [{"name": "access_token", "type": "string", "required": true}, {"name": "page_limit", "type": "integer", "required": false, "constraints": ["value <= 20"]}], "response_schemas": {"success": [{"song_id": 1, "title": "Song"}]}},
  "show_album_library": {"description": "Get a list of albums in the user's album library.", "parameters": [{"name": "access_token", "type": "string", "required": true}], "response_schemas": {"success": [{"album_id": 2}]}},
  "show_playlist_library": {"description": "Get a list of playlists in the user's playlist library.", "parameters": [{"name": "access_token", "type": "string", "required": true}], "response_schemas": {"success": [{"playlist_id": 3}]}}
}
""",
        encoding="utf-8",
    )

    context = build_official_api_docs_context(
        appworld_root=tmp_path,
        task={
            "instruction_text": "Give me top played r&b song titles across my Spotify song, album and playlist libraries.",
            "required_apps": ["spotify"],
        },
        max_apis_per_app=3,
    )

    assert "apis.spotify.show_song_library" in context
    assert "constraints value <= 20" in context
    assert "apis.spotify.show_album_library" in context
    assert "apis.spotify.show_playlist_library" in context
    assert "apis.spotify.signup" not in context


def test_official_api_docs_context_ignores_eval_only_api_refs_by_default(tmp_path):
    docs_dir = tmp_path / "data" / "api_docs" / "standard"
    docs_dir.mkdir(parents=True)
    (docs_dir / "spotify.json").write_text(
        """
{
  "create_playlist": {"description": "Create a new playlist.", "parameters": [], "response_schemas": {"success": {"playlist_id": 1}}},
  "show_song_library": {"description": "Get songs from the user's song library.", "parameters": [], "response_schemas": {"success": [{"song_id": 1}]}}
}
""",
        encoding="utf-8",
    )

    context = build_official_api_docs_context(
        appworld_root=tmp_path,
        task={
            "instruction_text": "Inspect my Spotify songs.",
            "required_apps": ["spotify"],
            "api_refs": ["spotify.create_playlist"],
            "step_label_eval_only": True,
        },
        max_apis_per_app=1,
    )

    assert "apis.spotify.show_song_library" in context
    assert "apis.spotify.create_playlist" not in context


def test_official_api_docs_context_promotes_goal_matched_write_api_without_solution_refs(tmp_path):
    docs_dir = tmp_path / "data" / "api_docs" / "standard"
    docs_dir.mkdir(parents=True)
    (docs_dir / "spotify.json").write_text(
        """
{
  "create_playlist": {"description": "Create a new playlist.", "parameters": [{"name": "title", "type": "string", "required": true}], "response_schemas": {"success": {"playlist_id": 1}}},
  "search_songs": {"description": "Search songs by play count in playlists with title filters.", "parameters": [{"name": "query", "type": "string", "required": false}], "response_schemas": {"success": [{"song_id": 1}]}},
  "show_playlist_library": {"description": "Get a list of playlists in the user's playlist library with songs and title metadata.", "parameters": [], "response_schemas": {"success": [{"playlist_id": 1}]}},
  "show_playlist": {"description": "Get detailed playlist songs, title, play count, and playlist metadata.", "parameters": [], "response_schemas": {"success": {"playlist_id": 1}}},
  "search_playlists": {"description": "Search playlists by playlist title, songs, play count and owner.", "parameters": [], "response_schemas": {"success": [{"playlist_id": 1}]}}
}
""",
        encoding="utf-8",
    )

    context = build_official_api_docs_context(
        appworld_root=tmp_path,
        task={
            "instruction_text": "Make a Spotify playlist called Mix containing songs by play count.",
            "required_apps": ["spotify"],
            "api_refs": ["spotify.create_playlist"],
            "step_label_eval_only": True,
        },
        max_apis_per_app=3,
    )

    assert "apis.spotify.create_playlist" in context


def test_official_clstr_state_text_can_include_schema_inventory_without_skill_labels():
    state_text = build_official_clstr_state_text(
        task={
            "task_id": "task_1",
            "instruction_text": "Make a Spotify playlist from my most played songs.",
            "required_apps": ["spotify"],
            "api_refs": ["spotify.create_playlist"],
        },
        history=[],
        api_docs_context=(
            "[Task-Relevant AppWorld API Docs]\n"
            "- apis.spotify.create_playlist(title: string required): Create playlist.\n"
        ),
    )

    assert "[User Goal]" in state_text
    assert "Required apps: spotify" in state_text
    assert "[Available API Inventory]" in state_text
    assert "apis.spotify.create_playlist" in state_text
    assert "spotify.create_playlist" not in state_text.split("[Available API Inventory]", 1)[0]


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate_text(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls == 1:
            return "```python\nprint(apis.api_docs.show_app_descriptions())\n```"
        return "```python\napis.supervisor.complete_task()\n```"


class FakeWorld:
    def __init__(self):
        self.outputs = []
        self.completed = False

    def execute(self, code):
        self.outputs.append(code)
        if "complete_task" in code:
            self.completed = True
            return "completed"
        return "[{'name': 'spotify'}]"

    def task_completed(self):
        return self.completed

    def evaluate(self):
        class Report:
            def report(self):
                return {"success": True}

        return Report()


def test_official_react_loop_persists_history_and_stops_on_completion():
    row = run_official_react_task(
        task={"task_id": "fake_1", "instruction": "finish it", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=4),
    )
    assert row["task_completed"] is True
    assert row["evaluation_success"] is True
    assert row["steps"][0]["execute_output"] == "[{'name': 'spotify'}]"
    assert "Output:" in row["steps"][1]["user_prompt"]
    assert "password_by_app" in row["steps"][0]["user_prompt"]
    assert "returns a list" in row["steps"][0]["user_prompt"]


def test_official_react_loop_emits_step_progress_events():
    events = []

    row = run_official_react_task(
        task={"task_id": "fake_progress", "instruction": "finish it", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=2),
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert row["success"] is True
    assert [event for event, _ in events] == [
        "official_executor_step_prepare_start",
        "official_executor_step_evidence_done",
        "official_executor_step_start",
        "official_executor_step_done",
        "official_executor_step_prepare_start",
        "official_executor_step_evidence_done",
        "official_executor_step_start",
        "official_executor_step_done",
    ]
    assert events[0][1]["task_id"] == "fake_progress"
    assert events[0][1]["step_idx"] == 0
    assert events[-1][1]["task_completed"] is True
    assert events[-1][1]["evaluation_success"] is True


def test_official_react_prompt_explains_task_datetime_for_relative_dates():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task()\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_datetime",
            "instruction": "Like all transactions from the ongoing year.",
            "required_apps": ["venmo"],
            "task_datetime": "2023-05-18T12:00:00",
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    prompt = generator.prompts[0]
    assert "Task datetime: 2023-05-18T12:00:00" in prompt
    assert "relative-date" in prompt
    assert 'task_datetime = "2023-05-18T12:00:00"' in prompt
    assert "split('T')[0]" in prompt
    assert "YYYY-MM-DD" in prompt
    assert "ongoing/current year starts on YYYY-01-01" in prompt
    assert "do not append time" in prompt


def test_official_react_prompt_guides_cross_app_identity_mapping():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task()\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_cross_app",
            "instruction": "Like all venmo transactions to and from my coworkers.",
            "required_apps": ["phone", "venmo"],
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    prompt = generator.prompts[0]
    assert "relationship=" in prompt
    assert "contact email" in prompt
    assert "not phone_number" in prompt


def test_official_react_prompt_adds_spotify_library_schema_guard():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task()\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_spotify_scope",
            "instruction": "Give me the top 6 most played edm songs across my Spotify song, album and playlist libraries.",
            "required_apps": ["spotify"],
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    prompt = generator.prompts[0]
    assert "Spotify Library Schema Guard" in prompt
    assert "song library rows use `song_id`" in prompt
    assert "album and playlist rows use `song_ids`" in prompt
    assert "show_playlist(...)[\"songs\"] rows use `id`; call `show_song(song_id=row[\"id\"])` before reading `play_count`" in prompt
    assert "filter by genre before ranking by play_count" in prompt
    assert "do not set `is_public=True`" in prompt
    assert "omit `is_public`" in prompt


def test_official_react_loop_reads_appworld_testtracker_success_property():
    class TrackerWorld(FakeWorld):
        def evaluate(self, suppress_errors=False):
            del suppress_errors

            class Tracker:
                success = True

                def report(self):
                    return "Num Passed Tests : 2\nNum Failed Tests : 0\nNum Total  Tests : 2"

            return Tracker()

    row = run_official_react_task(
        task={"task_id": "fake_tracker", "instruction": "finish it", "required_apps": ["spotify"]},
        world=TrackerWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=2),
    )

    assert row["task_completed"] is True
    assert row["evaluation_success"] is True
    assert row["success"] is True


def test_official_react_loop_continues_after_wrong_completion_with_verifier_feedback():
    class VerifierFeedbackGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return "```python\napis.supervisor.complete_task(answer='wrong')\n```"
            assert "official evaluator did not pass" in user_prompt
            assert "missing required genre filter" in user_prompt
            assert "SECRET_EXPECTED_ANSWER" not in user_prompt
            return "```python\napis.supervisor.complete_task(answer='right')\n```"

    class WrongThenRightWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.answer = ""

        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
            self.answer = "right" if "right" in code else "wrong"
            return "Execution successful."

        def evaluate(self):
            answer = self.answer

            class Report:
                success = answer == "right"

                def to_dict(self, stats_only=False):
                    if stats_only:
                        return {"success": self.success, "num_tests": 1}
                    if self.success:
                        return {"success": True, "num_tests": 1, "passes": [], "failures": []}
                    return {
                        "success": False,
                        "num_tests": 1,
                        "passes": [],
                        "failures": [
                            {
                                "requirement": "missing required genre filter",
                                "trace": "expected SECRET_EXPECTED_ANSWER but got wrong answer",
                                "label": "no_op_fail",
                            }
                        ],
                    }

            return Report()

    generator = VerifierFeedbackGenerator()
    row = run_official_react_task(
        task={"task_id": "fake_wrong_completion", "instruction": "Give me filtered songs.", "required_apps": []},
        world=WrongThenRightWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=3, max_wrong_completion_retries=1),
    )

    assert row["success"] is True
    assert row["evaluation_success"] is True
    assert len(row["steps"]) == 2
    assert row["steps"][0]["task_completed"] is True
    assert row["steps"][0]["evaluation_success"] is False
    assert row["steps"][1]["evaluation_success"] is True


def test_official_react_loop_caps_wrong_completion_retries():
    class AlwaysWrongGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\napis.supervisor.complete_task(answer='still wrong')\n```"

    class AlwaysWrongWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            self.completed = True
            return "Execution successful."

        def evaluate(self):
            class Report:
                success = False

                def report(self):
                    return "Num Passed Tests : 0\nNum Failed Tests : 1\nwrong answer"

            return Report()

    row = run_official_react_task(
        task={"task_id": "fake_retry_cap", "instruction": "Give me the right answer.", "required_apps": []},
        world=AlwaysWrongWorld(),
        generator=AlwaysWrongGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=5, max_wrong_completion_retries=1),
    )

    assert row["success"] is False
    assert len(row["steps"]) == 2
    assert row["steps"][-1]["stop_reason"] == "wrong_completion_retry_limit"


def test_official_react_loop_does_not_retry_wrong_completion_by_default():
    class AlwaysWrongGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\napis.supervisor.complete_task(answer='wrong')\n```"

    class AlwaysWrongWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            self.completed = True
            return "Execution successful."

        def evaluate(self):
            class Report:
                success = False

                def report(self):
                    return "Num Passed Tests : 0\nNum Failed Tests : 1\nwrong answer"

            return Report()

    row = run_official_react_task(
        task={"task_id": "fake_no_retry_default", "instruction": "Give me the right answer.", "required_apps": []},
        world=AlwaysWrongWorld(),
        generator=AlwaysWrongGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=5),
    )

    assert row["success"] is False
    assert len(row["steps"]) == 1
    assert row["steps"][-1]["stop_reason"] == "wrong_completion_retry_limit"


def test_official_react_loop_completion_precheck_blocks_answer_missing_task_constraint():
    class ConstraintRepairGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return (
                    "```python\n"
                    "songs = apis.spotify.show_song_library(access_token='token')\n"
                    "top_songs = sorted(songs, key=lambda row: row['play_count'], reverse=True)[:6]\n"
                    "apis.supervisor.complete_task(answer=','.join(row['title'] for row in top_songs))\n"
                    "```"
                )
            assert "Completion precheck blocked" in user_prompt
            assert "edm" in user_prompt.lower()
            return (
                "```python\n"
                "songs = apis.spotify.show_song_library(access_token='token')\n"
                "top_songs = [row for row in songs if row['genre'] == 'EDM'][:6]\n"
                "apis.supervisor.complete_task(answer=','.join(row['title'] for row in top_songs))\n"
                "```"
            )

    class ConstraintWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "[{'title': 'A', 'genre': 'EDM', 'play_count': 5}]"

    world = ConstraintWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_edm_answer",
            "instruction": "Give me a comma-separated list of top 6 most played edm song titles.",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=ConstraintRepairGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=2,
            completion_precheck_mode="constraint_tokens",
            valid_api_refs={
                ("spotify", "show_song_library"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert len(row["steps"]) == 2
    assert row["steps"][0]["completion_precheck"]["ok"] is False
    assert row["steps"][0]["completion_precheck"]["reason"] == "missing_answer_constraint_tokens"
    assert world.outputs == [row["steps"][1]["code"]]


def test_official_react_loop_repairs_completion_precheck_before_execution():
    class PrecheckRepairGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return (
                    "```python\n"
                    "file = '~/my_downloads/report.txt'\n"
                    "new_file_name = '2023_01_01_' + file.split('/')[-1]\n"
                    "apis.file_system.move_file(source_file_path=file, destination_file_path='~/archive/' + file.split('/')[-1], access_token='token')\n"
                    "apis.supervisor.complete_task()\n"
                    "```"
                )
            assert "[Code Repair Feedback]" in user_prompt
            assert "violates_file_move_prefix_constraint" in user_prompt
            assert "destination_file_path='~/archive/' + file.split('/')[-1]" in user_prompt
            return (
                "```python\n"
                "file = '~/my_downloads/report.txt'\n"
                "new_file_name = '2023_01_01_' + file.split('/')[-1]\n"
                "apis.file_system.move_file(source_file_path=file, destination_file_path='~/archive/' + new_file_name, access_token='token')\n"
                "apis.supervisor.complete_task()\n"
                "```"
            )

    class RepairWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            assert "destination_file_path='~/archive/' + file.split('/')[-1]" not in code
            self.completed = True
            return "Execution successful."

    def evidence_builder(task, history):
        del task, history
        return EvidenceResult(
            prompt_text="[Evidence]\n- apis.file_system.move_file",
            diagnostics={
                "constraint_hints": [
                    "When files are renamed/prefixed and then moved, move the renamed/prefixed basename into the target directory; never archive the original basename."
                ]
            },
        )

    world = RepairWorld()
    generator = PrecheckRepairGenerator()
    row = run_official_react_task(
        task={
            "task_id": "fake_precheck_repair",
            "instruction": "Prefix files by creation date, then archive old files.",
            "required_apps": ["file_system"],
        },
        world=world,
        generator=generator,
        evidence_builder=evidence_builder,
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            completion_precheck_mode="constraint_tokens",
            preflight_max_repairs=1,
            valid_api_refs={
                ("file_system", "move_file"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert len(generator.prompts) == 2
    assert len(world.outputs) == 1
    assert row["steps"][0]["completion_precheck"]["ok"] is True
    assert row["steps"][0]["repair_count"] == 1
    assert row["steps"][0]["repair_errors"][0]["stage"] == "completion_precheck"
    assert "new_file_name" in row["steps"][0]["code"]


def test_official_react_loop_completion_precheck_blocks_missing_indie_genre_constraint():
    class IndieConstraintGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return (
                    "```python\n"
                    "songs = apis.spotify.show_song_library(access_token='token')\n"
                    "# Extract top 3 most played indie song titles.\n"
                    "top_songs = sorted(songs, key=lambda row: row['play_count'], reverse=True)[:3]\n"
                    "apis.supervisor.complete_task(answer=','.join(row['title'] for row in top_songs))\n"
                    "```"
                )
            assert "Completion precheck blocked" in user_prompt
            assert "indie" in user_prompt.lower()
            return (
                "```python\n"
                "songs = apis.spotify.show_song_library(access_token='token')\n"
                "top_songs = [row for row in songs if row['genre'].lower() == 'indie'][:3]\n"
                "apis.supervisor.complete_task(answer=','.join(row['title'] for row in top_songs))\n"
                "```"
            )

    class ConstraintWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "[{'title': 'A', 'genre': 'Indie', 'play_count': 5}]"

    world = ConstraintWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_indie_answer",
            "instruction": "Give me a comma-separated list of top 3 most played indie song titles.",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=IndieConstraintGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=2,
            completion_precheck_mode="constraint_tokens",
            valid_api_refs={
                ("spotify", "show_song_library"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert len(row["steps"]) == 2
    assert row["steps"][0]["completion_precheck"]["ok"] is False
    assert row["steps"][0]["completion_precheck"]["missing_constraint_tokens"] == ["indie"]


def test_completion_precheck_blocks_cross_library_genre_filter_only_on_song_library():
    class CrossLibraryGenreGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return (
                    "```python\n"
                    "song_library = apis.spotify.show_song_library(access_token='token')\n"
                    "album_library = apis.spotify.show_album_library(access_token='token')\n"
                    "playlist_library = apis.spotify.show_playlist_library(access_token='token')\n"
                    "all_song_ids = set()\n"
                    "for album in album_library:\n"
                    "    all_song_ids.update(album.get('song_ids', []))\n"
                    "for playlist in playlist_library:\n"
                    "    all_song_ids.update(playlist.get('song_ids', []))\n"
                    "indie_songs = []\n"
                    "for song in song_library:\n"
                    "    if song.get('genre') == 'indie':\n"
                    "        indie_songs.append(song)\n"
                    "apis.supervisor.complete_task(answer=','.join(row['title'] for row in indie_songs[:3]))\n"
                    "```"
                )
            assert "combined song id set" in user_prompt.lower()
            return (
                "```python\n"
                "all_song_ids = {1, 2, 3}\n"
                "indie_songs = []\n"
                "for song_id in all_song_ids:\n"
                "    song = apis.spotify.show_song(song_id=song_id)\n"
                "    if song.get('genre') == 'indie':\n"
                "        indie_songs.append(song)\n"
                "apis.supervisor.complete_task(answer=','.join(row['title'] for row in indie_songs[:3]))\n"
                "```"
            )

    class ConstraintWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "[]"

    row = run_official_react_task(
        task={
            "task_id": "fake_cross_library_indie",
            "instruction": (
                "Give me a comma-separated list of top 3 most played indie song titles "
                "from across my Spotify song, album and playlist libraries."
            ),
            "required_apps": ["spotify"],
        },
        world=ConstraintWorld(),
        generator=CrossLibraryGenreGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=2,
            completion_precheck_mode="constraint_tokens",
            valid_api_refs={
                ("spotify", "show_song_library"),
                ("spotify", "show_album_library"),
                ("spotify", "show_playlist_library"),
                ("spotify", "show_song"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert len(row["steps"]) == 2
    assert row["steps"][0]["completion_precheck"]["ok"] is False
    assert row["steps"][0]["completion_precheck"]["reason"] == "cross_library_genre_filter_not_over_union"


def test_official_react_loop_completion_precheck_is_off_by_default():
    class MissingConstraintGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "songs = apis.spotify.show_song_library(access_token='token')\n"
                "apis.supervisor.complete_task(answer=','.join(row['title'] for row in songs[:6]))\n"
                "```"
            )

    world = FakeWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_default_no_precheck",
            "instruction": "Give me a comma-separated list of top 6 most played edm song titles.",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=MissingConstraintGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            valid_api_refs={
                ("spotify", "show_song_library"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["task_completed"] is True
    assert row["steps"][0]["completion_precheck"]["reason"] == "disabled"


def test_official_react_completion_precheck_allows_answer_when_high_signal_constraints_are_preserved():
    class VenmoAnswerGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "transactions = apis.venmo.show_feed(access_token='token')\n"
                "venue = 'Azure Harbor Bistro'\n"
                "my_share = 38\n"
                "matching = [row for row in transactions if venue in row['note']]\n"
                "total_amount_paid_by_manager = sum(row['amount'] for row in matching) + my_share\n"
                "apis.supervisor.complete_task(answer=total_amount_paid_by_manager)\n"
                "```"
            )

    class VenmoWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "[]"

    row = run_official_react_task(
        task={
            "task_id": "fake_venmo_answer",
            "instruction": (
                "I went on dinner with my coworkers yesterday at Azure Harbor Bistro. "
                "My manager paid for food and everyone venmoed them. Everyones' transactions except mine "
                "should be on my social feed. My share was $38. How much did my manager pay for the others, "
                "including me, yesterday?"
            ),
            "required_apps": ["venmo"],
        },
        world=VenmoWorld(),
        generator=VenmoAnswerGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            completion_precheck_mode="constraint_tokens",
            valid_api_refs={
                ("venmo", "show_feed"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert row["steps"][0]["completion_precheck"]["ok"] is True


def test_official_react_completion_precheck_does_not_require_literal_amount_answer_token():
    class VenmoComputedAnswerGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "transactions = apis.venmo.show_feed(access_token='token')\n"
                "venue = 'Azure Harbor Bistro'\n"
                "matching = [row for row in transactions if venue in row['note']]\n"
                "manager_total = sum(row['amount'] for row in matching)\n"
                "my_share = matching[0]['amount']\n"
                "apis.supervisor.complete_task(answer=manager_total + my_share)\n"
                "```"
            )

    class VenmoWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "[]"

    row = run_official_react_task(
        task={
            "task_id": "fake_venmo_computed_amount_answer",
            "instruction": (
                "I went on dinner with my coworkers yesterday at Azure Harbor Bistro. "
                "My manager paid for food and everyone venmoed them. Everyones' transactions except mine "
                "should be on my social feed. My share was $38. How much did my manager pay for the others, "
                "including me, yesterday?"
            ),
            "required_apps": ["venmo"],
        },
        world=VenmoWorld(),
        generator=VenmoComputedAnswerGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            completion_precheck_mode="constraint_tokens",
            valid_api_refs={
                ("venmo", "show_feed"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["success"] is True
    assert row["steps"][0]["completion_precheck"]["ok"] is True


def test_official_react_loop_compacts_large_history_outputs():
    class VerboseWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            return "x" * 2000

    row = run_official_react_task(
        task={"task_id": "fake_long", "instruction_text": "finish it", "required_apps": ["spotify"]},
        world=VerboseWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=2, max_history_chars=600),
    )

    second_prompt = row["steps"][1]["user_prompt"]
    assert len(second_prompt) < 1400
    assert "...[truncated]" in second_prompt


def test_official_react_loop_includes_compressed_api_docs_context():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return super().generate_text(system_prompt, user_prompt)

    generator = CapturingGenerator()
    run_official_react_task(
        task={"task_id": "fake_docs", "instruction_text": "inspect library", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            api_docs_context="[Task-Relevant AppWorld API Docs]\n- apis.spotify.show_song_library(...)",
        ),
    )

    assert "[Task-Relevant AppWorld API Docs]" in generator.prompts[0]
    assert "apis.spotify.show_song_library" in generator.prompts[0]


def test_official_react_loop_includes_generic_execution_correctness_guard():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task()\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_guard",
            "instruction_text": "Send the owed money as per my phone text conversation and move files to ~/trash/.",
            "required_apps": ["phone", "venmo", "file_system"],
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    prompt = generator.prompts[0]
    assert "Never guess task values" in prompt
    assert "phone text conversation" in prompt
    assert "inspect messages" in prompt
    assert "use only the basename" in prompt
    assert "destination path must include the target directory and final filename" in prompt
    assert "filter before ranking" in prompt


def test_official_react_loop_includes_complete_ledger_guard_for_money_tasks():
    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task(answer=0)\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_venmo_amount",
            "instruction_text": "How much did my manager pay for dinner after everyone venmoed them?",
            "required_apps": ["venmo"],
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    prompt = generator.prompts[0]
    assert "For money or transaction aggregation" in prompt
    assert "prefer complete ledger APIs such as `show_transactions` or payment-request APIs" in prompt
    assert "Do not rely on social/feed APIs as the only source" in prompt


def test_official_react_loop_blocks_unrequested_account_mutations():
    class SignupGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\napis.spotify.signup(username='new_user', password='pw', email='x@y.com')\n```"

    world = FakeWorld()
    row = run_official_react_task(
        task={"task_id": "fake_mutation", "instruction": "inspect my library", "required_apps": ["spotify"]},
        world=world,
        generator=SignupGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    assert world.outputs == []
    assert row["steps"][0]["preflight_ok"] is False
    assert "Safety preflight blocked" in row["steps"][0]["execute_output"]


def test_official_react_loop_normalizes_positional_complete_task_answer():
    class AnswerGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\nresult = 'A, B, C'\napis.supervisor.complete_task(result)\n```"

    class AnswerWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task(answer=result)" in code:
                self.completed = True
                return "completed"
            return "wrong complete_task call"

    world = AnswerWorld()
    row = run_official_react_task(
        task={"task_id": "fake_answer", "instruction": "answer it", "required_apps": ["spotify"]},
        world=world,
        generator=AnswerGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    assert row["task_completed"] is True
    assert row["steps"][0]["code_normalizer"]["changed"] is True
    assert "complete_task(answer=result)" in world.outputs[0]


def test_official_react_loop_removes_playlist_visibility_filter_for_all_playlists():
    class PlaylistCountGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "def get_all_song_ids(api_function, access_token, **kwargs):\n"
                "    return api_function(access_token=access_token, **kwargs)\n"
                "items = get_all_song_ids(apis.spotify.show_playlist_library, 'token', is_public=False)\n"
                "apis.supervisor.complete_task(answer=len(items))\n"
                "```"
            )

    class CapturingWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            self.completed = True
            return "completed"

    world = CapturingWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_all_playlists",
            "instruction": "How many unique songs are there across my Spotify song library, albums library and all playlists?",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=PlaylistCountGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    assert row["steps"][0]["code_normalizer"]["changed"] is True
    assert row["steps"][0]["code_normalizer"]["reason"] == "spotify_playlist_library_visibility_filter_removed"
    assert "is_public" not in world.outputs[0]


def test_official_react_loop_normalizes_short_page_break_after_extend():
    class BadPaginationGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "page_index = 0\n"
                "page_limit = 20\n"
                "items = []\n"
                "while True:\n"
                "    page = apis.spotify.show_playlist_library(access_token='token', page_index=page_index, page_limit=page_limit)\n"
                "    if not page or len(page) < page_limit:\n"
                "        break\n"
                "    items.extend(page)\n"
                "    page_index += 1\n"
                "apis.supervisor.complete_task(answer=len(items))\n"
                "```"
            )

    class CapturingWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            self.completed = True
            return "completed"

    world = CapturingWorld()
    row = run_official_react_task(
        task={"task_id": "fake_pagination", "instruction": "How many playlists do I have?", "required_apps": ["spotify"]},
        world=world,
        generator=BadPaginationGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    normalized = world.outputs[0]
    assert row["steps"][0]["code_normalizer"]["changed"] is True
    assert "pagination_short_page_break_after_extend" in row["steps"][0]["code_normalizer"]["reason"]
    assert "if not page:\n        break\n    items.extend(page)\n    if len(page) < page_limit:\n        break" in normalized


def test_official_react_loop_keeps_playlist_visibility_filter_when_task_explicitly_public():
    class PublicPlaylistGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "items = apis.spotify.show_playlist_library(access_token='token', is_public=True)\n"
                "apis.supervisor.complete_task(answer=len(items))\n"
                "```"
            )

    class CapturingWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            self.completed = True
            return "completed"

    world = CapturingWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_public_playlists",
            "instruction": "How many public Spotify playlists are in my playlist library?",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=PublicPlaylistGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
    )

    assert row["steps"][0]["code_normalizer"]["changed"] is False
    assert "is_public=True" in world.outputs[0]


def test_official_react_loop_removes_completion_narration_answer_for_state_change_task():
    class StateChangeCompletionGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "apis.spotify.like_song(song_id=1, access_token='token')\n"
                "apis.supervisor.complete_task(answer='All songs from followed artists have been liked.')\n"
                "```"
            )

    class StateChangeWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task()" in code and "answer=" not in code:
                self.completed = True
                return "completed"
            return "wrong completion"

    world = StateChangeWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_state_change",
            "instruction": "Like all the songs from the artists I follow on Spotify.",
            "required_apps": ["spotify"],
        },
        world=world,
        generator=StateChangeCompletionGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            valid_api_refs={
                ("spotify", "like_song"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["task_completed"] is True
    assert row["steps"][0]["code_normalizer"]["changed"] is True
    assert row["steps"][0]["code_normalizer"]["reason"] == "state_change_complete_task_answer_removed"
    assert "complete_task()" in world.outputs[0]
    assert "answer=" not in world.outputs[0]


def test_official_react_loop_removes_completion_narration_answer_for_make_task():
    class MakePlaylistGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return (
                "```python\n"
                "playlist = apis.spotify.create_playlist(title='Most Listened Playlist Songs', access_token='token')\n"
                "apis.supervisor.complete_task(answer='Created playlist with requested songs.')\n"
                "```"
            )

    class MakePlaylistWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task()" in code and "answer=" not in code:
                self.completed = True
                return "completed"
            return "wrong completion"

    world = MakePlaylistWorld()
    row = run_official_react_task(
        task={
            "task_id": "fake_make_playlist",
            "instruction": 'Make me a Spotify playlist called "Most Listened Playlist Songs".',
            "required_apps": ["spotify"],
        },
        world=world,
        generator=MakePlaylistGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            valid_api_refs={
                ("spotify", "create_playlist"),
                ("supervisor", "complete_task"),
            },
        ),
    )

    assert row["task_completed"] is True
    assert row["steps"][0]["code_normalizer"]["reason"] == "state_change_complete_task_answer_removed"
    assert "complete_task()" in world.outputs[0]
    assert "answer=" not in world.outputs[0]


def test_official_react_system_prompt_does_not_globally_bias_idempotent_recovery():
    prompt = build_official_react_system_prompt(max_interactions=40)
    assert "raise_on_failure=False" not in prompt
    assert "already done" not in prompt


def test_official_react_loop_adds_idempotent_recovery_after_write_failure():
    class RetryGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append((system_prompt, user_prompt))
            if len(self.prompts) == 1:
                return "```python\napis.spotify.like_song(song_id=1, access_token='token')\n```"
            return "```python\napis.supervisor.complete_task()\n```"

    class AlreadyLikedWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "Execution failed. Response status code is 422: already liked"

    generator = RetryGenerator()
    row = run_official_react_task(
        task={
            "task_id": "fake_already_liked",
            "instruction": "Like all the songs from the artists I follow on Spotify.",
            "required_apps": ["spotify"],
        },
        world=AlreadyLikedWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=2),
    )

    assert row["task_completed"] is True
    assert "raise_on_failure=False" not in generator.prompts[0][0]
    assert "already satisfied" in generator.prompts[1][1]
    assert "raise_on_failure=False" in generator.prompts[1][1]


def test_official_react_loop_adds_list_return_shape_recovery_after_type_error():
    class ShapeErrorGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append((system_prompt, user_prompt))
            if len(self.prompts) == 1:
                return "```python\ncontacts = apis.phone.search_contacts(access_token='token')['results']\n```"
            return "```python\napis.supervisor.complete_task()\n```"

    class ShapeErrorWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return "Execution failed. TypeError: list indices must be integers or slices, not str"

    generator = ShapeErrorGenerator()
    run_official_react_task(
        task={"task_id": "fake_shape", "instruction": "Send money to a phone contact.", "required_apps": ["phone"]},
        world=ShapeErrorWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=2),
    )

    retry_prompt = generator.prompts[1][1]
    assert "[API Error Recovery]" in retry_prompt
    assert "returned a list" in retry_prompt
    assert "do not access ['results']" in retry_prompt


def test_official_react_loop_adds_date_format_recovery_after_api_422():
    class DateErrorGenerator:
        def __init__(self):
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append((system_prompt, user_prompt))
            if len(self.prompts) == 1:
                return (
                    "```python\n"
                    "apis.venmo.show_transactions(access_token='token', min_created_at='2023-01-01T00:00:00')\n"
                    "```"
                )
            return "```python\napis.supervisor.complete_task()\n```"

    class DateErrorWorld(FakeWorld):
        def execute(self, code):
            self.outputs.append(code)
            if "complete_task" in code:
                self.completed = True
                return "completed"
            return (
                "Execution failed. Exception: Response status code is 422: "
                '{"message":"Invalid min_created_at format. Use YYYY-MM-DD."}'
            )

    generator = DateErrorGenerator()
    run_official_react_task(
        task={"task_id": "fake_date", "instruction": "Like transactions from this year.", "required_apps": ["venmo"]},
        world=DateErrorWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=2),
    )

    retry_prompt = generator.prompts[1][1]
    assert "[API Error Recovery]" in retry_prompt
    assert "Convert datetime values to date-only strings" in retry_prompt
    assert "YYYY-MM-DD" in retry_prompt


def test_official_react_loop_blocks_unknown_api_when_schema_available():
    class UnknownApiGenerator:
        def generate_text(self, system_prompt, user_prompt):
            del system_prompt, user_prompt
            return "```python\napis.spotify.set_access_token('token')\n```"

    world = FakeWorld()
    row = run_official_react_task(
        task={"task_id": "fake_unknown_api", "instruction": "inspect library", "required_apps": ["spotify"]},
        world=world,
        generator=UnknownApiGenerator(),
        config=OfficialReActExecutorConfig(
            max_interactions=1,
            valid_api_refs={
                ("api_docs", "show_app_descriptions"),
                ("api_docs", "show_api_descriptions"),
                ("api_docs", "show_api_doc"),
                ("spotify", "login"),
                ("supervisor", "complete_task"),
                ("supervisor", "show_account_passwords"),
                ("supervisor", "show_profile"),
            },
        ),
    )

    assert world.outputs == []
    assert row["steps"][0]["preflight_ok"] is False
    assert row["steps"][0]["preflight"]["reason"] == "invalid_api_ref"
    assert "apis.spotify.set_access_token" in row["steps"][0]["execute_output"]


def test_official_executor_hides_raw_skill_text_and_records_evidence():
    captured = {}

    class EvidenceBuilder:
        def __call__(self, task, history):
            captured["called"] = True
            return EvidenceResult(
                prompt_text=(
                    "Retrieved evidence:\n"
                    "- Relevant valid APIs: apis.spotify.show_song()\n"
                    "- Useful workflow hint: inspect detail rows"
                ),
                diagnostics={"selected_skill_ids": ["skillx/appworld/spotify/example"]},
            )

    row = run_official_react_task(
        task={"task_id": "fake_2", "instruction": "count songs", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
        evidence_builder=EvidenceBuilder(),
    )
    assert captured["called"] is True
    assert "Retrieved evidence:" in row["steps"][0]["user_prompt"]
    assert "skillx/appworld" not in row["steps"][0]["user_prompt"].lower()
    assert row["steps"][0]["skill_evidence"]["selected_skill_ids"] == [
        "skillx/appworld/spotify/example"
    ]


def test_official_executor_warns_skill_evidence_must_not_drop_user_constraints():
    class EvidenceBuilder:
        def __call__(self, task, history):
            del task, history
            return EvidenceResult(
                prompt_text=(
                    "Retrieved evidence:\n"
                    "- Useful workflow hint: rank songs by play_count"
                ),
                diagnostics={},
            )

    class CapturingGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_text(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return "```python\napis.supervisor.complete_task(answer='x')\n```"

    generator = CapturingGenerator()
    run_official_react_task(
        task={
            "task_id": "fake_constraints",
            "instruction": "Give top 4 r&b song titles across song, album, and playlist libraries.",
            "required_apps": ["spotify"],
        },
        world=FakeWorld(),
        generator=generator,
        config=OfficialReActExecutorConfig(max_interactions=1),
        evidence_builder=EvidenceBuilder(),
    )

    prompt = generator.prompts[0]
    assert "User constraints override retrieved skill evidence" in prompt
    assert "filters such as genre" in prompt
    assert "source/library scope" in prompt
    assert "ranking/count/limit" in prompt


def test_official_executor_notifies_evidence_builder_after_execution():
    observed = []

    class EvidenceBuilder:
        def __call__(self, task, history):
            del task, history
            return "Retrieved evidence:\n- Relevant valid APIs: apis.spotify.show_song_library()"

        def observe(self, *, code, execute_output, step):
            observed.append((code, execute_output, step["step_idx"]))

    row = run_official_react_task(
        task={"task_id": "fake_3", "instruction": "inspect library", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
        evidence_builder=EvidenceBuilder(),
    )

    assert row["steps"][0]["execute_output"] == "[{'name': 'spotify'}]"
    assert observed == [("print(apis.api_docs.show_app_descriptions())", "[{'name': 'spotify'}]", 0)]
