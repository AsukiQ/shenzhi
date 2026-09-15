import json
from pathlib import Path

from clstr import appworld_executor
from clstr.appworld_executor import (
    NullSkillProvider,
    PredictionFileSkillProvider,
    QwenCodeGenerator,
    QwenCodeGeneratorConfig,
    build_executor_comparison,
    build_api_docs_context,
    build_appworld_executor_prompt,
    extract_python_code,
    run_appworld_executor_eval,
)
from scripts.run_appworld_qwen_executor_eval import DEFAULT_PREDICTIONS


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
            "login": {
                "app_name": "spotify",
                "api_name": "login",
                "description": "Log in to Spotify.",
                "parameters": [{"name": "username", "type": "string", "required": True}],
                "response_schemas": {"success": {"access_token": "string"}},
            },
            "show_song": {
                "app_name": "spotify",
                "api_name": "show_song",
                "description": "Show song metadata.",
                "parameters": [{"name": "song_id", "type": "integer", "required": True}],
                "response_schemas": {"success": {"title": "string", "play_count": 1}},
            },
            "show_song_library": {
                "app_name": "spotify",
                "api_name": "show_song_library",
                "description": "List songs in the library.",
                "parameters": [
                    {"name": "access_token", "type": "string", "required": True},
                    {
                        "name": "page_limit",
                        "type": "integer",
                        "required": False,
                        "default": 5,
                        "constraints": ["value >= 1.0, <= 20.0"],
                    },
                ],
                "response_schemas": {"success": [{"song_id": 1, "title": "string"}]},
            },
        },
    )
    return root


def _skill_pool(path: Path) -> Path:
    _write_jsonl(
        path,
        [
            {
                "skill_id": "skillx/appworld/spotify-auth",
                "name": "spotify authenticate",
                "description": "Authenticate to Spotify.",
                "executor_desc": "apis.spotify.login",
                "body": "Call apis.spotify.login before Spotify reads.",
                "failure_modes": [],
            },
            {
                "skill_id": "skillx/appworld/spotify-song",
                "name": "spotify inspect song",
                "description": "Inspect a Spotify song.",
                "executor_desc": "apis.spotify.show_song",
                "body": "Use apis.spotify.show_song(song_id=...).",
                "failure_modes": [],
            },
        ],
    )
    return path


def test_extract_python_code_accepts_fenced_and_raw_code():
    fenced = "```python\nprint('ready')\n```\nextra"
    raw = "apis.supervisor.complete_task(answer='done')"

    assert extract_python_code(fenced) == "print('ready')"
    assert extract_python_code(raw) == raw


def test_prompt_builder_includes_instruction_skills_and_filtered_api_docs(tmp_path):
    root = _fake_appworld_root(tmp_path)
    api_docs = build_api_docs_context(
        appworld_root=root,
        required_apps=["spotify"],
        api_refs=["spotify.login"],
        max_apis_per_app=3,
    )
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Give me a comma-separated list of top 4 most played r&b song titles from across my Spotify song, album and playlist libraries.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-auth",
                "name": "spotify authenticate",
                "description": "Authenticate to Spotify.",
                "executor_desc": "apis.spotify.login",
                "body": "Call apis.spotify.login before Spotify reads.",
            }
        ],
        api_docs_context=api_docs,
    )

    assert "Give me a comma-separated list of top 4 most played r&b song titles" in prompt
    assert "skillx/appworld/spotify-auth" in prompt
    assert "Call apis.spotify.login" in prompt
    assert "apis.spotify.login" in prompt
    assert "returns {\"access_token\": \"string\"}" in prompt
    assert "page_limit: integer optional default 5 constraints value >= 1.0, <= 20.0" in prompt
    assert "apis.supervisor.show_profile()" in prompt
    assert "apis.supervisor.show_account_passwords()" in prompt
    assert "Do not invent placeholder credentials" in prompt
    assert "keyword arguments" in prompt
    assert 'supervisor_passwords["spotify"]' in prompt
    assert "Never call `requester.json()`" in prompt
    assert 'apis.supervisor.complete_task(status="success")' in prompt
    assert "Do not call `apis.skillx`" in prompt
    assert "Use only API names listed in [AppWorld API Docs]" in prompt
    assert "Do not pass parameters that are not listed" in prompt
    assert "Never use `page_limit=100`" in prompt
    assert "A SkillX skill name is not a function name" in prompt
    assert "If a needed field is absent from a list/library response schema" in prompt
    assert "call `apis.spotify.show_song(song_id=song_id)` before reading `genre` or `play_count`" in prompt
    assert "union song IDs from `show_song_library`, `show_album_library`, and `show_playlist_library`" in prompt
    assert "For this task, filter songs where `song[\"genre\"]` matches `r&b` case-insensitively" in prompt
    assert "sort the filtered songs by `song[\"play_count\"]` descending" in prompt
    assert "return exactly 4 song titles as a comma-separated string" in prompt
    assert 'show_song_library` returns song dicts; add `item["song_id"]`' in prompt
    assert 'show_album_library` returns album dicts; add every id from `album["song_ids"]`' in prompt
    assert 'show_playlist_library` returns playlist dicts; add every id from `playlist["song_ids"]`' in prompt
    assert "Do not expect `show_playlist_library` to return a `songs` field" in prompt
    assert "Return Python code only" in prompt


def test_prompt_builder_safe_metadata_omits_unsafe_skill_body(tmp_path):
    root = _fake_appworld_root(tmp_path)
    api_docs = build_api_docs_context(
        appworld_root=root,
        required_apps=["spotify"],
        api_refs=["spotify.login"],
        max_apis_per_app=1,
    )
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Log in and inspect my Spotify library.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/login-to-venmo-and-phone-17",
                "name": "login to venmo and phone",
                "description": "Log in to both Venmo and Phone applications.",
                "executor_desc": "apis.supervisor.show_account_passwords, apis.venmo.login, apis.phone.login",
                "body": (
                    "venmo_login_result = apis.venmo.login("
                    "username='nan_ritt@gmail.com', password=venmo_password)\n"
                    "phone_login_result = apis.phone.login("
                    "username='2307354647', password=phone_password)"
                ),
            }
        ],
        api_docs_context=api_docs,
        skill_context_mode="safe_metadata",
    )

    assert "skillx/appworld/login-to-venmo-and-phone-17" in prompt
    assert "login to venmo and phone" in prompt
    assert "Log in to both Venmo and Phone applications." in prompt
    assert "apis.venmo.login" in prompt
    assert "raw_body_omitted: true" in prompt
    assert "not schema-validated" in prompt
    assert "nan_ritt@gmail.com" not in prompt
    assert "2307354647" not in prompt
    assert "body:" not in prompt


def test_preflight_can_use_full_schema_when_prompt_api_docs_are_truncated(tmp_path):
    root = _fake_appworld_root(tmp_path)
    api_docs = build_api_docs_context(
        appworld_root=root,
        required_apps=["spotify"],
        api_refs=["spotify.login"],
        max_apis_per_app=1,
    )
    load_appworld_api_refs = getattr(appworld_executor, "load_appworld_api_refs", None)

    assert callable(load_appworld_api_refs)
    assert "apis.spotify.login" in api_docs
    assert "apis.spotify.show_song" not in api_docs

    valid_refs = load_appworld_api_refs(appworld_root=root, required_apps=["spotify"])

    assert ("spotify", "show_song") in valid_refs
    assert appworld_executor.preflight_appworld_code(
        "song = apis.spotify.show_song(song_id=1)",
        api_docs,
        valid_api_refs=valid_refs,
    )["ok"] is True
    assert appworld_executor.preflight_appworld_code(
        "song = apis.spotify.fake_api(song_id=1)",
        api_docs,
        valid_api_refs=valid_refs,
    )["invalid_api_refs"] == ["apis.spotify.fake_api"]


def test_preflight_rejects_literal_id_filters_on_search_apis():
    preflight = appworld_executor.preflight_appworld_code(
        "songs = apis.spotify.search_songs(artist_id=1, min_play_count=990)",
        "[AppWorld API Docs]\n- apis.spotify.search_songs(...)\n",
        valid_api_refs={("spotify", "search_songs")},
    )

    assert preflight["ok"] is False
    assert preflight["reason"] == "unsafe_literal_id_filter"
    assert preflight["unsafe_literal_id_filters"] == [
        {
            "api_ref": "apis.spotify.search_songs",
            "parameter": "artist_id",
            "literal": 1,
        }
    ]
    assert "artist_id=1" in preflight["error"]


def test_sanitizer_drops_literal_id_filters_on_search_apis():
    result = appworld_executor.sanitize_appworld_code(
        "songs = apis.spotify.search_songs(artist_id=1, min_play_count=990)\n"
        "song = apis.spotify.show_song(song_id=1)"
    )

    assert result["changed"] is True
    assert "artist_id" not in result["code"]
    assert "min_play_count=990" in result["code"]
    assert "show_song(song_id=1)" in result["code"]
    assert result["removed_unsafe_literal_id_filters"] == [
        {
            "api_ref": "apis.spotify.search_songs",
            "parameter": "artist_id",
            "literal": 1,
        }
    ]


def test_prompt_builder_uses_phone_number_login_and_payment_text_hints():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": (
                "Kristin paid for my grocery recently as my payment cards were not working at the time. "
                "Send them the owed money with a description note \"Groceries\" as per my phone text conversation, "
                "and then send them a phone text message, \"It is done.\"."
            ),
            "required_apps": ["phone", "venmo"],
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.phone.login(...)\napis.venmo.create_transaction(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert 'phone_access_token = apis.phone.login(username=supervisor_profile["phone_number"], password=supervisor_passwords["phone"])["access_token"]' in prompt
    assert 'venmo_access_token = apis.venmo.login(username=supervisor_profile["email"], password=supervisor_passwords["venmo"])["access_token"]' in prompt
    assert 'phone_access_token = apis.phone.login(username=supervisor_profile["email"]' not in prompt
    assert "Phone login username is the supervisor phone number, not email." in prompt
    assert "Find the payer and amount from phone text messages before creating the Venmo transaction." in prompt
    assert "Use `apis.phone.search_contacts(query=...)` to find the payer contact first; read that contact's `email` and `phone_number` fields." in prompt
    assert "`apis.phone.search_text_messages` rows use the text body field `message`, not `text`." in prompt
    assert "Text message `sender` and `receiver` contain `contact_id`, `name`, and `phone_number`; they do not contain `email`." in prompt
    assert "Use the contact email from `search_contacts` as `receiver_email`; never use a phone number as a Venmo email." in prompt
    assert "Call `search_text_messages(phone_number=contact[\"phone_number\"], page_limit=20, page_index=...)` and paginate until an empty page." in prompt
    assert "Extract the grocery amount from `message[\"message\"]` with a dollar-amount regex." in prompt
    assert "Scan every paginated text message; do not assume page 0 or the first message contains the grocery amount." in prompt
    assert "The amount regex must require a literal dollar sign, for example `r\"\\$(\\d+(?:\\.\\d+)?)\"`; do not use optional-dollar patterns like `\\$?`." in prompt
    assert "Filter to messages whose `message` mentions grocery, groceries, paid, payment, card, owed, or bill before choosing the amount." in prompt
    assert "If several relevant messages contain dollar amounts, choose the most recent relevant message by `sent_at`." in prompt
    assert "Do not run one regex over all messages joined together; select one relevant message first, then extract its dollar amount." in prompt
    assert "Skip messages without a dollar amount instead of defaulting to 0 or matching unrelated numbers." in prompt
    assert "Use the payer first name from the task instruction as the contact query." in prompt
    assert "Valid solution order: log in, search the contact, paginate text messages by that contact phone number, regex the amount from `message`, create the Venmo transaction, send the phone text, then call `complete_task`." in prompt
    assert "Import `re` before using a dollar-amount regex." in prompt
    assert "Do not call any `apis.simple_note.*` API for this phone+venmo grocery/text task." in prompt
    assert "Use `apis.venmo.create_transaction(receiver_email=..., amount=..., description=..., access_token=venmo_access_token)`" in prompt
    assert "Use `apis.phone.send_text_message(phone_number=..., message=..., access_token=phone_access_token)`" in prompt
    assert "Do not call `apis.supervisor.complete_task(status=\"success\")` until the Venmo transaction and phone text have both succeeded." in prompt


def test_enrich_task_with_appworld_specs_adds_datetime_without_overwriting(tmp_path):
    root = tmp_path / "appworld_root"
    _write_json(
        root / "data" / "tasks" / "task_1" / "specs.json",
        {"datetime": "2023-05-18T12:00:00", "instruction": "from specs"},
    )
    enrich_task_with_appworld_specs = getattr(appworld_executor, "enrich_task_with_appworld_specs", None)

    assert callable(enrich_task_with_appworld_specs)

    enriched = enrich_task_with_appworld_specs({"task_id": "task_1", "instruction_text": "from row"}, root)
    already = enrich_task_with_appworld_specs(
        {"task_id": "task_1", "instruction_text": "from row", "task_datetime": "2024-01-01T00:00:00"},
        root,
    )

    assert enriched["task_datetime"] == "2023-05-18T12:00:00"
    assert enriched["datetime"] == "2023-05-18T12:00:00"
    assert enriched["instruction_text"] == "from row"
    assert already["task_datetime"] == "2024-01-01T00:00:00"


def test_prompt_builder_includes_appworld_task_datetime_for_relative_dates():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "How many likes did all Venmo transactions, I sent this month, have in total?",
            "required_apps": ["venmo"],
            "task_datetime": "2023-05-18T12:00:00",
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.venmo.show_transactions(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert "[Task Environment]" in prompt
    assert "AppWorld task datetime: 2023-05-18T12:00:00" in prompt
    assert "Use this datetime for relative phrases such as `today`, `this month`, and `this year`." in prompt
    assert "Do not use the host machine date/time for AppWorld relative-date tasks." in prompt


def test_prompt_builder_uses_venmo_like_count_pagination_and_date_hints():
    month_prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "How many likes did all Venmo transactions, I sent this month, have in total?",
            "required_apps": ["venmo"],
            "task_datetime": "2023-05-18T12:00:00",
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.venmo.show_transactions(...)\n",
        skill_context_mode="safe_metadata",
    )
    year_prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_2",
            "instruction_text": "How many likes did all Venmo transactions, I received this year, have in total?",
            "required_apps": ["venmo"],
            "task_datetime": "2023-05-18T12:00:00",
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.venmo.show_transactions(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert "Use `apis.venmo.show_transactions(access_token=venmo_access_token, min_like_count=1, page_limit=20, page_index=...)` for Venmo like-count tasks." in month_prompt
    assert "Paginate Venmo transactions until an empty page; sum `transaction[\"like_count\"]` from every returned page." in month_prompt
    assert "For `this month`, set `min_created_at=\"2023-05-01\"` from the AppWorld task datetime." in month_prompt
    assert "For sent transactions, pass `direction=\"sent\"`; for received transactions, pass `direction=\"received\"`." in month_prompt
    assert "Do not add a broad `max_created_at=\"2023-12-31\"` unless the instruction explicitly asks for an end date." in month_prompt
    assert "Maintain a local integer `page_index` and increment it by 1 each loop; transaction rows do not contain a `page_index` field." in month_prompt
    assert "Never use `apis.supervisor.show_account_passwords()[0]`; always build `supervisor_passwords` by `account_name`." in month_prompt
    assert "For `this year`, set `min_created_at=\"2023-01-01\"` from the AppWorld task datetime." in year_prompt
    assert "For sent transactions, pass `direction=\"sent\"`; for received transactions, pass `direction=\"received\"`." in year_prompt


def test_prompt_builder_adds_file_system_move_and_date_hints():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": (
                'In my file system, add the prefix "YYYY-MM-DD_" to all file names in the ~/downloads/ directory, '
                "based on their creation dates, and then move all files not from this year to ~/trash/."
            ),
            "required_apps": ["file_system"],
            "task_datetime": "2023-05-18T12:00:00",
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.file_system.show_directory(...)\napis.file_system.move_file(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert "The file_system app has no `rename_file` or `categorize_files_by_creation_date` API." in prompt
    assert "Use `show_directory` to list paths, then `show_file(file_path=..., access_token=file_system_access_token)` to read `created_at`." in prompt
    assert "`move_file` requires `source_file_path` and a full `destination_file_path` including the final filename." in prompt
    assert "Do not pass a destination directory alone to `move_file`." in prompt
    assert "Build destination paths with exactly one `/` between directory and filename; avoid `//`." in prompt
    assert "Use `file_detail[\"created_at\"][:10]` for date prefixes." in prompt
    assert "For `this year`, compare against year `2023` from the AppWorld task datetime, not the host clock." in prompt
    assert "Rename every listed file by moving it to the requested date-prefixed filename; do not only move old files." in prompt
    assert "For files moved to trash or recycle directories, the destination path must still include the requested date-prefixed filename." in prompt
    assert "If execution history says `No API named`, do not retry that API; replace it with manual `show_directory` + `show_file` + `move_file` logic." in prompt


def test_prompt_builder_adds_simple_note_export_hints():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": (
                'Export all my Simple Note notes to "~/backups/notes/" directory in my file system. '
                'The files should be named according to the note title, replacing white space with "_", '
                'and the extension should be ".md".'
            ),
            "required_apps": ["file_system", "simple_note"],
        },
        skills=[],
        api_docs_context="[AppWorld API Docs]\napis.simple_note.search_notes(...)\napis.file_system.create_file(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert "`apis.simple_note.search_notes(...)` returns a list of note summaries, not a dict with a `notes` key." in prompt
    assert "Paginate `search_notes(access_token=simple_note_access_token, page_limit=20, page_index=...)` until an empty list." in prompt
    assert "Increment `page_index += 1` inside the `search_notes` loop after processing each non-empty page." in prompt
    assert "Call `show_note(note_id=..., access_token=simple_note_access_token)` for each note before exporting content." in prompt
    assert "Write exactly `note_detail[\"content\"]` to each `.md` file; do not prepend markdown titles or headers." in prompt
    assert "Create the destination directory if needed, then create one file per note with whitespace in the title replaced by `_`." in prompt
    assert "After creating all requested files, call `apis.supervisor.complete_task(status=\"success\")`." in prompt


def test_prompt_builder_filters_low_relevance_skill_metadata_from_executor_context():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": (
                'Export all my Simple Note notes to "~/backups/notes/" directory in my file system. '
                'The files should be named according to the note title, replacing white space with "_", '
                'and the extension should be ".md".'
            ),
            "required_apps": ["file_system", "simple_note"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/simple-note-get-expense-shares-from-note-47",
                "name": "simple_note get expense shares from note",
                "description": "Parse expense shares from a note.",
                "executor_desc": "apis.simple_note.search_notes, apis.simple_note.show_note",
            },
            {
                "skill_id": "skillx/appworld/file-system-export-notes-to-markdown-1",
                "name": "file_system export notes to markdown files",
                "description": "Export notes into markdown files in a backup directory.",
                "executor_desc": "apis.simple_note.search_notes, apis.simple_note.show_note, apis.file_system.create_file",
            },
        ],
        api_docs_context="[AppWorld API Docs]\napis.simple_note.search_notes(...)\napis.file_system.create_file(...)\n",
        skill_context_mode="safe_metadata",
    )

    assert "skillx/appworld/file-system-export-notes-to-markdown-1" in prompt
    assert "skillx/appworld/simple-note-get-expense-shares-from-note-47" not in prompt
    assert "Filtered 1 retrieved SkillX skill(s) with low task relevance before prompting." in prompt


def test_prompt_builder_safe_metadata_blocks_nonexistent_api_from_skill_title():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": (
                'In my file system, add the prefix "YYYY-MM-DD_" to files based on creation dates, '
                "then move old files to trash."
            ),
            "required_apps": ["file_system"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/file-system-categorize-files-by-creation-date-9",
                "name": "file system categorize files by creation date",
                "description": "Categorizes files by creation date.",
                "executor_desc": "apis.file_system.show_file",
            }
        ],
        api_docs_context=(
            "[AppWorld API Docs]\n"
            "- apis.file_system.show_directory(access_token: string required): List files.\n"
            "- apis.file_system.show_file(file_path: string required, access_token: string required): Show metadata.\n"
            "- apis.file_system.move_file(source_file_path: string required, destination_file_path: string required, access_token: string required): Move a file.\n"
        ),
        skill_context_mode="safe_metadata",
        schema_guard_skill_metadata=True,
    )

    assert "executable_appworld_apis_from_skill: apis.file_system.show_file" in prompt
    assert "blocked_nonexistent_skill_title_api: apis.file_system.categorize_files_by_creation_date" in prompt
    assert "reference_name_not_function: file system categorize files by creation date" not in prompt
    assert "skillx/appworld/file-system-categorize-files-by-creation-date-9" not in prompt


def test_prompt_builder_safe_metadata_keeps_read_only_retrieval_skill_title():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Find the top played Spotify songs.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
        api_docs_context=(
            "[AppWorld API Docs]\n"
            "- apis.spotify.show_song_library(access_token: string required): List song ids.\n"
            "- apis.spotify.show_song(song_id: integer required): Show song metadata.\n"
        ),
        skill_context_mode="safe_metadata",
        schema_guard_skill_metadata=True,
    )

    assert "skillx/appworld/spotify-find-songs-based-on-play-count-59" in prompt
    assert "reference_name_not_function: spotify find songs based on play count" in prompt
    assert "executable_appworld_apis_from_skill: apis.spotify.show_song" in prompt
    assert "apis.spotify.show_song_library" in prompt
    assert "blocked_nonexistent_skill_title_api: apis.spotify.find_songs_based_on_play_count" in prompt


def test_prompt_builder_schema_plan_hides_title_and_filters_action_apis():
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Add all songs from Aria Sterling that have been played over 990 times to my Spotify player queue.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-like-songs-by-source-67",
                "name": "spotify like songs by source",
                "description": "Like songs from a specific source.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.like_song, apis.spotify.fake_api",
                "body": "apis.spotify.like_song(song_id=1)",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(access_token: string required): List songs.\n",
        skill_context_mode="schema_plan",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "like_song"),
            ("supervisor", "complete_task"),
        },
    )

    assert "schema-grounded constraints" in prompt
    assert "valid_appworld_apis: apis.spotify.show_song_library" in prompt
    assert "read_support_apis: apis.spotify.show_song_library" in prompt
    assert "state_changing_action_apis: (none)" in prompt
    assert "filtered_state_changing_api_count: 1" in prompt
    assert "not_an_api_allowlist: true" in prompt
    assert "Use any API in [AppWorld API Docs] if required by the user goal." in prompt
    assert "raw_body_omitted: true" in prompt
    assert "skillx/appworld/spotify-like-songs-by-source-67" not in prompt
    assert "spotify like songs by source" not in prompt
    assert "apis.spotify.like_song" not in prompt
    assert "apis.spotify.fake_api" not in prompt


def test_prompt_builder_adaptive_schema_keeps_aligned_skill_and_hides_misaligned_skill():
    skill = {
        "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
        "name": "spotify find songs based on play count",
        "description": "Find songs and rank them by play count.",
        "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
    }
    valid_refs = {
        ("spotify", "show_song_library"),
        ("spotify", "show_song"),
        ("supervisor", "complete_task"),
    }

    aligned_prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Give me the top 4 most played r&b Spotify song titles.",
            "required_apps": ["spotify"],
        },
        skills=[skill],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="adaptive_schema",
        valid_api_refs=valid_refs,
    )
    misaligned_prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_2",
            "instruction_text": "How many unique songs are there across my Spotify song library, albums library and all playlists?",
            "required_apps": ["spotify"],
        },
        skills=[skill],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="adaptive_schema",
        valid_api_refs=valid_refs,
    )

    assert "reference_name_not_function: spotify find songs based on play count" in aligned_prompt
    assert "schema-grounded constraints" not in aligned_prompt
    assert "spotify find songs based on play count" not in misaligned_prompt
    assert "skillx/appworld/spotify-find-songs-based-on-play-count-59" not in misaligned_prompt
    assert "schema-grounded constraints" in misaligned_prompt
    assert "valid_appworld_apis: apis.spotify.show_song, apis.spotify.show_song_library" in misaligned_prompt


def test_prompt_builder_gated_schema_keeps_downgrades_and_drops_skills():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_1",
            "instruction_text": "Add all songs by Aria Sterling to my Spotify queue.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-songs-to-queue-1",
                "name": "spotify add songs to queue",
                "description": "Add matching Spotify songs to the player queue.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.add_song_to_queue",
            },
            {
                "skill_id": "skillx/appworld/spotify-like-songs-by-source-2",
                "name": "spotify like songs by source",
                "description": "Like songs from a source.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.like_song",
            },
            {
                "skill_id": "skillx/appworld/spotify-no-schema-3",
                "name": "spotify vague helper",
                "description": "A vague helper with no executable API evidence.",
                "executor_desc": "Use Spotify.",
            },
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.add_song_to_queue(...)\n",
        skill_context_mode="gated_schema",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "add_song_to_queue"),
            ("spotify", "like_song"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "skillx/appworld/spotify-add-songs-to-queue-1" in prompt
    assert "reference_name_not_function: spotify add songs to queue" in prompt
    assert "schema-grounded constraints" in prompt
    assert "skillx/appworld/spotify-like-songs-by-source-2" not in prompt
    assert "spotify like songs by source" not in prompt
    assert "apis.spotify.like_song" not in prompt
    assert "skillx/appworld/spotify-no-schema-3" not in prompt
    assert "Gated SkillX handoff kept" in prompt
    assert diagnostics["skill_context_mode"] == "gated_schema"
    assert diagnostics["handoff_decision_counts"]["safe_metadata"] == 1
    assert diagnostics["handoff_decision_counts"]["schema_plan"] == 1
    assert diagnostics["handoff_decision_counts"]["drop"] == 1
    assert diagnostics["safe_metadata_skill_count"] == 1
    assert diagnostics["schema_plan_skill_count"] == 1
    assert diagnostics["dropped_skill_count"] == 1
    assert diagnostics["kept_skill_ids"] == [
        "skillx/appworld/spotify-add-songs-to-queue-1",
        "skillx/appworld/spotify-like-songs-by-source-2",
    ]
    assert diagnostics["dropped_skill_ids"] == ["skillx/appworld/spotify-no-schema-3"]


def test_prompt_builder_gated_schema_hides_known_polluting_playcount_skill_for_unique_count():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_2",
            "instruction_text": "How many unique songs are there across my Spotify song library, albums library, and all playlists?",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="gated_schema",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "show_song"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "schema-grounded constraints" in prompt
    assert "valid_appworld_apis: apis.spotify.show_song, apis.spotify.show_song_library" in prompt
    assert "skillx/appworld/spotify-find-songs-based-on-play-count-59" not in prompt
    assert "spotify find songs based on play count" not in prompt
    assert diagnostics["handoff_decisions"][0]["decision"] == "schema_plan"
    assert "pollution_pattern" in diagnostics["handoff_decisions"][0]["reasons"]


def test_prompt_builder_gated_schema_downgrades_non_exact_unique_count_metadata():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_3",
            "instruction_text": "How many unique songs are there across my Spotify libraries?",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-extract-unique-artist-ids-from-songs-54",
                "name": "spotify extract unique artist ids from songs",
                "description": "Extract unique artist identifiers from songs.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="gated_schema",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "show_song"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "schema-grounded constraints" in prompt
    assert "spotify extract unique artist ids from songs" not in prompt
    assert diagnostics["handoff_decisions"][0]["decision"] == "schema_plan"
    assert "unique_song_goal_requires_schema_only" in diagnostics["handoff_decisions"][0]["reasons"]


def test_prompt_builder_gated_schema_downgrades_queue_goal_without_queue_write_api():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_4",
            "instruction_text": "Add all songs by Aria Sterling to my Spotify player queue.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11",
                "name": "spotify identify songs played so far including current",
                "description": "Inspect the current player queue and listening history.",
                "executor_desc": "apis.spotify.show_song_queue, apis.spotify.show_song",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_queue(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="gated_schema",
        valid_api_refs={
            ("spotify", "show_song_queue"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "schema-grounded constraints" in prompt
    assert "spotify identify songs played so far including current" not in prompt
    assert diagnostics["handoff_decisions"][0]["decision"] == "schema_plan"
    assert "add_queue_goal_without_queue_write_api" in diagnostics["handoff_decisions"][0]["reasons"]


def test_prompt_builder_api_evidence_hides_skill_text_and_exposes_valid_api_refs():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_5",
            "instruction_text": "How many unique Spotify songs are available across my libraries?",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
                "body": "DO NOT LEAK BODY. Use apis.spotify.show_song_library and apis.spotify.show_song.",
                "skill_md": "# spotify find songs based on play count\nDO NOT LEAK SKILL MD",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n- apis.spotify.show_song(...)\n",
        skill_context_mode="api_evidence",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "show_song"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "[CLSTR-Prioritized API Evidence]" in prompt
    assert "skill_text_hidden: true" in prompt
    assert "valid_appworld_apis_from_selected_skills: apis.spotify.show_song, apis.spotify.show_song_library" in prompt
    assert "read_support_apis: apis.spotify.show_song, apis.spotify.show_song_library" in prompt
    assert "state_changing_action_apis: (none)" in prompt
    assert "Use any API in [AppWorld API Docs] if required by the user goal." in prompt
    assert "skillx/appworld/spotify-find-songs-based-on-play-count-59" not in prompt
    assert "spotify find songs based on play count" not in prompt
    assert "Find songs and rank them by play count" not in prompt
    assert "DO NOT LEAK BODY" not in prompt
    assert "DO NOT LEAK SKILL MD" not in prompt
    assert diagnostics["skill_context_mode"] == "api_evidence"
    assert diagnostics["selected_skill_ids"] == ["skillx/appworld/spotify-find-songs-based-on-play-count-59"]
    assert diagnostics["api_evidence_refs"] == ["apis.spotify.show_song", "apis.spotify.show_song_library"]
    assert diagnostics["read_support_apis"] == ["apis.spotify.show_song", "apis.spotify.show_song_library"]
    assert diagnostics["state_changing_action_apis"] == []
    assert diagnostics["hidden_skill_text_fields"] == ["skill_id", "name", "description", "body", "skill_md"]


def test_prompt_builder_api_evidence_records_invalid_refs_without_trusting_them():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_6",
            "instruction_text": "Inspect my Spotify library.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-fake-helper-60",
                "name": "spotify fake helper",
                "description": "Should not expose invalid API as valid.",
                "executor_desc": "apis.spotify.fake_api",
                "body": "apis.spotify.fake_api()",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.show_song_library(...)\n",
        skill_context_mode="api_evidence",
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "[CLSTR-Prioritized API Evidence]" in prompt
    assert "valid_appworld_apis_from_selected_skills: (none)" in prompt
    assert "invalid_or_filtered_refs: apis.spotify.fake_api" in prompt
    assert "apis.spotify.fake_api" not in diagnostics["api_evidence_refs"]
    assert diagnostics["invalid_skill_api_refs"] == ["apis.spotify.fake_api"]


def test_prompt_builder_verified_hints_hides_skill_text_and_emits_optional_hints():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_verified_hints",
            "instruction_text": "Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.add_to_queue",
                "body": "DO NOT LEAK BODY. Add matching songs to queue.",
            }
        ],
        api_docs_context="[AppWorld API Docs]\n- apis.spotify.search_songs(...)\n- apis.spotify.show_song(...)\n- apis.spotify.add_to_queue(...)\n",
        skill_context_mode="verified_hints",
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "[Optional Retrieved Evidence]" in prompt
    assert "[CLSTR-Prioritized API Evidence]" not in prompt
    assert "decision: schema_only_evidence" in prompt
    assert "- apis.spotify.add_to_queue" in prompt
    assert "Add each matching song to the queue" not in prompt
    assert "skillx/appworld/spotify-add-filtered-songs-to-queue-3" not in prompt
    assert "spotify add filtered songs to queue" not in prompt
    assert "DO NOT LEAK BODY" not in prompt
    assert diagnostics["skill_context_mode"] == "verified_hints"
    assert diagnostics["gate_decision"] == "schema_only_evidence"
    assert diagnostics["handoff_decision_counts"]["inject_hint"] == 1
    assert diagnostics["hidden_skill_text_fields"] == ["skill_id", "name", "description", "body", "skill_md"]


def test_prompt_builder_verified_metadata_keeps_verified_metadata_and_hides_suppressed_skills():
    diagnostics: dict = {}
    prompt = build_appworld_executor_prompt(
        task={
            "task_id": "task_verified_metadata",
            "instruction_text": "Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
            "required_apps": ["spotify"],
        },
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.add_to_queue",
                "body": "DO NOT LEAK VERIFIED BODY.",
            },
            {
                "skill_id": "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11",
                "name": "spotify identify songs played so far including current",
                "description": "Inspect current queue and played songs.",
                "executor_desc": "apis.spotify.show_song_queue, apis.spotify.show_song",
                "body": "DO NOT LEAK SUPPRESSED BODY.",
            },
        ],
        api_docs_context=(
            "[AppWorld API Docs]\n"
            "- apis.spotify.search_songs(...)\n"
            "- apis.spotify.add_to_queue(...)\n"
            "- apis.spotify.show_song_queue(...)\n"
            "- apis.spotify.show_song(...)\n"
        ),
        skill_context_mode="verified_metadata",
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "add_to_queue"),
            ("spotify", "show_song_queue"),
            ("spotify", "show_song"),
            ("supervisor", "complete_task"),
        },
        diagnostics_out=diagnostics,
    )

    assert "spotify add filtered songs to queue" in prompt
    assert "Search matching songs and add them to queue." in prompt
    assert "DO NOT LEAK VERIFIED BODY" not in prompt
    assert "spotify identify songs played so far including current" not in prompt
    assert "DO NOT LEAK SUPPRESSED BODY" not in prompt
    assert "[Optional Retrieved Evidence]" in prompt
    assert "- apis.spotify.add_to_queue" in prompt
    assert diagnostics["skill_context_mode"] == "verified_metadata"
    assert diagnostics["gate_decision"] == "schema_only_evidence"
    assert diagnostics["handoff_decision_counts"] == {
        "inject_hint": 1,
        "schema_only": 0,
        "suppress": 1,
    }
    assert diagnostics["verified_metadata_skill_count"] == 1
    assert diagnostics["suppressed_skill_ids"] == [
        "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11"
    ]


def test_prediction_file_skill_provider_returns_ranked_skill_rows(tmp_path):
    skill_pool = _skill_pool(tmp_path / "skill_pool.jsonl")
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [{"query_id": "task_1", "ranked_skill_ids": ["skillx/appworld/spotify-song"]}])

    provider = PredictionFileSkillProvider(skill_pool_path=skill_pool, predictions_path=predictions)

    selected = provider.get_skills({"query_id": "task_1", "task_id": "task_1"}, top_k=1)
    assert selected[0]["skill_id"] == "skillx/appworld/spotify-song"
    assert selected[0]["name"] == "spotify inspect song"


def test_prediction_file_skill_provider_can_pack_duplicate_auth_heavy_topk(tmp_path):
    skill_pool = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool,
        [
            {
                "skill_id": "spotify-auth-a",
                "name": "spotify authenticate",
                "executor_desc": "apis.spotify.login",
                "body": "Log in to Spotify.",
            },
            {
                "skill_id": "spotify-auth-b",
                "name": "spotify_authenticate",
                "executor_desc": "apis.spotify.login",
                "body": "Log in to Spotify.",
            },
            {
                "skill_id": "phone-auth",
                "name": "phone authenticate",
                "executor_desc": "apis.phone.login",
                "body": "Log in to Phone.",
            },
            {
                "skill_id": "songs",
                "name": "spotify inspect songs",
                "executor_desc": "apis.spotify.show_song_library apis.spotify.show_song",
                "body": "Inspect Spotify songs.",
            },
            {
                "skill_id": "playlist",
                "name": "spotify inspect playlists",
                "executor_desc": "apis.spotify.show_playlist_library",
                "body": "Inspect Spotify playlists.",
            },
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "query_id": "task_1",
                "ranked_skill_ids": ["spotify-auth-a", "spotify-auth-b", "phone-auth", "songs", "playlist"],
            }
        ],
    )

    provider = PredictionFileSkillProvider(
        skill_pool_path=skill_pool,
        predictions_path=predictions,
        dedupe_canonical_skills=True,
        max_auth_like_skills=1,
    )

    selected = provider.get_skills({"query_id": "task_1", "task_id": "task_1"}, top_k=3)
    assert [skill["skill_id"] for skill in selected] == ["spotify-auth-a", "songs", "playlist"]


def test_qwen_executor_default_clstr_base_predictions_use_routing_only_fit():
    assert DEFAULT_PREDICTIONS["clstr_base"] == "outputs/appworld_clstr_eval/routing_only_fit_v1_dev/predictions.jsonl"


def test_qwen_executor_default_clstr_skillrouter_init_predictions_are_registered():
    assert (
        DEFAULT_PREDICTIONS["clstr_skillrouter_init"]
        == "outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/predictions.jsonl"
    )


class _FakeQwenBackend:
    def __init__(self):
        self.calls = []

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "apis.supervisor.complete_task(status='success')"

    def metadata(self) -> dict:
        return {
            "qwen_backend_version": "shared_generation_v1",
            "model_name_or_path": "models/Qwen3-8B",
            "local_files_only": True,
            "enable_thinking": False,
        }


def test_qwen_code_generator_uses_shared_backend_metadata():
    backend = _FakeQwenBackend()
    generator = QwenCodeGenerator(
        QwenCodeGeneratorConfig(model_name_or_path="models/Qwen3-8B"),
        backend=backend,
    )

    response = generator.generate("return code")

    assert response == "apis.supervisor.complete_task(status='success')"
    assert backend.calls == [("You are an AppWorld Python-code executor. Return code only.", "return code")]
    assert generator.metadata()["qwen_backend_version"] == "shared_generation_v1"
    assert generator.metadata()["model_name_or_path"] == "models/Qwen3-8B"
    assert generator.metadata()["local_files_only"] is True
    assert generator.metadata()["enable_thinking"] is False


class _FakeGenerator:
    def generate(self, prompt: str) -> str:
        assert "Find my Spotify song title." in prompt
        return "```python\napis.supervisor.complete_task(answer='done')\n```"


class _FakeWorld:
    def __init__(self, task_id, experiment_name, **kwargs):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self.executed_code = ""
        self.closed = False

    def execute(self, code):
        self.executed_code = code
        return "Execution successful."

    def task_completed(self):
        return "complete_task" in self.executed_code

    def evaluate(self, suppress_errors=True):
        return {"suppress_errors": suppress_errors, "success": True, "ok": True}

    def close(self):
        self.closed = True


class _FakeTracker:
    def to_dict(self, stats_only=False):
        return {"stats_only": stats_only, "success": True, "checks": {"ok": True}}


class _FakeEvaluateObjectWorld(_FakeWorld):
    def evaluate(self, suppress_errors=True):
        return _FakeTracker()


class _FakeFailedExecutionWorld(_FakeWorld):
    def execute(self, code):
        self.executed_code = code
        return "Execution failed. Traceback:\n  File \"<python-input>\", line 1, in <module>\nValueError: bad"


class _FakeIncorrectWorld(_FakeWorld):
    def evaluate(self, suppress_errors=True):
        return {"suppress_errors": suppress_errors, "success": False, "failures": ["wrong answer"]}


def test_executor_eval_runs_fake_world_and_writes_report(tmp_path):
    root = _fake_appworld_root(tmp_path)
    skill_pool = _skill_pool(tmp_path / "skill_pool.jsonl")
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "query_id": "task_1",
                "task_id": "task_1",
                "instruction_text": "Find my Spotify song title.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.login"],
            }
        ],
    )

    report = run_appworld_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool,
        output_dir=tmp_path / "executor",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only",
        skill_provider=NullSkillProvider(),
        generator=_FakeGenerator(),
        world_factory=_FakeWorld,
        max_tasks=1,
        top_k=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "qwen_only"
    assert report["task_count"] == 1
    assert report["success_count"] == 1
    assert report["success_rate"] == 1.0
    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text().splitlines()]
    assert runs[0]["task_completed"] is True
    assert runs[0]["evaluation_success"] is True
    assert "complete_task" in runs[0]["code"]


def test_executor_eval_serializes_appworld_tracker_objects(tmp_path):
    root = _fake_appworld_root(tmp_path)
    skill_pool = _skill_pool(tmp_path / "skill_pool.jsonl")
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "query_id": "task_1",
                "task_id": "task_1",
                "instruction_text": "Find my Spotify song title.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.login"],
            }
        ],
    )

    report = run_appworld_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool,
        output_dir=tmp_path / "executor",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only",
        skill_provider=NullSkillProvider(),
        generator=_FakeGenerator(),
        world_factory=_FakeEvaluateObjectWorld,
        max_tasks=1,
        top_k=2,
    )

    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text().splitlines()]
    assert runs[0]["evaluate"] == {"stats_only": False, "success": True, "checks": {"ok": True}}


def test_executor_eval_requires_evaluate_success_for_task_success(tmp_path):
    root = _fake_appworld_root(tmp_path)
    skill_pool = _skill_pool(tmp_path / "skill_pool.jsonl")
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "query_id": "task_1",
                "task_id": "task_1",
                "instruction_text": "Find my Spotify song title.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.login"],
            }
        ],
    )

    report = run_appworld_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool,
        output_dir=tmp_path / "executor",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only",
        skill_provider=NullSkillProvider(),
        generator=_FakeGenerator(),
        world_factory=_FakeIncorrectWorld,
        max_tasks=1,
        top_k=2,
    )

    assert report["success_count"] == 0
    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text().splitlines()]
    assert runs[0]["execution_ok"] is True
    assert runs[0]["task_completed"] is True
    assert runs[0]["evaluation_success"] is False
    assert runs[0]["success"] is False


def test_executor_eval_counts_traceback_execute_output_as_execution_failure(tmp_path):
    root = _fake_appworld_root(tmp_path)
    skill_pool = _skill_pool(tmp_path / "skill_pool.jsonl")
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "query_id": "task_1",
                "task_id": "task_1",
                "instruction_text": "Find my Spotify song title.",
                "required_apps": ["spotify"],
                "api_refs": ["spotify.login"],
            }
        ],
    )

    report = run_appworld_executor_eval(
        tasks_path=tasks_path,
        skill_pool_path=skill_pool,
        output_dir=tmp_path / "executor",
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        method="qwen_only",
        skill_provider=NullSkillProvider(),
        generator=_FakeGenerator(),
        world_factory=_FakeFailedExecutionWorld,
        max_tasks=1,
        top_k=2,
    )

    assert report["execution_failures"] == 1
    runs = [json.loads(line) for line in Path(report["runs_path"]).read_text().splitlines()]
    assert runs[0]["execution_ok"] is False
    assert runs[0]["success"] is False


def test_build_executor_comparison_collects_success_rates(tmp_path):
    report_a = tmp_path / "qwen_only" / "report.json"
    report_b = tmp_path / "clstr_base" / "report.json"
    _write_json(
        report_a,
        {
            "status": "ok",
            "method": "qwen_only",
            "task_count": 3,
            "success_count": 1,
            "success_rate": 0.333333,
            "generation_failures": 0,
            "execution_failures": 1,
        },
    )
    _write_json(
        report_b,
        {
            "status": "ok",
            "method": "clstr_base",
            "task_count": 3,
            "success_count": 2,
            "success_rate": 0.666667,
            "generation_failures": 0,
            "execution_failures": 0,
        },
    )

    comparison = build_executor_comparison([report_a, report_b], tmp_path / "comparison")

    assert comparison["status"] == "ok"
    assert comparison["rows"][0]["method"] == "qwen_only"
    assert comparison["rows"][1]["success_rate"] == 0.666667
    assert (tmp_path / "comparison" / "comparison.md").exists()
