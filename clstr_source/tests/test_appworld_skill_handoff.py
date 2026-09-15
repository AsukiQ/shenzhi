from clstr.appworld_skill_handoff import build_verified_skill_handoff


def test_verified_handoff_suppresses_queue_skill_without_queue_write_api():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11",
                "name": "spotify identify songs played so far including current",
                "description": "Inspect the current queue and played songs.",
                "executor_desc": "apis.spotify.show_song_queue, apis.spotify.show_song",
            }
        ],
        valid_api_refs={
            ("spotify", "show_song_queue"),
            ("spotify", "show_song"),
            ("spotify", "search_songs"),
            ("spotify", "add_to_queue"),
        },
        required_apps={"spotify"},
    )

    assert handoff["decisions"][0]["decision"] == "suppress"
    assert "add_queue_goal_without_queue_write_api" in handoff["decisions"][0]["reasons"]
    assert "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11" not in handoff["prompt_block"]
    assert "spotify identify songs played so far including current" not in handoff["prompt_block"]
    assert "apis.spotify.show_song_queue" not in handoff["prompt_block"]
    assert handoff["suppressed_schema_refs"] == ["apis.spotify.show_song", "apis.spotify.show_song_queue"]
    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["prompt_block"] == ""


def test_verified_handoff_injects_queue_workflow_when_queue_write_api_matches():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.add_to_queue",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.93, 0.84], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "inject_hint"
    assert handoff["state_changing_action_apis"] == ["apis.spotify.add_to_queue"]
    assert "[Optional Retrieved Evidence]" in handoff["prompt_block"]
    assert "valid_tools_or_apis:" in handoff["prompt_block"]
    assert "- apis.spotify.add_to_queue" in handoff["prompt_block"]
    assert "Add each matching song to the queue" in handoff["prompt_block"]
    assert "skillx/appworld/spotify-add-filtered-songs-to-queue-3" not in handoff["prompt_block"]


def test_verified_handoff_adds_schema_fallback_for_queue_when_all_skills_suppressed():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-identify-songs-played-so-far-including-current-11",
                "name": "spotify identify songs played so far including current",
                "description": "Inspect the current queue and played songs.",
                "executor_desc": "apis.spotify.show_song_queue, apis.spotify.show_song",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
            ("spotify", "show_song_queue"),
        },
        required_apps={"spotify"},
    )

    assert handoff["decisions"][0]["decision"] == "suppress"
    assert handoff["fallback_source"] == "task_schema"
    assert handoff["useful_apis"] == [
        "apis.spotify.add_to_queue",
        "apis.spotify.search_songs",
        "apis.spotify.show_song",
    ]
    assert "- apis.spotify.show_song_queue" not in handoff["prompt_block"]
    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["prompt_block"] == ""


def test_verified_handoff_adds_schema_fallback_for_payment_and_text_workflow():
    handoff = build_verified_skill_handoff(
        instruction=(
            'Joseph paid for my grocery recently. Send them the owed money with a description note "Grocery Bill" '
            'as per my phone text conversation, and then send them a phone text message, "Done.".'
        ),
        skills=[
            {
                "skill_id": "skillx/appworld/phone-delete-all-messages-from-contact-43",
                "name": "phone delete all messages from contact",
                "description": "Delete text messages from a contact.",
                "executor_desc": "apis.phone.delete_text_message",
            }
        ],
        valid_api_refs={
            ("phone", "search_contacts"),
            ("phone", "search_text_messages"),
            ("phone", "send_text_message"),
            ("venmo", "create_transaction"),
        },
        required_apps={"phone", "venmo"},
        prompt_style="legacy_hints",
    )

    assert handoff["fallback_source"] == "task_schema_payment_message"
    assert "apis.phone.search_text_messages" in handoff["useful_apis"]
    assert "apis.venmo.create_transaction" in handoff["state_changing_action_apis"]
    assert "apis.phone.send_text_message" in handoff["state_changing_action_apis"]
    assert "read the conversation before sending money" in handoff["prompt_block"].lower()
    assert "preserve quoted notes/messages exactly" in handoff["prompt_block"].lower()
    assert handoff["gate_decision"] == "schema_only_evidence"


def test_verified_handoff_compresses_unique_count_without_playcount_pollution():
    handoff = build_verified_skill_handoff(
        instruction="How many unique songs are there across my Spotify song library, albums library and all playlists?",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            },
            {
                "skill_id": "skillx/appworld/spotify-count-unique-songs-across-libraries-1",
                "name": "spotify count unique songs across libraries",
                "description": "Traverse song, album, and playlist libraries and count unique song ids.",
                "executor_desc": (
                    "apis.spotify.show_song_library, apis.spotify.show_album_library, "
                    "apis.spotify.show_playlist_library, apis.spotify.show_song"
                ),
            },
        ],
        valid_api_refs={
            ("spotify", "show_song_library"),
            ("spotify", "show_album_library"),
            ("spotify", "show_playlist_library"),
            ("spotify", "show_song"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.91, 0.86], "transition_scores_available": True},
    )

    decisions = {row["skill_id"]: row for row in handoff["decisions"]}
    assert decisions["skillx/appworld/spotify-find-songs-based-on-play-count-59"]["decision"] == "suppress"
    assert "unique_count_goal_vs_ranking_skill" in decisions["skillx/appworld/spotify-find-songs-based-on-play-count-59"]["reasons"]
    assert decisions["skillx/appworld/spotify-count-unique-songs-across-libraries-1"]["decision"] == "inject_hint"
    assert "- apis.spotify.show_album_library" in handoff["prompt_block"]
    assert "- apis.spotify.show_playlist_library" in handoff["prompt_block"]
    assert "Collect song ids across song library, album library, and playlist library" in handoff["prompt_block"]
    assert "play count" not in handoff["prompt_block"].lower()


def test_verified_handoff_accepts_reference_model_scores_as_secondary_evidence():
    handoff = build_verified_skill_handoff(
        instruction="Export my Simple Note notes to a CSV file.",
        skills=[
            {
                "skill_id": "skillx/appworld/simple-note-export-notes-1",
                "name": "simple note export notes",
                "description": "Export notes with exact content.",
                "executor_desc": "apis.simple_note.search_notes, apis.simple_note.show_note",
            }
        ],
        valid_api_refs={
            ("simple_note", "search_notes"),
            ("simple_note", "show_note"),
            ("file_system", "write_file"),
        },
        required_apps={"simple_note", "file_system"},
        reference_scores={
            "skillx/appworld/simple-note-export-notes-1": {
                "operation_match_score": 0.92,
                "label": "operation_match",
            }
        },
        controller_diagnostics={"scores": [0.91], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "inject_hint"
    assert handoff["decisions"][0]["reference_label"] == "operation_match"
    assert handoff["decisions"][0]["reference_operation_match_score"] == 0.92
    assert "- apis.simple_note.search_notes" in handoff["prompt_block"]
    assert "Search notes, fetch full note content, then write/export the requested file" in handoff["prompt_block"]


def test_verified_handoff_suppresses_write_api_evidence_for_read_only_question():
    handoff = build_verified_skill_handoff(
        instruction=(
            "I went on dinner with my coworkers yesterday at Azure Harbor Bistro. "
            "My manager paid for food and everyone venmoed them. "
            "How much did my manager pay for the others, including me, yesterday?"
        ),
        skills=[
            {
                "skill_id": "skillx/appworld/venmo-send-transaction-34",
                "name": "venmo send transaction",
                "description": "Create a Venmo transaction and optional comment.",
                "executor_desc": (
                    "apis.venmo.login, apis.venmo.create_transaction, "
                    "apis.venmo.create_transaction_comment"
                ),
            }
        ],
        valid_api_refs={
            ("venmo", "login"),
            ("venmo", "show_social_feed"),
            ("venmo", "create_transaction"),
            ("venmo", "create_transaction_comment"),
        },
        required_apps={"venmo"},
        controller_diagnostics={"scores": [0.91], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "suppress"
    assert "read_only_goal_with_state_changing_skill" in handoff["decisions"][0]["reasons"]
    assert "apis.venmo.create_transaction" not in handoff["prompt_block"]
    assert "apis.venmo.create_transaction_comment" not in handoff["prompt_block"]
    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["prompt_block"] == ""


def test_verified_handoff_omits_auth_only_evidence_after_intent_filtering():
    handoff = build_verified_skill_handoff(
        instruction=(
            "I went on dinner with my coworkers yesterday at Azure Harbor Bistro. "
            "My manager paid for food and everyone venmoed them. "
            "How much did my manager pay for the others, including me, yesterday?"
        ),
        skills=[
            {
                "skill_id": "skillx/appworld/venmo-send-transaction-34",
                "name": "venmo send transaction",
                "description": "Create a Venmo transaction and optional comment.",
                "executor_desc": (
                    "apis.venmo.login, apis.venmo.create_transaction, "
                    "apis.venmo.create_transaction_comment"
                ),
            },
            {
                "skill_id": "skillx/appworld/venmo-authenticate-32",
                "name": "venmo authenticate",
                "description": "Login to Venmo with supervisor password.",
                "executor_desc": "apis.supervisor.show_account_passwords, apis.venmo.login",
            },
        ],
        valid_api_refs={
            ("supervisor", "show_account_passwords"),
            ("venmo", "login"),
            ("venmo", "show_social_feed"),
            ("venmo", "create_transaction"),
            ("venmo", "create_transaction_comment"),
        },
        required_apps={"venmo"},
        controller_diagnostics={"scores": [0.91, 0.88], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "suppress"
    assert handoff["decisions"][1]["decision"] == "schema_only"
    assert handoff["useful_apis"] == []
    assert handoff["fallback_source"] == "auth_only_evidence_suppressed"
    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["prompt_block"] == ""


def test_verified_handoff_suppresses_read_only_evidence_for_state_changing_goal_without_write_api():
    handoff = build_verified_skill_handoff(
        instruction='Make me a Spotify playlist called "Most Listened Playlist Songs" containing only the most-played song from each of my playlists.',
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find and rank songs by play count.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.show_playlist",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "show_playlist"),
            ("spotify", "create_playlist"),
            ("spotify", "add_songs_to_playlist"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.91], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "suppress"
    assert "state_changing_goal_without_state_changing_api" in handoff["decisions"][0]["reasons"]
    assert "apis.spotify.search_songs" not in handoff["prompt_block"]
    assert "apis.spotify.show_song" not in handoff["prompt_block"]
    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["prompt_block"] == ""


def test_verified_handoff_treats_containing_playlist_goal_as_add_action():
    handoff = build_verified_skill_handoff(
        instruction='Make me a Spotify playlist called "Most Listened Playlist Songs" containing only the most-played song from each of my playlists.',
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-songs-to-playlist-24",
                "name": "spotify add songs to playlist",
                "description": "Add selected songs to a playlist.",
                "executor_desc": "apis.spotify.add_song_to_playlist",
            }
        ],
        valid_api_refs={
            ("spotify", "add_song_to_playlist"),
            ("spotify", "create_playlist"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.91], "transition_scores_available": True},
    )

    assert handoff["decisions"][0]["decision"] == "inject_hint"
    assert handoff["state_changing_action_apis"] == ["apis.spotify.add_song_to_playlist"]
    assert "- apis.spotify.add_song_to_playlist" in handoff["prompt_block"]


def test_verified_handoff_gate_omits_prompt_when_all_evidence_is_suppressed_without_schema_fallback():
    handoff = build_verified_skill_handoff(
        instruction="How many unique songs are there across my Spotify song library?",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-rank-play-count",
                "name": "spotify rank play count",
                "description": "Rank songs by play count.",
                "executor_desc": "apis.spotify.show_song_library",
            }
        ],
        valid_api_refs={("spotify", "show_song_library")},
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.4]},
    )

    assert handoff["gate_decision"] == "no_evidence_fallback"
    assert handoff["exact_base_executor_fallback"] is True
    assert handoff["prompt_block"] == ""
    assert "[Optional Retrieved Hints]" not in handoff["prompt_block"]


def test_verified_handoff_gate_emits_structured_workflow_evidence_for_high_confidence_hints():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.add_to_queue",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.93, 0.84], "transition_scores_available": True},
    )

    assert handoff["gate_decision"] == "workflow_hint_evidence"
    assert "[Optional Retrieved Evidence]" in handoff["prompt_block"]
    assert "[Optional Retrieved Hints]" not in handoff["prompt_block"]
    assert "decision: workflow_hint_evidence" in handoff["prompt_block"]
    assert "Add each matching song to the queue" in handoff["prompt_block"]


def test_verified_handoff_can_render_legacy_hint_prompt_style_with_gate_decision():
    handoff = build_verified_skill_handoff(
        instruction="Add all songs from Astrid Nightshade that have been played over 980 times to my Spotify player queue.",
        skills=[
            {
                "skill_id": "skillx/appworld/spotify-add-filtered-songs-to-queue-3",
                "name": "spotify add filtered songs to queue",
                "description": "Search matching songs and add them to queue.",
                "executor_desc": "apis.spotify.search_songs, apis.spotify.show_song, apis.spotify.add_to_queue",
            }
        ],
        valid_api_refs={
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
        },
        required_apps={"spotify"},
        controller_diagnostics={"scores": [0.93, 0.84], "transition_scores_available": True},
        prompt_style="legacy_hints",
    )

    assert handoff["gate_decision"] == "workflow_hint_evidence"
    assert handoff["prompt_style"] == "legacy_hints"
    assert "[Optional Retrieved Hints]" in handoff["prompt_block"]
    assert "[Optional Retrieved Evidence]" not in handoff["prompt_block"]
    assert "Potentially useful APIs:" in handoff["prompt_block"]
    assert "- apis.spotify.add_to_queue" in handoff["prompt_block"]
    assert "Possible workflow hints:" in handoff["prompt_block"]
    assert "Add each matching song to the queue" in handoff["prompt_block"]


def test_verified_handoff_exposes_read_only_arithmetic_constraints_without_workflow_hint():
    handoff = build_verified_skill_handoff(
        instruction=(
            "How much did my manager pay for the others, including me? "
            "Everyones' transactions except mine should be on my social feed. My share was $38."
        ),
        skills=[
            {
                "skill_id": "skillx/appworld/venmo-get-transactions-21",
                "name": "venmo get transactions",
                "description": "Inspect Venmo transactions and social feed.",
                "executor_desc": "apis.venmo.show_social_feed, apis.venmo.show_transactions",
            }
        ],
        valid_api_refs={
            ("venmo", "show_social_feed"),
            ("venmo", "show_transactions"),
        },
        required_apps={"venmo"},
        controller_diagnostics={"scores": [0.8, 0.7], "transition_scores_available": False},
        prompt_style="legacy_hints",
    )

    assert handoff["gate_decision"] == "schema_only_evidence"
    assert "Constraint checks:" in handoff["prompt_block"]
    assert "add the explicitly stated user amount/share to the aggregate" in handoff["prompt_block"]
    assert "do not subtract it from records that already omit it" in handoff["prompt_block"]
    assert "Possible workflow hints:" in handoff["prompt_block"]
    assert "Possible workflow hints:\n- (none)" in handoff["prompt_block"]


def test_verified_handoff_exposes_file_move_destination_constraints_without_workflow_hint():
    handoff = build_verified_skill_handoff(
        instruction=(
            'Add the prefix "YYYY_MM_DD_" to all file names in ~/my_downloads/ based on creation dates, '
            "and then move all files not from this year to ~/archive/."
        ),
        skills=[
            {
                "skill_id": "skillx/appworld/file-system-organize-files-by-date-9",
                "name": "file system organize files by date",
                "description": "Rename files with date prefixes and move files into an archive.",
                "executor_desc": "apis.file_system.show_file, apis.file_system.move_file",
            }
        ],
        valid_api_refs={
            ("file_system", "show_file"),
            ("file_system", "move_file"),
            ("file_system", "show_directory"),
        },
        required_apps={"file_system"},
        controller_diagnostics={"scores": [0.8, 0.7], "transition_scores_available": False},
        prompt_style="legacy_hints",
    )

    assert handoff["gate_decision"] == "schema_only_evidence"
    assert "Constraint checks:" in handoff["prompt_block"]
    assert "destination path must include the target directory and final file name" in handoff["prompt_block"]
    assert "derive file names from the basename, not the full source path" in handoff["prompt_block"]
    assert "combine the final directory with the renamed/prefixed basename" in handoff["prompt_block"]


def test_verified_handoff_limits_read_only_visible_schema_refs():
    handoff = build_verified_skill_handoff(
        instruction="Give me a comma-separated list of top 4 most played r&b song titles from across my Spotify libraries.",
        skills=[
            {
                "skill_id": "s1",
                "name": "spotify play count",
                "description": "Rank songs by play count.",
                "executor_desc": "apis.spotify.show_song",
            },
            {
                "skill_id": "s2",
                "name": "spotify search songs",
                "description": "Search songs.",
                "executor_desc": "apis.spotify.search_songs",
            },
            {
                "skill_id": "s3",
                "name": "spotify album library",
                "description": "Read albums.",
                "executor_desc": "apis.spotify.show_album_library",
            },
            {
                "skill_id": "s4",
                "name": "spotify playlist library",
                "description": "Read playlists.",
                "executor_desc": "apis.spotify.show_playlist_library",
            },
        ],
        valid_api_refs={
            ("spotify", "show_song"),
            ("spotify", "search_songs"),
            ("spotify", "show_album_library"),
            ("spotify", "show_playlist_library"),
        },
        required_apps={"spotify"},
        visible_skill_limit=2,
    )

    assert handoff["useful_apis"] == ["apis.spotify.search_songs", "apis.spotify.show_song"]
    assert "apis.spotify.show_album_library" not in handoff["prompt_block"]
    assert "apis.spotify.show_playlist_library" not in handoff["prompt_block"]
    assert handoff["visible_skill_ids"] == ["s1", "s2"]
    assert handoff["rescued_action_skill_ids"] == []


def test_verified_handoff_rescues_deep_action_api_from_limited_visible_prefix():
    handoff = build_verified_skill_handoff(
        instruction='Make me a Spotify playlist called "Most Listened Playlist Songs" containing songs.',
        skills=[
            {
                "skill_id": "s1",
                "name": "spotify rank songs",
                "description": "Rank songs.",
                "executor_desc": "apis.spotify.show_song",
            },
            {
                "skill_id": "s2",
                "name": "spotify show playlists",
                "description": "Read playlists.",
                "executor_desc": "apis.spotify.show_playlist",
            },
            {
                "skill_id": "s3",
                "name": "spotify create playlist",
                "description": "Create a playlist.",
                "executor_desc": "apis.spotify.create_playlist",
            },
            {
                "skill_id": "s4",
                "name": "spotify add songs",
                "description": "Add songs to a playlist.",
                "executor_desc": "apis.spotify.add_song_to_playlist",
            },
        ],
        valid_api_refs={
            ("spotify", "show_song"),
            ("spotify", "show_playlist"),
            ("spotify", "create_playlist"),
            ("spotify", "add_song_to_playlist"),
        },
        required_apps={"spotify"},
        visible_skill_limit=2,
    )

    assert "apis.spotify.create_playlist" in handoff["prompt_block"]
    assert "apis.spotify.add_song_to_playlist" in handoff["prompt_block"]
    assert handoff["visible_skill_ids"] == ["s1", "s2"]
    assert handoff["rescued_action_skill_ids"] == ["s3", "s4"]
