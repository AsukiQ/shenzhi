import json
import subprocess
import sys
from pathlib import Path

from clstr.appworld_current_route import build_appworld_current_route_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_sample_inputs(root: Path) -> None:
    _write_jsonl(
        root / "data/appworld_skill_pool/skill_pool.jsonl",
        [
            {
                "skill_id": "skillx/appworld/spotify-auth-0",
                "name": "spotify authenticate",
                "description": "Log in to Spotify.",
                "executor_desc": "apis.spotify.login",
                "body": "access_token = apis.spotify.login(...)",
            },
            {
                "skill_id": "skillx/appworld/spotify-library-1",
                "name": "spotify list library",
                "description": "List Spotify songs.",
                "executor_desc": "apis.spotify.show_song_library",
                "body": "songs = apis.spotify.show_song_library(...)",
            },
            {
                "skill_id": "skillx/appworld/spotify-song-2",
                "name": "spotify inspect song",
                "description": "Inspect a Spotify song.",
                "executor_desc": "apis.spotify.show_song",
                "body": "song = apis.spotify.show_song(...)",
            },
        ],
    )
    _write_jsonl(
        root / "data/appworld_routing/train_tasks.jsonl",
        [
            {
                "task_id": "train-task",
                "query_id": "train-task",
                "split": "train",
                "instruction_text": "Find the most played Spotify song.",
                "query": "Instruction: Find the most played Spotify song.",
                "required_apps": ["spotify"],
                "positive_skill_ids": ["skillx/appworld/spotify-auth-0", "skillx/appworld/spotify-library-1"],
            },
            {
                "task_id": "dev-leak",
                "query_id": "dev-leak",
                "split": "dev",
                "query": "must not enter training",
                "positive_skill_ids": ["skillx/appworld/spotify-auth-0"],
            },
        ],
    )
    _write_jsonl(
        root / "data/appworld_routing/train_qrels.jsonl",
        [
            {"query_id": "train-task", "skill_id": "skillx/appworld/spotify-auth-0", "relevance": 1, "split": "train"},
            {"query_id": "train-task", "skill_id": "skillx/appworld/spotify-library-1", "relevance": 1, "split": "train"},
            {"query_id": "dev-leak", "skill_id": "skillx/appworld/spotify-auth-0", "relevance": 1, "split": "dev"},
        ],
    )
    _write_jsonl(
        root / "data/appworld_routing/train_replay.jsonl",
        [
            {
                "task_id": "train-task",
                "split": "train",
                "instruction_text": "Find the most played Spotify song.",
                "query": "Instruction: Find the most played Spotify song.",
                "steps": [
                    {
                        "skill_id": "skillx/appworld/spotify-auth-0",
                        "skill_name": "spotify authenticate",
                        "observation": "AppWorld oracle API call: spotify.login POST /spotify/auth/token",
                    },
                    {
                        "skill_id": "skillx/appworld/spotify-library-1",
                        "skill_name": "spotify list library",
                        "observation": "AppWorld oracle API call: spotify.show_song_library GET /spotify/songs",
                    },
                ],
                "success": True,
                "reward": 1.0,
            }
        ],
    )
    _write_jsonl(
        root / "data/appworld_act/verified_pairs_train.jsonl",
        [
            {
                "task_id": "train-task",
                "step_idx": 0,
                "action_at_t": 0,
                "a_next_plus": 2,
                "candidates_next": [0, 2, 1],
                "obs_at_t": "AppWorld oracle API call: spotify.show_song GET /spotify/songs/1",
                "state_before": {
                    "query": "Instruction: Find the most played Spotify song.",
                    "history": [["spotify authenticate", "logged in"]],
                    "observation": "logged in",
                    "artifact": {"split": "train"},
                },
                "was_in_raw_topk": True,
                "replay_prefix": [],
            }
        ],
    )


def test_build_appworld_current_route_dataset_maps_routing_replay_and_verified_pairs(tmp_path):
    _write_sample_inputs(tmp_path)

    manifest = build_appworld_current_route_dataset(
        repo_root=tmp_path,
        output_dir="data/clstr_appworld_current_route_v1",
    )

    out_dir = tmp_path / "data/clstr_appworld_current_route_v1"
    retrieval = _read_jsonl(out_dir / "retrieval.jsonl")
    trajectories = _read_jsonl(out_dir / "trajectories.jsonl")
    skills = _read_jsonl(out_dir / "skill_pool.jsonl")

    assert manifest["status"] == "ok"
    assert manifest["skill_pool"]["skill_count"] == 3
    assert {row["query_id"] for row in retrieval} >= {
        "train-task",
        "appworld-trajectory::train-task::0::current",
        "appworld-trajectory::train-task::0::next",
        "appworld-verified-pair::train-task::0::next",
    }
    assert all(row.get("split") == "train" for row in retrieval)
    assert all(row["query_id"] != "dev-leak" for row in retrieval)
    assert all(row["positive_skill_id"] != "dev-leak" for row in retrieval)
    assert all(row["canonical_skill_id"] == row["skill_id"] for row in skills)

    replay_row = next(row for row in trajectories if row["provenance"]["source_id"] == "appworld_train_replay")
    assert replay_row["benchmark"] == "appworld"
    assert replay_row["skill_id"] == "skillx/appworld/spotify-auth-0"
    assert replay_row["next_skill_id"] == "skillx/appworld/spotify-library-1"
    assert replay_row["loss_mask"]["L_policy"] is True
    assert replay_row["loss_mask"]["L_trans_skill_ce"] is True

    verified_row = next(row for row in trajectories if row["provenance"]["source_id"] == "appworld_verified_pairs_train")
    assert verified_row["candidate_next_skill_ids"] == [
        "skillx/appworld/spotify-auth-0",
        "skillx/appworld/spotify-song-2",
        "skillx/appworld/spotify-library-1",
    ]
    assert verified_row["next_skill_id"] == "skillx/appworld/spotify-song-2"
    assert verified_row["candidate_source"] == "appworld_verified_pair_topk"
    assert verified_row["loss_mask"]["L_trans_skill_ce"] is True


def test_build_appworld_current_route_cli_writes_manifest(tmp_path):
    _write_sample_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_clstr_appworld_current_route.py",
            "--repo_root",
            str(tmp_path),
            "--output_dir",
            "data/clstr_appworld_current_route_v1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "data/clstr_appworld_current_route_v1/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert "clstr_appworld_current_route_v1" in result.stdout


def test_build_appworld_current_route_can_add_large_pool_distractors(tmp_path):
    _write_sample_inputs(tmp_path)
    _write_jsonl(
        tmp_path / "data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl",
        [
            {
                "skill_id": "toolbench-g3/weather.lookup",
                "name": "weather lookup",
                "description": "A non-AppWorld tool skill that should be a routing distractor only.",
            }
        ],
    )

    manifest = build_appworld_current_route_dataset(
        repo_root=tmp_path,
        output_dir="data/clstr_appworld_largepool_current_route_v1",
        extra_skill_pool_paths=["data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl"],
    )

    out_dir = tmp_path / "data/clstr_appworld_largepool_current_route_v1"
    skills = _read_jsonl(out_dir / "skill_pool.jsonl")
    retrieval = _read_jsonl(out_dir / "retrieval.jsonl")
    trajectories = _read_jsonl(out_dir / "trajectories.jsonl")

    assert manifest["skill_pool"]["skill_count"] == 4
    assert manifest["skill_pool"]["extra_skill_pool_count"] == 1
    distractor = next(row for row in skills if row["skill_id"] == "toolbench-g3/weather.lookup")
    assert distractor["executor_domain"] == "non_appworld_distractor"
    assert distractor["appworld_executor_compatible"] is False
    assert all(str(row["positive_skill_id"]).startswith("skillx/appworld/") for row in retrieval)


def test_build_appworld_current_route_can_append_appworld_after_checkpoint_base_pool(tmp_path):
    _write_sample_inputs(tmp_path)
    _write_jsonl(
        tmp_path / "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
        [
            {
                "skill_id": "toolbench/search",
                "name": "toolbench search",
                "executor_domain": "toolbench",
            },
            {
                "skill_id": "traject/open-tab",
                "name": "open browser tab",
                "executor_domain": "traject",
            },
        ],
    )

    manifest = build_appworld_current_route_dataset(
        repo_root=tmp_path,
        output_dir="data/clstr_appworld_dynamic_v4_1b_append",
        base_skill_pool_path="data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
    )

    out_dir = tmp_path / "data/clstr_appworld_dynamic_v4_1b_append"
    skills = _read_jsonl(out_dir / "skill_pool.jsonl")
    retrieval = _read_jsonl(out_dir / "retrieval.jsonl")
    trajectories = _read_jsonl(out_dir / "trajectories.jsonl")

    assert manifest["skill_pool"]["skill_count"] == 5
    assert manifest["dynamic_skill_registry"]["pool_order"] == "base_then_appworld_append"
    assert manifest["dynamic_skill_registry"]["checkpoint_prefix_skill_count"] == 2
    assert manifest["dynamic_skill_registry"]["appended_appworld_skill_count"] == 3
    assert [row["skill_id"] for row in skills[:2]] == ["toolbench/search", "traject/open-tab"]
    assert skills[0]["appworld_executor_compatible"] is False
    assert skills[0]["skill_pool_role"] == "checkpoint_base"
    assert [row["skill_id"] for row in skills[2:]] == [
        "skillx/appworld/spotify-auth-0",
        "skillx/appworld/spotify-library-1",
        "skillx/appworld/spotify-song-2",
    ]
    assert all(row["is_appended_after_checkpoint"] is True for row in skills[2:])
    assert all(row["appworld_executor_compatible"] is True for row in skills[2:])
    assert all(str(row["positive_skill_id"]).startswith("skillx/appworld/") for row in retrieval)
    verified_row = next(row for row in trajectories if row["provenance"]["source_id"] == "appworld_verified_pairs_train")
    assert verified_row["skill_id"] == "skillx/appworld/spotify-auth-0"
    assert verified_row["next_skill_id"] == "skillx/appworld/spotify-song-2"
    assert verified_row["candidate_next_skill_ids"] == [
        "skillx/appworld/spotify-auth-0",
        "skillx/appworld/spotify-song-2",
        "skillx/appworld/spotify-library-1",
    ]
