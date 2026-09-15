# Visible Evidence Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate CLSTR internal candidate depth from the compact evidence shown to the executor.

**Architecture:** Keep broad selected/candidate skills for CLSTR diagnostics and Stage4 data, but limit schema/hint aggregation to a compact visible prefix. Rescue action-matched state-changing APIs from deeper candidates so write tasks do not lose required APIs.

**Tech Stack:** Python, pytest, AppWorld official executor scripts, existing `appworld_skill_handoff` compressor.

---

### Task 1: Add Handoff Packing Controls

**Files:**
- Modify: `clstr/appworld_skill_handoff.py`
- Test: `tests/test_appworld_skill_handoff.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `build_verified_skill_handoff(..., visible_skill_limit=2)` and assert:

```python
def test_verified_handoff_limits_read_only_visible_schema_refs():
    handoff = build_verified_skill_handoff(
        instruction="Give me a comma-separated list of top 4 most played r&b song titles from across my Spotify libraries.",
        skills=[
            {"skill_id": "s1", "name": "play count", "description": "rank by play count", "executor_desc": "apis.spotify.show_song"},
            {"skill_id": "s2", "name": "artist search", "description": "search songs", "executor_desc": "apis.spotify.search_songs"},
            {"skill_id": "s3", "name": "album library", "description": "album details", "executor_desc": "apis.spotify.show_album_library"},
            {"skill_id": "s4", "name": "playlist library", "description": "playlist details", "executor_desc": "apis.spotify.show_playlist_library"},
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
    assert handoff["visible_skill_ids"] == ["s1", "s2"]
```

```python
def test_verified_handoff_rescues_deep_action_api_from_limited_visible_prefix():
    handoff = build_verified_skill_handoff(
        instruction='Make me a Spotify playlist called "Most Listened Playlist Songs" containing songs.',
        skills=[
            {"skill_id": "s1", "name": "rank songs", "description": "rank", "executor_desc": "apis.spotify.show_song"},
            {"skill_id": "s2", "name": "show playlists", "description": "read", "executor_desc": "apis.spotify.show_playlist"},
            {"skill_id": "s3", "name": "create playlist", "description": "create", "executor_desc": "apis.spotify.create_playlist"},
            {"skill_id": "s4", "name": "add songs", "description": "add", "executor_desc": "apis.spotify.add_song_to_playlist"},
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
    assert handoff["rescued_action_skill_ids"] == ["s3", "s4"]
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_skill_handoff.py -q
```

Expected: fail because `visible_skill_limit`, `visible_skill_ids`, and `rescued_action_skill_ids` do not exist.

- [ ] **Step 3: Implement minimal packing**

Add `visible_skill_limit: int | None = None` to `build_verified_skill_handoff`. Compute evidence decisions from the visible prefix plus any deeper decision with non-empty `state_changing_action_apis`. Keep `selected_skill_ids` unchanged for diagnostics.

- [ ] **Step 4: Verify green**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_skill_handoff.py
```

Expected: pass.

### Task 2: Wire Runtime Parameter

**Files:**
- Modify: `scripts/run_appworld_official_executor_eval.py`
- Modify: `scripts/sbatch/run_appworld_official_executor_eval.sh`
- Test: `tests/test_appworld_official_executor_cli.py`
- Test: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Write failing tests**

Add CLI/sbatch tests asserting `--handoff_visible_skill_limit` and `HANDOFF_VISIBLE_SKILL_LIMIT` are supported.

- [ ] **Step 2: Implement pass-through**

Thread the value from CLI to `CLSTREvidenceBuilder` and into `build_verified_skill_handoff`.

- [ ] **Step 3: Verify**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_official_executor_cli.py tests/test_sbatch_scripts.py::test_appworld_official_executor_sbatch_normalizes_task_ids_for_sbatch_export
```

Expected: pass.

### Task 3: CPU Regression Verification

**Files:**
- Modify: `.planning/2026-06-14-clstr-next-direction/progress.md`

- [ ] **Step 1: Run focused tests**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_skill_handoff.py tests/test_appworld_official_executor_cli.py tests/test_sbatch_scripts.py::test_appworld_official_executor_sbatch_normalizes_task_ids_for_sbatch_export
```

- [ ] **Step 2: Run syntax checks**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_skill_handoff.py scripts/run_appworld_official_executor_eval.py
bash -n scripts/sbatch/run_appworld_official_executor_eval.sh
git diff --check -- clstr/appworld_skill_handoff.py scripts/run_appworld_official_executor_eval.py scripts/sbatch/run_appworld_official_executor_eval.sh tests/test_appworld_skill_handoff.py tests/test_appworld_official_executor_cli.py tests/test_sbatch_scripts.py
```

- [ ] **Step 3: Stop before GPU**

Do not submit Slurm jobs until CPU tests pass and the user agrees to the small 6-task validation.
