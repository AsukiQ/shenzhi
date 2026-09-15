# Structured SkillX Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a schema-grounded SkillX handoff mode that lets CLSTR provide AppWorld executor constraints without exposing misleading SkillX titles/metadata.

**Architecture:** Extend the existing AppWorld prompt builder with a new `skill_context_mode="schema_plan"`. The mode uses full AppWorld schema refs already loaded for preflight, formats only valid executable API constraints, and drops skills with no valid schema evidence.

**Tech Stack:** Python, pytest, AppWorld JSON API docs, existing CLSTR AppWorld executor modules.

---

### Task 1: Add Prompt-Level Tests

**Files:**
- Modify: `tests/test_appworld_executor.py`
- Modify: `tests/test_appworld_multistep.py`

- [ ] **Step 1: Add executor prompt test**

Add a test that builds a `schema_plan` prompt with a misleading SkillX title such as `spotify-like-songs-by-source`, where the skill body references both valid and invalid APIs. Assert:

```python
assert "skillx/appworld/spotify-like-songs-by-source" not in prompt
assert "spotify like songs by source" not in prompt
assert "valid_appworld_apis:" in prompt
assert "apis.spotify.show_song_library" in prompt
assert "apis.spotify.like_song" not in prompt
assert "raw_body_omitted: true" in prompt
```

- [ ] **Step 2: Add multistep full-schema wiring test**

Add a test that calls `build_multistep_executor_prompt(..., skill_context_mode="schema_plan", valid_api_refs={...})` with prompt API docs that omit `show_song`, and assert `show_song` can still appear in the schema-grounded skill constraints because full refs were passed separately.

- [ ] **Step 3: Run red tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py::test_prompt_builder_schema_plan_hides_title_and_filters_action_apis \
  tests/test_appworld_multistep.py::test_multistep_schema_plan_uses_full_schema_refs -q
```

Expected: both fail because `schema_plan` is unsupported.

### Task 2: Implement `schema_plan`

**Files:**
- Modify: `clstr/appworld_executor.py`
- Modify: `clstr/appworld_multistep.py`
- Modify: `scripts/sbatch/run_appworld_multistep_executor_eval.sh`

- [ ] **Step 1: Extend valid modes**

Change:

```python
VALID_SKILL_CONTEXT_MODES = {"raw", "metadata", "safe_metadata"}
```

to include:

```python
"schema_plan"
```

- [ ] **Step 2: Add schema-plan helpers**

Add helpers near `_format_skill()`:

```python
def _api_name_tokens(api_name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(api_name).lower()))

def _is_state_changing_api_name(api_name: str) -> bool:
    return str(api_name).startswith(("add_", "archive", "batch_", "create_", "delete_", "like_", "move_", "remove_", "rename_", "send_", "set_", "share_", "update_", "write_"))

def _schema_plan_refs(skill: dict[str, Any], valid_api_refs: set[tuple[str, str]], instruction: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]], int]:
    ...
```

The helper must keep all valid read/support refs and keep state-changing refs only when the API verb appears in the user-goal tokens.

- [ ] **Step 3: Format schema-plan blocks**

In `_format_skill()`, for `mode == "schema_plan"`, return a block that contains:

```text
[Reference Skill N - schema-grounded constraints]
not_callable: true
schema_grounded: true
valid_appworld_apis: ...
read_support_apis: ...
state_changing_action_apis: ...
filtered_state_changing_api_count: ...
raw_body_omitted: true
```

Do not include `skill_id`, `name`, `description`, or raw body.

- [ ] **Step 4: Wire full valid refs into prompt construction**

Add optional `valid_api_refs` to `build_appworld_executor_prompt()` and `build_multistep_executor_prompt()`. In multistep eval, pass the full schema refs already loaded by `load_appworld_api_refs()` to both prompt construction and preflight.

- [ ] **Step 5: Run green tests**

Run the two tests from Task 1. Expected: pass.

### Task 3: Verification and Low-Cost Gate

**Files:**
- Modify: `.planning/2026-06-11-dynamic-skill-registry/progress.md`
- Modify: `description.md`

- [ ] **Step 1: Run focused suite**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_phase1_prior_residual_defaults.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_executor.py clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run regression6 sbatch only if tests pass**

Submit exactly one small job:

```bash
OUTPUT_DIR="outputs/appworld_dynamic_multistep_executor_dev57/regression6_stage4_schema_plan_<timestamp>" \
TASKS_PATH="data/appworld_multistep/dev57_stage4_regression6_dynamic.jsonl" \
MAX_TASKS=6 MAX_STEPS=3 TOP_K=5 \
CLSTR_CHECKPOINT_PATH="outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt" \
RANKING_MODE=transition_blend POLICY_BLEND_ALPHA=0.25 \
TRANSITION_SCORING_MODE=v4_1b_action_observation TRANSITION_RESIDUAL_LAMBDA=0.0 \
CANDIDATE_TOP_K=350 CANDIDATE_SOURCE=routing \
SKILL_CONTEXT_MODE=schema_plan APPWORLD_EXECUTOR_COMPATIBLE_ONLY=1 \
MODEL_NAME_OR_PATH=models/Qwen3-8B \
sbatch --time=00:45:00 --gpus=1 -p gpu_a800 scripts/sbatch/run_appworld_dynamic_multistep_executor_smoke.sh
```

- [ ] **Step 3: Gate decision**

If regression6 remains `0/6`, stop and inspect row-level failures before changing anything else. If it improves and has no preflight failures, proceed later to dev10, not dev57.
