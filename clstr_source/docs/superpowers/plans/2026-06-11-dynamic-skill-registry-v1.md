# Dynamic Skill Registry v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLSTR support for appending new skills after checkpoint creation, with coverage-aware fallback for low-supervision skills.

**Architecture:** Extend `SkillTable` and `CLSTRModel` with append APIs that preserve old parameters and initialize new skill rows from text embeddings. Add a small blending helper that downweights CLSTR heads for appended/unseen skills while preserving full weight for old skills.

**Tech Stack:** Python, PyTorch, pytest.

---

### Task 1: SkillTable append API

**Files:**
- Modify: `clstr/encoders.py`
- Test: `tests/test_dynamic_skill_registry.py`

- [ ] Write a failing test that constructs a `SkillTable` with two skills, appends a third skill, and verifies old embeddings/biases are preserved while the new row is initialized and marked as appended.
- [ ] Run `python -m pytest tests/test_dynamic_skill_registry.py::test_skill_table_append_skills_preserves_existing_rows_and_marks_new_rows -q` and verify it fails because `append_skills` does not exist.
- [ ] Implement `SkillTable.append_skills()` with duplicate-skill skipping, metadata defaults, tensor expansion for `E`, `skill_bias_retr`, and `skill_bias_belief`, and a structured report.
- [ ] Run the focused test and verify it passes.

### Task 2: CLSTRModel append wrapper

**Files:**
- Modify: `clstr/model.py`
- Test: `tests/test_dynamic_skill_registry.py`

- [ ] Write a failing test that uses a lightweight fake `CLSTRModel` instance and verifies `append_skills()` keeps `model.skills`, `model.skill_table.skills`, `stop_idx`, and cache state consistent.
- [ ] Run the focused test and verify it fails because `CLSTRModel.append_skills` does not exist.
- [ ] Implement the wrapper by delegating to `skill_table.append_skills()`, copying `skill_table.skills` back to `model.skills`, and clearing the cross-encoder cache.
- [ ] Run the focused test and verify it passes.

### Task 3: Coverage-aware blending helper

**Files:**
- Create: `clstr/dynamic_skill_registry.py`
- Test: `tests/test_dynamic_skill_registry.py`

- [ ] Write failing tests for appended/unseen skill weights and old-skill default full weights.
- [ ] Run focused tests and verify they fail because the helper does not exist.
- [ ] Implement `skill_head_coverage_weights()` and `blend_skill_scores_with_coverage()`.
- [ ] Run focused tests and verify they pass.

### Task 4: Verification and documentation

**Files:**
- Modify: `description.md`
- Modify: `docs/clstr_v4_plan.md`

- [ ] Run `python -m pytest tests/test_dynamic_skill_registry.py tests/test_v4_action_proj_sharing.py tests/test_stage_checkpoint_init.py -q`.
- [ ] Run `python -m py_compile clstr/encoders.py clstr/model.py clstr/dynamic_skill_registry.py`.
- [ ] Run `git diff --check -- clstr/encoders.py clstr/model.py clstr/dynamic_skill_registry.py tests/test_dynamic_skill_registry.py description.md docs/clstr_v4_plan.md`.
- [ ] Document that dynamic append is supported for text retrieval immediately, while multi-step CLSTR heads remain coverage-aware for unseen skills.
