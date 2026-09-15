# Hybrid Verified SkillX Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or equivalent task-by-task execution. This plan is scoped to AppWorld executor handoff only; no full GPU jobs are part of implementation.

**Goal:** Add `skill_context_mode=verified_hints`, where CLSTR-selected SkillX rows are verified and compressed into low-priority executor hints instead of being exposed as raw SkillX metadata.

**Architecture:** CLSTR still selects skills. A deterministic schema guard verifies app, operation, state-changing API, and pollution risks. An optional reference-model scorer interface can provide semantic scores, but v1 defaults to deterministic scoring so local tests and CPU diagnostics are stable. The compressor emits only schema-valid APIs and short workflow primitives; raw SkillX id/name/body remain internal diagnostics.

**Tech Stack:** Python, pytest, existing AppWorld API schema utilities, existing Qwen executor path. No training or full sbatch job in this plan.

---

## Files

- Create: `clstr/appworld_skill_handoff.py`
  - Owns verifier decisions, deterministic semantic scoring, compressed hint formatting, and diagnostics.
- Modify: `clstr/appworld_executor.py`
  - Adds `verified_hints` mode and delegates handoff decisions to the new module.
- Modify: `tests/test_appworld_executor.py`
  - Adds RED/GREEN tests for unique-count, queue mutation, simple-note export, and raw SkillX hiding.
- Modify: `tests/test_appworld_multistep.py`
  - Adds a diagnostics persistence test for multistep `verified_hints`.
- Modify: `tests/test_appworld_multistep_cli.py`
  - Adds CLI mode availability test.
- Modify: `description.md`, `.planning/2026-06-11-dynamic-skill-registry/progress.md`
  - Records the design boundary and gate rule.

## Tasks

### Task 1: RED tests

- [ ] Add prompt tests showing `verified_hints` is unsupported.
- [ ] Run targeted tests and confirm they fail for missing mode / behavior.

### Task 2: Handoff module

- [ ] Implement deterministic verifier:
  - `inject_hint` for operation/app/schema-aligned skills.
  - `schema_only` for weak but usable evidence.
  - `suppress` for no schema evidence or harmful operation mismatch.
- [ ] Implement compressor:
  - Emits `[Optional Retrieved Hints]`.
  - Includes `Potentially useful APIs`, `Possible workflow hints`, and `Suppressed retrieved hints`.
  - Never includes `skill_id`, `name`, `description`, `body`, or `skill_md` in prompt text.
- [ ] Include a reference-model scorer interface but keep v1 default deterministic.

### Task 3: Prompt integration

- [ ] Add `verified_hints` to `VALID_SKILL_CONTEXT_MODES`.
- [ ] In `build_appworld_executor_prompt()`, call the new module when mode is `verified_hints`.
- [ ] Preserve full diagnostics in `diagnostics_out`.
- [ ] Add neutral prompt rules; avoid `CLSTR`/`Prioritized` wording.

### Task 4: Verification

- [ ] Run targeted tests:
  - `tests/test_appworld_executor.py`
  - `tests/test_appworld_multistep.py`
  - `tests/test_appworld_multistep_cli.py`
- [ ] Run `py_compile` on affected modules.
- [ ] Run `git diff --check`.

### Task 5: Low-cost result gate

- [ ] If local tests pass, run a CPU-only handoff audit if available or add one before any GPU job.
- [ ] Only after diagnostics look correct, run one regression6 sbatch gate with `SKILL_CONTEXT_MODE=verified_hints`.
- [ ] Stop unless regression6 reaches at least `5/6` with zero preflight failures.
