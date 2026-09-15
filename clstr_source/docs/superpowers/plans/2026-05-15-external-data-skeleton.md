# CLSTR External Data Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SkillsBench / ALFWorld data-status checks, Stage0 audit overlap checks, and clean_router_data skeleton generation without leaking SkillRouter eval labels into training.

**Architecture:** External benchmark roots are inspected through a small CLSTR module that emits explicit `ok`, `partial`, or `missing` statuses. Stage0 audit consumes split ids and reports overlap violations. clean_router_data generation is manifest-first: it refuses to produce formal train files unless SkillsBench trajectories, skill pool, harness results, and audit exclusions are present.

**Tech Stack:** Python 3.11, pytest, existing CLSTR JSONL/TOML helpers via standard library, shell wrappers under `scripts/`.

---

### Task 1: Tests First

**Files:**
- Modify: `tests/test_data.py`
- Modify: `tests/test_leakage.py`
- Create: `tests/test_external_data.py`

- [ ] Write failing tests for SkillsBench split generation/checking from task directories.
- [ ] Write failing tests for Stage0 query-overlap reporting.
- [ ] Write failing tests for clean_router_data missing-input manifest behavior.
- [ ] Write failing tests for ALFWorld missing environment status.

### Task 2: Data Skeleton Implementation

**Files:**
- Create: `clstr/external_data.py`
- Modify: `clstr/leakage.py`
- Create: `scripts/check_skillsbench_data.py`
- Create: `scripts/build_clean_router_data.py`
- Create: `scripts/check_alfworld_data.py`

- [ ] Implement SkillsBench task discovery and deterministic task-level split generation.
- [ ] Implement ALFWorld root/package/env checks with missing status.
- [ ] Extend Stage0 audit with train/dev/test overlap checks against SkillRouter eval query ids.
- [ ] Implement clean_router_data manifest generation that blocks formal files when required inputs are absent.

### Task 3: Docs And Verification

**Files:**
- Modify: `README.md`

- [ ] Document SkillsBench data check, audit rerun, clean_router_data build/check, and ALFWorld check commands.
- [ ] Run `pytest -q tests/test_data.py tests/test_leakage.py tests/test_train_smoke.py`.
- [ ] Run full `pytest -q`.
- [ ] Report external data status and whether AutoDL network_turbo was used.
