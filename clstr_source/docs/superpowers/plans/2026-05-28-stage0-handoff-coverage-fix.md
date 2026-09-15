# Stage0 Handoff Coverage Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage2 candidate handoff diagnose and tolerate Stage0 positive misses without hiding true Stage0 recall.

**Architecture:** Add a Stage0 handoff coverage audit path that evaluates current/next positive coverage under multiple query modes. Update Stage2 handoff to use SkillRouter-style sequential queries by default, inject missing current positives with provenance for supervised training, and mask next-skill CE when next positives are absent instead of dropping the whole row.

**Tech Stack:** Python, PyTorch, pytest, existing CLSTR training monitor and sbatch wrappers.

---

### Task 1: Stage2 Handoff Query And Masking

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Test: `tests/test_clstr_topm_candidate_handoff.py`

- [ ] Add tests proving Stage2 uses SkillRouter-style handoff query text and no longer drops rows solely because `next_skill_id` is missing from Stage0 top-M.
- [ ] Add a handoff query builder with modes `raw_state` and `skillrouter_state`.
- [ ] Change current-positive behavior to allow injection with provenance when policy is `inject`.
- [ ] Change next-positive missing behavior to mask `L_trans_skill_ce` on that row and record `stage0_next_positive_missing_masked`.
- [ ] Extend handoff report with current/next coverage and masking counts.
- [ ] Expose `--stage0_handoff_query_mode` in the Stage2 CLI and sbatch script.

### Task 2: Stage0 Handoff Coverage Audit

**Files:**
- Create: `clstr/stage0_handoff_audit.py`
- Create: `scripts/audit_clstr_stage0_handoff_coverage.py`
- Test: `tests/test_stage0_handoff_audit.py`

- [ ] Add a reusable audit function that loads trajectory rows and skill pool, computes Stage0 top-K over current and next handoff queries, and reports coverage at K values.
- [ ] Support query modes `raw_state` and `skillrouter_state`.
- [ ] Report benchmark buckets and global metrics for current and next positives.
- [ ] Add CLI writing JSON output.

### Task 3: Verification And Documentation

**Files:**
- Modify: `description.md`

- [ ] Run targeted pytest for handoff and audit tests.
- [ ] Run a small local or sbatch smoke with `MAX_ROWS=256`, `MAX_STEPS=5`, no full job.
- [ ] Record changed behavior and smoke findings in `description.md`.
