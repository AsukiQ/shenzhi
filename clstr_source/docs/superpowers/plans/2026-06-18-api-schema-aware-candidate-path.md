# API/Schema-Aware Candidate Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AppWorld routing audits label-space aware and build a current v4.2 dynamic SkillX append pool before any further Stage4 work.

**Architecture:** Keep CLSTR's base skill pool unchanged for trained checkpoints, append AppWorld SkillX rows only through the dynamic registry path, and report absent positives separately from in-pool retrieval misses. Use deterministic file-level audits before GPU jobs.

**Tech Stack:** Python 3.10, JSONL skill pools, pytest, existing CLSTR `appworld_current_route` and `stage0_retrieval_coverage_audit` modules.

---

### Task 1: Pool-Membership-Aware Stage0 Coverage Audit

**Files:**
- Modify: `clstr/stage0_retrieval_coverage_audit.py`
- Modify: `scripts/audit_stage0_retrieval_coverage.py`
- Test: `tests/test_stage0_retrieval_coverage_audit.py`

- [ ] **Step 1: Write the failing test**

Add a test with a skill pool containing only `skill/in-pool`, a retrieval row whose positive is `skill/absent`, and predictions that cannot contain it. Assert:

```python
report = build_stage0_retrieval_coverage_audit(
    retrieval_rows_path=retrieval_path,
    predictions_path=predictions_path,
    output_dir=output_dir,
    k_values=[1],
    skill_pool_path=skill_pool_path,
)
assert report["overall"]["positive_absent_from_pool"] == 1
assert report["overall"]["missing_at_max_k"] == 0
assert report["written_missing_rows"] == 0
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_retrieval_coverage_audit.py::test_stage0_retrieval_coverage_audit_separates_absent_positive_from_topk_miss
```

Expected: failure because `skill_pool_path` is not accepted yet.

- [ ] **Step 3: Implement minimal audit support**

Add helper functions to load skill ids from `skill_pool_path`, add `positive_absent_from_pool` counters to source and overall stats, and skip writing correction rows for absent positives.

- [ ] **Step 4: Add CLI flag**

Add `--skill_pool_path` to `scripts/audit_stage0_retrieval_coverage.py` and pass it through.

- [ ] **Step 5: Verify**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_retrieval_coverage_audit.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/stage0_retrieval_coverage_audit.py scripts/audit_stage0_retrieval_coverage.py
```

Expected: tests pass and compilation exits 0.

### Task 2: Current v4.2 Dynamic AppWorld Pool Build

**Files:**
- Existing: `scripts/build_clstr_appworld_current_route.py`
- Output: `data/clstr_appworld_dynamic_v4_2_nowweak_append/`

- [ ] **Step 1: Build the dynamic pool**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/build_clstr_appworld_current_route.py \
  --output_dir data/clstr_appworld_dynamic_v4_2_nowweak_append \
  --base_skill_pool_path data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl
```

Expected: manifest status `ok`, base rows first, AppWorld SkillX rows appended.

- [ ] **Step 2: Verify target positives are in pool**

Run a small membership script for:

```text
skillx/appworld/spotify-add-songs-to-playlist-24
skillx/appworld/spotify-create-new-playlist-21
skillx/appworld/spotify-add-songs-to-playlist-35
```

Expected: all three return `in_pool=True`.

### Task 3: Low-Cost Routing-Only Decision Point

**Files:**
- Update: `description.md`
- Update: `.planning/2026-06-14-clstr-next-direction/task_plan.md`
- Update: `.planning/2026-06-14-clstr-next-direction/progress.md`

- [ ] **Step 1: Record design and build results**

Add the absent-positive diagnosis and dynamic v4.2 append result to project docs.

- [ ] **Step 2: Decide whether to run Slurm routing-only audit**

If the dynamic pool exists and positives are present, the next Slurm job should be a small routing-only AppWorld dynamic audit. It should not be an executor or full training job.
