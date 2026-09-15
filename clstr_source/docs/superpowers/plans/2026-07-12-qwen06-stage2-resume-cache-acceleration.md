# Qwen06 Stage2 Resume Cache Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume Stage2 from step 2000 to the original total step 10000 while enabling the frozen-Qwen embedding cache.

**Architecture:** Keep `max_steps=10000` as the original data/sampling schedule and add an optional absolute `target_total_steps`. The training loop derives remaining steps from the restored checkpoint; launchers forward this target and allow an explicit cache-mode override.

**Tech Stack:** Python 3.10, PyTorch, pytest, Bash, Slurm.

---

### Task 1: Absolute resume target

**Files:**
- Modify: `tests/test_full_base_train.py`
- Modify: `clstr/full_base_train.py`

- [ ] **Step 1: Write the failing resume test**

Update the existing resume test so the second run passes `max_steps=3` and `target_total_steps=3`, then assert `trained_steps_this_run == 2`, `final_step == 3`, optimizer restoration, and metric steps `[2, 3]`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_full_base_train.py::test_train_clstr_full_base_resume_checkpoint_continues_step_optimizer_and_counterfactual_warmup`

Expected: FAIL because `target_total_steps` is not accepted.

- [ ] **Step 3: Implement the minimal training-loop behavior**

Add `target_total_steps: int | None = None` to `train_clstr_full_base_with_model()` and `run_clstr_full_base_train()`. When set, require it to exceed `resume_start_step`, set `trained_steps_this_run = target_total_steps - resume_start_step`, and set `final_step = target_total_steps`. Keep `max_steps` unchanged for subset construction and warmup semantics.

- [ ] **Step 4: Verify GREEN**

Run the focused test from Step 2 and expect PASS.

### Task 2: CLI and launcher propagation

**Files:**
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify: `tests/test_qwen06_training_launchers.py`

- [ ] **Step 1: Write failing launcher assertions**

Assert that the Python CLI exposes `--target_total_steps`, the lower launcher forwards `TARGET_TOTAL_STEPS`, and the Qwen06 wrapper uses inherited cache settings rather than overwriting them.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_qwen06_training_launchers.py`

Expected: FAIL on missing target/cache override contracts.

- [ ] **Step 3: Implement propagation**

Add the CLI argument and pass it to `run_clstr_full_base_train()`. Append `--target_total_steps` in the lower launcher when `TARGET_TOTAL_STEPS` is non-empty. In the Qwen06 wrapper, default the full target to 10000 and use `${EMBEDDING_CACHE_MODE:-auto}` and `${EMBEDDING_CACHE_MAX_ROWS:-20000}` before exporting them.

- [ ] **Step 4: Verify GREEN**

Run the launcher test and expect PASS.

### Task 3: Regression and delivery

**Files:**
- Modify: `.planning/2026-07-12-qwen06-multibench-eval-acceleration/progress.md`

- [ ] **Step 1: Run focused regressions**

Run: `pytest -q tests/test_full_base_train.py tests/test_qwen06_training_launchers.py tests/test_sbatch_scripts.py`

Expected: all pass.

- [ ] **Step 2: Commit implementation**

Commit the tested source, tests, design, and plan with a focused Stage2 resume/cache message.

- [ ] **Step 3: Submit replacement Stage2**

Submit the canonical wrapper with `FULL_RUN=1`, the verified resume checkpoint, `TARGET_TOTAL_STEPS=10000`, and `EMBEDDING_CACHE_MODE=always`.

- [ ] **Step 4: Gate before Stage4**

Verify the cache report and resume report, then submit a fresh Stage4 smoke/full dependency chain only after Stage2 succeeds.

