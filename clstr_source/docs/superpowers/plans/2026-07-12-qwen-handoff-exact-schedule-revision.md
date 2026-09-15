# Qwen Handoff Exact-Schedule Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parity-unsafe exact-query-deduplicated compact handoff path with an exact legacy-schedule implementation, verify it on real Qwen, and promote it only if every gate passes.

**Architecture:** Preserve legacy current/next encoder and scorer batch composition and order. Optimize only selected-score transfer, key compact records by the complete legacy schedule context, and recompute a whole schedule batch whenever any row misses.

**Tech Stack:** Python 3.11, PyTorch, pytest, Bash/Slurm, immutable Torch shards, JSON manifests.

---

## File Structure

- Modify `clstr/stage0_handoff_acceleration.py`: schedule-aware row-key plan and global identity.
- Modify `clstr/full_base_train.py`: shared candidate-index selection, vectorized score transfer, exact schedule raw computation, and batch-atomic cache orchestration.
- Modify `tests/test_stage0_handoff_acceleration.py`: global/schedule identity tests.
- Modify `tests/test_clstr_topm_candidate_handoff.py`: score, call-order, and complete-batch reuse tests.
- Modify `docs/superpowers/specs/2026-07-11-qwen-handoff-acceleration-design.md`: mark the original dedup design superseded.
- Modify `docs/superpowers/plans/2026-07-11-qwen-handoff-acceleration.md`: mark unsafe tasks historical.
- Modify Qwen Stage1/2/4 wrappers only after the real-Qwen gate passes.

### Task 1: Lock the exact-schedule behavior with RED tests

**Files:**
- Modify: `tests/test_stage0_handoff_acceleration.py`
- Modify: `tests/test_clstr_topm_candidate_handoff.py`

- [x] **Step 1: Add a test that different encode batch sizes produce different global identities**
- [x] **Step 2: Add a test that row keys change with batch context and duplicate position**
- [x] **Step 3: Add a test that vectorized selected scores exactly equal legacy selected scores**
- [x] **Step 4: Add a test that current and next encoder calls retain the legacy split order**
- [x] **Step 5: Add a test that only complete schedule batches are reused across subsets**
- [x] **Step 6: Run the five tests and confirm they fail because the revision APIs or behavior are absent**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_handoff_acceleration.py::test_global_identity_tracks_legacy_execution_batch_size \
  tests/test_stage0_handoff_acceleration.py::test_scheduled_row_keys_track_batch_context_and_duplicate_position \
  tests/test_clstr_topm_candidate_handoff.py::test_vectorized_stage0_score_transfer_matches_legacy_inventory_topk_exactly \
  tests/test_clstr_topm_candidate_handoff.py::test_exact_schedule_raw_candidates_preserve_legacy_calls_and_outputs \
  tests/test_clstr_topm_candidate_handoff.py::test_row_sharded_handoff_reuses_only_complete_legacy_schedule_batches
```

Expected: RED failures from missing APIs and the old content-only/deduplicated behavior.

### Task 2: Implement the exact schedule and schedule-aware keys

**Files:**
- Modify: `clstr/stage0_handoff_acceleration.py`
- Modify: `clstr/full_base_train.py`

- [x] **Step 1: Add `LegacyScheduleRowKeyPlan` and `build_legacy_schedule_row_key_plan()`**
- [x] **Step 2: Add encode batch size and `legacy_split_batches_v1` to the global identity**
- [x] **Step 3: Extract shared inventory-aware candidate-index selection**
- [x] **Step 4: Keep the legacy scalar-score function and add the vectorized-score function**
- [x] **Step 5: Add `_compute_stage0_raw_candidates_exact_schedule()` with current encode/score followed by next encode/score**
- [x] **Step 6: Make the compact cache reuse or recompute complete schedule batches**
- [x] **Step 7: Run the five focused tests and confirm GREEN**

Expected: all five tests pass without changing the legacy label-aware post-processing loop.

### Task 3: Remove the disproven implementation surface

**Files:**
- Modify: `clstr/stage0_handoff_acceleration.py`
- Modify: `clstr/full_base_train.py`
- Modify: `tests/test_stage0_handoff_acceleration.py`
- Modify: `tests/test_clstr_topm_candidate_handoff.py`
- Delete: `scripts/diagnose_qwen06_handoff_parity.py`

- [x] **Step 1: Delete `OrderedQueryPlan`, `build_ordered_query_plan()`, and the content-only row key**
- [x] **Step 2: Delete `_compute_stage0_raw_candidates_deduplicated()` and its tests**
- [x] **Step 3: Delete the one-off diagnostic script that imports the unsafe helper**
- [x] **Step 4: Run the cache and handoff tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_handoff_acceleration.py \
  tests/test_clstr_topm_candidate_handoff.py
```

Expected: 49 tests pass.

### Task 4: Run complete CPU verification

**Files:**
- Verify only.

- [x] **Step 1: Run the focused integration suite**
- [x] **Step 2: Compile Python sources and check all relevant shell launchers**
- [x] **Step 3: Run `git diff --check` and inspect the complete diff**
- [x] **Step 4: Commit the exact-schedule revision (`c8d1cfc`)**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_handoff_acceleration.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_full_base_train.py \
  tests/test_stage12_consolidated_train.py \
  tests/test_sbatch_scripts.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
bash -n scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh
bash -n scripts/sbatch/run_clstr_unified_stage1_heads_init.sh
bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh
git diff --check
```

Expected: all commands exit 0.

### Task 5: Re-run the real-Qwen gate and promote conditionally

**Files:**
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage1_train.sh`
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`

- [x] **Step 1: Submit the 2,048-row gate through Slurm (`110509`)**

```bash
sbatch --parsable \
  --export=ALL,PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel \
  scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh
```

- [x] **Step 2: Supervise the job to `COMPLETED|0:0` and inspect the JSON report**
- [x] **Step 3: Require exact candidate IDs/order, scores within `1e-5`, identical masks/decisions, zero trainable Qwen parameters, cold speedup at least 1.25x, warm speedup at least 5x, and compact storage at most 25% of legacy**
- [x] **Step 4: Promote only selected batch 16 and the shared cache root**
- [x] **Step 5: Re-run wrapper tests and shell syntax after promotion**

## Self-Review

- Spec coverage: exact call order, vectorized scores, schedule identity, batch-atomic reuse, unsafe-code removal, CPU verification, GPU gate, and guarded promotion are each mapped to a task.
- Placeholder scan: no deferred implementation instructions or ambiguous test commands remain.
- Type consistency: `LegacyScheduleRowKeyPlan`, `RawStage0Candidates`, `encode_batch_size`, and `execution_schedule_version` match the implemented APIs.
