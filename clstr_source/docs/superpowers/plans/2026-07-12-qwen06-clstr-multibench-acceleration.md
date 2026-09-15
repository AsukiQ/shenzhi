# Qwen06 CLSTR Multi-Benchmark Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate final Qwen06 CLSTR ToolBench, ToolSandbox, Tau2, and ALFWorld evaluation while preserving benchmark semantics and fail-closed resume behavior.

**Architecture:** Remove provably redundant native model passes, expose explicit benchmark-specific batch profiles, add a standalone semantic parity gate for paired fallback/accelerated smoke outputs, and make vectorized ALFWorld persist only complete batches through atomic replacement. The final checkpoint determines whether the accelerated or fallback profile is used for full evaluation.

**Tech Stack:** Python 3.10, PyTorch, pytest, JSON/JSONL, Bash, Slurm.

---

### Task 1: Remove redundant native evaluator passes

**Files:**

- Modify: `tests/test_toolbench_full_clstr_route_eval.py`
- Modify: `tests/test_toolsandbox_route_eval.py`
- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`

- [ ] **Step 1: Make ToolBench call-count expectations fail**

Update the existing candidate-union test so it distinguishes overall and
grouped evaluators, expects only two overall calls, and verifies the grouped
reports wrap those exact results:

```python
overall_results = [
    {**fake_metrics, "marker": "prior"},
    {**fake_metrics, "marker": "stage4"},
]
overall_calls = []

def fake_overall(_model, _rows, **kwargs):
    overall_calls.append(dict(kwargs))
    return dict(overall_results[len(overall_calls) - 1])

def fail_grouped(*_args, **_kwargs):
    raise AssertionError("single-benchmark ToolBench must not rescore by benchmark")

assert len(overall_calls) == 2
assert report["prior_eval_by_benchmark"] == {"toolbench_g3": overall_results[0]}
assert report["stage4_eval_by_benchmark"] == {"toolbench_g3": overall_results[1]}
```

- [ ] **Step 2: Make ToolSandbox discarded-call expectations fail**

Change the full-eval test to count overall calls and make the grouped evaluator
raise. Assert exactly two overall calls and unchanged report metrics.

- [ ] **Step 3: Run RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_toolsandbox_route_eval.py -q
```

Expected: ToolBench fails because it still invokes the grouped evaluator twice;
ToolSandbox fails because it still invokes it once.

- [ ] **Step 4: Implement the minimal pass removal**

In ToolBench replace both grouped calls with:

```python
prior_eval_by_benchmark = {"toolbench_g3": dict(prior_eval)}
stage4_eval_by_benchmark = {"toolbench_g3": dict(stage4_eval)}
```

In ToolSandbox delete the ignored
`evaluate_logged_online_stage4_rows_by_benchmark(...)` call and its stale
side-effect comment. Remove now-unused grouped-evaluator imports in both files.

- [ ] **Step 5: Run GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_toolsandbox_route_eval.py -q
git diff --check -- clstr/toolbench_full_clstr_route_eval.py \
  clstr/toolsandbox_route_eval.py tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_toolsandbox_route_eval.py
git add clstr/toolbench_full_clstr_route_eval.py clstr/toolsandbox_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py tests/test_toolsandbox_route_eval.py
git commit -m "perf: remove redundant native benchmark passes"
```

### Task 2: Add semantic batch-parity reports

**Files:**

- Create: `clstr/qwen_clstr_eval_batch_parity.py`
- Create: `scripts/compare_qwen06_clstr_eval_batches.py`
- Create: `tests/test_qwen_clstr_eval_batch_parity.py`

- [ ] **Step 1: Write failing frozen parity tests**

Create fallback and accelerated fixture directories with frozen prediction
rows whose execution identities and hashes differ but semantic fields match.
The desired API is:

```python
report = compare_qwen_clstr_eval_batches(
    kind="frozen_tau2",
    fallback_dir=fallback_dir,
    accelerated_dir=accelerated_dir,
)
assert report["status"] == "ok"
assert report["effective_profile"] == "accelerated"
```

Change one `ranked_skill_ids` order and assert `status == "action_required"`,
`effective_profile == "fallback"`, and a `ranked_skill_ids_mismatch` blocker.

- [ ] **Step 2: Write failing native Tau2 and ALFWorld parity tests**

For native Tau2 compare `tau2_ranked_rows.jsonl` candidate order plus
`stage0_prior_report`, `stage0_prior_eval`, `base_eval`, `stage4_eval`, and
`strict`. Permit only a `1e-8` numeric tolerance. For ALFWorld compare ordered
gamefiles, action traces, success/reward/goal-condition/step fields, and
aggregate metrics. Missing accelerated artifacts must select fallback rather
than report success.

- [ ] **Step 3: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_qwen_clstr_eval_batch_parity.py -q
```

Expected: import failure because the parity module does not exist.

- [ ] **Step 4: Implement comparison and self-hashed report**

Implement:

```python
def compare_qwen_clstr_eval_batches(
    *,
    kind: str,
    fallback_dir: str | Path,
    accelerated_dir: str | Path,
) -> dict[str, Any]: ...
```

Supported kinds are `frozen_tau2`, `native_tau2`, and `alfworld`. Return a
deterministic object containing `status`, `blockers`, `kind`, both resolved
directories, semantic row counts, `effective_profile`, and `report_sha256`.
Do not compare execution identity/self-hash fields or raw floating scores.

The CLI accepts `--kind`, `--fallback_dir`, `--accelerated_dir`, and
`--output_path`; it writes the full JSON report and exits zero only when the
accelerated profile passes.

- [ ] **Step 5: Run GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_qwen_clstr_eval_batch_parity.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/qwen_clstr_eval_batch_parity.py \
  scripts/compare_qwen06_clstr_eval_batches.py
git diff --check
git add clstr/qwen_clstr_eval_batch_parity.py \
  scripts/compare_qwen06_clstr_eval_batches.py \
  tests/test_qwen_clstr_eval_batch_parity.py
git commit -m "feat: gate clstr evaluation batch parity"
```

### Task 3: Expose accelerated and fallback batch profiles

**Files:**

- Modify: `clstr/qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_multibench_submit.py`

- [ ] **Step 1: Write failing stage-export tests**

Extend the deterministic smoke-stage test with:

```python
by_key = {stage.key: stage for stage in stages}
assert by_key["frozen/tau2"].exports["BATCH_SIZE"] == "128"
assert by_key["native/tau2"].exports["BATCH_SIZE"] == "64"
assert by_key["native/tau2"].exports["STAGE0_CANDIDATE_BATCH_SIZE"] == "128"
assert by_key["closed_loop/alfworld_valid_seen"].exports["BATCH_SIZE"] == "4"
assert by_key["frozen/toolbench_g3"].exports["BATCH_SIZE"] == "16"
assert by_key["native/toolbench_g3"].exports["BATCH_SIZE"] == "8"
```

Add a config override test setting all accelerated values to their fallback
sizes and assert those exact exports. Assert the native launcher already reads
`STAGE0_CANDIDATE_BATCH_SIZE` and the value is explicit in every stage plan.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_qwen_clstr_multibench_submit.py -q
```

Expected: Tau2 and ALFWorld still export the global 16/8/1 defaults and native
stages omit an explicit Stage0 batch export.

- [ ] **Step 3: Implement validated batch profile resolution**

Add a private resolver accepting optional runtime config:

```python
batch_profile = {
    "frozen_tau2": 128,
    "native_tau2": 64,
    "native_tau2_stage0": 128,
    "alfworld": 4,
}
```

All values must be positive non-boolean integers. Unspecified values use the
accelerated defaults above. Other benchmarks retain frozen 16, native 8, and
native Stage0 16. Export `STAGE0_CANDIDATE_BATCH_SIZE` explicitly for every
native stage so submission identity captures it. A runtime config can select
fallback 16/8/16/1 after a failed parity gate.

- [ ] **Step 4: Run GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_qwen_clstr_multibench_submit.py -q
git diff --check -- clstr/qwen_clstr_multibench_submit.py \
  tests/test_qwen_clstr_multibench_submit.py
git add clstr/qwen_clstr_multibench_submit.py \
  tests/test_qwen_clstr_multibench_submit.py
git commit -m "perf: add benchmark-specific clstr eval batches"
```

### Task 4: Make vectorized ALFWorld resume batch-safe

**Files:**

- Modify: `clstr/alfworld_eval.py`
- Modify: `tests/test_alfworld_eval.py`

- [ ] **Step 1: Add a vectorized fake environment and aligned-resume test**

Create a deterministic fake environment whose `reset()` returns exactly
`batch_size` sequential gamefiles. Run four episodes at batch two, reopen with
six episodes at batch two, and assert the first four rows are preserved and
only the next environment batch is scored.

- [ ] **Step 2: Add fail-closed resume tests**

Add tests that reject:

```python
# incomplete mid-batch file
len(existing_rows) == 1 and batch_size == 2

# identity drift
existing_rows[0]["split"] != requested_split
existing_rows[0]["method"] != run_name
existing_rows[0]["episode_index"] != 0

# environment order drift while replay-skipping
stored_gamefile != reset_gamefile
```

Also prove a complete three-episode run produced with batch two can be reopened
unchanged even though its final batch is partial.

- [ ] **Step 3: Add atomic persistence failure test**

Seed a valid `run.jsonl`, monkeypatch `os.replace` to raise during the next
batch commit, and assert the canonical file still contains only the previously
committed rows and no temporary file remains.

- [ ] **Step 4: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_alfworld_eval.py -k 'closed_loop_eval and (resume or batch or atomic)' -q
```

Expected: nonempty batch-two resume raises the legacy batch-size-one guard and
the atomic failure test observes append semantics.

- [ ] **Step 5: Implement validation, replay verification, and atomic commits**

Add a local atomic JSONL writer using a same-directory temporary file,
`flush()`, `os.fsync()`, and `os.replace()`. Validate loaded rows before
initializing continuation. If the run is incomplete require
`len(run_rows) % batch_size == 0`. Replay one reset per committed batch and
compare the returned gamefiles with the stored chunk. Build all rows from a
new environment batch in memory, extend `run_rows`, then atomically rewrite
the canonical file once per completed batch. Use the same atomic writer for
the final normalization write.

- [ ] **Step 6: Run GREEN, full ALFWorld tests, and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_alfworld_eval.py -q
git diff --check -- clstr/alfworld_eval.py tests/test_alfworld_eval.py
git add clstr/alfworld_eval.py tests/test_alfworld_eval.py
git commit -m "feat: resume batched alfworld evaluation safely"
```

### Task 5: Regression, operational parity, and delivery

**Files:**

- Modify: `.planning/2026-07-12-qwen06-multibench-eval-acceleration/task_plan.md`
- Modify: `.planning/2026-07-12-qwen06-multibench-eval-acceleration/progress.md`

- [ ] **Step 1: Run focused and full regression suites**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_toolsandbox_route_eval.py \
  tests/test_qwen_clstr_eval_batch_parity.py \
  tests/test_qwen_clstr_multibench_submit.py \
  tests/test_qwen_clstr_multibench_report.py \
  tests/test_qwen_clstr_frozen_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_alfworld_eval.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q
bash -n scripts/sbatch/run_qwen06_clstr_frozen_route_eval.sh \
  scripts/sbatch/run_qwen06_clstr_native_route_eval.sh \
  scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh
git diff --check
```

- [ ] **Step 2: Keep training gates ahead of evaluation**

Before any GPU parity job, require Stage2 step 10000 quality/lineage success,
Stage4 smoke success, and Stage4 full success. Record cache use, resume
metadata, throughput, and final checkpoint digests.

- [ ] **Step 3: Run paired final-checkpoint smoke jobs through Slurm**

Use separate fallback and accelerated output directories for:

- frozen Tau2: batch 16 versus 128 on the same 32 rows;
- native Tau2: batch 8/Stage0 16 versus batch 64/Stage0 128 on the same 64
  rows;
- ALFWorld valid-seen and valid-unseen: batch 1 versus 4 on the same five
  episodes.

Run the comparator CLI for each pair and retain its self-hashed report. Record
wall time, GPU peak memory, and effective profile.

- [ ] **Step 4: Select the full profile and submit all four benchmarks**

If every accelerated pair passes, use the accelerated runtime profile. For
each failed pair, override only that benchmark to its fallback batch. Submit
the canonical smoke-to-full multibench chain without altering data, prompt,
candidates, checkpoint, or metrics.

- [ ] **Step 5: Verify, integrate, and record results**

Use `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. The user has already selected
direct integration without a PR: fast-forward `master` only after all tests
and runtime gates pass, preserving unrelated worktrees and changes.

## Execution Choice

The user authorized inline execution, requested no PR, and asked not to pause
benchmark acceleration work once training is stable. Execute this plan in the
current isolated worktree without subagents; training state changes and faults
always preempt implementation work.
