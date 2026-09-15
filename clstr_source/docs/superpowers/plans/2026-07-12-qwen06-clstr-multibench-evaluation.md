# Qwen 0.6B CLSTR Multi-Benchmark Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve one final Qwen CLSTR checkpoint chain, evaluate it on frozen ToolBench-G3/tau2/ToolSandbox/ALFWorld routing inputs, run CLSTR-native causal evaluations and official ALFWorld closed loop, and aggregate paper-usable evidence.

**Architecture:** Add a focused frozen current-state route evaluator and an immutable checkpoint-chain resolver. Reuse existing native CLSTR evaluators through a manifest-driven smoke/full submitter; keep comparable routing and task-success results separate.

**Tech Stack:** Python 3.11, PyTorch, pytest, Bash/Slurm, JSON/JSONL manifests, existing CLSTR checkpoint loaders and ALFWorld harness.

---

## File Structure

- Create `clstr/qwen_clstr_final_chain.py`: validate and resolve Stage0/1/2/4 gates, lineages, checkpoint digests, and frozen-backbone identity.
- Create `clstr/frozen_clstr_route_eval.py`: validate frozen corpora, adapt rows, expand the skill table safely, score declared candidates, and write strict metrics/predictions.
- Create `clstr/qwen_clstr_multibench_submit.py`: deterministic smoke/full stage plans and resumable job registry.
- Create `clstr/qwen_clstr_multibench_report.py`: validate outputs and aggregate routing versus task-success sections.
- Create `scripts/resolve_qwen06_clstr_final_chain.py`: final-chain CLI.
- Create `scripts/run_frozen_clstr_route_eval.py`: frozen-routing CLI.
- Create `scripts/submit_qwen06_clstr_multibench.py`: smoke/full submission CLI.
- Create `scripts/build_qwen06_clstr_multibench_report.py`: final aggregation CLI.
- Create `scripts/sbatch/run_qwen06_clstr_frozen_route_eval.sh`: one frozen benchmark route job.
- Create `scripts/sbatch/run_qwen06_clstr_native_route_eval.sh`: dispatch existing ToolBench/tau2/ToolSandbox native evaluators.
- Create `scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh`: official ALFWorld seen/unseen job.
- Create focused tests under `tests/test_qwen_clstr_{final_chain,frozen_route_eval,multibench_submit,multibench_report}.py`.

### Task 1: Immutable final checkpoint-chain resolution

**Files:**
- Create: `tests/test_qwen_clstr_final_chain.py`
- Create: `clstr/qwen_clstr_final_chain.py`
- Create: `scripts/resolve_qwen06_clstr_final_chain.py`

- [x] **Step 1: Write a failing test for one valid Stage0/1/2/4 chain**

The test creates minimal checkpoint files, `status=ok` gates, and linked lineage JSON, then asserts:

```python
manifest = resolve_qwen_clstr_final_chain(run_root)
assert manifest["status"] == "ok"
assert manifest["checkpoint_chain_digest"]
assert manifest["checkpoints"]["stage4"]["path"].endswith("clstr_stage4_act-step3000.pt")
```

- [x] **Step 2: Run the test and verify RED from the missing resolver**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_clstr_final_chain.py`

- [x] **Step 3: Implement strict gate, lineage, parent, path, digest, and frozen-backbone validation**

The resolver must require the exact Qwen run layout and use streaming SHA-256 for checkpoint and JSONL files. It must reject a Stage4 lineage whose identity parent is not Stage2 or whose parent digests differ.

- [x] **Step 4: Add rejection tests for a bad gate, wrong Stage4 identity parent, changed checkpoint, and `freeze_backbone=false`**
- [x] **Step 5: Add the CLI and run focused GREEN tests**
- [x] **Step 6: Commit**

### Task 2: Frozen current-state route evaluator

**Files:**
- Create: `tests/test_qwen_clstr_frozen_route_eval.py`
- Create: `clstr/frozen_clstr_route_eval.py`
- Create: `scripts/run_frozen_clstr_route_eval.py`

- [x] **Step 1: Write RED tests for frozen-row adaptation**

Assert `raw_state -> state_text`, `positive_skill_id -> next_skill_id`, row candidates become legal inventory, ToolBench uses the complete frozen pool, and no replay prefix is fabricated.

- [x] **Step 2: Write RED tests for strict ranking metrics and complete predictions**

Use a small deterministic model to verify Recall@1/5, MRR, positive rank, declared candidate count, strict source denominator, and per-row top-K IDs/scores.

- [x] **Step 3: Implement manifest self-hash/count/positive-coverage validation and row adaptation**
- [x] **Step 4: Implement base-prefix skill merging and safe unseen-skill expansion**
- [x] **Step 5: Implement checkpoint-prompted batched scoring over declared candidates**

For every row, score `unified_route_logits(h, initial_belief(h), candidate_rows=...)`. Record `memory_active_rows=0` and `zero_history_fallback=exact_static`; do not call transition replay on frozen rows.

- [x] **Step 6: Write atomic report, identity, and prediction artifacts with resume validation**
- [x] **Step 7: Run focused GREEN tests and commit**

### Task 3: Smoke/full Slurm orchestration

**Files:**
- Create: `tests/test_qwen_clstr_multibench_submit.py`
- Create: `clstr/qwen_clstr_multibench_submit.py`
- Create: `scripts/submit_qwen06_clstr_multibench.py`
- Create: three Qwen CLSTR sbatch wrappers listed above.

- [x] **Step 1: Write a RED test for deterministic smoke stage order**

Expected order:

```text
frozen/toolbench_g3
frozen/toolsandbox
frozen/tau2
frozen/alfworld_offline
native/toolbench_g3
native/toolsandbox
native/tau2
closed_loop/alfworld_valid_seen
closed_loop/alfworld_valid_unseen
```

- [x] **Step 2: Require one final-chain digest and pinned corpus manifest SHA per stage**
- [x] **Step 3: Implement resumable submission registry and dry-run output**
- [x] **Step 4: Implement wrappers that restore `PROJECT_ROOT` after sourcing the shared GPU environment**
- [x] **Step 5: Add shell syntax and launcher-contract tests**
- [x] **Step 6: Run GREEN tests and commit**

### Task 4: Structural smoke gate

**Files:**
- Create: `tests/test_qwen_clstr_multibench_report.py`
- Create: `clstr/qwen_clstr_multibench_report.py`
- Create: `scripts/build_qwen06_clstr_multibench_report.py`

- [x] **Step 1: Write RED tests that reject incomplete predictions, denominator drift, non-finite scores, checkpoint drift, and mislabeled routing success**
- [x] **Step 2: Implement smoke validation with separate `routing` and `task_success` sections**
- [x] **Step 3: Run CPU tests, compile, shell syntax, and diff checks**
- [ ] **Step 4: Submit bounded smoke jobs only after Stage4 quality gate is `ok`**
- [ ] **Step 5: Supervise all smoke jobs to terminal states and build a self-hashed smoke gate**
- [ ] **Step 6: Commit compact smoke evidence**

### Task 5: Full evaluation and aggregation

**Files:**
- Use the implemented submitter/report builder; write outputs under the final CLSTR run root.

- [ ] **Step 1: Submit full jobs using the accepted smoke-gate SHA**
- [ ] **Step 2: Require exact full denominators: ToolBench 1,362, tau2 13,907, ToolSandbox 115, ALFWorld offline 5,858, and all official ALFWorld seen/unseen episodes**
- [ ] **Step 3: Supervise every job to `COMPLETED|0:0` or record a distinct failure artifact**
- [ ] **Step 4: Build `comparison_report.json` and `comparison_table.md`**
- [ ] **Step 5: Verify checkpoint/corpus digests, report self-hash, and paper labels**
- [ ] **Step 6: Commit only compact code, manifests, and summary evidence; do not commit large predictions or caches**

## Self-Review

- Spec coverage: final-chain identity, frozen comparable routing, native causal routing, ALFWorld success, smoke/full gating, strict denominators, and aggregation each map to a task.
- Placeholder scan: every implementation stage has explicit files, expected behavior, and verification commands; no benchmark choice remains implicit.
- Type consistency: `checkpoint_chain_digest`, frozen `manifest_sha256`, `memory_active_rows`, and routing/task-success section names are consistent throughout.
