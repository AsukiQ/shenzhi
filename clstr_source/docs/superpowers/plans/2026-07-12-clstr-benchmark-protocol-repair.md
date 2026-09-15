# CLSTR Benchmark Protocol Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align CLSTR native benchmark candidate handling with the already-valid SR and ToolREx protocols and rerun only affected CLSTR evaluations.

**Architecture:** Benchmark-local row builders will mark their rows with `candidate_pool_protocol=benchmark_local`; the shared candidate-recall layer will then resolve the legal pool from `candidate_next_skill_ids` only for those marked rows, while explicit environment inventory fields retain precedence. The current-state route evaluator will construct the static/dynamic union and then retain only the top `final_k` candidates according to the fused score before computing ranking metrics. Global-pool rows remain legal over the full skill table even when they carry Stage0 handoff candidates.

**Tech Stack:** Python, PyTorch, pytest executed in Slurm CPU-mode jobs, Slurm evaluation launchers.

---

### Task 1: Benchmark-local legal pool

**Files:**
- Modify: `clstr/memory_candidate_recall.py`
- Test: `tests/test_memory_candidate_recall.py`

- [ ] Add a failing test where a row has `candidate_next_skill_ids` but no explicit inventory field and assert only those skill IDs are legal.
- [ ] Run the focused test in a Slurm CPU-mode job and verify it fails because the current mask is all-true.
- [ ] Mark ToolSandbox/Tau2 rows as `benchmark_local` and implement an ordered legal-pool resolver that uses `candidate_next_skill_ids` only for marked rows.
- [ ] Run the focused test and existing candidate-recall tests in Slurm and verify they pass.

### Task 2: Effective final candidate budget

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/memory_candidate_recall.py`
- Test: `tests/test_current_state_route_eval.py`
- Test: `tests/test_memory_candidate_recall.py`

- [ ] Add a failing route-eval test with a union larger than `final_k` and assert `stage4_candidate_count == final_k`.
- [ ] Add a deterministic helper test proving fused-score top-k selection preserves stable declared-pool tie order.
- [ ] Run both focused tests in Slurm and verify the pre-fix failures.
- [ ] Implement stable fused top-k truncation of the union before ranking metrics while retaining pre-truncation candidate-recall diagnostics.
- [ ] Run focused route and candidate tests in Slurm and verify they pass.

### Task 3: Evaluation contract tests

**Files:**
- Modify: `tests/test_toolsandbox_route_eval.py`
- Modify: `tests/test_tau2_route_eval.py`
- Modify: `tests/test_toolbench_full_clstr_route_eval.py`

- [ ] Assert ToolSandbox and Tau2 native evaluators pass their row-local pool sizes and produce no candidates outside each row's declared pool.
- [ ] Assert ToolBench remains `known_global` and records an effective candidate count equal to the configured final budget.
- [ ] Run the three focused files in a Slurm CPU-mode job.

### Task 4: Commit and rerun

**Files:**
- Update: `.planning/2026-07-12-clstr-benchmark-protocol-repair/progress.md`
- Update: `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/finalwork/clstr_benchmark_results.md`

- [ ] Review `git diff` and ensure unrelated dirty planning files are not staged.
- [ ] Commit only the protocol repair, tests, and this plan.
- [ ] Submit corrected CLSTR ToolSandbox native full evaluation.
- [ ] Submit corrected CLSTR Tau2 native full evaluation.
- [ ] Submit aligned CLSTR ToolBench native evaluation with the chosen final budget and preserve the existing result as a diagnostic.
- [ ] Monitor meaningful progress, collect completed metrics, and update the result ledger with provenance paths.
