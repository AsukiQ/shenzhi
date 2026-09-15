# Candidate-Provenance Positive Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-training CMC reliability mode that applies bounded positive memory residuals only to dynamic-only candidates and otherwise preserves exact static scoring.

**Architecture:** Candidate provenance already exists in `CandidateUnion`. A reusable mask builder and fusion primitive produce candidate-wise memory activation. Current-state and ALFWorld scorers share the mode; a final-chain overlay exposes it to the aligned multibench launchers without changing the Stage4 checkpoint.

**Tech Stack:** Python 3, PyTorch, existing CMC candidate union, pytest through Slurm, Slurm multibench launchers.

---

### Task 1: Implement provenance masks and positive bounded fusion

**Files:**
- Modify: `tests/test_memory_candidate_recall.py`
- Modify: `tests/test_safe_memory_ranking.py`
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `clstr/safe_memory_ranking.py`

- [ ] Write RED tests asserting exact shared/static rows, positive-only dynamic-extra boosts, bound enforcement, invalid masking, and exact zero-history fallback.
- [ ] Run the focused tests through Slurm and confirm failures for missing APIs.
- [ ] Add `candidate_provenance_mask(candidate_rows, dynamic_extra_rows, valid_mask)` and `candidate_provenance_positive_residual_fusion(...)`.
- [ ] Run GREEN through Slurm and commit.

### Task 2: Apply the mode consistently in current-state and ALFWorld scoring

**Files:**
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_alfworld_eval.py`
- Modify: `clstr/memory_utility_gate.py`
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/alfworld_eval.py`

- [ ] Write RED tests for the new `cmc_candidate_provenance` mode before and after final-k truncation, exact local-pool static behavior, and ALFWorld exact-static fallback when no dynamic provenance exists.
- [ ] Run RED through Slurm.
- [ ] Add the reliability mode, build provenance masks at both scoring points, and route through the shared fusion primitive. Preserve existing `cmc` behavior byte-for-byte.
- [ ] Run GREEN through Slurm and commit.

### Task 3: Bind the mode into final-chain and multibench execution

**Files:**
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `clstr/qwen_clstr_multibench_submit.py`
- Modify: `scripts/resolve_qwen06_clstr_final_chain.py`

- [ ] Write RED tests requiring a self-hashed candidate-provenance reliability overlay with no gate checkpoint and runtime mode propagation.
- [ ] Run RED through Slurm.
- [ ] Add an explicit resolver flag and reliability contract binding residual bound, Stage4 SHA, base/deployed sources, and unchanged checkpoint-chain digest.
- [ ] Run GREEN through Slurm and commit.

### Task 4: Verify and run aligned smoke

- [ ] Run `py_compile`, `bash -n`, and `git diff --check` locally.
- [ ] Run the complete affected pytest matrix through Slurm.
- [ ] Resolve a fresh candidate-provenance final chain and build a smoke configuration aligned with the existing CMC evaluation manifests.
- [ ] Submit ToolBench, ToolSandbox, and Tau2-base smoke jobs.
- [ ] Stop unless ToolBench beats current CMC fused MRR while ToolSandbox/Tau2 remain within `0.005` of static and all identity/protocol checks pass.
- [ ] Only after smoke passes, submit aligned full evaluation and continue ALFWorld control import.
