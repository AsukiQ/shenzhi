# Qwen Safe Memory Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain Qwen CLSTR Stage2/4 with a frozen static route, raw dynamic candidate recall, and jointly trained bounded gated local ranking.

**Architecture:** A pure ranking module builds bounded residual fusion and deterministic small-pool masks. `CLSTRModel` owns a causal route-utility gate. Stage2/4 keep full-pool raw-dynamic NLL for recall while directly supervising safe fused ranking on local candidate masks.

**Tech Stack:** Python, PyTorch, pytest through Slurm, existing CLSTR Stage2/Stage4 trainers and Qwen launchers.

---

## File structure

- Create `clstr/safe_memory_ranking.py`: pure bounded fusion, local masks, margin objective, and metrics.
- Modify `clstr/model.py`: add `RouteMemoryUtilityGate` and a method returning alpha.
- Modify `clstr/full_base_train.py`: freeze policy, Stage2 joint objective, metrics, and report schema.
- Modify `clstr/stage4_act_train.py`: use the identical objective and freeze contract.
- Modify `clstr/current_state_route_eval.py`: raw-dynamic candidate union plus safe fused ranking.
- Modify Qwen Stage2/4 CLI and sbatch launchers: explicit hyperparameters and schema identity.
- Modify readiness/quality modules: fail closed on missing safe-memory evidence.
- Add focused tests under `tests/` for each changed boundary.

### Task 1: Integrate the repaired mainline

- [ ] Merge `master` into `exp/qwen06-clstr-postfix` and resolve only conflicts that preserve the selected Qwen artifacts and repaired benchmark protocols.
- [ ] Confirm `git merge-base --is-ancestor b61f569 HEAD` succeeds and `git merge-base --is-ancestor 8ce8222 HEAD` fails.
- [ ] Run `git diff --check` and commit the merge if Git does not create a merge commit automatically.

### Task 2: Add bounded safe-fusion primitives

**Files:**
- Create: `clstr/safe_memory_ranking.py`
- Create: `tests/test_safe_memory_ranking.py`

- [ ] Write failing tests showing that alpha zero exactly equals static, residuals are centered and bounded, invalid masks fail, and gradients reach dynamic logits and alpha.
- [ ] Submit the RED test with `sbatch --wait --wrap='cd <worktree> && CUDA_VISIBLE_DEVICES="" pytest -q tests/test_safe_memory_ranking.py'`; expect failures because the module is absent.
- [ ] Implement `bounded_memory_fusion(static_logits, dynamic_logits, alpha, valid_mask, residual_bound)` using valid-mask centering followed by `residual_bound * tanh(centered_delta / residual_bound)`.
- [ ] Re-run the same Slurm test and require all tests to pass.

### Task 3: Add deterministic local candidate masks and objective

**Files:**
- Modify: `clstr/safe_memory_ranking.py`
- Modify: `tests/test_safe_memory_ranking.py`

- [ ] Write failing tests for explicit inventories, sizes 2/3/4/5/8/10, mandatory positives, frozen-static hard negatives, dynamic disagreement negatives, all-positive exclusion, and deterministic ties.
- [ ] Verify RED through Slurm.
- [ ] Implement `build_local_candidate_masks(...)` without additional model forward calls.
- [ ] Write failing tests for static-correct margin preservation, static-wrong gain, multi-positive NLL, row weights, and zero eligible rows.
- [ ] Implement `safe_local_route_objective(...)` returning loss terms, eligibility counts, violations, and endpoint ranks.
- [ ] Verify GREEN through Slurm and commit the pure module.

### Task 4: Add the causal route-utility gate

**Files:**
- Modify: `clstr/model.py`
- Modify: `tests/test_model_pipeline.py`
- Modify: `tests/test_stage_checkpoint_init.py`

- [ ] Write failing tests for alpha shape/range, near-static initialization, zero-history exact alpha zero, causal tensor dependence, gradient flow, and old-checkpoint missing-key allowlisting.
- [ ] Verify RED through Slurm.
- [ ] Add `RouteMemoryUtilityGate`, using normalized state, static memory, memory delta, and state-delta interaction, with the final bias initialized for alpha approximately 0.01.
- [ ] Add `route_memory_alpha(h_t, static_memory, dynamic_memory, causal_update_count)` to `CLSTRModel`.
- [ ] Extend checkpoint schema/allowlists so Stage0/1 checkpoints may omit the gate but new Stage2/4 resumes may not.
- [ ] Verify GREEN and commit.

### Task 5: Integrate Stage2 joint training

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`
- Modify: `tests/test_stage2_quality_gate.py`

- [ ] Write failing tests proving the static endpoint is detached and unchanged, the raw full-pool loss reaches transition/correction, the local loss reaches route utility, and `unified_retriever` is excluded from the optimizer.
- [ ] Verify RED through Slurm.
- [ ] Replace the current Stage2 route objective with raw dynamic full-pool NLL plus safe local objective; retain the old counterfactual metrics only as an endpoint diagnostic.
- [ ] Freeze encoder projection, skill table, initial-belief head, and unified retriever; train transition, correction gate, action projection if required, and route-utility gate.
- [ ] Add report fields for residual bound, local sizes, alpha statistics, local endpoint ranks, and safety/gain violations.
- [ ] Wire explicit launcher defaults and quality-gate assertions.
- [ ] Verify GREEN and commit.

### Task 6: Integrate Stage4 calibration

**Files:**
- Modify: `clstr/stage4_act_train.py`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`
- Modify: `tests/test_stage4_act_train.py`
- Modify: `tests/test_stage4_quality_gate.py`

- [ ] Write failing tests proving Stage4 uses the same safe objective, preserves the frozen static endpoint, and trains only causal memory modules plus route utility.
- [ ] Verify RED through Slurm.
- [ ] Integrate the shared objective and report schema, retaining trajectory-disjoint data and current stage selection logic.
- [ ] Require the new Stage2 checkpoint schema before full Stage4 training.
- [ ] Verify GREEN and commit.

### Task 7: Align inference and candidate recall

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `clstr/frozen_clstr_route_eval.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_memory_candidate_recall.py`
- Modify: `tests/test_qwen_clstr_frozen_route_eval.py`

- [ ] Write failing tests showing raw dynamic scores supply extra candidates, safe fused scores rank the shared union, local pools receive no injected candidates, and zero-history rows equal static exactly.
- [ ] Verify RED through Slurm.
- [ ] Implement the shared inference contract and persist static/raw-dynamic/safe-fused metrics plus alpha diagnostics.
- [ ] Verify GREEN and commit.

### Task 8: Verify source and run Qwen smoke

- [ ] Submit focused pytest suites through Slurm with `CUDA_VISIBLE_DEVICES=""` and require zero failures.
- [ ] Submit the complete affected Stage2/Stage4/evaluation suite through Slurm and require zero failures.
- [ ] Run local `python -m compileall` only on changed Python modules, `bash -n` on changed launchers, CLI `--help`, stale-symbol scans, and `git diff --check`.
- [ ] Submit representative Qwen Stage2 smoke from the selected Stage1 checkpoint; inspect finite loss, optimizer membership, frozen-module evidence, alpha distribution, and local/full-pool metrics.
- [ ] If Stage2 smoke passes, submit Stage4 smoke and apply the same checks.

### Task 9: Run the Qwen full chain and benchmarks

- [ ] Launch Stage2 full from the existing selected Qwen Stage1 artifact, then Stage4 full from the selected new Stage2 checkpoint.
- [ ] Run ToolBench, Tau2, ToolSandbox, and ALFWorld using the repaired benchmark protocols.
- [ ] Update `finalwork/clstr_benchmark_results.md` with static, raw dynamic, safe fused, and equal-budget results.
- [ ] Require ToolSandbox safe-fused MRR delta versus static to be at least `-0.005`, nonnegative ToolBench/Tau2/ToolSandbox macro-MRR delta versus the current selected Qwen route, and positive strict dynamic-extra recall on at least one applicable global-pool benchmark.
- [ ] Keep BGE Stage2/4 cancelled if any gate fails; otherwise write a separate BGE execution plan using the same protocol.

