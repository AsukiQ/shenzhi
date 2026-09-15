# Anchored Harm-Suppression Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and deploy a fail-closed linear gate that preserves the train-selected fixed CMC mixture by default and lowers alpha only when memory is predicted to be harmful.

**Architecture:** The oracle audit is changed from static-versus-dynamic to static-versus-train-selected-fixed-alpha. A shared linear gate predicts harmful-memory probability from the existing eleven reliability features; runtime alpha is `alpha_base * (1 - p_harm)`, with exact zero-history fallback. The immutable Stage4 residual adapter and candidate union remain active.

**Tech Stack:** Python 3, PyTorch, JSON/JSONL route manifests, pytest through Slurm, existing CMC final-chain and benchmark launchers.

---

## File structure

- Modify `scripts/audit_clstr_memory_utility_oracle.py`: construct the fixed-alpha deployed endpoint and make audit eligibility compare a learned harm selector against that endpoint.
- Modify `tests/test_memory_utility_oracle_audit.py`: cover fixed-alpha oracle headroom, selector gain, and source regret.
- Modify `clstr/memory_utility_gate.py`: add an inference-only anchored wrapper that converts harmful-memory probability into alpha.
- Modify `clstr/memory_utility_gate_train.py`: train harmful-memory probability with sign/source balancing and promote against fixed alpha.
- Modify `tests/test_memory_utility_gate_train.py`: cover anchored targets, alpha mapping, filtering, checkpoint schema, and promotion.
- Modify `clstr/qwen_clstr_final_chain.py`: accept and bind the anchored checkpoint schema and alpha base.
- Modify `tests/test_qwen_clstr_final_chain.py`: cover anchored overlay lineage.
- Modify `tests/test_qwen06_cmc_direct_utility_recalibration.py`: require fail-closed anchored orchestration.
- Reuse `scripts/run_qwen06_cmc_direct_utility_recalibration.py` and its Slurm launcher after the report contract changes.

### Task 1: Audit the exact fixed-versus-static deployment decision

**Files:**
- Modify: `tests/test_memory_utility_oracle_audit.py`
- Modify: `scripts/audit_clstr_memory_utility_oracle.py`

- [ ] **Step 1: Write failing anchored-audit tests**

Add a synthetic case where `alpha_base` is selected on train, fixed alpha is strong on dev, and a feature-separated harmful minority permits a selector to improve over fixed. Assert:

```python
anchored = report["anchored_harm_eligibility"]
assert report["best_fixed_alpha"] == 0.85
assert anchored["fixed_or_static_utility_oracle_gain_over_fixed"] >= 0.005
assert anchored["linear_harm_selector_gain_over_fixed"] >= 0.002
assert anchored["worst_source_regret_vs_fixed"] <= 0.005
assert anchored["eligible"] is True
assert report["learned_gate_recommended"] is True
```

Add rejection cases for no harmful rows and selector MRR not exceeding fixed alpha.

- [ ] **Step 2: Run RED through Slurm**

Run:

```bash
sbatch --wait -p gpu_a800 --gpus=1 --cpus-per-task=4 --time=00:12:00 \
  --job-name=anchored-audit-red \
  --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_memory_utility_oracle_audit.py'\'''
```

Expected: failure for missing `anchored_harm_eligibility`.

- [ ] **Step 3: Implement fixed endpoint rows and anchored eligibility**

After selecting `best_fixed_alpha` on train, create dev rows whose baseline rank/utility comes from that alpha. Reuse `_fit_linear_selector()` with fixed as the positive endpoint and static as fallback. Persist:

```python
"anchored_harm_eligibility": {
    "alpha_base": best_fixed_alpha,
    "both_harm_signs_present": both_signs,
    "fixed_macro_mrr": fixed_mrr,
    "fixed_or_static_utility_oracle_macro_mrr": oracle_mrr,
    "fixed_or_static_utility_oracle_gain_over_fixed": oracle_mrr - fixed_mrr,
    "linear_harm_selector_macro_mrr": selector_mrr,
    "linear_harm_selector_gain_over_fixed": selector_mrr - fixed_mrr,
    "worst_source_regret_vs_fixed": worst_regret,
    "eligible": eligible,
}
```

Require oracle gain `>=0.005`, selector gain `>=0.002`, both signs, and worst source regret `<=0.005`. Make `learned_gate_recommended` depend on this anchored eligibility.

- [ ] **Step 4: Run GREEN and commit**

Run the Step 2 command and require zero failures, then commit the two files.

### Task 2: Train and load a baseline-preserving harmful-memory gate

**Files:**
- Modify: `tests/test_memory_utility_gate_train.py`
- Modify: `clstr/memory_utility_gate.py`
- Modify: `clstr/memory_utility_gate_train.py`

- [ ] **Step 1: Write failing target, mapping, and promotion tests**

Add tests for:

```python
target, weight, delta = direct_harm_targets(
    static, fixed, valid, positive, temperature=0.5
)
assert target[static_better] > 0.5
assert target[fixed_better] < 0.5

wrapper = AnchoredMemoryUtilityGate(harm_gate, alpha_base=0.85)
alpha = wrapper(features)
assert torch.allclose(alpha, 0.85 * (1.0 - harm_gate(features)))
```

Update the successful training fixture so the harmful minority is feature-separable and assert:

```python
assert report["promoted"] is True
assert report["alpha_base"] == 0.85
assert report["dev_summary"]["balanced_macro_mrr_improvement_vs_fixed"] > 0
assert report["validation"]["promotion_checks"]["harmful_rows_improve"] is True
assert report["validation"]["promotion_checks"]["helpful_rows_preserved"] is True
```

Retain the regression test that filters ineligible rows within selected trajectories.

- [ ] **Step 2: Run RED through Slurm**

Run focused gate tests and require failures only for the new anchored contracts.

- [ ] **Step 3: Implement anchored mapping and targets**

Add `AnchoredMemoryUtilityGate`, which owns a `MemoryUtilityGate` and a validated `alpha_base` buffer. Its forward method returns:

```python
harm_probability = self.harm_gate(features)
return self.alpha_base * (1.0 - harm_probability)
```

Add `direct_harm_targets()` using `utility(static)-utility(fixed)`. Construct fixed logits with `fuse_route_scores()` and the audit-bound alpha base.

- [ ] **Step 4: Implement sign/source-balanced training**

Compute one BCE mean for every present `(source, harm_sign)` group and average the group means. Train:

```python
p_harm = harm_gate(train_features)
raw_alpha = alpha_base * (1.0 - p_harm)
alpha = effective_memory_alpha(raw_alpha, counts)
fused = fuse_route_scores(static, dynamic, alpha, valid)
loss = fused_rank_loss + base_no_regret_loss + balanced_harm_bce
```

The no-regret loss uses fixed-alpha utility as its detached reference.

- [ ] **Step 5: Replace proxy promotion with outcome promotion**

Summarize fixed, learned, harmful, and helpful subsets. Require exact endpoint/fallback checks, positive macro MRR delta over fixed, source regret `<=0.005`, harmful-row improvement, helpful-row regret `<=0.005`, and harmful/helpful mean-alpha separation `>=0.10`.

Save `memory_utility_gate_checkpoint_v2` containing `gate_output_semantics="anchored_harm_suppression_alpha"`, `alpha_base`, the core harm-gate state dict, audit SHA, Stage4 SHA, feature caps, and validation results. The loader returns an `AnchoredMemoryUtilityGate`, so existing scorers continue to consume alpha directly.

- [ ] **Step 6: Run GREEN and commit**

Run all gate tests through Slurm, require zero failures, and commit the three files.

### Task 3: Bind the anchored checkpoint into the release chain

**Files:**
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_qwen06_cmc_direct_utility_recalibration.py`
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `scripts/run_qwen06_cmc_direct_utility_recalibration.py`

- [ ] **Step 1: Write failing lineage and workflow tests**

Require the final overlay and recalibration report to bind checkpoint schema v2, alpha base, anchored audit SHA, immutable Stage4 SHA, and `gate_output_semantics`. Reject v1 free-alpha checkpoints for the new run.

- [ ] **Step 2: Run RED through Slurm**

Run the two focused test files and confirm failures for missing anchored identities.

- [ ] **Step 3: Implement fail-closed lineage propagation**

Validate v2 checkpoint/report identities in the final-chain resolver. Add `alpha_base` and output semantics to the reliability overlay without changing the base checkpoint-chain digest. Keep workflow status `not_recommended` or `not_promoted` when either gate fails.

- [ ] **Step 4: Run GREEN and commit**

Run the focused tests through Slurm and commit the four files.

### Task 4: Verify on real records, train, and gate benchmark submission

**Files:**
- Verify all affected source and tests.
- Produce artifacts under `outputs/qwen06_clstr_postfix/cmc_anchored_harm/recalibration`.

- [ ] **Step 1: Run local source-only checks**

Run `py_compile`, `bash -n`, and `git diff --check`. Do not load torch locally.

- [ ] **Step 2: Run the complete affected Slurm pytest matrix**

Run oracle audit, gate training, CMC deployment, final-chain, workflow, ALFWorld scorer, and multibench tests. Require zero failures.

- [ ] **Step 3: Submit the real anchored audit and training**

Use the immutable step-3000 selection and route manifest with a fresh output directory. The workflow must audit first and train only when recommended.

- [ ] **Step 4: Apply stop conditions**

If the anchored audit is negative or the gate is not promoted, do not alter thresholds and do not submit benchmarks. Record fixed `alpha=0.85` as the release fallback.

- [ ] **Step 5: Resolve and smoke-test the promoted overlay**

Only after promotion, resolve the final reliability overlay and run aligned smoke evaluation for ToolBench, ToolSandbox, Tau2-base, and ALFWorld.

- [ ] **Step 6: Submit aligned full benchmarks**

Submit the full four-benchmark chain only if smoke artifacts pass identity, denominator, and protocol checks. Update the shared results ledger after aggregation.
