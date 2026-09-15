# Stage2 Counterfactual Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain Stage2 dynamic memory so true causal history ranks the next skill above shuffled history while preserving the frozen static endpoint.

**Architecture:** Reuse the existing unified-memory full-pool Stage2 trainer. Within each benchmark-balanced batch, cyclically permute current memories inside `(benchmark, replay-length)` groups, run the same post-action update for true and shuffled memories, and optimize true next-skill NLL plus true-over-shuffled utility and static no-regret. The Qwen backbone and Stage2 static foundation remain frozen for the warm-start smoke.

**Tech Stack:** Python, PyTorch, existing CLSTR Stage2 trainer, Slurm-only pytest/training.

---

### Task 1: Shuffled-history objective

**Files:**
- Modify: `clstr/counterfactual_ranking.py`
- Modify: `tests/test_counterfactual_ranking.py`

- [ ] Add a failing test where true logits have higher positive log utility than shuffled logits and assert zero loss; reverse them and assert positive loss.
- [ ] Run `pytest -q tests/test_counterfactual_ranking.py` through Slurm and verify RED.
- [ ] Add `shuffled_history_utility_loss(true_logits, shuffled_logits, positive_mask, valid_mask, margin)` using `relu(margin + u_shuffled - u_true).mean()` over finite rows with legal positives and negatives.
- [ ] Re-run the focused Slurm test and commit.

### Task 2: Benchmark-matched memory permutation

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `tests/test_full_base_train.py`

- [ ] Add a failing unit test for `_counterfactual_memory_permutation(rows, device)` asserting deterministic cyclic swaps within benchmark and replay-length groups, no cross-benchmark donor, and `-1` for singleton groups.
- [ ] Run the focused test through Slurm and verify RED.
- [ ] Implement the helper using `(source_benchmark, len(replay_prefix))` groups and stable row order.
- [ ] Re-run the focused test and commit.

### Task 3: Stage2-CF training integration

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Test: `tests/test_full_base_train.py`
- Test: `tests/test_sbatch_scripts.py`

- [ ] Add failing tests that require a `counterfactual_history` loss weight, true/shuffled utility metrics, and zero eligible rows when no valid donor exists.
- [ ] Add `counterfactual_history` to canonical loss weights with default `0.0`.
- [ ] For full-pool unified routing, build shuffled current memory with the permutation, detach donor memory, reuse the same action/observation/current-skill inputs, compute shuffled next memory/logits, and add the weighted shuffled-history utility loss.
- [ ] Persist `counterfactual_history_loss`, eligible rows, violation rows, true utility, shuffled utility, and true-minus-shuffled utility.
- [ ] Forward `COUNTERFACTUAL_HISTORY_LOSS_WEIGHT` and `COUNTERFACTUAL_HISTORY_MARGIN` through the CLI and sbatch launcher.
- [ ] Run focused Stage2/full-base/sbatch tests through Slurm and commit.

### Task 4: Warm-start smoke launcher and promotion

**Files:**
- Create: `scripts/sbatch/run_qwen06_clstr_stage2_counterfactual_repair.sh`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `clstr/full_base_train.py`
- Test: `tests/test_qwen_clstr_training_launchers.py`

- [ ] Add a failing launcher test requiring 1200 steps, current Stage2 step10000 warm-start, frozen static foundation, frozen Qwen cache, `counterfactual_history=1.0`, `L_trans_skill_ce=1.0`, and legacy/dead losses at zero.
- [ ] Add an optional compatible warm-start checkpoint loaded after Stage0/Stage1 initialization but before optimizer creation; persist its SHA and loaded-key report.
- [ ] Implement the launcher with batch size 16, learning rate `3e-5` to `3e-6`, validation every 300 steps, and output `outputs/qwen06_clstr_postfix/stage2_counterfactual_repair_smoke`.
- [ ] Run launcher contract tests, `bash -n`, `py_compile`, and `git diff --check`; commit.
- [ ] Submit the 1200-step smoke through Slurm.
- [ ] At steps 300/600/900/1200, require finite nonzero gradients and positive true-minus-shuffled utility; after training rerun the ToolSandbox and Tau2 counterfactual audits.
- [ ] Promote only if true replay beats shuffled replay and final routing does not regress below static; otherwise stop before the clean Stage1-to-Stage2 retrain.
