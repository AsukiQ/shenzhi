# Transition Prior + Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the stable old skill-level transition signal while preserving the action-aware + next-observation semantics and later engineering fixes.

**Architecture:** Do not git-reset the repository. Keep the current action-aware transition path as a residual branch, add a skill-level prior branch based on `m_t/current_skill`, and score candidate next skills with `prior_logits + lambda * residual_logits`. With `lambda=0`, the model falls back to the stable skill-transition prior; with `lambda>0`, action/observation feedback can refine ranking.

**Tech Stack:** Python, PyTorch, existing CLSTR Stage1/2/4 training utilities, pytest.

---

### Task 1: Preserve Current Engineering Fixes

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage4_act_train.py`
- Test: `tests/test_full_base_train.py`
- Test: `tests/test_stage4_act_train.py`
- Test: `tests/test_v4_action_proj_sharing.py`

- [x] Do not use `git reset --hard` or wholesale checkout of old files.
- [x] Keep current sbatch/gate/readiness/checkpoint-safety behavior intact.
- [x] Keep action-aware transition semantics:
  - action channel: actual `action_text` embedding through `action_proj` when available;
  - observation channel: `next_observation_text` embedding;
  - no action-text-as-observation regression.

### Task 2: Add Prior + Residual Transition Scoring

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage4_act_train.py`

- [x] Add helper(s) that compute transition candidate logits from a prediction vector and candidate ids.
- [x] Compute prior prediction with current-skill soft-shared action input and a neutral observation tensor.
- [x] Compute residual prediction with actual action embedding and next-observation embedding.
- [x] Combine logits as:

```text
transition_logits = prior_logits + transition_residual_lambda * residual_logits
```

- [x] Default `transition_residual_lambda` should be conservative and globally configured, not benchmark-specific.
- [x] Record the scoring mode and lambda in metrics/train_report.

### Task 3: TDD Coverage

**Files:**
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_stage4_act_train.py`
- Modify: `tests/test_v4_action_proj_sharing.py`

- [x] Add a test proving `lambda=0` ignores residual action/observation logits and uses only prior logits.
- [x] Add a test proving `lambda>0` adds residual logits without replacing the prior.
- [x] Preserve existing tests that assert action and next-observation are sent to the correct channels.

### Task 4: Verification

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py \
  tests/test_stage4_act_train.py \
  tests/test_v4_action_proj_sharing.py \
  tests/test_stage1_heads_quality_gate.py

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  clstr/stage4_act_train.py
```

Expected:

```text
all selected tests pass
py_compile exits 0
```

No Slurm/GPU job should be submitted in this implementation step.

**2026-06-08 update:** Implemented in the current dirty working tree without git reset/checkout. Stage1, Stage2, and Stage4 now expose `transition_residual_lambda`; metrics and train reports record `skill_prior_plus_action_observation_residual`. CPU tests and compile checks passed; no Slurm/GPU job was submitted.
