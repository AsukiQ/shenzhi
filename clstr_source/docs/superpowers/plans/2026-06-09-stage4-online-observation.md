# Stage4 Online Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect post-action observations into the v4.1b ALFWorld online HRPO path without leaking future observations into pre-action scoring.

**Architecture:** Keep action selection pre-action and policy-only/controller-compatible. Extend rollout records with `observation_text` and `next_observation_text`, then train post-action auxiliary transition/belief consistency from `(state_t, action_t, obs_{t+1})` alongside HRPO policy loss.

**Tech Stack:** Python, PyTorch, pytest, existing CLSTR `online_hrpo.py`, existing Stage4/FullBase transition helpers.

---

### Task 1: Rollout Records Carry Post-Action Observation

**Files:**
- Modify: `clstr/online_hrpo.py`
- Test: `tests/test_online_hrpo.py`

- [x] **Step 1: Write failing test**

Add a test that runs a tiny fake ALFWorld env through `_rollout_once()` and asserts every recorded step includes `observation_text`, `next_observation_text`, `done`, and `next_admissible_actions`.

- [x] **Step 2: Verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_online_hrpo.py::test_online_hrpo_rollout_records_post_action_observation -q
```

Expected: fail because `HrpoStep` does not expose the new fields.

- [x] **Step 3: Implement minimal schema change**

Add fields to `HrpoStep` and populate them in `_rollout_once()` immediately after `env.step([action])`.

- [x] **Step 4: Verify pass**

Run the same pytest target and confirm it passes.

### Task 2: No-Leakage Guard for Action Scoring

**Files:**
- Modify: `tests/test_online_hrpo.py`
- Modify: `clstr/online_hrpo.py`

- [x] **Step 1: Write failing/no-regression test**

Add a test showing `_sample_action()` / `compute_hrpo_action_logits()` only receives `state_text` and candidate actions, never `next_observation_text`.

- [x] **Step 2: Verify**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_online_hrpo.py::test_online_hrpo_action_scoring_does_not_encode_next_observation -q
```

Expected: pass or fail only for the intended no-leakage assertion.

### Task 3: Post-Action Auxiliary Observation Loss

**Files:**
- Modify: `clstr/online_hrpo.py`
- Test: `tests/test_online_hrpo.py`

- [x] **Step 1: Write failing tests**

Add tests for a helper that builds an auxiliary loss from rollout rows with `next_observation_text`, and returns zero/count-zero when no post-action observations exist.

- [x] **Step 2: Implement helper**

Implement a small `_compute_online_observation_aux_loss()` that:

- encodes `state_text`;
- builds current skill memory `m_t`;
- encodes chosen actions and `next_observation_text`;
- uses `model.transition(m_t, action_emb, next_obs_emb)` when available;
- aligns the prediction to detached next-observation skill memory with MSE;
- optionally runs `model.gate(pred, next_obs_memory, next_obs_emb)` and aligns it to the same target.

- [x] **Step 3: Integrate into training loop**

Add auxiliary loss to `train_clstr_online_hrpo_with_model()` with weights from `HrpoLossConfig.transition_weight` and `HrpoLossConfig.belief_weight`. Report `online_transition_loss`, `online_belief_loss`, `online_observation_aux_count`, `post_action_observation_used`, and `pre_action_next_observation_leakage=false`.

### Task 4: Documentation and Verification

**Files:**
- Modify: `description.md`
- Modify: `docs/clstr_v4_plan.md`

- [x] **Step 1: Document the change**

Record that Stage4-online now uses `obs_{t+1}` only after `env.step`, while pre-action scoring remains leakage-free.

- [x] **Step 2: Run targeted verification**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_online_hrpo.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/online_hrpo.py scripts/run_clstr_v4_1b_alfworld_online_hrpo.py
git diff --check -- clstr/online_hrpo.py tests/test_online_hrpo.py description.md docs/clstr_v4_plan.md docs/superpowers/plans/2026-06-09-stage4-online-observation.md
```

Expected: all pass.

### Execution Notes

- 2026-06-09 observation smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_observation_smoke` verified post-action observation recording and no pre-action leakage, but reward/advantage was degenerate.
- 2026-06-09 progress reward smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_progress_smoke` fixed nonzero reward variation and HRPO advantage.
- 2026-06-10 multi-update/action-quality smokes verified persistent online observation auxiliary loss and nonzero HRPO updates.
- 2026-06-10 goal-action-prior smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_smoke` produced at least one target-object interaction (`take tissuebox 2 from diningtable 1`), but did not yet show stable goal interaction or task success.
