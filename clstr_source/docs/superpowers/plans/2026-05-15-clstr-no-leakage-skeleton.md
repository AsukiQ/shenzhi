# CLSTR No-Leakage Experiment Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the no-leakage SkillsBench + SkillRouter eval-only + ALFWorld experiment skeleton.

**Architecture:** Keep CLSTR training generic: mock smoke fixtures and generic replay trajectories are allowed, while SkillRouter benchmark files are eval-only. Add Stage0 audit as a small data-inspection module plus CLI, and wrap the read-only SkillRouter benchmark runner from inside this repo without modifying `/root/autodl-tmp/skillrouter`.

**Tech Stack:** Python 3.11, pytest, PyYAML, existing CLSTR package, read-only upstream SkillRouter scripts.

---

### Task 1: Tests First

**Files:**
- Modify: `tests/test_rollout.py`
- Modify: `tests/test_train_smoke.py`
- Modify: `tests/test_validator.py`
- Modify: `tests/test_data.py`
- Create: `tests/test_leakage.py`
- Create: `tests/test_skillrouter_eval.py`

- [ ] Write failing tests for explicit SkillsBench / SkillRouter eval-only / ALFWorld data roles, Stage0 audit outputs, raw-topK semantics, and eval-only SkillRouter baseline wrapper.
- [ ] Run the targeted tests and confirm the new tests fail for the expected missing behavior.

### Task 2: Core Implementation

**Files:**
- Modify: `clstr/rollout.py`
- Modify: `clstr/train.py`
- Modify: `clstr/validator.py`
- Modify: `configs/data/base.yaml`
- Create: `clstr/leakage.py`
- Create: `scripts/run_leakage_audit.py`
- Create: `scripts/run_skillrouter_eval_only.py`
- Create: `scripts/run_clstr_smoke_train.sh`

- [ ] Rename legacy replay APIs to `ReplayTrajectoryEnv` / `load_replay_trajectories`.
- [ ] Add data role validation and train-time protection against SkillRouter eval supervision leakage.
- [ ] Implement audit artifact extraction from SkillRouter eval_core and missing split reporting for SkillsBench.
- [ ] Implement eval-only SkillRouter baseline wrapper that writes under `outputs/`.
- [ ] Fix verified template raw recall to read only `raw_topk` / `raw_candidates`.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `scripts/train_stage.sh`
- Modify: historical docs only to mark superseded benchmark-specific content archived if needed.

- [ ] Document Stage0 audit, SkillRouter eval-only baseline, and CLSTR mock/generic replay smoke training commands.
- [ ] Run the required targeted pytest command.
- [ ] Run full `pytest -q`.
- [ ] Run a legacy benchmark residue audit and classify any remaining archived references.
