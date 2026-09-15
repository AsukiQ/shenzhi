# Auxiliary Native Pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run formal auxiliary trajectory pretraining from CLSTR-native routing init with reproducible train and diagnostic reports.

**Architecture:** Keep auxiliary trajectories separate from SkillsBench clean-router data. Extend the existing auxiliary pretrain runner with split filtering, readiness accounting, routing-init loading, and held-out diagnostic evaluation while mapping all engineering targets back to `L_act`, `L_trans`, STOP proxy, and `L_retr`.

**Tech Stack:** Python, PyTorch, Transformers, pytest, existing CLSTR model and auxiliary trajectory JSONL format.

---

### Task 1: RED Tests For Auxiliary Readiness And Routing Init

**Files:**
- Modify: `tests/test_aux_trajectories.py`

- [ ] **Step 1: Add tests that build a tiny normalized auxiliary dataset**

Use `pseudo_skills.jsonl`, `trajectories.jsonl`, and `manifest.json` in a temporary directory with `train`, `valid_seen`, and `test` splits.

- [ ] **Step 2: Add a failing split-filter/readiness test**

Run: `pytest -q tests/test_aux_trajectories.py::test_aux_pretrain_filters_splits_and_records_readiness`

Expected: FAIL because `run_aux_trajectory_pretrain` does not accept `include_splits` / `exclude_splits` or emit readiness fields.

- [ ] **Step 3: Add a failing routing-init/eval diagnostic test**

Run: `pytest -q tests/test_aux_trajectories.py::test_aux_pretrain_loads_native_routing_init_and_writes_eval_report`

Expected: FAIL because routing init manifest loading and `eval_report.json` do not exist yet.

### Task 2: Implement Auxiliary Pretrain Support

**Files:**
- Modify: `clstr/aux_pretrain.py`
- Modify: `scripts/run_aux_trajectory_pretrain.py`

- [ ] **Step 1: Implement split filtering and readiness accounting**

Add `include_splits`, `exclude_splits`, selected/excluded split counts, environment counts, flattened step counts, pseudo-skill counts, low-confidence counts, and coverage fields.

- [ ] **Step 2: Implement routing init manifest loading**

Load `base_clstr_checkpoint`, build `CLSTRModel` from checkpoint config, load matching model state, load native residual rerank sidecar for provenance, and reject q/doc adapter manifests.

- [ ] **Step 3: Implement diagnostic eval**

Evaluate held-out selected split rows after training and write `eval_report.json` with action recall, STOP metrics, and transition proxy metrics.

- [ ] **Step 4: Update CLI arguments**

Expose `--routing_init_manifest`, `--include_splits`, `--exclude_splits`, `--max_trajectories`, `--max_steps`, and `--batch_size`.

### Task 3: Reports And Documentation

**Files:**
- Modify: `clstr/aux_pretrain.py`
- Modify: `README.md`

- [ ] **Step 1: Write design report**

Write `outputs/aux_trajectory_pretrain_full/design_report.json` from the training runner.

- [ ] **Step 2: Update README commands**

Document native routing init pretrain, diagnostic eval, loss taxonomy, and SkillsBench boundary.

### Task 4: Formal Run And Verification

**Files:**
- Outputs only under `outputs/aux_trajectory_pretrain_full/`

- [ ] **Step 1: Run targeted tests**

Run: `pytest -q tests/test_aux_trajectories.py tests/test_train_smoke.py tests/test_external_data.py tests/test_encoders.py`

- [ ] **Step 2: Run formal-ish training**

Run with `--routing_init_manifest outputs/clstr_native_routing_init/manifest.json`, exclude `test`, use at least 512 steps if hardware allows, and write checkpoint, `train_report.json`, `eval_report.json`, and `design_report.json`.

- [ ] **Step 3: Run full verification**

Run: `pytest -q`

- [ ] **Step 4: Final audit**

Check `/root/autodl-tmp/skillrouter` status is untouched and no SkillsBench clean-router/harness/successful trajectory files were written.
