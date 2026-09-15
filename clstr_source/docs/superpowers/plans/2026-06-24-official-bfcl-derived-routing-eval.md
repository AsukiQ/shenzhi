# Official BFCL-Derived Routing Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a smoke-first official BFCL-derived next-function routing benchmark for comparing frozen SkillRouter-style retrieval against full CLSTR.

**Architecture:** Convert official `gorilla-llm/Berkeley-Function-Calling-Leaderboard` v3 files into CLSTR-style routing rows without using UniToolCall conversion. The evaluator uses the same row/candidate/positive set for SkillRouter frozen bi-encoder and CLSTR Stage0/base/Stage4, then reports strict r@1/r@5/MRR.

**Tech Stack:** Python, PyTorch, HuggingFace local BFCL JSON/JSONL files, existing CLSTR Stage0/Stage4 evaluators, pytest, Slurm sbatch wrappers.

---

### Task 1: Official BFCL Converter

**Files:**
- Create: `clstr/bfcl_route_eval.py`
- Test: `tests/test_bfcl_route_eval.py`

- [ ] Write tests for JSONL parsing, single-turn `multiple` rows, parallel multi-positive rows, and multi-turn execution-string rows.
- [ ] Implement `load_bfcl_route_corpus(data_root, categories, max_rows_per_category)` returning `skills`, `source_rows`, and an audit report.
- [ ] Use official BFCL provenance fields: `bfcl_id`, `bfcl_category`, `turn_index`, `call_index`, and `raw_source_path`.
- [ ] Exclude categories with candidate count <= 1 from the default main routing set; keep counts in the audit report.

### Task 2: SkillRouter Frozen Baseline

**Files:**
- Modify: `clstr/bfcl_route_eval.py`
- Create: `scripts/run_bfcl_skillrouter_frozen_eval.py`
- Create: `scripts/sbatch/run_bfcl_skillrouter_frozen_eval.sh`
- Test: `tests/test_bfcl_route_eval.py`

- [ ] Write tests that monkeypatch embedding calls and verify multi-positive scoring when duplicate names exist.
- [ ] Implement frozen SkillRouter-style bi-encoder ranking over the exact BFCL candidate set.
- [ ] Emit `bfcl_skillrouter_frozen_eval_report.json` with strict r@1/r@5/MRR and per-category metrics.
- [ ] Run local tests and sbatch syntax check before any Slurm job.

### Task 3: CLSTR Full Route Eval

**Files:**
- Modify: `clstr/bfcl_route_eval.py`
- Create: `scripts/run_bfcl_full_clstr_route_eval.py`
- Create: `scripts/sbatch/run_bfcl_full_clstr_route_eval.sh`
- Test: `tests/test_bfcl_route_eval.py`

- [ ] Write tests for converting BFCL rows into Stage4-compatible rows without positive injection.
- [ ] Reuse the existing Stage0 checkpoint loader and Stage4 logged-online evaluator.
- [ ] Report Stage0 prior, CLSTR base, and full Stage4 strict metrics on the same BFCL rows.
- [ ] Keep BFCL as zero-shot eval: no parameter training, no BFCL rows added to training.

### Task 4: Smoke And Full Eval

**Files:**
- Modify: `.planning/2026-06-20-stage2-prior-residual-retrain/progress.md`
- Modify: `.planning/2026-06-20-stage2-prior-residual-retrain/task_plan.md`

- [ ] Run local tests: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_bfcl_route_eval.py -q`.
- [ ] Run `bash -n` for both BFCL sbatch wrappers.
- [ ] Submit one smoke job at a time, with small category caps.
- [ ] Only if smoke has no blockers, submit full BFCL SkillRouter and CLSTR evals.
- [ ] Record final metrics and keep BFCL wording as “official BFCL-derived next-function routing,” not official BFCL leaderboard score.
