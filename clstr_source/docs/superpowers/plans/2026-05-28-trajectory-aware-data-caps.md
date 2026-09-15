# Trajectory-Aware Data Caps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trajectory-derived Stage0 retrieval samples and benchmark-aware Stage2 row caps so auxiliary ALFWorld/ScienceWorld improve sequential routing without dominating ToolBench-G3 and TRAJECT-Bench.

**Architecture:** Extend the unified pretrain builder to append capped current/next trajectory retrieval pairs from `trajectories.jsonl`. Extend Stage2 row filtering with explicit benchmark caps applied after train/benchmark filters and before Stage0 candidate handoff.

**Tech Stack:** Python JSONL data builders, pytest unit tests, existing CLSTR Stage0/Stage2 scripts.

---

### Task 1: Stage0 Trajectory-Derived Retrieval

**Files:**
- Modify: `scripts/build_clstr_unified_pretrain.py`
- Modify: `tests/test_unified_pretrain_v2.py`
- Modify: `scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh`

- [x] Add tests that ALFWorld/ScienceWorld trajectory rows are converted into capped current/next retrieval pairs.
- [x] Add builder helpers for cap parsing, trajectory query text, and capped per-benchmark row selection.
- [x] Add CLI/sbatch env controls through `TRAJECTORY_RETRIEVAL_CAPS`.

### Task 2: Stage2 Benchmark Caps

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_sbatch_scripts.py`

- [x] Add tests that `benchmark_caps` keeps ToolBench/TRAJECT unlimited and caps ALFWorld/ScienceWorld.
- [x] Add cap parsing and report fields.
- [x] Wire CLI/sbatch defaults to `toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000`.

### Task 3: Documentation and Verification

**Files:**
- Modify: `description.md`

- [x] Document that ALFWorld/ScienceWorld are auxiliary sequential-routing data with default 10k caps.
- [x] Run focused pytest for unified pretrain, full-base filters, and sbatch scripts.

### Task 4: Build v3 Data and Stage0 Defaults

**Files:**
- Modify: Stage0/Stage1/Stage2/Stage3/Stage4 CLI and sbatch defaults
- Modify: `scripts/build_clstr_unified_pretrain.py`
- Modify: `description.md`

- [x] Rebuild `data/clstr_unified_pretrain_v3_trajectory_retrieval_caps`.
- [x] Fix `retrieval_stream.total_pairs` so manifest excludes skipped held-out counts and matches `retrieval.jsonl` line count.
- [x] Set Stage0 default sampling to `source_balanced`.
- [x] Point Stage0 baseline/eval/readiness and downstream handoff defaults at the v3 data/Stage0 output paths.
- [x] Record that no Slurm/full training job was submitted.
