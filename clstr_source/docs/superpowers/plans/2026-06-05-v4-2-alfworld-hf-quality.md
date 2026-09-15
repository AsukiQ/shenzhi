# v4.2 ALFWorld HF Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a v4.2 unified pretrain data view that adds high-quality HuggingFace ALFWorld admissible-success trajectories and prioritizes them over weak AgentGym ALFWorld rows.

**Architecture:** Extend `scripts/build_clstr_unified_pretrain.py` with a new recipe `v4_2_alfworld_hf_quality`. The builder reads local or HF-downloaded ALFWorld parquet rows, converts success/admissible conversations into the existing trajectory schema, writes source-quality metadata, and keeps Stage0/Stage1/Stage2 consumers unchanged.

**Tech Stack:** Python JSONL/parquet builder, pytest, existing CLSTR unified schema.

---

### Task 1: Add HF ALFWorld Parser Tests

**Files:**
- Modify: `tests/test_unified_pretrain_v2.py`
- Modify: `scripts/build_clstr_unified_pretrain.py`

- [ ] Add a fixture with two HF-style ALFWorld rows: one success/admissible and one failure/non-admissible.
- [ ] Assert the parser keeps only the success/admissible row.
- [ ] Assert parsed rows include `source_quality=hf_alfworld_admissible_success`, train provenance, `skill_id`, `next_skill_id`, `loss_mask`, and admissible actions when present.

### Task 2: Implement v4.2 Builder Hook

**Files:**
- Modify: `scripts/build_clstr_unified_pretrain.py`

- [ ] Add `v4_2_alfworld_hf_quality` to recipe choices.
- [ ] Add local path discovery for:
  `data/hf_alfworld_admissible_success/`
  and `.tmp/hf_alfworld_admissible_success/`.
- [ ] Convert HF `messages` + `metadata` rows into canonical trajectory rows.
- [ ] Add HF ALFWorld rows before AgentGym weak rows so ALFWorld caps naturally prefer quality rows.

### Task 3: Build/Download Data Safely

**Files:**
- Add or modify only under `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/data` or `.tmp`.

- [ ] Try to load existing local parquet first.
- [ ] If missing, download from HuggingFace mirror using `HF_ENDPOINT=https://hf-mirror.com`.
- [ ] Do not include ScienceWorld in the v4.2 main recipe.

### Task 4: Validate Dataset and Docs

**Files:**
- Modify: `description.md`

- [ ] Build `data/clstr_unified_pretrain_v4_2_alfworld_hf_quality`.
- [ ] Verify leakage audit is ok.
- [ ] Report trajectory/retrieval/skill counts and ALFWorld quality breakdown.
- [ ] Run focused pytest and py_compile.
