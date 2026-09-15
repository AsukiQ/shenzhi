# Stage0 Negative-Aware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in explicit-negative auxiliary loss to Stage0 retrieval warmup.

**Architecture:** Preserve the current full-pool multi-positive NLL as the anchor. Extend unified-v2 retrieval loading to carry `negative_skill_ids` into filtered query rows, add a small margin loss over resolved negative indices, expose CLI/sbatch knobs, and validate through unit tests before any training job.

**Tech Stack:** Python, PyTorch, pytest, existing CLSTR Stage0 scripts.

---

### Task 1: Preserve Explicit Negatives In Stage0 Rows

**Files:**
- Modify: `clstr/retrieval_warmup.py`
- Test: `tests/test_stage0_biencoder_protocol.py`

- [x] Add a test proving unified-v2 loading aggregates `negative_skill_ids` per query/source group.
- [x] Add a test proving `_filter_queries_to_skill_pool` resolves `negative_indices` and excludes positives from negatives.
- [x] Implement loader and filter changes.
- [x] Run the targeted tests.

### Task 2: Add Explicit-Negative Stage0 Loss

**Files:**
- Modify: `clstr/retrieval_warmup.py`
- Test: `tests/test_stage0_biencoder_protocol.py`

- [x] Add a test for `_explicit_negative_margin_loss` where a negative above a positive creates loss.
- [x] Add a test proving `run_stage0_biencoder_train` forwards explicit-negative settings.
- [x] Implement the loss and include it in total Stage0 loss only when weight is greater than zero.
- [x] Add metrics/report fields for unweighted and weighted explicit-negative loss and usable row count.
- [x] Run the targeted tests.

### Task 3: Expose CLI And Sbatch Knobs

**Files:**
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Test: `tests/test_stage0_biencoder_protocol.py`

- [x] Extend the existing CLI/sbatch test to check `--explicit_negative_loss_weight`, `--explicit_negative_margin`, and their env defaults.
- [x] Add CLI args and sbatch env vars.
- [x] Run the targeted tests.

### Task 4: Verify And Commit

**Files:**
- Modified files from Tasks 1-3.

- [x] Run `python -m compileall -q clstr scripts tests`.
- [x] Run `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_biencoder_protocol.py -q`.
- [x] Update planning progress.
- [x] Commit the code and docs.

### Task 5: Smoke-Gate The Negative-Aware Objective

**Files:**
- Output: `outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_neg*_smoke1200/`
- Audit: `outputs/clstr_stage0_handoff_coverage_audit/function_aug_v2_true_adapt_proj_neg*_smoke1200_rows2048/`

- [x] Run `explicit_negative_loss_weight=0.05`, margin `0.1`, 1200-step smoke.
- [x] Run dependent Stage0 handoff coverage audit on 2048 rows.
- [x] Compare against `function_aug_v2_true_adapt_proj_smoke1200_rows2048`.
- [x] Run one stronger-weight smoke before deciding whether the objective is worth full Stage0.

2026-07-02 `neg005` result:

- Jobs: smoke `104736`, audit `104738`; both completed with exit code `0:0`.
- Smoke checkpoint: `outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_neg005_smoke1200/checkpoints/clstr_unified_retrieval_v2-step1200.pt`.
- Audit gate passed, but did **not** show a meaningful improvement over the no-explicit-negative 1200-step baseline.
- Baseline vs `neg005`, global: current@100 `0.750283 -> 0.751416`, current@200 `0.796149 -> 0.798981`, next@100 unchanged `0.779661`, next@200 `0.833686 -> 0.831568`, next@500 `0.893008 -> 0.891949`.
- ToolBench-G3 next@100/200/500 unchanged: `0.647059 / 0.764706 / 0.846154`.
- TrajectBench next@200/500 regressed slightly: `0.524887 -> 0.515837`, `0.696833 -> 0.692308`.
- Decision: do **not** launch full Stage0 from `neg005`. The explicit-negative margin at weight `0.05` is too weak or misaligned for the handoff coverage goal. Try a stronger smoke, then stop this direction if audit still fails to improve.

2026-07-02 `neg050` result:

- Jobs: smoke `104801`, audit `104803`.
- Cancelled at step `333/1200` after about 20 minutes because it did not show a qualitatively different signal and the `neg005` audit already showed no handoff benefit.
- Last observed batch metric before cancellation: weighted explicit-negative loss about `0.050`, recall@100 `0.820`, recall@350 `0.898`. These are batch diagnostics only and not enough to justify completing the smoke.
- Decision: stop tuning explicit `negative_skill_ids` weight. The more relevant failure mode is current-model high-scoring false positives, so the next prototype should use online mined hard negatives from the full-pool logits.

### Task 6: Add Online Mined Hard Negatives

**Files:**
- Modify: `clstr/retrieval_warmup.py`
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Test: `tests/test_stage0_biencoder_protocol.py`

- [x] Add an opt-in `_mined_hard_negative_margin_loss` that masks positives and mines the top scoring non-positive logits per batch.
- [x] Keep defaults disabled so existing Stage0 behavior and checkpoints remain interpretable.
- [x] Expose `mined_hard_negative_loss_weight`, `mined_hard_negative_margin`, and `mined_hard_negative_top_k` through wrapper, CLI, and sbatch.
- [x] Fix Stage0 CLI `--help` to lazy-import the training stack so help/tests do not load the heavy training path.
- [x] Run targeted protocol tests and compile checks.
- [x] Run mined-hard-negative 1200-step smoke plus the same 2048-row handoff audit.
- [ ] Run mined-hard-negative full Stage0 if the 1200-step audit improves ToolBench-G3 and TrajectBench handoff coverage.

2026-07-02 `minedhn020` smoke result:

- Jobs: smoke `104819`, audit `104820`; both completed with exit code `0:0`.
- Config: `MINED_HARD_NEGATIVE_LOSS_WEIGHT=0.2`, `MINED_HARD_NEGATIVE_MARGIN=0.1`, `MINED_HARD_NEGATIVE_TOP_K=32`; explicit negative loss disabled.
- Smoke checkpoint: `outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_minedhn020_smoke1200/checkpoints/clstr_unified_retrieval_v2-step1200.pt`.
- Audit gate passed and showed a real same-row improvement over both baseline and `neg005`.
- Global baseline -> minedhn020: current@100 `0.750283 -> 0.759343`, current@200 `0.796149 -> 0.810872`, current@500 `0.869196 -> 0.880521`, next@100 `0.779661 -> 0.791314`, next@200 `0.833686 -> 0.844280`, next@500 `0.893008 -> 0.904661`.
- ToolBench-G3 baseline -> minedhn020: next@100 `0.647059 -> 0.669683`, next@200 `0.764706 -> 0.782805`, next@500 `0.846154 -> 0.855204`.
- TrajectBench baseline -> minedhn020: current@500 `0.649321 -> 0.687783`, next@100 `0.411765 -> 0.438914`, next@200 `0.524887 -> 0.552036`, next@500 `0.696833 -> 0.737557`.
- Decision: this objective is worth one full Stage0 run. It directly addresses current-model false positives and does not add a new stage or model component.
