# Unified Memory-Conditioned CLSTR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current h-only Stage0 prior plus weak Stage4 residual route with a unified memory-conditioned skill retriever while preserving Stage0-style static retrieval warm-up.

**Architecture:** Keep the training curriculum, but make the inference scorer one model: encode the current state as `h_t`, initialize/update recurrent belief as `m_t`, fuse them into a retrieval vector `z_t`, and rank skills by `z_t dot E_skill`. Stage0 becomes the `t=0` warm-up objective for the same scorer, not a separate h-only prior used at inference.

**Tech Stack:** PyTorch, existing CLSTR model modules (`clstr/model.py`, `clstr/belief.py`, `clstr/full_base_train.py`, `clstr/stage4_act_train.py`), existing sbatch training/eval scripts, pytest.

---

## Current Evidence

- tau2 local-pool true ablation:
  - static no replay MRR: `0.244131`
  - dynamic replay no online MRR: `0.244131`
  - dynamic replay with exact online memory MRR: `0.296849`
  - `m_l2_mean=7.7217`, `residual_max_abs_diff_mean=0.7101`, but `final_positive_rank_changed_rows=0`.
- ToolSandbox local-pool true ablation:
  - static no replay MRR: `0.634369`
  - dynamic replay no online MRR: `0.634369`
  - dynamic replay with online memory MRR: `0.634369`
  - `m_l2_mean=2.7016`, `residual_max_abs_diff_mean=0.3720`, but `final_positive_rank_changed_rows=0`.
- Interpretation: the repaired `m_t` path is active, but the current prior/residual design makes `m_t` too weak to affect ranking.

## Design Decision

Do not delete Stage0-style training. Delete the h-only Stage0 prior as the final routing backbone.

Use this final route:

```text
h_t = Encoder(state_text_t)
b_t = SparseBelief(h_t, E)
m_0 = InitBelief(h_0, b_0)
m_t = Update(m_{t-1}, action_{t-1}, h_t, b_t)
z_t = Fuse(h_t, m_t)
score_t(skill) = z_t dot E_skill + optional skill_bias
```

Use this curriculum:

```text
Warm-up: train z_0 retrieval on single-step/static rows.
Sequential: train m_t/z_t retrieval on trajectory rows with replay prefixes.
Adaptation: fine-tune the same scorer online/offline, not a separate residual reranker.
```

## Success Criteria

- Static retrieval does not collapse: unified `z_0` MRR/top-M recall is close to current h-only Stage0 on ToolBench/APIBank/BFCL static slices.
- Memory matters: on tau2/ToolSandbox/ToolBench trajectory rows, `dynamic z_t` improves over `static z_t` and creates nonzero positive-rank flips.
- Global pool remains viable: clean67k ToolBench strict MRR should not regress more than a small tolerance before full retraining.
- Method is simpler: main paper route can be described without Stage0 prior plus Stage4 residual.

## Stage-Complete Requirement

The redesign is not a Stage0-only change. Stage1/2/4 all must use the same route semantics before any full run is considered valid:

```text
h_t -> m_t -> z_t = Fuse(h_t, m_t) -> z_t dot E_skill
```

- Stage0 trains `z_0` retrieval with `m_0 = initial_belief(h_0)`.
- Stage1/2 trains teacher-forced prefix `m_t` and scores candidates with `unified_route_logits(h_t, m_t)`.
- Stage4 continues the same scorer on logged/online adaptation rows; it must not reintroduce Stage0 prior + transition residual as the main score.
- Eval must call the same scorer as training. Legacy prior/residual is allowed only as an explicit ablation or rollback mode.

## Current Implementation Status

- Done: unified scorer API in `clstr/model.py`.
- Done: sparse `m_0` initialization in `clstr/belief.py`.
- Done: Stage0 `route_scorer=unified_memory` warm-up path and full clean67k run.
  - Checkpoint: `outputs/clstr_unified_memory_stage0_full_b128_20260707_052137/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
  - Final step: loss `2.993819`, R@1 `0.4375`, R@20 `0.796875`, R@50 `0.875`, R@100 `0.9375`.
- Done: Stage1/2 `route_scorer=unified_memory` is wired into `clstr/full_base_train.py`, consolidated scripts, gates, reports, and checkpoint metadata.
- Done: Stage4 `route_scorer=unified_memory` is wired through training, freeze/report metadata, memory-margin diagnostics, and tests.
- Done: Stage2/Stage4 now record dynamic-vs-static rank deltas and support a lightweight memory-margin loss so `m_t` has direct ranking evidence beyond CE.
- Done: common eval scripts accept `route_scorer=unified_memory`; final validation still needs a trained full Stage1/2/4 unified checkpoint.
- Blocked: the first full Stage1/2 submission failed after completing the expensive Stage0 handoff cache and starting Stage1. The next action is to diagnose/fix that failure, then rerun Stage1/2 from the completed Stage0 checkpoint using the cached handoff.

Immediate blocker: repair the full Stage1/2 run, then run Stage4 full with the same `route_scorer=unified_memory`. A full CLSTR checkpoint is not valid unless Stage1/2, Stage4, and eval all use this same route.

## Stage1/2/4 Change Contract

These stages must change together. A run is invalid if any of them falls back to legacy prior/residual scoring as the main route.

- Stage1/2 trains `unified_route_logits(h_t, m_t)` on static rows and replay-prefix trajectory rows.
- Stage1/2 gates validate the route mode and active losses instead of requiring legacy `belief` or action-aware residual fields.
- Stage2 routing losses that depend only on frozen h-only retrieval are diagnostics, not optimization evidence.
- Stage4 continues from the Stage1/2 unified scorer and trains dynamic next-skill retrieval with replayed/logged prefixes.
- Stage4 may use Stage0/topM only to form candidate sets; it must not add `candidate_next_prior_scores` as the main score in unified mode.
- Eval must use the same `route_scorer=unified_memory` path and table metrics must use strict numbers.

## Task 1: Add Unified Scorer Module

**Files:**
- Modify: `clstr/model.py`
- Create or modify tests: `tests/test_model_pipeline.py`

- [x] Add a small `UnifiedMemoryRetriever` module.
- [x] Add `CLSTRModel.unified_route_logits(h_t, m_t, candidate_ids=None)`.
- [x] Unit test full-pool and candidate subset scoring.

## Task 2: Fix Belief Initialization for `m_0`

**Files:**
- Modify: `clstr/belief.py`
- Modify: `clstr/model.py`
- Test: `tests/test_belief.py`

- [x] Implement sparse belief initialization:

```text
b_t = TopKSoftmax(belief_logits(h_t), top_k=K_belief) @ E
m_0 = LayerNorm(W_h h_t + W_b b_t)
```

- [x] Make `belief_top_k`, `logit_scale_belief`, and belief bias trainable in the unified warm-up checkpoint.
- [x] Add tests that `m_0` depends on sparse belief and changes with the query.

## Task 3: Unified Static Warm-Up

**Files:**
- Modify: `clstr/retrieval_warmup.py`
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Test: `tests/test_stage0_biencoder_protocol.py`, `tests/test_stage_checkpoint_init.py`

- [x] Train on existing Stage0/static retrieval data, but optimize:

```text
z_0 = Fuse(h_0, m_0)
CE(z_0 dot E, gold_skill)
```

- [x] Keep h-only retrieval as a monitored baseline, not the final inference path.
- [x] Save checkpoint fields clearly:

```json
{
  "route_scorer": "unified_memory",
  "warmup_objective": "z0_static_retrieval",
  "uses_h_only_prior_at_inference": false
}
```

- [x] Full clean67k warm-up completed.

## Task 4: Stage1/2 Unified Sequential Training

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `scripts/run_clstr_stage12_consolidated_train.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage12_consolidated_function_aug_v2.sh`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Test: `tests/test_full_base_train.py`, `tests/test_stage12_consolidated_train.py`

Stage1/2 must train the same route that Stage0 warm-up and eval will use. It is not enough to keep the current transition-head/residual objective, because that leaves the final retriever as a h-only prior plus a weak add-on.

- [x] Build replayed `m_t` for each trajectory step:

```text
m_t = ReplayUpdate(m_0, prefix_actions, prefix_observations)
z_t = Fuse(h_t, m_t)
score = z_t dot E_candidate
```

- [x] Add `route_scorer` / `sequential_scorer` config with:

```text
legacy_prior_residual
unified_memory
```

- [x] In `unified_memory`, compute all routing CE/listwise losses from:

```text
static/no-history row: m_t = initial_belief(h_t), logits = unified_route_logits(h_t, m_t)
prefix/history row:   m_t = replay_prefix_belief(...), logits = unified_route_logits(h_t, m_t)
```

- [x] Replace the Stage2 transition residual CE main loss with direct unified retrieval CE/listwise loss over candidate rows. The old transition residual loss remains only when `route_scorer=legacy_prior_residual`.
- [x] Make trainable modules explicit for this mode:

```text
initial_belief_head
unified_retriever
belief logit scale/bias if used by m_0
transition/gate only if they are actually used to update m_t
```

- [x] Do not train dead losses. If encoder/skill_table are frozen, any h-only routing contrastive loss is a diagnostic only and must not be logged as an optimization objective.
- [x] Save checkpoint metadata:

```json
{
  "route_scorer": "unified_memory",
  "stage1_2_objective": "teacher_forced_prefix_unified_retrieval",
  "uses_stage0_prior_at_inference": false,
  "legacy_prior_residual_available": true
}
```

- [x] Add a counterfactual memory loss:

```text
score_dynamic(gold) - score_static(gold) > margin
```

only on rows where prefix length is positive and the next skill differs from the static/no-history prediction.

- [x] Track these diagnostics every eval:

```text
static_z_mrr
dynamic_z_mrr
dynamic_vs_static_delta_mrr
positive_rank_improved_rows
positive_rank_worsened_rows
z_final_argmax_changed_rows
```

- [ ] Diagnose and repair the current full Stage1/2 failure, then rerun full Stage1/2 from the completed Stage0 checkpoint and cached handoff.

## Task 5: Stage4 Unified Adaptation

**Files:**
- Modify: `clstr/stage4_act_train.py`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- Test: `tests/test_stage4_act_train.py`

Stage4 must be a continuation of the Stage1/2 unified scorer, not a separate residual reranker over Stage0 scores.

- [x] Add `route_scorer=unified_memory` to Stage4 training.
- [x] In `route_scorer=unified_memory`, compute:

```text
h_t = encode(state_text)
m_static = initial_belief(h_t)
m_dynamic = replay_prefix_belief(m_static, logged prefix)
logits_static = unified_route_logits(h_t, m_static, candidate_rows)
logits_dynamic = unified_route_logits(h_t, m_dynamic, candidate_rows)
loss = CE(logits_dynamic, gold_next_skill) + lambda_margin * max(0, margin - logits_dynamic_gold + logits_static_gold)
```

- [x] Stage4 may still use Stage0/topM handoff to build candidates, but it must not use `candidate_next_prior_scores` as a main additive score in unified mode.
- [x] Keep the existing `stage0_positive_missing_policy=skip` behavior unless explicitly running a miss-recovery experiment. Do not inject gold into candidates for main training/eval.
- [x] Freeze/unfreeze policy for the first version:

```text
freeze encoder and skill_table
train initial_belief_head, unified_retriever, transition/gate modules used by replay
do not train old trans_head residual unless legacy mode is selected
```

- [x] Report diagnostics every eval:

```text
stage4_unified_dynamic_mrr
stage4_unified_static_mrr
stage4_unified_delta_mrr
stage4_unified_rank_improved_rows
stage4_unified_rank_worsened_rows
stage4_unified_argmax_changed_rows
stage4_unified_candidate_source_topm
```

- [x] Save checkpoint metadata:

```json
{
  "route_scorer": "unified_memory",
  "stage4_objective": "unified_dynamic_next_skill_retrieval",
  "uses_stage0_prior_at_inference": false,
  "legacy_prior_residual_available": true
}
```

- [ ] Run Stage4 full after Stage1/2 full succeeds.

## Task 6: Compatibility Evaluators

**Files:**
- Modify: `clstr/tau2_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/global_pool_route_eval.py`
- Test: corresponding eval tests

- [x] Add `route_scorer=unified_memory` to eval scripts.
- [x] Ensure train/eval use the same `m_t` path:

```text
static/no-history = z_0
dynamic/history = z_t after replay prefix
```

- [x] Keep old prior/residual eval as `legacy_prior_residual` only for ablation and rollback.
- [x] Main table should use strict metrics only.

## Task 7: Fast Validation Before Full Retraining

**Files:**
- Create/update: `clstr/mt_ablation_eval.py`

- [x] Run static warm-up smoke on a small slice.
- [x] Run tau2 and ToolSandbox local-pool dynamic-vs-static diagnostics.
- [x] Run ToolBench clean67k smoke.
- [ ] Stop if any of these happen:

```text
z_0 static retrieval collapses versus h-only Stage0.
dynamic z_t still changes logits but has zero rank flips.
ToolBench strict MRR drops sharply before sequential gains appear.
```

## Task 8: Full Training and Tables

**Files:**
- Update: `finalwork/table.md`
- Update: `.planning/.../progress.md` and `findings.md`

- [x] Train unified static warm-up full.
- [ ] Train unified sequential full.
- [ ] Train unified Stage4 full.
- [ ] Run main benchmarks in this order:

```text
ToolBench-G3 clean67k
tau2 local and global-transfer
ToolSandbox local and global-transfer
APIBank
ALFWorld if executor setup is stable
TRAJECT last because it is slow and currently weak
```

- [ ] Run fair SkillRouter baselines trained on the same data where applicable.
- [ ] Report old CLSTR prior/residual as legacy ablation only if it helps explain the redesign.

## Expected Outcome

Best case: unified CLSTR preserves static retrieval and finally shows nonzero, positive dynamic memory contribution.

Acceptable case: static is preserved, dynamic gains are limited but measurable on sequential rows; paper can claim recurrent belief-conditioned retrieval with honest ablations.

Failure case: unified `m_t` hurts static retrieval or still cannot flip ranks. Then the correct decision is to demote recurrent `m_t` and write CLSTR as a stronger static/dynamic evidence router, not as a recurrent memory method.
