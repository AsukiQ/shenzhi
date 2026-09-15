# CLSTR Stage4 Completion And Comparison Plan 2026-06-22

## Objective

完成一版可发表口径下站得住的 CLSTR 训练与评估闭环：

1. Stage0/1/2 保持当前 prior-preserving 主线。
2. Stage4 修成通用 logged-online trajectory adaptation，不做 AppWorld 或 benchmark 特化规则。
3. 在同一候选池、同一 eval split 上报告 CLSTR 与 SkillRouter-style baseline 的差异。
4. 只有 smoke/gate 证明有效才提交 full job，避免继续烧无效任务。

## Current Baseline

当前最稳 Stage4 是 source-gated online memory：

- Full report: `outputs/logged_online_stage4_full/mixed_v4_2_rankprior_stage2full_trajprefix_balanced_memory_sourcegate_l0_eval512_20260621_b/train_stdout.json`
- Aggregate MRR: `0.2955 -> 0.3078`
- ToolBench-only MRR: `0.1765 -> 0.2225`
- Gate keeps helpful sources and disables unsupported sources.

新的 trainable Stage4 calibrator 已经接入，但还不能直接 full：

- It is zero-initialized and therefore no-op before training.
- It trains only a small adapter over generic score evidence.
- It restores best held-out adapter instead of blindly using the last update.
- It is row-wise no-op when a source has no online memory evidence, so source gate remains protective.
- Latest smoke with 3-feature safe calibrator was safe but weak: MRR improved over initial by only about `+0.0006`.

## Stage4 Repair Plan

### Step 1: Finish 6-Feature Trainable Calibrator

Features:

- Stage0 prior score.
- Stage2 transition residual score.
- Online memory score.
- Stage0 prior x online memory.
- Stage2 residual x online memory.
- Online memory hit indicator.

Safety design:

- Zero init preserves previous logits.
- Calibrator-only checkpoint stores only adapter weights.
- `prior_eval` disables calibrator explicitly.
- Rows without online-memory evidence get zero calibrator delta.
- Source-gated disabled benchmarks stay unchanged.

Pass gate for moving beyond smoke:

- Final/best held-out MRR must beat initial memory-gated MRR by at least `+0.005`.
- No benchmark may regress MRR or recall@5 by more than `0.01` relative to its own initial/gated value.
- ToolBench and at least one other active source should show non-negative delta.

If this gate fails after two low-cost variants, stop trainable Stage4 full and use source-gated memory as the main Stage4 result.

### Step 2: Low-Cost Smoke Variants

Run at most two additional GPU smokes:

1. **6-feature calibrator smoke**
   - Same sanitized mixed caps as current smoke.
   - `TRAIN_SCORE_CALIBRATOR=1`.
   - `MAX_UPDATES=80`.
   - Best-on-heldout selection.

2. **Learning-rate/regularization smoke only if needed**
   - Same data and split.
   - Try smaller LR if the first smoke overfits or oscillates.
   - Do not change architecture again unless both smokes fail with a clear diagnostic.

### Step 3: Stage4 Full If Smoke Passes

Full trainable Stage4 job:

- Input: `.tmp/stage4_sanitized_mixed_full/trajectories_visible_global_full.jsonl`
- Skill pool: `data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl`
- Stage0 checkpoint: `outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- Stage2 checkpoint: `outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt`
- Stage0 top-M: 350.
- Online memory: `auto_variant`, no `state_conditioned` by default.
- Trainable params: score calibrator only.
- Save latest and best adapter checkpoint continuously.

Full pass gate:

- Aggregate final MRR must exceed both Stage0/Stage2 prior and initial memory-gated Stage4.
- Per-source regressions must be gated off or below `0.01`.
- Report must include prior, initial, best, final, delta-vs-prior, delta-vs-initial, and per-source tables.

If trainable full fails but memory-gated full remains positive, report trainable Stage4 as a negative/ablation result rather than forcing it into the main claim.

## Complete CLSTR Training Definition

For the paper-facing run, "complete CLSTR" should mean:

1. Stage0 SkillRouter-style bi-encoder retrieval over the full CLSTR skill pool.
2. Stage1/Stage2 prior-preserving trajectory-aware reranking/transition residual.
3. Stage4 source-gated logged-online memory/calibrator adaptation.
4. Evaluation with no GT-derived inventory, no candidate injection at eval, and no AppWorld-specific prompt rule.

This avoids claiming executor RL if the robust result is actually logged-online routing adaptation.

## Comparison Plan

### Primary Baselines

Use existing baselines first, rerun only if a required split is missing:

- SkillRouter frozen retrieval:
  - `outputs/skillret_official_eval/baselines/skillrouter_frozen_full/metrics.json`
- SkillRouter-style finetune:
  - `outputs/skillret_official_eval/baselines/skillrouter_style_finetune_full/metrics.json`
- CLSTR SkillRouter-initialized retrieval:
  - `outputs/skillret_official_eval/clstr_skillrouter_init_full/metrics.json`
- Current Stage0 SkillRouter-frozen baseline:
  - `outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json`

### Routing Metrics

Report the same metrics for SkillRouter-style retrieval and CLSTR:

- recall@1, recall@5, recall@20, recall@50, recall@100, recall@350 where available.
- MRR.
- positive coverage@M.
- candidate count and missing-positive rate.
- per-benchmark breakdown.

### Multi-Step Stage4 Metrics

SkillRouter is primarily a single-step router, so the fair Stage4 comparison is:

- Stage0/SkillRouter-style prior on the same Stage4 rows.
- CLSTR Stage2 prior-residual on the same rows.
- CLSTR Stage4 memory/calibrator final on the same rows.

This demonstrates CLSTR's added value as trajectory-aware adaptation rather than claiming SkillRouter failed at a task it was not designed for.

### Optional Downstream Checks

Only after routing metrics are stable:

- AppWorld/Qwen executor smoke as a noisy case study.
- Do not use AppWorld as the main Stage4 success criterion.
- Do not add task-specific API or prompt hacks to rescue the result.

## Stop Rules

Stop and write a diagnosis instead of running full if:

- Smoke does not beat initial memory-gated Stage4.
- A source-gated disabled benchmark is modified by the trainable adapter.
- Stage0 next-positive coverage falls below the previous sanitized smoke level.
- Job runs longer than expected without metrics/checkpoints.
- The same failure repeats in two smoke variants.

## Next Actions

1. Finish verification of the 6-feature calibrator.
2. Run one 6-feature mixed sanitized smoke.
3. If it passes, submit trainable Stage4 full and monitor.
4. If it fails, keep source-gated memory full as Stage4 mainline and generate the final comparison report.
5. Build a concise CLSTR vs SkillRouter table from existing metrics and only rerun missing baselines.
