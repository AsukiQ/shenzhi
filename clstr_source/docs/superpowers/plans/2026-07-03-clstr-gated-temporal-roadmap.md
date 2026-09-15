# CLSTR Gated Temporal Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a time-efficient CLSTR improvement path that first tests whether prior-preserving gated temporal reranking can reduce Stage0-prior damage, then advances the remaining worthwhile method ideas only after clear go/no-go decisions.

**Architecture:** Keep the current clean Stage0 and Stage1/2 baseline as the control. Add a new gated temporal scoring path that treats Stage0 as a semantic prior and uses trainable, state-conditioned residual correction only when useful. Avoid repeated full retraining by using no-train gate sweeps, cached handoff artifacts, frozen-head gate tuning, hard-slice evaluation, and explicit promotion gates.

**Tech Stack:** Python, PyTorch, existing CLSTR training/eval scripts, Slurm on A800, JSONL metrics, existing Stage0 top-M handoff and Stage1/2 training pipeline.

---

## Control Run

Current control run:

- Job: `105539`
- Output: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000`
- Data root: `data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean`
- Stage0 checkpoint: `outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- Training: Stage1 `3000`, Stage2 `10000`, batch `16`, quota sampler.

Do not discard this run. It is the clean baseline for every later decision.

---

## Phase 0: Finish And Audit The Current Baseline

**Purpose:** Establish whether the current clean retrain is already acceptable before adding method changes.

**Files:**
- Read: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/stage1_quality_gate.json`
- Read: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/stage2_quality_gate.json`
- Read: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/stage2_full_base/training_metrics.jsonl`
- Produce summary: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/baseline_decision_summary.json`

- [ ] Monitor `105539` until completion or failure.
- [ ] If Stage1 gate fails, inspect blockers before changing code.
- [ ] If Stage2 gate fails, inspect whether failure is absolute quality or only drop-vs-prior.
- [ ] Compute rolling metrics over first/last 200 Stage2 rows:
  - `loss`
  - `transition_skill_ce_loss`
  - `transition_skill_recall@5`
  - `stage0_prior_transition_skill_recall@5`
  - `transition_skill_mrr`
  - `stage0_prior_transition_skill_mrr`
  - `transition_delta_vs_stage0_prior_mrr`
  - `transition_worse_than_stage0_prior_fraction`
- [ ] Run quick eval only if Stage2 gate is `ok`:
  - ToolBench-G3
  - tau2
  - ToolSandbox
- [ ] Record whether this baseline is good enough to keep as mainline even if gated experiments fail.

**Baseline promotion rule:** Keep the baseline if Stage2 gate is `ok` and fast eval does not regress relative to the latest accepted main-table CLSTR by more than normal run noise.

---

## Phase 1: Build Hard-Slice Diagnostics Before Training New Modules

**Purpose:** Avoid full retraining for ideas that cannot fix known failure modes.

**Files:**
- Create: `scripts/audit_clstr_prior_damage_slices.py`
- Create: `clstr/prior_damage_slices.py`
- Test: `tests/test_prior_damage_slices.py`
- Output: `outputs/clstr_prior_damage_slices/<run_id>/slice_report.json`

**Hard slices to create:**

- `stage0_top1_wrong_top5_contains_gt`
- `stage0_top1_wrong_top20_contains_gt`
- `clstr_worse_than_stage0_prior`
- `transition_switch_rows`
- `stage0_or_qwen_correct_clstr_wrong` when the eval format supports this comparison
- `gt_missing_from_topM` for Stage0 coverage boundary analysis

- [ ] Write tests for slice construction on small synthetic rows.
- [ ] Implement deterministic slice builder from existing metrics/eval reports.
- [ ] Run slice builder on the current baseline.
- [ ] Save slice sizes and examples.

**Decision rule:** If most failures are `gt_missing_from_topM`, gated reranking cannot solve them; defer to Phase 5 Stage0/hybrid recall. If failures are mostly `clstr_worse_than_stage0_prior`, proceed to gated reranking.

---

## Phase 2: No-Train Gate Sweep

**Purpose:** Test whether conservative gating can reduce damage without training.

**Files:**
- Create: `scripts/sweep_clstr_prior_gate.py`
- Create: `clstr/prior_gate_sweep.py`
- Test: `tests/test_prior_gate_sweep.py`
- Output: `outputs/clstr_prior_gate_sweep/<run_id>/gate_sweep_report.json`

**Gate formulas:**

- `fixed_0_10`
- `fixed_0_15`
- `fixed_0_25`
- `margin_gate`: lower lambda when Stage0 top1-top2 margin is large.
- `entropy_gate`: lower lambda when Stage0 distribution entropy is low.
- `margin_entropy_gate`: combine margin and entropy.

**Metrics:**

- `transition_delta_vs_stage0_prior_mrr`
- `transition_worse_than_stage0_prior_fraction`
- `transition_skill_recall@5`
- `rank_drop_count`
- hard-slice MRR/R@5

- [ ] Write tests for deterministic gate formula outputs.
- [ ] Implement sweep over saved Stage2 logits/candidate reports if available.
- [ ] If logits are not available, add a lightweight replay mode that recomputes scores without training.
- [ ] Run no-train sweep on hard slices first.
- [ ] Run no-train sweep on a representative validation/eval subset.

**Promotion rule:** Continue to trainable gate only if no-train gate reduces `worse_than_stage0_prior_fraction` without materially lowering R@5.

---

## Phase 3: First Batch Method Change

**Name:** CLSTR-Gated Temporal Reranker.

**Purpose:** Add the minimum method changes likely to improve CLSTR without exploding variables.

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify or create: `clstr/gated_temporal_reranker.py`
- Modify: `scripts/run_clstr_stage12_consolidated_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage12_consolidated_function_aug_v2.sh`
- Test: `tests/test_gated_temporal_reranker.py`
- Test: `tests/test_full_base_train.py`
- Test: `tests/test_stage12_consolidated_train.py`

**Method components to implement together:**

1. State-conditioned gate:

```text
final_score_i = stage0_score_i + lambda_t * residual_score_i
lambda_t = sigmoid(g(features_t)) * lambda_max
```

Initial settings:

- `lambda_max = 0.5`
- gate bias initialized so mean lambda is close to `0.25`
- features: Stage0 margin, candidate entropy, current-skill-present flag, transition relation features, optional state embedding summary.

2. Prior-preserving loss:

```text
loss = listwise_nll
     + alpha * KL(final_distribution || stage0_distribution)
     + beta * rank_drop_penalty
```

Initial settings:

- `alpha = 0.03`
- `beta = 0.05`
- rank-drop penalty only applies when Stage0 ranks a positive in top5/top10 and final score moves it below that bucket.

3. Candidate-set context pooling:

```text
candidate_context = attention_or_deepsets(query_embedding, candidate_embeddings)
residual_i = MLP([query_embedding, candidate_embedding_i, candidate_context, stage0_score_i])
```

Initial implementation should be lightweight:

- no full transformer
- no all-67k attention
- top50/top64 candidate context only
- toggle via scoring mode

4. Freeze-first training mode:

Initial gated training should freeze existing Stage1/2 heads and train only:

- gate MLP
- candidate context projection
- any new residual adapter parameters

Do not retrain Stage0 during this phase.

**New scoring mode:**

```text
stage0_prior_gated_temporal_residual
```

Keep existing mode:

```text
stage0_rank_prior_plus_transition_residual
```

**Training shortcut:**

- Reuse cached Stage0 candidate handoff when possible.
- First smoke: Stage1 `500`, Stage2 `1000`, batch `32`, LR `1e-4`.
- If OOM, fallback to batch `16` with gradient accumulation to effective batch `32`.

**Promotion rule:**

Promote to gated full only if:

- `transition_worse_than_stage0_prior_fraction` decreases by at least 20% relative to baseline on hard slices, or absolute value is below a clearly acceptable threshold.
- `transition_delta_vs_stage0_prior_mrr` is non-negative.
- R@5 drop vs Stage0 prior is not worse than baseline.
- No ToolBench/tau2 quick eval regression beyond run noise.

---

## Phase 4: Gated Full And Fast Benchmarks

**Purpose:** Only run expensive full training after Phase 2/3 evidence is positive.

**Files:**
- Output: `outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_gated_<tag>`
- Reports: fast eval outputs for ToolBench-G3, tau2, ToolSandbox.

- [ ] Submit gated full Stage1/2 only after smoke passes promotion rule.
- [ ] Monitor full with 15-minute interval while Stage1/2 runs.
- [ ] Compute final Stage2 gate and rolling windows.
- [ ] Run quick eval on ToolBench-G3, tau2, ToolSandbox.
- [ ] Compare against current baseline and old accepted table.

**Decision rule:**

- If gated full wins or clearly reduces prior damage without hurting main metrics, it becomes the new Stage1/2 mainline.
- If gated full is neutral, keep current baseline and do not send gated mode to Stage4.
- If gated full loses, revert to baseline and move to Phase 5 only if failure is Stage0 coverage.

---

## Phase 5: Skill Pool Audit And Stage0 / Candidate Recall Improvements

**Purpose:** Address failures that gated reranking cannot solve because gt skill is absent from candidates.

**Only start if Phase 1 shows many `gt_missing_from_topM` or low Stage0 coverage.**

### Phase 5A: Non-Destructive Skill Pool Audit

**Purpose:** Determine whether the expanded skill pool itself is hurting retrieval before changing ids, deleting skills, or retraining Stage0.

**Files:**
- Create: `scripts/audit_clstr_skill_pool_quality.py`
- Create: `clstr/skill_pool_quality_audit.py`
- Test: `tests/test_skill_pool_quality_audit.py`
- Output: `outputs/clstr_skill_pool_quality_audit/<data_tag>/skill_pool_quality_report.json`

**Audit dimensions:**

- exact duplicate skill text
- normalized duplicate skill text
- near-duplicate clusters by embedding similarity
- alias/equivalent group coverage
- source-wise skill counts
- source-wise gt coverage@50/@100/@350/@500
- benchmark-wise gt coverage@50/@100/@350/@500
- high-confusion skill pairs where Stage0 ranks a near-duplicate above gt
- low-quality or under-specified skill text rows

**First rule:** Do not delete skills or change existing `skill_id` during audit.

**Preferred first fix after audit:**

```text
keep all existing skill_id values
+ add alias/equivalent groups
+ add duplicate_cluster_id
+ add source_quality / low_quality flags
+ treat equivalent skills as positives during training/eval
```

This avoids breaking checkpoint compatibility while still reducing false negatives caused by duplicate or equivalent skills.

**Promotion rule:** Only move from audit to destructive pool cleaning if non-destructive alias/equivalent handling cannot recover coverage or ranking quality.

### Phase 5B: Candidate Recall Improvements

**Candidate improvements:**

1. Hard-negative curriculum:

```text
easy random negatives
-> same-benchmark hard negatives
-> same API/tool-family hard negatives
-> rows where gt rank is outside topM
```

2. Hybrid retrieval:

```text
dense topK
+ lexical/schema topK
+ trajectory-prior topK
+ learned or deterministic non-oracle fusion
```

3. Skill-pool cleaning after Phase 5A audit:

```text
near-duplicate detection
alias/equivalence group update
source-quality filtering
per-benchmark coverage audit
```

**Promotion rule:** Only retrain Stage0 if Phase 5A/5B shows coverage@M or downstream fast eval can plausibly improve. Do not run 5000-step Stage0 full for cosmetic loss changes.

---

## Phase 6: Stage4 Preference Adaptation

**Purpose:** Make Stage4 useful without depending on noisy free-form executor RL.

**Only start after a stable Stage1/2 mainline exists.**

**Method:**

```text
positive = gt next skill or successful trajectory skill
negative = CLSTR wrong selected skill or failed trajectory skill
loss = pairwise ranking / DPO-style preference loss
```

**Negative buckets:**

- gt in candidates but model selected wrong skill
- gt missing from candidates, use only for Stage0/candidate generator update
- transition switch mistaken as self
- self mistaken as switch
- qwen-only-correct but CLSTR-handoff-wrong

**Promotion rule:** Stage4 must improve at least one temporal/multistep benchmark without hurting static routing benchmarks materially.

---

## Phase 7: Later Method Extensions

These are worthwhile but should not be mixed into the first batch.

1. Small trajectory encoder:

```text
last-k [state, selected skill, action, observation] -> trajectory embedding -> next skill rerank
```

Use only if current-state gated reranker saturates.

2. Dynamic / unseen skill calibration:

```text
new skill enters pool
-> rely more on text retrieval
-> lower transition prior for unseen skills
-> few-shot preference update if trajectories exist
```

Use only after mainline is stable.

3. Full candidate-set attention:

Start with lightweight pooling in Phase 3. Consider transformer-style candidate interaction only if pooling helps and complexity is justified.

4. Batch-size and optimization sweep:

Only after method is stable:

- batch32 LR `1e-4`
- batch32 LR `1.5e-4`
- gradient accumulation effective batch64

Do not mix LR changes into first gated smoke.

---

## Time-Saving Rules

- No new full run without passing a hard-slice gate.
- No Stage0 full retrain unless coverage analysis demands it.
- No destructive skill pool cleaning before non-destructive skill pool audit.
- Do not change existing `skill_id` values unless a separate migration plan is written.
- Reuse Stage0 candidate handoff whenever the data/checkpoint/topM/scoring input are unchanged.
- Use no-train gate sweeps before training new gate parameters.
- Train only new gate/context parameters before unfreezing old heads.
- Run ToolBench/tau2/ToolSandbox quick eval before long benchmarks.
- TrajectBench remains last because it is slow and currently lower signal.

---

## Immediate Next Actions After Current Job

1. Finish `105539`.
2. Write `baseline_decision_summary.json`.
3. Build prior-damage hard slices.
4. Run no-train gate sweep.
5. If sweep is promising, implement Phase 3.
6. If sweep is not promising and failures are mostly missing-candidate, skip gating and move to Phase 5A skill pool audit before any Stage0 retrain.
