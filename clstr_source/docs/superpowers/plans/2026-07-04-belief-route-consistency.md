# CLSTR Belief/Route Consistency Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair CLSTR so `m_t`/belief is nonconstant, trained and evaluated through the same path, checkpointed correctly, and reported with strict non-leaky metrics.

**Architecture:** Keep clean Stage0 as the large-pool retrieval foundation. Repair Stage1/2/4 memory semantics, candidate feature semantics, checkpoint persistence, and reporting gates before rerunning Stage1/2 and Stage4. Treat all pre-fix CLSTR table rows as legacy diagnostics until post-fix benchmarks are available.

**Tech Stack:** Python, PyTorch, Slurm, pytest, JSONL route corpora, existing CLSTR scripts under `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`.

---

## Files To Modify

- `clstr/full_base_train.py`
  - Replace retrieval-based memory construction with a shared belief-memory helper.
  - Unfreeze belief scale/bias during Stage1/2.
  - Checkpoint belief scale/bias even when routing foundation is otherwise excluded.
  - Remove frozen routing loss from promotion gates or mark it monitor-only.
- `clstr/belief.py`
  - Keep `subspace_obs()` as the canonical inference memory path.
  - Add optional diagnostics helpers if needed.
- `clstr/stage4_act_train.py`
  - Unfreeze/load belief calibration where Stage4 needs recurrent `m_t`.
  - Report skipped missing-positive rows prominently.
- `clstr/stage_checkpoint_init.py`
  - Ensure Stage2/Stage4/eval checkpoint loading includes belief calibration keys.
- `clstr/model.py`
  - Align `skill_head_context` defaults with training or make context explicit in train/eval reports.
- `scripts/audit_clstr_belief_state.py`
  - New audit CLI for entropy, effective support, m variance, and pairwise cosine.
- `scripts/audit_clstr_report_contract.py`
  - New or extended report guard for strict metrics, no-inject, memory semantics, and train/eval feature semantics.
- `tests/test_belief_route_consistency.py`
  - New tests for train/eval memory path, checkpoint persistence, and non-dead reporting.
- `finalwork/table.md`
  - Keep current pre-fix results clearly marked as legacy diagnostics until rerun.

## Task 1: Add Belief State Audit

- [ ] Create `scripts/audit_clstr_belief_state.py`.
- [ ] Inputs:
  - `--stage0_checkpoint_path`
  - `--skills_path`
  - `--trajectories_path`
  - `--sample_rows`
  - `--output_path`
- [ ] Compute for both `retrieval_logits` and `belief_logits`:
  - logit scale and bias stats
  - entropy and effective support of `softmax(logits)`
  - top1/top5 probability mass
  - `m_t` vector norm
  - pairwise cosine of sampled `m_t`
  - per-benchmark summaries
- [ ] Output blockers:
  - `belief_effective_support_too_large`
  - `belief_pairwise_cosine_too_high`
  - `belief_top5_mass_too_low`
- [ ] Add tests using a tiny fake skill table where high-scale belief passes and low-scale belief fails.

Run:

```bash
pytest -q tests/test_belief_route_consistency.py
```

## Task 2: Canonicalize Memory Semantics

- [ ] In `clstr/full_base_train.py`, replace `_skill_logits_and_memory()` with a helper whose memory path defaults to `subspace_obs(model.skill_table, h)`.
- [ ] Keep retrieval logits available only for Stage0/routing monitors.
- [ ] Add train report fields:
  - `memory_semantics = belief_logits_subspace_obs`
  - `routing_logits_semantics = retrieval_logits_stage0_monitor`
  - `policy_context_semantics`
- [ ] Add a regression test proving Stage2 memory uses `belief_logits`, not `retrieval_logits`.

## Task 3: Train And Persist Belief Calibration

- [ ] In Stage1/2 freeze logic, unfreeze only:
  - `skill_table.logit_scale_belief`
  - `skill_table.skill_bias_belief`
- [ ] Do not unfreeze `skill_table.E`, `skill_table.W`, encoder, or retrieval scale/bias in this repair pass.
- [ ] Initialize belief scale from retrieval scale when loading Stage0 unless explicitly disabled:
  - `logit_scale_belief <- logit_scale_retr`
  - `skill_bias_belief <- skill_bias_retr`
- [ ] Modify checkpoint filtering so belief calibration keys are saved even when frozen routing foundation is excluded.
- [ ] Modify checkpoint loading so belief calibration keys override the Stage0 defaults.
- [ ] Add tests:
  - saved Stage2 checkpoint contains belief calibration keys
  - Stage4/eval load report says belief calibration loaded
  - `logit_scale_belief` is not reset to `log(0.2)`

## Task 4: Align Policy Head Candidate Features

- [ ] Pick one mainline semantics and enforce it everywhere:
  - Preferred conservative option: train policy head on the same cross-encoded state-skill candidate embeddings used by `policy_forward_from_candidates()`.
- [ ] If cross-encoding all policy candidates is too slow, add a `policy_candidate_feature_mode` flag and do not mix modes in the same checkpoint.
- [ ] Make `skill_head_context` explicit:
  - if using `belief_residual`, train with `m_t - h_t`;
  - if using `belief`, eval with `m_t`.
- [ ] Add train/eval report fields:
  - `policy_candidate_feature_mode`
  - `skill_head_context`
- [ ] Add tests that fail if train uses action-text embeddings while eval uses cross-encoder embeddings without an explicit compatibility flag.

## Task 5: Clean Loss And Report Semantics

- [ ] Keep frozen Stage2 routing loss as `routing_monitor_loss`, not a weighted training objective.
- [ ] Remove it from quality-gate improvement claims.
- [ ] Keep `L_trans_skill_ce` and transition candidate reranking as trainable evidence.
- [ ] Report no-gradient replay prefix explicitly:
  - `replay_prefix_gradient = detached`
  - no claim of end-to-end recurrent gradient unless separately enabled.

## Task 6: Strict Metrics And Leakage Guards

- [ ] Extend clean preflight with text near-duplicate checks:
  - normalized query/state text exact match
  - token Jaccard high-overlap match
  - optional embedding nearest-neighbor audit if cached embeddings exist
- [ ] Apply to ToolBench-G3 and TRAJECT-Bench eval sets.
- [ ] Add report contract guard:
  - main metric must come from `strict`
  - retained-only metrics must be nested under `diagnostic_retained_only`
  - `positive_injected_rows > 0` blocks promotion
  - missing-positive skipped rows must be reported

## Task 7: Smoke Gates Before Full Training

- [ ] Run belief audit on clean Stage0 before code changes and save report.
- [ ] Run repaired Stage1/2 smoke from clean Stage0:
  - Stage1 `500`
  - Stage2 `1000`
  - batch `16`
  - same clean data root
- [ ] Required smoke pass:
  - Stage2 transition MRR positive versus Stage0 prior
  - belief audit after Stage2 shows nonconstant `m_t`
  - belief calibration keys load into a fresh model
  - policy feature report matches eval feature report
- [ ] If smoke fails, do not submit full. Diagnose the exact failed gate.

## Task 8: Repaired Full Training

- [ ] Run repaired Stage1/2 full from clean Stage0.
- [ ] Run repaired Stage4 from repaired Stage2.
- [ ] Do not rerun Stage0 unless post-fix strict eval shows candidate recall is the bottleneck.
- [ ] Preserve old-path clean Stage4 as:
  - `legacy_pre_belief_fix_clean_stage4`
  - diagnostic only

## Task 9: Repaired Benchmark Evaluation

- [ ] Re-run in this order:
  - ToolBench-G3 clean strict
  - tau2
  - ToolSandbox
  - ALFWorld only if route diagnostics do not regress
  - TRAJECT-Bench last because it is slower and teacher-forced
- [ ] Update `finalwork/table.md` only with strict post-fix rows.
- [ ] Keep old rows in an appendix/diagnostic section until replaced.

## Task 10: Paper Claim Guard

- [ ] Paper wording may claim belief-state routing only if:
  - `m_t` nonconstant audit passes
  - no-`m_t` or retrieval-only ablation is lower on at least two route benchmarks
  - Stage4 has positive strict contribution on at least one multi-step route benchmark
- [ ] Otherwise, narrow claim to:
  - clean large-pool route reranking with optional state memory diagnostics
  - do not emphasize recurrent belief as proven contributor
