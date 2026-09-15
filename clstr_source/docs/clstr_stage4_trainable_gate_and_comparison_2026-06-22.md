# CLSTR Stage4 Trainable Gate And Comparison Notes 2026-06-22

## Current Decision State

The robust Stage4 result is still the source-gated logged-online memory route.

Trainable Stage4 has been gated:

- `93918` low-LR 6-feature calibrator smoke failed the trainable-full gate.
- The smoke also exposed a frozen-base buffer mutation bug in calibrator-only training.
- The bug is fixed and covered by regression tests.
- Repaired smoke `93921` completed at:
  `outputs/logged_online_stage4_smoke/mixed_sanitized_visible_global_calibrator6_train_caps512_eval64_lr1e2_updates160_20260622_g`

Repaired smoke `93921` result:

| Metric | Prior | Initial memory-gated | Best/final trainable | Trainable delta vs initial |
|---|---:|---:|---:|---:|
| recall@1 | 0.0625 | 0.0917 | 0.0958 | +0.0042 |
| recall@5 | 0.5542 | 0.5750 | 0.5750 | +0.0000 |
| MRR | 0.2816 | 0.3113 | 0.3146 | +0.0033 |

Decision:

- Do not run trainable Stage4 full: the repaired smoke is positive but below the pre-set `+0.005` MRR full gate.
- Keep the source-gated logged-online memory result as the main Stage4 result.
- Report the trainable calibrator as a safe but weak ablation unless a future, better-justified training signal is introduced.

## Static SkillRET Retrieval

These are official SkillRET static retrieval metrics. They should not be mixed with multi-step Stage4 metrics as if they were the same benchmark.

| Method | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | Recall@15 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| SkillRouter frozen | 0.6812 | 0.7014 | 0.7012 | 0.7532 | 0.7794 | 0.6304 |
| SkillRouter-style finetune | 0.6837 | 0.7040 | 0.7033 | 0.7552 | 0.7820 | 0.6330 |
| CLSTR SkillRouter-init | 0.6822 | 0.7033 | 0.6994 | 0.7548 | 0.7841 | 0.6327 |

Interpretation:

- CLSTR is competitive on static SkillRET retrieval but does not clearly dominate SkillRouter there.
- The paper claim should not be "we simply beat SkillRouter on static retrieval"; the stronger CLSTR claim is trajectory-aware correction/adaptation on top of a strong retriever.

## Unified CLSTR Stage0 Retrieval

These use the CLSTR unified Stage0 qrels/run setup.

| Method | NDCG@20 | NDCG@50 | NDCG@100 | Recall@20 | Recall@50 | Recall@100 | MAP@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SkillRouter frozen over CLSTR qrels | 0.3535 | 0.3736 | 0.3859 | 0.5387 | 0.6305 | 0.6993 | 0.3121 |
| CLSTR Stage0 trained | 0.4700 | 0.4934 | 0.5074 | 0.6903 | 0.7973 | 0.8744 | 0.4162 |

Interpretation:

- Stage0 training materially improves retrieval over the frozen SkillRouter-style baseline on the unified CLSTR data.
- This supports keeping Stage0 as the foundation rather than replacing CLSTR with pure SkillRouter.

## Stage2 Trajectory-Aware Residual

Stage2 full checkpoint:

`outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt`

| Metric | Stage0 prior on Stage2 candidates | Stage2 final |
|---|---:|---:|
| recall@1 | 0.3486 | 0.4148 |
| recall@5 | 0.8602 | 0.8627 |
| MRR | 0.5397 | 0.5846 |

Additional Stage2 diagnostics:

- delta recall@1: `+0.0661`
- delta recall@5: `+0.0024`
- delta MRR: `+0.0449`
- worse-than-prior fraction: `0.1464`

Interpretation:

- The prior-preserving residual design fixes the earlier negative-transfer failure.
- Most improvement is rank refinement/MRR, not broad recall@5, which is expected because Stage0 already retrieves many positives into top-5/top-M.

## Stage4 Source-Gated Logged-Online Memory

Current best full source-gated/autovariant Stage4:

`outputs/logged_online_stage4_full/mixed_v4_2_rankprior_stage2full_trajprefix_balanced_memory_autovariant_sourcegate_l0_eval512_20260621_a/train_stdout.json`

| Metric | Stage2/prior | Stage4 final | Delta |
|---|---:|---:|---:|
| recall@1 | 0.0681 | 0.0785 | +0.0105 |
| recall@5 | 0.6995 | 0.7152 | +0.0157 |
| MRR | 0.2955 | 0.3078 | +0.0123 |

Selected memory variants:

| Source | Selected variant |
|---|---|
| ALFWorld | none |
| ToolBench-G3 | latest_exact |
| TrajectBench | none |
| WebShop | latest_exact |

Interpretation:

- Stage4 adds positive logged-online trajectory adaptation without executor-specific rules.
- The source gate is important: unsupported sources are not forced to use memory.
- If trainable calibrator remains weak after the repaired smoke, the paper-facing Stage4 should be this calibrated logged-online memory residual, with trainable calibrator reported as an ablation or negative result.

## Final Gate Outcome

Phase 16 trainable full is skipped. The current paper-facing route is:

1. Stage0 trained SkillRouter-style retrieval over the unified CLSTR skill pool.
2. Stage2 prior-preserving trajectory-aware residual.
3. Stage4 source-gated logged-online memory adaptation.
4. Trainable Stage4 score calibrator as an ablation/negative result, not the main claim.
