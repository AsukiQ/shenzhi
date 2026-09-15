# Stage0 Negative-Aware Design

## Goal

Make Stage0 use the explicit `negative_skill_ids` already present in the unified retrieval data, while preserving the existing full-pool multi-positive objective as the coarse-recall anchor.

## Scope

This is a low-cost prototype for the current SkillRouter-initialized mainline. It does not change the backbone, does not unfreeze `skill_table.E`, does not add reranker distillation, and does not add top-K latent skill attention.

## Design

Stage0 continues to optimize `L_retr_full_pool_multi_positive_nll`. The loader additionally keeps resolved explicit negative skill ids and negative indices for each query/source group. A new optional auxiliary loss compares each query's positive logits against the explicitly labeled negative logits:

`L_explicit_neg = mean(max(0, margin + negative_logit - positive_logsumexp))`

The total Stage0 loss is:

`L_stage0 = L_retr_full_pool_multi_positive_nll + weight * L_explicit_neg + preservation_anchor`

The new loss is disabled by default with `explicit_negative_loss_weight=0.0`, so existing configs behave the same unless a run opts in.

## Evaluation

Do not run a blind 5000-step full job. First run unit tests, then a small Stage0 smoke with explicit negatives enabled, then compare fixed handoff coverage against the existing 1200/2400/3600 checkpoints. Continue training only if ToolBench-G3 and TrajectBench coverage improves.

## Paper Positioning

This remains Stage0 retrieval-foundation training. It is not a new stage. In the paper, it can be described as hard-negative-aware retrieval adaptation using existing route supervision.
