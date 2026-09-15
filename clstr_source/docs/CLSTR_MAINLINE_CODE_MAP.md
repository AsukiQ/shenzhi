# CLSTR vNext mainline code map

This document describes the only code path used by the current Qwen-0.6B
CLSTR experiment. Files not listed here are not automatically obsolete: many
belong to baselines, benchmark adapters, or retained legacy experiments.

## Canonical execution chain

```text
precompute
  -> Stage0 static proposal foundation
  -> static route-query adapter
  -> Stage2 recurrent memory
  -> ToolBench / ToolSandbox / Tau2-base evaluation
```

The orchestration entry point is `scripts/submit_clstr_vnext_full_chain.sh`.

## What to read first

The method-critical implementation is concentrated in these files:

1. `clstr/vnext_core.py` — static query, recurrent memory, transition,
   correction, and final candidate-local scoring;
2. `clstr/vnext_stage0_train.py` — frozen-backbone static proposal training;
3. `clstr/vnext_compressor_train.py` — the active static route-query adapter
   (the historical filename is retained for checkpoint compatibility);
4. `clstr/vnext_stage2_train.py` — causal replay, losses, curriculum, gates,
   checkpointing, and held-out selection;
5. `clstr/vnext_eval.py` — common causal replay and benchmark scorer;
6. `clstr/history_channel.py`, `clstr/vnext_candidates.py`,
   `clstr/vnext_losses.py`, and `clstr/vnext_training.py` — shared contracts.

`clstr/model.py` is the compatibility container around these components. Most
other top-level modules are benchmark adapters, baselines, or archived
pre-vNext experiments; they are not additional stages of the current method.

## Method architecture

`clstr/vnext_core.py`

- `StaticRouteQueryDelta` (around line 102): bounded, zero-initialized static
  route adapter.
- `CLSTRVNextCore` (around line 407): owns the static proposal, route adapter,
  recurrent memory transition/correction, and memory route delta.
- `static_query_components` (around line 461): returns the frozen Stage0 query,
  explicit static route delta, and their sum.
- `queries` (around line 480): combines static and recurrent memory queries.
- `candidate_route_scores` (around line 562): the sole canonical scorer,
  decomposed as base logits + static-route delta logits + memory-delta logits.
- `update_memory` (around line 653): action-conditioned prediction followed by
  optional executed-result correction.

`clstr/model.py` wraps these operations with skill-table lookup and frozen-Qwen
encoding. Its legacy heads remain compatibility dependencies and are not part of
the current vNext objective.

## Stage0 static foundation

`clstr/vnext_stage0_train.py`

- `_build_model` (around line 309): creates the frozen-Qwen CLSTR model.
- `_evaluate_stage0` (around line 510): held-out proposal/static-route metrics.
- `train_vnext_stage0` (around line 780): trains only the accepted Stage0
  projection, initial belief, skill adapter/bias, and unified static query.

The active objective is full-pool multi-positive NLL plus online Top-32 hard
negative margin. The backbone remains frozen.

## Static route-query adapter

`clstr/vnext_compressor_train.py`

- `_prepare_successor_route_rows` (around line 131): creates ordered next-skill
  supervision without target injection.
- `_natural_support_listwise_nll` (around line 240): listwise CE over the
  naturally recalled Top-500 support.
- `evaluate_vnext_candidate_compressor` (around line 420): support-invariant
  static route evaluation.
- `train_vnext_candidate_compressor` (around line 664): canonical mode is
  `static_route_query_residual`; the old candidate MLP remains an explicit
  ablation and is not used by Stage2.

The filename retains historical terminology and is a cleanup target after the
metric gate is passed.

## Stage2 recurrent memory

`clstr/vnext_stage2_train.py`

- `_batch_memory_at_start` and `_apply_replay_events` (around lines 1116-1232):
  reconstruct causal recurrent memory from replay prefixes.
- `_causal_pair_loss` (around line 1370): shuffled/mismatched-history causal
  ranking supervision.
- `_evaluate_stage2_ordinary_dev` (around line 1766): static safety and ordinary
  routing metrics.
- `_evaluate_stage2_causal_dev` (around line 2240): factual versus intervened
  memory evidence.
- `train_vnext_stage2` (around line 2886): trains only recurrent transition,
  correction, and memory-route modules over a frozen Stage0 + route adapter.

## Contracts and evaluation

- `clstr/vnext_training.py`: trainability masks, cache/run contracts, checkpoint
  state filtering, and foundation digests.
- `clstr/vnext_candidates.py`: label-free natural support construction.
- `clstr/vnext_losses.py`: Stage0 hard-negative, no-regret, and causal ranking
  losses.
- `clstr/history_channel.py`: compact causal state and executed-result policy.
- `clstr/vnext_eval.py`: canonical checkpoint restore and common routing eval.
- `clstr/global_pool_skillrouter_eval.py`: ToolBench global-pool corpus.
- `clstr/toolsandbox_route_eval.py`: ToolSandbox corpus/protocol adapter.
- `clstr/tau2_route_eval.py`: Tau2-base corpus/protocol adapter.

## Cleanup policy

Legacy files are removed only after proving that they are unreachable from:

1. the canonical vNext chain above;
2. the retained SR and ToolREx baselines;
3. ToolBench, ToolSandbox, and Tau2-base evaluation;
4. focused regression tests and checkpoint compatibility.

The pre-cleanup repository is recoverable from Git branch
`backup/pre-vnext-prune-20260717`.

Additional exact recovery refs created before the current cleanup are:

- `backup/routequery-0d94a9a-pre-prune-20260718`;
- `backup/recallfix-419a100-pre-prune-20260718`;
- `archive/source-*-20260718` for completed immutable scheduler sources.

The obsolete identity-candidate promotion path has been deleted. A static
import audit finds 24 modules in the train-only closure and 61 after adding the
ToolBench, ToolSandbox, and Tau2-base corpus adapters. The remaining legacy
files will therefore be removed only after those loader dependencies are
extracted and the three final evaluations pass; a naive bulk deletion would
silently break the benchmark protocol even though training still imports.
