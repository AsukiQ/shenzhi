# CLSTR Hybrid Causal Negative Mining Design

**Date:** 2026-07-11
**Status:** Draft for written user review
**Applies to:** Qwen3-Embedding-0.6B CLSTR ablations after the mechanical Stage0 smoke
**Does not modify:** Slurm job `110269` or the current online-mined-negative control recipe

## Goal

Improve CLSTR's negative supervision without copying SkillRouter or ToolREx in a way that is redundant with CLSTR's full-pool objective. The design should:

- reduce false-negative pressure on equivalent or near-duplicate skills;
- add a small, diverse set of reweighted structured negatives;
- mine history-confusable trajectory examples that directly test whether memory improves next-skill ranking;
- keep the Qwen3-Embedding-0.6B backbone frozen unless the user separately approves unfreezing; and
- earn adoption through a paired fixed-row Stage0 audit rather than replacing the current recipe by assumption.

## Non-Goals

- Do not change the current 80-step smoke or reinterpret its gate.
- Do not import SR/ToolREx checkpoints, prompts, expanded documents, or training artifacts into CLSTR lineage.
- Do not add random negatives merely to increase count; full-pool NLL already sees the complete skill inventory.
- Do not generate synthetic queries in the first ablation.
- Do not add a new training stage or enable the learned reliability gate.
- Do not unfreeze the embedding backbone.

## Why CLSTR Needs a Different Mining Design

The current Stage0 objective is:

```text
stage0_loss = full_pool_multi_positive_nll
            + 0.2 * online_top32_hard_negative_margin
```

For each row, `full_pool_multi_positive_nll` already places every non-positive skill in the denominator. This differs from SR and ToolREx, whose sampled InfoNCE or listwise groups depend heavily on choosing a small explicit negative set. Copying SR's ten negatives into CLSTR without a separate weight or mask would mostly repeat candidates that Stage0 already scores.

The useful SR ideas are instead:

- false-negative filtering;
- semantic, lexical, and same-family diversity; and
- deterministic, auditable mining artifacts.

CLSTR also has a method-specific opportunity that SR and ToolREx lack: mining examples where the visible current state is ambiguous but the trajectory history changes the correct next skill.

## Design Overview

The proposal has three independent components. Each can be disabled without changing baseline behavior.

### Component A: False-Negative Safety Layer

Every proposed explicit or online hard negative receives one of three labels:

```text
known_positive
safe_negative
uncertain_ignore
```

A candidate is `known_positive` when its canonical ID or alias belongs to the row's complete positive closure.

A candidate is `uncertain_ignore` when any of these conditions holds relative to a positive skill:

1. normalized skill name is equal;
2. canonical skill-document character-trigram Jaccard is greater than `0.60`; or
3. frozen base-Qwen document cosine similarity is greater than `0.92`.

`uncertain_ignore` does not become a positive label. It is removed from auxiliary negative losses so incomplete annotation cannot force the model to separate likely equivalents.

Online top-32 mining scans deeper than rank 32 as needed and selects the first 32 `safe_negative` candidates. Reports record:

- candidates inspected;
- exclusions by reason;
- safe-negative fill count and fill rate;
- queries with fewer than 32 safe negatives; and
- representative excluded pairs for audit.

The full-pool NLL remains unchanged in the first ablation. The safety layer initially protects only auxiliary hard-negative terms, avoiding a broad objective change before evidence exists.

### Component B: Structured Stage0 Negative Bank

For each training query, build up to eight filtered negatives:

```text
4 semantic-confusable negatives
2 BM25 lexical negatives
2 same taxonomy / environment / source-family negatives
```

All candidates pass Component A. No random quota is used because easy random skills already contribute to the full-pool denominator.

Source behavior:

- semantic candidates come from frozen Qwen retrieval over the complete skill inventory;
- BM25 reranks the semantic top-50 to avoid an inventory-wide lexical scan per query;
- taxonomy resolves in order from explicit category, environment, then source;
- a source shortfall is filled deterministically from the remaining semantic or BM25 stream, never by an unrecorded random substitution.

The bank is stored by skill ID rather than repeated full document text. Each row records source, source rank, raw score, filter decisions, fallback reason, query identity, and positive IDs.

The Stage0 hybrid objective is:

```text
hybrid_stage0_loss = full_pool_multi_positive_nll
                   + 0.2 * filtered_online_top32_margin
                   + 0.1 * structured_bank_margin
```

The structured-bank weight `0.1` is fixed for the first paired ablation. It is not tuned on held-out benchmark rows.

### Component C: History-Confusable Causal Mining

This component supplies harder rows to the existing Stage2/Stage4 counterfactual utility objective; it does not introduce another core loss.

Candidate row pairs must:

1. use two different trajectory IDs, with both IDs confined to the training split;
2. have highly similar current-state query embeddings;
3. have different valid next-skill labels; and
4. differ in replay prefix, prior action, or earlier observation in a way that can explain the label difference.

Priority is given when the static route ranks the other row's next skill highly, while the repaired dynamic route can use history to prefer the correct skill. This creates a direct training and evaluation target for the paper's central claim: history-conditioned memory should resolve state ambiguity.

Each mined causal pair stores:

- both trajectory and row identities;
- current-state similarity;
- positive and counterfactual skill IDs;
- prefix lengths and history fields used;
- static and dynamic ranks when available;
- split and leakage metadata; and
- the exact Stage0/Stage2 checkpoint fingerprints used for mining.

## Data and Artifact Boundaries

The design creates two versioned artifact families:

```text
outputs/qwen06_clstr_postfix/negative_mining/stage0_bank_v1/
outputs/qwen06_clstr_postfix/negative_mining/causal_pairs_v1/
```

Every artifact manifest must include:

- schema and protocol version;
- Qwen model-directory digest;
- skill-pool and data-manifest digests;
- query/pair counts and source distributions;
- thresholds and quotas;
- filter and fallback counts;
- train/protected-eval overlap report;
- shard identities and merge coverage; and
- producer commit and command configuration.

Mining is resumable and sharded. A merge fails closed on duplicate query IDs, missing shards, fingerprint mismatch, protected-eval overlap, or incomplete expected coverage.

No artifact from the separate SR/ToolREx worktree is a valid CLSTR parent. Code may reuse reviewed generic algorithms, but CLSTR artifacts must be rebuilt from CLSTR's own prompt, skill pool, training rows, and lineage.

## Experimental Protocol

### Step 1: Preserve the Current Control

The currently approved online-only Stage0 recipe remains the control. Job `110269` is only a mechanical smoke and is not a mining comparison.

### Step 2: Audit-Only Pilot

Before full mining or training, process a deterministic 10,000-query training sample and report:

- online top-32 false-negative exclusion rate;
- fraction of rows that fill all eight structured slots;
- semantic/BM25/taxonomy overlap;
- missing-taxonomy and fallback rates; and
- examples of high-cosine or high-overlap exclusions.

The full miner proceeds only if:

- at least 95% of sampled rows fill six or more safe structured negatives;
- fewer than 10% of rows fail to fill 24 safe online negatives; and
- no protected-eval overlap is found.

A failed pilot stops this direction without affecting the current training chain.

### Step 3: Paired 1200-Step Ablation

The control and hybrid runs start from the exact same stable step-0 checkpoint, optimizer state, data order, and random seed. The only changed variables are the safety mask and structured-bank auxiliary term.

Both runs use:

- Qwen3-Embedding-0.6B;
- frozen backbone;
- unified-memory Stage0;
- `clstr_causal_state_v1`;
- the same 67,557-skill pool; and
- the same fixed 2,048-row `checkpoint_state_query` audit.

### Step 4: Adoption Gate

The hybrid checkpoint replaces neither the control nor the main recipe unless all conditions hold:

```text
primary next@100 composite gain versus control >= 0.005
at least one of ToolBench-G3 or TrajectBench next@100 gain >= 0.01
global next@500 does not regress
no benchmark next@500 regression is worse than 0.005
all existing Stage0 release floors remain status=ok
```

Here `primary next@100 composite` is the unweighted mean of global, ToolBench-G3, and TrajectBench `next_recall@100`, matching the existing Stage0 selection metric.

Borderline or failed results retain the online-only control. No weight search follows automatically.

### Step 5: Causal-Pair Evaluation

History-confusable mining is evaluated separately after a valid Stage2 checkpoint exists. Required evidence includes:

- static versus dynamic accuracy on mined ambiguity pairs;
- counterfactual utility margin;
- dynamic safety on non-ambiguous rows;
- trajectory-disjoint validation; and
- performance with replay prefixes removed as a negative control.

This component is adopted only if memory-conditioned ranking improves ambiguity pairs without violating the existing safety tolerance.

## Error Handling and Stop Rules

- Missing or mismatched lineage: stop before mining or training.
- Unfilled quota: record deterministic fallback; fail merge if minimum fill criteria are not met.
- Excessive uncertain filtering: stop at the audit-only pilot.
- Any protected-eval overlap: hard failure.
- Hybrid Stage0 gate miss: retain control and stop tuning.
- Causal pairs that cannot attribute a difference to history: exclude and report.
- Any request to unfreeze Qwen: stop and obtain separate user approval.

## Testing Requirements

Unit and integration tests must cover:

- canonical-ID and alias positive closure;
- normalized-name, trigram, and cosine uncertain filters;
- top-32 safe-negative backfill;
- deterministic 4/2/2 source quotas and fallbacks;
- compact ID-only artifacts and fingerprint validation;
- shard resume and strict merge coverage;
- protected-eval and trajectory-group leakage rejection;
- causal-pair construction with similar state and different history/label;
- disabled-mode parity with the current Stage0 loss and outputs; and
- paired-run lineage proving identical step-0 initialization.

## Risks and Mitigations

### Over-filtering true hard negatives

The first version masks uncertain candidates only from auxiliary margins, not from full-pool NLL. Filter rates and examples are audited before training.

### Redundant structured candidates

Source overlap is measured. The small eight-negative bank and adoption gate prevent a large mining system from becoming mandatory without gain.

### Missing or weak taxonomy

Taxonomy shortfalls fall back to semantic/BM25 candidates and remain visible in reports. The miner does not invent categories.

### Frozen-backbone ceiling

The ablation tests whether cleaner negatives improve the trainable CLSTR projection, skill adapter, initial-belief head, and unified retriever. A weak result does not authorize backbone unfreezing.

### Mining cost and storage

The audit-only pilot precedes full sharding. Artifacts store IDs and scores instead of repeated documents.

## Alternatives Considered

### Keep online-only mining permanently

Fastest and already evidence-backed, but it leaves false-negative pressure and provides no history-specific data contribution.

### Copy SR's complete 4/3/2/1 recipe

Well audited for sampled SR training, but random negatives are redundant under CLSTR full-pool NLL and the design does not target memory utility.

### Generate synthetic negative queries with an LLM

Potentially useful later, but it introduces grounding, leakage, judging, and cost questions before simpler mined evidence is exhausted.

## Recommended Decision

Proceed with the current online-only chain as the main control. In parallel, implement the audit-only pilot for Components A and B after this written design is reviewed and a separate implementation plan is approved. Implement Component C only after Stage2 exists, because its mining signal depends on a valid repaired dynamic checkpoint.
