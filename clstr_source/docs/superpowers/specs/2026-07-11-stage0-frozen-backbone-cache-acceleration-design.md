# Stage0 Frozen-Backbone Cache Acceleration Design

Date: 2026-07-11
Status: Approved for implementation
Scope: Qwen3-Embedding-0.6B CLSTR Stage0 only

## Context

The current Qwen CLSTR Stage0 run keeps the Qwen backbone frozen while training
the encoder projection, the unified-memory initialization and routing heads, the
skill adapter, and retrieval calibration parameters. The online-only control
uses a full 67,557-skill pool, batch size 128, gradient accumulation 2, and
online top-32 hard negatives.

The active 1200-step control is intentionally left unchanged. Its steady-state
throughput is about 16 optimizer steps per minute on one A800, compared with
about 19.8 steps per minute for an older and methodologically different run.
GPU utilization is 100%, checkpoint writes take only seconds, and the process
uses about 45 GB of an 80 GB GPU. Repeated frozen-Qwen forward passes are the
dominant avoidable cost.

The loaded training corpus contains 550,694 usable query groups, but the
deterministic `handoff_balanced` sampler has only 15 buckets. A 1200-step
segment with two microbatches per optimizer step repeatedly visits a much
smaller scheduled subset, expected to contain roughly 30,000--40,000 distinct
query rows. Precomputing every corpus query would therefore add unnecessary
cold-start work and memory traffic.

## Decision

Implement two semantics-preserving accelerations before implementing hybrid
causal negative mining:

1. a schedule-aware CPU BF16 cache of frozen-backbone pooled query features,
   captured before the trainable encoder projection; and
2. a verified Stage0 resume fast path that loads checkpointed `skill_table.E`
   without re-encoding the complete skill pool.

The cache is enabled only when the encoder backbone is fully frozen. It does
not cache the projected state representation `h_t`. The trainable projection,
`initial_belief_head`, `unified_retriever`, skill adapter, scale, and bias paths
remain in the live computation graph.

The current 1200-step control is not restarted or modified. The accelerated
path may be used for the 1200-to-2400 continuation only after numerical parity
and speed gates pass.

## Alternatives Considered

### Full-corpus pooled cache

Caching all 550,694 pooled Qwen outputs is straightforward and would require
about 1.13 GB in BF16. It is not selected because a 1200-step balanced segment
touches only a small fraction of those rows. Encoding the full corpus could
consume more backbone work than the continuation itself needs.

### Lazy per-batch cache

Encoding only cache misses avoids an up-front schedule scan. It is not selected
because adjacent balanced batches overlap heavily, leaving small fragmented
miss batches that underutilize the GPU and complicate deterministic performance
measurement.

### Larger batch size or DDP

The current batch already consumes about 45 GB on an A800. Batch 256 may OOM,
and DDP changes resource requirements and introduces additional engineering
risk. Neither is required for this first acceleration pass.

## Architecture

### Split the encoder at the projection boundary

`StateEncoder` gains two explicit operations:

- `encode_backbone_pooled(text_batch)`: tokenize, run the frozen backbone, and
  pool the last hidden state without applying `proj` or final normalization;
- `project_pooled(pooled)`: cast exactly as the existing path does, apply the
  trainable projection, and apply configured embedding normalization.

`StateEncoder.forward` composes these operations. Uncached behavior therefore
continues through the same implementation used by the cached path.

The cached value is:

```text
u_i = Pool(FrozenQwen(FormatStateQuery(x_i)))
```

Each training microbatch still computes:

```text
h_i = Normalize(Projection(u_i))
m_0 = InitialBelief(h_i)
logits = UnifiedRoute(h_i, m_0)
```

Only `u_i` is detached and cached. No trainable output is cached.

### Deterministic schedule planning

After query filtering, shuffling, and bucket construction, every query row is
assigned a private stable training index. For `start_step ... max_steps`, the
trainer invokes the existing batch sampler with the same microbatch indices it
will use in training. It records the distinct query indices in first-use order.

The planner must not implement a second sampling algorithm. It calls
`_batch_queries_for_step` directly, so cache construction and training cannot
silently diverge when sampling behavior changes.

For modes whose schedule cannot be planned deterministically, explicit cache
mode fails closed. The generic default remains cache-off.

### Cache construction and lookup

The scheduled raw query texts are passed through the model's existing state
query formatting boundary, including the causal prompt, truncation, tokenizer,
pooling mode, and maximum length. Pooled outputs are produced under inference
mode, copied to contiguous CPU BF16 storage, and optionally pinned for
nonblocking transfers.

The cache contains:

- one contiguous `[scheduled_unique_rows, backbone_hidden_size]` tensor;
- a mapping from private training-row index to cache row;
- an identity record covering backbone, prompt contract, tokenizer-relevant
  settings, pooling, dtype, corpus order, sampler, seed, and step interval.

During training, batch row indices gather pooled features in original batch
order. The gathered tensor moves to the model device and is passed through
`project_pooled`. A missing scheduled row is an error, not a silent uncached
fallback, because it indicates schedule drift.

The first implementation keeps the pooled cache in node RAM only. Persistent
cross-job cache files are out of scope until the in-memory path demonstrates a
useful end-to-end speedup.

### Verified skill-table resume fast path

Stage0 checkpoints already contain `model_state_dict["skill_table.E"]`, but the
current trainer rebuilds all 67,557 skill embeddings before loading the resume
checkpoint. The verified fast path skips that rebuild only if all of the
following hold:

- a resume checkpoint is present;
- `skill_table.E` exists and has the exact expected rank and shape;
- the ordered skill-pool identity matches the current pool;
- the relevant checkpoint encoder and skill serialization configuration is
  compatible; and
- Qwen lineage validation has not reported a mismatch.

The ordered skill-pool identity hashes each skill's stable ID together with the
exact serialized skill text used by `SkillTable`. New checkpoints store this
identity in their config and payload. For the existing step-1200 checkpoint,
the trainer may derive and compare the sibling `selected_skills.jsonl` written
by that run before the current output file is rewritten.

When verified resume mode is explicitly requested, missing or mismatched
identity is a hard error. It must never load a same-shaped skill table under a
different skill ordering. Generic legacy training keeps rebuild mode as its
default.

## Configuration

The Stage0 Python entrypoint and shared Slurm launcher expose:

- `frozen_backbone_cache_mode`: `off` or `schedule`;
- `frozen_backbone_cache_batch_size`;
- `resume_skill_table_mode`: `rebuild` or `verified_checkpoint`.

The Qwen Stage0 launcher requests schedule caching and verified checkpoint skill
loading only for accelerated experiments. It continues to set
`TRAIN_ENCODER_BACKBONE=0`. Any request combining schedule caching with a
trainable backbone fails before model-scale work begins.

The cache options, identity, row count, byte count, build duration, lookup hit
count, and skill-table rebuild decision are recorded in setup status, checkpoint
config, and the final training report.

## Numerical Correctness Gates

CPU unit tests must cover:

1. split encoder composition matches the original forward path;
2. schedule planning reproduces the exact training batches and deduplicates only
   storage, not batch occurrences or order;
3. cached lookup produces the same projected embeddings and unified logits as
   uncached encoding;
4. projection, initial-belief, unified-retriever, and retrieval-path gradients
   match;
5. one optimizer update from identical state matches within declared tolerance;
6. cache mode rejects a trainable backbone and detects schedule misses;
7. verified resume skips skill encoding for a matching ordered pool; and
8. verified resume rejects missing, reordered, text-changed, or shape-mismatched
   pools.

A real Qwen GPU parity smoke uses the same checkpoint and fixed query batch for
cached and uncached passes. It records maximum and mean absolute differences for
pooled outputs, projected states, logits, loss, selected gradients, and updated
parameters. BF16 pooled caching is expected to preserve the frozen backbone's
native pooled values; tolerances must be fixed in the audit script rather than
chosen after observing results.

## Performance Gate

The accelerated path is not enabled for the 1200-to-2400 continuation unless a
single-A800 benchmark shows all of the following:

- numerical and gradient parity passes;
- steady-state training throughput is at least 1.5 times the uncached path;
- projected end-to-end continuation time, including cache construction, is at
  least 1.5 times faster than the current resume path; and
- peak GPU memory remains within the A800 allocation without reducing batch
  size, effective batch size, full skill pool, prompt, negative count, or
  precision.

If the speed gate fails, production continuation remains on the uncached path.
The implementation may still keep the verified skill-table resume fast path if
its own parity checks pass.

## Experiment Isolation

Acceleration changes execution only. It must not change:

- query rows or their sampler order;
- positives, aliases, online top-32 negative selection, or loss weights;
- full-pool scoring;
- causal state prompt and truncation;
- batch size or gradient accumulation;
- optimizer state or learning-rate schedule;
- checkpoint continuation step; or
- any Stage0 quality or promotion gate.

Hybrid causal negative mining remains a separate experiment. Its online-only
control is never silently replaced. Once acceleration passes, both the control
and mined-negative branches use the same validated execution path so their
difference remains attributable to supervision.

## Rollout

1. Add RED unit and launcher tests.
2. Implement the encoder split and schedule cache behind default-off options.
3. Implement ordered skill-pool identity and verified resume loading.
4. Run focused tests, the existing Stage0 suite, and checkpoint-lineage tests.
5. Submit a small real-Qwen parity and timing job to Slurm.
6. Enable acceleration in the Qwen continuation launcher only if all gates pass.
7. Complete the fixed step-1200 handoff audit and Stage0 promotion gate.
8. Continue to step 2400 if the Stage0 gate and acceleration gate are both OK.
9. Implement hybrid causal negative mining as a separate paired ablation.

## Non-Goals

- unfreezing any Qwen backbone layer;
- caching projected `h_t`, memory states, logits, or trainable skill features;
- changing the Stage0 method or negative-mining objective;
- DDP or multi-GPU training;
- persistent full-corpus cache infrastructure; and
- altering Stage1, Stage2, or Stage4 semantics in this acceleration pass.
