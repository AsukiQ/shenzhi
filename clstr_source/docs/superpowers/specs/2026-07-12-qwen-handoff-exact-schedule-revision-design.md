# Qwen Handoff Exact-Schedule Revision Design

## Context

The first 2,048-row real-Qwen acceleration gate (`110477`) correctly rejected the original compact cold path. Warm-cache reuse was 16.05–16.29x and compact storage was 10.38% of legacy JSONL, but cold execution was only 0.95–1.02x and candidate parity failed on most rows.

Diagnostics isolated two independent causes:

- changing the BF16 Qwen batch composition changed current/next embeddings by as much as 0.01003 and changed current top-500 sets on every row of the 64-row probe;
- merging current and next scorer calls preserved sets on that probe but changed exact candidate order on current or next rows.

The same diagnostics found one exact-safe optimization: preserve every legacy encoder/scorer call and replace selected-score scalar `.item()` transfers with one tensor gather. This reduced the probe scoring phase from 1.583 seconds to 0.0693 seconds while preserving all current and next candidate IDs and order.

## Goal

Make the compact `row_sharded_v1` handoff path exactly reproduce the legacy batch-16 candidate computation while retaining compact storage and warm-cache reuse. The Qwen backbone remains fully frozen.

## Invariants

For each legacy batch, execution remains:

1. encode the ordered current queries;
2. score the current embeddings with the unified-static scorer;
3. encode the ordered next queries;
4. score the next embeddings with the unified-static scorer;
5. run the existing stable inventory-aware candidate selection for current and next logits.

No query deduplication, batch recomposition, current/next concatenation, scorer merging, dtype conversion, loss change, belief change, or optimizer-cache change is allowed.

## Components

### Vectorized score transfer

Candidate index selection remains centralized in one inventory-aware helper. The legacy function transfers selected scores one scalar at a time and remains the audit baseline. The compact cold function uses the same selected indices, creates a padded index tensor, gathers selected scores once, and transfers the gathered tensor to CPU once. Padding is discarded before constructing `RawStage0Candidates`.

### Schedule-aware identity

The global cache identity includes `encode_batch_size` and `execution_schedule_version=legacy_split_batches_v1` in addition to the existing checkpoint, scorer, skill-pool, prompt, top-M, and stable-ranking digests.

Each row key hashes:

- the complete ordered current-query batch;
- the complete ordered next-query batch;
- every row's ordered visible inventory in that batch;
- the configured encode batch size;
- the row offset within the batch.

This distinguishes duplicate row content at different positions and prevents reuse across a changed BF16 execution context.

### Batch-atomic reuse

Cache lookup is row-sharded for storage, but reuse is schedule-batch atomic. A batch is a hit only when every scheduled row key is present. If any row is missing, all rows in that original batch are recomputed together and the full batch is appended. Cached records remain raw candidates only; label-aware masking, injection, provenance, and filtering run from the current training rows every time.

### Gate and promotion

The real-Qwen gate compares the compact path against a legacy batch-16 reference. Batch 16, 64, and 128 may be measured, but only configurations passing candidate IDs/order, score tolerance, masks/decisions, frozen-backbone, cold-speed, warm-speed, and storage thresholds are eligible. Production wrappers use the fastest passing configuration. If none passes, wrappers remain on legacy JSONL.

## Testing

CPU tests cover schedule-aware identity, duplicate row position, changed batch context, vectorized-vs-legacy score equality, exact current/next call order, batch-atomic subset reuse, shard validation, trainer integration, CLI forwarding, and shell syntax. The final acceptance test is the 2,048-row Slurm gate; CPU tests cannot establish BF16 Qwen parity.

## Non-Goals

- unfreezing any Qwen layer;
- changing Stage0/1/2/4 losses, memory, prompts, or sampling;
- caching optimizer-step Qwen activations;
- weakening the existing parity or speed thresholds;
- treating approximate cosine agreement as candidate parity.
