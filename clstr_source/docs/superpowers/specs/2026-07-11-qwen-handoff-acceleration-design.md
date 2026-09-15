# Qwen CLSTR Handoff Acceleration Design

> **Superseded implementation strategy (2026-07-12):** Real-Qwen jobs `110477`, `110485`, and `110492` proved that cross-batch exact-query deduplication and merged current/next scoring change BF16 embeddings or candidate order. Keep this document as the original design record, but do not implement Sections 1 or the selected deduplication alternative. The accepted replacement is `2026-07-12-qwen-handoff-exact-schedule-revision-design.md`.

## Goal

Reduce the frozen-Qwen Stage0 top-M handoff cost before CLSTR Stage1/2/4 without changing candidate identities, stable tie-breaking, row filtering, loss masks, memory semantics, or trainable parameters.

The measured Stage1 baseline is the reference:

- 83,998 selected source rows;
- 83,167 retained rows;
- batch size 16;
- roughly 53 minutes for handoff and cache materialization;
- 6,412,419,918-byte exact-key JSONL cache;
- current/next top-500 coverage about 0.989/0.991.

## Scope

This change implements the approved recommended scheme only:

1. combine current and next query planning;
2. encode each exact state query once per handoff run;
3. keep row-specific unified-static scoring and inventory filtering exact;
4. persist raw current/next candidate indices and scores in a compact row-keyed sharded cache;
5. reuse cache hits across different Stage1/2/4 row subsets;
6. provide fixed-row real-Qwen parity, storage, memory, and speed gates.

The schedule-aware training embedding cache is explicitly deferred. It will be designed separately after the handoff gate and Stage1 baseline are complete.

## Alternatives Considered

### Batch-size-only tuning

Increasing the Qwen encode batch from 16 to 64 or 128 is simple and may improve throughput, but it still encodes duplicate current/next states and cannot reuse work across Stage1/2/4. It remains part of the GPU benchmark, not the complete implementation.

### Recommended: exact-query deduplication plus compact row-keyed cache

This is the selected design. It removes redundant Qwen forwards inside one run and permits Stage2/4 to reuse Stage1 rows even when the selected row subset differs. It preserves the existing per-row top-K and post-processing semantics.

### Full training cache in the same change

Adding schedule-aware frozen-Qwen optimizer caching at the same time could save more wall time, but it changes a separate execution boundary and expands the parity surface to gradients and optimizer updates. It is deferred so handoff candidate parity can be isolated and audited first.

## Architecture

### 1. Query planning and in-memory deduplication

For each selected row, serialize the exact current and next query using the existing `_stage0_handoff_query_text()` contract. Build an ordered unique-query table keyed by `(query_mode, serialized_text)` and record the current/next query index for every row.

Encode unique queries in deterministic first-occurrence order. Store the resulting tensors on CPU without changing dtype. During row scoring, gather current and next tensors for the original row order, concatenate them into one scoring batch, call the existing unified-static scorer, split current/next logits, and call the existing stable inventory-aware top-K function.

The optimization caches only frozen no-grad query representations. It does not cache belief updates, transition outputs, route logits used by training, replay memory, or losses.

### 2. Raw candidate records

Separate raw candidate computation from the existing label-aware post-processing. Each raw record contains only:

- current candidate indices;
- current candidate scores;
- next candidate indices;
- next candidate scores.

Positive-missing policy, current-skill hard-negative insertion, loss masking, provenance, coverage metrics, and retained/skipped-row decisions continue to run against the current row every time. This prevents cached labels or historical loss masks from leaking into a later stage.

### 3. Compact row-keyed sharded cache

The cache has two identity levels.

Global identity includes every model/pool setting that changes raw candidate values:

- Stage0 checkpoint content digest;
- unified-static scorer digest;
- ordered skill-pool and skill-embedding digests;
- query mode and prompt-bearing model state;
- top-M and inventory minimum;
- candidate selection version and tie-break policy.

The global identity deliberately excludes the complete selected-row digest.

Each row key hashes candidate-affecting row content:

- exact current query text;
- exact next query text;
- ordered explicit/tool inventory skill IDs.

The cache directory contains an atomic JSON manifest and fixed-size Torch shards. A shard stores row keys plus four dense tensors: current/next int32 indices and current/next float32 scores. It contains no full row text, provenance, replay prefix, or label metadata. Missing rows are computed and appended as new shards; existing shards are immutable.

### 4. Compatibility and rollout

The legacy exact-key JSONL cache remains the default. Add an opt-in cache format `row_sharded_v1`. Existing BGE, SkillRouter, and legacy CLSTR launchers keep their behavior.

The Qwen Stage1/2/4 wrappers may switch to `row_sharded_v1` only after the real-Qwen gate is `ok`. Before promotion, the benchmark wrapper supplies the new format explicitly.

## Data Flow

1. Select the deterministic training row subset.
2. Build the global cache identity and per-row keys.
3. Load compact-cache hits by shard.
4. Plan exact current/next queries for cache misses.
5. Deduplicate and encode miss queries.
6. Score misses with the existing unified-static and stable top-K functions.
7. Append miss records atomically to new shards.
8. Restore raw records to original row order.
9. Run the unchanged label-aware post-processing and report generation.

## Failure Handling

- A global identity mismatch selects a different cache directory; it never reuses stale records.
- A malformed manifest, missing shard, duplicate key with conflicting values, wrong tensor shape/dtype, or incomplete lookup fails closed.
- Cache writes use temporary files followed by atomic rename.
- Cache mode `refresh` ignores existing entries and creates a fresh global entry.
- Unsupported cache formats fail before model or data-heavy work.
- Any candidate-ID, retained-row, loss-mask, or stable-order mismatch blocks production promotion.

## Verification

### CPU tests

- exact current/next queries are deduplicated in first-occurrence order;
- duplicate query texts invoke the fake encoder once;
- accelerated raw records match the legacy raw records exactly;
- label-aware post-processing remains unchanged;
- a second identical run is all cache hits;
- a superset run reuses old rows and computes only misses;
- row keys change when query text or explicit inventory changes;
- compact shards contain only keys and dense candidate tensors;
- corrupt or incompatible cache artifacts fail closed.

### Real-Qwen Slurm gate

Use the selected Stage0 step-5000 checkpoint and a fixed deterministic 2,048-row sample. Compare legacy batch-16 against accelerated batch-16/64/128.

Promotion requires:

- identical retained row identities and order;
- identical current/next candidate IDs and stable tie order;
- maximum score difference within the declared tolerance;
- identical loss masks and positive-missing decisions after post-processing;
- backbone trainable parameter count equal to zero;
- cold accelerated speedup at least 1.25x;
- warm-cache speedup at least 5x;
- compact cache bytes no more than 25% of the legacy JSONL bytes on the same rows;
- no CUDA error, OOM, non-finite value, or fallback to a different scorer/query mode.

Production remains on the legacy path if any required check fails.

## Non-Goals

- no backbone unfreezing or LoRA;
- no change to top-M, candidate union, full-pool loss, counterfactual loss, replay depth, or memory equations;
- no cache of trainable memory/transition/gate outputs;
- no learned reliability gate;
- no deletion of legacy cache support.
