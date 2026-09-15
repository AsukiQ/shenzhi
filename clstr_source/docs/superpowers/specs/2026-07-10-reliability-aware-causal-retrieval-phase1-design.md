# Reliability-Aware Causal Retrieval Phase 1 Design

## Goal

Phase 1 strengthens CLSTR's causal-memory claim without changing the backbone or
adding a new policy-training stage. It makes the unified retriever train over the
declared legal skill pool, replaces the positive-logit-only memory margin with a
listwise counterfactual utility objective, and gives Stage2 and Stage4 the same
post-action routing semantics.

This phase implements Tasks 1-5 of
`docs/superpowers/plans/2026-07-10-reliability-aware-causal-retrieval.md`.
Memory-conditioned candidate union and the learned reliability gate remain later,
separately gated work.

## Alternatives considered

### A. Replace only `transition_memory_margin`

This is the smallest code change, but it keeps supervision restricted to the
Stage0 shortlist. Rows whose correct next skill is absent from that shortlist
still cannot teach memory to recover the skill, so the resulting causal claim is
incomplete.

### B. Use full-pool supervision only in Stage2

This fixes the main offline objective but leaves Stage4 optimizing a different
candidate-limited geometry. The same checkpoint would then be trained under two
incompatible definitions of next-skill routing.

### C. Shared full-pool Stage2 and Stage4 objective

This is the selected design. Stage2 and Stage4 use the same tensor-level
full-pool objective and differ only in their outer training loop and loss weight
schedule. It is a larger change than A, but it directly supports the paper claim
and prevents objective drift.

## Causal state contract

For a transition row at time `t`, both comparison branches are evaluated at the
same post-action state:

```text
h_next       = Encode(next_state_text)
m_dynamic    = Correct(Transition(m_t, action_t), Observe(next_observation_text))
m_static     = InitialBelief(h_next)

dynamic_logits = UnifiedRouteFull(h_next, m_dynamic)
static_logits  = UnifiedRouteFull(h_next, m_static)
```

The two branches must use the same skill table, legal-pool mask, positive mask,
state encoding, dtype, and device. Their only semantic difference is the memory
input. The static branch is a detached comparator and receives no gradient from
the counterfactual term.

An empty replay prefix is not a zero-history comparison after the current action:
the post-action transition and observation update still make `m_dynamic`
counterfactual-eligible.

## Full-pool unified scoring API

`CLSTRModel` exposes two focused operations:

```text
unified_route_full_logits(h, m) -> [batch, skill_count]
gather_unified_route_logits(full_logits, candidate_rows) -> padded subset logits
```

The public candidate-subset API delegates to these operations. Full logits are
computed once per branch; gathering must not invoke the retriever or recompute
the skill-table matrix product. Invalid candidate indices fail loudly, and padded
positions use the dtype's minimum finite value.

This split is required by both full-pool training and later static-plus-dynamic
candidate recall.

## Canonical Stage0 handoff

The Stage0 handoff must use the same unified static scorer used by the repaired
model:

```text
h = Encode(text)
m0 = InitialBelief(h)
scores = UnifiedRouteFull(h, m0)
```

The h-only `skill_table.retrieval_logits(h)` path is not a valid fallback in
unified-memory mode. Missing `initial_belief` or full unified scoring is an error.

Candidate selection is deterministic under tied logits: higher score first, then
declared skill-pool index ascending. Cache identity includes the effective scorer,
checkpoint chain, skill embeddings, declared-pool order, query format, selection
version, and tie policy. A cache produced by the previous h-only scorer cannot be
reused.

## Full-pool training eligibility

The model skill table defines the default legal pool. A row may narrow that pool
only through explicit inference-available inventory fields. The implementation
must not infer legality from the target skill, target namespace, benchmark label,
or root prefix.

A causal row is eligible for the full-pool objective when:

- all required causal text fields are present;
- at least one known positive is inside the declared legal pool;
- at least one legal negative exists;
- both score branches are finite on valid entries.

Static Top-M membership is diagnostic only. A static miss remains trainable and
the gold skill is never inserted into the natural shortlist. Unknown-table,
known-but-illegal, no-negative, and nonfinite rows are counted separately.

Equivalent skill IDs form a multi-positive mask. With no explicit inventory,
every column of the supplied skill table is legal. A nonempty explicit inventory
that maps to no known skill yields an all-false legal mask and a named exclusion.

## Shared causal routing objective

For valid skills, define multi-positive log utility as:

```text
U(logits) = logsumexp(logits over positive skills)
          - logsumexp(logits over all legal skills)
```

The main next-skill loss is the weighted mean of `-U(dynamic_logits)` over
eligible rows.

The counterfactual term compares dynamic memory against the detached static
branch:

```text
gain = U(dynamic_logits) - stop_gradient(U(static_logits))
```

Rows are separated by whether the static branch's legal Top-1 is positive:

```text
weak-static row:
    gain_loss = relu(gain_margin - gain)

strong-static row:
    safety_loss = relu(-safety_tolerance - gain)

counterfactual_utility = gain_weight * mean(gain_loss)
                       + safety_weight * mean(safety_loss)
```

The weak branch asks memory to improve difficult rows. The strong branch prevents
memory from damaging rows the static retriever already solves. Weak and strong
means are computed separately so the larger group cannot dominate by count.

Masked or empty groups return an explicit finite scalar zero. The implementation
must not derive zero by multiplying padded minimum logits, which could produce
NaN. Gradients flow through dynamic routing, transition, observation correction,
the recurrent belief gate, and trainable initial belief paths used by replay;
they do not flow through the static comparator.

## Stage2 integration

The unified Stage2 launcher defaults to:

```text
next_skill_pool_mode = full_pool
transition_inventory_mask_mode = explicit_only
counterfactual_utility weight = 0.05
gain_margin = 0.1
safety_tolerance = 0.01
gain_weight = 1.0
safety_weight = 1.0
warmup_fraction = 0.1
```

Generic compatibility helpers may retain candidate-limited behavior only when
explicitly configured. Full-pool mode is valid only with `unified_memory`.

`transition_memory_margin` is removed from active loss keys, metrics, CLI flags,
checkpoint metadata, quality checks, and readiness checks. It is replaced rather
than supplemented, so experiments have one unambiguous memory-specific auxiliary
objective.

Counterfactual weight uses the existing loss-weight dictionary as its single
outer weight source. A linear warmup uses the restored optimizer global step, so
Stage2 resume does not restart the schedule.

## Stage4 integration

Stage4 retains causally valid rows even when the natural Stage0 shortlist is empty
or misses the next skill. It receives the complete declared skill mapping from the
loaded model and validates that it matches the skill table.

Stage4 calls the same tensor-level full-pool objective as Stage2. It uses unit row
weights because full-pool Stage4 forbids positive injection. Its default
counterfactual weight is `0.05` with a `0.05` local warmup fraction.

Stage4 currently has no resume contract. An interrupted run restarts Stage4 and
records that fact; this phase does not claim resume-safe Stage4 warmup.

## Reporting and failure behavior

Reports expose, at minimum:

```text
route_scorer
static_candidate_scorer
next_skill_pool_mode
full_pool_eligible_rows
static_hit_rows
static_miss_rows
counterfactual_weak_rows
counterfactual_strong_rows
counterfactual_gain_violations
counterfactual_safety_violations
positive_not_in_skill_table_rows
positive_outside_legal_pool_rows
explicit_inventory_no_known_skill_rows
no_legal_negative_rows
nonfinite_logit_rows
```

The implementation fails loudly for non-unified full-pool mode, mismatched skill
tables, target-derived global masks, invalid enums, invalid margins/weights,
noncontiguous skill mappings, and stale incompatible handoff caches.

## Testing strategy

Implementation follows red-green-refactor commits.

1. Model API tests prove full-score reuse, subset parity, padding, validation, and
   gradient flow.
2. Handoff tests forbid h-only scoring, verify initial-belief calls, deterministic
   ties, scorer-aware cache invalidation, and no gold injection.
3. Pure objective tests cover weak gain, strong safety, multi-positive aliases,
   masks, finite empty groups, and detached static gradients.
4. Stage2 tests retain static misses, validate explicit legal pools, remove old
   memory-margin surfaces, verify resume-aware warmup, and prove post-action
   gradients reach causal memory modules.
5. Stage4 tests verify static-miss retention and raw-objective parity with Stage2.
6. Existing causal-mainline, readiness, launcher, and evaluator tests run after
   each integration task.

No full training job is launched until CPU tests, Python compilation, shell syntax,
and the candidate-selection latency/memory benchmark pass.

## Phase boundary

Phase 1 ends when Tasks 1-5 pass their source verification and produce a checkpoint-
compatible training surface. It does not claim empirical method improvement until
new Stage2/Stage4 training and held-out comparisons are run.

The next phase will implement static-preserving dynamic candidate recall. The
learned reliability gate remains conditional on an oracle/learnability audit and
is not automatically added merely because the code can support it.
