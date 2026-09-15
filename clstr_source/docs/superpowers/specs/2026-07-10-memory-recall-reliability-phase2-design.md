# Memory Recall and Reliability Phase 2 Design

## Goal

Phase 2 completes the two remaining approved CLSTR method changes after the
full-pool causal Stage2/Stage4 objective:

1. let recurrent memory add skills that the corrected static retriever omitted;
2. make final routing fall back toward the static branch when dynamic memory is
   not useful.

The two changes are implemented as separate, ordered subprojects. Candidate
recall is completed first because every reliability comparison must use the same
candidate set.

## Scope

Phase 2 includes:

- a deterministic static-preserving candidate union;
- strict candidate-recall and rescue metrics;
- shared evaluator integration for global, local, and environment-candidate
  protocols;
- mask-safe static/dynamic score fusion;
- fixed-alpha, heuristic, rank-oracle, and utility-oracle reliability baselines;
- a trajectory-disjoint learnability audit;
- a learned scalar gate only when the persisted audit satisfies the predeclared
  evidence threshold.

Phase 2 does not unfreeze the text backbone, add Stage3/HRPO, replace recurrent
memory with a categorical belief, use benchmark/skill identities as gate
features, adapt the dynamic budget per row, reset memory during score backoff, or
claim candidate recall on environment-provided or saturated local candidate sets.

## Alternatives considered

### Candidate retrieval

1. Replacing the static shortlist with dynamic Top-K makes memory influential
   but can remove already-correct static candidates and confounds candidate
   quality with budget.
2. Scoring only the static shortlist preserves behavior but cannot support a
   memory-recall claim.
3. Preserving static Top-M and appending dynamic-only Top-D is selected. It
   cannot reduce coarse recall by candidate removal and has a clean equal-budget
   static control.

### Reliability control

1. Resetting to initial memory changes recurrent state and can erase useful
   history. It remains an oracle diagnostic only.
2. Fixed or heuristic score interpolation is cheap and auditable, so it is
   always implemented as a baseline.
3. A learned scalar score gate is selected conditionally. It is enabled only if
   held-out data show meaningful and predictable static/dynamic complementarity.

## Subproject A: static-preserving memory candidate recall

For one evaluated state, compute full-pool scores over one declared legal pool:

```text
static_scores  = UnifiedRouteFull(h, InitialBelief(h))
dynamic_scores = UnifiedRouteFull(h, m_dynamic)
```

For post-action evaluation, `h` is `h_next` and `m_dynamic` is the corrected
post-action memory. For current-state evaluation, `h` is the current state and
`m_dynamic` is the canonically replayed trajectory memory. Both branches use the
same state, skill table, legal mask, dtype, and device.

Candidate selection is:

```text
C_static = StableTopK(static_scores, M)
C_extra  = StableTopK(dynamic_scores outside C_static, D)
C_union  = C_static followed by C_extra

C_equal_budget = StableTopK(static_scores, M + D)
```

The union preserves both selected orders, contains no duplicates, and uses
declared skill-table index as the tie break. If dynamic and static scores are
identical, `C_union` must exactly equal `C_equal_budget`, including tied scores
and sparse legal masks.

The first implementation uses fixed `D`; adaptive budgets remain out of scope.
Training does not cache candidate IDs because sequential heads change each
optimizer step. Evaluation caching is allowed only in `model.eval()` mode with a
key covering the effective model state, ordered checkpoint overlays, row/prefix
digest, legal-pool embedding/order, M, D, mask mode, causal contract, and selector
version.

## Candidate-recall metrics and denominators

Every applicable report records:

```text
static_recall@M
dynamic_top_recall@D
dynamic_extra_recall@D
union_recall@(M+D)
static_equal_budget_recall@(M+D)
static_miss_rows
dynamic_rescue_rows
static_miss_recovery_rate
static_dynamic_overlap
dynamic_only_candidate_count
```

Exact and alias-aware variants use the full source denominator. A target absent
from the declared legal pool contributes a strict zero and a named protocol
blocker; it is never silently filtered out.

Two denominator families are always reported:

- `all_eligible_source_strict_*`: paper-primary, including zero-history rows;
- `memory_active_strict_*`: rows with at least one causal update, used only for
  mechanism analysis.

Candidate-recall claims apply only to known-pool sequential global retrieval.
Benchmark-local pools with `M + D >= legal_pool_size` report saturation and make
no recall claim. ALFWorld and WebShop report non-applicability because the
environment supplies admissible actions. Appended unseen skills are a separate
transfer slice and cannot count as trained dynamic-recall evidence.

## Subproject B: reliability-aware score backoff

Reliability changes only route scores:

```text
fused_scores = static_scores + alpha * (dynamic_scores - static_scores)
```

The implementation evaluates valid entries only, so padded infinities never
enter subtraction. `alpha=0` returns the exact masked static tensor and
`alpha=1` returns the exact masked dynamic tensor. A row with
`causal_update_count=0` always receives effective `alpha=0`.

The route gate never changes `m_dynamic`; a low-alpha decision cannot erase
trajectory history needed later.

Always-available modes are static, dynamic, one global fixed alpha, an
entropy/margin heuristic, and rank/utility oracles for diagnostics.

The audit-only feature vector contains no benchmark or skill identity:

```text
static/dynamic normalized entropy
static/dynamic standardized top1-top2 gap
top1 agreement
Jensen-Shannon divergence
mean absolute standardized-logit difference
1 - cosine(static_memory, dynamic_memory)
relative memory L2 difference
log-normalized causal update count
log-normalized valid candidate count
```

All features and branch scores are detached for gate calibration.

## Learned-gate decision boundary

The audit freezes qualifying benchmarks before inspecting utilities. A benchmark
qualifies with at least 50 memory-active gate-eligible rows from at least 10
trajectories, and at least four qualifying benchmarks are required.

`learned_gate_recommended=true` only when all conditions hold:

```text
static-better and dynamic-better each >= 5% of non-tie rows
each winner type >= 20 held-out rows total
each winner type >= 5 rows in at least two sequential benchmarks
rank-oracle headroom over best endpoint >= 0.01 macro MRR
rank-oracle headroom over best fixed alpha >= 0.005 macro MRR
held-out linear AUROC >= 0.60
    or utility regret improves >= 10% over a constant predictor
```

Splits, normalization, count caps, and fixed-alpha selection use
trajectory-disjoint train/dev data only. Bootstrap intervals resample whole
trajectories/tasks.

If the audit fails, no learned-gate checkpoint or active learned-gate path is
produced. The method keeps fixed-alpha/heuristic baselines and records the
negative learned-gate result.

## Component boundaries

- `clstr/memory_candidate_recall.py`: candidate union, legal masks, and strict
  recall/rescue accounting.
- `clstr/current_state_route_eval.py`: shared branch scoring and union evaluation.
- `clstr/logged_online_stage4_train.py`: batch aggregation and metadata flow.
- `clstr/global_pool_route_eval.py`: strict global-pool and equal-budget controls.
- benchmark wrappers: declare protocol applicability and budgets without
  reimplementing selection.
- `clstr/memory_utility_gate.py`: fusion, feature extraction, and oracle/heuristic
  utilities.
- `scripts/audit_clstr_memory_utility_oracle.py`: trajectory-disjoint audit and
  persisted recommendation.
- `clstr/memory_utility_gate_train.py`: created only after a passing audit.

## Failure behavior

Fail loudly for mismatched branch shapes/dtypes/devices, nonfinite valid logits,
invalid alpha or budgets, target-derived global masks, incomplete skill mappings,
unknown learned-gate checkpoints, and endpoint claims with `final_k > static_k`.

Rows with no legal positive or negative remain visible in named exclusion
counters. Empty groups return explicit finite zeros; padded minimum values are
never multiplied to synthesize zero.

## Testing strategy

1. Pure union tests prove static preservation, tie determinism, sparse masks,
   exact equal-score equivalence, and strict rescue accounting.
2. Shared evaluator tests prove full-pool scoring, source-denominator retention,
   current-state semantics, and zero-history behavior.
3. Protocol tests prove global/local/environment applicability and reject
   target-derived masking.
4. Fusion tests prove exact endpoints, mask safety, feature detachment, and no
   memory mutation.
5. Oracle tests prove trajectory-disjoint splits, frozen benchmark qualification,
   clustered bootstrap, and recommendation thresholds.
6. Learned-gate tests and implementation are conditional on the persisted audit.

## Acceptance boundary

Source implementation is complete when candidate union and reliability audit
tests pass, retained evaluators use the shared contracts, active launchers expose
declared budgets/modes, Python and shell checks pass, and readiness rejects
unsupported claims.

Empirical inclusion in the paper remains conditional on predeclared ablation
thresholds. Code completion alone is not evidence that candidate recall or a
learned gate improves benchmark performance.
