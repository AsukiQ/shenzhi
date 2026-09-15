# CMC Direct-Utility Gate Recalibration Design

## Status and scope

Approved conceptually on 2026-07-13. This is a Stage4-only calibration
experiment on the current frozen Qwen-0.6B CLSTR chain. It does not retrain
Stage0, Stage1, Stage2, the Qwen backbone, the current Stage4 residual adapter,
or any routing/memory module.

The historical `full3000` result uses a different base chain and is not a
target, regression baseline, or checkpoint-selection input. All decisions use
the current chain's static, raw-dynamic, and fused validation endpoints.

## Problem

The current candidate-aware CMC gate already receives eleven detached,
benchmark-agnostic utility features and emits one interpolation coefficient per
route row. Its deployed equation is retained:

```text
s_fused = (1 - alpha) * s_static + alpha * s_dynamic
```

The current objective trains alpha only indirectly through dynamic ranking,
fused ranking, and a static no-regret term. Because the gate starts near
`alpha=0.01` and static routing is strong, a nearly-static solution is an easy
low-risk optimum. On the matched full evaluations, alpha does not track actual
memory utility: ToolBench benefits from raw dynamic memory but receives a lower
alpha than harmful Tau2 dynamic memory.

## Selected design

Keep the existing `MemoryUtilityGate`, its eleven-feature schema, exact endpoint
fusion, and zero-history hard fallback. Refit only the gate under explicit
row-level counterfactual utility supervision.

For each trustworthy causal calibration row, compute frozen utilities:

```text
u_static  = log P(positive skill | s_static, legal candidates)
u_dynamic = log P(positive skill | s_dynamic, legal candidates)
delta_u   = stop_gradient(u_dynamic - u_static)
```

Convert the signed utility difference into a soft target and confidence:

```text
target_alpha = sigmoid(delta_u / temperature)
row_weight   = clamp(abs(delta_u) / temperature, 0, 1)
```

Rows with negligible static/dynamic difference therefore have negligible gate
supervision. Positive-utility rows supervise alpha upward; negative-utility
rows supervise it downward. The calibration objective is:

```text
L_calibration = L_fused_rank
              + L_static_no_regret
              + lambda_gate * weighted_BCE(alpha, target_alpha)
```

The dynamic endpoint is frozen and is not optimized in this calibration pass.
The current Stage4 residual adapter is also frozen, so the experiment isolates
whether utility supervision fixes selection without changing either endpoint.

## Feature normalization and trainable scope

The gate is reset and refit with feature mean and scale computed only from the
calibration training split. The trainable state is exactly:

```text
route_memory_candidate_utility_gate.net.0.weight
route_memory_candidate_utility_gate.net.0.bias
```

The normalization buffers are derived artifacts, not optimizer parameters.
No benchmark name, source ID, skill ID, task ID, or evaluation identity is a
feature. The existing features remain:

```text
static/dynamic normalized entropy
static/dynamic standardized top gap
top-1 agreement
Jensen-Shannon divergence
mean standardized logit difference
memory cosine and relative-L2 distance
normalized causal update count
normalized valid candidate count
```

Calibration batches and validation metrics are macro-balanced by training
source so a large source cannot dominate the gate.

## Pre-training oracle and separability audit

Gate refitting is allowed only if a trajectory-grouped calibration audit shows:

1. both positive-utility and negative-utility rows exist;
2. the per-row static/dynamic oracle improves source-balanced validation MRR by
   at least `0.005` over static;
3. a feature-only cross-validated linear gate captures at least half of that
   oracle gain;
4. its worst source-level validation regret versus static is no worse than
   `0.005`.

If these conditions fail, the experiment stops. No benchmark-specific rule or
threshold is added.

## Data and selection

Calibration uses only training/validation trajectories already permitted for
Stage4. Final ToolBench, ToolSandbox, Tau2-base, and ALFWorld rows are excluded
from fitting, normalization, early stopping, and checkpoint selection.

Checkpoint selection uses trajectory-disjoint, source-balanced validation and
requires:

- exact alpha-zero/static and alpha-one/dynamic endpoints;
- zero-history exact static fallback;
- positive source-balanced fused MRR delta versus static;
- worst source-level fused regret versus static no worse than `0.005`;
- a non-degenerate alpha distribution containing both low- and high-utility
  rows.

The current CMC checkpoint remains the immutable control.

## Evaluation and interpretation

The recalibrated gate is evaluated once under the already pinned SR/ToolRex
protocols. Tau2 uses only official base. The primary question is whether one
benchmark-agnostic gate preserves useful ToolBench dynamic memory while keeping
the ToolSandbox and Tau2 static fallbacks.

The experiment does not aim to reproduce the different-base historical
`full3000` score. A successful result demonstrates selective causal memory; a
failed separability audit or validation gate is reported as evidence that the
current eleven features are insufficient.

## ALFWorld control reuse

The aligned Qwen3-14B-only executor control already contains complete canonical
`valid_seen` and `valid_unseen` runs. A prefix audit found the repeated current
runs identical on the first thirty episodes of each split. The final CLSTR
evaluation therefore reuses those immutable control artifacts and runs only
Qwen+CLSTR.

The imported control must record its source paths, metrics and run SHA-256
digests, episode denominators, split identities, and the prefix-equivalence
audit. The aggregate gate must validate the imported control and the newly run
CLSTR artifacts separately; copying files without provenance is forbidden.

## Explicit non-goals

- No benchmark-specific alpha or branching.
- No full-test threshold search.
- No attempt to recover a different-base historical score.
- No Stage2, residual-adapter, memory, router, or backbone update.
- No Tau2-full diagnostic; only official Tau2-base is reported.
