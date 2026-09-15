# Minimal Stage4 Counterfactual Memory Calibration Design

## Status

Approved for implementation on 2026-07-13. The implementation must keep the
Qwen backbone frozen and preserve all prior checkpoints and reports.

This document supersedes
`2026-07-12-qwen06-clstr-stage4-safe-memory-design.md` for the next Qwen CLSTR
experiment. The previous design and all of its artifacts remain preserved as a
completed control.

## Decision Summary

The previous safe-memory implementation made Stage4 too weak to repair a
degraded routing foundation and too complicated to calibrate reliably. The next
experiment separates the responsibilities cleanly:

1. **Stage2 learns a strong routing foundation.** It may adapt
   `initial_belief_head`, `unified_retriever`, and belief calibration while the
   Qwen backbone remains frozen. A score-level anchor limits zero-history
   routing drift.
2. **Stage4 calibrates memory influence.** It freezes the complete selected
   Stage2 model and trains only one zero-initialized memory residual adapter and
   one candidate-aware utility gate.
3. **The deployed fusion has exact endpoints.** Alpha zero returns exact static
   scores; alpha one returns exact dynamic scores. There is no tanh residual
   clipping.
4. **Stage4 optimizes the deployed route directly.** It uses actual legal
   candidate sets and one counterfactual regret objective rather than six
   synthetic local candidate sizes and a weak 0.05 fused-loss term.
5. **ALFWorld uses protocol-matched reporting.** The main closed-loop result
   uses the historical shared Qwen3-14B executor protocol; the CLSTR-only greedy
   admissible-action controller is retained only as an ablation.

The Qwen embedding backbone remains fully frozen throughout.

## Motivation From the Completed Safe-Memory Run

The completed safe-memory chain provides four concrete design findings:

- Stage4 selected step 0. Every trained safe-fused checkpoint was worse than
  the Stage2-initialized endpoint.
- The safe Stage2 froze `initial_belief_head`, `unified_retriever`, and belief
  calibration. Its Tau2 static MRR fell to `0.219014`, compared with the earlier
  adapted route's stronger static behavior.
- On all 218 Tau2 batches, effective alpha exactly equaled the memory-active row
  fraction. The causal gate produced alpha one for memory-active rows and alpha
  zero only for zero-history rows. It learned memory presence, not utility.
- The deployed fused loss had weight `0.05`, while raw dynamic full-pool NLL
  dominated training. In addition, tanh-bounded residual fusion meant alpha one
  did not preserve raw dynamic ranking.

These are architectural findings, not evidence for increasing the number of
steps or unfreezing Qwen.

## What Stage4 Means

The method stages have distinct semantic roles:

- **Stage0:** learn the skill space and zero-history retrieval geometry.
- **Stage1/Stage2:** learn causal trajectory memory and a strong route scorer.
- **Stage4:** learn whether and how memory should alter the final candidate
  ranking under the deployment protocol.

Stage4 is not another full router training phase. In the paper, it should be
described as **Counterfactual Memory Calibration**, not primarily as “the fourth
training stage.” The training schedule is an implementation detail; the method
component is a frozen static teacher, a lightweight memory residual, and a
utility-aware gate.

## Alternatives Considered

### A. Delete Stage4 and merge every objective into Stage2

This is the smallest pipeline, but it removes the explicit mechanism that
prevents harmful history-conditioned ranking. Stage2 would need to learn memory
dynamics, final ranking, and reliability jointly, making failures hard to
attribute and weakening the safe-fallback claim.

### B. Put a complete second router inside Stage4

A cloned `initial_belief_head` plus a cloned `unified_retriever` gives maximum
capacity. It also duplicates responsibilities, complicates checkpoint loading,
and makes Stage4 look like benchmark-driven engineering rather than a focused
method component.

### C. Strong anchored Stage2 plus minimal Stage4 calibration

This is the selected design. Stage2 owns the full route foundation. Stage4 adds
only the minimum capacity needed to express memory-specific ranking corrections
and to decide when those corrections are useful.

## Stage2: Anchored Routing Foundation

### Canonical loss cleanup

The anchored rerun must not carry legacy objectives that have no effective
training or deployment role. Canonical Stage1/Stage2 configuration removes
`routing`, `hard_negative_margin`, `Q_success`, and
`transition_hard_negative_margin`; these objectives had zero weight in the
completed Qwen runs even though some forward computations and report fields
remained active. Experimental ablation utilities may retain their standalone
implementations, but the canonical launchers, main loss interface, optimizer
scope, quality gates, and paper objective must not expose them.

Stage2 also disables `STOP`. The head learned a nontrivial offline done-label
classifier, but ToolBench, Tau2, and ToolSandbox do not consume it, ALFWorld
uses environment termination, and the current candidate controller supplies a
candidate-invariant STOP value that cannot change action ranking. Stage1 may
retain STOP only as an explicitly optional learned-termination experiment; it
is not a required canonical component and Stage2 must freeze the head.

The old Stage2 `counterfactual_utility` loss is removed because Stage4 CMC now
owns memory-utility calibration, fused ranking, and no-regret behavior. Stage2
therefore trains the causal next-skill route plus the static teacher anchor and
any explicitly retained component-initialization auxiliaries; it does not
pre-calibrate the Stage4 utility gate.

`L_trans` and `belief` are not removed in this change. They have real gradients
through transition/correction and require a controlled ablation before either
can be declared redundant.

### Trainable scope

Stage2 continues to train its existing causal memory modules and task heads. In
unified-memory mode it also restores training for:

```text
initial_belief_head.*
unified_retriever.*
skill_table.logit_scale_belief
skill_table.skill_bias_belief
```

The following remain frozen:

```text
encoder.backbone.*
```

The existing frozen-Qwen embedding cache remains valid because none of these
changes modify the backbone or state-query prompt.

### Step-zero routing teacher

Before the Stage2 optimizer is created, the loaded Stage1/selected-Stage0 route
is frozen as the **Stage2 step-zero teacher**. Teacher route logits are detached
and may be cached under the existing prompt, candidate-pool, checkpoint, and
dtype identities. No second Qwen model is loaded.

For a zero-history row with legal candidate set `C`, define:

```text
p_teacher = softmax(s_teacher(C))
p_stage2  = softmax(s_stage2_static(C))

L_anchor = KL(p_teacher || p_stage2)
```

After the canonical cleanup, the Stage2 objective becomes:

```text
L_stage2 = L_next_skill_full_pool
         + lambda_policy * L_policy_preservation
         + lambda_trans * L_trans
         + lambda_belief * L_belief
         + 0.1 * L_anchor
```

Here `lambda_policy`, `lambda_trans`, and `lambda_belief` remain explicit
ablation-controlled weights. There is no Stage2 STOP, routing, Q-success,
hard-negative-margin, or legacy counterfactual-utility term.

The anchor is evaluated only on zero-history/static routing rows. It protects
the Stage0 geometry while still allowing supervised Stage2 ranking to improve
it. This is a score-level behavioral anchor, not a parameter-distance penalty.

### Stage2 selection requirement

Stage2 uses its existing trajectory-disjoint training-source validation. The
selected checkpoint must satisfy both:

- static macro MRR is no more than `0.005` below the step-zero teacher;
- dynamic macro MRR is higher than the step-zero dynamic route.

If no nonzero Stage2 checkpoint satisfies both, the experiment stops before
Stage4. Stage4 is not allowed to repair a failed routing foundation.

## Stage4 Architecture

### Frozen Stage2 teacher

Stage4 loads the selected Stage2 checkpoint and freezes every existing model
parameter, including:

```text
encoder.*
skill_table.*
initial_belief_head.*
unified_retriever.*
transition.*
gate.*
action_proj.*
all policy, stop, transition, and success heads
```

Stage4 therefore cannot change `m_t`, the Stage2 static endpoint, or the Stage2
base dynamic endpoint. It only calibrates how an already learned memory state
affects routing.

### Memory residual adapter

Add one module:

```text
route_memory_residual_adapter
```

For current state embedding `h_t`, initial/static memory `m_0`, and causal
memory `m_t`, define:

```text
delta_m = m_t - m_0

adapter_input = concat(h_t, delta_m, h_t * delta_m)
delta_z       = Adapter(adapter_input)
```

The adapter is:

```text
LayerNorm(3d)
Linear(3d, 64)
GELU
Linear(64, d), zero-initialized output
```

The route queries and logits are:

```text
z_static       = UnifiedRouteVector_stage2(h_t, m_0)
z_dynamic_base = UnifiedRouteVector_stage2(h_t, m_t)
z_dynamic      = z_dynamic_base + delta_z

s_static  = z_static  @ E.T
s_dynamic = z_dynamic @ E.T
```

For zero-history rows, `delta_z` is hard-masked to zero. The adapter is a small
memory-specific correction, not a duplicate router.

### Candidate-aware utility gate

The Stage4 utility gate uses the existing versioned 11-feature schema computed
from detached static/dynamic logits and memories:

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

The gate is:

```text
train-only feature normalization
Linear(11, 1)
Sigmoid
```

It receives no benchmark ID, skill ID, task ID, or final-test identity. Feature
extraction is detached, but gradients flow through alpha into the gate.

Zero-history rows receive effective alpha zero regardless of gate output.

### Exact endpoint fusion

Stage4 uses the existing mask-safe linear fusion:

```text
s_final = (1 - alpha) * s_static + alpha * s_dynamic
```

The implementation must use `fuse_route_scores()` so padded invalid entries do
not participate. The endpoint contract is exact on valid entries:

```text
alpha = 0 -> s_final is bitwise-equal to s_static
alpha = 1 -> s_final is bitwise-equal to s_dynamic
```

The previous centered tanh residual bound is removed from this Stage4 path.

## Stage4 Data Flow

For every memory-active causal training row:

```text
h_t      = FrozenEncode(state_t)
m_0      = FrozenStage2Init(h_t)
m_t      = FrozenStage2ReplayAndPostActionMemory(prefix, action, observation)

s_static  = FrozenStage2Route(h_t, m_0)
s_dynamic = FrozenStage2Route(h_t, m_t) + Adapter(h_t, m_t - m_0)
alpha     = UtilityGate(detached candidate-aware features)
s_final   = Fuse(s_static, s_dynamic, alpha)
```

Stage4 does not reconstruct memory differently from inference. The same replay,
post-action update, state-query prompt, legal-pool mask, and skill table are used
in training, validation, and evaluation.

Rows without a trustworthy trajectory, current action, next observation, next
state, or positive next skill are excluded with explicit counts. Zero-history
rows are validation controls, not optimizer examples.

## Counterfactual Calibration Objective

For legal candidate logits `s` and one or more positive skills, define:

```text
U(s) = logsumexp(s over positives) - logsumexp(s over legal candidates)
```

Stage4 uses three terms:

```text
L_dynamic = -mean(U(s_dynamic))
L_final   = -mean(U(s_final))
L_regret  = mean(relu(stop_gradient(U(s_static)) - U(s_final)))

L_stage4 = L_dynamic + L_final + L_regret
```

All three coefficients are `1.0` in the first experiment. There is no gain
margin, safety tolerance, synthetic candidate-size sweep, or 0.05 fused-loss
multiplier.

`L_dynamic` prevents the gate from hiding a weak adapter by collapsing to zero.
`L_final` trains the deployed fused route directly. `L_regret` pushes the gate
toward static whenever fused memory is worse than the detached static teacher.

## Candidate Protocol

Training uses the row's actual legal candidate set:

- benchmark-local rows use the complete declared local pool;
- global-pool rows use the complete declared global pool;
- positives are never injected into evaluation candidate unions;
- training rows whose positive is outside the declared legal pool are excluded
  and counted rather than silently repaired.

Candidate recall remains separate from safe reranking:

```text
C_static = StableTopK(s_static, M)
C_extra  = StableTopK(s_dynamic outside C_static, D)
C_union  = C_static followed by C_extra
```

Memory may therefore improve candidate recall even when the gate later chooses
a mostly static final ranking.

## Train and Validation Protocol

### Training-source-only split

No final ToolBench, ToolSandbox, Tau2, or ALFWorld evaluation split may select a
checkpoint or hyperparameter. Splits are trajectory-disjoint and deterministic.

Validation is balanced by routing regime rather than benchmark name:

```text
global-pool rows
local-small rows with 2-10 legal candidates
local-medium rows with 11-64 legal candidates
long-history rows with at least three causal updates
```

The four existing training sources—ToolBench-G3 train, TrajectBench train,
ALFWorld train, and WebShop train—supply these strata. A row may enter only one
stratum, selected by a fixed priority and hash rule recorded in the split
manifest. Each stratum contributes 256 validation rows when available.

### Optimization

Stage4 keeps the existing compute envelope:

```text
maximum steps        = 3000
batch size           = 16
optimizer            = AdamW
peak learning rate   = 3e-5
warmup               = 5%
minimum learning rate= 3e-6
schedule             = cosine
weight decay         = 0.01
```

Every batch is balanced across the four routing regimes when enough training
rows are available. Frozen-Qwen scheduled embedding caches remain enabled.

### Checkpoint selection

Validation runs at step 0 and every 400 steps, plus the final step. The selected
checkpoint maximizes balanced fused macro MRR, with ties broken by lower regret
and then earlier step.

A nonzero Stage4 checkpoint is release-eligible only when:

- fused macro MRR exceeds the best Stage2 endpoint by at least `0.005`;
- no validation regime falls more than `0.01` below Stage2 static MRR;
- memory-active fused regret is lower than at step 0;
- zero-history static logits are exact;
- adapter and utility-gate gradients are finite and nonzero;
- all frozen Stage2 parameter digests are unchanged.

If no nonzero checkpoint passes, Stage2 remains final and Stage4 is removed from
the paper's main method rather than described as a successful component.

## Checkpoint and Loader Contract

The Stage4 delta checkpoint may contain only:

```text
route_memory_residual_adapter.*
route_memory_utility_gate.*
```

It must not contain any Stage2 parameter. The loader order is:

```text
Stage0 -> Stage1 -> selected anchored Stage2 -> selected Stage4 calibration delta
```

After loading, the complete Stage2 digest must match the recorded parent. Old
safe-memory checkpoints remain loadable through their existing lineage path but
cannot be silently relabeled as the new calibration design.

The final-chain manifest records:

- selected Stage2 checkpoint and anchor report;
- selected Stage4 delta or explicit `stage4_not_promoted`;
- adapter/gate architecture and feature schema;
- validation split and selection hashes;
- exact endpoint verification;
- candidate protocol and prompt identity.

## ALFWorld Evaluation Protocol

The earlier positive ALFWorld result and the current zero result use different
controllers. The next experiment keeps them separate.

### Main closed-loop protocol

Use the historical shared executor configuration:

```text
Qwen3-14B direct admissible-action executor
official ALFWorld environment
loop guard
historical prompt and generation configuration
Qwen score weight = 1.0
CLSTR score weight = 0.25
historical Qwen/CLSTR score normalization
CLSTR prior mode = unified_memory_concrete_action
current recurrent memory protocol
```

The executor model identity, prompt template, generation arguments, loop-guard
configuration, score normalization, and the two numeric weights above are
pinned in the run manifest. A run that changes any of them is a new protocol
and cannot replace the historical-comparison row.

The coarse `unified_memory` scorer remains an interface ablation. It cannot be
substituted for the positive concrete-action protocol because the candidate
representation and override policy differ.

Run, under identical seeds and denominators:

```text
Qwen3-14B only
Qwen3-14B + Stage2 static prior
Qwen3-14B + Stage2 raw-dynamic prior
Qwen3-14B + selected Stage4 fused prior
```

The current safe-memory reliability mode and selected Stage4 delta must be
explicitly threaded into the executor-gate scorer. The old script's implicit
raw-dynamic default is not sufficient.

### Retriever-only ablation

The no-external-generator greedy admissible-action controller remains a valid
ablation. It is labeled `CLSTR-only admissible-action controller` and must not
replace or be compared numerically with the Qwen3-14B executor result.

ALFWorld success is a secondary Stage4 check because the shared executor adds
generation variance. The primary Stage4 selection remains training-source route
validation.

## Benchmark Protocol Clarifications

### Tau2

The paper main route result uses the official `base` split from each domain's
`split_tasks.json`. Tool-action decisions and refusal/no-tool decisions are
reported separately, with refusal rows evaluated through the STOP contract.

The existing 13,907-row airline/retail/telecom corpus remains a full-routing
stress test. It must not replace the main result because 13,215 rows come from
the telecom `full` split. The already pinned `0.423038` three-benchmark stress
reference remains a promotion gate; the reconstructed base split receives its
own same-protocol pre-safe reference before CMC training results are opened.

### ToolSandbox

The main result restores the exact training skill-table prefix, appends the 31
ToolSandbox skills, and masks final scoring to each row's legal candidates. This
uses the same candidate inventory as SR and ToolREx without rebuilding the
trained CLSTR checkpoint into a different model.

A paired `local_table_rebuild` ablation uses the same CLSTR checkpoint, rows,
legal candidates, corrected causal replay, dtype, and batch schedule while
constructing the model directly on the 31-skill ToolSandbox table. The ablation
measures sensitivity to model-table instantiation and cannot select a checkpoint.

## Paper Presentation

The method component is named **Counterfactual Memory Calibration (CMC)**:

```text
strong anchored causal router
+ zero-initialized memory residual adapter
+ candidate-aware utility gate
+ counterfactual no-regret calibration
```

The central claim is:

> CMC preserves the zero-history static route exactly while allowing causal
> trajectory memory to improve candidate recall and final skill ranking only
> when it is useful.

Required ablations are:

- Stage2 without CMC;
- residual adapter without utility gate;
- utility gate without counterfactual regret;
- CMC with static-only candidate recall;
- raw dynamic versus fused dynamic;
- current tanh-bounded safe-memory control versus exact-endpoint CMC.

## Testing Strategy

Implementation must follow red-green-refactor cycles. Required focused tests
include:

1. Stage2 router parameters are trainable while the Qwen backbone remains
   frozen.
2. Stage2 anchor loss is zero for identical teacher/student logits and positive
   after controlled drift.
3. Stage2 selection rejects static degradation beyond `0.005`.
4. Stage4 optimizer parameters are exactly the residual adapter and utility
   gate.
5. One Stage4 optimizer step changes dynamic/fused logits while every Stage2
   tensor remains bitwise unchanged.
6. Zero-history rows force zero adapter contribution and alpha zero.
7. Alpha zero and one produce exact static and dynamic valid logits.
8. Candidate-aware gate features contain no benchmark, task, or skill identity.
9. Counterfactual regret pushes alpha down when dynamic is worse and permits it
   to rise when dynamic is better.
10. Training uses complete legal pools and rejects positives outside them.
11. Validation strata are deterministic, disjoint, and input-order invariant.
12. Checkpoint selection chooses a nonfinal better checkpoint and rejects step
    0 as a positive Stage4 claim.
13. Delta checkpoints reject every Stage2 key and loaders verify the parent
    digest.
14. The Qwen3-14B executor gate receives the selected reliability mode and
    Stage4 delta explicitly.
15. Retriever-only and shared-executor ALFWorld reports cannot be aggregated
    into one metric row.

Lightweight unit tests and syntax checks may run locally. Heavy pytest suites,
model loads, training, and benchmark evaluations run only through Slurm.

## Failure Handling

- Stage2 static-anchor failure stops before Stage4.
- Any Stage2 tensor change during Stage4 is a hard error.
- Nonfinite adapter/gate loss, logits, gradients, or validation metrics block
  the checkpoint.
- A gate that improves average MRR but violates a regime floor is not promoted.
- A best Stage4 step of zero is an explicit negative result and removes CMC from
  the main method.
- Missing ALFWorld executor identity, Qwen checkpoint identity, prompt identity,
  or scorer mode blocks main-table aggregation.
- No failure path authorizes Qwen backbone unfreezing.

## Acceptance Criteria

The experiment is successful only when all of the following hold:

- anchored Stage2 restores a strong static/dynamic routing foundation without
  unfreezing Qwen;
- selected Stage4 step is nonzero;
- Stage4 improves balanced fused validation MRR by at least `0.005` over the
  best Stage2 endpoint;
- no routing regime regresses more than `0.01` from Stage2 static;
- exact alpha-zero static and alpha-one dynamic endpoints are verified;
- ToolBench has strictly positive dynamic-extra candidate recall,
  `Recall(C_union) - Recall(C_static) > 0`;
- ToolSandbox fused-minus-static MRR is at least `-0.005`;
- the arithmetic mean of protocol-matched ToolBench, ToolSandbox, and Tau2 MRR
  is at least `0.423038`;
- the reconstructed Tau2 base result meets or exceeds the frozen same-protocol
  pre-safe reference, and the 13,907-row result is labeled as a stress test;
- ToolSandbox checkpoint-faithful and local-table-rebuild results are both
  reported under identical row-local candidates;
- ALFWorld is rerun under the protocol-matched shared Qwen3-14B executor, with
  the retriever-only result reported separately;
- all artifacts, lineage, and focused/Slurm verification gates pass before a
  paper claim is made.
