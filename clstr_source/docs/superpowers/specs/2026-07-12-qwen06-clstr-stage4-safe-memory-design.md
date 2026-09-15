# Qwen0.6B CLSTR Stage4 Safe-Memory Design

## Status

Draft for user review. The architecture direction was approved in conversation
on 2026-07-12, but production implementation remains blocked until the user
reviews this committed document.

## Goal

Repair Stage4 so that it can improve causal trajectory memory without changing
the Stage2 static routing endpoint. The final route must support an exact
score-level fallback to Stage2 when memory is absent or judged unreliable.

The repaired method is:

```text
h_t       = Encode(state_t)
m_t       = ReplayFrom(Init_stage2(h_0), trajectory_prefix)

m_hat     = Transition(m_t, action_t, next_observation)
m_next    = Correction(m_hat, Observe(h_next), next_observation)

static    = Route_stage2(h_next, Init_stage2(h_next))
dynamic   = Route_stage2(h_next, m_next)

final     = Fuse(static, dynamic, alpha)
```

`Init_stage2` and `Route_stage2` are immutable. Stage4 changes only the memory
supplied to the frozen Stage2 router and, conditionally, the scalar reliability
decision used to fuse its two endpoints.

## Why the Current Stage4 Is Not Safe

The current implementation computes `static_full` under `torch.no_grad()`, but
the static branch still calls the Stage4 model's trainable
`initial_belief_head` and `unified_retriever`. A no-gradient forward pass is not
an immutable teacher: optimizer updates from the dynamic branch change both
modules and therefore move the supposed static comparator.

The fixed balanced four-domain probe confirms this is a real failure mode rather
than a loss-plot interpretation issue:

| Checkpoint | Dynamic MRR | Static MRR | Dynamic - static |
|---|---:|---:|---:|
| Stage2 | 0.644277 | 0.555301 | +0.088977 |
| Stage4 step800 | 0.665475 | 0.568834 | +0.096641 |
| Stage4 step1200 | 0.708205 | 0.527829 | +0.180377 |

Dynamic routing improves, but by step1200 the nominal static endpoint is
0.027472 below the original Stage2 static endpoint. Part of the increasing gap
can therefore come from static degradation.

The current training protocol has a second, independent weakness: it uses one
seeded shuffle, cyclic training batches, constant AdamW learning rate `1e-4`, no
fixed validation split, and no best-checkpoint selection. The launcher forces
step3000 as the final checkpoint, while the current quality gate mainly checks a
loose training-loss window and tail Recall@5.

The native Tau2 probe gives an additional boundary condition. Stage4 improves
the shared router, but memory-specific improvement is not yet stable under
transfer. Tau2 is therefore a transfer diagnostic, not a checkpoint-selection
set.

## Scope

This design includes:

- an immutable Stage2 routing endpoint inside Stage4;
- a delta-only Stage4 checkpoint containing only causal-memory updates;
- the existing full-pool causal next-skill objective anchored to Stage2;
- trajectory-disjoint fixed validation and benchmark-balanced training;
- warmup plus cosine learning-rate scheduling;
- validation-selected best checkpoints and revised quality gates;
- exact static/dynamic score fusion through the existing memory-utility API;
- conditional learned reliability-gate calibration;
- final-chain and lineage changes required to select the validated checkpoint.

This design does not include:

- Qwen backbone unfreezing, top-layer unfreezing, LoRA, or full fine-tuning;
- a second Qwen teacher model;
- a new Stage3, HRPO, or online RL stage;
- a categorical belief replacement;
- a new dynamic residual adapter in the first implementation;
- selecting checkpoints with Tau2, ToolBench test data, or any final benchmark
  test split;
- deleting or overwriting the currently running Stage4 control.

## Alternatives Considered

### A. Duplicate the complete Stage2 model as a teacher

This gives a clear frozen reference but duplicates the Qwen encoder, skill table,
and routing heads on GPU. It adds unnecessary memory and checkpoint complexity
even though both branches use the same state embedding and skill table.

### B. Freeze the Stage2 router in place and train only the memory updater

This is the selected design. Stage4 loads the existing Stage0-to-Stage2 chain,
freezes every parameter used by the static router, and uses that same immutable
router for both static and dynamic memory. It adds no second backbone and makes
static equality mechanically testable.

### C. Keep the current trainable router and add distillation or EMA anchoring

Distillation and EMA can reduce drift but do not make `alpha=0` exactly equal to
Stage2. They introduce a tolerance-dependent approximation where the method
needs a strict fallback guarantee. This approach is rejected.

## Component Boundaries

Two different gates must remain clearly separated:

1. `model.gate` is the recurrent belief-correction gate. It changes `m_t` and is
   trainable in Stage4.
2. The memory-utility gate produces the scalar `alpha`. It changes route scores
   only, never changes or resets `m_t`, and is promoted only after the existing
   learnability audit passes.

The repaired implementation reuses these existing interfaces:

- `model.initial_belief(h)`;
- `model.unified_route_full_logits(h, m)`;
- `model.gather_unified_route_logits(...)`;
- the shared post-action memory helper;
- replay-prefix initialization through `model.initial_belief()`;
- `clstr.memory_utility_gate.memory_utility_features()`;
- `effective_memory_alpha()` and `fuse_route_scores()`;
- the static-preserving candidate union used by current-state evaluation.

`route_logits_from_candidates()` is explicitly outside this Stage4 path because
its skill ranking is h-only and uses memory only for STOP.

## Immutable Stage2 Router

Stage4 first reconstructs the exact Stage0-to-Stage2 model currently used for
evaluation. It then records a digest over every tensor that can affect static
routing, including:

```text
encoder.*
skill_table.*
initial_belief_head.*
unified_retriever.*
```

The exact implementation may use a named allowlist rather than raw prefixes,
but it must cover the complete effective static scorer. These parameters are
frozen before the optimizer is created and remain frozen for the entire run.

The Stage4 trainable module allowlist is:

```text
transition.*
gate.*
action_proj.*
```

The following are not trainable in the first safe-memory implementation:

```text
encoder.*
skill_table.*
initial_belief_head.*
unified_retriever.*
trans_head.*
skill_head.*
stop_head.*
gated_temporal_reranker.*
```

Replay remains trainable through `transition`, `gate`, and `action_proj`, but its
initial memory comes from the frozen Stage2 initializer. Observation-derived
memory continues to use the frozen skill geometry.

Focused toy-model tests compare every immutable router tensor after an optimizer
step. A real run computes the complete router content digest at step0 and final
selection. At each 400-step validation event it performs a cheaper continuous
audit: optimizer-name equality, exact digest equality for the small Stage2
routing heads and calibration tensors, and exact static-logit equality on the
fixed validation cache. This avoids repeatedly transferring the complete Qwen
backbone solely for hashing while still detecting any change that affects the
declared static endpoint. Any mismatch blocks training immediately.

## Causal Training Data Flow

For each eligible row:

```text
h_t       = FrozenEncode(state_text)
m_start   = FrozenInit(h_first)
m_t       = TrainableReplay(m_start, replay_prefix)

h_next    = FrozenEncode(next_state_text)
m_next    = PostActionMemory(
                m_t,
                current_skill,
                action_text,
                next_observation_text,
                h_next,
            )

m_static  = FrozenInit(h_next)
s_static  = FrozenRoute(h_next, m_static)
s_dynamic = FrozenRoute(h_next, m_next)
```

Both branches use the same cached `h_next`, skill table, legal-pool mask,
positive mask, dtype, and full-pool scorer. Their only semantic difference is
the memory input.

The Qwen backbone and its projection remain frozen. Frozen state embeddings may
use the existing schedule-aware cache, but cache identity must retain the exact
prompt, ordered batch schedule, model identity, and dtype. Validation embeddings
are built once under one fixed schedule and reused across every checkpoint so
static equality is not confounded by BF16 batch-composition effects.

Rows missing a trustworthy trajectory ID, step index, next state, current
action, next observation, current skill, or next-skill target cannot enter the
safe-memory train or validation sets. Full training fails if trajectory identity
is unavailable for any otherwise eligible row because trajectory-disjoint
validation could not be guaranteed.

## Teacher-Anchored Objective

The current counterfactual utility objective is retained rather than replaced
again. Its static utility is redefined as the immutable Stage2 teacher.

For legal skills, define:

```text
U(scores) = logsumexp(scores over positive skills)
          - logsumexp(scores over all legal skills)

L_dynamic = mean(-U(s_dynamic))
```

The detached teacher gain is:

```text
gain = U(s_dynamic) - stop_gradient(U(s_static_stage2))
```

For a row where Stage2 static Top-1 is wrong:

```text
L_gain = relu(gain_margin - gain)
```

For a row where Stage2 static Top-1 is correct:

```text
L_safety = relu(-safety_tolerance - gain)
```

The Stage4 memory loss is:

```text
L_stage4 = L_dynamic
         + counterfactual_weight
           * (gain_weight * mean(L_gain)
              + safety_weight * mean(L_safety))
```

Defaults remain evidence-compatible with the current repaired objective:

```text
counterfactual_weight = 0.05
gain_margin           = 0.10
safety_tolerance      = 0.01
gain_weight           = 1.0
safety_weight         = 1.0
```

Gradients must reach `transition`, recurrent `gate`, and `action_proj`. No
gradient or optimizer state may exist for the immutable router. A new KL or
distillation term is unnecessary because exact parameter freezing is stronger
than approximate anchoring.

## Static-Preserving Candidate Recall

The existing memory-conditioned candidate-union design remains part of the
method:

```text
C_static = StableTopK(s_static, M)
C_extra  = StableTopK(s_dynamic outside C_static, D)
C_union  = C_static followed by C_extra
```

Freezing the router does not make this feature redundant. Stage4 trains
`m_next`, so dynamic full-pool scores can still recover skills omitted by the
Stage2 static Top-M. Static candidates are never removed, and equal-budget
static retrieval remains the recall control.

The final route budget must satisfy `final_k <= M`. Therefore, when `alpha=0`,
dynamic-only union entries cannot change the final Top-K and the selected skills
are exactly the Stage2 static result.

## Reliability Fusion

Final valid scores use the existing mask-safe fusion:

```text
final = static + alpha * (dynamic - static)
```

The implementation continues to use `fuse_route_scores()` rather than applying
the expression to padded minimum values. Its exact endpoint contract is:

```text
alpha = 0  -> bitwise-equal valid static scores
alpha = 1  -> bitwise-equal valid dynamic scores
```

Rows with zero causal updates receive effective `alpha=0` regardless of the raw
gate output. A low alpha changes only the current route decision; it never erases
trajectory memory.

Always-available reliability modes are:

- static;
- dynamic;
- fixed alpha selected from `{0, 0.25, 0.5, 0.75, 1}` on validation only;
- the existing identity-free heuristic;
- rank and utility oracles for diagnostics only.

### Conditional learned gate

The learned gate is not trained jointly with the memory updater. Joint training
could let alpha collapse toward zero and hide a weak dynamic branch. Instead:

1. select the best dynamic Stage4 checkpoint first;
2. emit detached route records from Stage4 training trajectories only;
3. run the existing internally trajectory-disjoint memory-utility oracle and
   learnability audit on those training-source records;
4. create a learned-gate checkpoint only when that persisted audit reports
   `learned_gate_recommended=true`;
5. fit `Linear(11, 1) + Sigmoid` on the audit's train partition, using the
   versioned 11-feature schema with no benchmark or skill identity and no
   gradients into CLSTR;
6. use the audit's internal dev partition for calibration diagnostics, then make
   the promotion decision on the fixed Stage4 validation set.

The gate calibration loss is the multi-positive listwise NLL of detached fused
scores. The gate checkpoint stores its feature schema, train-only normalization,
audit digest, validation metrics, zero-history policy, and state dict.

The learned gate is promoted over the best fixed alpha only if it:

- improves balanced validation macro MRR by at least `0.005`;
- does not reduce any benchmark MRR by more than `0.01` versus the best fixed
  alpha;
- preserves exact zero-history fallback;
- passes the predeclared Phase-2 audit thresholds.

If these conditions fail, no learned-gate artifact is declared active. The final
chain records the best validation-selected fixed alpha. This is a negative gate
result, not a training failure.

Frozen comparable corpora that contain no lossless causal replay prefix receive
effective `alpha=0` and are reported as static fallback, not as memory evidence.
Memory-specific claims come from native causal ToolBench, Tau2, and ToolSandbox
routes plus ALFWorld closed-loop evaluation, where real recurrent updates exist.

## Train/Validation Split

The split is created before row caps, shuffling, handoff materialization, or
optimizer construction.

For each of the four Stage4 training benchmarks:

1. group all eligible rows by trajectory ID;
2. compute SHA-256 over `(split_version, seed, benchmark, trajectory_id)`;
3. assign a trajectory to validation when the first 64 hash bits modulo 100 are
   below 10, and assign the rest to training;
4. exclude every row from a validation trajectory from training;
5. hash `(validation_row_version, seed, benchmark, row_id)` and select the 256
   lowest-hash eligible validation rows per benchmark.

The target validation set therefore contains 1024 rows, balanced across:

```text
toolbench_g3
traject_bench
alfworld
webshop
```

The split manifest records trajectory and row counts, hashes, benchmark counts,
source-data identity, prompt contract, and selected row IDs. Fewer than 128
eligible validation rows for any declared benchmark blocks the full run instead
of silently changing the denominator.

Tau2 and all final benchmark test data remain outside this split and cannot
affect checkpoint or alpha selection.

## Training Sampler and Optimization

Full Stage4 retains a 3000-step maximum budget and batch size 16 for direct
comparison with the current control. Each training batch contains four rows from
each benchmark, drawn from independent deterministic shuffled queues. Queues
cycle only within their benchmark and reshuffle with a recorded epoch seed.

The optimizer schedule is:

```text
optimizer             = AdamW
peak learning rate    = 3e-5
warmup                 = first 5% of optimizer steps
decay                  = cosine
minimum learning rate = 3e-6
weight decay           = 0.01
```

The counterfactual weight keeps its current 5% warmup, keyed to the same local
optimizer step. The scheduler state is stored in rolling and selected
checkpoints. Stage4 resume is added only if model, optimizer, scheduler, split,
sampler, and router digests all match; otherwise resume fails closed.

No automatic early stopping is required in the first implementation. The run
may consume the full 3000-step budget, while best-checkpoint selection prevents
the arbitrary final step from becoming the paper checkpoint.

## Validation and Best-Checkpoint Selection

Validation runs at:

```text
step 0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3000
```

It uses `model.eval()`, no gradients, fixed cached embeddings, the complete
legal skill pool, and the same four-domain row set every time.

Each validation report includes, globally and per benchmark:

- static, dynamic, and fused Recall@1/5;
- static, dynamic, and fused MRR;
- dynamic-minus-static MRR;
- memory-active metrics;
- static-miss recovery and equal-budget candidate recall;
- fixed-alpha grid results;
- router digest and maximum static-logit difference from the Stage2 baseline;
- nonfinite and exclusion counts.

A checkpoint is mechanically eligible only when:

- the immutable router digest exactly matches Stage2;
- maximum static-logit difference from the cached Stage2 baseline is exactly
  zero on valid entries;
- all required metrics and gradients are finite;
- its checkpoint contains only approved Stage4 delta keys;
- all four validation benchmarks are present.

Among eligible checkpoints, selection is lexicographic:

1. highest balanced all-eligible dynamic macro MRR;
2. highest memory-active dynamic-minus-static macro MRR;
3. highest balanced dynamic Recall@5;
4. earliest step.

The selected dynamic checkpoint is release-eligible only when:

- balanced dynamic macro MRR improves over the Stage2 dynamic baseline by at
  least `0.01`;
- balanced memory-active dynamic-minus-static macro MRR is at least `+0.01`;
- at least three of four benchmarks have nonnegative dynamic-minus-static MRR;
- no benchmark dynamic MRR is more than `0.03` below its Stage2 dynamic
  baseline;
- the immutable static endpoint remains exact.

Here, the Stage2 static and Stage2 dynamic baselines are step0 evaluations on
the identical cached validation rows. The Stage2 dynamic baseline uses the same
replay and post-action contract as Stage4, with the unmodified Stage2 recurrent
modules.

After fixed-alpha and optional learned-gate calibration, the final fused route is
release-eligible only when:

- balanced fused macro MRR is at least `0.005` above
  `max(Stage2 static macro MRR, Stage2 dynamic macro MRR)`;
- no benchmark fused MRR is more than `0.01` below its Stage2 static endpoint;
- zero-history rows remain exact static fallback.

If no checkpoint passes, Stage4 is marked `action_required`; the pipeline keeps
Stage2 as the safe endpoint and does not claim a Stage4 memory improvement.

## Checkpoint and Lineage Contract

The safe Stage4 checkpoint is delta-only. Its model state may contain only:

```text
transition.*
gate.*
action_proj.*
```

It must not contain:

```text
encoder.*
skill_table.*
initial_belief_head.*
unified_retriever.*
```

The loader overlays Stage2 first and the Stage4 delta second, then verifies that
the effective router digest is unchanged. A forbidden key, missing required
delta, shape mismatch, or digest change is a hard error.

Each validation checkpoint stores optimizer and scheduler state for resume, but
the final evaluation checkpoint remains a compact delta overlay. Checkpoint
metadata records:

- Stage0 and Stage2 parent identities;
- immutable router digest;
- trainable/frozen parameter names and counts;
- split and sampler manifest hashes;
- objective and scheduler configuration;
- selected validation report hash;
- frozen Qwen evidence.

Stage4 writes a self-hashed `stage4_selection.json` containing the selected
checkpoint path and digest, selected validation report, release decision, and
active reliability artifact or fixed alpha.

The Qwen final-chain resolver must stop hard-coding `step3000`. It resolves the
selected Stage4 checkpoint through `stage4_selection.json`, verifies the Stage4
quality gate and lineage against that exact checkpoint, and includes the active
reliability mode in the final-chain identity. A promoted learned gate is an
optional separately hashed artifact; its absence is valid only when a fixed
alpha is explicitly recorded.

The current final-chain manifest generated for the moving-static control cannot
be reused for the safe-memory run. It must be regenerated after the new Stage4
selection is complete.

## Quality Gate

The revised Stage4 quality gate checks four independent categories:

1. Causal mechanics: post-action updates, replay evidence, next-state rows, and
   gradients into every required memory module.
2. Static safety: exact Stage2 router digest, exact cached static logits, no
   forbidden checkpoint keys, and alpha-zero parity.
3. Validation quality: selected-checkpoint absolute dynamic metrics, memory
   delta, per-benchmark floors, fixed-alpha or gate results, and no test-set
   selection.
4. Lineage and execution: frozen Qwen, parent identities, split/sampler hashes,
   scheduler state, selected checkpoint existence, and self-hashed reports.

The old loose loss-window and tail Recall@5 checks may remain as health
diagnostics, but they are no longer sufficient for release.

## Current Stage4 Control

Slurm job `110640` and its existing output directory remain unchanged. They are
the moving-static control for diagnosis and ablation. The control may finish and
its dependent tests may run, but its step3000 checkpoint is not automatically
accepted as the final CLSTR paper checkpoint.

No current checkpoint, log, probe, or quality report is deleted. The safe-memory
implementation uses a separate output directory and lineage.

## Testing Strategy

Implementation follows red-green-refactor cycles.

Required focused tests include:

1. Stage4 optimizer parameters are exactly `transition`, recurrent `gate`, and
   `action_proj`.
2. One optimizer step changes a dynamic route while leaving every immutable
   router tensor and digest unchanged.
3. Static logits before and after Stage4 updates are exactly equal to the cached
   Stage2 baseline.
4. Changing only the action or next observation changes dynamic logits and sends
   gradients to all required memory modules.
5. No gradient reaches `initial_belief_head`, `unified_retriever`, encoder, or
   skill table.
6. Replay starts through the frozen Stage2 initializer and remains trainable
   through recurrent update modules.
7. Alpha zero and one return exact endpoints; zero-history rows force alpha zero.
8. Delta-only checkpoint writing and loading reject every forbidden router key.
9. The trajectory split is deterministic, disjoint, four-domain balanced, and
   invariant to input row order.
10. The sampler emits four rows per benchmark for batch size 16.
11. Validation selects an earlier better checkpoint over a worse final step and
    uses deterministic tie-breaking.
12. Learned-gate creation is impossible after a failed audit and promotion is
    impossible without validation improvement.
13. Final-chain resolution uses `stage4_selection.json` and rejects a hard-coded,
    stale, or mismatched Stage4 checkpoint.
14. Frozen and native evaluators reconstruct Stage0, Stage2, selected Stage4
    delta, and optional reliability artifact in the declared order.

Lightweight focused tests and syntax checks may run locally. Repository-wide or
otherwise heavy pytest suites run only through Slurm, consistent with the
storage-node resource constraint.

## Failure Handling

- Any immutable-router drift stops Stage4 immediately.
- Nonfinite loss, logits, gradients, or validation metrics block the checkpoint.
- Split overlap or missing trajectory identity blocks full training.
- A Stage4 checkpoint containing router keys is rejected even when tensor values
  happen to equal Stage2.
- A failed learned-gate audit produces no learned-gate checkpoint.
- A best fixed alpha of zero is reported explicitly; it means validation did not
  support a memory-dependent final route.
- Failure to meet release thresholds keeps Stage2 as the safe endpoint and stops
  automatic full-benchmark submission for the Stage4 claim.
- No failure path authorizes Qwen backbone unfreezing.

## Acceptance Criteria

The design is implemented successfully when:

- the Stage2 static route is immutable and exactly recoverable at `alpha=0`;
- Stage4 checkpoints contain only the causal-memory delta;
- the causal next-skill objective directly trains post-action memory while using
  the immutable Stage2 teacher;
- memory can still improve full-pool ranking and static-preserving candidate
  recall;
- checkpoint selection uses fixed trajectory-disjoint validation rather than the
  final training step;
- the final reliability mode is selected and recorded without test data;
- Qwen remains fully frozen;
- the existing control remains preserved;
- all focused and Slurm verification gates pass before the four full benchmarks
  are submitted.
