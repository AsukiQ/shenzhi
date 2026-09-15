# Qwen Safe Memory Fusion Design

## Scope

This change retrains the Qwen-0.6B CLSTR Stage2 and Stage4 memory path while
reusing the selected Stage0 and Stage1 artifacts. Its purpose is to preserve the
static route on small candidate pools and retain memory-driven full-pool recall.
The BGE implementation is explicitly out of scope until Qwen benchmark evidence
passes the promotion gate.

## Why the current objective is insufficient

The current Stage2 and Stage4 counterfactual objective compares dynamic and
static multi-positive log probability over the full legal skill pool. That is a
useful full-pool signal, but it is not the metric used by a benchmark such as
ToolSandbox, whose average candidate count is approximately 3.6. A model can
increase full-pool gold probability while moving one local negative above the
gold skill. The existing objective also does not directly teach a route-utility
gate; the two post-hoc gate experiments did not pass their promotion gates.

## Alternatives considered

### Increase the current counterfactual loss weight

This is the smallest change but preserves the mismatch between full-pool log
utility and local rank. It is rejected.

### Train another post-hoc reliability classifier

This leaves the Stage2 memory route unchanged and asks Stage4 or evaluation to
repair it. Two empirical gate-only attempts failed. It is retained only as an
ablation, not the main design.

### Joint raw-recall and safe-local fusion

This is the selected design. It separates the two roles that memory must serve:
raw dynamic logits can discover candidates outside the static shortlist, while
a bounded gated residual controls final local ordering.

## Architecture

The selected Stage0 static routing foundation is frozen during Stage2 and
Stage4: encoder projection, skill table, initial-belief head, and unified
retriever do not update. Static logits therefore remain the same function of the
current state throughout memory training.

For each causal next-state row:

1. Compute static memory from the next state with `initial_belief`.
2. Compute dynamic memory from the replayed prefix and the current
   action/observation correction.
3. Score both memories through the same frozen unified retriever.
4. Use raw dynamic logits for full-pool training and dynamic-extra candidate
   recall.
5. Center `raw_dynamic - static` over valid skills, bound it with `tanh`, and
   multiply it by a learned scalar alpha in `[0, 1]`.
6. Add that residual to static logits for final candidate-set ranking.

The route-utility gate consumes the next-state embedding, static memory,
dynamic-minus-static memory, and their elementwise interaction. It is initialized
near static and uses no benchmark name, candidate label, or target-derived
feature.

## Candidate-set supervision

Rows with an explicit legal inventory use that inventory directly. Other rows
receive deterministic candidate masks of sizes 2, 3, 4, 5, 8, and 10. Every
mask contains all known positives that fit, the strongest frozen-static hard
negatives, and deterministic raw-dynamic disagreement negatives. The masks add
no encoder or skill-table forward pass because both full-logit tensors already
exist.

The local objective has three terms:

- fused multi-positive NLL over the candidate mask;
- static-correct safety, preserving the best-positive versus best-negative
  margin up to a declared tolerance;
- static-wrong gain, requiring the fused margin to improve over static by a
  declared amount.

The raw dynamic full-pool NLL remains active so the safe local scorer does not
collapse candidate recall.

## Stage responsibilities

Stage2 trains transition, belief correction gate, action projection where
required by the existing causal path, and the new route-utility gate. It does
not update the static routing foundation.

Stage4 continues the same causal objective on its trajectory-disjoint training
split. It calibrates memory dynamics and route utility but cannot unfreeze or
replace the static foundation.

## Inference

Global-pool evaluation builds the existing static Top-M plus raw-dynamic extra-D
union. It ranks that shared union with the safe fused logits. Local benchmark
candidate sets skip candidate injection and only use safe fused ranking. Rows
with zero causal updates force alpha to zero exactly.

Reports persist static, raw dynamic, and safe fused endpoint metrics, alpha
statistics, local safety violations, and candidate-recall deltas. The old
gate-only checkpoints are not accepted as the new trainable gate.

## Compatibility and failure handling

Older Stage0/1 checkpoints may omit the new gate parameters. Loading initializes
only the missing route-utility gate at its near-static default and reports the
missing-key allowlist. Stage2/4 resume checkpoints must contain the gate once the
new schema is active. Invalid candidate masks, all-positive masks, non-finite
logits, or inconsistent checkpoint schemas fail loudly.

## Verification and promotion

All pytest execution runs as a Slurm job with `CUDA_VISIBLE_DEVICES=""`; the
login/storage node is used only for compilation, shell syntax, small JSON reads,
and git inspection.

Qwen is promoted only when:

- focused and affected source tests pass;
- Stage2 and Stage4 representative smoke jobs are finite and their reports show
  the intended trainable/frozen modules;
- ToolSandbox safe-fused MRR is within 0.005 of static or better;
- the ToolBench/Tau2/ToolSandbox macro MRR does not regress against the current
  selected Qwen route;
- raw dynamic candidate additions improve strict recall on at least one
  supported global-pool evaluation.

BGE Stage2/4 remains cancelled unless all applicable Qwen conditions pass.

