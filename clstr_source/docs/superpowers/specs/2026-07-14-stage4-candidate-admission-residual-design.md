# Stage4 Candidate Admission and Constrained Residual Design

## Goal

Replace the failed hand-written candidate-provenance boost with one trainable
Stage4 module that learns both which memory-retrieved candidates are useful and
how memory should change their ranking. Stage0, Stage1, Stage2, the frozen Qwen
backbone, candidate budgets, and the CMC parent checkpoint remain unchanged.

## Evidence motivating the change

On the aligned full ToolBench split:

- frozen-static MRR: `0.299425`;
- existing CMC MRR: `0.307634`;
- raw dynamic MRR: `0.350136`;
- hand-written candidate-provenance MRR: `0.294632`;
- dynamic retrieval rescued 85 positive rows.

Memory therefore contains useful ranking and recall signal. The failed rule
boosted all dynamic-only candidates with a positive centered residual, so many
negative extras displaced useful static candidates. Exact-static scoring for
all shared candidates also discarded the strongest part of the dynamic ranking
signal.

## Architecture

Stage4 receives the unchanged candidate union:

```text
static top-M union dynamic-only top-D
```

For each candidate it builds label-free features from the current state,
recurrent memory, candidate embedding, static/dynamic logits and ranks,
centered residual, provenance, causal update count, and uncertainty statistics.
Benchmark, task, source, skill identity, and gold identity are forbidden
features.

Two small heads are trained:

1. `admission_head` predicts whether a dynamic-only candidate deserves memory
   influence. It is supervised only on dynamic extras using gold/equivalent
   skills as positives and retrieved extras as hard negatives.
2. `candidate_residual_head` predicts a signed candidate-wise correction for
   every shared or extra candidate.

The deployed score is:

```text
bounded_delta_i = 2 * tanh(raw_delta_i / 2)

shared candidate:
    final_i = static_i + bounded_delta_i

dynamic-only candidate:
    final_i = static_i + sigmoid(admission_i) * bounded_delta_i
```

This is not a row-level static/dynamic gate. Shared candidates may be reranked,
while low-confidence extras naturally fall back to their below-top-M static
scores. There is no benchmark-specific threshold or alpha.

## Objective

Train only the two new Stage4 heads. The selected CMC route residual adapter,
all earlier stages, and the backbone stay frozen so the experiment isolates
reliability learning and can reuse the existing caches.

```text
L_stage4 = L_listwise + lambda_admit * L_admit + lambda_safe * L_cf_rank
```

- `L_listwise` is multi-positive listwise cross-entropy on the final candidate
  union.
- `L_admit` is class-balanced binary cross-entropy over dynamic extras. It
  directly distinguishes rescued positives from the much larger hard-negative
  extra set.
- `L_cf_rank` is a counterfactual no-regret margin. On rows where static has a
  valid positive and negative, it penalizes any reduction of the positive-vs-
  hardest-negative margin. On rows where memory supplies a missing positive,
  it trains a positive gain relative to the static counterfactual.

The old hand-written provenance boost and the obsolete scalar memory-margin
objective are not part of this training path.

## Data and leakage contract

- Candidate generation never injects the target skill.
- Admission labels are used only in the loss, never to construct candidates.
- Splits remain trajectory-grouped; evaluation trajectories never enter Stage4
  training or calibration.
- Gold-equivalent skills follow the existing equivalence contract.
- Training reports positive/negative counts separately for shared and dynamic-
  only candidates and fails closed if dynamic-extra positives are absent.

## Fast schedule

Use the existing frozen-Qwen embedding and handoff caches.

1. Run a 300-step smoke with frequent held-out evaluation.
2. Continue only if admission AUPRC exceeds the prevalence baseline and final
   validation MRR is not below static.
3. Run at most 1,200 full steps with early stopping; do not default to another
   3,000-step run unless the curve is still improving.

## Promotion gate

Evaluate in this order and stop immediately on failure:

1. ToolBench full: MRR must be strictly greater than existing CMC `0.3076335`,
   with positive dynamic-extra rescue count.
2. ToolSandbox full: fused MRR must be within `0.005` of static.
3. Tau2-base full: fused MRR must be within `0.005` of static.
4. Only then run ALFWorld.

No benchmark-specific tuning is allowed after observing these results. A failed
candidate-admission branch remains an ablation; the existing CMC chain remains
the release fallback.

## Verification

- Unit tests cover candidate-wise shapes, masks, invalid candidates, bounded
  residuals, zero-history fallback, and no-target candidate construction.
- Integration tests cover pre-final-k and post-final-k scoring, final-chain
  identity, runtime mode propagation, and frozen parent-router digests.
- All Torch tests, training, and evaluations run through Slurm. Local checks are
  limited to syntax, JSON, shell, and diff validation.
