# Candidate-Provenance Positive Residual Design

## Goal

Use recurrent memory only where it provides direct candidate-retrieval evidence:
dynamic-only candidates. Preserve exact static scoring for every shared candidate
and for every row without dynamic candidate expansion.

## Scoring rule

Let `C_static` be the static top-k set and `C_extra` be the dynamic candidates
added outside `C_static`. For the final candidate union, construct an
`extra_mask` from candidate provenance.

```text
raw_residual      = dynamic_logits - static_logits
centered_residual = raw_residual - mean(raw_residual over valid candidates)
bounded_positive = bound * tanh(relu(centered_residual) / bound)
final_logits      = static_logits + extra_mask * bounded_positive
```

The bound is the existing predeclared safe memory residual bound, `2.0`.
Invalid candidates retain the dtype floor.

## Invariants

- Shared candidates are bit-exact static.
- Dynamic-only candidates can only be boosted, never demoted.
- Rows with no dynamic-only candidates are bit-exact static.
- Rows with no causal history are bit-exact static even if candidate provenance
  is malformed or unexpectedly nonempty.
- Candidate selection before final-k truncation and final scoring after
  truncation use the identical fusion rule.
- No benchmark, source, task, skill, or positive-label identity enters the rule.
- The Stage2 router, Stage4 residual adapter, backbone, and candidate budgets are
  unchanged.

## Integration

Add one reusable fusion function in `clstr/safe_memory_ranking.py`. Build the
candidate-position mask from `CandidateUnion.dynamic_extra_rows` and
`CandidateUnion.candidate_rows`. Use the new fusion under a new reliability
mode, `cmc_candidate_provenance`, in current-state routing and both ALFWorld CMC
scorers. Existing `cmc` behavior remains immutable for comparison.

The final-chain reliability record binds the mode and residual bound. No learned
gate checkpoint is loaded.

## Validation and stop rule

Run Slurm RED/GREEN tests for exact-static invariants, positive-only bounded
boosts, provenance-mask construction, final-k consistency, current-state
scoring, ALFWorld scoring, final-chain validation, and multibench exports.

Then run aligned smoke evaluation for ToolBench, ToolSandbox, and Tau2-base.
Promote to full evaluation only if:

- ToolSandbox and Tau2 are exact or within `0.005` MRR of their static branch;
- ToolBench MRR is strictly above the current CMC fused value;
- ToolBench retains positive dynamic-extra recall;
- no protocol, identity, denominator, or nonfinite blocker appears.

If smoke fails, do not tune benchmark-specific thresholds or alphas. Stop this
branch and retain the immutable CMC result.
