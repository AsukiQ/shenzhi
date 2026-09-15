# Anchored Harm-Suppression Gate Design

## Objective

Replace the failed free-form direct-utility alpha gate with a baseline-preserving
gate that starts from the strongest train-selected fixed CMC mixture and only
suppresses memory when counterfactual evidence predicts that memory is harmful.
The Stage2 router, Stage4 residual adapter, candidate union, and Qwen backbone
remain immutable.

## Evidence and scope

The selected validation split gives:

- static macro MRR: `0.478447`
- dynamic endpoint macro MRR: `0.666430`
- train-selected fixed `alpha=0.85` macro MRR: `0.674411`
- fixed-or-static utility-oracle macro MRR: `0.696175`
- fixed-or-static rank-oracle macro MRR: `0.702589`

The one-sided architecture therefore has `0.021763` utility-oracle MRR
headroom over the fixed baseline. A two-sided choice among static, fixed, and
dynamic gives no material utility-oracle improvement over the one-sided design,
so upward alpha correction is out of scope.

## Scoring contract

For a route row with causal history:

```text
base_logits = fuse(static_logits, dynamic_logits, alpha_base)
p_harm      = HarmGate(memory_utility_features)
alpha       = alpha_base * (1 - p_harm)
final       = fuse(static_logits, dynamic_logits, alpha)
```

For a row without causal history, `alpha` is exactly zero. `alpha_base` is
selected on calibration-train only from the declared fixed-alpha grid and is
bound into the audit, checkpoint, final-chain overlay, and benchmark manifests.

The gate is a shared linear `11 -> 1 -> sigmoid` model. Benchmark names, source
IDs, task IDs, and skill IDs are forbidden inputs.

## Supervision and optimization

The counterfactual target is based on the deployed endpoints:

```text
harm_delta = utility(static_logits) - utility(base_logits)
harm_target = sigmoid(harm_delta / temperature)
confidence = clamp(abs(harm_delta) / temperature, 0, 1)
```

The BCE term is balanced across harmful and helpful utility signs inside each
source so that the harmful minority is not drowned by the helpful majority.
The full objective is:

```text
L = L_fused_rank
  + L_base_no_regret
  + L_balanced_harm_BCE
```

`L_base_no_regret` compares the learned fused utility against the fixed-alpha
baseline, not against static. Only the gate weight and bias are trainable. The
existing trajectory-disjoint train/dev split, seed 17, temperature 0.5,
learning rate 0.01, and 300 full-batch steps remain fixed.

## Fail-closed audit and promotion

Before fitting, the audit must evaluate the exact static-versus-fixed decision.
It must show both harm signs, positive fixed-or-static oracle headroom, a
trajectory-disjoint linear selector improvement over fixed alpha, and bounded
source regret. If this audit fails, no gate is trained.

After fitting, promotion requires all of the following:

- exact alpha-zero/static and alpha-one/dynamic endpoint behavior;
- exact zero-history static fallback;
- dev source-balanced macro MRR strictly above the train-selected fixed-alpha
  baseline;
- worst source regret versus fixed alpha at most `0.005`;
- harmful-row fused MRR above fixed-alpha harmful-row MRR;
- helpful-row fused MRR regret versus fixed alpha at most `0.005`;
- mean alpha on harmful rows at least `0.10` below mean alpha on helpful rows.

There is no arbitrary requirement that any alpha be below `0.25`. Outcome and
separation checks replace that proxy.

If either audit or promotion fails, no checkpoint is emitted and benchmark
submission is forbidden. The release fallback is the immutable Stage4 CMC
checkpoint with the train-selected fixed alpha.

## Verification and benchmark gate

All torch/pytest, recalibration, and evaluation commands run through Slurm.
Focused RED/GREEN tests cover endpoint construction, sign balancing, anchored
alpha mapping, audit thresholds, promotion, serialization, overlay loading, and
fail-closed orchestration. Only a promoted checkpoint may enter the aligned
ToolBench, ToolSandbox, Tau2-base, and ALFWorld benchmark chain.
