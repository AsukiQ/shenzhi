# BGE-M3 CLSTR Backbone-Substitution Design

## Objective

Produce a paper-grade BGE-M3 CLSTR run that differs from the Qwen0.6B CLSTR
mainline only where the backbone interface requires it. Train Stage0/1/2/4,
select the safe-memory Stage4 artifact on fixed validation, and evaluate the
same immutable final chain on ToolBench, Tau2, ToolSandbox, and ALFWorld.

The primary scientific role is a controlled backbone-substitution comparison,
not a separately tuned BGE method.

## Invariants Shared with Qwen

- Use the identical data manifest, trajectory-disjoint split, benchmark caps,
  sampling seed, negative-mining protocol, candidate budgets, losses, and
  Stage0/1/2/4 step schedules.
- Keep the complete BGE backbone frozen in every stage.
- Preserve `d=1024`, identity projection initialization, normalized
  embeddings, the `clstr_causal_state_v1` state-query contract, and the
  `skillret_official` skill serializer.
- Use the same Stage4 immutable-router policy, delta-only checkpoint contract,
  fixed-validation checkpoint selection, optional audit-bound learned gate,
  and exact static fallback.
- Use the same frozen/native/closed-loop benchmark corpora, denominators,
  smoke gate, and full evaluation protocol.
- Never reuse Qwen checkpoint, lineage, final-chain, handoff-cache, or
  evaluation output directories for BGE artifacts.

## Backbone-Specific Profile

The BGE profile changes only the encoder interface:

- Backbone: local `models/BAAI/bge-m3`.
- Pooling: CLS for state and skill encodings.
- Tokenizer padding: right.
- Hidden and CLSTR dimension: 1024.
- Projection: identity-initialized and trainable exactly as in Qwen Stage0.
- Query serialization: retain the Qwen CLSTR causal state prompt; do not add a
  BGE retrieval instruction in the primary comparison.
- Precision: start with float32 for compatibility and numerical stability.
  A bfloat16 variant is allowed only after a parity smoke proves equivalent
  routing behavior and materially lowers runtime or memory.

Any further BGE-specific prompt, learning-rate, unfreezing, or mining change
belongs to a separately labeled tuned-BGE ablation and must not replace the
primary substitution result.

## Architecture and Artifact Model

Introduce a small backbone-profile layer used by launchers and artifact
validators. The profile declares family, model path, basename, pooling,
padding, precision, frozen status, and expected hidden dimension. Existing
Qwen behavior remains the default and existing Qwen manifests remain readable.

Generalize the Qwen-specific lineage/final-chain/multibench orchestration only
as far as necessary to accept an explicitly declared `bge_m3` profile. Keep
the same checkpoint roles and reliability identity. Artifact validation must
fail closed if a BGE chain points to a Qwen backbone, uses non-CLS pooling,
unfreezes the backbone, or mixes checkpoint families.

Use an isolated branch and worktree with an independent run root such as
`outputs/bge_m3_clstr_base`. Cache identities must include the backbone
profile, so BGE can use the schedule-aware cache without colliding with Qwen.

## Training Flow

1. Run a BGE Stage0 smoke proving CLS/right-padding configuration, frozen
   backbone, identity projection, finite loss, and correct cache identity.
2. Run Stage0 full for 5000 steps and apply the existing selection gate.
3. Run Stage1 for 3000 steps and Stage2 for 10000 steps with the same quality
   and lineage gates as Qwen.
4. Run safe-memory Stage4 smoke and full for 3000 steps. Select by fixed
   validation rather than final optimizer step, then qualify fixed-alpha or
   learned reliability under the same thresholds.
5. Resolve one immutable BGE final-chain manifest.

## Evaluation Flow

Run the same nine-stage evaluation chain:

- Frozen: ToolBench, ToolSandbox, Tau2, ALFWorld-offline.
- Native: ToolBench, ToolSandbox, Tau2.
- Closed loop: ALFWorld valid-seen and valid-unseen with
  `stateful_post_action_v1`.

Smoke must pass semantic parity and structural validation before full jobs are
submitted. All full jobs must share the same BGE final-chain SHA, checkpoint
chain digest, reliability identity, and accepted smoke-gate SHA.

## Failure Handling

- A failed BGE smoke blocks full training; it does not change Qwen artifacts.
- A Stage4 release failure preserves Stage2 as the safe endpoint and records
  the failed Stage4 attempt without weakening thresholds.
- A learned-gate failure falls back to the validation-selected fixed alpha.
- OOM or throughput issues may change batch size or cache batch size, but not
  effective examples, data order, optimizer-step count, or method semantics.
- Backbone unfreezing requires explicit user approval and is outside this
  design.

## Verification and Reporting

Tests must cover profile resolution, BGE checkpoint/lineage identity,
cache-key separation from Qwen, launcher invariants, final-chain loading, and
multibench export/report validation. Pytest runs only through Slurm with a GPU
allocation and `CUDA_VISIBLE_DEVICES=""`.

The final comparison reports both Qwen0.6B and BGE-M3 with identical protocol
metadata, plus any unavoidable backbone-interface differences. No tuned-BGE
result may be presented as the strict substitution row.
