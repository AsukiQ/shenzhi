# Qwen06 CLSTR Multi-Benchmark Acceleration Design

## Goal

Reduce the wall time of the final ToolBench-G3, ToolSandbox, Tau2, and
ALFWorld evaluations without changing checkpoint lineage, benchmark rows,
prompts, candidate pools, model scores, prediction ordering, metric formulas,
or report labels.

## Selected Scope

The user selected the bounded, semantics-preserving approach:

1. remove native evaluator passes whose result is provably redundant or
   discarded;
2. use benchmark-specific batches only where the workload and available A800
   memory justify them;
3. make ALFWorld batch sizes greater than one safely resumable;
4. require final-checkpoint parity evidence before accelerated full runs.

Two broader alternatives were rejected. Merely increasing every batch size
would leave duplicate model passes and ALFWorld's resume failure unresolved.
Multi-GPU sharding and new persistent embedding/candidate caches would add
merge and identity complexity that is unnecessary for this evaluation round.

## Native ToolBench and ToolSandbox

ToolBench-G3 evaluates one benchmark label but currently invokes the full
Stage2/4 evaluator four times: prior overall, prior grouped by benchmark,
Stage4 overall, and Stage4 grouped by benchmark. The grouped outputs are
mathematically identical to wrapping the corresponding overall result as
`{"toolbench_g3": result}`. The two grouped model passes will be replaced by
those mappings. The Stage4 overall pass remains the only pass that writes route
records, so route-record semantics do not change.

ToolSandbox currently performs base and Stage4 evaluation, then executes a
third grouped evaluator whose return value is discarded. The third call will
be removed. Characterization tests will assert the same top-level report,
row-derived metrics, and evaluator arguments for the two retained passes.

Tau2 already follows the selected single-benchmark mapping pattern and remains
the reference behavior.

## Benchmark-Specific Batch Profiles

Existing batch values remain the fallback profile:

- frozen routing: batch 16;
- native Stage2/4 routing: batch 8;
- native Stage0 candidate ranking: batch 16;
- closed-loop ALFWorld: batch 1.

The accelerated Tau2 profile is:

- frozen Tau2: batch 128;
- native Tau2 Stage2/4 evaluation: batch 64;
- native Tau2 Stage0 candidate ranking: batch 128.

Other frozen and native benchmarks retain their current values. The selected
batch values are explicit stage exports and therefore participate in the
existing submission-plan fingerprint and resume drift checks.

Before a full Tau2 job is accepted, the final checkpoint must be evaluated on
the same bounded rows with the fallback and accelerated profiles. Frozen Tau2
requires identical row IDs, candidate IDs, ranked skill IDs, positive ranks,
and strict metrics. Native Tau2 requires identical Stage0 candidate order and
all Stage0/base/Stage4 ranking-derived metrics; fixture tests separately prove
per-row Stage2/4 batch invariance. The gate also records elapsed time and peak
memory. A mismatch, OOM, or non-finite score selects the fallback profile; it
never weakens the metric comparison or edits predictions after scoring.

## Resumable Batched ALFWorld

ALFWorld already executes vectorized environments and scores all active
episodes in one model call. Its current resume guard rejects any nonempty run
when `batch_size != 1`.

Resume will become batch-aware and fail closed:

- existing rows must have contiguous `episode_index` values and matching
  method/split identity;
- a complete run may be reopened even when its final partial batch is smaller
  than the configured batch;
- an incomplete run must contain a whole number of configured batches;
- skipped environment resets must reproduce the stored gamefile sequence;
- a non-aligned or identity-drifted file is rejected rather than guessed.

Each completed environment batch is persisted by atomically replacing the
canonical `run.jsonl` with all committed rows. A process failure while writing
the temporary file therefore leaves the previous complete-batch checkpoint
intact. Progress and final metrics continue to use the existing schema.

The accelerated ALFWorld profile is batch 4. Before full valid-seen or
valid-unseen evaluation, a bounded final-checkpoint smoke compares batch 1 and
batch 4 on identical episode identities. The gate requires identical
gamefiles, chosen action traces, per-episode success/reward/goal-condition
values, and aggregate metrics. Any mismatch or OOM selects batch 1 for that
split.

## Orchestration and Failure Handling

Training gates preempt evaluation work. All heavy parity and benchmark runs
execute through Slurm after the immutable Stage4 chain is available.

Acceleration metadata records the requested profile, effective profile,
parity result, row or episode denominator, checkpoint-chain digest, and corpus
manifest digest. Existing smoke-to-full acceptance remains mandatory. Full
jobs fail closed on plan drift, checkpoint/corpus drift, incomplete outputs,
non-finite scores, resume misalignment, or parity mismatch. A parity failure
is not a benchmark failure when the fallback profile completes successfully.

## Testing

Automated tests cover:

- ToolBench evaluator call count dropping from four to two while preserving
  both grouped report mappings;
- ToolSandbox evaluator call count dropping from three to two with unchanged
  report content;
- Tau2-only batch exports and unchanged batches for other benchmarks;
- exact frozen/native ranking comparison and fallback selection;
- ALFWorld batch-4 clean execution, aligned resume, complete partial-final
  resume, gamefile drift rejection, mid-batch rejection, and atomic-write
  recovery;
- batch-1/batch-4 episode and aggregate parity comparison;
- existing submitter, validator, report, launcher, and benchmark evaluator
  regression suites.

## Non-Goals

- changing benchmark data, prompts, candidates, decoding, or metrics;
- modifying SR/ToolREx worktrees or rerunning their completed results;
- unfreezing Qwen;
- adding multi-GPU evaluation, approximate ranking, or lossy caches;
- representing routing recall as end-to-end task success.
