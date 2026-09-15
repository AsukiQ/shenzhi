# Stage2 Counterfactual Memory Audit Design

## Goal

Determine whether recurrent memory `m_t` contains causal next-skill evidence or
merely perturbs Stage2 scores. The audit is diagnostic only and must not train,
select, or promote a checkpoint.

## Compared Conditions

For identical current-state rows and legal candidate pools, evaluate:

1. `static_no_replay`: remove every replay prefix and use `m_0`.
2. `dynamic_replay_no_online`: use the true causal replay prefix and raw
   Stage2 unified-memory scores, without online bonuses or a Stage4 gate.
3. `shuffled_replay_no_online`: replace each non-empty prefix with a prefix
   from a different trajectory in the same benchmark and prefix-length bucket.
4. `masked_replay_no_online`: preserve prefix structure but blank its skill,
   action, and observation evidence; this must behave like the static control.
5. Keep the existing `dynamic_replay_with_online` condition as a separate
   legacy diagnostic, not as evidence about Stage2 `m_t`.

Existing benchmark-provided corrected causal prefixes take precedence over
generic auto replay. Rows without a prefix may receive the existing deterministic
auto replay with at most three prior steps.

## Metrics

Persist MRR, Recall@1, Recall@5, true-minus-static, true-minus-shuffled,
shuffled-minus-static, memory-active coverage, and the existing pairwise rank,
memory-distance, and logit-difference diagnostics.

## Decision Rule

- If true replay beats shuffled replay and static on ToolBench, `m_t` contains
  useful trajectory evidence; repair reliability/fusion before redesigning
  Stage2.
- If true and shuffled replay are indistinguishable, redesign Stage2 with
  counterfactual history negatives.
- ToolSandbox and Tau2-base diagnose whether history is irrelevant or harmful;
  they are not used to select a checkpoint.

## Execution

Run through the existing ToolSandbox and Tau2-base evaluators with their aligned
candidate pools and Stage2 checkpoint. Use small diagnostic subsets first. Run
ToolBench from the already aligned raw-dynamic/static evidence unless a shuffled
variant is needed after the first decision gate.
