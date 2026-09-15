# ToolSandbox Causal Replay Evaluation Design

## Goal

Measure the existing Qwen CLSTR checkpoint under a causally aligned ToolSandbox replay prefix without changing training or checkpoint weights.

## Problem

ToolSandbox rows describe the tool required next: `skill_id/action_text` identify the previously executed tool, while `next_skill_id` identifies the tool executed to advance to the next milestone. The generic auto-replay builder currently replays the former, creating an off-by-one event such as `START -> oracle arguments` instead of replaying the required tool call. The corpus also lacks real tool-result observations.

## Design

Add a ToolSandbox-only replay adapter with two explicit modes:

- `legacy_auto`: retain the current generic replay behavior for reproducibility.
- `corrected_causal`: for every prior milestone, replay `next_skill_id` as the executed skill, serialize its oracle arguments as action context, and use the following milestone's `state_text` as a weak post-action state observation. Mark every step as `next_state_without_tool_result` so it cannot be confused with a real environment observation.

The adapter attaches explicit `replay_prefix` objects before the shared evaluator runs. The shared evaluator already preserves explicit prefixes, so no Stage2/Stage4 training code changes are required.

## Evaluation contract

Use the same Stage0, Stage2, Stage4, skill pool, candidate pool, row denominator, and reliability settings for:

1. static endpoint;
2. legacy auto replay;
3. corrected causal replay.

Report dynamic-minus-static MRR and replay provenance. Do not promote the corrected adapter to shared training until this diagnostic is complete.

## Retraining

None for this phase. A later method-level replay change would reuse Stage0 and Stage1 and retrain Stage2, Stage4, and the reliability gate.
