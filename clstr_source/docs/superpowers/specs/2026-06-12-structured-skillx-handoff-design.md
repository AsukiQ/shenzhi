# Structured SkillX Handoff Design

## Goal

Make CLSTR-selected SkillX references help the AppWorld executor without exposing misleading SkillX titles or free-form metadata that Qwen can turn into nonexistent APIs or wrong task strategies.

## Problem

The current `safe_metadata` prompt still exposes SkillX IDs, names, descriptions, and executor metadata as natural language. On dev57, this made Stage4 worse than qwen-only: retrieved skills were often topically related but operationally misleading. The failure is mostly at the SkillX-to-executor boundary, not in Stage4 checkpoint loading or transition tensor shape.

## Design

Add a new `skill_context_mode="schema_plan"` for AppWorld executor prompts.

In this mode, each selected skill is converted into a schema-grounded constraint block:

- Hide SkillX `skill_id`, `name`, raw body, and title-derived API candidates.
- Extract explicit `apis.<app>.<api>` refs from `executor_desc`, `body`, and `skill_md`.
- Keep only refs that exist in the full AppWorld API schema for the current required apps.
- Separate read/support APIs from state-changing action APIs.
- Filter state-changing APIs unless their action verb appears in the user goal.
- Tell the executor that the user goal is authoritative and the block is only an API constraint.
- Drop skills that have no valid schema-grounded refs.

Prompt-visible content should describe allowed APIs and constraints, not SkillX titles. Preflight remains the final safety net and already uses the full schema.

## Non-Goals

- Do not retrain Stage0/Stage1/Stage2/Stage4 in this phase.
- Do not change CLSTR ranking scores in this phase.
- Do not run dev57/full until regression6 and dev10 gates look healthy.

## Validation Gates

1. Unit tests must prove `schema_plan` hides misleading titles while preserving valid APIs.
2. Unit tests must prove multistep prompt construction can use full schema refs even when prompt API docs are truncated.
3. Existing AppWorld executor/multistep tests must pass.
4. Only after tests pass, run a 6-task regression sbatch. Do not proceed to dev57 unless this gate improves over the previous `0/6`.
