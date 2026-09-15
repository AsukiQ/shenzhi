# CLSTR Qwen State-Query Prompt Consistency Design

## Status

Approved after conversation review on 2026-07-11.

## Goal

Give frozen Qwen3-Embedding-0.6B one explicit, versioned causal state-query contract across CLSTR Stage0 training, Stage0 handoff audits, Stage1/2/4 training, route evaluation, and real inference.

## Problem

The current Stage0 trainer and handoff audit prepend the SkillRouter-style Qwen instruction, while a model reconstructed from the Stage0 checkpoint defaults to raw state serialization. Stage1/2/4 embedding caches and route scoring can therefore encode raw `state_text` even though Stage0 was trained and selected with instructed queries.

This is a representation-contract mismatch, not merely a wording preference. A Stage0 gate can approve one input distribution while downstream dynamic memory routing uses another.

## Canonical Main Prompt

The primary CLSTR query prompt is English and benchmark agnostic:

```text
Instruct: Given an agent task or current execution state and interaction history, retrieve the skill or tool document most useful for the next action.
Query:{state_text}
```

Its version is:

```text
clstr_causal_state_v1
```

The wording deliberately specifies:

- task or current execution state, so static schema queries and trajectory states share one contract;
- interaction history, matching CLSTR's trajectory-memory claim;
- skill or tool document, matching the mixed 67,557-item pool;
- next action, matching causal next-skill routing rather than generic topical similarity.

No benchmark name is inserted into the instruction. ToolBench-, BFCL-, ALFWorld-, or TrajectBench-specific prompt variants are out of scope because they would introduce benchmark-specific tuning and weaken the general routing claim.

## Skill Documents and Transition Inputs

Skill documents retain the existing `skillret_official` serialization:

```text
{name} | {description} | {body}
```

They receive no `Instruct:` prefix. This follows Qwen3-Embedding's asymmetric retrieval contract: query inputs use an instruction, retrieval documents do not.

Action and observation strings are transition inputs rather than retrieval queries. They remain unchanged in this work. Adding `Action:` or `Observation:` role prompts is a separate architecture ablation and must not be mixed into the prompt-consistency fix.

## Text-Length Contract

The causal prompt uses this raw-state budget:

```text
state_query_max_chars = 2000
state_query_truncation = head_tail_v1
```

For an over-length state, `head_tail_v1` retains the beginning and end of the raw state with a visible truncation marker between them. This preserves the task goal commonly serialized near the beginning and the most recent observation/history commonly serialized near the end. The instruction text is outside the 2000-character raw-state budget.

## Single Serialization Boundary

One shared formatter owns state-query serialization. Callers pass raw state text and a prompt version; they do not prepend instructions themselves.

The model exposes two explicit encoding roles:

```text
encode_states(raw_state_texts)        -> apply the checkpoint state-query prompt
encode_observations(raw_other_texts)  -> encode text without a retrieval prompt
```

State and next-state call sites use `encode_states`. Actions, observations, next observations, and executor text use `encode_observations`. Shared embedding-cache helpers must receive an explicit text role instead of routing every field through one generic encoder function.

Input beginning with an `Instruct:` query wrapper is rejected by `encode_states` for the new Qwen run. Failing loudly exposes stale callers and prevents both double-prefixing and accidental use of a different pre-rendered instruction.

The formatter applies to every state-like retrieval representation:

- Stage0 retrieval training queries;
- current state `h_t`;
- next state `h_{t+1}`;
- Stage0 current/next handoff candidate generation;
- Stage0 fixed-row audits;
- Stage1/2/4 state and next-state embedding caches;
- current-state candidate-union evaluation;
- benchmark route evaluation and real rollout routing.

The formatter does not apply to:

- skill documents;
- action embeddings;
- observation and next-observation embeddings;
- executor LLM system/user prompts.

Centralizing this boundary prevents double-prefixing and prevents one stage from silently reverting to raw state text.

## Checkpoint and Lineage Contract

Every Stage0 checkpoint records:

```text
state_query_prompt_version
state_query_instruction
state_query_max_chars
state_query_truncation
skill_text_format
```

Checkpoint reconstruction restores these fields into the model configuration. Stage1/2/4 launchers, resumes, audits, and evaluators fail before model execution when their requested prompt contract differs from the parent Stage0 lineage.

Legacy checkpoints that only contain `query_text_format=skillrouter` map explicitly to:

```text
sr_task_description_v1
```

They must never be silently interpreted as the new causal prompt.

Reports include the rendered prompt version and contract but do not duplicate complete user/task text into lineage manifests.

## Tests and Failure Handling

Targeted tests must prove that:

1. the causal prompt version renders its exact instruction;
2. the causal prompt uses the specified head-tail state truncation;
3. already formatted state text is rejected rather than double-prefixed;
4. Stage0 receives raw query text and serializes it through the shared formatter;
5. checkpoint reconstruction restores the prompt contract;
6. Stage1/2/4 state and next-state encoding calls `encode_states` with the restored contract;
7. actions and observations call `encode_observations` and remain outside the query formatter;
8. lineage validation rejects prompt-version, instruction, length, or truncation mismatches;
9. legacy `query_text_format=skillrouter` checkpoints map only to the explicit legacy version.

Any missing or unsupported prompt version is an error for the new Qwen run. No fallback to raw state text is permitted.

## Out of Scope

- benchmark-specific query instructions;
- learned prompt tokens or prompt tuning;
- action/observation role-prompt ablations;
- Qwen reranker relevance prompts;
- changes to executor LLM prompts;
- backbone unfreezing or LoRA.

## Completion Criteria

The prompt-consistency work is complete when:

- all state-query paths use the shared versioned formatter;
- Stage0 training, fixed audit, Stage1/2/4, evaluation, and inference cannot disagree silently;
- checkpoint and lineage reports make prompt provenance auditable;
- all focused prompt, checkpoint, Stage0, Stage1/2/4, and route-evaluation tests pass.
