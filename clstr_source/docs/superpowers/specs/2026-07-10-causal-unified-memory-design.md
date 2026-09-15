# Causal Unified-Memory Mainline Design

## Objective

Make the implemented CLSTR paper method exactly match a four-stage mainline—Stage0, Stage1, Stage2, and Stage4—and remove code that suggests an unused Stage3/HRPO or memory-blind rollout contribution.

The repaired mainline must directly train the causal claim:

\[
m_0=\operatorname{Init}(h_0),
\quad
\hat m_{t+1}=\operatorname{Transition}(m_t,a_t,o_{t+1}),
\quad
b_{t+1}=\operatorname{Observation}(h_{t+1}),
\]

\[
m_{t+1}=\operatorname{Correction}(\hat m_{t+1},b_{t+1},o_{t+1}),
\quad
s_{t+1}=\operatorname{UnifiedRoute}(h_{t+1},m_{t+1}).
\]

The `next_skill_id` loss is computed from `s_{t+1}`. It must not silently fall back to scoring `(h_t,m_t)`.

## Repository Safety

- Baseline commit: `bfa56f0`.
- Backup branch: `backup/pre-causal-memory-fix-20260710`.
- Implementation branch: `fix/causal-unified-memory`.
- Generated results, datasets, checkpoints, caches, `.tmp/`, and `slurm_logs/` are not part of the source snapshot.
- Existing checkpoints remain historical comparison artifacts. Because the loss graph changes, they cannot be reported as checkpoints trained with the repaired semantics.

## Mainline Boundary

The retained training pipeline is:

1. Stage0: skill retrieval and candidate handoff.
2. Stage1: initialization of trainable CLSTR heads.
3. Stage2: supervised multi-loss training, including causal next-skill supervision.
4. Stage4: offline transition-conditioned next-skill specialization.

There is no Stage3 in the paper method. Closed-loop benchmark evaluation and teacher-trajectory extraction remain available, but neither is described as an HRPO training stage.

## Canonical Memory Initialization

Every new trajectory starts through one public initializer:

```python
m_0 = model.initial_belief(h_0)
```

This applies to static training, replay-prefix training, Stage4, and retained evaluators. Replay must encode the first state in the prefix and call `model.initial_belief()`; it must not initialize from raw `subspace_obs()`.

When replay is trainable, text embeddings may remain detached according to the frozen-encoder policy, but gradients must be able to reach `initial_belief_head`, `transition`, and `gate` across the bounded replay window. Non-trainable evaluation replay runs under `torch.no_grad()`.

The observation correction at later steps still uses an observation-derived memory target; canonicalizing `m_0` does not mean applying the initializer head at every recurrent update.

## Next-State Data Semantics

The next-skill target belongs to the state after the current action. Training therefore requires all of:

- `state_text` for `h_t`;
- `action_text` and current `skill_id` for `a_t`;
- `next_observation_text` for `o_{t+1}`;
- `next_state_text` for `h_{t+1}`;
- `next_skill_id` for the target.

`next_state_text` is attached from the immediately following row only when trajectory identity and step ordering agree. Attachment happens before smoke caps or batch shuffling so the last retained row does not lose an otherwise available next state.

Rows lacking a trustworthy adjacent next state are excluded from `L_trans_skill_ce` and Stage4 next-skill training with an auditable skip reason. The main loss does not fabricate a next state from the current state or quietly substitute `next_observation_text` for full state context.

## Shared Post-Action Update

Stage2 and Stage4 use one shared helper with the following logical interface:

```python
post_action_memory(
    model,
    h_t,
    m_t,
    current_skill_labels,
    action_embeddings,
    observation_embeddings,
    h_next,
) -> (m_hat_next, observation_memory_next, m_next)
```

The helper:

1. applies the current action and next observation to `transition`;
2. computes the observation memory from `h_next` and the skill table;
3. applies `gate`/correction;
4. returns `m_next` without detaching the main loss graph.

Both stages then call:

```python
dynamic_logits = model.unified_route_logits(h_next, m_next, candidate_rows=...)
static_memory = model.initial_belief(h_next)
static_logits = model.unified_route_logits(h_next, static_memory, candidate_rows=...)
```

This makes dynamic-versus-static comparisons share the same next state, candidate set, and scorer. Their only intended information difference is recurrent trajectory memory.

`L_trans` may remain as an auxiliary transition-consistency objective, but it is no longer the only current-action supervision reaching the memory updater.

## Stage4 Contract

Stage4 is an offline causal next-skill stage, not a hidden policy/RL stage.

For unified memory it trains the modules used by its loss graph:

- `initial_belief_head`;
- `transition`;
- `gate`;
- `action_proj`;
- `unified_retriever`.

The Stage4 entrypoint removes optional HRPO/preference arguments and reports. In particular, `lambda_pref`, `beta_kl`, preference rows, reference modules, and `joint_transition_act_with_optional_preference` naming are removed. The training objective is reported as causal transition-conditioned next-skill cross-entropy plus explicitly retained auxiliary terms.

## Removal and Migration Policy

The pre-change snapshot is the recovery mechanism; misleading source does not need to remain in the active branch.

Delete:

- Stage3 and Qwen-Stage3 modules, entrypoints, Slurm scripts, submission wrappers, and dedicated tests;
- HRPO/online-policy helpers used only by deleted Stage3 or the deleted Stage4 preference branch;
- generic memory-blind rollout plus legacy `train.py`/`infer.py` entrypoints and dedicated tests;
- stale README/current-method documentation and active scripts that advertise a Stage3 pipeline;
- temporary patch artifacts that are not part of the method.

Before deleting a mixed-purpose module, migrate retained pure utilities to a neutral module. For example, candidate-list positive injection used by validators must not force retention of a policy rollout module.

Retain:

- Stage0/1/2/4 training and checkpoint initialization;
- benchmark evaluators, including ALFWorld concrete-action closed-loop evaluation;
- teacher/expert rollout extraction used to build supervised data;
- route evaluation and memory-ablation tools after they adopt canonical initialization;
- historical planning and result records, clearly treated as records rather than active entrypoints.

Deletion is dependency-driven: a file is removed only after retained callers are migrated and import collection succeeds.

## Evaluation Semantics

The ALFWorld concrete-action scorer remains memory-conditioned through `skill_head(candidate_action_embeddings, m_t)`. Its replay initialization is changed to the canonical initializer.

Offline route evaluators must compare:

- dynamic causal memory at the evaluated state; and
- `model.initial_belief(h_t)` at that same state.

Existing pre-fix dynamic-versus-static numbers are historical and must be regenerated.

## Error Handling and Auditability

- A next-skill row missing `next_state_text`, current action, next observation, current skill, target skill, or target candidate is skipped or rejected before loss computation with a named reason.
- Stage2 and Stage4 reports include counts for causal rows, missing-next-state rows, replay-initialized rows, and direct post-action-update rows.
- Training fails loudly if unified-memory mode lacks `initial_belief`, `transition`, `gate`, or `unified_route_logits`.
- Declared trainable modules are audited on a representative batch; a required module with no gradient is a test failure.

## Testing Strategy

Implementation follows red-green-refactor cycles.

Required semantic tests:

1. Replay of a non-empty prefix calls `initial_belief_head` and does not initialize from bare `subspace_obs`.
2. Static and replay initialization use identical top-k support and initializer semantics.
3. Changing only the current action changes Stage2 and Stage4 dynamic next-skill logits/loss.
4. Changing only the next observation changes Stage2 and Stage4 dynamic next-skill logits/loss.
5. Changing `next_state_text` changes the routing state used by next-skill logits.
6. The main next-skill CE produces gradients for transition, gate, action projection, and unified retriever on a representative row.
7. Static comparison uses `initial_belief(h_next)`, not the current-state initializer.
8. Missing or non-adjacent next-state rows are excluded and counted.
9. Active code, scripts, README, and tests contain no Stage3/HRPO training entrypoint references.
10. Retained validators and evaluators import after generic rollout removal.

Verification includes targeted unit tests, the complete affected test modules, Python import/compile checks, shell syntax checks for retained launchers, and a repository-wide reference scan.

## Success Criteria

The change is complete when:

- Stage2 and Stage4 next-skill logits are causally downstream of the current action and next observation;
- replay and static paths share the same trainable initializer;
- Stage4 has no Stage3/HRPO dependency or preference branch;
- the active repository presents only Stage0/1/2/4 as the CLSTR training pipeline;
- retained evaluation code uses consistent memory initialization;
- all affected tests pass and semantic counterfactual tests fail on the backup implementation but pass on the repaired implementation.
