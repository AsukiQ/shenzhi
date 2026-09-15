# Causal Unified-Memory Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage2 and Stage4 train next-skill ranking from the causally updated post-action memory, canonicalize trajectory initialization, and remove the unused Stage3/HRPO/generic-rollout training surface.

**Architecture:** Enrich ordered training rows with a verified adjacent `next_state_text`, initialize every trajectory through `model.initial_belief`, and share one strict differentiable post-action memory helper between Stage2 and Stage4. Stage4 becomes a pure offline causal next-skill stage. Retained utilities are moved out of legacy modules before the Stage3/HRPO/rollout dependency closure is deleted.

**Tech Stack:** Python 3.11, PyTorch, pytest, Bash launchers, Git.

---

## File Structure

### Core files modified

- `clstr/full_base_train.py`: adjacent-next-state enrichment, canonical replay initialization, shared post-action memory update, causal Stage2 CE, embedding/cache semantics.
- `clstr/stage4_act_train.py`: causal Stage4 CE, unified freeze policy, pure Stage4 training loop without preference/HRPO.
- `scripts/run_clstr_stage4_act_train.py`: remove preference and conditional-transition CLI surface.
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`: launch the causal unified Stage4 contract.
- `clstr/stage4_quality_gate.py`: accept and audit only the causal Stage4 objective.
- `clstr/current_state_route_eval.py`: keep offline route benchmarks on their current-state prediction contract instead of reusing causal Stage4 training loss.
- `scripts/audit_clstr_unified_training_readiness.py`: expose only Stage0/1/2/4 readiness.
- `README.md`: document the four-stage mainline and distinguish evaluation/data rollouts from training.

### New focused utility files

- `clstr/candidate_utils.py`: pure candidate-list positive injection migrated from generic rollout.
- `clstr/device_utils.py`: retained device resolution migrated from legacy training.
- `tests/test_candidate_utils.py`: behavior tests for positive-candidate injection.
- `tests/test_device_utils.py`: device resolution tests independent of legacy training.

### Primary tests modified

- `tests/test_full_base_train.py`: next-state enrichment, initializer, causal Stage2 counterfactuals, gradient graph.
- `tests/test_stage4_act_train.py`: causal Stage4 counterfactuals, gradient graph, freeze behavior, preference removal.
- `tests/test_stage4_quality_gate.py`: causal objective only.
- `tests/test_current_state_route_eval.py`: current-state route semantics and canonical dynamic/static initialization.
- `tests/test_unified_training_readiness.py`: four-stage readiness only.
- `tests/test_sbatch_scripts.py`: no Stage3 launchers and updated Stage4 arguments.
- `tests/test_qwen_external_encoder.py`, `tests/test_v4_action_proj_sharing.py`: remove Stage3-only cases while retaining Stage0/1/2/4 behavior.

### Files deleted after callers are migrated

- `clstr/stage3_unified_hrpo.py`
- `clstr/qwen_stage3_unified_hrpo.py`
- `clstr/stage3_quality_gate.py`
- `clstr/online_hrpo.py`
- `clstr/appworld_act_hrpo.py`
- `clstr/rollout.py`
- `clstr/train.py`
- `clstr/infer.py`
- Their dedicated launchers, configs, submission wrappers, and tests listed in Task 7.

`clstr/logged_online_stage4_train.py`, teacher rollout extraction, and closed-loop benchmark evaluators remain because they are retained offline-data/evaluation infrastructure rather than Stage3 HRPO.

---

### Task 1: Attach Trustworthy Adjacent Next States

**Files:**
- Modify: `clstr/full_base_train.py:174-262,440-460,572-610,708-750,2040-2070,4452-4492`
- Modify: `clstr/stage4_act_train.py:220-468,1084-1137`
- Test: `tests/test_full_base_train.py:997-1065`
- Test: `tests/test_stage4_act_train.py:220-470`

- [ ] **Step 1: Write failing adjacent-state tests**

Add tests that exercise real row enrichment rather than asserting on a mock call:

```python
def test_attach_adjacent_next_states_requires_consecutive_matching_target():
    rows = [
        {
            "benchmark": "alfworld",
            "trajectory_id": "traj",
            "step_index": 0,
            "state_text": "state zero",
            "skill_id": "skill/a",
            "next_skill_id": "skill/b",
            "loss_mask": {"L_trans_skill_ce": True},
        },
        {
            "benchmark": "alfworld",
            "trajectory_id": "traj",
            "step_index": 1,
            "state_text": "state one",
            "skill_id": "skill/b",
        },
    ]

    prepared, report = _attach_adjacent_next_states(rows)

    assert prepared[0]["next_state_text"] == "state one"
    assert prepared[0]["next_state_source"] == "adjacent_trajectory_row"
    assert report["attached_rows"] == 1
    assert report["skip_reasons"] == {}
```

Also add separate cases for a step gap, duplicate step, different trajectory, empty next state, and `next_skill_id != next_row.skill_id`. Assert Stage2 masks only `L_trans_skill_ce` and records `causal_next_state_skip_reason`; assert Stage4 skips those rows entirely.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'adjacent_next_state or next_state_before_cap' \
  tests/test_stage4_act_train.py -k 'missing_next_state or copies_next_state'
```

Expected: collection/import failure for `_attach_adjacent_next_states` or assertion failures because no `next_state_text` is attached.

- [ ] **Step 3: Implement the enrichment helper**

Add a shared helper with a stable identity key and explicit masking:

```python
def _trajectory_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    provenance = _provenance(row)
    return (
        str(row.get("benchmark") or ""),
        str(provenance.get("source_id") or provenance.get("source_dataset") or ""),
        str(row.get("split") or provenance.get("split") or ""),
        str(row.get("trajectory_id") or ""),
    )


def _strict_step_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row["step_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _attach_adjacent_next_states(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = [dict(row) for row in rows]
    grouped: dict[tuple[str, str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        identity = _trajectory_identity(row)
        if identity[3]:
            grouped.setdefault(identity, []).append((index, row))

    attached = 0
    skip_reasons: Counter[str] = Counter()
    for indexed_rows in grouped.values():
        ordered = sorted(
            indexed_rows,
            key=lambda item: (_strict_step_index(item[1]) is None, _strict_step_index(item[1]) or 0, item[0]),
        )
        step_counts = Counter(_strict_step_index(row) for _index, row in ordered)
        for position, (index, row) in enumerate(ordered):
            if not (row.get("loss_mask") or {}).get("L_trans_skill_ce"):
                continue
            reason = ""
            current_step = _strict_step_index(row)
            if current_step is None:
                reason = "invalid_step_index"
            elif step_counts[current_step] != 1:
                reason = "duplicate_step_index"
            elif position + 1 >= len(ordered):
                reason = "missing_adjacent_next_row"
            else:
                next_row = ordered[position + 1][1]
                next_step = _strict_step_index(next_row)
                if next_step != current_step + 1:
                    reason = "nonconsecutive_next_step"
                elif str(row.get("next_skill_id") or "") != str(next_row.get("skill_id") or ""):
                    reason = "adjacent_next_skill_mismatch"
                elif not str(next_row.get("state_text") or "").strip():
                    reason = "missing_adjacent_next_state_text"
            updated = dict(prepared[index])
            if reason:
                loss_mask = dict(updated.get("loss_mask") or {})
                loss_mask["L_trans_skill_ce"] = False
                updated["loss_mask"] = loss_mask
                updated["causal_next_state_skip_reason"] = reason
                skip_reasons[reason] += 1
            else:
                updated["next_state_text"] = str(next_row["state_text"])
                updated["next_state_source"] = "adjacent_trajectory_row"
                attached += 1
            prepared[index] = updated
    return prepared, {
        "row_count": len(rows),
        "attached_rows": attached,
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }
```

Call it immediately after reading the raw trajectory rows, before split filtering, benchmark filtering/caps, smoke limits, Stage0 handoff, or shuffling. Group identity includes split/source so the helper cannot cross those boundaries. In Stage4, materialize raw source rows without an early source limit, attach adjacent states before eligibility/caps, require non-empty action/observation/next state, and copy `next_state_text` into the Stage4 row.

Add `next_state_text -> _next_state_embedding` to the embedding cache. Include the new field in Stage0 handoff identity/digest, and use it for the next-skill handoff query when present.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 command again, then:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'attach_auto_replay_prefixes or handoff_rows_digest or embedding_cache' \
  tests/test_stage4_act_train.py -k 'build_stage4_next_skill_rows'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add clstr/full_base_train.py clstr/stage4_act_train.py tests/test_full_base_train.py tests/test_stage4_act_train.py
git commit -m "fix: attach causal next states before training caps"
```

---

### Task 2: Canonicalize Replay Initialization

**Files:**
- Modify: `clstr/full_base_train.py:1794-1887`
- Modify: `tests/test_full_base_train.py:1842-1903`
- Modify: `clstr/alfworld_eval.py:599-791`
- Modify: `scripts/run_webshop_clstr_eval.py:130-270`
- Modify: `clstr/prior_gate_sweep.py:350-400`
- Modify: `clstr/stage2_transition_row_diagnostics.py:260-300`
- Modify: `clstr/stage2_memory_rerank_eval.py:90-130`
- Modify: `tests/test_alfworld_eval.py`
- Modify: `tests/test_webshop_eval.py`
- Modify: `tests/test_mt_ablation_eval.py`

- [ ] **Step 1: Write a failing initializer regression test**

Use a small real module whose initializer has a visible trainable transformation:

```python
def test_replay_prefix_initializes_through_model_initial_belief():
    model = _TrainableReplayPrefixModel()
    model.initial_belief_calls = 0
    original = model.initial_belief

    def recording_initial_belief(h, top_k=None):
        model.initial_belief_calls += 1
        return original(h, top_k=top_k)

    model.initial_belief = recording_initial_belief
    replayed, used = _replay_prefix_belief(
        model,
        [{
            "observation_text": "initial observation",
            "action_text": "prefix action",
            "next_observation_text": "prefix next observation",
            "skill_id": "skill",
        }],
        fallback_m=torch.full((1, 2), 99.0),
        skill_id_to_idx={"skill": 0},
        skill_count=2,
        device=torch.device("cpu"),
        trainable=True,
    )

    assert used is True
    assert model.initial_belief_calls == 1
    replayed.sum().backward()
    assert model.initial_belief_head.weight.grad is not None
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'replay_prefix_initializes_through_model_initial_belief'
```

Expected: `initial_belief_calls == 0` or the initializer-head gradient is absent.

- [ ] **Step 3: Replace raw replay initialization**

Inside `_replay_prefix_belief()` replace `_skill_logits_and_memory()` initialization with:

```python
initial_belief = getattr(model, "initial_belief", None)
if not callable(initial_belief):
    raise ValueError("causal replay requires model.initial_belief")
m = initial_belief(h_step)
if not trainable:
    m = m.detach()
```

Remove `detach_initial`; trainable replay keeps the bounded gradient path through the initializer and recurrent heads. Preserve `torch.no_grad()` for evaluator replay. Tighten auto-replay construction so prefix steps must also be consecutive and share benchmark/source/split identity; do not connect filtered step 0 directly to step 2.

For retained unified evaluators, remove raw-memory fallbacks: ALFWorld and WebShop unified scorers must raise a clear error when `initial_belief` is missing. Update retained prior/memory diagnostics to call `model.initial_belief(h)` for trajectory initialization; if an old diagnostic cannot represent the causal/current-state contract, delete its active launcher instead of silently keeping raw `subspace_obs` initialization.

- [ ] **Step 4: Verify canonical replay users**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'replay_prefix or initial_belief' \
  tests/test_alfworld_eval.py -k 'unified_memory' \
  tests/test_webshop_eval.py -k 'unified_memory' \
  tests/test_mt_ablation_eval.py
```

Expected: all selected tests pass, with static and replay paths using the same initializer semantics.

- [ ] **Step 5: Commit Task 2**

```bash
git add clstr/full_base_train.py clstr/alfworld_eval.py scripts/run_webshop_clstr_eval.py \
  clstr/prior_gate_sweep.py clstr/stage2_transition_row_diagnostics.py \
  clstr/stage2_memory_rerank_eval.py tests/test_full_base_train.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py tests/test_mt_ablation_eval.py
git commit -m "fix: canonicalize replay memory initialization"
```

---

### Task 3: Make Stage2 Next-Skill CE Causal

**Files:**
- Modify: `clstr/full_base_train.py:1901-1952,2040-2070,3990-4145`
- Modify: `tests/test_full_base_train.py:1599-1674,2645-2740`

- [ ] **Step 1: Replace the old behavior test with causal assertions**

The Stage2 test row must contain `next_state_text`. Replace the assertion `transition_calls == 0` with assertions that the current action/observation reach the updater and both route calls use `h_next`:

```python
assert model.transition_calls == 1
dynamic_h, dynamic_m, candidates = model.unified_route_calls[-2]
static_h, static_m, static_candidates = model.unified_route_calls[-1]
assert torch.equal(dynamic_h, model.encode_observations(["next full state"]))
assert torch.equal(static_h, dynamic_h)
assert candidates == static_candidates == [[2, 1, 0]]
assert not torch.equal(dynamic_m, static_m)
```

Add separate counterfactual tests that change only `action_text`, only `next_observation_text`, and only `next_state_text`. Compare recorded dynamic memory/logits and confirm static logits stay unchanged for action/observation changes.

Add one real backward test with only `L_trans_skill_ce=1` and assert finite nonzero gradients on `initial_belief_head`, `transition`, `gate`, `action_proj`, and `unified_retriever`; `trans_head` must have no gradient.

- [ ] **Step 2: Run Stage2 causal tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_full_base_train.py \
  -k 'stage2_unified_memory and (post_action or counterfactual or main_ce_gradient)'
```

Expected: old unified branch does not call transition, action/observation changes do not affect logits, or required gradients are missing.

- [ ] **Step 3: Add the strict shared post-action helper**

Add:

```python
def _post_action_memory(
    model: Any,
    *,
    m_t: torch.Tensor,
    current_skill_labels: torch.Tensor,
    action_embeddings: torch.Tensor,
    observation_embeddings: torch.Tensor,
    h_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transition = getattr(model, "transition", None)
    gate = getattr(model, "gate", None)
    skill_table = getattr(model, "skill_table", None)
    if transition is None or gate is None or skill_table is None:
        raise ValueError("causal unified memory requires transition, gate, and skill_table")
    if action_embeddings is None:
        raise ValueError("causal unified memory requires current action embeddings")
    action_input = _transition_action_input(
        model,
        current_skill_labels,
        observation_embeddings,
        action_emb=action_embeddings,
    )
    m_hat_next = transition(m_t, action_input, observation_embeddings)
    observation_memory_next = subspace_obs(skill_table, h_next)
    gamma = gate(m_hat_next, observation_memory_next, observation_embeddings)
    m_next = gamma * observation_memory_next + (1.0 - gamma) * m_hat_next
    return m_hat_next, observation_memory_next, m_next
```

Do not use the detached auxiliary target helper and do not retain a `TypeError` fallback in this mainline helper.

- [ ] **Step 4: Route Stage2 CE from post-action state and memory**

Encode `next_state_text` as `_next_state_embedding`, call `_post_action_memory`, and in both candidate and full-pool paths use:

```python
static_memory_next = model.initial_belief(h_next)
trans_skill_logits = model.unified_route_logits(h_next, m_next, candidate_rows=candidate_rows)
static_logits = model.unified_route_logits(h_next, static_memory_next, candidate_rows=candidate_rows)
residual_logits = trans_skill_logits - static_logits
```

Remove the current-state `m_static_selected` from this branch. Pass `static_logits.detach()` into memory-margin comparisons so the loss cannot improve by depressing the static comparator.

Add metrics `transition_post_action_update_rows` and `transition_next_state_rows`.

- [ ] **Step 5: Run Stage2 tests and verify GREEN**

Run the Task 3 command, then:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_full_base_train.py
```

Expected: the full module passes.

- [ ] **Step 6: Commit Task 3**

```bash
git add clstr/full_base_train.py tests/test_full_base_train.py
git commit -m "fix: supervise post-action memory in stage2 routing"
```

---

### Task 4: Make Stage4 a Pure Causal Next-Skill Stage

**Files:**
- Modify: `clstr/stage4_act_train.py:1-50,705-1018,1021-1380`
- Modify: `scripts/run_clstr_stage4_act_train.py:50-240`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- Modify: `clstr/stage4_quality_gate.py`
- Modify: `tests/test_stage4_act_train.py`
- Modify: `tests/test_stage4_quality_gate.py`

- [ ] **Step 1: Write failing Stage4 causal and freeze tests**

Replace `_ForbiddenStage4Transition` with real recording/trainable transition and gate modules. Every Stage4 causal row includes `next_state_text`.

Assert:

```python
assert len(model.transition.calls) == 1
assert torch.equal(model.transition.calls[0].action_input, expected_action_embedding)
assert torch.equal(model.transition.calls[0].observation, expected_observation_embedding)
assert torch.equal(model.unified_route_calls[0].h, expected_next_state_embedding)
```

Add action, next-observation, and next-state counterfactual tests plus a main-CE gradient test. Update the freeze test so unified Stage4 with default arguments has trainable `initial_belief_head`, `transition`, `gate`, `action_proj`, and `unified_retriever`, while `trans_head` and `skill_head` stay frozen.

- [ ] **Step 2: Run Stage4 causal tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_act_train.py \
  -k 'unified_memory and (post_action or counterfactual or main_ce_gradient or freeze)'
```

Expected: transition is not called for the current row, route state is `h_t`, or gate remains frozen.

- [ ] **Step 3: Use the shared post-action update in Stage4**

Import `_post_action_memory`, encode `_next_state_embedding`, and compute:

```python
_, _, m_next = _post_action_memory(
    model,
    m_t=m_obs,
    current_skill_labels=labels,
    action_embeddings=action_emb,
    observation_embeddings=obs_emb,
    h_next=h_next,
)
static_memory = model.initial_belief(h_next)
logits = model.unified_route_logits(h_next, m_next, candidate_rows=candidate_rows)
static_logits = model.unified_route_logits(h_next, static_memory, candidate_rows=candidate_rows)
```

Rename unified diagnostic variables and report fields from ambiguous `prior` to `static` where they refer to `initial_belief(h_next)`. Detach `static_logits` in memory-margin loss.

- [ ] **Step 4: Repair Stage4 trainability**

For unified Stage4, `_freeze_for_stage4_act()` always enables exactly:

```python
for name in ("initial_belief_head", "transition", "gate", "action_proj", "unified_retriever"):
    _set_trainable(getattr(model, name, None), True)
```

Remove `train_transition` as a unified-mode switch and report `frozen_belief_gate=False`.

- [ ] **Step 5: Remove the hidden preference/HRPO branch**

Delete the `online_hrpo` and `stage3_unified_hrpo` imports, `_compute_stage4_preference_loss`, preference arguments/batches/reference modules, and `train_preference_head`. Reduce `_compute_joint_stage4_act_loss` to the causal Stage4 loss or replace its callers directly with `_compute_stage4_act_loss`.

Set report values to:

```python
"training_objective": "causal_transition_conditioned_next_skill_ce",
"training_regime": "offline_train_split_causal_next_skill",
"on_policy_rollout_used": False,
```

Remove `lambda_pref`, `beta_kl`, and preference CLI/environment options from all active Stage4 launchers. Update the Stage4 quality gate to accept only the causal objective.

- [ ] **Step 6: Run Stage4 tests and verify GREEN**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_act_train.py tests/test_stage4_quality_gate.py
```

Expected: both modules pass and no Stage4 preference/HRPO symbols remain.

- [ ] **Step 7: Commit Task 4**

```bash
git add clstr/stage4_act_train.py clstr/stage4_quality_gate.py \
  scripts/run_clstr_stage4_act_train.py scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  tests/test_stage4_act_train.py tests/test_stage4_quality_gate.py
git commit -m "fix: make stage4 causal and remove hrpo preference"
```

---

### Task 5: Separate Current-State Route Evaluation from Causal Stage4

**Files:**
- Create: `clstr/current_state_route_eval.py`
- Create: `tests/test_current_state_route_eval.py`
- Modify: `clstr/logged_online_stage4_train.py:1166-1345`
- Modify: `tests/test_logged_online_stage4_train.py`
- Verify callers: `clstr/toolbench_full_clstr_route_eval.py`, `clstr/tau2_route_eval.py`, `clstr/toolsandbox_route_eval.py`, `clstr/bfcl_route_eval.py`, `clstr/apibank_route_eval.py`, `clstr/global_pool_route_eval.py`

- [ ] **Step 1: Write failing current-state evaluator tests**

Use rows whose contract is “select the target skill at the supplied current state.” Do not provide `next_state_text`, because these rows are not causal Stage4 training rows.

```python
def test_current_state_route_eval_uses_state_and_replayed_memory_without_post_action_update():
    row = {
        "state_text": "current route state",
        "skill_idx": 1,
        "positive_next_skill_idx": 1,
        "candidate_next_skill_indices": [0, 1, 2],
        "replay_prefix": [{
            "observation_text": "initial route state",
            "action_text": "historical action",
            "next_observation_text": "historical observation",
            "skill_idx": 0,
        }],
    }

    _loss, metrics = _compute_current_state_route_loss(
        model, [row], torch.device("cpu")
    )

    assert model.route_calls[-2].state_text == "current route state"
    assert model.post_action_transition_calls == 0
    assert metrics["current_state_route_count"] == 1.0
```

Add tests proving that a current-row `action_text` change does not change current-state logits, while a replay-prefix change does. Assert static memory is `initial_belief(h_t)` and dynamic replay starts from the same initializer.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_current_state_route_eval.py
```

Expected: the module/function does not exist.

- [ ] **Step 3: Implement a dedicated unified current-state evaluator**

Create a focused function with no current-action update:

```python
def _compute_current_state_route_loss(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    h_t = _batch_cached_or_encode(model, rows, "_state_embedding", "state_text", device)
    static_memory = model.initial_belief(h_t)
    dynamic_memory, replay_used = _apply_replay_prefix_beliefs(
        model,
        rows,
        static_memory,
        _stage4_skill_id_to_idx(rows),
        int(model.skill_table.E.size(0)),
        device,
        trainable=False,
    )
    candidate_rows, mask, targets = _pad_candidate_indices(rows, device)
    dynamic_logits = model.unified_route_logits(h_t, dynamic_memory, candidate_rows=candidate_rows)
    static_logits = model.unified_route_logits(h_t, static_memory, candidate_rows=candidate_rows)
    dynamic_logits = dynamic_logits.masked_fill(~mask, torch.finfo(dynamic_logits.dtype).min)
    loss = F.cross_entropy(dynamic_logits, targets)
    metrics = {
        **_masked_ranking_metrics(dynamic_logits, targets, mask),
        "current_state_route_count": float(len(rows)),
        "current_state_replay_prefix_used_count": float(replay_used),
        "route_scorer": "unified_memory",
        "transition_skill_head_type": "unified_memory_retriever",
        "uses_stage0_prior_at_inference": False,
    }
    static_metrics = _masked_ranking_metrics(static_logits, targets, mask)
    metrics.update({
        f"stage4_unified_static_{key.removeprefix('stage4_')}": value
        for key, value in static_metrics.items()
    })
    metrics.update(_dynamic_static_rank_metrics(
        dynamic_logits,
        static_logits.masked_fill(~mask, torch.finfo(static_logits.dtype).min),
        targets,
        prefix="stage4_unified",
    ))
    return loss, metrics
```

Import `_stage4_skill_id_to_idx`, `_pad_candidate_indices`, and `_masked_ranking_metrics` from `stage4_act_train`, plus `_dynamic_static_rank_metrics` and the replay/encoding helpers from `full_base_train`. This module supports the retained unified-memory paper route scorer only and fails loudly if `initial_belief` or `unified_route_logits` is absent. Preserve compatibility metric names required by current table wrappers while also emitting explicit `current_state_route_*` names.

- [ ] **Step 4: Switch logged route evaluation to the new contract**

In `evaluate_logged_online_stage4_rows()`, replace `_compute_stage4_act_loss()` with `_compute_current_state_route_loss()`. Keep batching, resumption, progress files, and aggregation unchanged. The causal `_compute_stage4_act_loss()` remains used only for Stage4 training/causal validation.

- [ ] **Step 5: Run evaluator and wrapper tests GREEN**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py
```

Expected: all selected tests pass; route rows without `next_state_text` remain valid and no wrapper calls the causal Stage4 loss.

- [ ] **Step 6: Commit Task 5**

```bash
git add clstr/current_state_route_eval.py clstr/logged_online_stage4_train.py \
  tests/test_current_state_route_eval.py tests/test_logged_online_stage4_train.py
git commit -m "refactor: separate current-state route evaluation"
```

---

### Task 6: Migrate Retained Utilities

**Files:**
- Create: `clstr/candidate_utils.py`
- Create: `clstr/device_utils.py`
- Create: `tests/test_candidate_utils.py`
- Create: `tests/test_device_utils.py`
- Modify: `clstr/validator.py`
- Modify: `clstr/appworld_act_verified_pairs.py`
- Modify: `clstr/appworld_clstr_eval.py`
- Modify: `scripts/run_appworld_multistep_executor_eval.py`

- [ ] **Step 1: Write failing utility tests**

Candidate behavior:

```python
def test_inject_positive_candidate_replaces_last_negative_without_duplicates():
    assert inject_positive_candidate([2, 4, 6], positive=8, k=3) == [2, 4, 8]
    assert inject_positive_candidate([2, 8, 6], positive=8, k=3) == [2, 8, 6]
```

Device behavior:

```python
def test_resolve_device_honors_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
```

Copy the complete existing behavior contract from `rollout._inject_positive_candidate` and `train.resolve_device`, including empty rows, `k<=0`, CUDA availability, and auto selection.

- [ ] **Step 2: Run utility tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_candidate_utils.py tests/test_device_utils.py
```

Expected: new modules do not exist.

- [ ] **Step 3: Add focused utilities and migrate imports**

Move the production implementations without behavior changes. Update retained callers to import from the new modules. Do not retain compatibility re-exports in `rollout.py` or `train.py`, because those files will be deleted.

- [ ] **Step 4: Verify retained callers**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_candidate_utils.py tests/test_device_utils.py \
  tests/test_validator.py tests/test_appworld_act_verified_pairs.py \
  tests/test_appworld_clstr_eval.py tests/test_appworld_multistep_cli.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add clstr/candidate_utils.py clstr/device_utils.py clstr/validator.py \
  clstr/appworld_act_verified_pairs.py clstr/appworld_clstr_eval.py \
  scripts/run_appworld_multistep_executor_eval.py \
  tests/test_candidate_utils.py tests/test_device_utils.py
git commit -m "refactor: migrate utilities out of legacy rollout"
```

---

### Task 7: Delete Stage3, HRPO, Generic Rollout, and Legacy Train/Infer

**Files:**
- Delete source: `clstr/stage3_unified_hrpo.py`, `clstr/qwen_stage3_unified_hrpo.py`, `clstr/stage3_quality_gate.py`, `clstr/online_hrpo.py`, `clstr/appworld_act_hrpo.py`, `clstr/rollout.py`, `clstr/train.py`, `clstr/infer.py`
- Delete Stage3/HRPO scripts: `scripts/audit_clstr_stage3_quality.py`, `scripts/run_clstr_unified_stage3_hrpo.py`, `scripts/run_clstr_qwen3_unified_stage3_hrpo.py`, `scripts/run_clstr_qwen3_online_hrpo.py`, `scripts/run_clstr_v4_1b_alfworld_online_hrpo.py`, `scripts/run_appworld_clstr_hrpo_train.py`, `scripts/submit_clstr_stage3_after_stage2_gate.sh`, `scripts/submit_clstr_stage4_after_stage3_gate.sh`, `scripts/plot_clstr_loss_curve.py`, `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`, `scripts/sbatch/run_clstr_v4_1b_alfworld_online_hrpo_smoke.sh`, `scripts/sbatch/run_appworld_clstr_hrpo_train.sh`, `scripts/sbatch/run_phase1_dev10_gate.sh`
- Delete legacy scripts/config: `scripts/train_stage.sh`, `scripts/infer_static.sh`, `scripts/infer_agentic.sh`, `scripts/run_clstr_smoke_train.sh`, `scripts/run_clstr_clean_router_mini_train.sh`, `scripts/sbatch/run_appworld_clstr_train.sh`, `configs/train/`
- Delete dedicated tests: `tests/test_stage3_unified_hrpo.py`, `tests/test_stage3_quality_gate.py`, `tests/test_online_hrpo.py`, `tests/test_appworld_act_hrpo.py`, `tests/test_appworld_hrpo_cli.py`, `tests/test_phase1_advantage_fallback.py`, `tests/test_phase1_reward_shaping_audit.py`, `tests/test_phase1_stop_head_training_signal.py`, `tests/test_phase1_sbatch_plumbing.py`, `tests/test_phase2_logging.py`, `tests/test_rollout.py`, `tests/test_train_smoke.py`, `tests/test_infer_pipeline.py`
- Modify: `tests/test_qwen_external_encoder.py`, `tests/test_v4_action_proj_sharing.py`, `tests/test_phase1_closed_loop_gradient.py`, `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Add a failing active-surface scan test**

In `tests/test_mainline_surface.py`, define the retained active roots and assert that source/scripts/README do not advertise Stage3/HRPO training:

```python
def test_active_training_surface_contains_only_stage0_stage1_stage2_stage4():
    forbidden = (
        "stage3_unified_hrpo",
        "stage3_joint",
        "run_clstr_qwen3_online_hrpo",
        "run_clstr_v4_1b_alfworld_online_hrpo",
    )
    active_paths = [Path("clstr"), Path("scripts"), Path("README.md")]
    hits = []
    for root in active_paths:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    hits.append((str(path), token))
    assert hits == []
```

Exclude historical `.planning/`, `docs/superpowers/plans/`, and output records from this active-surface assertion.

- [ ] **Step 2: Run the scan and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_mainline_surface.py
```

Expected: it reports current Stage3/HRPO files and active references.

- [ ] **Step 3: Delete the closed dependency set**

Delete the listed files using `apply_patch`. From mixed tests, remove only Stage3/HRPO imports and cases; preserve Qwen encoder, action-projection, and direct causal gradient coverage. Do not delete teacher rollout, benchmark eval rollout, `logged_online_trajectory.py`, or `logged_online_stage4_train.py`.

- [ ] **Step 4: Clean mixed active references**

Update:

- `clstr/qwen_stage4_act_train.py`: Stage2 heads only.
- `clstr/stage2_quality_gate.py`: handoff to Stage4/evaluation.
- `clstr/encoders.py`: remove Stage3 use-case wording.
- `clstr/qwen_full_base_train.py`: stop enumerating HRPO methods as current comparisons.
- `scripts/run_appworld_multistep_executor_eval.py`: remove HRPO checkpoint-name auto-routing.
- `tests/test_appworld_multistep_cli.py` and other mixed tests: remove HRPO-only cases.
- `docs/superpowers/plans/test_patch_tmp.md`: delete the temporary patch artifact.

- [ ] **Step 5: Verify imports and scan GREEN**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_mainline_surface.py tests/test_qwen_external_encoder.py \
  tests/test_v4_action_proj_sharing.py tests/test_phase1_closed_loop_gradient.py \
  tests/test_sbatch_scripts.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
```

Expected: selected tests and compile check pass; deleted-module imports are absent.

- [ ] **Step 6: Commit Task 7**

```bash
git add -A
git commit -m "refactor: remove non-mainline stage3 and rollout code"
```

---

### Task 8: Make Readiness and Documentation Four-Stage Only

**Files:**
- Modify: `scripts/audit_clstr_unified_training_readiness.py`
- Modify: `scripts/sbatch/run_clstr_unified_readiness_audit.sh`
- Modify: `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
- Modify: `tests/test_unified_training_readiness.py`
- Modify: `tests/test_sbatch_scripts.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing four-stage readiness assertions**

Update readiness tests to assert exact active gates:

```python
assert set(report["gates"]) == {
    "stage0_routing",
    "stage1_heads",
    "stage2_full_base",
    "stage4_causal_next_skill",
}
assert "stage3_checkpoint" not in report["provenance"]
```

Add README assertions that the current pipeline contains `Stage0 → Stage1 → Stage2 → Stage4` and does not describe HRPO as a training stage.

- [ ] **Step 2: Run readiness/docs tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py tests/test_mainline_surface.py
```

Expected: current readiness report still contains Stage3 gates or scripts still accept Stage3 paths.

- [ ] **Step 3: Remove Stage3 readiness state**

Delete Stage3 defaults, row counts, quality-gate calls, gates, CLI arguments, and exported environment variables. Rename the Stage4 gate to `stage4_causal_next_skill` and require the new causal objective plus direct post-action row count.

Rewrite the README current-method section around the four-stage causal equations. Explicitly label teacher rollouts as supervised-data extraction and closed-loop rollouts as evaluation.

- [ ] **Step 4: Verify readiness/docs GREEN**

Run the Task 8 command again and run `bash -n` on the two modified Slurm scripts.

Expected: all selected tests pass and shell syntax is valid.

- [ ] **Step 5: Commit Task 8**

```bash
git add scripts/audit_clstr_unified_training_readiness.py \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py \
  tests/test_mainline_surface.py README.md
git commit -m "docs: expose only the four-stage causal mainline"
```

---

### Task 9: Full Verification and Independent Review

**Files:**
- Modify as required by review findings only.

- [ ] **Step 1: Run affected test modules**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_candidate_utils.py \
  tests/test_device_utils.py \
  tests/test_alfworld_eval.py \
  tests/test_mt_ablation_eval.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py \
  tests/test_mainline_surface.py
```

Expected: zero failures.

- [ ] **Step 2: Run retained evaluator and import checks**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_validator.py \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_appworld_clstr_eval.py \
  tests/test_appworld_multistep.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
```

Expected: zero failures and compile exit code 0.

- [ ] **Step 3: Run repository semantic scans**

```bash
rg -n "stage3_unified_hrpo|stage3_joint|run_clstr_.*hrpo|from clstr\.rollout|from clstr\.train" \
  clstr scripts tests README.md
rg -n "unified_route_logits\(h_selected, m_selected|unified_route_logits\(h, m_obs" \
  clstr/full_base_train.py clstr/stage4_act_train.py
git diff --check bfa56f0..HEAD
```

Expected: no active Stage3/HRPO/legacy imports, no old current-state next-skill call sites, and no whitespace errors introduced after the snapshot.

- [ ] **Step 4: Request independent code review**

Provide the reviewer with the approved design, implementation plan, base `c85070e`, current HEAD, and explicit questions about time indexing, gradient paths, missing-row handling, and deletion safety. Resolve all critical and important findings with new failing tests before production changes.

- [ ] **Step 5: Run final verification after review fixes**

Repeat Steps 1–3 after the last review change. Record exact commands, pass counts, and any intentionally excluded historical references in `.planning/2026-07-10-causal-unified-memory-mainline/progress.md`.

- [ ] **Step 6: Commit final review fixes**

```bash
git add -A
git commit -m "test: verify causal unified-memory mainline"
```

Skip this commit only if independent review requires no changes and the worktree is already clean.
