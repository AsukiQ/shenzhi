# Static-Preserving Memory Candidate Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let causal recurrent memory add dynamic-only skills outside the corrected static shortlist while preserving every static Top-M candidate and reporting strict equal-budget rescue metrics.

**Architecture:** A pure selector builds `StaticTopM + DynamicExtraTopD` from full unified static/dynamic logits over one legal pool. A shared current-state evaluator computes both branches once, applies the union, and exposes source-normalized recall records; retained benchmark wrappers declare whether candidate recall is global-applicable, local-saturated, or environment-inapplicable.

**Tech Stack:** Python 3.10+, PyTorch, pytest, JSON/JSONL reports, Bash/Slurm.

---

## Prerequisites

- Start from commit `2426736` or later on `feat/reliability-aware-causal-retrieval`.
- Phase 1 commits `0ad1a8e`, `ee5baca`, `c0b2c9d`, `30798e5`, and `35ec77e` must be ancestors.
- Keep `route_scorer=unified_memory`; candidate recall must not revive the legacy prior/residual scorer.
- Preserve the compatibility mode `stage0_candidates` for historical tests and explicit ablations.

## File structure

- `clstr/memory_candidate_recall.py`: union selection, strict recall accounting, and protocol applicability.
- `clstr/current_state_route_eval.py`: one shared full-pool branch computation and union-ranked current-state evaluation.
- `clstr/logged_online_stage4_train.py`: batch aggregation, progress identity, and argument propagation.
- `clstr/global_pool_route_eval.py`: all-source strict global-pool report and equal-budget control.
- benchmark wrappers: pass protocol metadata and do not implement their own top-k logic.
- `scripts/audit_clstr_unified_training_readiness.py`: reject unsupported candidate-recall claims.

### Task 1: Implement the pure static-plus-dynamic union

**Files:**
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `tests/test_memory_candidate_recall.py`

- [ ] **Step 1: Write failing union and strict-metric tests**

Add tests for static preservation, dynamic-only selection, deterministic ties,
sparse masks, small pools, and exact equal-score equivalence:

```python
def test_static_dynamic_union_preserves_static_and_adds_dynamic_only_candidates():
    static = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    dynamic = torch.tensor([[1.0, 2.0, 3.0, 10.0, 11.0]])
    valid = torch.ones_like(static, dtype=torch.bool)

    union = build_static_dynamic_union(
        static,
        dynamic,
        valid,
        static_k=2,
        dynamic_extra_k=2,
    )

    assert union.static_rows == [[0, 1]]
    assert union.dynamic_top_rows == [[4, 3]]
    assert union.dynamic_extra_rows == [[4, 3]]
    assert union.candidate_rows == [[0, 1, 4, 3]]
    assert union.static_equal_budget_rows == [[0, 1, 2, 3]]
```

Also add:

```python
def test_equal_static_dynamic_scores_reproduce_equal_budget_static_topk_exactly():
    logits = torch.tensor([[4.0, 3.0, 3.0, 3.0, 1.0]])
    valid = torch.tensor([[True, True, False, True, True]])

    union = build_static_dynamic_union(
        logits,
        logits,
        valid,
        static_k=2,
        dynamic_extra_k=2,
    )

    assert union.candidate_rows == union.static_equal_budget_rows
```

Write strict accounting tests where one row is a static miss rescued by a
dynamic extra, one target is absent from the declared table, and one row has no
causal update. Assert the absent target remains in the all-source denominator.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py -k 'union or strict_recall or applicability'
```

Expected: import failures for the new dataclasses/functions.

- [ ] **Step 3: Implement union selection**

Add:

```python
CANDIDATE_UNION_VERSION = "memory_union_v1"
CANDIDATE_RECALL_MODES = {"stage0_candidates", "static_plus_dynamic_extra"}


@dataclass(frozen=True)
class CandidateUnion:
    candidate_rows: list[list[int]]
    static_rows: list[list[int]]
    dynamic_top_rows: list[list[int]]
    dynamic_extra_rows: list[list[int]]
    static_equal_budget_rows: list[list[int]]


def build_static_dynamic_union(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    static_k: int,
    dynamic_extra_k: int,
) -> CandidateUnion:
    if static_logits.ndim != 2 or static_logits.shape != dynamic_logits.shape:
        raise ValueError("static and dynamic logits must have matching rank-2 shapes")
    if static_logits.shape != valid_mask.shape:
        raise ValueError("valid_mask must match branch logits")
    if static_logits.dtype != dynamic_logits.dtype or static_logits.device != dynamic_logits.device:
        raise ValueError("static and dynamic logits must share dtype and device")
    if int(static_k) < 0 or int(dynamic_extra_k) < 0:
        raise ValueError("candidate budgets must be nonnegative")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    static_rows = stable_masked_topk_rows(static_logits, valid, k=int(static_k))
    dynamic_top_rows = stable_masked_topk_rows(dynamic_logits, valid, k=int(dynamic_extra_k))
    static_equal_budget_rows = stable_masked_topk_rows(
        static_logits,
        valid,
        k=int(static_k) + int(dynamic_extra_k),
    )
    extra_valid = valid.clone()
    for row_idx, static_ids in enumerate(static_rows):
        if static_ids:
            ids = torch.tensor(static_ids, dtype=torch.long, device=extra_valid.device)
            extra_valid[row_idx, ids] = False
    dynamic_extra_rows = stable_masked_topk_rows(
        dynamic_logits,
        extra_valid,
        k=int(dynamic_extra_k),
    )
    candidate_rows = [
        static_ids + extra_ids
        for static_ids, extra_ids in zip(static_rows, dynamic_extra_rows)
    ]
    return CandidateUnion(
        candidate_rows=candidate_rows,
        static_rows=static_rows,
        dynamic_top_rows=dynamic_top_rows,
        dynamic_extra_rows=dynamic_extra_rows,
        static_equal_budget_rows=static_equal_budget_rows,
    )
```

- [ ] **Step 4: Implement strict recall accounting and applicability**

Add:

```python
def _candidate_hit_rows(candidate_rows, positive_mask):
    hits = torch.zeros(len(candidate_rows), dtype=torch.bool, device=positive_mask.device)
    for row_idx, indices in enumerate(candidate_rows):
        if indices:
            ids = torch.tensor(indices, dtype=torch.long, device=positive_mask.device)
            hits[row_idx] = bool(positive_mask[row_idx].index_select(0, ids).any())
    return hits


def candidate_recall_report(
    union: CandidateUnion,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> dict[str, Any]:
    if positive_mask.ndim != 2 or int(positive_mask.size(0)) != len(union.candidate_rows):
        raise ValueError("positive_mask must have one row per candidate union")
    if legal_mask.shape != positive_mask.shape:
        raise ValueError("legal_mask must match positive_mask")
    if causal_update_count.ndim != 1 or int(causal_update_count.numel()) != len(union.candidate_rows):
        raise ValueError("causal_update_count must have one value per row")
    known_positive = positive_mask.to(dtype=torch.bool)
    legal = legal_mask.to(device=known_positive.device, dtype=torch.bool)
    positive = known_positive & legal
    counts = causal_update_count.to(device=positive.device, dtype=torch.float32)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")

    static_hit = _candidate_hit_rows(union.static_rows, positive)
    dynamic_top_hit = _candidate_hit_rows(union.dynamic_top_rows, positive)
    dynamic_extra_hit = _candidate_hit_rows(union.dynamic_extra_rows, positive)
    union_hit = _candidate_hit_rows(union.candidate_rows, positive)
    equal_budget_hit = _candidate_hit_rows(union.static_equal_budget_rows, positive)
    overlap_values = []
    dynamic_only_counts = []
    for static_ids, dynamic_ids, extra_ids in zip(
        union.static_rows,
        union.dynamic_top_rows,
        union.dynamic_extra_rows,
    ):
        denominator = max(1, min(len(static_ids), len(dynamic_ids)))
        overlap_values.append(len(set(static_ids).intersection(dynamic_ids)) / denominator)
        dynamic_only_counts.append(len(extra_ids))

    def summarize(mask: torch.Tensor) -> dict[str, float]:
        denom = int(mask.sum().detach().cpu().item())
        selected = lambda values: values[mask]
        static_miss = selected(~static_hit)
        rescue = selected((~static_hit) & union_hit)
        miss_count = int(static_miss.sum().detach().cpu().item())
        return {
            "source_rows": float(denom),
            "static_recall": float(selected(static_hit).float().mean().item()) if denom else 0.0,
            "dynamic_top_recall": float(selected(dynamic_top_hit).float().mean().item()) if denom else 0.0,
            "dynamic_extra_recall": float(selected(dynamic_extra_hit).float().mean().item()) if denom else 0.0,
            "union_recall": float(selected(union_hit).float().mean().item()) if denom else 0.0,
            "static_equal_budget_recall": float(selected(equal_budget_hit).float().mean().item()) if denom else 0.0,
            "static_miss_rows": float(miss_count),
            "dynamic_rescue_rows": float(rescue.sum().detach().cpu().item()),
            "static_miss_recovery_rate": float(rescue.sum().item()) / miss_count if miss_count else 0.0,
            "static_dynamic_overlap": (
                float(sum(value for value, keep in zip(overlap_values, mask.detach().cpu().tolist()) if keep)) / denom
                if denom else 0.0
            ),
            "dynamic_only_candidate_count": (
                float(sum(value for value, keep in zip(dynamic_only_counts, mask.detach().cpu().tolist()) if keep)) / denom
                if denom else 0.0
            ),
        }

    all_rows = torch.ones(len(union.candidate_rows), dtype=torch.bool, device=positive.device)
    memory_active = counts > 0
    return {
        "all_eligible_source_strict": summarize(all_rows),
        "memory_active_strict": summarize(memory_active),
        "memory_active_coverage": float(memory_active.float().mean().item()) if len(union.candidate_rows) else 0.0,
        "positive_not_in_declared_pool_rows": float((~known_positive.any(dim=-1)).sum().item()),
        "positive_outside_legal_pool_rows": float(
            (known_positive.any(dim=-1) & ~positive.any(dim=-1)).sum().item()
        ),
        "candidate_union_version": CANDIDATE_UNION_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "tie_break_policy": TIE_BREAK_POLICY,
    }
```

Add `candidate_recall_applicability(pool_protocol, legal_pool_size, static_k,
dynamic_extra_k, causal_sequential)` returning the exact fields:

```text
candidate_recall_applicable
candidate_recall_applicability_reason
candidate_recall_saturated
```

Use reasons `known_global_sequential`, `legal_pool_fully_enumerated`,
`environment_admissible_actions`, `nonsequential_without_causal_update`, and
`appended_untrained_skill_transfer`.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py
```

Expected: all tests pass.

Commit:

```bash
git add clstr/memory_candidate_recall.py tests/test_memory_candidate_recall.py
git commit -m "feat: add static-preserving dynamic candidate union"
```

### Task 2: Add one shared full-pool current-state route batch

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_logged_online_stage4_train.py`

- [ ] **Step 1: Write failing full-pool evaluator tests**

Extend the recording model with `unified_route_full_logits()` and
`gather_unified_route_logits()`. Add a row whose target is absent from static
Top-M but is selected by dynamic extra. Assert:

```python
loss, metrics = _compute_current_state_route_loss(
    model,
    rows,
    torch.device("cpu"),
    candidate_recall_mode="static_plus_dynamic_extra",
    skill_id_to_idx={"skill/a": 0, "skill/b": 1, "skill/c": 2, "skill/d": 3},
    equivalent_skill_ids_by_skill_id={},
    static_k=1,
    dynamic_extra_k=1,
    final_k=1,
)

assert metrics["candidate_recall_all_union_recall"] == 1.0
assert metrics["candidate_recall_all_static_recall"] == 0.0
assert metrics["candidate_recall_all_dynamic_rescue_rows"] == 1.0
assert metrics["stage4_next_skill_recall@1"] == 1.0
```

Add tests that changing the current row action/observation still does not affect
current-state scores, replay changes dynamic but not static scores, zero replay
forces `causal_update_count=0`, and `stage0_candidates` preserves the old API.

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_current_state_route_eval.py -k 'candidate_recall or full_pool_union'
```

Expected: new keyword arguments and metrics are absent.

- [ ] **Step 3: Implement a reusable batch-output dataclass**

Add:

```python
@dataclass
class CurrentStateRouteBatchOutput:
    loss: torch.Tensor
    metrics: dict[str, Any]
    static_full_logits: torch.Tensor
    dynamic_full_logits: torch.Tensor
    legal_mask: torch.Tensor
    positive_mask: torch.Tensor
    candidate_union: CandidateUnion
    static_memory: torch.Tensor
    dynamic_memory: torch.Tensor
    causal_update_count: torch.Tensor
```

Implement `_build_current_state_route_batch(...)` with the same arguments as the
public loss helper plus candidate budgets. In union mode it must:

1. encode `h_t` once;
2. compute `static_memory = model.initial_belief(h_t)`;
3. replay prefixes canonically into `dynamic_memory`;
4. compute full static/dynamic logits once each;
5. build exact/alias positive masks and one legal mask;
6. build the shared union;
7. gather both branches on the union without recomputing the retriever;
8. rank the dynamic endpoint on the union;
9. count every source row, including unknown/out-of-legal targets, as a strict
   zero in recall metrics.

The existing `_compute_current_state_route_loss()` delegates to this builder and
returns `(output.loss, output.metrics)` so existing callers remain compatible.
Require `final_k <= static_k` whenever exact static fallback is advertised.

- [ ] **Step 4: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_current_state_route_eval.py tests/test_logged_online_stage4_train.py
```

Commit:

```bash
git add clstr/current_state_route_eval.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py
git commit -m "feat: evaluate current-state routes on memory candidate union"
```

### Task 3: Propagate the union through retained logged evaluators

**Files:**
- Modify: `clstr/logged_online_stage4_train.py`
- Modify: `tests/test_logged_online_stage4_train.py`

- [ ] **Step 1: Write failing propagation and resume-identity tests**

Add tests that `build_logged_online_stage4_rows_with_stage0_handoff(...,
next_skill_pool_mode="full_pool")` retains static misses without injection, and
that `evaluate_logged_online_stage4_rows()` forwards the complete mapping,
aliases, M, D, and final K.

Write a progress-resume test showing a batch log created with D=16 is not reused
when D=64.

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_logged_online_stage4_train.py -k 'candidate_union or full_pool_handoff or progress_identity'
```

- [ ] **Step 3: Thread the explicit contract**

Add these keyword arguments to both overall and per-benchmark evaluators:

```python
candidate_recall_mode: str = "stage0_candidates"
skill_id_to_idx: dict[str, int] | None = None
equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None
static_k: int = 500
dynamic_extra_k: int = 64
final_k: int = 64
```

Pass them only to the unified current-state path. Include all six values plus
`CANDIDATE_UNION_VERSION` in progress JSON and completed-batch identity; stale
batch logs must not be reused across configurations. Preserve legacy scorer
behavior unchanged.

For union-mode row construction, call both handoff and Stage4 builders with
`next_skill_pool_mode="full_pool"`; preserve untouched natural static candidates
for diagnostics and forbid positive injection.

- [ ] **Step 4: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_logged_online_stage4_train.py tests/test_current_state_route_eval.py
```

Commit:

```bash
git add clstr/logged_online_stage4_train.py tests/test_logged_online_stage4_train.py
git commit -m "feat: propagate memory candidate union through logged evaluation"
```

### Task 4: Apply strict protocol semantics to benchmark evaluators

**Files:**
- Modify: `clstr/global_pool_route_eval.py`
- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/tau2_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `clstr/bfcl_route_eval.py`
- Modify: `clstr/apibank_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `scripts/run_webshop_clstr_eval.py`
- Test: `tests/test_global_pool_route_eval.py`
- Test: `tests/test_toolbench_full_clstr_route_eval.py`
- Test: `tests/test_tau2_route_eval.py`
- Test: `tests/test_toolsandbox_route_eval.py`
- Test: `tests/test_bfcl_route_eval.py`
- Test: `tests/test_apibank_route_eval.py`
- Test: `tests/test_alfworld_eval.py`
- Test: `tests/test_webshop_eval.py`

- [ ] **Step 1: Write failing protocol tests**

Assert global known-pool sequential reports contain an applicable union and
equal-budget control. Assert local pools with `M + D >= pool_size` contain:

```json
{
  "candidate_recall_applicable": false,
  "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
  "candidate_recall_saturated": true
}
```

Assert ALFWorld/WebShop contain:

```json
{
  "candidate_recall_applicable": false,
  "candidate_recall_applicability_reason": "environment_admissible_actions",
  "candidate_source": "environment_admissible_actions"
}
```

Add a global target-outside-pool test that keeps the row as a strict zero and
adds `target_outside_declared_legal_pool` to blockers.

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py
```

- [ ] **Step 3: Integrate without duplicating selection**

Global and ToolBench known-pool evaluators pass the complete mapping and alias
table to the logged evaluator with:

```text
candidate_recall_mode=static_plus_dynamic_extra
static_k=stage0_top_m
dynamic_extra_k=64
final_k=candidate_count
next_skill_pool_mode=full_pool
```

They report both `all_eligible_source_strict` and `memory_active_strict`, and
retain static misses before evaluation. The primary strict table uses all-source
metrics.

tau2, ToolSandbox, and APIBank use the same helper only when their legal pool is
not saturated; otherwise they report applicability metadata and make no recall
claim. BFCL enables recall only for a predeclared sequential subset with positive
causal update count. ALFWorld/WebShop do not compute a union.

No wrapper may infer a global legal pool from target namespace, benchmark name,
or gold skill.

- [ ] **Step 4: Run GREEN and commit**

Run the Step 2 command again. Expected: all pass.

Commit:

```bash
git add clstr/global_pool_route_eval.py clstr/toolbench_full_clstr_route_eval.py \
  clstr/tau2_route_eval.py clstr/toolsandbox_route_eval.py clstr/bfcl_route_eval.py \
  clstr/apibank_route_eval.py clstr/alfworld_eval.py scripts/run_webshop_clstr_eval.py \
  tests/test_global_pool_route_eval.py tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py
git commit -m "feat: report strict memory candidate recall by protocol"
```

### Task 5: Expose budgets, readiness, and final verification

**Files:**
- Modify: `scripts/run_global_pool_clstr_route_eval.py`
- Modify: active global/ToolBench evaluator Slurm scripts
- Modify: `scripts/audit_clstr_unified_training_readiness.py`
- Modify: `tests/test_unified_training_readiness.py`
- Modify: `tests/test_sbatch_scripts.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing launcher/readiness tests**

Require active applicable launchers to expose `STATIC_K`, `DYNAMIC_EXTRA_K`, and
`FINAL_K`, and require readiness metadata:

```json
{
  "candidate_recall_mode": "static_plus_dynamic_extra",
  "candidate_union_version": "memory_union_v1",
  "equal_budget_comparator_mode": "same_final_scorer_static_top_m_plus_d",
  "requested_dynamic_extra_d": 64,
  "candidate_selection_version": "stable_declared_pool_v1",
  "tie_break_policy": "declared_pool_index_ascending"
}
```

Readiness must reject retained-only denominators, gold injection, target-derived
global masks, missing equal-budget controls, and `final_k > static_k`.

- [ ] **Step 2: Run RED, implement, and run GREEN**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py
```

- [ ] **Step 3: Run candidate-recall source verification**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py \
  tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py \
  tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py

env PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts

git ls-files '*.sh' | xargs bash -n
git diff --check
```

Expected: zero failures and zero syntax/whitespace errors.

- [ ] **Step 4: Commit readiness/documentation**

```bash
git add scripts/run_global_pool_clstr_route_eval.py scripts/sbatch \
  scripts/audit_clstr_unified_training_readiness.py \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py README.md
git commit -m "docs: expose strict memory candidate recall protocol"
```

## Empirical checkpoint

Do not claim the component improves the paper result until a retrained Phase 1
checkpoint passes the predeclared comparison against the same-scorer static
Top(M+D) control. Store source-row digests, checkpoint hashes, M/D/final-K, and
trajectory-clustered bootstrap intervals with every report.
