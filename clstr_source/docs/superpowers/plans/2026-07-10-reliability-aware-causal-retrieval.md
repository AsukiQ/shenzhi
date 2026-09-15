# Reliability-Aware Causal Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CLSTR train and evaluate causal memory as a full-pool retriever, replace the weak positive-logit memory margin with a detached-static listwise counterfactual objective, add static-preserving dynamic candidate recall, and conditionally calibrate a score-level memory utility gate.

**Architecture:** The repaired causal Stage2/Stage4 model produces two score branches at the same post-action state and over the same legal skill pool: a static branch from `InitialBelief(h)` and a dynamic branch from recurrent `m_t`. Training uses full-pool multi-positive next-skill supervision and a low-weight counterfactual utility loss. Inference preserves `TopM(static)` and appends `TopD(dynamic \ static)` before score fusion; a learned scalar gate is added only if a post-fix oracle/learnability audit proves row-level complementarity.

**Tech Stack:** Python 3.11, PyTorch, pytest, JSON/JSONL reports, Bash/Slurm launchers, Git.

---

## Approved Phase 1 execution contract

The user-approved design for the current implementation cycle is
`docs/superpowers/specs/2026-07-10-reliability-aware-causal-retrieval-phase1-design.md`.
This cycle executes only Tasks 1-5 below, in order:

1. expose reusable full-pool unified scores;
2. align Stage0 handoff with `UnifiedRoute(h, InitialBelief(h))`;
3. add the pure listwise counterfactual utility objective;
4. make unified Stage2 full-pool and replace `transition_memory_margin`;
5. make Stage4 full-pool and reuse the identical tensor-level objective.

Tasks 6-10 remain part of the overall method roadmap but are outside this cycle.
In particular, Phase 1 does not enable static-plus-dynamic candidate union, train a
memory utility gate, or launch full training. After Task 5, run the Phase 1 source
verification below and stop for a code/result checkpoint before candidate-recall
work begins.

Phase 1 source verification:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_model_pipeline.py \
  tests/test_counterfactual_ranking.py \
  tests/test_memory_candidate_recall.py \
  tests/test_full_base_train.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage2_quality_gate.py \
  tests/test_stage12_consolidated_train.py \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_current_state_route_eval.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py

env PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts

git ls-files '*.sh' | xargs bash -n
git diff --check master..HEAD
```

Expected: zero test failures, compile exit code 0, shell syntax exit code 0,
and no whitespace errors. Also require:

```bash
! rg -n "transition_memory_margin|memory_margin_weight|memory_margin_loss" \
  clstr scripts tests
```

Historical documentation and archived result files may retain the old term, but
active source, launchers, and tests may not.

## Scope and decisions

This plan contains only the three approved method changes:

1. reliability-aware static/dynamic score backoff;
2. replacement of `transition_memory_margin` with listwise counterfactual utility;
3. memory-conditioned candidate recall.

It does not add Stage3/HRPO, unfreeze the backbone, add a categorical belief, use benchmark IDs in the gate, train on shuffled prefixes, or make the first gate control the candidate budget.

The following distinction must remain explicit in code and paper artifacts:

```text
Correctness repair:
    Stage0 handoff must use UnifiedRoute(h, InitialBelief(h)).

Method contribution:
    causal m_t supplies dynamic-only candidates outside the corrected static shortlist,
    and a reliability controller limits damage in final ranking.
```

## Execution prerequisite

Do not start Task 1 until the existing causal-mainline plan has completed canonical replay initialization, causal Stage2/Stage4 post-action scoring, and current-state evaluator separation.

Required preflight:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py \
  tests/test_stage4_act_train.py \
  tests/test_current_state_route_eval.py

rg -n "def _post_action_memory|def _compute_current_state_route_loss" \
  clstr/full_base_train.py clstr/current_state_route_eval.py
```

Expected:

- all three modules pass;
- `_post_action_memory` exists and Stage2/Stage4 call it before next-skill scoring;
- current-state route evaluation does not call the causal Stage4 training loss;
- static and dynamic comparisons use the same `h_next` and differ only in memory.

If any condition fails, stop and finish `docs/superpowers/plans/2026-07-10-causal-unified-memory-mainline.md` first.

## File structure

### New focused modules

- `clstr/counterfactual_ranking.py`: multi-positive log utility and detached-static gain/safety loss.
- `clstr/memory_candidate_recall.py`: stable declared-pool selection, legal-pool masking, fixed static-plus-dynamic-extra union, strict recall/rescue metrics.
- `clstr/memory_utility_gate.py`: low-dimensional gate features, scalar gate, mask-safe score fusion, oracle diagnostics.
- `clstr/memory_utility_gate_train.py`: conditional trajectory-split, gate-only calibration over frozen branch records; create only if the Task 8 audit passes.
- `scripts/audit_clstr_memory_utility_oracle.py`: preflight oracle and learnability report.
- `scripts/train_clstr_memory_utility_gate.py`: conditional Stage4 gate calibration entrypoint; create only if the Task 8 audit passes.

### Existing modules modified

- `clstr/model.py`: expose reusable full-pool unified logits without recomputing them before candidate gather.
- `clstr/full_base_train.py`: corrected static handoff/cache, full-pool Stage2 supervision, counterfactual loss, candidate-recall records.
- `clstr/stage4_act_train.py`: retain static-miss rows, full-pool causal supervision, shared counterfactual loss.
- `clstr/current_state_route_eval.py`: static/dynamic/union/fused evaluation over the current-state contract.
- `clstr/mt_ablation_eval.py`: strict candidate-rescue and reliability diagnostics.
- global/local route evaluators: use the shared candidate-union and gate helpers.
- active Stage2/Stage4 CLI and Slurm launchers: rename loss arguments and expose fixed candidate budgets.

### New tests

- `tests/test_counterfactual_ranking.py`
- `tests/test_memory_candidate_recall.py`
- `tests/test_memory_utility_gate.py`
- `tests/test_memory_utility_gate_train.py`: conditional; create only if the Task 8 audit passes.

---

### Task 1: Expose Reusable Full-Pool Unified Scores

**Files:**
- Modify: `clstr/model.py:429-462`
- Test: `tests/test_model_pipeline.py:249-280`

- [ ] **Step 1: Write failing full-score reuse tests**

Add tests proving the full logits are computed once and candidate gathering is a pure subset operation:

```python
def test_unified_route_full_logits_and_gather_match_public_subset_api():
    model = _minimal_unified_model()
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    m = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    rows = [[2, 0], [1]]

    full = CLSTRModel.unified_route_full_logits(model, h, m)
    retriever_calls_after_full = model.unified_retriever.calls
    gathered = CLSTRModel.gather_unified_route_logits(model, full, rows)
    public = CLSTRModel.unified_route_logits(model, h, m, candidate_rows=rows)

    assert torch.equal(gathered, public)
    assert model.unified_retriever.calls == retriever_calls_after_full + 1  # public API only
    assert gathered.shape == (2, 2)
    assert torch.isneginf(gathered[1, 1]) or gathered[1, 1] == torch.finfo(gathered.dtype).min
```

Also assert gradients from a gathered candidate reach `unified_retriever`, while padded entries do not contribute. Preserve and test the existing API behavior where one row of `full_logits` may be expanded across multiple `candidate_rows`; gathering an already-computed tensor must not call the retriever.

- [ ] **Step 2: Run the tests and verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_model_pipeline.py -k 'unified_route_full_logits or gather_unified_route'
```

Expected: FAIL because the two focused APIs do not exist.

- [ ] **Step 3: Split full scoring from gathering**

Implement:

```python
def unified_route_full_logits(self, h_t: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
    skill_table = getattr(self, "skill_table", None)
    if skill_table is None or not hasattr(skill_table, "E"):
        raise RuntimeError("unified_route_full_logits requires skill_table.E")
    z_t = self.unified_retriever(h_t, m_t)
    skill_embs = skill_table.E.to(device=z_t.device, dtype=z_t.dtype)
    return z_t @ skill_embs.t()


def gather_unified_route_logits(
    self,
    full_logits: torch.Tensor,
    candidate_rows: list[list[int]],
) -> torch.Tensor:
    if full_logits.ndim != 2:
        raise ValueError("full_logits must have shape [batch, skills]")
    if int(full_logits.size(0)) == 1 and len(candidate_rows) > 1:
        full_logits = full_logits.expand(len(candidate_rows), -1)
    if len(candidate_rows) != int(full_logits.size(0)):
        raise ValueError("candidate_rows batch size must match full_logits")
    width = max((len(row) for row in candidate_rows), default=0)
    if width == 0:
        return full_logits[:, :0]
    ids = torch.zeros(len(candidate_rows), width, dtype=torch.long, device=full_logits.device)
    valid = torch.zeros(len(candidate_rows), width, dtype=torch.bool, device=full_logits.device)
    for row_idx, row in enumerate(candidate_rows):
        if any(int(idx) < 0 or int(idx) >= int(full_logits.size(1)) for idx in row):
            raise ValueError("candidate skill index is outside full_logits")
        if row:
            row_ids = torch.tensor(row, dtype=torch.long, device=full_logits.device)
            ids[row_idx, : len(row)] = row_ids
            valid[row_idx, : len(row)] = True
    return full_logits.gather(1, ids).masked_fill(~valid, torch.finfo(full_logits.dtype).min)
```

Refactor `unified_route_logits()` to call these helpers. The gather helper must not call the retriever or recompute `z @ E.T`.

- [ ] **Step 4: Run targeted and model tests GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_model_pipeline.py tests/test_v4_candidate_sampling.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add clstr/model.py tests/test_model_pipeline.py
git commit -m "refactor: expose reusable unified full-pool logits"
```

---

### Task 2: Align Stage0 Candidate Handoff with the Trained Unified Scorer

**Files:**
- Create: `clstr/memory_candidate_recall.py`
- Modify: `clstr/full_base_train.py:77,744-781,987-1090,4772-4859`
- Create: `scripts/benchmark_memory_candidate_selection.py`
- Create: `tests/test_memory_candidate_recall.py`
- Test: `tests/test_full_base_train.py`
- Test: `tests/test_clstr_topm_candidate_handoff.py`

- [ ] **Step 1: Write failing scorer-alignment tests**

Use a model whose h-only retrieval scorer raises if called:

```python
def test_stage0_handoff_uses_initial_belief_and_unified_full_logits():
    model = _RecordingUnifiedHandoffModel()
    model.skill_table.retrieval_logits = _forbidden_h_only_retrieval

    retained, report = _attach_stage0_topm_candidates(
        model,
        _handoff_rows(),
        _skills(),
        _skill_id_to_idx(),
        top_m=2,
        positive_missing_policy="skip",
        encode_batch_size=16,
        device=torch.device("cpu"),
    )

    assert retained
    assert model.initial_belief_calls == 2  # current and next queries
    assert model.unified_full_calls == 2
    assert report["static_candidate_scorer"] == "unified_static"
```

Add a cache-key test asserting different `initial_belief_top_k`, scorer name, or checkpoint digest produces a different key and that a v1 cache cannot hit.

Add a tied-logit test proving handoff uses declared-pool index order, plus cache tests proving that changing the tie policy, candidate-selection version, or declared-pool order digest causes a miss. A valid NaN or infinity must raise instead of entering candidate selection.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'handoff and unified_static'
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py -k 'cache or scorer'
```

Expected: the h-only scorer is called or required report/cache fields are missing.

- [ ] **Step 3: Implement the canonical static scorer**

Inside handoff batching use:

```python
state_h = _encode(model, state_texts).to(device)
state_m0 = model.initial_belief(state_h)
state_logits = model.unified_route_full_logits(state_h, state_m0)

next_h = _encode(model, next_texts).to(device)
next_m0 = model.initial_belief(next_h)
next_logits = model.unified_route_full_logits(next_h, next_m0)
```

Create one shared tensor-level selector in `memory_candidate_recall.py` and use it inside handoff instead of `torch.topk`:

```python
CANDIDATE_SELECTION_VERSION = "stable_declared_pool_v1"
TIE_BREAK_POLICY = "declared_pool_index_ascending"


def stable_masked_topk_rows(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
) -> list[list[int]]:
    if logits.ndim != 2 or logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must be matching rank-2 tensors")
    if not logits.is_floating_point():
        raise ValueError("candidate logits must use a floating dtype")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("valid candidate logits must be finite")
    limit = max(0, int(k))
    if limit == 0 or int(logits.size(1)) == 0:
        return [[] for _ in range(int(logits.size(0)))]
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    probe_k = min(limit, int(logits.size(1)))
    thresholds = torch.topk(masked, k=probe_k, dim=-1, largest=True, sorted=False).values.min(dim=-1).values
    selected: list[list[int]] = []
    for row_idx in range(int(logits.size(0))):
        row_valid = valid[row_idx]
        effective_k = min(limit, int(row_valid.sum().detach().cpu().item()))
        threshold = thresholds[row_idx]
        strict_ids = (row_valid & (logits[row_idx] > threshold)).nonzero(as_tuple=False).view(-1)
        boundary_ids = (row_valid & (logits[row_idx] == threshold)).nonzero(as_tuple=False).view(-1)
        fill = max(0, effective_k - int(strict_ids.numel()))
        chosen = torch.cat([strict_ids, boundary_ids[:fill]], dim=0)
        # chosen starts in declared-index order; stable sorting keeps that order on ties.
        chosen_scores = logits[row_idx].gather(0, chosen)
        chosen_order = torch.argsort(chosen_scores, descending=True, stable=True)
        chosen = chosen.gather(0, chosen_order)
        selected.append([int(item) for item in chosen.detach().cpu().tolist()])
    return selected
```

This uses device-side `topk` only to find the kth threshold, takes all strictly better skills, fills the boundary tie by declared-pool index, and stable-sorts at most k selected items. It avoids a full 67k-column argsort and transfers only selected indices. The declared pool order, not a registry ID that may be absent in local pools, is the tie-break source.

Replace every `torch.topk` path inside the existing inventory-aware handoff helper, including min-candidate backfill and score-output ordering, with this selector. Add an end-to-end explicit-inventory test whose kth boundary is tied and whose min-candidate backfill is required. On the target GPU, benchmark batch 64, pool 67,557, K=500 and D=64: candidate selection must allocate less than one additional full score matrix, run within 2x unsorted `torch.topk`, and add no more than 20% to measured full-logit-plus-selection latency. If this gate fails, optimize selection before launching full handoff/evaluation.

Fail loudly if `initial_belief` or unified full scoring is absent. Do not fall back to `_skill_logits_and_memory()` in unified handoff mode.

Bump:

```python
STAGE0_HANDOFF_CACHE_VERSION = "stage0_handoff_cache_v2_unified_static"
```

Add these cache-key/report fields:

```text
static_candidate_scorer = unified_static
initial_belief_top_k
route_scorer
state_text_format
skill_text_format
checkpoint_digest
skills_digest
skill_embedding_digest
declared_pool_order_digest
candidate_selection_version = stable_declared_pool_v1
tie_break_policy = declared_pool_index_ascending
unified_static_scorer_digest
```

Thread every field explicitly through `_stage0_handoff_cache_key()` and its only caller; do not append report-only fields after the digest is formed. Define `unified_static_scorer_digest` by hashing the canonical effective in-memory scorer state after every checkpoint/overlay/local-pool rebuild: encoder parameters used by `_encode`, `skill_table.W`, `skill_table.E`, `logit_scale_belief`, `skill_bias_belief`, `initial_belief_head`, and `unified_retriever`, plus their relevant config. Hash ordered checkpoint file contents, not only path/mtime/size, and record the checkpoint chain. Compute the separate `skill_embedding_digest` once from `skill_table.E.detach().cpu().contiguous()`—including tensor shape, dtype, and bytes—for easy pool diagnostics. This prevents a local pool with the same skill IDs but different loaded calibration/heads or rebuilt embeddings from reusing stale rankings. The existing complete row digest, query mode, positive policy, and inventory settings remain.

- [ ] **Step 4: Verify GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_memory_candidate_recall.py
```

Expected: all tests pass and old cache fixtures are explicitly invalidated.

On the target GPU run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/benchmark_memory_candidate_selection.py \
  --batch_size 64 --skill_count 67557 --hidden_size 1024 \
  --static_k 500 --dynamic_extra_k 64 --warmup 10 --repeats 50 \
  --max_topk_latency_ratio 2.0 --max_end_to_end_overhead_fraction 0.20 \
  --max_extra_score_matrix_equivalents 1.0
```

Expected: exit 0 and a JSON report with selector/topk latency, synthetic full-logit-plus-selection overhead, and peak allocated bytes. Do not launch full handoff if any threshold fails.

- [ ] **Step 5: Commit**

```bash
git add clstr/full_base_train.py clstr/memory_candidate_recall.py \
  scripts/benchmark_memory_candidate_selection.py \
  tests/test_full_base_train.py tests/test_clstr_topm_candidate_handoff.py \
  tests/test_memory_candidate_recall.py
git commit -m "fix: align stage0 handoff with unified static retrieval"
```

---

### Task 3: Add the Listwise Counterfactual Utility Loss

**Files:**
- Create: `clstr/counterfactual_ranking.py`
- Create: `tests/test_counterfactual_ranking.py`

- [ ] **Step 1: Write failing loss-contract tests**

Cover weak-static gain, strong-static safety, multi-positive aliases, masks, and gradients:

```python
def test_counterfactual_utility_detects_negative_growth_that_fools_positive_margin():
    static = torch.tensor([[2.0, 2.5, 0.0]], requires_grad=True)
    dynamic = torch.tensor([[2.2, 3.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    valid = torch.ones_like(positives)

    output = counterfactual_utility_loss(
        dynamic_logits=dynamic,
        static_logits=static,
        positive_mask=positives,
        valid_mask=valid,
        gain_margin=0.1,
        safety_tolerance=0.0,
    )

    assert output.gain_loss > 0
    output.loss.backward()
    assert dynamic.grad is not None
    assert static.grad is None
```

Also test:

- a static-correct row that dynamic damages activates safety;
- positive aliases are permutation invariant;
- weak and strong branches are averaged separately;
- no weak/strong eligible rows return a finite scalar zero;
- a row needs at least one valid positive and one valid negative;
- masked extreme values never produce NaN.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_counterfactual_ranking.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the pure loss**

Use a dataclass result and explicit finite zeros:

```python
@dataclass
class CounterfactualUtilityOutput:
    loss: torch.Tensor
    gain_loss: torch.Tensor
    safety_loss: torch.Tensor
    weak_count: int
    strong_count: int
    gain_violation_count: int
    safety_violation_count: int


def multi_positive_log_utility(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != positive_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, positive_mask, and valid_mask must be matching rank-2 tensors")
    if not logits.is_floating_point():
        raise ValueError("utility logits must use a floating dtype")
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool) & valid_mask
    masked_all = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    masked_pos = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    return torch.logsumexp(masked_pos.float(), dim=-1) - torch.logsumexp(masked_all.float(), dim=-1)


def counterfactual_utility_loss(
    *,
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    gain_margin: float,
    safety_tolerance: float,
    gain_weight: float = 1.0,
    safety_weight: float = 1.0,
) -> CounterfactualUtilityOutput:
    if dynamic_logits.ndim != 2 or dynamic_logits.shape != static_logits.shape:
        raise ValueError("dynamic and static logits must have matching rank-2 shapes")
    if dynamic_logits.shape != positive_mask.shape or dynamic_logits.shape != valid_mask.shape:
        raise ValueError("logits and masks must have matching shapes")
    if dynamic_logits.device != static_logits.device or dynamic_logits.dtype != static_logits.dtype:
        raise ValueError("dynamic and static logits must share dtype and device")
    if gain_margin < 0 or safety_tolerance < 0 or gain_weight < 0 or safety_weight < 0:
        raise ValueError("counterfactual margins, tolerance, and branch weights must be nonnegative")
    valid_mask = valid_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    positive_mask = positive_mask.to(device=dynamic_logits.device, dtype=torch.bool) & valid_mask
    static = static_logits.detach()
    u_static = multi_positive_log_utility(static, positive_mask, valid_mask).detach()
    u_dynamic = multi_positive_log_utility(dynamic_logits, positive_mask, valid_mask)
    gain = u_dynamic - u_static
    static_top1 = static.masked_fill(~valid_mask, torch.finfo(static.dtype).min).argmax(dim=-1)
    static_correct = positive_mask.gather(1, static_top1.unsqueeze(1)).squeeze(1)
    has_positive = positive_mask.any(dim=-1)
    has_negative = (valid_mask & ~positive_mask).any(dim=-1)
    finite = torch.where(valid_mask, torch.isfinite(dynamic_logits) & torch.isfinite(static), True).all(dim=-1)
    eligible = has_positive & has_negative & finite
    weak = eligible & ~static_correct
    strong = eligible & static_correct
    gain_terms = F.relu(float(gain_margin) - gain[weak])
    safety_terms = F.relu(-float(safety_tolerance) - gain[strong])
    zero = dynamic_logits.new_zeros(())
    gain_loss = gain_terms.mean() if gain_terms.numel() else zero
    safety_loss = safety_terms.mean() if safety_terms.numel() else zero
    total = float(gain_weight) * gain_loss + float(safety_weight) * safety_loss
    return CounterfactualUtilityOutput(
        loss=total,
        gain_loss=gain_loss,
        safety_loss=safety_loss,
        weak_count=int(weak.sum().detach().cpu().item()),
        strong_count=int(strong.sum().detach().cpu().item()),
        gain_violation_count=int((gain_terms > 0).sum().detach().cpu().item()),
        safety_violation_count=int((safety_terms > 0).sum().detach().cpu().item()),
    )


@dataclass
class FullPoolCausalRouteOutput:
    main_loss: torch.Tensor
    counterfactual: CounterfactualUtilityOutput
    eligible_mask: torch.Tensor
    exclusion_counts: dict[str, int]


def full_pool_causal_route_objective(
    *,
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    row_weights: torch.Tensor,
    gain_margin: float,
    safety_tolerance: float,
    gain_weight: float = 1.0,
    safety_weight: float = 1.0,
) -> FullPoolCausalRouteOutput:
    if dynamic_logits.ndim != 2 or dynamic_logits.shape != static_logits.shape:
        raise ValueError("full-pool branch logits must have matching rank-2 shapes")
    if dynamic_logits.shape != positive_mask.shape or dynamic_logits.shape != valid_mask.shape:
        raise ValueError("full-pool logits and masks must have matching shapes")
    if row_weights.ndim != 1 or int(row_weights.numel()) != int(dynamic_logits.size(0)):
        raise ValueError("row_weights must have one value per row")
    if dynamic_logits.device != static_logits.device or dynamic_logits.dtype != static_logits.dtype:
        raise ValueError("full-pool branch logits must share dtype and device")

    valid = valid_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    known_positive = positive_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    legal_positive = known_positive & valid
    has_legal_positive = legal_positive.any(dim=-1)
    has_legal_negative = (valid & ~legal_positive).any(dim=-1)
    finite = torch.where(
        valid,
        torch.isfinite(dynamic_logits) & torch.isfinite(static_logits),
        True,
    ).all(dim=-1)
    eligible = has_legal_positive & has_legal_negative & finite
    weights = row_weights.to(device=dynamic_logits.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("row_weights must be finite and nonnegative")

    zero = dynamic_logits.new_zeros(())
    if bool(eligible.any()):
        eligible_weights = weights[eligible]
        if not bool((eligible_weights.sum() > 0).item()):
            raise ValueError("eligible full-pool rows have zero total weight")
        row_nll = -multi_positive_log_utility(
            dynamic_logits[eligible],
            legal_positive[eligible],
            valid[eligible],
        )
        main_loss = (row_nll * eligible_weights).sum() / eligible_weights.sum()
    else:
        main_loss = zero
    counterfactual = counterfactual_utility_loss(
        dynamic_logits=dynamic_logits[eligible],
        static_logits=static_logits[eligible],
        positive_mask=legal_positive[eligible],
        valid_mask=valid[eligible],
        gain_margin=gain_margin,
        safety_tolerance=safety_tolerance,
        gain_weight=gain_weight,
        safety_weight=safety_weight,
    )
    return FullPoolCausalRouteOutput(
        main_loss=main_loss,
        counterfactual=counterfactual,
        eligible_mask=eligible,
        exclusion_counts={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "no_valid_positive_rows": int((~has_legal_positive).sum().detach().cpu().item()),
            "no_valid_negative_rows": int((~has_legal_negative).sum().detach().cpu().item()),
            "nonfinite_logit_rows": int((~finite).sum().detach().cpu().item()),
        },
    )
```

Never use `logits.sum() * 0` on tensors containing padded minimum values. Build zero with `dynamic_logits.new_zeros(())`.

- [ ] **Step 4: Run GREEN and gradient checks**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_counterfactual_ranking.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add clstr/counterfactual_ranking.py tests/test_counterfactual_ranking.py
git commit -m "feat: add listwise counterfactual memory utility loss"
```

---

### Task 4: Make Stage2 Full-Pool and Replace Memory Margin

**Files:**
- Modify: `clstr/counterfactual_ranking.py`
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `clstr/full_base_train.py:35-41,3830-3858,3903-4456,4560-4588,5010-5310`
- Modify: `scripts/run_clstr_stage2_full_base_train.py:78-90,214-255`
- Modify: `scripts/run_clstr_stage12_consolidated_train.py:129-190,270-335,419-555`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Modify: `scripts/sbatch/run_clstr_unified_stage12_consolidated_function_aug_v2.sh`
- Test: `tests/test_memory_candidate_recall.py`
- Test: `tests/test_full_base_train.py`
- Test: `tests/test_clstr_topm_candidate_handoff.py`
- Test: `tests/test_stage12_consolidated_train.py`
- Test: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Write failing Stage2 tests**

Add a causal row whose gold next skill is outside static top-M and assert it remains trainable:

```python
def test_stage2_full_pool_keeps_causal_static_miss_and_backpropagates():
    row = _causal_stage2_row(
        next_skill_id="skill/gold",
        stage0_next_candidate_skill_indices=[0, 1],
        full_pool_gold_index=4,
    )

    loss, metrics = _compute_full_base_loss(
        model,
        adapter,
        [row],
        _five_skill_mapping(),
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        loss_weights={
            key: (1.0 if key == "L_trans_skill_ce" else 0.05 if key == "counterfactual_utility" else 0.0)
            for key in LOSS_WEIGHT_KEYS
        },
        counterfactual_gain_margin=0.1,
        counterfactual_safety_tolerance=0.01,
        counterfactual_gain_weight=1.0,
        counterfactual_safety_weight=1.0,
        counterfactual_scale=1.0,
    )

    loss.backward()
    assert metrics["transition_full_pool_rows"] == 1.0
    assert metrics["transition_static_miss_train_rows"] == 1.0
    assert model.transition.weight.grad is not None
```

Add tests for:

- aliases form a multi-positive full-pool mask;
- explicit legal inventory masks both branches identically;
- target-derived root/benchmark masks are rejected in global protocol mode;
- `replay_prefix=[]` is still counterfactual-eligible after one post-action update;
- static logits receive no gradient through the counterfactual term;
- the old `transition_memory_margin` metric/config keys disappear.
- full-pool mode keeps the row when either current or next target is absent from static top-M, without injecting either target into the recorded natural shortlist.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py -k 'full_pool or counterfactual or static_miss'
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py -k 'static_miss'
```

Expected: rows are masked/dropped or new arguments/metrics are absent.

- [ ] **Step 3: Replace candidate-limited CE in unified Stage2**

Add arguments:

```text
generic helper default: next_skill_pool_mode = stage0_candidates
main unified Stage2 CLI default: next_skill_pool_mode = full_pool
loss_weights[counterfactual_utility] = 0.05
counterfactual_gain_margin = 0.1
counterfactual_safety_tolerance = 0.01
counterfactual_gain_weight = 1.0
counterfactual_safety_weight = 1.0
counterfactual_warmup_fraction = 0.1
```

`full_pool` is valid only with `route_scorer=unified_memory`. Add `explicit_only` to `TRANSITION_INVENTORY_MASK_MODES`, CLI choices, reports, and tests. In full-pool mode it calls only the shared explicit-inventory mask builder and bypasses `_filter_transition_candidates_by_inventory`; reject target/root-derived modes such as `root_namespace`, `stage0_topk_trajectory_prior`, and legacy `auto`. The unified CLI defaults to `transition_inventory_mask_mode=explicit_only`. Generic compatibility callers retain `stage0_candidates` unless they opt in explicitly. Validate every enum and require `counterfactual_warmup_fraction` in `[0, 1]`.

For `unified_memory + full_pool`, after causal `_post_action_memory`:

```python
causal_rows = trans_skill_rows  # exactly the rows represented by h_next/m_next
dynamic_full = model.unified_route_full_logits(h_next, m_next)
with torch.no_grad():
    static_full = model.unified_route_full_logits(h_next, model.initial_belief(h_next))
legal_pool = legal_skill_pool_mask(
    causal_rows,
    skill_id_to_idx,
    skill_count=skill_count,
    device=device,
)
valid_mask = legal_pool.mask
positive_mask = full_pool_positive_mask(
    causal_rows,
    skill_id_to_idx,
    equivalent_skill_ids_by_skill_id,
    skill_count=skill_count,
    device=device,
)

row_weights = _transition_ce_row_weights(
    causal_rows,
    real_multiplier=transition_real_candidate_ce_multiplier,
    injected_multiplier=transition_injected_candidate_ce_multiplier,
    device=device,
    dtype=dynamic_full.dtype,
)
objective = full_pool_causal_route_objective(
    dynamic_logits=dynamic_full,
    static_logits=static_full,
    positive_mask=positive_mask,
    valid_mask=valid_mask,
    row_weights=row_weights,
    gain_margin=counterfactual_gain_margin,
    safety_tolerance=counterfactual_safety_tolerance,
    gain_weight=counterfactual_gain_weight,
    safety_weight=counterfactual_safety_weight,
)
main_loss = objective.main_loss
cf = objective.counterfactual
eligible = objective.eligible_mask
schema_exclusions = {
    "positive_not_in_skill_table_rows": int((~positive_mask.any(dim=-1)).sum().detach().cpu().item()),
    "positive_outside_legal_pool_rows": int(
        (positive_mask.any(dim=-1) & ~(positive_mask & valid_mask).any(dim=-1)).sum().detach().cpu().item()
    ),
    "explicit_inventory_no_known_skill_rows": int(
        legal_pool.explicit_inventory_no_known_skill_mask.sum().detach().cpu().item()
    ),
}
```

Put this reduction in one reusable `full_pool_causal_route_objective()` in `counterfactual_ranking.py`, returning a dataclass with `main_loss`, `counterfactual`, `eligible_mask`, and named exclusion counts. Its inputs are branch logits, masks, per-row weights, margins, and gain/safety weights; it has no model or row-schema dependency. Stage2 and Stage4 must call this function so their raw objective geometry cannot drift.

Do not construct `list(range(skill_count))` per row. Build a tensor mask by scatter or ragged positive indices.

Training eligibility depends on causal fields, target membership in the declared skill table/legal pool, at least one legal negative, and finite branch logits—not static top-M hit. Record named counts for `positive_not_in_skill_table`, `positive_outside_legal_pool`, `explicit_inventory_no_known_skill`, `no_legal_negative`, and `nonfinite_logits`; never average a sentinel utility from an ineligible row. Other losses that genuinely require a candidate shortlist receive their own mask. A batch with no full-pool-eligible row returns an explicit finite zero for these two terms while retaining other valid objectives.

Build `causal_rows` from every row passing the causal-field/loss mask before checking target membership or Stage0 hits. Schema-specific mask builders/outside code record the named reasons; `full_pool_causal_route_objective()` reports only generic `no_valid_positive`, `no_valid_negative`, and `nonfinite_logits`. This keeps the pure tensor helper schema-free while ensuring unknown-target rows are observable rather than prefiltered away.

Thread `next_skill_pool_mode` into Stage0 handoff/cache semantics. In `full_pool` mode:

```text
current positive missing from static top-M:
    keep the row;
    mask only candidate-limited current-routing losses;
    do not append the current skill into either natural shortlist;
    record stage0_current_positive_hit=false.

next positive missing from static top-M:
    keep L_trans_skill_ce enabled;
    do not inject the target into natural candidates;
    record stage0_next_positive_hit=false.
```

The handoff never injects a current or next gold, never copies the current gold into the next shortlist, and never drops the whole row because a current-routing target missed. `stage0_current_positive_hit` and `stage0_next_positive_hit` are computed on the untouched natural lists before any loss-specific masking.

In `stage0_candidates` compatibility mode, preserve the existing skip/mask policy. Add `next_skill_pool_mode` to the handoff cache key so retained-row sets cannot cross modes.

Add the two shared mask builders in `memory_candidate_recall.py`; Stage2 and Stage4 import them rather than importing training internals from one another:

```python
def full_pool_positive_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    *,
    skill_count: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(len(rows), skill_count, dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        target = str(row.get("next_skill_id") or "")
        equivalents = equivalent_skill_ids_by_skill_id or {}
        positive_ids = {target, *equivalents.get(target, [])}
        indices = [skill_id_to_idx[item] for item in positive_ids if item in skill_id_to_idx]
        if indices:
            mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


@dataclass
class LegalSkillPoolMaskOutput:
    mask: torch.Tensor
    explicit_inventory_no_known_skill_mask: torch.Tensor


def legal_skill_pool_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    skill_count: int,
    device: torch.device,
) -> LegalSkillPoolMaskOutput:
    mask = torch.ones(len(rows), skill_count, dtype=torch.bool, device=device)
    no_known = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        explicit = explicit_inventory_skill_ids_ordered(row)
        if explicit:
            mask[row_idx].zero_()
            indices = [skill_id_to_idx[item] for item in explicit if item in skill_id_to_idx]
            if indices:
                mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
            else:
                no_known[row_idx] = True
    return LegalSkillPoolMaskOutput(mask=mask, explicit_inventory_no_known_skill_mask=no_known)
```

The supplied skill table itself defines the global or local pool. With no explicit inventory, every table column is legal. With a nonempty explicit inventory that maps to no table skill, the row remains all-false and `explicit_inventory_no_known_skill_mask` records the schema-specific reason. The caller combines this audit with `positive_mask.any()` and `(positive_mask & legal_pool.mask).any()` to distinguish unknown-table targets from known-but-illegal targets. The helper must never infer a mask from target/root namespace.

Move the existing inventory-value flattening contract into this shared module as `explicit_inventory_skill_ids_ordered()`; preserve the existing `EXPLICIT_INVENTORY_SKILL_ID_KEYS` order and first-occurrence deduplication. Unit-test nested strings/dicts/lists and an empty inventory before replacing the private Stage2 call sites.

Replace `transition_memory_margin` in `LOSS_WEIGHT_KEYS`, reports, CLI, checkpoint metadata, quality/readiness checks, standalone Stage2 launcher, and consolidated Stage1+Stage2 launcher with `counterfactual_utility`. Remove the deprecated CLI flags entirely; do not accept aliases that silently keep the old loss runnable. Stage1 sets the new weight to zero.

Stage2 has one weight source: `loss_weights["counterfactual_utility"]`. Margin, tolerance, branch weights, and a loop-computed scale are direct parameters, but no second direct outer weight exists. Define warmup as a linear ramp during the first `fraction * max_steps` optimizer steps, with no extra zero plateau:

```python
warmup_steps = float(counterfactual_warmup_fraction) * float(max_steps)
counterfactual_scale = 1.0 if warmup_steps <= 0 else min(1.0, float(global_step) / warmup_steps)
weighted_cf = (
    loss_weights["counterfactual_utility"]
    * counterfactual_scale
    * cf.loss
)
```

Use the restored optimizer `global_step` on resume so the schedule does not restart. Tests cover fraction 0, midpoint, completion, 1, and resumed steps.

- [ ] **Step 4: Run Stage2 GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_counterfactual_ranking.py \
  tests/test_memory_candidate_recall.py \
  tests/test_full_base_train.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage2_quality_gate.py \
  tests/test_stage12_consolidated_train.py \
  tests/test_sbatch_scripts.py
```

Expected: all tests pass; reports separately count real, injected, static-hit, and static-miss causal rows.

Run `rg -l "transition_memory_margin|memory_margin_weight|memory_margin_loss" clstr scripts tests`. Expected: no active source, launcher, or test references; historical docs/results may still contain the names.

- [ ] **Step 5: Commit**

```bash
git add clstr/counterfactual_ranking.py clstr/memory_candidate_recall.py clstr/full_base_train.py \
  scripts/run_clstr_stage2_full_base_train.py \
  scripts/run_clstr_stage12_consolidated_train.py \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage12_consolidated_function_aug_v2.sh \
  tests/test_full_base_train.py tests/test_clstr_topm_candidate_handoff.py \
  tests/test_memory_candidate_recall.py tests/test_stage2_quality_gate.py \
  tests/test_stage12_consolidated_train.py tests/test_sbatch_scripts.py
git commit -m "feat: train causal stage2 routing over the full skill pool"
```

---

### Task 5: Make Stage4 Full-Pool and Reuse the Counterfactual Loss

**Files:**
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `clstr/stage4_act_train.py:290-425,738-875,1020-1420`
- Modify: `scripts/run_clstr_stage4_act_train.py:96-118,165-225`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- Modify: `clstr/stage4_quality_gate.py`
- Test: `tests/test_stage4_act_train.py`
- Test: `tests/test_stage4_quality_gate.py`

- [ ] **Step 1: Write failing Stage4 static-miss and parity tests**

```python
def test_stage4_builder_retains_static_miss_for_full_pool_training():
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        [_causal_source_row(stage0_candidates=["skill/a"], next_skill="skill/gold")],
        {"skill/a": 0, "skill/gold": 1},
        stage0_candidate_handoff_report={"enabled": True},
        next_skill_pool_mode="full_pool",
    )

    assert len(rows) == 1
    assert rows[0]["stage0_next_positive_hit"] is False
    assert rows[0].get("positive_next_skill_position") is None
    assert report["stage0_next_static_miss_retained_rows"] == 1
```

Repeat the test with an empty natural Stage0 shortlist. Add an end-to-end handoff-to-builder test proving full-pool mode cannot drop the row before the builder. Add a parity test feeding identical branch logits/masks to the shared objective and asserting equal raw full-pool NLL/counterfactual outputs from Stage2 and Stage4; do not compare totals with different outer weights.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_act_train.py -k 'static_miss or counterfactual or full_pool'
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_quality_gate.py -k 'counterfactual or full_pool'
```

Expected: the builder skips the row or Stage4 still exposes memory-margin fields.

- [ ] **Step 3: Integrate full-pool Stage4**

For full-pool mode:

- retain causally valid rows regardless of Stage0 hit;
- preserve the untouched Stage0 candidates/scores as diagnostics, allowing the list to be empty;
- set `positive_next_skill_position=None` when the natural shortlist misses; call `_pad_candidate_indices()` only in `stage0_candidates` compatibility mode;
- compute `h_next`, `m_next`, `static_full`, and `dynamic_full` after the shared post-action helper;
- use the same positive/legal masks and `counterfactual_utility_loss()` as Stage2;
- remove `memory_margin_weight` / `memory_margin` and all corresponding metrics;
- report the causal objective as `full_pool_causal_next_skill_with_counterfactual_utility`.

Expose the same Stage2 arguments in Stage4:

```text
next_skill_pool_mode=full_pool
counterfactual_utility_weight=0.05
counterfactual_gain_margin=0.1
counterfactual_safety_tolerance=0.01
counterfactual_gain_weight=1.0
counterfactual_safety_weight=1.0
counterfactual_warmup_fraction=0.05
```

Stage4 preserves the canonical handoff fields `stage0_current_positive_hit` and `stage0_next_positive_hit`; do not introduce a second `static_candidate_hit` alias. It receives the complete `skill_id_to_idx` and equivalent-ID mapping from the runner's loaded skill table; never reconstruct them from shortlists. Validate that mapping indices are contiguous, their declared skill order matches the model table, and `len(skill_id_to_idx) == model.skill_table.E.size(0)`. `full_pool` with any non-unified route scorer fails loudly.

Compute the static branch under `torch.no_grad()` and call the same shared eligible-row/main-NLL/counterfactual helper as Stage2, passing `torch.ones(len(causal_rows), device=device)` as Stage4 row weights because full-pool Stage4 forbids injection. Stage4 keeps its direct outer `counterfactual_utility_weight`; its fresh-training loop ramps by local `step/max_steps` over the first configured fraction. The current Stage4 trainer has no resume contract, so this plan does not claim resume-safe Stage4 warmup; an interrupted run restarts Stage4 and records that restart. Tests cover local step 0, midpoint, and ramp completion. The Stage4 quality gate requires finite full-pool main loss, fields for both gain/safety counts, total counterfactual-eligible rows greater than zero, static-hit/miss counts, and no gold injection in evaluation records. Either weak or strong count may legitimately be zero.

- [ ] **Step 4: Run GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_act_train.py tests/test_stage4_quality_gate.py \
  tests/test_counterfactual_ranking.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add clstr/memory_candidate_recall.py clstr/stage4_act_train.py clstr/stage4_quality_gate.py \
  scripts/run_clstr_stage4_act_train.py \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  tests/test_stage4_act_train.py tests/test_stage4_quality_gate.py
git commit -m "feat: train causal stage4 routing over the full skill pool"
```

---

### Task 6: Add Fixed Static-Plus-Dynamic Candidate Recall

**Files:**
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `tests/test_memory_candidate_recall.py`
- Modify: `clstr/mt_ablation_eval.py`
- Test: `tests/test_mt_ablation_eval.py`

- [ ] **Step 1: Write failing fixed-union tests**

```python
def test_static_dynamic_union_preserves_static_and_adds_dynamic_only_candidates():
    static = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    dynamic = torch.tensor([[1.0, 2.0, 3.0, 10.0, 11.0]])
    valid = torch.ones_like(static, dtype=torch.bool)

    union = build_static_dynamic_union(static, dynamic, valid, static_k=2, dynamic_extra_k=2)

    assert union.candidate_rows == [[0, 1, 4, 3]]
    assert union.static_rows == [[0, 1]]
    assert union.dynamic_extra_rows == [[4, 3]]
```

Also test:

- no duplicate candidates;
- deterministic tie breaking by global skill index;
- invalid pool entries never appear;
- `dynamic == static` gives exact static `Top(M+D)`;
- here `dynamic == static` means `m_dynamic = InitialBelief(h_next)` at the same evaluated `h_next`, not reuse of the trajectory-start memory;
- tied logits plus a sparse invalid mask still give row-by-row equality with stable static `Top(M+D)`;
- union contains every static TopM item;
- small pool returns fewer than M+D without padding fake skills;
- raw dynamic TopD and static equal-budget Top(M+D) are retained separately;
- exact and alias-aware rescue metrics use all eligible source rows as denominator.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py
```

Expected: import failure.

- [ ] **Step 3: Implement the fixed union and strict metrics**

Define:

```python
@dataclass
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
    if static_logits.shape != dynamic_logits.shape or static_logits.shape != valid_mask.shape:
        raise ValueError("static, dynamic, and valid tensors must have matching shapes")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    static_rows = stable_masked_topk_rows(static_logits, valid, k=static_k)
    dynamic_top_rows = stable_masked_topk_rows(dynamic_logits, valid, k=dynamic_extra_k)
    static_equal_budget_rows = stable_masked_topk_rows(
        static_logits,
        valid,
        k=max(0, int(static_k)) + max(0, int(dynamic_extra_k)),
    )
    extra_valid = valid.clone()
    for row_idx, static_ids in enumerate(static_rows):
        if static_ids:
            ids = torch.tensor(static_ids, dtype=torch.long, device=extra_valid.device)
            extra_valid[row_idx, ids] = False
    dynamic_extra_rows = stable_masked_topk_rows(
        dynamic_logits,
        extra_valid,
        k=dynamic_extra_k,
    )
    union_rows = [static_ids + extra_ids for static_ids, extra_ids in zip(static_rows, dynamic_extra_rows)]
    return CandidateUnion(
        candidate_rows=union_rows,
        static_rows=static_rows,
        dynamic_top_rows=dynamic_top_rows,
        dynamic_extra_rows=dynamic_extra_rows,
        static_equal_budget_rows=static_equal_budget_rows,
    )
```

Select dynamic extras only after masking every static candidate. Keep fixed M+D budget when the legal pool is large enough. The shared stable tensor selector performs sorting on device; only the final M/D indices cross to CPU. Assert in tests that `dynamic_logits is static_logits` produces `candidate_rows == static_equal_budget_rows` exactly, including ties and invalid entries.

Add metrics:

```text
static_recall@M
dynamic_top_recall@D
dynamic_extra_recall@D
union_recall@(M+D)
static_equal_budget_recall@(M+D)
static_miss_rows
dynamic_rescue_rows
static_miss_recovery_rate
raw_static_dynamic_overlap
dynamic_only_candidate_count
gold_static_rank
gold_dynamic_rank
```

Report exact and alias-aware variants, benchmark/prefix/known-vs-appended slices, latency, and peak memory.

Training never persists static or dynamic candidate indices because sequential heads change every optimizer step. Cache only frozen text embeddings and frozen skill embeddings. Checkpoint evaluation may cache unions only in `model.eval()` mode and only when the key includes an `effective_model_state_digest`, ordered Stage0/Stage2/Stage4 checkpoint-overlay chain, row/prefix digest, legal-pool embedding digest, declared-pool order digest, static K, dynamic D, mask mode, union algorithm version, tie policy, current-state versus post-action-next-state contract, causal replay semantic version, and query format. The effective digest is computed after all partial loads/local rebuilds from every in-memory parameter used by encoding, initial belief, transition/correction, and static/dynamic routing; it is not the digest of a single checkpoint path. Closed-loop evaluation recomputes the dynamic branch every step.

- [ ] **Step 4: Run GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py tests/test_mt_ablation_eval.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add clstr/memory_candidate_recall.py clstr/mt_ablation_eval.py \
  tests/test_memory_candidate_recall.py tests/test_mt_ablation_eval.py
git commit -m "feat: add static-preserving dynamic candidate recall"
```

---

### Task 7: Use the Same Union Contract in Retained Evaluators

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/global_pool_route_eval.py`
- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/tau2_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `clstr/bfcl_route_eval.py`
- Modify: `clstr/apibank_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `scripts/run_webshop_clstr_eval.py`
- Test: corresponding evaluator tests

- [ ] **Step 1: Write failing evaluator-contract tests**

For global-pool evaluation assert static misses remain in the strict denominator and may be rescued by dynamic extras. For benchmark-local pools where `M+D >= pool_size`, assert union recall equals one and no candidate-recall gain is claimed. For ALFWorld/WebShop assert report value:

```json
{
  "candidate_recall_applicable": false,
  "candidate_source": "environment_admissible_actions"
}
```

Saturated local-pool reports must additionally contain:

```json
{
  "candidate_recall_applicable": false,
  "candidate_recall_applicability_reason": "legal_pool_fully_enumerated",
  "candidate_recall_saturated": true
}
```

Add tests that a target absent from the declared legal pool is reported as a protocol blocker and a strict zero, never silently removed from the denominator.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_current_state_route_eval.py \
  tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py \
  tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py
```

Expected: reports lack fixed-union metrics or still exclude static misses.

- [ ] **Step 3: Integrate shared recall selection**

Every skill-pool evaluator must:

1. declare one legal pool independent of the target;
2. compute static and dynamic full logits over that pool;
3. build `C_static + C_extra` using the shared helper;
4. gather both score branches on the same union;
5. report strict source-normalized metrics with missing targets as zero.

Every global evaluation emits two predeclared denominator families: `all_eligible_source_strict_*` includes zero-update and memory-active rows, while `memory_active_strict_*` restricts to `causal_update_count > 0` and reports `memory_active_coverage`. The paper's primary routing table and Task 10 acceptance use all-source strict metrics; memory-active metrics are mechanism analysis and cannot replace a negative all-source result. Candidate-recall mechanism claims are enabled only for known global-pool sequential rows with active memory. The predeclared primary evidence is ToolBench-G3 clean67k known-pool and TrajectBench known-pool after registry membership and canonical replay are verified. tau2, ToolSandbox, and APIBank benchmark-local runs are rerank/gate diagnostics; ALFWorld/WebShop are not applicable because the environment supplies actions. BFCL is auxiliary global evidence only for subsets with a real sequential causal update.

Default first experiment:

```text
static_k = existing Stage0 M
dynamic_extra_k = 64
final_k = existing downstream K
```

Do not use adaptive D. Do not use target namespace/root to infer the global legal pool. Appended unseen skills must be a separate transfer slice. Validate `final_k <= static_k` whenever an exact alpha-zero static fallback is claimed; for smaller pools compare against the effective legal-pool size. Fail loudly on an incompatible configuration instead of claiming endpoint equivalence.

- [ ] **Step 4: Run evaluator GREEN**

Run the command from Step 2 again.

Expected: all modules pass; static-only and `dynamic_full == static_full` union-equivalence tests pass. Learned/fixed-alpha fusion is not required until Task 8.

- [ ] **Step 5: Commit**

```bash
git add clstr/current_state_route_eval.py clstr/global_pool_route_eval.py \
  clstr/toolbench_full_clstr_route_eval.py clstr/tau2_route_eval.py \
  clstr/toolsandbox_route_eval.py clstr/bfcl_route_eval.py \
  clstr/apibank_route_eval.py clstr/alfworld_eval.py \
  scripts/run_webshop_clstr_eval.py tests/test_current_state_route_eval.py \
  tests/test_global_pool_route_eval.py tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py tests/test_webshop_eval.py
git commit -m "feat: evaluate memory-conditioned candidate recall strictly"
```

---

### Task 8: Implement the Gate Oracle and Learnability Audit

**Files:**
- Create: `clstr/memory_utility_gate.py`
- Create: `scripts/audit_clstr_memory_utility_oracle.py`
- Create: `tests/test_memory_utility_gate.py`
- Modify: `clstr/mt_ablation_eval.py`

- [ ] **Step 1: Write failing feature, fusion, and oracle tests**

```python
def test_score_fusion_has_exact_static_and_dynamic_endpoints():
    static = torch.tensor([[3.0, 2.0, float("-inf")]])
    dynamic = torch.tensor([[1.0, 4.0, float("-inf")]])
    valid = torch.tensor([[True, True, False]])
    floor = torch.finfo(static.dtype).min
    expected_static = static.masked_fill(~valid, floor)
    expected_dynamic = dynamic.masked_fill(~valid, floor)

    assert torch.equal(fuse_route_scores(static, dynamic, torch.tensor([0.0]), valid), expected_static)
    assert torch.equal(fuse_route_scores(static, dynamic, torch.tensor([1.0]), valid), expected_dynamic)
```

Add tests that fusion never evaluates `-inf - -inf`, rejects invalid shapes/devices/dtypes/alpha values and nonfinite valid logits, does not mutate dynamic memory, excludes benchmark/skill IDs from features, forces alpha zero when `causal_update_count=0`, and computes rank/utility oracle metrics on identical candidates.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py -k 'fusion or feature or oracle'
```

Expected: module/functions absent.

- [ ] **Step 3: Implement audit-only gate utilities**

Implement mask-safe score fusion:

```python
def fuse_route_scores(static, dynamic, alpha, valid_mask):
    if static.ndim != 2 or static.shape != dynamic.shape or static.shape != valid_mask.shape:
        raise ValueError("static, dynamic, and valid_mask must have matching rank-2 shapes")
    if not static.is_floating_point() or not dynamic.is_floating_point() or not alpha.is_floating_point():
        raise ValueError("route scores and alpha must use floating dtypes")
    if static.dtype != dynamic.dtype or static.device != dynamic.device:
        raise ValueError("static and dynamic scores must share dtype and device")
    if alpha.ndim != 1 or int(alpha.numel()) != int(static.size(0)):
        raise ValueError("alpha must have one scalar per row")
    if alpha.device != static.device:
        raise ValueError("alpha and route scores must share a device")
    if not torch.isfinite(alpha).all() or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha must be finite and in [0, 1]")
    output = torch.full_like(static, torch.finfo(static.dtype).min)
    valid = valid_mask.to(device=static.device, dtype=torch.bool)
    if not torch.isfinite(static[valid]).all() or not torch.isfinite(dynamic[valid]).all():
        raise ValueError("valid route scores must be finite")
    a = alpha.to(device=static.device, dtype=static.dtype).view(-1, 1).expand_as(static)
    static_valid = static[valid]
    dynamic_valid = dynamic[valid]
    alpha_valid = a[valid]
    interior = torch.lerp(static_valid, dynamic_valid, alpha_valid)
    output[valid] = torch.where(
        alpha_valid == 0,
        static_valid,
        torch.where(alpha_valid == 1, dynamic_valid, interior),
    )
    return output


def effective_memory_alpha(raw_alpha: torch.Tensor, causal_update_count: torch.Tensor) -> torch.Tensor:
    if raw_alpha.shape != causal_update_count.shape:
        raise ValueError("raw_alpha and causal_update_count must have matching shapes")
    counts = causal_update_count.to(raw_alpha.device)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")
    return torch.where(counts > 0, raw_alpha, torch.zeros_like(raw_alpha))
```

Extract only low-dimensional, inference-available features:

```text
static/dynamic normalized entropy
static/dynamic top1-top2 standardized gap
top1 agreement
JS divergence
mean absolute standardized-logit difference
1 - cosine(m_static, m_dynamic)
relative memory L2 difference
log-normalized causal update count
log-normalized valid candidate count
```

All returned feature tensors are detached. Define finite edge behavior in tests: entropy and gap are zero with one candidate; standardized gaps/differences are zero at zero variance; JS uses only valid probabilities; zero-norm memory produces finite cosine/L2 features; train-derived count caps clamp update/candidate counts at inference. `causal_update_count=0` is passed through `effective_memory_alpha()` and forces alpha zero in audit, training, and inference.

The audit script uses trajectory-disjoint train/dev splits and reports:

```text
static endpoint, dynamic endpoint, best fixed alpha
rank oracle and utility oracle
static/dynamic/tie fractions
entropy/margin heuristic
linear held-out winner AUROC or utility regret
leave-one-benchmark-out results
paired-bootstrap confidence intervals
```

Endpoint/oracle metrics use all source rows and score union misses as zero. Gate utility/learnability records require `causal_update_count > 0`, at least one valid positive, and one valid negative; report `source_rows`, `union_positive_hit_rows`, `memory_active_rows`, `gate_eligible_rows`, and every exclusion reason. Zero-history rows keep exact static fallback in endpoint metrics but never enter gate fitting or feature-normalization statistics. Bootstrap resamples whole trajectories/tasks, never individual correlated steps.

- [ ] **Step 4: Add the explicit decision gate**

The audit report sets `learned_gate_recommended=true` only when all hold:

```text
both static-better and dynamic-better >= 5% of non-tie rows
each winner type has >= 20 held-out rows total and >= 5 rows in each of at least two sequential benchmarks
rank oracle headroom over best endpoint >= 0.01 macro MRR
oracle headroom over best fixed alpha >= 0.005 macro MRR
held-out linear AUROC >= 0.60 or utility regret improves >= 10% over the constant predictor
```

Before winner labels are computed, persist an audit manifest listing the sequential benchmark set. A benchmark qualifies only under the predeclared rule of at least 50 memory-active gate-eligible rows from at least 10 trajectories; require at least four qualifying benchmarks or set `learned_gate_recommended=false`. Freeze this set before inspecting utilities, report every excluded benchmark as an audit blocker, and compute macro MRR as its unweighted mean. `best fixed alpha` is one global value selected from `0.05 * j for j in range(21)` on train/dev trajectories, never separately tuned per benchmark. No test-set rows participate in thresholds, alpha selection, or feature normalization. Report trajectory-clustered paired-bootstrap confidence intervals for headroom and regret.

- [ ] **Step 5: Run GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py tests/test_mt_ablation_eval.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add clstr/memory_utility_gate.py clstr/mt_ablation_eval.py \
  scripts/audit_clstr_memory_utility_oracle.py \
  tests/test_memory_utility_gate.py tests/test_mt_ablation_eval.py
git commit -m "feat: audit memory utility gate headroom"
```

- [ ] **Step 7: Run the post-fix oracle audit and stop if it fails**

Run the audit on held-out trajectory splits for at least ToolBench, tau2, TrajectBench, ToolSandbox, APIBank, and BFCL. Store the command, checkpoint hashes, row digests, and report path in the experiment ledger.

Expected to continue: `learned_gate_recommended=true`. If false, do not execute Task 9; keep only static/dynamic endpoints, fixed-alpha, and heuristic diagnostics.

---

### Task 9: Conditionally Train the Memory Utility Gate

**Condition:** Execute only when Task 8 recommends a learned gate.

**Files:**
- Modify: `clstr/memory_utility_gate.py`
- Create: `clstr/memory_utility_gate_train.py`
- Create: `scripts/train_clstr_memory_utility_gate.py`
- Create: `tests/test_memory_utility_gate_train.py`
- Modify: `clstr/model.py`
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/global_pool_route_eval.py`
- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/tau2_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `clstr/bfcl_route_eval.py`
- Modify: `clstr/apibank_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `scripts/run_webshop_clstr_eval.py`
- Test: `tests/test_current_state_route_eval.py`
- Test: `tests/test_global_pool_route_eval.py`
- Test: `tests/test_toolbench_full_clstr_route_eval.py`
- Test: `tests/test_tau2_route_eval.py`
- Test: `tests/test_toolsandbox_route_eval.py`
- Test: `tests/test_bfcl_route_eval.py`
- Test: `tests/test_apibank_route_eval.py`
- Test: `tests/test_alfworld_eval.py`
- Test: `tests/test_webshop_eval.py`

- [ ] **Step 1: Write failing gate-module and gradient-isolation tests**

```python
def test_gate_calibration_backward_updates_only_memory_utility_gate():
    model = _causal_model_with_utility_gate()
    batch = _gate_records_with_static_and_dynamic_winners()
    batch.features.requires_grad_(True)
    batch.static_logits.requires_grad_(True)
    batch.dynamic_logits.requires_grad_(True)

    loss, metrics = compute_memory_utility_gate_loss(model.memory_utility_gate, batch)
    loss.backward()

    assert any(p.grad is not None for p in model.memory_utility_gate.parameters())
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("memory_utility_gate."))
    assert batch.features.grad is None
    assert batch.static_logits.grad is None
    assert batch.dynamic_logits.grad is None
    assert model.memory_utility_gate is not model.gate
    assert not ({id(p) for p in model.memory_utility_gate.parameters()} & {id(p) for p in model.gate.parameters()})
    assert metrics["gate_static_winner_rows"] > 0
    assert metrics["gate_dynamic_winner_rows"] > 0
```

Add synthetic tests showing alpha rises for predictable dynamic winners and falls for predictable static winners, plus checkpoint missing/enabled behavior.

- [ ] **Step 2: Run RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate_train.py
```

Expected: module/training functions absent.

- [ ] **Step 3: Implement the small scalar gate**

```python
class MemoryUtilityGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 16, initial_alpha: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        probability = min(max(float(initial_alpha), 1.0e-4), 1.0 - 1.0e-4)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(probability / (1.0 - probability)))

    @torch.no_grad()
    def set_feature_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError("gate normalization tensors have the wrong feature shape")
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std).clamp_min(1.0e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1.0e-6)
        return torch.sigmoid(self.net(normalized)).squeeze(-1)
```

The gate training records contain detached features, static/dynamic logits, valid mask, positive mask, causal update count, trajectory ID, and benchmark. Split by trajectory before normalization. Compute feature mean/std and count-feature caps only from train-split rows that have dynamic history and pass gate eligibility, call `set_feature_normalization()`, and save the buffers/caps in the checkpoint so every evaluator uses the identical transform.

Train only the gate with:

```text
L_gate = multi_positive_NLL(fused_scores)
       + beta * utility_magnitude_weighted_BCE(alpha, dynamic_is_better)
```

Implement the batch loss without sending gradients into either route branch:

```python
@dataclass
class GateBatch:
    features: torch.Tensor
    static_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    valid_mask: torch.Tensor
    positive_mask: torch.Tensor
    causal_update_count: torch.Tensor
    utility_tie_tolerance: float = 0.01


def gate_metrics(alpha: torch.Tensor, delta: torch.Tensor, non_tie: torch.Tensor) -> dict[str, float]:
    return {
        "gate_alpha_mean": float(alpha.detach().mean().cpu().item()),
        "gate_static_winner_rows": float(((delta < 0) & non_tie).sum().detach().cpu().item()),
        "gate_dynamic_winner_rows": float(((delta > 0) & non_tie).sum().detach().cpu().item()),
        "gate_tie_rows": float((~non_tie).sum().detach().cpu().item()),
    }


def compute_memory_utility_gate_loss(gate: MemoryUtilityGate, batch: GateBatch, beta: float = 0.2):
    if batch.static_logits.ndim != 2 or batch.static_logits.shape != batch.dynamic_logits.shape:
        raise ValueError("gate branch logits must have matching rank-2 shapes")
    if batch.static_logits.shape != batch.valid_mask.shape or batch.static_logits.shape != batch.positive_mask.shape:
        raise ValueError("gate logits and masks must have matching shapes")
    if batch.static_logits.device != batch.dynamic_logits.device or batch.static_logits.dtype != batch.dynamic_logits.dtype:
        raise ValueError("gate branch logits must share dtype and device")
    score_device = batch.static_logits.device
    valid_all = batch.valid_mask.to(device=score_device, dtype=torch.bool)
    positives_all = batch.positive_mask.to(device=score_device, dtype=torch.bool) & valid_all
    update_count_all = batch.causal_update_count.detach().to(score_device)
    if update_count_all.ndim != 1 or int(update_count_all.numel()) != int(valid_all.size(0)):
        raise ValueError("causal_update_count must have one value per row")
    if not torch.isfinite(update_count_all).all() or bool((update_count_all < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")
    has_history = update_count_all > 0
    has_positive = positives_all.any(dim=-1)
    has_negative = (valid_all & ~positives_all).any(dim=-1)
    finite = torch.where(
        valid_all,
        torch.isfinite(batch.static_logits) & torch.isfinite(batch.dynamic_logits),
        True,
    ).all(dim=-1)
    eligible = has_positive & has_negative & finite & has_history
    if not bool(eligible.any()):
        raise ValueError("gate batch has no memory-active row with finite positive and negative candidates")

    features = batch.features.detach().to(score_device)[eligible]
    static = batch.static_logits.detach()[eligible]
    dynamic = batch.dynamic_logits.detach()[eligible]
    valid = valid_all[eligible]
    positives = positives_all[eligible]
    update_count = update_count_all[eligible]
    raw_alpha = gate(features)
    alpha = effective_memory_alpha(raw_alpha, update_count)
    fused = fuse_route_scores(static, dynamic, alpha, valid)
    fused_nll = -multi_positive_log_utility(fused, positives, valid).mean()

    u_static = multi_positive_log_utility(static, positives, valid).detach()
    u_dynamic = multi_positive_log_utility(dynamic, positives, valid).detach()
    delta = u_dynamic - u_static
    non_tie = delta.abs() > float(batch.utility_tie_tolerance)
    if non_tie.any():
        targets = (delta[non_tie] > 0).to(alpha.dtype)
        magnitudes = delta[non_tie].abs()
        cap = torch.quantile(magnitudes.detach().float(), 0.95).to(magnitudes.dtype)
        weights = magnitudes.clamp_max(cap.clamp_min(1.0e-6))
        positive_fraction = targets.mean().clamp(1.0e-3, 1.0 - 1.0e-3)
        class_balance = torch.where(
            targets > 0,
            0.5 / positive_fraction,
            0.5 / (1.0 - positive_fraction),
        )
        weights = weights * class_balance
        weights = weights / weights.mean().clamp_min(1.0e-6)
        selector = F.binary_cross_entropy(alpha[non_tie], targets, weight=weights)
    else:
        selector = alpha.new_zeros(())
    metrics = gate_metrics(alpha, delta, non_tie)
    metrics.update({
        "gate_source_rows": float(valid_all.size(0)),
        "gate_eligible_rows": float(eligible.sum().detach().cpu().item()),
        "gate_missing_positive_rows": float((~has_positive).sum().detach().cpu().item()),
        "gate_missing_negative_rows": float((has_positive & ~has_negative).sum().detach().cpu().item()),
        "gate_nonfinite_rows": float((~finite).sum().detach().cpu().item()),
        "gate_no_dynamic_history_rows": float((~has_history).sum().detach().cpu().item()),
    })
    return fused_nll + float(beta) * selector, metrics
```

Exclude near-tie rows from the BCE term; they remain in fused NLL. Winsorize utility magnitude at the batch 95th percentile and apply class-balance weights as above. Stratify batches by benchmark and winner class. Early-stop on unweighted held-out macro MRR/NLL. If the complete training split contains only one winner class, fail the calibration report before batching and select the best endpoint.

- [ ] **Step 4: Integrate checkpoint and inference semantics**

- Add `memory_utility_gate` to model/checkpoint metadata only when trained.
- `gate_enabled=true` with missing parameters fails loudly.
- `gate_enabled=false` means the ungated dynamic endpoint (`alpha=1`) on memory-active rows and exact static on zero-history rows; static-only and fixed-alpha are separate explicit ablation modes.
- A low alpha changes only the current route scores; it never overwrites or resets recurrent `m_t`.
- `causal_update_count=0` always yields effective alpha zero and masked-static scores.
- Feature normalization statistics and count caps are checkpointed from train only.
- Learned inference computes both branches in the same candidate-union order and applies the same legal mask.

Add checkpoint round-trip tests for one global-pool evaluator, one benchmark-local evaluator, and one environment-candidate evaluator; all must reproduce the same normalized features and effective alpha as the calibration module.

- [ ] **Step 5: Run GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py \
  tests/test_memory_utility_gate_train.py \
  tests/test_model_pipeline.py \
  tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py \
  tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py \
  tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py \
  tests/test_webshop_eval.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py \
  clstr/model.py clstr/current_state_route_eval.py clstr/global_pool_route_eval.py \
  clstr/toolbench_full_clstr_route_eval.py clstr/tau2_route_eval.py \
  clstr/toolsandbox_route_eval.py clstr/bfcl_route_eval.py \
  clstr/apibank_route_eval.py clstr/alfworld_eval.py \
  scripts/run_webshop_clstr_eval.py scripts/train_clstr_memory_utility_gate.py \
  tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py \
  tests/test_model_pipeline.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py tests/test_bfcl_route_eval.py \
  tests/test_apibank_route_eval.py tests/test_alfworld_eval.py \
  tests/test_webshop_eval.py
git commit -m "feat: calibrate reliability-aware memory score fusion"
```

---

### Task 10: Run Controlled Ablations and Final Verification

**Files:**
- Modify: `scripts/audit_clstr_unified_training_readiness.py`
- Modify: `README.md`
- Modify: `finalwork/table.md` only after new runs complete
- Modify: active Stage2/Stage4 and evaluator Slurm scripts
- Test: `tests/test_unified_training_readiness.py`
- Test: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Add readiness/report tests**

Require reports to identify:

```json
{
  "static_candidate_scorer": "unified_static",
  "next_skill_pool_mode": "full_pool",
  "candidate_recall_mode": "static_plus_dynamic_extra",
  "pool_protocol": "known_global|benchmark_local|environment_candidates",
  "candidate_recall_applicable": true,
  "candidate_recall_applicability_reason": "known_global_sequential",
  "candidate_recall_saturated": false,
  "requested_static_k": 500,
  "effective_static_k": 500,
  "requested_dynamic_extra_d": 64,
  "effective_dynamic_extra_d": 64,
  "requested_final_k": 64,
  "effective_final_k": 64,
  "candidate_selection_version": "stable_declared_pool_v1",
  "tie_break_policy": "declared_pool_index_ascending",
  "union_cache_schema_version": "memory_union_v1",
  "equal_budget_comparator_mode": "same_final_scorer_static_top_m_plus_d",
  "counterfactual_utility_enabled": true,
  "memory_utility_gate_mode": "disabled|fixed_alpha|learned"
}
```

The readiness gate rejects h-only handoff caches, retained-only denominators, gold-injected evaluation, target-derived global masks, and learned gate checkpoints without a passing oracle report.

- [ ] **Step 2: Run RED, implement metadata/readiness, then GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py
```

Expected before implementation: failures for missing metadata. Expected after implementation: all pass.

- [ ] **Step 3: Run the predeclared ablation matrix**

Use at least three seeds for trainable variants:

```text
A. corrected unified-static only
B. corrected static candidates + causal dynamic rerank, no counterfactual loss
C. B + listwise counterfactual gain only
D. B + listwise counterfactual gain and safety
E. dynamic-only full retrieval
F. static Top(M+D) equal-budget candidate source, ranked by the same final scorer as G
G. fixed static TopM + dynamic-extra TopD, dynamic endpoint (alpha=1), no learned gate
H. G + best fixed alpha
I. G + entropy/margin heuristic
J. G + learned gate, only if Task 8 passed
K. G with dynamic memory set to InitialBelief(h_next) at the same evaluated h_next
```

For the paired F/G recall comparison, hold legal mask, final scorer/alpha, final K, and source rows fixed; only candidate IDs differ. F uses stable static `Top(M+D)` and G uses stable static `TopM + dynamic-extra TopD`. Do not compare static-scored F against dynamic/fused-scored G. In K, directly reuse `dynamic_full = static_full` in eval mode and assert candidate rows, fused logits, and ranks match the static control under ties and sparse masks.

Sweep `D in {16, 32, 64}` only after the default D=64 run is valid. Adaptive D is not part of this plan.

- [ ] **Step 4: Apply method acceptance criteria**

Counterfactual loss enters the final method only if:

- held-out dynamic MRR improves by at least about 0.005 over causal CE baseline;
- static-correct to dynamic-wrong flips do not increase;
- no principal benchmark drops by more than 0.01 MRR.

Dynamic recall enters the main claim only if, on known global-pool sequential tasks:

- all thresholds below pass on `all_eligible_source_strict_*`; `memory_active_strict_*` is supporting mechanism analysis only;
- fixed union beats the same-scorer static `Top(M+D)` control with a positive trajectory/task-clustered paired-bootstrap 95% confidence interval;
- strict Recall@K improves by about one point or at least 5% of static misses are rescued;
- strict MRR and R@1 do not decline;
- the effect reproduces on at least two sequential domains.

The first three comparisons use F, the equal-budget same-scorer control. Also retain a separate safety comparison against A, the corrected unified-static endpoint, so a candidate-source gain cannot hide an overall routing regression.

Learned gate enters the final method only if:

- it beats static, dynamic, best fixed alpha, and heuristic on held-out macro MRR;
- it improves at least 0.005 macro MRR over best fixed alpha;
- it recovers at least 25% of oracle headroom;
- it reduces dynamic-worsened rows by at least 20% while retaining at least 80% of dynamic-improved rows;
- no main benchmark is more than 0.005 MRR below static.
- the observed dynamic-win rate in the highest-alpha quartile exceeds the lowest-alpha quartile by at least 10 percentage points.

If a component fails its criterion, disable it in the final checkpoint and report it honestly as an ablation.

- [ ] **Step 5: Run full source verification**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_model_pipeline.py \
  tests/test_counterfactual_ranking.py \
  tests/test_memory_candidate_recall.py \
  tests/test_memory_utility_gate.py \
  tests/test_full_base_train.py \
  tests/test_stage4_act_train.py \
  tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py \
  tests/test_global_pool_route_eval.py \
  tests/test_toolbench_full_clstr_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_toolsandbox_route_eval.py \
  tests/test_bfcl_route_eval.py \
  tests/test_apibank_route_eval.py \
  tests/test_alfworld_eval.py \
  tests/test_webshop_eval.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py

# Run only when Task 8's persisted report recommends and Task 9 implements the learned gate.
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate_train.py \
  tests/test_model_pipeline.py \
  tests/test_current_state_route_eval.py \
  tests/test_global_pool_route_eval.py \
  tests/test_tau2_route_eval.py \
  tests/test_alfworld_eval.py

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts

git diff --check
```

Also run `bash -n` on every modified Slurm script.

If Task 8 reports `learned_gate_recommended=false`, skip the conditional command, require `memory_utility_gate_mode=disabled|fixed_alpha`, and verify readiness rejects any learned-gate checkpoint. The always-run audit/fusion tests in `tests/test_memory_utility_gate.py` must still pass.

- [ ] **Step 6: Request independent review**

Require separate reviewers for:

1. spec compliance and strict-denominator correctness;
2. code quality/gradient isolation;
3. experiment protocol and leakage risks.

Resolve all Critical/Important findings and rerun Step 5.

- [ ] **Step 7: Commit documentation/readiness**

```bash
git add scripts/audit_clstr_unified_training_readiness.py README.md \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py \
  scripts/sbatch
git commit -m "docs: define reliability-aware causal retrieval protocol"
```

Update `finalwork/table.md` in a separate commit only after reports exist and paths/hashes have been verified.

---

## Retraining order

```text
1. Finish and verify causal mainline fixes.
2. Rebuild Stage0 unified-static handoff/cache.
3. Train Stage2 with full-pool causal CE and counterfactual utility.
4. Train Stage4 from the matching Stage2 checkpoint.
5. Evaluate fixed static-plus-dynamic-extra recall and equal-budget controls.
6. Run gate oracle/learnability audit.
7. Train gate only if the audit passes.
8. Run strict benchmark matrix and paired-bootstrap analysis.
```

Old full3000 checkpoints remain historical diagnostics. They cannot be labeled as trained with this plan because both the causal loss graph and candidate-training denominator change.
