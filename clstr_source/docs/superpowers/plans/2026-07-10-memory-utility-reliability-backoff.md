# Memory Utility Reliability Backoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact, auditable static/dynamic score backoff and a trajectory-disjoint utility audit, enabling a learned scalar gate only when held-out evidence passes the predeclared recommendation threshold.

**Architecture:** Reliability operates on static and dynamic scores over the already-fixed candidate union and never changes recurrent memory. Pure utilities provide exact endpoints, fixed-alpha/heuristic baselines, features, and oracles; evaluators persist detached records for an audit that decides whether the optional gate-training task is allowed.

**Tech Stack:** Python 3.10+, PyTorch, pytest, JSON/JSONL reports, Bash/Slurm.

---

## Prerequisites

- Complete `docs/superpowers/plans/2026-07-10-static-preserving-memory-candidate-recall.md` first.
- Reliability comparisons must use identical candidate order, legal mask, state,
  and scorer parameters for static and dynamic branches.
- The learned-gate task is conditional. Implementing fixed/heuristic backoff and
  the oracle audit does not authorize creating or enabling a learned checkpoint.

## File structure

- `clstr/memory_utility_gate.py`: mask-safe fusion, feature extraction, endpoint,
  heuristic, and oracle utilities.
- `clstr/current_state_route_eval.py`: applies reliability modes to the shared
  candidate union and optionally emits detached audit records.
- `clstr/logged_online_stage4_train.py`: aggregates reliability metrics and writes
  route-record JSONL.
- `scripts/audit_clstr_memory_utility_oracle.py`: trajectory-disjoint fixed-alpha,
  heuristic, linear learnability, macro-MRR, and clustered-bootstrap audit.
- `clstr/memory_utility_gate_train.py`: optional gate-only calibration, created
  only after a passing audit.

### Task 1: Implement exact score fusion and inference-safe features

**Files:**
- Create: `clstr/memory_utility_gate.py`
- Create: `tests/test_memory_utility_gate.py`

- [ ] **Step 1: Write failing fusion tests**

```python
def test_score_fusion_has_exact_static_and_dynamic_endpoints():
    static = torch.tensor([[3.0, 2.0, float("-inf")]])
    dynamic = torch.tensor([[1.0, 4.0, float("-inf")]])
    valid = torch.tensor([[True, True, False]])
    floor = torch.finfo(static.dtype).min

    static_expected = static.masked_fill(~valid, floor)
    dynamic_expected = dynamic.masked_fill(~valid, floor)

    assert torch.equal(
        fuse_route_scores(static, dynamic, torch.tensor([0.0]), valid),
        static_expected,
    )
    assert torch.equal(
        fuse_route_scores(static, dynamic, torch.tensor([1.0]), valid),
        dynamic_expected,
    )
```

Add tests for shape/dtype/device validation, alpha outside `[0, 1]`, nonfinite
valid logits, padded invalid values, and no input mutation.

Add:

```python
def test_zero_history_forces_exact_static_alpha():
    raw = torch.tensor([0.9, 0.2])
    updates = torch.tensor([0.0, 3.0])
    assert torch.equal(effective_memory_alpha(raw, updates), torch.tensor([0.0, 0.2]))
```

- [ ] **Step 2: Write failing feature tests**

Construct one-candidate, tied-logit, zero-variance, zero-memory-norm, and sparse
mask cases. Assert a finite detached `[batch, 11]` tensor, zero entropy/gap for a
single valid candidate, and no benchmark/skill identity arguments.

- [ ] **Step 3: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py -k 'fusion or effective_alpha or feature'
```

Expected: module import failure.

- [ ] **Step 4: Implement exact fusion**

```python
RELIABILITY_MODES = {"static", "dynamic", "fixed_alpha", "heuristic", "learned"}
MEMORY_UTILITY_FEATURE_NAMES = (
    "static_normalized_entropy",
    "dynamic_normalized_entropy",
    "static_standardized_gap",
    "dynamic_standardized_gap",
    "top1_agreement",
    "js_divergence",
    "mean_abs_standardized_logit_difference",
    "memory_cosine_distance",
    "memory_relative_l2_difference",
    "log_normalized_causal_update_count",
    "log_normalized_valid_candidate_count",
)


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
    valid = valid_mask.to(device=static.device, dtype=torch.bool)
    if not torch.isfinite(static[valid]).all() or not torch.isfinite(dynamic[valid]).all():
        raise ValueError("valid route scores must be finite")
    output = torch.full_like(static, torch.finfo(static.dtype).min)
    expanded = alpha.to(dtype=static.dtype).view(-1, 1).expand_as(static)
    static_valid = static[valid]
    dynamic_valid = dynamic[valid]
    alpha_valid = expanded[valid]
    interior = torch.lerp(static_valid, dynamic_valid, alpha_valid)
    output[valid] = torch.where(
        alpha_valid == 0,
        static_valid,
        torch.where(alpha_valid == 1, dynamic_valid, interior),
    )
    return output


def effective_memory_alpha(raw_alpha, causal_update_count):
    if raw_alpha.shape != causal_update_count.shape:
        raise ValueError("raw_alpha and causal_update_count must have matching shapes")
    counts = causal_update_count.to(device=raw_alpha.device, dtype=raw_alpha.dtype)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")
    return torch.where(counts > 0, raw_alpha, torch.zeros_like(raw_alpha))
```

- [ ] **Step 5: Implement features and heuristic alpha**

Implement `memory_utility_features(...)` with branch logits, valid mask, static
and dynamic memories, causal update count, candidate-count cap, and update-count
cap. Standardize valid logits per row with variance clamped to `1e-6`; compute
entropy/JS only on valid softmax probabilities; clamp count features by the
train-derived caps. Return `.detach()`.

Implement a deterministic baseline:

```python
def heuristic_memory_alpha(features: torch.Tensor) -> torch.Tensor:
    js = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("js_divergence")]
    agreement = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("top1_agreement")]
    dynamic_gap = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("dynamic_standardized_gap")]
    static_gap = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("static_standardized_gap")]
    raw = torch.sigmoid(dynamic_gap - static_gap - js + agreement - 0.5)
    return raw.detach()
```

This is a baseline, not a trained gate. Its exact formula and version must be
persisted in reports.

- [ ] **Step 6: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py
```

Commit:

```bash
git add clstr/memory_utility_gate.py tests/test_memory_utility_gate.py
git commit -m "feat: add exact memory reliability score fusion"
```

### Task 2: Integrate static, dynamic, fixed, and heuristic modes

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/logged_online_stage4_train.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_logged_online_stage4_train.py`

- [ ] **Step 1: Write failing endpoint and no-memory-mutation tests**

For the same candidate union assert:

```python
static_output = _build_current_state_route_batch(..., reliability_mode="static")
dynamic_output = _build_current_state_route_batch(..., reliability_mode="dynamic")
fixed_output = _build_current_state_route_batch(..., reliability_mode="fixed_alpha", fixed_alpha=0.25)

assert torch.equal(static_output.fused_candidate_logits, static_output.static_candidate_logits)
assert torch.equal(dynamic_output.fused_candidate_logits, dynamic_output.dynamic_candidate_logits)
assert fixed_output.metrics["memory_utility_alpha_mean"] == pytest.approx(0.25)
assert torch.equal(static_output.dynamic_memory, dynamic_output.dynamic_memory)
```

Add a zero-history row and assert every non-static mode has effective alpha zero.
Add a learned-mode-without-checkpoint test that raises clearly.

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_current_state_route_eval.py tests/test_logged_online_stage4_train.py \
  -k 'reliability or fixed_alpha or heuristic or zero_history'
```

- [ ] **Step 3: Extend the shared batch output and evaluator APIs**

Add to `CurrentStateRouteBatchOutput`:

```python
static_candidate_logits: torch.Tensor
dynamic_candidate_logits: torch.Tensor
fused_candidate_logits: torch.Tensor
candidate_valid_mask: torch.Tensor
candidate_positive_mask: torch.Tensor
features: torch.Tensor
raw_alpha: torch.Tensor
effective_alpha: torch.Tensor
```

Add evaluator arguments:

```python
reliability_mode: str = "dynamic"
fixed_alpha: float = 1.0
memory_utility_gate: torch.nn.Module | None = None
feature_update_count_cap: float = 1.0
feature_candidate_count_cap: float = 1.0
```

Mode behavior:

```text
static      -> raw alpha 0
dynamic     -> raw alpha 1
fixed_alpha -> one validated scalar for all rows
heuristic   -> heuristic_memory_alpha(features)
learned     -> memory_utility_gate(features), require a supplied trained gate
```

Always pass raw alpha through `effective_memory_alpha()`. Rank and loss metrics
use fused scores; static/dynamic endpoint metrics remain separately reported.
Never assign to or interpolate `dynamic_memory`.

Thread the same arguments through logged overall/by-benchmark evaluators and
include them in progress-resume identity.

- [ ] **Step 4: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py
```

Commit:

```bash
git add clstr/current_state_route_eval.py clstr/logged_online_stage4_train.py \
  tests/test_current_state_route_eval.py tests/test_logged_online_stage4_train.py
git commit -m "feat: apply reliability backoff on shared route scores"
```

### Task 3: Persist route records and implement the oracle audit

**Files:**
- Modify: `clstr/memory_utility_gate.py`
- Modify: `clstr/logged_online_stage4_train.py`
- Create: `scripts/audit_clstr_memory_utility_oracle.py`
- Modify: `tests/test_memory_utility_gate.py`
- Create: `tests/test_memory_utility_oracle_audit.py`

- [ ] **Step 1: Write failing record/audit tests**

Create synthetic trajectory records with four qualifying benchmarks, both
winner types, and deterministic utilities. Assert:

- no trajectory appears in both train and dev;
- one global fixed alpha is selected from `{0.00, 0.05, ..., 1.00}`;
- benchmark qualification is frozen before winner statistics;
- macro MRR is an unweighted benchmark mean;
- bootstrap resamples trajectory IDs rather than rows;
- zero-history and missing-positive rows are counted but excluded from gate fit;
- recommendation becomes false when any threshold fails.

Add a logged-evaluator test that writes one JSONL record per source row, including
strict-zero rows.

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_oracle_audit.py \
  tests/test_logged_online_stage4_train.py -k 'route_record or oracle_audit'
```

- [ ] **Step 3: Define the persisted record schema**

When `route_records_path` is supplied, write detached JSONL rows:

```json
{
  "schema_version": "memory_utility_route_record_v1",
  "trajectory_id": "...",
  "task_id": "...",
  "benchmark": "...",
  "causal_update_count": 2,
  "features": [0.0],
  "static_logits": [0.0],
  "dynamic_logits": [0.0],
  "valid_mask": [true],
  "positive_mask": [false],
  "static_rank": 2,
  "dynamic_rank": 1,
  "static_utility": -1.2,
  "dynamic_utility": -0.4
}
```

Actual vector lengths are determined by the candidate union and feature schema.
Include candidate union/version, pool protocol, M/D/final-K, row digest, and
effective model/checkpoint-chain digest in a companion manifest. Refuse to append
records with a different manifest identity.

- [ ] **Step 4: Implement audit functions**

In `memory_utility_gate.py` add pure helpers for positive rank, multi-positive
utility, fixed-alpha sweep, rank oracle, utility oracle, and trajectory-clustered
bootstrap.

The CLI reads one or more manifests/JSONL files, freezes benchmarks satisfying
at least 50 memory-active eligible rows from 10 trajectories, requires at least
four benchmarks, splits by trajectory, and fits a detached linear logistic model
with PyTorch on normalized train features. No test rows participate in feature
normalization, fixed-alpha selection, or thresholds.

Persist:

```text
static/dynamic/best-fixed/heuristic macro MRR
rank and utility oracle macro MRR
winner/tie fractions and counts
held-out linear AUROC
constant and linear utility regret
leave-one-benchmark-out results
trajectory-bootstrap confidence intervals
every exclusion reason
learned_gate_recommended
```

Set `learned_gate_recommended=true` only when every threshold in the Phase 2
design passes:

```text
at least four qualifying benchmarks
static-better and dynamic-better each >= 5% of non-tie rows
each winner type >= 20 held-out rows total
each winner type >= 5 rows in at least two sequential benchmarks
rank-oracle headroom over best endpoint >= 0.01 macro MRR
rank-oracle headroom over best fixed alpha >= 0.005 macro MRR
held-out linear AUROC >= 0.60
    or utility regret improves >= 10% over the constant predictor
```

Never weaken a threshold because the current data are small.

- [ ] **Step 5: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py tests/test_memory_utility_oracle_audit.py \
  tests/test_logged_online_stage4_train.py
```

Commit:

```bash
git add clstr/memory_utility_gate.py clstr/logged_online_stage4_train.py \
  scripts/audit_clstr_memory_utility_oracle.py \
  tests/test_memory_utility_gate.py tests/test_memory_utility_oracle_audit.py \
  tests/test_logged_online_stage4_train.py
git commit -m "feat: audit causal memory utility headroom"
```

### Task 4: Run the decision audit

**Files:**
- Create runtime reports under `outputs/` only; do not commit checkpoints or raw
  route records.
- Modify the active planning progress log with report paths and hashes.

- [ ] **Step 1: Generate records from post-fix held-out trajectories**

Use matching retrained Stage0/2/4 checkpoints and fixed candidate-union settings
for ToolBench, TrajectBench, tau2, ToolSandbox, APIBank, and sequential BFCL.
Record command lines, source digests, checkpoint hashes, and manifests.

- [ ] **Step 2: Run the oracle audit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/audit_clstr_memory_utility_oracle.py \
  --record_manifest <manifest-1.json> \
  --record_manifest <manifest-2.json> \
  --output_path outputs/clstr_memory_utility_audit/report.json \
  --bootstrap_samples 2000 \
  --seed 17
```

Expected to authorize Task 5: report status `ok` and
`learned_gate_recommended=true`.

- [ ] **Step 3: Enforce the stop condition**

If records/checkpoints do not yet exist, or the recommendation is false, stop
before Task 5. Keep Tasks 1-3 as the completed fixed/heuristic reliability
implementation and set readiness to reject learned mode.

### Task 5: Conditionally train the scalar memory utility gate

**Condition:** Execute only with the exact passing audit report from Task 4.

**Files:**
- Modify: `clstr/memory_utility_gate.py`
- Create: `clstr/memory_utility_gate_train.py`
- Create: `scripts/train_clstr_memory_utility_gate.py`
- Create: `tests/test_memory_utility_gate_train.py`
- Modify: `clstr/model.py`
- Modify: evaluator checkpoint loaders/tests that enable learned mode

- [ ] **Step 1: Write failing gradient-isolation tests**

```python
def test_gate_calibration_backward_updates_only_memory_utility_gate():
    model = _causal_model_with_utility_gate()
    batch = _gate_batch()
    batch.features.requires_grad_(True)
    batch.static_logits.requires_grad_(True)
    batch.dynamic_logits.requires_grad_(True)

    loss, metrics = compute_memory_utility_gate_loss(model.memory_utility_gate, batch)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.memory_utility_gate.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("memory_utility_gate.")
    )
    assert batch.features.grad is None
    assert batch.static_logits.grad is None
    assert batch.dynamic_logits.grad is None
    assert metrics["gate_static_winner_rows"] > 0
    assert metrics["gate_dynamic_winner_rows"] > 0
```

- [ ] **Step 2: Run RED**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate_train.py
```

- [ ] **Step 3: Implement the gate and gate-only training**

Implement:

```python
class MemoryUtilityGate(torch.nn.Module):
    def __init__(self, feature_dim: int = 11, hidden_dim: int = 16, initial_alpha: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.register_buffer("update_count_cap", torch.tensor(1.0))
        self.register_buffer("candidate_count_cap", torch.tensor(1.0))
        probability = min(max(float(initial_alpha), 1.0e-4), 1.0 - 1.0e-4)
        final = self.net[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.constant_(final.bias, math.log(probability / (1.0 - probability)))

    @torch.no_grad()
    def set_normalization(self, mean, std, update_count_cap, candidate_count_cap):
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError("gate normalization shape does not match feature schema")
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std).clamp_min(1.0e-6))
        self.update_count_cap.copy_(torch.as_tensor(update_count_cap).to(self.update_count_cap).clamp_min(1.0))
        self.candidate_count_cap.copy_(
            torch.as_tensor(candidate_count_cap).to(self.candidate_count_cap).clamp_min(1.0)
        )

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1.0e-6)
        return torch.sigmoid(self.net(normalized)).squeeze(-1)


@dataclass
class GateBatch:
    features: torch.Tensor
    static_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    valid_mask: torch.Tensor
    positive_mask: torch.Tensor
    causal_update_count: torch.Tensor
    utility_tie_tolerance: float = 0.01
```

The complete training function first detaches features and branch logits,
filters to memory-active rows with a legal positive and negative, computes gate
alpha, calls `fuse_route_scores()`, and returns fused multi-positive NLL plus the
selector term below. Only gate parameters may receive gradients.

Training consumes detached route records, splits by trajectory before
normalization, and optimizes:

```text
L_gate = multi_positive_NLL(fused_scores)
       + 0.2 * utility_magnitude_weighted_BCE(alpha, dynamic_is_better)
```

Exclude utility ties from BCE but keep them in fused NLL. Winsorize utility
magnitude at the batch 95th percentile and class-balance static/dynamic winners.
Early stop on unweighted held-out macro MRR/NLL. Fail before batching if the
complete training split contains only one winner class.

Checkpoint metadata must include the passing audit hash. Learned mode with a
missing/mismatched audit or missing gate parameters fails loudly.

- [ ] **Step 4: Run GREEN and commit**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py \
  tests/test_model_pipeline.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py
```

Commit:

```bash
git add clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py \
  clstr/model.py scripts/train_clstr_memory_utility_gate.py \
  tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py \
  tests/test_model_pipeline.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py
git commit -m "feat: calibrate evidence-gated memory score reliability"
```

### Task 6: Readiness and final verification

**Files:**
- Modify: `scripts/audit_clstr_unified_training_readiness.py`
- Modify: active evaluator CLI/Slurm scripts
- Modify: `tests/test_unified_training_readiness.py`
- Modify: `tests/test_sbatch_scripts.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing readiness tests**

Require:

```json
{
  "memory_utility_gate_mode": "static|dynamic|fixed_alpha|heuristic|learned",
  "zero_history_fallback": "exact_static",
  "reliability_feature_schema": "memory_utility_features_v1",
  "reliability_changes_memory_state": false
}
```

Learned mode additionally requires a passing audit report hash and gate
checkpoint. Readiness rejects learned mode when Task 4 is false or unavailable.

- [ ] **Step 2: Run RED, implement, and run GREEN**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py
```

- [ ] **Step 3: Run unconditional source verification**

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_candidate_recall.py tests/test_memory_utility_gate.py \
  tests/test_memory_utility_oracle_audit.py tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py tests/test_global_pool_route_eval.py \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py

env PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts

git ls-files '*.sh' | xargs bash -n
git diff --check
```

If Task 4 did not authorize learned mode, `tests/test_memory_utility_gate_train.py`
must not exist and readiness must explicitly permit only static/dynamic/fixed or
heuristic modes.

- [ ] **Step 4: Run conditional learned-gate verification**

Only after Task 5:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_memory_utility_gate_train.py tests/test_model_pipeline.py \
  tests/test_current_state_route_eval.py tests/test_logged_online_stage4_train.py
```

- [ ] **Step 5: Commit readiness/documentation**

```bash
git add scripts/audit_clstr_unified_training_readiness.py scripts/sbatch README.md \
  tests/test_unified_training_readiness.py tests/test_sbatch_scripts.py
git commit -m "docs: expose evidence-gated memory reliability modes"
```

## Empirical acceptance

Fixed/heuristic backoff may be reported as baselines after source verification.
The learned gate enters the final method only if it beats static, dynamic, best
fixed alpha, and heuristic by the predeclared margins, recovers at least 25% of
oracle headroom, reduces dynamic-worsened rows by at least 20%, retains at least
80% of dynamic-improved rows, and has no main benchmark more than 0.005 MRR below
static.
