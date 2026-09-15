# Stage4 Candidate Admission and Constrained Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stage4-only candidate admission and candidate-wise constrained residual reranker initialized from the selected CMC checkpoint, then validate it with a 300-step smoke and at most 1,200 initial full steps.

**Architecture:** The selected CMC residual adapter supplies frozen memory-conditioned dynamic logits. A new candidate-wise module scores the unchanged static-top-500 plus dynamic-extra-64 union, using an admission probability only for dynamic extras and a signed residual for all candidates. Only the new module trains; the compact checkpoint embeds the frozen CMC adapter plus the new module so it remains a single Stage4 delta over Stage2.

**Tech Stack:** Python 3.10, PyTorch, existing CLSTR Stage4 trainer and validation pipeline, frozen Qwen embedding/handoff caches, pytest through Slurm, Slurm A800/H100/H200 launchers.

---

## File map

- Create `clstr/candidate_admission_residual.py`: candidate-wise feature construction, model head, deployed scoring, and three-term objective.
- Modify `clstr/model.py`: instantiate the new Stage4 module with zero-residual initialization.
- Modify `clstr/stage4_safe_memory.py`: freeze contract, delta serialization, and parent-CMC identity validation.
- Modify `clstr/stage4_act_train.py`: candidate-union training batch, objective metrics, method dispatch, and compact checkpoint contents.
- Modify `clstr/stage4_validation.py`: deployed validation scoring, admission AUPRC, no-regret checks, and checkpoint selection.
- Modify `clstr/stage4_quality_gate.py`: fail-closed quality contract for the new method.
- Modify `scripts/run_clstr_stage4_act_train.py`: load and validate the selected base CMC checkpoint before training.
- Modify `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`: forward the base CMC checkpoint argument.
- Create `scripts/sbatch/run_qwen06_clstr_candidate_admission_stage4.sh`: 300/1,200-step cached Qwen launcher.
- Modify `clstr/current_state_route_eval.py` and `clstr/alfworld_eval.py`: deploy the exact trained scorer.
- Modify `clstr/qwen_clstr_final_chain.py`, `clstr/qwen_clstr_lineage.py`, and `clstr/native_benchmark_checkpoint_adapter.py`: bind and load the new Stage4 delta.
- Add focused tests in `tests/test_candidate_admission_residual.py`, `tests/test_stage4_safe_memory.py`, `tests/test_stage4_act_train.py`, `tests/test_stage4_validation.py`, `tests/test_stage4_quality_gate.py`, `tests/test_current_state_route_eval.py`, `tests/test_alfworld_eval.py`, `tests/test_qwen_clstr_final_chain.py`, and `tests/test_qwen_clstr_training_launchers.py`.

## Task 1: Candidate-wise scorer and objective

**Files:**
- Create: `clstr/candidate_admission_residual.py`
- Create: `tests/test_candidate_admission_residual.py`

- [ ] **Step 1: Write RED tests for the deployed scoring equation**

Cover shared candidates, dynamic extras, invalid padding, zero causal history,
the exact `2.0` bound, and zero-initialized exact-static behavior:

```python
def test_candidate_admission_scores_shared_and_extra_candidates() -> None:
    class FixedHead(torch.nn.Module):
        def forward(self, h, memory_delta, candidate_embeddings, scalar_features):
            return (
                torch.tensor([[0.0, 4.0, -4.0]]),
                torch.tensor([[1.0, 1.0, 1.0]]),
            )

    head = FixedHead()
    output = score_candidate_admission_residual(
        head,
        h=torch.zeros(1, 4),
        memory_delta=torch.zeros(1, 4),
        candidate_embeddings=torch.zeros(1, 3, 4),
        static_logits=torch.tensor([[3.0, 2.0, 1.0]]),
        dynamic_logits=torch.tensor([[4.0, 3.0, 2.0]]),
        dynamic_extra_mask=torch.tensor([[False, True, True]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
        causal_update_count=torch.ones(1),
        residual_bound=2.0,
    )
    assert output.final_logits[0, 0] > output.static_logits[0, 0]
    assert output.final_logits[0, 1] > output.final_logits[0, 2]
    assert torch.all((output.final_logits - output.static_logits).abs() <= 2.0)
```

- [ ] **Step 2: Run the primitive RED tests through Slurm**

Run:

```bash
sbatch --parsable --job-name=admit-red --partition=gpu_a800 --gpus=1 \
  --time=00:10:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_candidate_admission_residual.py -q'"
```

Expected: failures because the module and APIs do not exist.

- [ ] **Step 3: Implement the head and exact deployed scorer**

Define these public APIs:

```python
CANDIDATE_ADMISSION_RESIDUAL_V1 = "candidate_admission_residual_v1"
CANDIDATE_ADMISSION_SCALAR_FEATURES = (
    "static_standardized_logit",
    "dynamic_standardized_logit",
    "centered_dynamic_minus_static",
    "normalized_static_rank",
    "normalized_dynamic_rank",
    "is_dynamic_extra",
    "log_normalized_update_count",
    "log_normalized_candidate_count",
)

class CandidateAdmissionResidualHead(torch.nn.Module):
    def __init__(self, model_dim: int, scalar_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        input_dim = 5 * int(model_dim) + int(scalar_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, int(hidden_dim)),
            torch.nn.GELU(),
        )
        self.admission_head = torch.nn.Linear(int(hidden_dim), 1)
        self.residual_head = torch.nn.Linear(int(hidden_dim), 1)
        torch.nn.init.zeros_(self.admission_head.weight)
        torch.nn.init.constant_(self.admission_head.bias, -4.0)
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        h: torch.Tensor,
        memory_delta: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = int(candidate_embeddings.size(1))
        h_rows = h.unsqueeze(1).expand(-1, width, -1)
        memory_rows = memory_delta.unsqueeze(1).expand(-1, width, -1)
        features = torch.cat(
            (
                h_rows,
                memory_rows,
                candidate_embeddings,
                h_rows * candidate_embeddings,
                memory_rows * candidate_embeddings,
                scalar_features,
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        return (
            self.admission_head(hidden).squeeze(-1),
            self.residual_head(hidden).squeeze(-1),
        )

@dataclass(frozen=True)
class CandidateAdmissionScoringOutput:
    final_logits: torch.Tensor
    static_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    admission_logits: torch.Tensor
    admission_probability: torch.Tensor
    bounded_residual: torch.Tensor

def candidate_admission_scalar_features(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.to(dtype=torch.bool)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)

    def standardized(values: torch.Tensor) -> torch.Tensor:
        mean = values.masked_fill(~valid, 0).sum(dim=-1, keepdim=True) / count
        centered = values - mean
        variance = centered.square().masked_fill(~valid, 0).sum(
            dim=-1, keepdim=True
        ) / count
        return centered / variance.sqrt().clamp_min(1.0e-6)

    def normalized_rank(values: torch.Tensor) -> torch.Tensor:
        floor = torch.finfo(values.dtype).min
        order = torch.argsort(
            values.masked_fill(~valid, floor),
            dim=-1,
            descending=True,
            stable=True,
        )
        ranks = torch.empty_like(order)
        positions = torch.arange(values.size(1), device=values.device).expand_as(order)
        ranks.scatter_(1, order, positions)
        return ranks.to(values.dtype) / count.to(values.dtype).clamp_min(1)

    static_z = standardized(static_logits)
    dynamic_z = standardized(dynamic_logits)
    centered_residual = standardized(dynamic_logits - static_logits)
    update_feature = torch.log1p(causal_update_count).unsqueeze(-1) / torch.log(
        static_logits.new_tensor(17.0)
    )
    update_feature = update_feature.expand_as(static_logits)
    count_feature = torch.log1p(count.to(static_logits.dtype)) / torch.log(
        static_logits.new_tensor(257.0)
    )
    count_feature = count_feature.expand_as(static_logits)
    return torch.stack(
        (
            static_z,
            dynamic_z,
            centered_residual,
            normalized_rank(static_logits),
            normalized_rank(dynamic_logits),
            dynamic_extra_mask.to(static_logits.dtype),
            update_feature,
            count_feature,
        ),
        dim=-1,
    ).masked_fill(~valid.unsqueeze(-1), 0)

def score_candidate_admission_residual(
    head: CandidateAdmissionResidualHead,
    *,
    h: torch.Tensor,
    memory_delta: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
    residual_bound: float = 2.0,
) -> CandidateAdmissionScoringOutput:
    scalar_features = candidate_admission_scalar_features(
        static_logits,
        dynamic_logits,
        dynamic_extra_mask,
        valid_mask,
        causal_update_count,
    )
    admission_logits, raw_residual = head(
        h,
        memory_delta,
        candidate_embeddings,
        scalar_features,
    )
    bounded = residual_bound * torch.tanh(raw_residual / residual_bound)
    admission = torch.sigmoid(admission_logits)
    scale = torch.where(dynamic_extra_mask, admission, torch.ones_like(admission))
    active = (causal_update_count > 0).unsqueeze(-1) & valid_mask
    final = torch.where(active, static_logits + scale * bounded, static_logits)
    final = final.masked_fill(~valid_mask, torch.finfo(final.dtype).min)
    return CandidateAdmissionScoringOutput(
        final_logits=final,
        static_logits=static_logits,
        dynamic_logits=dynamic_logits,
        admission_logits=admission_logits,
        admission_probability=admission,
        bounded_residual=bounded,
    )
```

Use a shared MLP trunk over `[h, memory_delta, candidate_embedding,
h*candidate_embedding, memory_delta*candidate_embedding, scalar_features]`,
with separate scalar admission and residual output layers. Zero-initialize the
residual output layer and initialize admission bias to `-4.0`. Force exact
static logits when `causal_update_count == 0`; invalid positions receive the
dtype floor.

- [ ] **Step 4: Write RED tests for the three-term objective**

Test listwise loss, class-balanced extra-only admission BCE, static-margin
no-regret, and a rescued-positive gain row:

```python
objective = candidate_admission_residual_loss(
    scoring=scoring,
    positive_mask=positive,
    valid_mask=valid,
    dynamic_extra_mask=extras,
    admission_weight=1.0,
    counterfactual_weight=1.0,
    gain_margin=0.1,
)
assert objective.admission_positive_count == 1
assert objective.admission_negative_count == 2
assert objective.no_regret_loss.item() > 0
```

- [ ] **Step 5: Implement `candidate_admission_residual_loss`**

Return a dataclass containing `loss`, `listwise_loss`, `admission_loss`,
`no_regret_loss`, `safety_loss`, `gain_loss`, and admission class counts. Use:

```text
loss = listwise_loss + admission_weight * admission_loss
       + counterfactual_weight * (safety_loss + gain_loss)
```

Admission labels must be read only by the loss. Clamp batch positive weight to
`[1, 64]`; batches without extra positives still train their negative extras.

- [ ] **Step 6: Run GREEN and commit**

Run the same Slurm pytest command. Expected: all primitive tests pass.

```bash
git add clstr/candidate_admission_residual.py tests/test_candidate_admission_residual.py
git commit -m "feat: add candidate admission residual objective"
```

## Task 2: Model, freeze, and compact checkpoint contract

**Files:**
- Modify: `clstr/model.py`
- Modify: `clstr/stage4_safe_memory.py`
- Modify: `tests/test_model_pipeline.py`
- Modify: `tests/test_stage4_safe_memory.py`

- [ ] **Step 1: Write RED model and freeze-contract tests**

Require `CLSTRModel.route_memory_candidate_admission_residual`, and require the
new freeze path to train only that module while keeping the loaded CMC adapter
frozen. Require compact state keys to contain exactly:

```text
route_memory_residual_adapter.*
route_memory_candidate_admission_residual.*
```

Also require checkpoint metadata fields `base_cmc_checkpoint_sha256`,
`parent_stage2_full_router_digest`, and `parent_stage2_fast_router_digest`.

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --parsable --job-name=admit-freeze-red --partition=gpu_a800 --gpus=1 \
  --time=00:10:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_model_pipeline.py tests/test_stage4_safe_memory.py -q'"
```

- [ ] **Step 3: Add model initialization and checkpoint helpers**

Add `self.route_memory_candidate_admission_residual =
CandidateAdmissionResidualHead(config.d)` in `CLSTRModel`. Add constants and
helpers in `stage4_safe_memory.py`:

```python
CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES = (
    "route_memory_residual_adapter.",
    "route_memory_candidate_admission_residual.",
)
CANDIDATE_ADMISSION_STAGE4_TRAINABLE_MODULES = (
    "route_memory_candidate_admission_residual",
)

def freeze_stage4_candidate_admission(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = model.route_memory_candidate_admission_residual
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    model.eval()
    module.train()
    return {
        "trainable_modules": ["route_memory_candidate_admission_residual"],
        "optimizer_parameter_names": sorted(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "full_router_digest": router_state_digest(model, scope="full"),
        "fast_router_digest": router_state_digest(model, scope="fast"),
    }

def stage4_candidate_admission_delta_state_dict(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES)
    }
    validate_stage4_candidate_admission_delta_state_dict(state)
    return state

def validate_stage4_candidate_admission_delta_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    forbidden = sorted(
        name for name in state
        if not str(name).startswith(CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"forbidden candidate-admission delta key: {forbidden[0]}")
    missing = [
        prefix for prefix in CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES
        if not any(str(name).startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(f"candidate-admission delta missing prefix: {missing[0]}")
    return {"status": "ok", "state_keys": sorted(state)}

def validate_stage4_candidate_admission_parent_cmc(
    payload: dict[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    recorded = str(payload.get("base_cmc_checkpoint_sha256") or "")
    if recorded != str(expected_sha256):
        raise ValueError("candidate-admission base CMC identity mismatch")
    return {"status": "ok", "base_cmc_checkpoint_sha256": recorded}
```

Do not reset or unfreeze `route_memory_residual_adapter`.

- [ ] **Step 4: Run GREEN and commit**

```bash
git add clstr/model.py clstr/stage4_safe_memory.py tests/test_model_pipeline.py tests/test_stage4_safe_memory.py
git commit -m "feat: add candidate admission Stage4 checkpoint contract"
```

## Task 3: Candidate-union Stage4 training batch

**Files:**
- Modify: `clstr/stage4_act_train.py`
- Modify: `tests/test_stage4_act_train.py`

- [ ] **Step 1: Write RED tests for the new batch builder**

Add `CANDIDATE_ADMISSION_RESIDUAL_V1` to `STAGE4_METHODS` and require
`_build_stage4_candidate_admission_route_batch` to:

- compute static and frozen-CMC dynamic full logits;
- build `static_k=500`, `dynamic_extra_k=64` candidates without target injection;
- gather candidate embeddings and provenance masks;
- train only the new module;
- report shared/extra positive and negative counts;
- fail closed when the complete training set contains no dynamic-extra positive.

Use a three-skill test model with `static_k=2`, `dynamic_extra_k=1` so the
positive appears only in the dynamic extra.

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --parsable --job-name=admit-batch-red --partition=gpu_a800 --gpus=1 \
  --time=00:15:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_stage4_act_train.py -k candidate_admission -q'"
```

- [ ] **Step 3: Implement the batch dataclass and builder**

Add:

```python
@dataclass(frozen=True)
class Stage4CandidateAdmissionRouteBatch:
    total_loss: torch.Tensor
    objective: CandidateAdmissionResidualObjective
    scoring: CandidateAdmissionScoringOutput
    candidate_union: CandidateUnion
    candidate_positive_mask: torch.Tensor
    candidate_valid_mask: torch.Tensor
    dynamic_extra_mask: torch.Tensor
    metrics: dict[str, Any]
```

Use existing `build_static_dynamic_union`, `candidate_provenance_mask`,
`full_pool_positive_mask`, and `legal_skill_pool_mask`. The existing selected
CMC adapter supplies dynamic logits under `torch.no_grad()`; only the new
candidate module remains in the autograd graph.

- [ ] **Step 4: Route the method through `_compute_stage4_act_loss` and freeze logic**

The new method must require unified-memory full-pool routing and frozen replay
prefix expansion, just like CMC. Add metrics:

```text
stage4_candidate_admission_listwise_loss
stage4_candidate_admission_bce_loss
stage4_candidate_admission_no_regret_loss
stage4_candidate_admission_auprc
stage4_candidate_admission_positive_count
stage4_candidate_admission_negative_count
stage4_candidate_admission_dynamic_rescue_rows
```

- [ ] **Step 5: Run GREEN and commit**

```bash
git add clstr/stage4_act_train.py tests/test_stage4_act_train.py
git commit -m "feat: train Stage4 candidate admission reranker"
```

## Task 4: Base-CMC initialization, validation, and quality gate

**Files:**
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- Modify: `clstr/stage4_validation.py`
- Modify: `clstr/stage4_quality_gate.py`
- Modify: `tests/test_stage4_validation.py`
- Modify: `tests/test_stage4_quality_gate.py`
- Modify: `tests/test_stage4_act_train.py`

- [ ] **Step 1: Write RED initialization and validation tests**

Require `--base_cmc_checkpoint_path` for the new method. Load it after the
Stage2 heads, validate its parent Stage2 router digests, and record its SHA-256.
Require step-0 exact static, finite gradients, admission AUPRC and prevalence,
final MRR, static MRR, and no-regret margin violations in every validation
report.

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --parsable --job-name=admit-validation-red --partition=gpu_a800 --gpus=1 \
  --time=00:15:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_stage4_act_train.py tests/test_stage4_validation.py tests/test_stage4_quality_gate.py -k candidate_admission -q'"
```

- [ ] **Step 3: Implement base-CMC initialization**

Add the CLI argument and load only the validated CMC delta prefixes. Fail if
the method is new and the argument is absent, or if the CMC checkpoint SHA or
parent router digests drift. Pass the identity into training and every compact
checkpoint.

- [ ] **Step 4: Implement deployed validation and selection**

Validation must call the same scorer used by inference before and after
final-k. Select checkpoints by final macro MRR subject to:

```text
finite gradients
dynamic-extra positive count > 0
admission AUPRC > admission prevalence
final macro MRR >= static macro MRR
no-regret margin violation rate <= configured validation tolerance
```

Do not introduce benchmark-specific thresholds.

- [ ] **Step 5: Extend the quality gate and compact payload validation**

Require the new method name, training objective, base CMC identity, frozen
adapter, trainable-module list, route digests, validation identities, and
promotion checks. Reject old CMC or provenance manifests masquerading as the
new method.

- [ ] **Step 6: Run GREEN and commit**

```bash
git add scripts/run_clstr_stage4_act_train.py scripts/sbatch/run_clstr_unified_stage4_act_train.sh clstr/stage4_validation.py clstr/stage4_quality_gate.py tests/test_stage4_act_train.py tests/test_stage4_validation.py tests/test_stage4_quality_gate.py
git commit -m "feat: validate candidate admission Stage4 checkpoints"
```

## Task 5: Exact inference and final-chain integration

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `clstr/memory_utility_gate.py`
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `clstr/qwen_clstr_lineage.py`
- Modify: `clstr/native_benchmark_checkpoint_adapter.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_alfworld_eval.py`
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_native_benchmark_checkpoint_adapter.py`

- [ ] **Step 1: Write RED inference contract tests**

Require mode `candidate_admission_residual` to use the frozen CMC dynamic
endpoint, build the same provenance mask before and after final-k, and call the
new scorer. Test zero-history exact static and ALFWorld supplied-pool behavior.
Require final chain reliability metadata:

```json
{
  "mode": "candidate_admission_residual",
  "fixed_alpha": null,
  "gate_checkpoint": null,
  "safe_memory_residual_bound": 2.0,
  "feature_update_count_cap": 16.0,
  "feature_candidate_count_cap": 256.0,
  "base_cmc_checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "deployed_reliability_source": "candidate_admission_constrained_residual"
}
```

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --parsable --job-name=admit-infer-red --partition=gpu_a800 --gpus=1 \
  --time=00:15:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_current_state_route_eval.py tests/test_alfworld_eval.py tests/test_qwen_clstr_final_chain.py tests/test_native_benchmark_checkpoint_adapter.py -q'"
```

- [ ] **Step 3: Implement shared inference routing**

Factor one helper that scores candidate unions with the new module and use it
in current-state and ALFWorld paths. Preserve all existing `cmc`,
`cmc_candidate_provenance`, static, and causal-gate behavior for ablations.

- [ ] **Step 4: Extend lineage, adapter, and final-chain validation**

Recognize the new Stage4 method and compact delta, preserve the Stage0/1/2
checkpoint chain, and bind the selected base CMC SHA. Runtime mode must remain
`candidate_admission_residual`; it must not map back to `cmc`.

- [ ] **Step 5: Run GREEN and commit**

```bash
git add clstr/current_state_route_eval.py clstr/alfworld_eval.py clstr/memory_utility_gate.py clstr/qwen_clstr_final_chain.py clstr/qwen_clstr_lineage.py clstr/native_benchmark_checkpoint_adapter.py tests/test_current_state_route_eval.py tests/test_alfworld_eval.py tests/test_qwen_clstr_final_chain.py tests/test_native_benchmark_checkpoint_adapter.py
git commit -m "feat: deploy candidate admission Stage4 scoring"
```

## Task 6: Accelerated Qwen launcher and 300-step smoke

**Files:**
- Create: `scripts/sbatch/run_qwen06_clstr_candidate_admission_stage4.sh`
- Modify: `tests/test_qwen_clstr_training_launchers.py`

- [ ] **Step 1: Write RED launcher tests**

Require the launcher to resolve the selected CMC checkpoint from
`stage4_cmc_full/stage4_selection.json`, use the existing row-sharded Stage0
handoff cache, freeze the backbone and all parent modules, and set:

```text
smoke: MAX_STEPS=300, validation every 100 steps
full:  MAX_STEPS=1200, validation every 300 steps
BATCH_SIZE=16
LEARNING_RATE=3e-5
MINIMUM_LEARNING_RATE=3e-6
STAGE4_METHOD=candidate_admission_residual_v1
```

- [ ] **Step 2: Run launcher RED/GREEN tests through Slurm**

```bash
sbatch --parsable --job-name=admit-launcher-tests --partition=gpu_a800 --gpus=1 \
  --time=00:10:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_qwen_clstr_training_launchers.py -k candidate_admission -q'"
```

- [ ] **Step 3: Implement and syntax-check the launcher**

The smoke output is
`outputs/qwen06_clstr_postfix/stage4_candidate_admission_smoke`; the full output
is `stage4_candidate_admission_full`. Use `bash -n` locally and never run Torch
on the storage node.

- [ ] **Step 4: Commit the launcher**

```bash
git add scripts/sbatch/run_qwen06_clstr_candidate_admission_stage4.sh tests/test_qwen_clstr_training_launchers.py
git commit -m "feat: launch accelerated candidate admission Stage4"
```

- [ ] **Step 5: Submit and supervise the 300-step smoke**

Submit through Slurm. Check once after setup, once near step 100, and at
completion. Stop unless admission AUPRC beats prevalence, validation final MRR
is at least static, gradients are finite/nonzero, dynamic-extra positives are
present, and throughput is consistent with cached frozen-Qwen training.

## Task 7: Initial 1,200-step run and benchmark promotion

**Files:**
- Generated: `outputs/qwen06_clstr_postfix/stage4_candidate_admission_full/`
- Generated: `outputs/qwen06_clstr_postfix/candidate_admission_evaluation/`
- Update: `finalwork/clstr_multibench_results.md`

- [ ] **Step 1: Submit the initial full run only after smoke passes**

Run to steps 300/600/900/1200 with validation every 300 steps. Continue from
the smoke checkpoint only if optimizer/scheduler and data-protocol identities
match. Do not extend beyond 1200 without checking whether validation is still
improving.

- [ ] **Step 2: Run the full affected regression matrix through Slurm**

```bash
sbatch --parsable --job-name=admit-final-tests --partition=gpu_a800 --gpus=1 \
  --time=00:30:00 --output=slurm-%j.out \
  --wrap="bash -lc 'export PROJECT_ROOT=$PWD; source scripts/sbatch/_clstr_gpu_env.sh; exec \"\${PYTHON_BIN}\" -m pytest tests/test_candidate_admission_residual.py tests/test_stage4_safe_memory.py tests/test_stage4_act_train.py tests/test_stage4_validation.py tests/test_stage4_quality_gate.py tests/test_current_state_route_eval.py tests/test_alfworld_eval.py tests/test_qwen_clstr_final_chain.py tests/test_native_benchmark_checkpoint_adapter.py tests/test_qwen_clstr_training_launchers.py -q'"
```

- [ ] **Step 3: Resolve a fresh final chain and run ToolBench full first**

Use an isolated evaluation root. Promotion requires MRR strictly greater than
`0.3076335`, positive dynamic-extra rescue rows, correct final-chain/corpus
identities, and no protocol/nonfinite blocker.

- [ ] **Step 4: Run ToolSandbox and Tau2-base only after ToolBench passes**

Each fused MRR must remain within `0.005` of its static branch. Do not run
ALFWorld or migrate to BGE if either fails.

- [ ] **Step 5: Run ALFWorld only after all three routing gates pass**

Record success rate, routing evidence, recurrent-memory use, executor identity,
and comparison with the existing CMC chain.

- [ ] **Step 6: Record results and decide whether to extend training**

Update the results table with every checkpoint and denominator. Extend from
1200 toward 2000/3000 only if validation MRR is still increasing without rising
no-regret violations; otherwise lock the selected <=1200 checkpoint.
