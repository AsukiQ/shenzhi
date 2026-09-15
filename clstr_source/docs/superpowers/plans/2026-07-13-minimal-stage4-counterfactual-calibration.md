# Minimal Stage4 Counterfactual Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an anchored Stage2 routing foundation and a minimal Stage4 Counterfactual Memory Calibration (CMC) delta while keeping Qwen frozen, preserving exact static/dynamic endpoints, and repairing the Tau2/ToolSandbox evaluation contracts.

**Architecture:** Stage2 restores trainability for the initial-belief/unified-router foundation and constrains its static route against an immutable step-zero teacher. Stage4 freezes the selected Stage2 checkpoint and trains only a zero-initialized memory residual adapter plus a candidate-aware utility gate under dynamic, fused, and no-regret objectives. Evaluation keeps dynamic-extra recall, uses exact linear fusion, reports Tau2 `base` separately from the 13,907-row stress corpus, and pairs ToolSandbox checkpoint-faithful scoring with a local-table rebuild ablation.

**Tech Stack:** Python 3, PyTorch, existing frozen Qwen embedding caches, pytest through Slurm, JSON/JSONL manifests, Bash Slurm launchers.

---

## File structure

- Create `clstr/stage2_static_route_anchor.py`: immutable router-teacher snapshot and score-level KL.
- Create `clstr/counterfactual_memory_calibration.py`: residual adapter, exact CMC objective, feature normalization contract, and delta-key validation.
- Modify `clstr/model.py`: own the new CMC adapter and candidate-aware gate without changing legacy checkpoint behavior.
- Modify `clstr/full_base_train.py`: anchored Stage2 trainability, loss integration, metrics, and checkpoint metadata.
- Modify Stage1/Stage2 launchers and quality gates: remove zero-weight legacy objectives, disable Stage2 STOP, and remove the Stage2 legacy counterfactual objective.
- Modify `clstr/stage2_quality_gate.py`: static-anchor and dynamic-improvement release checks.
- Modify `clstr/stage4_act_train.py`: freeze Stage2, compute CMC logits/features/losses, and save only the new delta.
- Modify `clstr/stage4_validation.py` and `clstr/stage4_quality_gate.py`: exact-endpoint selection and nonzero-step promotion.
- Modify `clstr/current_state_route_eval.py`, `clstr/memory_utility_gate.py`, and checkpoint loaders: deploy the same CMC fusion and dynamic-extra recall path.
- Modify `clstr/tau2_route_eval.py`: explicit official split selection and STOP/refusal accounting.
- Modify `clstr/toolsandbox_route_eval.py`: paired checkpoint-faithful and local-table-rebuild modes.
- Modify Qwen CLI/sbatch launchers and manifests for explicit schema identities.
- Add focused tests under `tests/`; run every pytest command through Slurm.

### Task 1: Add the immutable Stage2 static-route teacher

**Files:**
- Create: `clstr/stage2_static_route_anchor.py`
- Create: `tests/test_stage2_static_route_anchor.py`
- Modify: `tests/test_full_base_train.py`

- [x] **Step 1: Write failing teacher and KL tests**

```python
def test_static_route_anchor_is_zero_for_identical_student() -> None:
    model = TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.tensor([[True, True], [True, True]])
    loss = static_route_anchor_kl(model, teacher, h, valid)
    assert torch.equal(loss, torch.zeros_like(loss))


def test_static_route_anchor_detects_router_drift_without_teacher_gradients() -> None:
    model = TinyUnifiedModel()
    teacher = Stage2StaticRouteTeacher.from_model(model)
    with torch.no_grad():
        model.unified_retriever.weight.add_(0.25)
    loss = static_route_anchor_kl(model, teacher, torch.eye(2), torch.ones(2, 2, dtype=torch.bool))
    loss.backward()
    assert loss.item() > 0.0
    assert model.unified_retriever.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
```

- [x] **Step 2: Run RED through Slurm**

Run:

```bash
sbatch --wait --job-name=cmc-anchor-red --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:10:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_stage2_static_route_anchor.py'
```

Expected: import failure for `clstr.stage2_static_route_anchor`.

- [x] **Step 3: Implement the teacher without duplicating Qwen or `skill_table.E`**

```python
class Stage2StaticRouteTeacher(torch.nn.Module):
    @classmethod
    def from_model(cls, model: Any) -> "Stage2StaticRouteTeacher":
        teacher = cls(
            initial_belief_head=copy.deepcopy(model.initial_belief_head),
            unified_retriever=copy.deepcopy(model.unified_retriever),
            belief_logit_scale=model.skill_table.logit_scale_belief.detach().clone(),
            belief_bias=model.skill_table.skill_bias_belief.detach().clone(),
            initial_belief_top_k=int(model.config.initial_belief_top_k),
        )
        teacher.requires_grad_(False)
        teacher.eval()
        return teacher

    def full_logits(self, model: Any, h: torch.Tensor) -> torch.Tensor:
        cosine = model.skill_table._cosine_scores(h).detach()
        belief_logits = self.belief_logit_scale.exp().clamp(max=100.0) * cosine + self.belief_bias
        belief = sparse_subspace_from_logits(
            belief_logits,
            model.skill_table.E.detach(),
            top_k=self.initial_belief_top_k,
        )
        memory = self.initial_belief_head(h, belief)
        route = self.unified_retriever(h, memory)
        return route @ model.skill_table.E.detach().to(route).t()


def sparse_subspace_from_logits(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    k = min(max(1, int(top_k)), int(logits.size(-1)))
    values, indices = torch.topk(logits, k=k, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    selected = embeddings.index_select(0, indices.reshape(-1)).view(logits.size(0), k, -1)
    return torch.bmm(probabilities.unsqueeze(1), selected).squeeze(1)
```

`static_route_anchor_kl()` must mask invalid candidates, use `KL(p_teacher || p_student)`, detach teacher scores, and return exact zero when valid logits are identical.

- [x] **Step 4: Run GREEN and commit**

Run the Step 2 command and require all tests to pass, then:

```bash
git add clstr/stage2_static_route_anchor.py tests/test_stage2_static_route_anchor.py tests/test_full_base_train.py
git commit -m "feat: add stage2 static route anchor"
```

### Task 2: Restore and anchor the Stage2 routing foundation

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage2_quality_gate.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_stage2_quality_gate.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`

- [x] **Step 1: Write failing optimizer-scope tests**

```python
def test_anchored_unified_stage2_trains_router_but_freezes_qwen() -> None:
    audit = _freeze_for_full_base(model, route_scorer="unified_memory", anchored_routing_foundation=True)
    assert set(audit["trainable_modules"]) >= {
        "initial_belief_head",
        "unified_retriever",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    }
    assert all(not p.requires_grad for p in model.encoder.backbone.parameters())
    assert model.skill_table.E.requires_grad is False
```

Add a quality-gate test rejecting static MRR degradation below teacher by more than `0.005` and rejecting a checkpoint whose dynamic MRR does not exceed the step-zero dynamic route.

- [x] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-stage2-red --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_full_base_train.py tests/test_stage2_quality_gate.py tests/test_qwen_clstr_training_launchers.py'
```

Expected: failures for the missing `anchored_routing_foundation` contract and anchor metrics.

- [x] **Step 3: Implement anchored Stage2 scope and loss**

Add arguments:

```python
anchored_routing_foundation: bool = False
static_route_anchor_weight: float = 0.1
static_route_anchor_max_regression: float = 0.005
```

When enabled for unified memory, train transition/correction heads plus `initial_belief_head`, `unified_retriever`, `skill_table.logit_scale_belief`, and `skill_table.skill_bias_belief`; keep `encoder.*`, `skill_table.E`, retrieval scale/bias, and Qwen frozen. Construct `Stage2StaticRouteTeacher` before the optimizer. For rows with no replay prefix, add:

```python
anchor_loss = static_route_anchor_kl(model, teacher, h_next, legal_pool.mask)
total_loss = existing_component_complete_loss + static_route_anchor_weight * anchor_loss
```

Persist teacher digest, anchor row count, anchor KL, teacher/static/dynamic MRR, and trainable parameter names.

- [x] **Step 4: Wire launchers and selection checks**

Add launcher defaults:

```bash
ANCHORED_ROUTING_FOUNDATION=${ANCHORED_ROUTING_FOUNDATION:-1}
STATIC_ROUTE_ANCHOR_WEIGHT=${STATIC_ROUTE_ANCHOR_WEIGHT:-0.1}
STATIC_ROUTE_ANCHOR_MAX_REGRESSION=${STATIC_ROUTE_ANCHOR_MAX_REGRESSION:-0.005}
```

The quality gate must stop before Stage4 unless static MRR is within `0.005` of teacher and dynamic MRR is strictly higher than step-zero dynamic MRR.

- [x] **Step 5: Run GREEN and commit**

Run the Step 2 command, require zero failures, then commit:

```bash
git add clstr/full_base_train.py clstr/stage2_quality_gate.py scripts/run_clstr_stage2_full_base_train.py scripts/sbatch/run_qwen06_clstr_stage2_train.sh tests/test_full_base_train.py tests/test_stage2_quality_gate.py tests/test_qwen_clstr_training_launchers.py
git commit -m "feat: train anchored stage2 router foundation"
```

### Task 2A: Remove dead canonical Stage1/Stage2 losses

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage1_heads_quality_gate.py`
- Modify: `clstr/stage2_quality_gate.py`
- Modify: `scripts/run_clstr_stage1_heads_init.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage1_train.sh`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_stage1_heads_quality_gate.py`
- Modify: `tests/test_stage2_quality_gate.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`

- [x] **Step 1: Write failing canonical-loss tests**

Add tests asserting:

```python
def test_canonical_stage2_loss_contract_excludes_dead_objectives() -> None:
    weights = canonical_stage_loss_weights("stage2")
    assert {key: value for key, value in weights.items() if value > 0.0} == {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
    }
    assert all(
        weights[key] == 0.0
        for key in (
            "routing",
            "hard_negative_margin",
            "Q_success",
            "transition_hard_negative_margin",
            "STOP",
            "counterfactual_utility",
        )
    )


def test_zero_weight_losses_do_not_execute_forward_helpers(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("zero-weight loss helper executed")

    monkeypatch.setattr(full_base_train_module, "_retrieval_contrastive_loss_from_logits", fail)
    monkeypatch.setattr(full_base_train_module, "compute_q_success_loss", fail)
    monkeypatch.setattr(full_base_train_module, "_transition_hard_negative_margin_loss", fail)
    model = _NativePolicyHeadModel()
    model.q_success_head = _LinearQSuccessHead()
    rows = [{
        "state_text": "state current",
        "action_text": "expert action",
        "admissible_actions": ["wrong action", "expert action"],
        "expert_action": "expert action",
        "skill_id": "skill/a",
        "next_skill_id": "skill/b",
        "hard_negative_action": "wrong action",
        "loss_mask": {
            "L_policy": True,
            "routing": True,
            "hard_negative_margin": True,
            "Q_success": True,
            "transition_hard_negative_margin": True,
        },
    }]
    _compute_full_base_loss(
        model=model,
        action_adapter=_FailingActionAdapter(),
        batch=rows,
        skill_id_to_idx={"skill/a": 0, "skill/b": 1},
        device=torch.device("cpu"),
        loss_weights=canonical_stage_loss_weights("stage2"),
    )


def test_stage2_freezes_unused_stop_and_q_success_heads() -> None:
    audit = _freeze_for_full_base(
        model,
        route_scorer="unified_memory",
        active_loss_weights=canonical_stage_loss_weights("stage2"),
    )
    assert "stop_head" not in audit["trainable_modules"]
    assert "q_success_head" not in audit["trainable_modules"]
```

Quality-gate tests must assert Stage1 no longer requires zero-weight legacy
terms and Stage2 requires only `L_policy`, `L_trans`, `L_trans_skill_ce`, and
`belief`; neither gate requires `STOP`. Launcher tests must reject removed
canonical flags for routing, Q-success, hard-negative margins, STOP in Stage2,
and legacy Stage2 counterfactual utility.

- [x] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=cmc-loss-cleanup-red --gres=gpu:1 --cpus-per-task=4 --time=00:15:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; python -m pytest -q tests/test_full_base_train.py tests/test_stage1_heads_quality_gate.py tests/test_stage2_quality_gate.py tests/test_qwen_clstr_training_launchers.py'\'''
```

Expected: failures for the missing canonical loss contract, zero-weight
short-circuiting, optimizer-scope filtering, and old required-loss lists.

- [x] **Step 3: Implement the canonical contract**

Add one source of truth:

```python
CANONICAL_STAGE_ACTIVE_LOSS_WEIGHTS = {
    "stage1": {
        "L_policy": 0.6,
        "L_trans": 0.2,
        "L_trans_skill_ce": 0.6,
        "belief": 0.1,
    },
    "stage2": {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "belief": 0.1,
    },
}


def canonical_stage_loss_weights(stage_name: str) -> dict[str, float]:
    active = CANONICAL_STAGE_ACTIVE_LOSS_WEIGHTS[stage_name]
    return {key: float(active.get(key, 0.0)) for key in LOSS_WEIGHT_KEYS}
```

Stage1 may accept an explicit experimental STOP override outside the canonical
mapping; Stage2 must reject it. Remove canonical launcher flags for `routing`,
`hard_negative_margin`, `Q_success`, `transition_hard_negative_margin`, Stage2
`STOP`, and Stage2 `counterfactual_utility`. Preserve standalone ablation code
and legacy checkpoint loading.

Every optional loss block in `_compute_full_base_loss()` must require both a
positive normalized weight and an eligible row before doing tensor work. Pass
the normalized active weights into `_freeze_for_full_base()` so `stop_head` and
`q_success_head` stay frozen when their objectives are disabled.

- [x] **Step 4: Update quality gates and reports**

Set the canonical required terms to:

```python
STAGE1_REQUIRED = ["L_policy", "L_trans", "L_trans_skill_ce", "belief"]
STAGE2_REQUIRED = ["L_policy", "L_trans", "L_trans_skill_ce", "belief"]
```

Reports may retain a `legacy_loss_diagnostics` section for historical readers,
but canonical `loss_weights`, training-objective text, optimizer scope, and
paper-facing summaries must omit the removed terms.

- [x] **Step 5: Run GREEN and commit**

Run the Step 2 command and require zero failures, then:

```bash
git add clstr/full_base_train.py clstr/stage1_heads_quality_gate.py clstr/stage2_quality_gate.py scripts/run_clstr_stage1_heads_init.py scripts/run_clstr_stage2_full_base_train.py scripts/sbatch/run_qwen06_clstr_stage1_train.sh scripts/sbatch/run_qwen06_clstr_stage2_train.sh tests/test_full_base_train.py tests/test_stage1_heads_quality_gate.py tests/test_stage2_quality_gate.py tests/test_qwen_clstr_training_launchers.py
git commit -m "refactor: remove dead canonical stage losses"
```

### Task 3: Add CMC residual and counterfactual objective primitives

**Files:**
- Create: `clstr/counterfactual_memory_calibration.py`
- Create: `tests/test_counterfactual_memory_calibration.py`
- Modify: `clstr/model.py`
- Modify: `tests/test_model_pipeline.py`

- [x] **Step 1: Write failing primitive tests**

```python
def test_zero_initialized_adapter_preserves_dynamic_route() -> None:
    adapter = RouteMemoryResidualAdapter(4, hidden_dim=64)
    delta = adapter(torch.ones(2, 4), torch.ones(2, 4))
    assert torch.equal(delta, torch.zeros_like(delta))


def test_cmc_exact_endpoints_and_no_regret_gradient() -> None:
    alpha = torch.tensor([0.5], requires_grad=True)
    result = counterfactual_memory_calibration_loss(
        static_logits=torch.tensor([[4.0, 1.0]]),
        dynamic_logits=torch.tensor([[1.0, 4.0]], requires_grad=True),
        alpha=alpha,
        positive_mask=torch.tensor([[True, False]]),
        valid_mask=torch.tensor([[True, True]]),
    )
    assert torch.equal(result.alpha_zero_logits, result.static_logits)
    assert torch.equal(result.alpha_one_logits, result.dynamic_logits)
    result.loss.backward()
    # Harmful dynamic memory must push alpha down under gradient descent.
    assert alpha.grad is not None and alpha.grad.item() > 0
```

- [x] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-primitives-red --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:10:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_counterfactual_memory_calibration.py tests/test_model_pipeline.py'
```

- [x] **Step 3: Implement the adapter and loss exactly**

```python
class RouteMemoryResidualAdapter(torch.nn.Module):
    def __init__(self, d: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(3 * d),
            torch.nn.Linear(3 * d, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, d),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, memory_delta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((h, memory_delta, h * memory_delta), dim=-1))
```

Use existing `fuse_route_scores()` for exact endpoints. Define utility as:

```python
utility = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=-1) - torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), dim=-1)
loss = -dynamic_utility.mean() - fused_utility.mean() + torch.relu(static_utility.detach() - fused_utility).mean()
```

Add `route_memory_residual_adapter` and `route_memory_candidate_utility_gate = MemoryUtilityGate()` to `CLSTRModel`. Keep legacy `route_memory_utility_gate` for old checkpoint compatibility.

- [x] **Step 4: Run GREEN and commit**

```bash
git add clstr/counterfactual_memory_calibration.py clstr/model.py tests/test_counterfactual_memory_calibration.py tests/test_model_pipeline.py
git commit -m "feat: add counterfactual memory calibration primitives"
```

### Task 4: Integrate minimal Stage4 CMC training and delta checkpoints

**Files:**
- Modify: `clstr/stage4_act_train.py`
- Modify: `clstr/stage4_safe_memory.py`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`
- Modify: `tests/test_stage4_act_train.py`
- Modify: `tests/test_stage4_safe_memory.py`
- Modify: `tests/test_stage_checkpoint_init.py`

- [x] **Step 1: Write failing freeze/delta tests**

```python
def test_cmc_stage4_optimizer_contains_only_two_modules() -> None:
    audit = freeze_stage4_cmc(model)
    assert audit["trainable_modules"] == [
        "route_memory_residual_adapter",
        "route_memory_candidate_utility_gate",
    ]
    assert all(
        name.startswith(("route_memory_residual_adapter.", "route_memory_candidate_utility_gate."))
        for name in audit["optimizer_parameter_names"]
    )


def test_cmc_delta_rejects_stage2_keys() -> None:
    with pytest.raises(ValueError, match="forbidden CMC delta key"):
        validate_stage4_cmc_delta_state_dict({"transition.weight": torch.ones(1)})
```

- [x] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-stage4-red --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_stage4_act_train.py tests/test_stage4_safe_memory.py tests/test_stage_checkpoint_init.py'
```

- [x] **Step 3: Replace the safe-memory Stage4 path with CMC mode**

Add `stage4_method="counterfactual_memory_calibration_v1"`. In the CMC branch:

```python
static_vector = model.unified_retriever(h_next, static_memory)
dynamic_base_vector = model.unified_retriever(h_next, dynamic_memory)
memory_delta = dynamic_memory - static_memory
dynamic_vector = dynamic_base_vector + model.route_memory_residual_adapter(h_next, memory_delta)
static_logits = static_vector @ model.skill_table.E.to(static_vector).t()
dynamic_logits = dynamic_vector @ model.skill_table.E.to(dynamic_vector).t()
features = memory_utility_features(
    static_logits,
    dynamic_logits,
    legal_pool.mask,
    static_memory,
    dynamic_memory,
    causal_update_count,
    update_count_cap=16.0,
    candidate_count_cap=256.0,
).detach()
raw_alpha = model.route_memory_candidate_utility_gate(features)
alpha = effective_memory_alpha(raw_alpha, causal_update_count)
objective = counterfactual_memory_calibration_loss(static_logits, dynamic_logits, alpha, positive_mask, legal_pool.mask)
```

Exclude optimizer rows missing trustworthy trajectory/action/next observation/next state/positive skill and persist exclusion counts. Zero-history rows are validation-only controls. Remove tanh fusion, synthetic candidate-size sweeps, gain margin, safety tolerance, and the `0.05` fused multiplier from CMC mode.

- [x] **Step 4: Save and load only the CMC delta**

CMC checkpoint keys must be exactly:

```text
route_memory_residual_adapter.*
route_memory_candidate_utility_gate.*
```

Record the parent Stage2 full/fast router digests and reject loads whose parent differs.

- [x] **Step 5: Run GREEN and commit**

```bash
git add clstr/stage4_act_train.py clstr/stage4_safe_memory.py scripts/run_clstr_stage4_act_train.py scripts/sbatch/run_qwen06_clstr_stage4_train.sh tests/test_stage4_act_train.py tests/test_stage4_safe_memory.py tests/test_stage_checkpoint_init.py
git commit -m "feat: train minimal stage4 cmc delta"
```

### Task 5: Make validation and promotion match deployed CMC

**Files:**
- Modify: `clstr/stage4_validation.py`
- Modify: `clstr/stage4_quality_gate.py`
- Modify: `tests/test_stage4_validation.py`
- Modify: `tests/test_stage4_quality_gate.py`

- [x] **Step 1: Write failing selection tests**

Create fixtures where step 400 improves raw dynamic but harms fused ranking, step 800 improves fused MRR by `0.006` with no regime below `-0.01`, and step 0 remains exact static. Assert step 800 is promoted and a best step of 0 emits `stage4_not_promoted`.

- [x] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-select-red --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:10:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_stage4_validation.py tests/test_stage4_quality_gate.py'
```

- [x] **Step 3: Implement release rules**

Validate step 0 and every 400 steps. Select maximum balanced fused macro MRR, breaking ties by lower regret then earlier step. Promote only if fused macro improves by at least `0.005`, no routing regime loses more than `0.01` from Stage2 static, regret falls, endpoints are exact, gradients are finite/nonzero, and Stage2 digests are unchanged.

- [x] **Step 4: Run GREEN and commit**

```bash
git add clstr/stage4_validation.py clstr/stage4_quality_gate.py tests/test_stage4_validation.py tests/test_stage4_quality_gate.py
git commit -m "feat: select stage4 by deployed cmc ranking"
```

### Task 6: Align inference, candidate recall, and final-chain loading

**Files:**
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/memory_candidate_recall.py`
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `clstr/native_benchmark_checkpoint_adapter.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_memory_candidate_recall.py`
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_native_benchmark_checkpoint_adapter.py`

- [x] **Step 1: Write failing deployment-contract tests**

Test that dynamic scores add candidates outside static top-M, final fusion ranks the union, alpha zero/one are exact, zero-history forces alpha zero, and a CMC delta cannot load on the wrong Stage2 digest.

- [x] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-infer-red --gres=gpu:1 --cpus-per-task=4 --mem=28G --time=00:15:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_current_state_route_eval.py tests/test_memory_candidate_recall.py tests/test_qwen_clstr_final_chain.py tests/test_native_benchmark_checkpoint_adapter.py'
```

- [x] **Step 3: Implement one shared CMC scoring path**

Expose a helper returning static logits, raw dynamic logits, CMC dynamic logits, alpha, fused logits, and candidate-union records. Use it in validation, ToolBench, Tau2, ToolSandbox, and ALFWorld concrete-action scoring. Preserve old `bounded` and `causal_gate` reliability modes only for historical checkpoint replay.

Implementation note: the shared path also pins the Stage4 training feature caps
(`16/256`) in the reliability identity, computes the CMC gate on the complete
legal candidate set before dynamic-extra union selection, and validates the
selected delta's recorded Stage2 full/fast router parent digests.

- [x] **Step 4: Run GREEN and commit**

```bash
git add clstr/current_state_route_eval.py clstr/memory_candidate_recall.py clstr/qwen_clstr_final_chain.py clstr/native_benchmark_checkpoint_adapter.py clstr/alfworld_eval.py tests/test_current_state_route_eval.py tests/test_memory_candidate_recall.py tests/test_qwen_clstr_final_chain.py tests/test_native_benchmark_checkpoint_adapter.py
git commit -m "feat: deploy exact endpoint cmc routing"
```

### Task 7: Repair Tau2 base and ToolSandbox paired protocols

**Files:**
- Modify: `clstr/tau2_route_eval.py`
- Modify: `scripts/run_tau2_full_clstr_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `scripts/run_toolsandbox_full_clstr_route_eval.py`
- Modify: `tests/test_tau2_route_eval.py`
- Modify: `tests/test_toolsandbox_route_eval.py`

- [x] **Step 1: Write failing Tau2 split tests**

```python
def test_tau2_base_split_filters_tasks_and_counts_refusals(tmp_path: Path) -> None:
    corpus = load_tau2_route_corpus(tmp_path, domains=["airline"], task_split="base")
    assert corpus.report["task_split"] == "base"
    assert corpus.report["tool_action_row_count"] == 2
    assert corpus.report["no_tool_row_count"] == 1
    assert corpus.source_rows[-1]["route_target"] == "STOP"
```

Add input-order-invariant split membership and missing split-ID failures.

- [x] **Step 2: Write failing ToolSandbox paired-mode tests**

Assert `checkpoint_faithful` restores the training prefix and appends 31 skills, `local_table_rebuild` constructs exactly 31 skills, both expose identical row candidate IDs/order and corrected replay hashes, and the report forbids using local rebuild for checkpoint selection.

- [x] **Step 3: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-protocol-red --gres=gpu:1 --cpus-per-task=4 --mem=28G --time=00:15:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py'
```

- [x] **Step 4: Implement explicit protocols**

Add Tau2 `task_split` choices `base`, `train`, `test`, `full`; default the paper launcher to `base`. Use `split_tasks.json` membership and map empty-action tasks to a STOP-only routing row rather than silently dropping them. Retain `full` for the 13,907-row stress report.

Add ToolSandbox `model_skill_pool_mode` choices `checkpoint_faithful` and `local_table_rebuild`; keep identical legal candidates/replay and write a paired comparison manifest.

- [x] **Step 5: Run GREEN and commit**

```bash
git add clstr/tau2_route_eval.py scripts/run_tau2_full_clstr_route_eval.py clstr/toolsandbox_route_eval.py scripts/run_toolsandbox_full_clstr_route_eval.py tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py
git commit -m "fix: align tau2 and toolsandbox paper protocols"
```

### Task 8: Update manifests, docs, and launchers

**Files:**
- Modify: `clstr/qwen_clstr_lineage.py`
- Modify: `clstr/qwen_clstr_multibench_submit.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_native_route_eval.sh`
- Modify: `scripts/run_alfworld_qwen_clstr_executor_gate.py`
- Modify: `scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh`
- Modify: `tests/test_qwen_clstr_lineage.py`
- Modify: `tests/test_qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`
- Modify: `finalwork/clstr_benchmark_results.md`

- [ ] **Step 1: Write failing identity tests**

Require manifests to pin `counterfactual_memory_calibration_v1`, parent Stage2 digest, feature schema, Tau2 split/digest, ToolSandbox pool mode, ALFWorld `unified_memory_concrete_action`, Qwen weight `1.0`, and CLSTR weight `0.25`.

- [ ] **Step 2: Verify RED through Slurm**

```bash
sbatch --wait --job-name=cmc-manifest-red --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:12:00 --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix && source scripts/sbatch/_clstr_gpu_env.sh && python -m pytest -q tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_training_launchers.py'
```

- [ ] **Step 3: Implement fail-closed identities and documentation rows**

Do not aggregate a run whose checkpoint chain, task split, candidate protocol, prompt, weights, or scorer mode differs. Keep prior safe-memory rows unchanged and append new CMC rows only after complete jobs.

- [ ] **Step 4: Run GREEN and commit**

```bash
git add clstr/qwen_clstr_lineage.py clstr/qwen_clstr_multibench_submit.py scripts/sbatch/run_qwen06_clstr_native_route_eval.sh scripts/run_alfworld_qwen_clstr_executor_gate.py scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_training_launchers.py finalwork/clstr_benchmark_results.md
git commit -m "chore: pin cmc training and benchmark identities"
```

### Task 9: Verify, smoke, train, and evaluate Qwen CMC

**Files:**
- Update after completed jobs: `finalwork/clstr_benchmark_results.md`
- Create under outputs: immutable CMC chain manifests and run registries.

- [ ] **Step 1: Run source-only verification**

Run `git diff --check`, focused `python -m py_compile` on changed Python modules, `bash -n` on changed launchers, and stale-symbol scans locally. Do not run pytest on the login/storage node.

- [ ] **Step 2: Run the affected test matrix through Slurm**

Submit focused suites first, then the complete affected Stage2/Stage4/evaluation suite. Require zero failures and record job IDs, runtimes, and exact test counts in the active progress file.

- [ ] **Step 3: Run Stage2 smoke and full training**

Use the selected Qwen Stage1 checkpoint, frozen Qwen cache, 1,200-step smoke decision point, then continue to 10,000 steps only if anchor/static/dynamic gates pass. Monitor at coarse intervals; stop on nonfinite loss, digest drift, missing replay, or throughput regression.

- [ ] **Step 4: Run Stage4 smoke and full training**

Start only from the selected anchored Stage2. Verify optimizer scope, zero adapter initialization, nonzero adapter/gate gradients, exact endpoints, and step-zero baseline. Train at most 3,000 steps and promote only a nonzero checkpoint satisfying Task 5.

- [ ] **Step 5: Run the benchmark matrix**

Run ToolBench final100, ToolSandbox checkpoint-faithful plus local rebuild, Tau2 official base plus 13,907 stress, and ALFWorld Qwen3-14B concrete-action executor. Preserve static/raw-dynamic/CMC-fused/equal-budget branches.

- [ ] **Step 6: Apply release gates**

Require ToolBench dynamic-extra recall above zero, ToolSandbox fused-minus-static MRR at least `-0.005`, full-stress three-benchmark macro at least `0.423038`, Tau2 base at least its frozen same-protocol pre-safe reference, and a nonzero selected Stage4 step. If any condition fails, release the anchored Stage2 endpoint and omit CMC from the paper main method.

- [ ] **Step 7: Final verification and handoff**

Run manifest aggregation, checkpoint digest verification, `git diff --check`, and working-tree review. Commit only verified source/docs; never commit generated model checkpoints.
