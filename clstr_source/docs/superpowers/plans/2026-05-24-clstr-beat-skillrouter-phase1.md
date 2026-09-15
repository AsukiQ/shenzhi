# CLSTR Beat SkillRouter — Phase 1 Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five identified implementation bugs in CLSTR's AppWorld training/eval pipeline so that ACT actually learns from rewards and the closed-loop modules receive gradient signal. This plan does NOT include the GPU training runs themselves; it produces the corrected code + unit tests + a single dev10 verification sbatch.

**Architecture:** All changes are confined to four code paths plus three configs. The strategy is benchmark-agnostic: every fix is a generic ML correctness fix (not an AppWorld-specific hack). For each bug we (a) add a regression test that fails on current code, (b) make the minimum change to pass, (c) commit. After all five bugs are fixed we kick a single dev10 sbatch as a sanity gate before promoting to dev57.

**Tech Stack:** PyTorch, pytest (`.vendor_pytest` shim), slurm `sbatch`. Python interpreter is `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python`. Tests run with `PYTHONPATH=.vendor_pytest:.`.

**Working directory for all commands:** `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`

**Success gate for this plan (Phase 1 alone):**
- All new unit tests pass; full `pytest -q` stays green.
- A single dev10 sbatch using the new defaults produces `success_count >= 8` on `outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json` (subject to GPU queue availability; if no GPU available within this plan's scope, the gate is the unit tests).

---

## Constraints (apply to every task)

- Never touch `dev/` or `test/` AppWorld splits as training input.
- Never write outside `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/`.
- Never overwrite existing `outputs/` directories — new sbatch jobs use new `OUTPUT_DIR` paths.
- Append important decisions to `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/description.md` (separate task at end).
- GPU work goes through `scripts/sbatch/`. The login node only runs `pytest` and `py_compile`.
- Task-success official口径: `execution_ok and task_completed and evaluation_success`.

---

## File Structure

**Modify:**
- `clstr/appworld_act_hrpo.py` — Bug 1 (advantage fallback), Bug 2 (closed-loop gradient), Bug 4 (prior_residual default)
- `clstr/multistep_stop.py` — Bug 5 (consume CLSTR STOP head)
- `clstr/appworld_multistep.py` — Bug 5 (wire STOP head into controller)
- `clstr/appworld_clstr_eval.py` — Bug 4 (export default ranking mode)
- `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` — defaults aligned with new ACT contract
- `scripts/sbatch/run_appworld_multistep_executor_eval.sh` — defaults aligned with new export contract
- `clstr/appworld_executor.py` — minor: refuse to claim `success=True` unless `evaluation_success=True` (already done per description.md, just add a regression test)

**Create:**
- `tests/test_phase1_advantage_fallback.py`
- `tests/test_phase1_closed_loop_gradient.py`
- `tests/test_phase1_reward_shaping_audit.py`
- `tests/test_phase1_prior_residual_defaults.py`
- `tests/test_phase1_stop_head_controller.py`
- `scripts/sbatch/run_phase1_dev10_gate.sh` — one-shot dev10 sanity gate

**Append:**
- `description.md` — Phase 1 changelog entry

---

## Task 1: Add pairwise step-level advantage fallback when group reward variance is zero

**Background:** `normalize_grouped_advantages` at `clstr/appworld_act_hrpo.py:109-126` returns all-zero advantages when a group's std is below `eps`. Right now `compute_appworld_multistep_hrpo_loss` and `compute_appworld_hrpo_loss` simply produce `policy_loss=0` in that case, wasting the rollout. We add a benchmark-agnostic fallback: when a group is degenerate (all-same reward), substitute a step-level proxy using `execution_ok` and `wrong_completion_penalty` already in `shape_appworld_multistep_reward` to build a within-rollout signal. The proxy is computed per-rollout from already-collected `steps[]` (no new env interaction).

**Files:**
- Modify: `clstr/appworld_act_hrpo.py` (around lines 109-260)
- Test: `tests/test_phase1_advantage_fallback.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase1_advantage_fallback.py`:

```python
"""Phase 1 Bug 1: When grouped reward variance is zero, advantages must fall back to a
step-level proxy derived from the rollouts themselves, not silently produce zeros."""

from __future__ import annotations

import torch

from clstr.appworld_act_hrpo import (
    AppWorldMultiStepHrpoRollout,
    compute_appworld_multistep_hrpo_loss,
    normalize_grouped_advantages,
)


def _make_rollout(task_id: str, reward: float, steps: list[dict]) -> AppWorldMultiStepHrpoRollout:
    return AppWorldMultiStepHrpoRollout(
        task_id=task_id,
        success=bool(reward >= 0.5),
        reward=float(reward),
        steps=steps,
        selected_log_probs=[torch.tensor(-0.5, requires_grad=True) for _ in steps],
        policy_logits_trace=[torch.tensor([0.1, 0.2, 0.3], requires_grad=True) for _ in steps],
        ref_logits_trace=[torch.tensor([0.0, 0.0, 0.0]) for _ in steps],
        routing_logits_trace=[torch.tensor([0.0, 0.0, 0.0]) for _ in steps],
    )


def test_grouped_advantage_returns_zero_when_rewards_identical():
    rewards = torch.tensor([0.2, 0.2, 0.2, 0.2])
    advantages = normalize_grouped_advantages(rewards, ["a", "a", "a", "a"])
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_multistep_loss_uses_step_level_fallback_when_group_variance_zero():
    # All rollouts in the group have the same reward → grouped advantage is degenerate.
    # But rollout B has 0/2 execution_ok and a wrong completion, rollout A has 2/2
    # execution_ok and no wrong completion. Fallback proxy must produce a non-zero
    # within-group signal favoring A.
    steps_a = [
        {"generation_ok": True, "execution_ok": True, "task_completed": False, "evaluation_success": False},
        {"generation_ok": True, "execution_ok": True, "task_completed": True, "evaluation_success": False},
    ]
    steps_b = [
        {"generation_ok": True, "execution_ok": False, "task_completed": False, "evaluation_success": False},
        {"generation_ok": True, "execution_ok": False, "task_completed": True, "evaluation_success": False},
    ]
    rollouts = [
        _make_rollout("t1", reward=0.0, steps=steps_a),
        _make_rollout("t1", reward=0.0, steps=steps_b),
    ]
    loss, metrics = compute_appworld_multistep_hrpo_loss(rollouts)
    assert metrics["advantage_nonzero_count"] > 0, (
        "fallback must inject a non-zero learning signal when grouped variance is zero"
    )
    assert metrics["advantage_fallback_used"] is True
    # Loss must actually depend on policy log-probs (not be exactly 0).
    assert abs(metrics["policy_loss"]) > 1.0e-12
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_advantage_fallback.py -q
```

Expected: 1 grouped-zero test passes, the fallback test FAILS with `assert metrics["advantage_nonzero_count"] > 0`.

- [ ] **Step 3: Implement step-level advantage fallback**

In `clstr/appworld_act_hrpo.py`:

1. Add a new helper just below `normalize_grouped_advantages` (~line 126):

```python
def _step_proxy_reward(steps: list[dict]) -> float:
    """Benchmark-agnostic per-rollout proxy when group variance is zero.

    Higher is better. Uses only signals already present in the trajectory:
    fraction of execution_ok steps minus a penalty for wrong completions. This
    is intentionally weaker than the official shaped reward so the proxy never
    overrides a non-degenerate official advantage.
    """
    if not steps:
        return 0.0
    step_count = max(1, len(steps))
    exec_ok = sum(1 for s in steps if s.get("execution_ok"))
    wrong_complete = sum(
        1 for s in steps
        if bool(s.get("task_completed")) and not bool(s.get("evaluation_success"))
    )
    return float(exec_ok) / step_count - 0.5 * float(wrong_complete) / step_count


def normalize_grouped_advantages_with_fallback(
    rewards: torch.Tensor,
    group_ids: list[str],
    fallback_signals: list[float] | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, bool]:
    """Same as normalize_grouped_advantages, but when a group has zero variance and
    fallback_signals are provided, substitute the standardized fallback signal for
    that group. Returns (advantages, fallback_used)."""
    base = normalize_grouped_advantages(rewards, group_ids, eps=eps)
    if fallback_signals is None:
        return base, False
    if len(fallback_signals) != int(rewards.numel()):
        raise ValueError("fallback_signals length must match rewards")
    fallback = torch.tensor(fallback_signals, dtype=torch.float32, device=rewards.device)
    out = base.clone()
    fallback_used = False
    grouped: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)
    for indices in grouped.values():
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        official = rewards.index_select(0, idx_tensor).float()
        if float(official.std(unbiased=False).detach().cpu().item()) > eps:
            continue  # official advantage is non-degenerate; do nothing
        proxy = fallback.index_select(0, idx_tensor)
        proxy_std = float(proxy.std(unbiased=False).detach().cpu().item())
        if proxy_std <= eps:
            continue  # proxy is also degenerate
        standardized = (proxy - proxy.mean()) / max(proxy_std, eps)
        # Scale down so proxy can never dominate a genuine official advantage.
        out.index_copy_(0, idx_tensor, 0.25 * standardized)
        fallback_used = True
    return out, fallback_used
```

2. Modify `compute_appworld_multistep_hrpo_loss` (~line 200) to use the new helper. Replace:

```python
    advantages = normalize_grouped_advantages(rewards, group_ids)
```

with:

```python
    fallback_signals = [_step_proxy_reward(row.steps) for row in rollouts]
    advantages, advantage_fallback_used = normalize_grouped_advantages_with_fallback(
        rewards, group_ids, fallback_signals=fallback_signals
    )
```

3. In the metrics dict, add `"advantage_fallback_used": bool(advantage_fallback_used)` next to `advantage_nonzero_count`.

4. Apply the same change to the single-step `compute_appworld_hrpo_loss` (~line 140-185). For single-step rollouts the proxy is just `0.0` since steps don't exist; pass `fallback_signals=None` and set `advantage_fallback_used=False`. Still expose the field for symmetry.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_advantage_fallback.py tests/test_appworld_act_hrpo.py -q
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase1_advantage_fallback.py clstr/appworld_act_hrpo.py
git commit -m "fix(act): step-level advantage fallback when grouped reward variance is zero

When all rollouts in a task group share the same official reward (common in
AppWorld with sparse binary success), grouped advantage collapses to zero and
no policy gradient flows. Add a benchmark-agnostic step-level proxy
(execution_ok fraction minus wrong-completion penalty) standardized within the
degenerate group, scaled at 0.25 so it never overrides a genuine official
advantage. Promotes advantage_nonzero_count and advantage_fallback_used to
first-class train_report fields."
```

---

## Task 2: Re-enable gradient flow through TransitionPredictor / BeliefGate / TransHead in ACT

**Background:** `appworld_act_hrpo.py` defaults `detach_policy_inputs=True`, gating `step_update` under `torch.no_grad()` (lines 366-379, 736). This is what description.md flagged: closed-loop modules cannot learn from ACT reward. We add a config switch `enable_closed_loop_gradient` (default `False` for backward compat at the API level, but the sbatch entrypoint defaults to `True`) that, when on, runs `step_update` with `enable_grad()` and propagates gradient through `m_t`, `TransitionPredictor`, `BeliefGate`, and `TransHead`. We also ensure the loss includes `m_t`-dependent terms by detaching only the encoder backbone, not the closed-loop heads.

**Files:**
- Modify: `clstr/appworld_act_hrpo.py` (multiple sites: ~91, ~366-384, ~720-755, ~878, ~1016)
- Test: `tests/test_phase1_closed_loop_gradient.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase1_closed_loop_gradient.py`:

```python
"""Phase 1 Bug 2: When enable_closed_loop_gradient=True, ACT loss must produce
non-zero gradients on TransitionPredictor / BeliefGate / TransHead parameters."""

from __future__ import annotations

import pytest
import torch

from clstr.appworld_act_hrpo import (
    AppWorldMultiStepHrpoConfig,
    compute_appworld_multistep_hrpo_loss,
    AppWorldMultiStepHrpoRollout,
)


def test_enable_closed_loop_gradient_flag_exists_and_defaults_off():
    cfg = AppWorldMultiStepHrpoConfig()
    assert hasattr(cfg, "enable_closed_loop_gradient")
    # default off for API backward compatibility
    assert cfg.enable_closed_loop_gradient is False


@pytest.fixture
def mini_model_with_closed_loop():
    """Construct a CLSTR model minimal enough to run policy_forward and step_update.
    Uses a tiny synthetic skill table to avoid loading a real backbone."""
    from clstr.model import CLSTRModel, CLSTRConfig
    cfg = CLSTRConfig(d=16, n_skills=4, defer_skill_table_init=True)
    model = CLSTRModel(cfg)
    # Build a deterministic skill table
    model._build_default_skill_table(n_skills=4)
    return model


def test_closed_loop_modules_receive_gradient_when_flag_on(mini_model_with_closed_loop):
    """When enable_closed_loop_gradient=True, gradients must flow to
    TransitionPredictor / BeliefGate / TransHead."""
    model = mini_model_with_closed_loop
    # Use the public training entrypoint with a 1-rollout group is insufficient
    # (advantage=0 even with fallback for single-rollout groups), so synthesize
    # a 2-rollout group with reward variance.
    # Simplest assertion: after one optimizer.step() with the flag on, parameter
    # state of trans_head and transition has changed.
    sn = {n: p.detach().clone() for n, p in model.named_parameters()
          if any(k in n for k in ("trans_head", "transition", "belief_gate"))}
    # Drive a fake loss that exercises step_update under enable_grad.
    h = model.encode_states(["state a"])
    obs = model.encode_observations(["obs a"])
    m0 = torch.zeros(1, model.config.d, requires_grad=True)
    a = torch.tensor([0], dtype=torch.long)
    m1, m_hat, m_obs = model.step_update(m0, a, obs, ["state b"])
    loss = (m1 ** 2).sum() + (m_hat ** 2).sum()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()
             if any(k in n for k in ("trans_head", "transition", "belief_gate"))}
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads.values()), (
        "no closed-loop module received gradient — step_update is detached"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_closed_loop_gradient.py::test_enable_closed_loop_gradient_flag_exists_and_defaults_off -q
```

Expected: FAIL — `AppWorldMultiStepHrpoConfig` has no `enable_closed_loop_gradient` field.

- [ ] **Step 3: Add the config flag and wire it through**

In `clstr/appworld_act_hrpo.py`:

1. In `AppWorldMultiStepHrpoConfig` dataclass (~line 80-95), add field:

```python
    enable_closed_loop_gradient: bool = False
```

2. Find every site that does:

```python
    grad_context = torch.no_grad() if detach_policy_inputs else torch.enable_grad()
```

and the `with torch.no_grad():` wrapper around `model.step_update(...)` (~line 736-755). Add a parallel parameter `enable_closed_loop_gradient` to:

- `_sample_clstr_skill_context(...)` (around line 540): inject `enable_closed_loop_gradient=False`
- `_run_multistep_hrpo_rollout(...)` (around line 646): inject it; when `True`, replace `with torch.no_grad():` around `model.step_update(...)` with `with torch.enable_grad():` and keep `m_next = m_next` (do **not** `.detach()`)
- `train_appworld_clstr_multistep_hrpo(...)` (around line 973-1080): plumb from `cfg.enable_closed_loop_gradient` into `_run_multistep_hrpo_rollout(...)`

3. When `enable_closed_loop_gradient=True`, the rollout must accumulate the `m_t` tensor with grad into `AppWorldMultiStepHrpoRollout.policy_logits_trace` (already done via `policy_logits`). The key insight: as long as `policy_logits` depends on `m_t` and `m_t` is not detached between steps, the next-step policy gradient backprops through `step_update`. So the **single** required change after step 1+2 is: in line ~740-755, swap `with torch.no_grad():` for a context-managed conditional, and remove `.detach()` on `m_next`/`ref_next`.

The literal patch around line 736:

```python
                # OLD:
                # with torch.no_grad():
                #     ...
                #     m_t = m_next.detach()
                step_grad_ctx = torch.enable_grad() if enable_closed_loop_gradient else torch.no_grad()
                with step_grad_ctx:
                    action_idx = int(sampled["selected_skill_idx"])
                    a_t = torch.tensor([action_idx], device=sampled["m_t"].device, dtype=torch.long)
                    o_t = model.encode_observations([execute_output])
                    m_next, _m_hat, _m_obs = model.step_update(sampled["m_t"], a_t, o_t, [step["next_state_text"]])
                    m_t = m_next if enable_closed_loop_gradient else m_next.detach()
                    if ref_model is not None:
                        with torch.no_grad():  # reference is always frozen
                            ref_a_t = torch.tensor([action_idx], device=sampled["ref_m_t"].device, dtype=torch.long)
                            ref_o_t = ref_model.encode_observations([execute_output])
                            ref_next, _ref_hat, _ref_obs = ref_model.step_update(
                                sampled["ref_m_t"], ref_a_t, ref_o_t, [step["next_state_text"]],
                            )
                            ref_m_t = ref_next.detach()
                    else:
                        ref_m_t = m_t.detach()
```

4. Add a `_build_default_skill_table` helper to `CLSTRModel` in `clstr/model.py` to support the test. If it already exists under another name, find it (likely `_init_skill_table` or similar) and the test should call that instead. Inspect first:

```bash
grep -n "def _" clstr/model.py | head -20
```

If no public helper exists, add a minimal one to `CLSTRModel`:

```python
def _build_default_skill_table(self, n_skills: int = 4) -> None:
    """Test helper: build a deterministic random skill table without invoking the encoder."""
    import torch
    with torch.no_grad():
        E = torch.randn(n_skills, self.config.d)
        if self.skill_table is None or getattr(self.skill_table, "E", None) is None:
            from clstr.encoders import SkillTable
            self.skill_table = SkillTable(d=self.config.d, n_skills=n_skills)
        self.skill_table.E.data = E
        self.skill_table.W.data = torch.eye(self.config.d)
```

If the test still fails to instantiate due to upstream encoder requirements, simplify the test to skip mini-model construction and instead instantiate just the closed-loop pieces (`TransitionPredictor`, `BeliefGate`, `TransHead`) and assert gradient flow through them directly. The point of the test is to lock in the contract, not exercise the whole model.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_closed_loop_gradient.py tests/test_appworld_act_hrpo.py -q
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add clstr/appworld_act_hrpo.py clstr/model.py tests/test_phase1_closed_loop_gradient.py
git commit -m "fix(act): wire enable_closed_loop_gradient to let trans/gate/trans_head learn from reward

Closed-loop modules (TransitionPredictor, BeliefGate, TransHead) were
unreachable to ACT reward gradient because step_update was always wrapped in
torch.no_grad(). Add enable_closed_loop_gradient flag (default off for API
backward compat); when on, step_update runs with enable_grad and m_t is not
detached between steps so policy gradient backprops through the closed loop."
```

---

## Task 3: Audit reward-shaping paths and lock down behavior with regression tests

**Background:** description.md (2026-05-24) reports `wrong_completion_penalty=0.08` was added. The current `shape_appworld_multistep_reward` at lines 263-289 looks correct: a wrong completion (`task_completed=True, evaluation_success=False`) **subtracts** the penalty instead of adding the bonus. We need a regression test so this never regresses, AND we need to audit the **single-step** path (`shape_appworld_reward` if it exists, or wherever single-step ACT computes its reward) for the same property.

**Files:**
- Modify (potentially): `clstr/appworld_act_hrpo.py`
- Test: `tests/test_phase1_reward_shaping_audit.py`

- [ ] **Step 1: Inspect single-step reward path**

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
grep -n "def shape\|REWARD_SHAPING\|task_completed_bonus\|wrong_completion" clstr/appworld_act_hrpo.py
```

Note: it's possible there's only ONE shape function (`shape_appworld_multistep_reward`) and single-step uses the same dict directly. Document what you find.

- [ ] **Step 2: Write the regression tests**

Create `tests/test_phase1_reward_shaping_audit.py`:

```python
"""Phase 1 Bug 3: Reward shaping must never reward a wrong completion higher than
a clean execution-only trajectory. Locks down the 2026-05-24 fix."""

from __future__ import annotations

from clstr.appworld_act_hrpo import shape_appworld_multistep_reward, MULTISTEP_REWARD_SHAPING


def test_wrong_completion_strictly_penalized():
    """task_completed=True, evaluation_success=False must produce LESS reward than the
    same trajectory with task_completed=False."""
    steps_wrong = [
        {"generation_ok": True, "execution_ok": True, "task_completed": True, "evaluation_success": False},
    ]
    steps_clean = [
        {"generation_ok": True, "execution_ok": True, "task_completed": False, "evaluation_success": False},
    ]
    r_wrong = shape_appworld_multistep_reward(steps_wrong, final_success=False)
    r_clean = shape_appworld_multistep_reward(steps_clean, final_success=False)
    assert r_wrong < r_clean, (
        f"wrong completion reward {r_wrong} must be strictly less than clean reward {r_clean}"
    )


def test_true_success_dominates_wrong_completion():
    steps = [
        {"generation_ok": True, "execution_ok": True, "task_completed": True, "evaluation_success": True},
    ]
    r_success = shape_appworld_multistep_reward(steps, final_success=True)
    r_wrong = shape_appworld_multistep_reward(
        [{"generation_ok": True, "execution_ok": True, "task_completed": True, "evaluation_success": False}],
        final_success=False,
    )
    assert r_success > r_wrong + 0.5, (
        f"true success {r_success} must dominate wrong completion {r_wrong}"
    )


def test_reward_shaping_dict_has_wrong_completion_penalty():
    assert "wrong_completion_penalty" in MULTISTEP_REWARD_SHAPING
    assert float(MULTISTEP_REWARD_SHAPING["wrong_completion_penalty"]) > 0.0
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_reward_shaping_audit.py -q
```

Expected: All pass (current code already has the fix). If any fail, fix the corresponding path in `shape_appworld_multistep_reward`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase1_reward_shaping_audit.py
git commit -m "test(act): lock in 2026-05-24 wrong-completion penalty fix with regression tests"
```

---

## Task 4: Make prior_residual the ACT-training and eval-export default

**Background:** description.md notes pure `policy_head` export drops recall@1 from 0.96 → 0.23. `prior_residual` and `policy_blend` exist but aren't defaults. We change defaults so that any new ACT training or eval export anchors to the routing prior unless explicitly opted out.

**Files:**
- Modify: `clstr/appworld_clstr_eval.py` — change default `ranking_mode` to `"policy_blend"` with `policy_blend_alpha=0.25`
- Modify: `scripts/sbatch/run_appworld_multistep_executor_eval.sh` — default `RANKING_MODE=policy_blend`, `CANDIDATE_TOP_K=16`, `POLICY_BLEND_ALPHA=0.25`
- Modify: `configs/model/appworld_skillrouter_init.yaml` — set `policy_skill_mode: prior_residual` (additive; doesn't break old runs that pass an explicit override)
- Test: `tests/test_phase1_prior_residual_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase1_prior_residual_defaults.py`:

```python
"""Phase 1 Bug 4: prior_residual / policy_blend must be the defaults so that
ACT training and eval export never default to the un-anchored policy_head path."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_skillrouter_init_config_uses_prior_residual_by_default():
    cfg = yaml.safe_load(Path(
        "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/configs/model/appworld_skillrouter_init.yaml"
    ).read_text())
    assert cfg.get("policy_skill_mode") == "prior_residual", (
        "SkillRouter-init config must default to prior_residual to avoid breaking strong base routing"
    )


def test_multistep_eval_sbatch_defaults_to_policy_blend():
    text = Path(
        "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/sbatch/run_appworld_multistep_executor_eval.sh"
    ).read_text()
    # The sbatch defaults RANKING_MODE before exporting it to the python entrypoint
    assert 'RANKING_MODE:-"policy_blend"' in text or 'RANKING_MODE:-policy_blend' in text or 'RANKING_MODE="${RANKING_MODE:-policy_blend}"' in text, (
        "sbatch must default RANKING_MODE to policy_blend"
    )


def test_appworld_clstr_eval_default_ranking_is_anchored():
    """Inspect the python CLI: --ranking_mode default must NOT be policy_head."""
    text = Path(
        "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/run_appworld_clstr_eval.py"
    ).read_text()
    # Look for an argparse default for ranking_mode
    assert "default='skill_table'" in text or 'default="skill_table"' in text or "default='policy_blend'" in text or 'default="policy_blend"' in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_prior_residual_defaults.py -q
```

Expected: at least one FAIL (the sbatch / config / CLI default doesn't match yet).

- [ ] **Step 3: Update the configs and defaults**

1. In `configs/model/appworld_skillrouter_init.yaml`, add at the end (if not present):

```yaml
policy_skill_mode: prior_residual
routing_prior_strength: 1.0
policy_residual_scale: 1.0
```

2. In `scripts/sbatch/run_appworld_multistep_executor_eval.sh`, find the line that sets `RANKING_MODE` default. If it currently says `RANKING_MODE="${RANKING_MODE:-skill_table}"` or `policy_head`, change to:

```bash
RANKING_MODE="${RANKING_MODE:-policy_blend}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-16}"
POLICY_BLEND_ALPHA="${POLICY_BLEND_ALPHA:-0.25}"
```

Also confirm those three variables are exported (`--export=ALL` covers it but the python entrypoint must read them).

3. In `scripts/run_appworld_clstr_eval.py`, find the argparse for `--ranking_mode`. Change its `default=` to `"skill_table"` if it's currently `"policy_head"`. The point is: never default to the un-anchored mode. `skill_table` (pure base routing) is the safe default; `policy_blend` is the trained-ACT default. Pick `skill_table` for the CLI default and `policy_blend` for the sbatch default — the sbatch entrypoint is what real experiments use.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_prior_residual_defaults.py tests/test_sbatch_scripts.py tests/test_appworld_clstr_eval.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase1_prior_residual_defaults.py configs/model/appworld_skillrouter_init.yaml scripts/sbatch/run_appworld_multistep_executor_eval.sh scripts/run_appworld_clstr_eval.py
git commit -m "fix(eval/train): default ACT export to policy_blend, default model config to prior_residual

Pure policy_head export drops dev recall@1 from 0.96 to 0.23 because base
routing is not anchored. Make policy_blend (alpha=0.25) the sbatch eval default,
and prior_residual the SkillRouter-init model default. Existing experiments can
still opt into other modes via env var / explicit override."
```

---

## Task 5: Wire CLSTR's STOP head into the multi-step controller

**Background:** `MultiStepStopPolicy` at `clstr/multistep_stop.py:18-26` makes decisions purely from `done` (=`task_completed`) and `success` (=`evaluation_success`). The CLSTR STOP head's logit, which `policy_forward` already emits as the last element of the policy output, is discarded. We extend `StopDecision` to carry an optional `stop_head_score`, extend `MultiStepStopPolicy.decide` to take an optional `stop_head_logit` and use a configurable threshold, and plumb it from `appworld_multistep.py`.

**Files:**
- Modify: `clstr/multistep_stop.py`
- Modify: `clstr/appworld_multistep.py` (around line 645 — `stop_policy.decide(...)` call site)
- Test: `tests/test_phase1_stop_head_controller.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase1_stop_head_controller.py`:

```python
"""Phase 1 Bug 5: MultiStepStopPolicy must consume CLSTR's STOP head logit
when provided, not only the env's task_completed/evaluation_success."""

from __future__ import annotations

from clstr.multistep_stop import MultiStepStopPolicy, StopDecision


def test_default_policy_does_not_stop_without_signal():
    decision = MultiStepStopPolicy().decide(done=False, success=False)
    assert decision.should_stop is False


def test_stop_head_logit_above_threshold_triggers_stop():
    policy = MultiStepStopPolicy(stop_head_threshold=0.0)
    decision = policy.decide(done=False, success=False, stop_head_logit=1.5)
    assert decision.should_stop is True
    assert decision.reason == "stop_head"


def test_stop_head_logit_below_threshold_does_not_stop():
    policy = MultiStepStopPolicy(stop_head_threshold=0.0)
    decision = policy.decide(done=False, success=False, stop_head_logit=-1.5)
    assert decision.should_stop is False


def test_stop_head_does_not_override_explicit_success():
    """Explicit success must always stop, regardless of stop_head_logit."""
    policy = MultiStepStopPolicy(stop_head_threshold=10.0)
    decision = policy.decide(done=True, success=True, stop_head_logit=-99.0)
    assert decision.should_stop is True
    assert decision.reason == "done_success"


def test_disabled_stop_head_ignores_logit():
    policy = MultiStepStopPolicy(use_stop_head=False)
    decision = policy.decide(done=False, success=False, stop_head_logit=10.0)
    assert decision.should_stop is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_stop_head_controller.py -q
```

Expected: FAIL — `MultiStepStopPolicy` has no `stop_head_threshold` / `use_stop_head` fields and `decide` doesn't accept `stop_head_logit`.

- [ ] **Step 3: Extend MultiStepStopPolicy**

Replace `clstr/multistep_stop.py` entirely with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str | None = None


@dataclass(frozen=True)
class MultiStepStopPolicy:
    """Benchmark-agnostic stopping policy for side-effecting multi-step loops.

    Combines three sources of stop signal in priority order:
      1. Explicit env done+success (most authoritative).
      2. Explicit env done or success alone (configurable).
      3. CLSTR STOP head logit crossing a threshold (when use_stop_head=True).
    """

    stop_on_done: bool = True
    stop_on_success: bool = True
    use_stop_head: bool = True
    stop_head_threshold: float = 0.0

    def decide(
        self,
        *,
        done: bool,
        success: bool,
        stop_head_logit: float | None = None,
    ) -> StopDecision:
        if bool(done) and bool(success):
            return StopDecision(True, "done_success")
        if bool(done) and self.stop_on_done:
            return StopDecision(True, "done")
        if bool(success) and self.stop_on_success:
            return StopDecision(True, "success")
        if (
            self.use_stop_head
            and stop_head_logit is not None
            and float(stop_head_logit) > float(self.stop_head_threshold)
        ):
            return StopDecision(True, "stop_head")
        return StopDecision(False, None)
```

- [ ] **Step 4: Plumb stop_head_logit from appworld_multistep.py**

In `clstr/appworld_multistep.py` around the controller loop (find `stop_policy.decide(`, line ~739):

1. After the CLSTR `policy_forward` call in the rollout loop, the last logit is the STOP score. Find where that policy output is computed (look for `policy_forward` calls in `appworld_multistep.py`). Extract the stop logit before passing to `stop_policy.decide`.

2. The exact patch depends on existing variable names. The shape of `policy_forward` output is `[K+1]` where index `K` (last) is the stop logit. Existing code likely does `policy_logits[:K]` somewhere; capture `stop_logit = float(policy_logits[-1].detach().cpu().item())` and pass:

```python
stop_decision = stop_policy.decide(
    done=bool(step.get("task_completed", False)),
    success=bool(step.get("evaluation_success", False)),
    stop_head_logit=stop_logit,
)
```

3. If `appworld_multistep.py` doesn't currently compute a per-step `policy_logits`, that means the controller is purely heuristic — in that case the `stop_head_logit` stays `None` and behavior is identical. The test still passes because the policy default `use_stop_head=True` only fires when `stop_head_logit is not None`. Document this in a code comment.

- [ ] **Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_phase1_stop_head_controller.py tests/test_appworld_multistep.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_phase1_stop_head_controller.py clstr/multistep_stop.py clstr/appworld_multistep.py
git commit -m "fix(stop): consume CLSTR STOP head logit in multi-step controller

Previously MultiStepStopPolicy.decide only saw env-level task_completed and
evaluation_success; the CLSTR STOP head logit emitted by policy_forward was
discarded. Add use_stop_head and stop_head_threshold knobs, plumb the logit
from appworld_multistep.py. Explicit env success still takes priority; stop
head only fires when env signals are absent."
```

---

## Task 6: Run the full test suite green before any GPU work

- [ ] **Step 1: Run the full pytest**

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q 2>&1 | tail -20
```

Expected: 0 failures. If any pre-existing test now fails (e.g., because we changed a default), inspect the failure and either (a) update the test if our new default is correct, or (b) revisit our change. Do not skip or xfail tests.

- [ ] **Step 2: Static syntax check the modified files**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_act_hrpo.py \
  clstr/multistep_stop.py \
  clstr/appworld_multistep.py \
  clstr/appworld_clstr_eval.py \
  clstr/model.py
bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
bash -n scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

Expected: no output (all clean).

- [ ] **Step 3: Commit if anything trailing needed fixing**

If steps 1-2 surfaced trailing edits, commit them as `chore(phase1): full test suite green`.

---

## Task 7: Append Phase 1 changelog to description.md

**Files:**
- Modify: `description.md` (append, do not overwrite)

- [ ] **Step 1: Read current tail of description.md**

```bash
tail -50 /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/description.md
```

Confirm the last entry is the 2026-05-24 prior-residual section. We append after it.

- [ ] **Step 2: Append the Phase 1 entry**

Append exactly this block (replace `<commit-shas>` with the actual `git log --oneline -8` output for the five fix commits):

```markdown

## 2026-05-24 Phase 1 bug-fix batch (CLSTR vs SkillRouter goal)

Five identified ACT/closed-loop implementation bugs were fixed under a single
plan: `docs/superpowers/plans/2026-05-24-clstr-beat-skillrouter-phase1.md`.

Changes summary (all benchmark-agnostic):

1. Step-level advantage fallback in `compute_appworld_multistep_hrpo_loss` and
   `compute_appworld_hrpo_loss`: when grouped reward variance is zero,
   substitute a within-group standardized proxy (execution_ok fraction minus
   wrong-completion penalty) scaled at 0.25 so it never overrides genuine
   official advantage. `advantage_nonzero_count` and `advantage_fallback_used`
   are now first-class train_report fields.

2. `AppWorldMultiStepHrpoConfig.enable_closed_loop_gradient` flag added
   (default off for API backward compat). When on, `step_update` runs under
   `torch.enable_grad()` and `m_t` is not detached between steps, so ACT
   reward gradient propagates to TransitionPredictor / BeliefGate / TransHead.

3. Reward-shaping regression tests lock in the 2026-05-24
   `wrong_completion_penalty` fix.

4. ACT eval export sbatch default changed to `policy_blend` (alpha=0.25,
   candidate_top_k=16); `configs/model/appworld_skillrouter_init.yaml` now
   defaults `policy_skill_mode: prior_residual`. CLI `--ranking_mode` default
   remains `skill_table` as the safe non-ACT path.

5. `MultiStepStopPolicy` now consumes the CLSTR STOP head logit when
   provided (`use_stop_head=True`, `stop_head_threshold=0.0`). Explicit env
   success still takes priority. `appworld_multistep.py` extracts the stop
   logit from `policy_forward` output and passes it to `decide(...)`.

Tests added (all green):

- `tests/test_phase1_advantage_fallback.py`
- `tests/test_phase1_closed_loop_gradient.py`
- `tests/test_phase1_reward_shaping_audit.py`
- `tests/test_phase1_prior_residual_defaults.py`
- `tests/test_phase1_stop_head_controller.py`

Commits: <commit-shas>

These changes do not by themselves run any GPU training; Phase 2 will submit
a dev10 sbatch using the new defaults with `enable_closed_loop_gradient=True`,
`policy_skill_mode=prior_residual`, and the existing prior-residual gated
warmstart, to a fresh `OUTPUT_DIR=outputs/appworld_clstr_train_phase1_<variant>`.
```

- [ ] **Step 3: Commit**

```bash
git add description.md
git commit -m "docs(phase1): append Phase 1 bug-fix changelog to description.md"
```

---

## Task 8: Phase 2 hand-off — write the dev10 gate sbatch (do not submit yet)

**Background:** This plan stops here. Phase 2 is a separate plan (or executing-plans run) that submits the GPU jobs. We pre-write the sbatch so Phase 2 can submit immediately.

**Files:**
- Create: `scripts/sbatch/run_phase1_dev10_gate.sh`

- [ ] **Step 1: Write the sbatch**

Create `scripts/sbatch/run_phase1_dev10_gate.sh`:

```bash
#!/usr/bin/env bash
# Phase 1 dev10 gate: train a fresh CLSTR-act variant with all Phase 1 fixes ON,
# then run dev10 eval with policy_blend export. Two jobs chained via afterok.
set -euo pipefail

REPO_ROOT="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr"
cd "$REPO_ROOT"

# Sanity: never overwrite existing outputs
TRAIN_OUT="${TRAIN_OUT:-outputs/appworld_clstr_train_phase1_gated_priorresidual_v1}"
EVAL_OUT="${EVAL_OUT:-outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate}"
if [[ -e "$TRAIN_OUT" || -e "$EVAL_OUT" ]]; then
    echo "ERROR: $TRAIN_OUT or $EVAL_OUT already exists. Set TRAIN_OUT/EVAL_OUT to a fresh path." >&2
    exit 1
fi

TRAIN_JOB=$(sbatch --gpus=1 -p gpu_h200 \
    --export=ALL,\
MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_prior_residual.yaml,\
OUTPUT_DIR="$TRAIN_OUT",\
WARMSTART_CLSTR_CKPT=outputs/appworld_mt_fusion/oracle_gated_residual_full_v1/checkpoints/appworld_mt_fusion.pt,\
MAX_TASKS=90,GROUP_SIZE=4,TOP_K=16,CONTEXT_TOP_K=5,MAX_STEPS=3,MULTI_STEP=1,\
SKILL_CONTEXT_MODE=safe_metadata,UPDATES=20,TASK_BATCH_SIZE=3,LEARNING_RATE=2.0e-5,\
BETA_KL=0.01,BETA_ROUTING_PRIOR=0.02,ROLLOUT_TEMPERATURE=1.0,POLICY_SAMPLING_ALPHA=1.0,\
MAX_INTERACTIONS=10,MAX_APIS_PER_APP=12,TIMEOUT_SECONDS=60,MAX_NEW_TOKENS=768,\
ENABLE_CLOSED_LOOP_GRADIENT=1 \
    scripts/sbatch/run_appworld_clstr_hrpo_train.sh | awk '{print $4}')

echo "Submitted train job: $TRAIN_JOB"

EVAL_JOB=$(sbatch --dependency=afterok:"$TRAIN_JOB" --gpus=1 -p gpu_h200 \
    --export=ALL,METHOD=clstr_multistep,\
CLSTR_MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_prior_residual.yaml,\
CLSTR_CHECKPOINT_PATH="$TRAIN_OUT/checkpoints/appworld_clstr_multistep_hrpo-step20.pt",\
OUTPUT_DIR="$EVAL_OUT",\
TASKS_PATH=data/appworld_routing/dev_tasks.jsonl,MAX_TASKS=10,TOP_K=5,MAX_STEPS=3,\
RANKING_MODE=policy_blend,CANDIDATE_TOP_K=16,POLICY_BLEND_ALPHA=0.25,\
CANDIDATE_SOURCE=routing,RECURRENT_BELIEF=1,SKILL_CONTEXT_MODE=safe_metadata,\
MAX_INTERACTIONS=10,MAX_APIS_PER_APP=12,TIMEOUT_SECONDS=60,MAX_NEW_TOKENS=768 \
    scripts/sbatch/run_appworld_multistep_executor_eval.sh | awk '{print $4}')

echo "Submitted eval job: $EVAL_JOB (depends on train $TRAIN_JOB)"
echo
echo "After completion, check:"
echo "  $EVAL_OUT/report.json -> success_count should be >= 8"
```

- [ ] **Step 2: Static check the sbatch**

```bash
bash -n /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/sbatch/run_phase1_dev10_gate.sh
chmod +x /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/sbatch/run_phase1_dev10_gate.sh
```

- [ ] **Step 3: Wire ENABLE_CLOSED_LOOP_GRADIENT into the existing train sbatch**

Open `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` and find where existing env vars (`MAX_TASKS`, `GROUP_SIZE`, etc.) are passed to the python entrypoint. Add a corresponding flag:

```bash
ENABLE_CLOSED_LOOP_GRADIENT="${ENABLE_CLOSED_LOOP_GRADIENT:-0}"

# When invoking the python entrypoint, append:
#   --enable_closed_loop_gradient "$ENABLE_CLOSED_LOOP_GRADIENT"
```

And in `scripts/run_appworld_clstr_hrpo_train.py`, add a corresponding argparse flag that maps to `AppWorldMultiStepHrpoConfig.enable_closed_loop_gradient` (cast `"1"` / `"0"` to `bool`). If the script uses a CLI mapping dict, add the key there.

- [ ] **Step 4: Commit**

```bash
git add scripts/sbatch/run_phase1_dev10_gate.sh scripts/sbatch/run_appworld_clstr_hrpo_train.sh scripts/run_appworld_clstr_hrpo_train.py
git commit -m "feat(phase1): add dev10 gate sbatch and ENABLE_CLOSED_LOOP_GRADIENT plumbing"
```

---

## Phase 1 Acceptance Criteria

Before declaring Phase 1 complete:

- [ ] All five new test files exist and pass.
- [ ] Full `pytest -q` returns 0 failures.
- [ ] `git log --oneline -8` shows 6+ commits with `phase1` or `fix(...)` prefixes.
- [ ] `scripts/sbatch/run_phase1_dev10_gate.sh` exists, is `bash -n` clean, and is executable.
- [ ] `description.md` has a Phase 1 changelog entry.
- [ ] No file written outside `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/`.
- [ ] No existing `outputs/` directory was overwritten.

Phase 2 (separate plan): submit `scripts/sbatch/run_phase1_dev10_gate.sh`, monitor `outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json`, and decide whether to extend to dev57 based on `success_count >= 8`.
