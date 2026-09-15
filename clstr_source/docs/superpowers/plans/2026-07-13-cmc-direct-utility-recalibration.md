# CMC Direct-Utility Gate Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit only the Qwen-0.6B CLSTR CMC candidate-utility gate with explicit counterfactual utility supervision, deploy it without changing the selected Stage2 or Stage4 residual endpoint, reuse the canonical Qwen3-14B ALFWorld control with provenance, and rerun the SR/ToolRex-aligned benchmark chain only if the predeclared audit and validation gates pass.

**Architecture:** The selected step-3000 CMC validation route manifest is the sole calibration source; its 1,024 rows are split by trajectory into calibration-train and calibration-dev, while the four-row gate snapshot is rejected. Frozen static and CMC-dynamic logits produce direct utility targets for a reset eleven-feature gate; only the gate's linear weight and bias are optimized. Deployment keeps runtime mode `cmc` so the Stage4 residual adapter and dynamic-extra candidate union remain active, but an audit-bound external gate may override the checkpoint-native alpha predictor. ALFWorld imports immutable Qwen-only artifacts through a self-hashed provenance manifest and runs only Qwen+CLSTR.

**Tech Stack:** Python 3, PyTorch, JSON/JSONL route manifests, existing Qwen CLSTR final-chain and multibench modules, pytest through Slurm, Bash Slurm launchers.

---

## File structure

- Modify `scripts/audit_clstr_memory_utility_oracle.py`: implement the exact direct-utility separability thresholds and report source-level selector MRR/regret.
- Modify `tests/test_memory_utility_oracle_audit.py`: cover the approved four audit conditions, selected validation-manifest identity, and four-row rejection.
- Modify `clstr/memory_utility_gate_train.py`: add detached utility targets, confidence-weighted BCE, source-balanced fused/no-regret losses, CMC-schema binding, and gate-only promotion.
- Modify `scripts/train_clstr_memory_utility_gate.py`: expose the fixed calibration hyperparameters and consume the selected validation records as the train/dev source.
- Modify `tests/test_memory_utility_gate_train.py`: verify targets, weighting, trainable scope, checkpoint identity, and fail-closed promotion.
- Modify `clstr/current_state_route_eval.py`: keep the CMC residual endpoint while allowing an external gate to replace only alpha.
- Modify `clstr/alfworld_eval.py`: apply the same CMC-plus-external-gate rule to admissible and concrete-action scorers.
- Modify `clstr/qwen_clstr_final_chain.py`: accept an optional promoted direct-utility gate report as a release overlay without rewriting the immutable Stage4 selection.
- Modify `scripts/resolve_qwen06_clstr_final_chain.py`: expose the optional gate report.
- Modify `clstr/qwen_clstr_multibench_submit.py`: propagate the external gate and its audit identity while retaining runtime `cmc`.
- Modify the focused current-state, ALFWorld, final-chain, and multibench tests for the new deployment contract.
- Create `scripts/run_qwen06_cmc_direct_utility_recalibration.py`: bind selected route artifacts, run audit, stop if ineligible, train if eligible, and write one self-hashed run report.
- Create `scripts/sbatch/run_qwen06_cmc_direct_utility_recalibration.sh`: lightweight Slurm-only entry point with no Qwen/model load.
- Create `tests/test_qwen06_cmc_direct_utility_recalibration.py`: orchestration and launcher contract tests.
- Create `clstr/alfworld_control_import.py`: validate, normalize, copy, and provenance-bind canonical Qwen-only control artifacts.
- Create `scripts/import_alfworld_qwen_control.py`: CLI for full or manifest-filtered smoke control imports.
- Modify `scripts/run_alfworld_qwen_clstr_executor_gate.py`: support hybrid-only execution with an imported control and build a complete gate summary.
- Modify `scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh`: require imported-control provenance when `RUN_QWEN_ONLY=0`.
- Modify `clstr/qwen_clstr_multibench_report.py`: fail closed on imported-control provenance/SHA/denominator drift.
- Modify `tests/test_qwen_clstr_multibench_report.py`, `tests/test_qwen_clstr_training_launchers.py`, and `tests/test_sbatch_scripts.py`: cover control reuse and gate identities.
- Update `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/finalwork/clstr_benchmark_results.md`: append audit, recalibration, matched routing, ALFWorld, and ablation results without changing historical rows.

## Fixed experiment contract

- Calibration route source: `outputs/qwen06_clstr_postfix/stage4_cmc_full/validation/step3000.route_manifest.json` and its 1,024 route rows.
- Immutable Stage4 control SHA-256: `72a14a41c39a555c0c2ec06a64cb05263ff52cfc96a09518b291dd7625014f4b`.
- Split: deterministic trajectory-disjoint 75% train / 25% dev with seed `17`.
- Utility temperature: `0.5` natural-log-probability units.
- Loss weights: fused ranking `1.0`, static no-regret `1.0`, direct gate BCE `1.0`.
- Optimizer: Adam, learning rate `0.01`, maximum `300` steps, seed `17`.
- Trainable parameters: only `route_memory_candidate_utility_gate.net.0.weight` and `.bias` in the standalone gate checkpoint.
- Audit thresholds: both utility signs present; utility-oracle macro MRR gain over static at least `0.005`; linear selector captures at least `50%` of that gain; worst source-level selector regret versus static no worse than `0.005`.
- Promotion thresholds: exact alpha-zero/static and alpha-one/dynamic endpoints; exact zero-history static fallback; source-balanced dev fused MRR strictly above static; worst source-level fused regret at most `0.005`; alpha has at least one dev row `<=0.25` and one `>=0.75` among nonzero-history, non-tie rows.
- No benchmark names, source IDs, skill IDs, task IDs, or final benchmark rows enter gate features or fitting.
- Tau2 evaluation is official base only (1,208 tool-action rows). Do not submit Tau2-full.
- All pytest, torch/model loading, recalibration, smoke, and evaluation commands run through Slurm. Local checks are limited to JSON/text, `py_compile`, `bash -n`, and `git diff --check`.

### Task 1: Tighten the oracle audit to the approved direct-utility thresholds

**Files:**
- Modify: `scripts/audit_clstr_memory_utility_oracle.py`
- Modify: `tests/test_memory_utility_oracle_audit.py`

- [x] **Step 1: Write failing threshold and source-regret tests**

Add tests with synthetic trajectory groups that assert the exact report contract:

```python
def test_direct_utility_audit_uses_predeclared_gain_and_regret_thresholds() -> None:
    report = run_memory_utility_oracle_audit(
        _qualifying_records(),
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=10,
        seed=17,
    )
    direct = report["direct_utility_eligibility"]
    assert direct["both_utility_signs_present"] is True
    assert direct["utility_oracle_gain_over_static"] >= 0.005
    assert direct["linear_selector_gain_capture_fraction"] >= 0.5
    assert direct["worst_source_regret_vs_static"] <= 0.005
    assert direct["eligible"] is True
    assert report["recommendation_thresholds"]["minimum_utility_oracle_gain"] == 0.005
    assert report["recommendation_thresholds"]["minimum_gain_capture_fraction"] == 0.5
    assert report["recommendation_thresholds"]["maximum_source_regret"] == 0.005


def test_direct_utility_audit_rejects_four_row_gate_snapshot() -> None:
    report = run_memory_utility_oracle_audit(
        _qualifying_records()[:4],
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=0,
        seed=17,
    )
    assert report["learned_gate_recommended"] is False
    assert "insufficient_qualifying_benchmarks" in report["recommendation_blockers"]
```

Extend leave-one-source assertions so each result records `static_macro_mrr`, `linear_selector_macro_mrr`, and `regret_vs_static`.

- [x] **Step 2: Run RED through Slurm**

Run:

```bash
sbatch --wait -p gpu_a800 --job-name=du-audit-red --gpus=1 --cpus-per-task=4 --time=00:12:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_memory_utility_oracle_audit.py'\'''
```

Expected: failures for missing `direct_utility_eligibility` and source-level selector MRR fields.

- [x] **Step 3: Implement selector MRR and exact eligibility**

Make `_fit_linear_selector()` return dev choices and add a summary helper:

```python
def _selector_mrr_summary(
    rows: list[dict[str, Any]],
    choose_dynamic: list[bool],
) -> dict[str, Any]:
    selected = [
        {
            **row,
            "selector_rank": int(row["dynamic_rank"] if choose else row["static_rank"]),
        }
        for row, choose in zip(rows, choose_dynamic)
    ]
    by_source = {
        source: macro_mrr(
            [row for row in selected if row["benchmark"] == source],
            rank_key="selector_rank",
        )
        for source in sorted({row["benchmark"] for row in selected})
    }
    return {
        "macro_mrr": macro_mrr(selected, rank_key="selector_rank"),
        "by_source_mrr": by_source,
    }
```

Compute:

```python
oracle_gain = metrics["utility_oracle_macro_mrr"] - metrics["static_macro_mrr"]
selector_gain = linear_summary["macro_mrr"] - metrics["static_macro_mrr"]
capture = selector_gain / oracle_gain if oracle_gain > 1.0e-12 else 0.0
static_by_source = {
    source: macro_mrr(
        [row for row in static_rows if row["benchmark"] == source],
        rank_key="rank",
    )
    for source in sorted({row["benchmark"] for row in static_rows})
}
worst_regret = max(
    static_by_source[source] - linear_summary["by_source_mrr"][source]
    for source in static_by_source
)
direct_eligible = (
    len(dynamic_winners) > 0
    and len(static_winners) > 0
    and oracle_gain >= 0.005
    and capture >= 0.5
    and worst_regret <= 0.005
)
```

Persist every input metric, threshold, and boolean. Keep older diagnostics, but make `learned_gate_recommended` depend on the new direct-utility eligibility plus basic data qualification only; remove obsolete rank-oracle/fixed-alpha blockers from the gate-refit decision.

- [x] **Step 4: Run GREEN and commit**

Run the Step 2 command and require all tests to pass, then:

```bash
git add scripts/audit_clstr_memory_utility_oracle.py tests/test_memory_utility_oracle_audit.py
git commit -m "feat: audit direct memory utility separability"
```

### Task 2: Add direct utility targets and a source-balanced gate-only objective

**Files:**
- Modify: `clstr/memory_utility_gate_train.py`
- Modify: `scripts/train_clstr_memory_utility_gate.py`
- Modify: `tests/test_memory_utility_gate_train.py`

- [x] **Step 1: Write failing utility-target and trainable-scope tests**

Add tests:

```python
def test_direct_utility_targets_are_detached_signed_and_confidence_weighted() -> None:
    static = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    dynamic = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    valid = torch.ones_like(static, dtype=torch.bool)
    positive = torch.tensor([[True, False], [True, False]])
    target, weight, delta = direct_utility_targets(
        static, dynamic, valid, positive, temperature=0.5
    )
    assert target[0] > 0.5 and target[1] < 0.5
    assert torch.all((weight >= 0) & (weight <= 1))
    assert not target.requires_grad
    assert not weight.requires_grad
    assert not delta.requires_grad


def test_gate_training_changes_only_linear_gate_parameters(tmp_path: Path) -> None:
    records = _records("calibration")
    selection, route_manifest_path = _cmc_dynamic_selection(tmp_path, records)
    audit = _audit(
        recommended=True,
        route_manifest_path=route_manifest_path,
        route_records=records,
    )
    report = train_memory_utility_gate(
        route_records=records,
        audit_report=audit,
        dynamic_selection=selection,
        output_dir=tmp_path / "gate",
        temperature=0.5,
        fused_rank_weight=1.0,
        static_no_regret_weight=1.0,
        direct_gate_bce_weight=1.0,
        max_steps=100,
        learning_rate=0.01,
        seed=17,
    )
    payload = torch.load(report["checkpoint_path"], map_location="cpu")
    assert set(payload["trainable_parameter_names"]) == {
        "net.0.weight",
        "net.0.bias",
    }
    assert payload["objective"] == {
        "temperature": 0.5,
        "fused_rank_weight": 1.0,
        "static_no_regret_weight": 1.0,
        "direct_gate_bce_weight": 1.0,
        "source_balanced": True,
    }
```

Update fixtures to use the 1,024-style selected validation record identity and a CMC dynamic selection containing `cmc_fused_selection` and `stage2_baseline`; do not synthesize `fixed_alpha_selection`.

- [x] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=du-gate-red --gpus=1 --cpus-per-task=4 --time=00:12:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_memory_utility_gate_train.py'\'''
```

Expected: import/signature failures for `direct_utility_targets` and CMC source binding.

- [x] **Step 3: Implement the detached target and source-balanced losses**

Add reusable helpers:

```python
def direct_utility_targets(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    static_u = _positive_log_probability(static, valid, positive)
    dynamic_u = _positive_log_probability(dynamic, valid, positive)
    delta = (dynamic_u - static_u).detach()
    scale = float(temperature)
    target = torch.sigmoid(delta / scale)
    weight = (delta.abs() / scale).clamp(0.0, 1.0)
    return target, weight, delta


def _source_balanced_mean(values: torch.Tensor, sources: list[str]) -> torch.Tensor:
    return torch.stack([
        values[torch.tensor([source == name for source in sources], device=values.device)].mean()
        for name in sorted(set(sources))
    ]).mean()
```

For every training step compute per-row fused log probability, no-regret, and BCE:

```python
fused_rank = -fused_utility
no_regret = torch.relu(static_utility.detach() - fused_utility)
bce = F.binary_cross_entropy(alpha, target_alpha, reduction="none") * row_weight
loss = (
    _source_balanced_mean(fused_rank, train_sources)
    + _source_balanced_mean(no_regret, train_sources)
    + _source_balanced_mean(bce, train_sources)
)
```

Reset the gate with training-only mean/scale and default linear initialization; optimize only `gate.parameters()`. Bind calibration records to `dynamic_selection["validation_route_records"]` and to the audit manifest identity. Use the audit trajectory split for train/dev and compute promotion only on dev. Remove the separate all-validation scoring path that would include train trajectories.

- [x] **Step 4: Implement exact promotion and report identities**

The report must include static, dynamic, and learned dev summaries; exact endpoint checks; worst source regret; alpha quantiles/min/max; low/high row counts; target/weight/delta summaries; objective hyperparameters; route/audit/Stage4 SHAs; and the exact trainable parameter names. Promote only when every fixed promotion condition is true. If not promoted, write `gate_report.json` but no checkpoint.

- [x] **Step 5: Run GREEN and commit**

```bash
git add clstr/memory_utility_gate_train.py scripts/train_clstr_memory_utility_gate.py tests/test_memory_utility_gate_train.py
git commit -m "feat: train gate from direct counterfactual utility"
```

### Task 3: Deploy an external direct-utility gate without disabling the CMC adapter

**Files:**
- Modify: `clstr/memory_utility_gate.py`
- Modify: `clstr/memory_utility_gate_train.py`
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `tests/test_current_state_route_eval.py`
- Modify: `tests/test_alfworld_eval.py`
- Modify: `tests/test_memory_utility_gate_train.py`

- [x] **Step 1: Write failing CMC override tests**

Add a current-state test that gives the checkpoint-native gate and external gate opposite outputs, then asserts:

```python
output = _build_current_state_route_batch(
    model,
    rows,
    device,
    skill_id_to_idx={"skill_a": 0, "skill_b": 1, "skill_c": 2},
    equivalent_skill_ids_by_skill_id={},
    static_k=2,
    dynamic_extra_k=1,
    final_k=2,
    reliability_mode="cmc",
    memory_utility_gate=AlwaysDynamicGate(),
    feature_update_count_cap=4.0,
    feature_candidate_count_cap=3.0,
    safe_memory_residual_bound=2.0,
)
assert output.metrics["cmc_adapter_enabled"] is True
assert output.metrics["memory_utility_external_gate_loaded"] is True
assert torch.equal(output.raw_alpha, torch.ones_like(output.raw_alpha))
assert not torch.equal(output.dynamic_full_logits, output.raw_dynamic_full_logits)
```

Add equivalent admissible-action and concrete-action ALFWorld tests. Add resolver tests that allow a checkpoint for `cmc`, reject it for `static/dynamic/fixed_alpha`, and require expected gate/audit SHAs.

- [x] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=du-deploy-red --gpus=1 --cpus-per-task=4 --time=00:15:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_memory_utility_gate_train.py tests/test_current_state_route_eval.py tests/test_alfworld_eval.py'\'''
```

- [x] **Step 3: Permit an optional gate only for `cmc` and `learned`**

Change `resolve_reliability_gate()` so:

```python
if mode not in {"learned", "cmc"}:
    if normalized_path is not None:
        raise ValueError("this reliability mode must not load a gate checkpoint")
    return None, {"mode": mode, "gate_loaded": False}
if mode == "learned" and normalized_path is None:
    raise ValueError("learned reliability requires a gate checkpoint")
if mode == "cmc" and normalized_path is None:
    return None, {"mode": mode, "gate_loaded": False, "gate_source": "stage4_checkpoint"}
```

When `cmc` receives a checkpoint, load it with the same SHA/audit validation and report `gate_source="direct_utility_overlay"`.

- [x] **Step 4: Override only alpha in the CMC scoring path**

Keep `score_cmc_candidates()` responsible for the residual-adjusted dynamic logits and features. In current-state and both ALFWorld scorers:

```python
if reliability_mode == "cmc":
    raw_alpha = (
        memory_utility_gate(cmc_scoring.features)
        if memory_utility_gate is not None
        else cmc_scoring.raw_alpha
    )
    effective_alpha = effective_memory_alpha(raw_alpha, causal_update_count)
```

Do not switch to `learned`, do not use raw pre-CMC dynamic logits, and do not alter memory state. Persist gate source, checkpoint SHA, audit SHA, and caps in metrics.

- [x] **Step 5: Run GREEN and commit**

```bash
git add clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py clstr/current_state_route_eval.py clstr/alfworld_eval.py tests/test_current_state_route_eval.py tests/test_alfworld_eval.py tests/test_memory_utility_gate_train.py
git commit -m "feat: overlay direct utility gate on cmc scoring"
```

### Task 4: Build an immutable final-chain reliability overlay

**Files:**
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `scripts/resolve_qwen06_clstr_final_chain.py`
- Modify: `clstr/qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_frozen_route_eval.py`

- [x] **Step 1: Write failing final-chain overlay tests**

Create a promoted audit-bound gate fixture and assert:

```python
manifest = resolve_qwen_clstr_final_chain(
    run_root,
    reliability_gate_report_path=gate_report_path,
)
assert manifest["reliability"]["mode"] == "cmc_candidate_gate"
assert manifest["reliability"]["gate_checkpoint"]["sha256"] == gate_sha
assert manifest["reliability"]["audit_manifest_sha256"] == audit_sha
assert manifest["reliability"]["base_gate_source"] == "stage4_checkpoint"
assert manifest["reliability"]["deployed_gate_source"] == "direct_utility_overlay"
```

Reject unpromoted reports, mismatched selected Stage4 SHA, tampered self-hashes, wrong feature schema/caps, or a report that names any trainable endpoint beyond the gate linear layer.

- [x] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=du-chain-red --gpus=1 --cpus-per-task=4 --time=00:15:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_qwen_clstr_final_chain.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_frozen_route_eval.py'\'''
```

- [x] **Step 3: Implement an overlay without rewriting Stage4 artifacts**

Add `reliability_gate_report_path: str | Path | None = None` to the resolver. Validate the existing Stage4 selection and checkpoint first. If a promoted direct-utility report is supplied, replace only the final manifest's reliability block:

```python
reliability = {
    "mode": "cmc_candidate_gate",
    "fixed_alpha": None,
    "gate_checkpoint": sha256_path(gate_report["checkpoint_path"]),
    "audit_manifest_sha256": gate_report["audit_manifest_sha256"],
    "gate_report": sha256_path(gate_report_path),
    "selected_stage4_checkpoint_sha256": gate_report["selected_stage4_checkpoint_sha256"],
    "feature_update_count_cap": gate_report["feature_update_count_cap"],
    "feature_candidate_count_cap": gate_report["feature_candidate_count_cap"],
    "base_gate_source": "stage4_checkpoint",
    "deployed_gate_source": "direct_utility_overlay",
}
```

Self-hash the reliability block and final chain. Keep the existing final chain unchanged as the control; write the overlay chain to a new output path.

- [x] **Step 4: Propagate the external gate while keeping runtime mode `cmc`**

The multibench stage exports must contain the gate path/SHA/audit SHA and the gate-derived feature caps. `RELIABILITY_MODE` remains `cmc`. Add fail-closed tests that no stage silently maps the overlay to `learned`.

- [x] **Step 5: Run GREEN and commit**

```bash
git add clstr/qwen_clstr_final_chain.py scripts/resolve_qwen06_clstr_final_chain.py clstr/qwen_clstr_multibench_submit.py tests/test_qwen_clstr_final_chain.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_frozen_route_eval.py
git commit -m "feat: bind direct utility gate into final chain"
```

### Task 5: Add one fail-closed Slurm recalibration workflow

**Files:**
- Create: `scripts/run_qwen06_cmc_direct_utility_recalibration.py`
- Create: `scripts/sbatch/run_qwen06_cmc_direct_utility_recalibration.sh`
- Create: `tests/test_qwen06_cmc_direct_utility_recalibration.py`
- Modify: `tests/test_sbatch_scripts.py`

- [x] **Step 1: Write failing orchestration tests**

Test that the workflow:

1. requires a self-hashed CMC dynamic selection;
2. requires the supplied route manifest to equal `validation_route_records`, not `gate_route_records`;
3. writes the audit before any fitting;
4. returns `status="not_recommended"` with no checkpoint when audit eligibility fails;
5. calls gate fitting only when audit eligibility passes;
6. writes a self-hashed `direct_utility_recalibration_report.json` binding route, audit, gate report, checkpoint, and Stage4 identities.

The launcher test must assert no Qwen/model/checkpoint load, `CUDA_VISIBLE_DEVICES=` for the Python process, and the fixed `temperature=0.5`, weights `1/1/1`, seed `17`, steps `300`, and learning rate `0.01`.

- [x] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=du-flow-red --gpus=1 --cpus-per-task=4 --time=00:12:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_qwen06_cmc_direct_utility_recalibration.py tests/test_sbatch_scripts.py -k "direct_utility or memory_utility"'\'''
```

- [x] **Step 3: Implement the atomic workflow**

Use the selected route manifest path, load through `load_route_records_from_manifests()`, run and finalize the oracle report, then branch:

```python
if audit["learned_gate_recommended"] is not True:
    return write_run_report(status="not_recommended", checkpoint=None)
gate_report = train_memory_utility_gate(
    route_records=loaded["records"],
    audit_report=audit,
    dynamic_selection=dynamic,
    output_dir=output_dir / "gate",
    temperature=0.5,
    fused_rank_weight=1.0,
    static_no_regret_weight=1.0,
    direct_gate_bce_weight=1.0,
    max_steps=300,
    learning_rate=0.01,
    seed=17,
)
```

Never catch an identity mismatch and continue. A valid negative audit exits successfully with a negative report; corrupted inputs exit nonzero.

- [x] **Step 4: Run GREEN and commit**

```bash
git add scripts/run_qwen06_cmc_direct_utility_recalibration.py scripts/sbatch/run_qwen06_cmc_direct_utility_recalibration.sh tests/test_qwen06_cmc_direct_utility_recalibration.py tests/test_sbatch_scripts.py
git commit -m "feat: orchestrate direct utility gate recalibration"
```

### Task 6: Import canonical ALFWorld controls with provenance and run hybrid only

**Files:**
- Create: `clstr/alfworld_control_import.py`
- Create: `scripts/import_alfworld_qwen_control.py`
- Modify: `scripts/run_alfworld_qwen_clstr_executor_gate.py`
- Modify: `scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh`
- Modify: `clstr/qwen_clstr_multibench_report.py`
- Modify: `tests/test_qwen_clstr_multibench_report.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`
- Modify: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Write failing import and validator tests**

Create a canonical control fixture with raw `metrics.json`, `run.jsonl`, `method_executor_identity.json`, and `method_alfworld_executor_report.json`. Assert the importer writes:

```python
{
    "schema_version": "alfworld_qwen_control_import_v1",
    "status": "ok",
    "split": "valid_seen",
    "source": {
        "metrics": sha256_path(source_dir / "metrics.json"),
        "run": sha256_path(source_dir / "run.jsonl"),
        "executor_identity": sha256_path(source_dir / "method_executor_identity.json"),
        "executor_report": sha256_path(source_dir / "method_alfworld_executor_report.json"),
    },
    "imported": {
        "metrics": sha256_path(output_dir / "qwen_only" / "metrics.json"),
        "run": sha256_path(output_dir / "qwen_only" / "run.jsonl"),
    },
    "episode_count": 140,
    "gamefile_digest": canonical_digest([f"game-{index}" for index in range(140)]),
    "prefix_equivalence": {
        "rows": 30,
        "all_equal": True,
        "reference_run": sha256_path(prefix_reference_run),
        "compared_fields": ["gamefile", "chosen_action_trace", "success", "steps"],
    },
    "manifest_sha256": canonical_digest(payload_without_manifest_sha256),
}
```

The normalized imported metrics must add executor-gate control identity without changing measured success/reward/step values. Test tampered source, wrong split, duplicate gamefiles, denominator drift, nonmatching 30-row prefix, and a gate summary that does not exactly embed the imported metrics.

- [ ] **Step 2: Run RED through Slurm**

```bash
sbatch --wait -p gpu_a800 --job-name=du-alf-import-red --gpus=1 --cpus-per-task=4 --time=00:15:00 --wrap='bash -lc '\''export PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix; source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"; export CUDA_VISIBLE_DEVICES=; cd "${PROJECT_ROOT}"; python -m pytest -q tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_training_launchers.py tests/test_sbatch_scripts.py -k "alfworld or executor"'\'''
```

- [ ] **Step 3: Implement canonical import and optional manifest filtering**

For full evaluation import all canonical episode rows. For smoke, filter the canonical run to the exact frozen smoke manifest gamefiles and recompute aggregate metrics from those rows. Require `--prefix_reference_run_path` and compare the first 30 rows on gamefile, chosen action trace, success, and steps before importing. Copy artifacts atomically; do not symlink across worktrees. Record source and imported SHA-256 values, split manifest identity, gamefile digest, and prefix comparison identity.

Canonical full sources are:

```text
/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-alfworld-qwen14-executor/outputs/eval_corrections_v1/evaluations/full/qwen_only_alfworld_seen
/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-alfworld-qwen14-executor/outputs/eval_corrections_v1/evaluations/full/qwen_only_alfworld_unseen
```

- [ ] **Step 4: Make hybrid-only executor consume the imported control**

Add `--qwen_control_import_manifest`, `--memory_utility_gate_checkpoint_path`, `--expected_memory_utility_gate_checkpoint_sha256`, and `--expected_memory_utility_gate_audit_sha256`. If `--run_qwen_only` is false, require a valid import manifest and construct `qwen_report` from the imported metrics/run identity before `_delta()` and `gate_summary.json`. Load the overlay gate with:

```python
memory_utility_gate, gate_load_report = resolve_reliability_gate(
    reliability_mode=args.reliability_mode,
    gate_checkpoint_path=args.memory_utility_gate_checkpoint_path,
    expected_gate_sha256=args.expected_memory_utility_gate_checkpoint_sha256,
    expected_audit_sha256=args.expected_memory_utility_gate_audit_sha256,
    device=model.skill_table.E.device,
)
prior_scorer = ClstrUnifiedMemoryConcreteActionScorer(
    model,
    replay_prefix_max_steps=args.clstr_replay_prefix_max_steps,
    reliability_mode=args.reliability_mode,
    memory_utility_gate=memory_utility_gate,
    fixed_alpha=args.fixed_alpha,
    safe_memory_residual_bound=args.safe_memory_residual_bound,
    feature_update_count_cap=float(
        gate_load_report.get("feature_update_count_cap", args.feature_update_count_cap)
    ),
    feature_candidate_count_cap=float(
        gate_load_report.get("feature_candidate_count_cap", args.feature_candidate_count_cap)
    ),
)
```

Apply the same arguments to the admissible-action scorer. The sbatch launcher must require `QWEN_CONTROL_IMPORT_MANIFEST` when `RUN_QWEN_ONLY=0`, pass the external memory gate path/SHA/audit SHA, and keep `RUN_HYBRID=1`.

- [ ] **Step 5: Validate import provenance in the multibench aggregate**

When `control_import.json` is present, require its self-hash, source/imported file hashes, split and denominator, gamefile set equality with Qwen+CLSTR, executor identity, Qwen model digest, and prefix-equivalence result. Reject a bare copied `qwen_only` directory without provenance.

- [ ] **Step 6: Run GREEN and commit**

```bash
git add clstr/alfworld_control_import.py scripts/import_alfworld_qwen_control.py scripts/run_alfworld_qwen_clstr_executor_gate.py scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh clstr/qwen_clstr_multibench_report.py tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_training_launchers.py tests/test_sbatch_scripts.py
git commit -m "feat: reuse canonical alfworld qwen control"
```

### Task 7: Verify, audit, recalibrate, and create the promoted chain

**Files:**
- Create under outputs: `outputs/qwen06_clstr_postfix/cmc_direct_utility/`
- Update: `.planning/2026-07-13-minimal-stage4-calibration-design/progress.md` (not committed)

- [ ] **Step 1: Run local source checks only**

Run `git diff --check`, `python3 -m py_compile` on changed Python files, and `bash -n` on changed shell launchers. Do not run pytest locally.

- [ ] **Step 2: Run the complete affected pytest matrix through Slurm**

Run all focused suites from Tasks 1-6, then one combined affected suite. Record Slurm job IDs, exact pass counts, and runtimes. Require zero failures.

- [ ] **Step 3: Submit the direct-utility audit/recalibration job**

Use:

```text
DYNAMIC_SELECTION_PATH=outputs/qwen06_clstr_postfix/stage4_cmc_full/stage4_dynamic_selection.json
ROUTE_MANIFEST_PATH=outputs/qwen06_clstr_postfix/stage4_cmc_full/validation/step3000.route_manifest.json
OUTPUT_DIR=outputs/qwen06_clstr_postfix/cmc_direct_utility/recalibration
```

The launcher must bind the exact selected Stage4 SHA above. Monitor once after startup and near completion; this is a small route-record job and should not load Qwen or occupy a GPU for model computation.

- [ ] **Step 4: Apply the audit stop condition**

If `learned_gate_recommended=false`, do not train, do not create an overlay final chain, and do not rerun benchmarks. Record the negative audit as the experimental result and retain the current CMC final chain.

- [ ] **Step 5: Apply the promotion stop condition**

If the audit passes but `gate_report.promoted=false`, do not create an overlay final chain and do not rerun full benchmarks. Preserve the report as evidence that direct supervision did not satisfy source-balanced validation.

- [ ] **Step 6: Resolve a new final chain only after promotion**

Write:

```text
outputs/qwen06_clstr_postfix/cmc_direct_utility/final_chain_manifest.json
```

using the promoted gate report. Verify the Stage0/1/2/4 checkpoint SHAs and base `checkpoint_chain_digest` equal the current control chain; only the reliability block and final-manifest SHA change.

- [ ] **Step 7: Commit verified source before benchmark submission**

Run final source checks and commit any remaining verified source/test changes. Do not commit generated checkpoints or outputs.

### Task 8: Run aligned smoke/full benchmarks and update the result ledger

**Files:**
- Create under outputs: `outputs/qwen06_clstr_postfix/cmc_direct_utility/evaluation/`
- Modify: `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/finalwork/clstr_benchmark_results.md`

- [ ] **Step 1: Build overlay-chain smoke/full configs**

Clone the pinned current CMC evaluation config but change only `final_chain_manifest_path` and `output_root`. Keep ToolBench full-pool/final100, ToolSandbox row-local checkpoint-faithful, Tau2 official base, ALFWorld Qwen3-14B generate/proposal-bonus, weights `1.0/0.25`, loop guard, batch size one, and 50 steps.

- [ ] **Step 2: Import smoke controls and pass the nine-stage smoke gate**

Import the exact smoke gamefile subset from canonical seen/unseen controls, then submit the standard nine-stage smoke matrix. ALFWorld stages run hybrid only. Require completed Slurm evidence, final-chain/audit/gate identities, and aggregate `status=ok` before full submission.

- [ ] **Step 3: Import full controls and submit the nine-stage full matrix**

Import all 140 seen and 134 unseen canonical control rows. Submit the standard frozen/native/closed-loop stages. Do not submit Tau2-full or the already completed ToolSandbox local-table-rebuild ablation unless a code-identity change invalidates that diagnostic.

- [ ] **Step 4: Monitor at coarse, health-oriented intervals**

Check startup logs, throughput after representative batches, and terminal state. Prioritize training/evaluation health over auxiliary work; avoid frequent polling. Stop on nonfinite metrics, wrong denominators, missing external-gate identity, Qwen-only rerun, or protocol drift.

- [ ] **Step 5: Aggregate and compare against the immutable control**

Report static, raw dynamic, old CMC fused, and direct-utility CMC fused R@1/R@5/MRR for ToolBench, ToolSandbox, and Tau2-base; alpha mean/min/max and low/high counts; ALFWorld seen/unseen success and delta versus imported Qwen-only; SR/ToolRex matched rows; and the existing ToolSandbox local-table-rebuild sensitivity result.

- [ ] **Step 6: Update the shared ledger atomically**

Append a dated section to `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/finalwork/clstr_benchmark_results.md` containing exact checkpoint/audit/final-chain/corpus/provenance SHAs and Slurm job IDs. Do not overwrite historical `full3000`, old CMC, SR, or ToolRex rows.

- [ ] **Step 7: Final verification and handoff**

Run `git status --short`, `git diff --check`, final report self-hash validation, checkpoint lineage validation, and result-table consistency checks. The work is complete only when the promoted-or-negative audit outcome is documented and, if promoted, all four aligned benchmark families have terminal validated artifacts.
