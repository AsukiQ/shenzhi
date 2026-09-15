# Qwen0.6B CLSTR Stage4 Safe-Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use subagent-driven-development: the user has explicitly prohibited sub-agent dispatch for this work.

**Goal:** Train and select a Qwen0.6B CLSTR Stage4 memory delta that improves causal routing while keeping the Stage2 static endpoint exactly immutable and recoverable through reliability fallback.

**Architecture:** Reconstruct Stage0 plus Stage2 once, freeze the complete effective router in place, and optimize only `transition`, recurrent `gate`, and `action_proj`. Split trajectories before handoff, train with deterministic four-domain balanced batches, select a delta-only checkpoint on fixed validation, then optionally calibrate a standalone memory-utility gate and pin the selected reliability artifact in a new final-chain manifest.

**Tech Stack:** Python 3.10+, PyTorch, JSON/JSONL, pytest, Bash, Slurm, existing CLSTR checkpoint/lineage/evaluation utilities.

---

## Execution Constraints

- Work only in `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel` on branch `exp/qwen06-handoff-accel`.
- Keep the Qwen backbone fully frozen. Do not add LoRA or any unfreeze flag.
- Preserve `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix/outputs/qwen06_clstr_postfix/stage4_full` as the moving-static control.
- Write the new smoke/full outputs under `stage4_safe_smoke` and `stage4_safe_full`.
- Do not modify the SR/ToolREx worktrees or `clstr-qwen06-full-readiness`.
- Never run pytest directly on the login/storage node. Every RED/GREEN pytest command below is a Slurm payload requesting one GPU for allocation but setting `CUDA_VISIBLE_DEVICES=""` inside pytest.
- Local checks are limited to `python -m py_compile`, `bash -n`, `git diff --check`, small JSON reads, and Slurm submission/status inspection.
- Commit after every task. Do not include the pre-existing multibench-plan worktree changes in these commits.

The pytest interpreter is:

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python
```

Every pytest step below includes its complete Slurm command and exact test targets. Do not substitute a local pytest invocation.

## File Structure

### New focused modules

- `clstr/stage4_safe_memory.py`: immutable-router parameter policy, content digests, delta-only checkpoint state, and checkpoint validation.
- `clstr/stage4_data_protocol.py`: trajectory-disjoint split, deterministic validation probes, gate-calibration probes, and four-domain balanced batching.
- `clstr/stage4_validation.py`: fixed Stage4 validation, Stage2 baseline cache, fixed-alpha sweep, dynamic checkpoint selection, and final selection manifests.
- `clstr/memory_utility_gate_train.py`: conditional scalar-gate training, checkpoint loading, and audit-bound promotion reports.

### Existing modules modified

- `clstr/stage4_act_train.py`: safe unified-memory objective integration, scheduler/resume, validation cadence, and delta checkpoints.
- `clstr/stage4_quality_gate.py`: selected-checkpoint, static-safety, and fixed-validation release gates.
- `clstr/qwen_clstr_lineage.py`: strict Stage4 delta allowlist and selected-checkpoint metadata.
- `clstr/qwen_clstr_final_chain.py`: selection-manifest resolution and reliability identity.
- `clstr/frozen_clstr_route_eval.py`: final-chain v2 validation and explicit zero-history reliability reporting.
- `clstr/memory_utility_gate.py`: normalized learned scalar gate and reliability-mode validation.
- `clstr/native_benchmark_checkpoint_adapter.py`: strict safe-memory delta overlay and router-integrity audit.
- `clstr/toolbench_full_clstr_route_eval.py`, `clstr/global_pool_route_eval.py`, `clstr/tau2_route_eval.py`, `clstr/toolsandbox_route_eval.py`: validated learned-gate forwarding.
- `clstr/alfworld_eval.py`: stateful post-action memory and static/dynamic admissible-action fusion.
- `clstr/qwen_clstr_multibench_submit.py`: reliability exports for all nine evaluation jobs.
- `clstr/qwen_clstr_multibench_report.py`: stateful ALFWorld evidence validation and consolidated reporting.

### New or modified scripts

- `scripts/train_clstr_memory_utility_gate.py`
- `scripts/audit_clstr_memory_utility_oracle.py`
- `scripts/finalize_clstr_stage4_selection.py`
- `scripts/run_clstr_stage4_act_train.py`
- `scripts/audit_clstr_stage4_quality.py`
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`
- `scripts/sbatch/run_qwen06_clstr_native_route_eval.sh`
- `scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh`

### Tests

- Create `tests/test_stage4_safe_memory.py`
- Create `tests/test_stage4_data_protocol.py`
- Create `tests/test_stage4_validation.py`
- Create `tests/test_memory_utility_gate_train.py`
- Modify existing Stage4, lineage, final-chain, evaluator, ALFWorld, multibench, and launcher tests named in each task.

---

### Task 1: Immutable Stage2 Router and Delta-Only State Contract

**Files:**

- Create: `clstr/stage4_safe_memory.py`
- Create: `tests/test_stage4_safe_memory.py`
- Modify: `clstr/stage4_act_train.py:1232-1277`
- Modify: `tests/test_stage4_act_train.py:1089-1152`

- [ ] **Step 1: Write failing immutable-router tests**

Create a tiny model fixture and the exact desired contract in `tests/test_stage4_safe_memory.py`:

```python
from __future__ import annotations

import torch

from clstr.stage4_safe_memory import (
    freeze_stage4_safe_memory,
    router_state_digest,
    set_stage4_safe_training_mode,
    stage4_delta_state_dict,
    validate_stage4_delta_state_dict,
)


class _SkillTable(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.E = torch.nn.Parameter(torch.eye(dim), requires_grad=False)
        self.logit_scale_belief = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.skill_bias_belief = torch.nn.Parameter(torch.zeros(dim), requires_grad=False)


class _TinySafeMemoryModel(torch.nn.Module):
    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.proj = torch.nn.Linear(dim, dim, bias=False)
        self.skill_table = _SkillTable(dim)
        self.initial_belief_head = torch.nn.Linear(dim, dim)
        self.unified_retriever = torch.nn.Linear(dim * 2, dim)
        self.transition = torch.nn.Linear(dim, dim)
        self.gate = torch.nn.Linear(dim, dim)
        self.action_proj = torch.nn.Linear(dim, dim, bias=False)
        self.trans_head = torch.nn.Linear(dim, dim)
        self.skill_head = torch.nn.Linear(dim, dim)
        self.stop_head = torch.nn.Linear(dim, 1)


def test_safe_memory_freeze_and_delta_state_are_exact() -> None:
    model = _TinySafeMemoryModel()
    before = router_state_digest(model, scope="full")

    report = freeze_stage4_safe_memory(model)
    set_stage4_safe_training_mode(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = sum(parameter.square().sum() for group in optimizer.param_groups for parameter in group["params"])
    loss.backward()
    optimizer.step()

    assert report["trainable_modules"] == ["transition", "gate", "action_proj"]
    assert router_state_digest(model, scope="full") == before
    assert all(parameter.grad is None for parameter in model.initial_belief_head.parameters())
    assert all(parameter.grad is None for parameter in model.unified_retriever.parameters())
    assert model.encoder.training is False
    assert model.initial_belief_head.training is False
    assert model.unified_retriever.training is False
    assert model.transition.training is True
    assert model.gate.training is True
    assert model.action_proj.training is True

    state = stage4_delta_state_dict(model)
    assert state
    assert all(key.startswith(("transition.", "gate.", "action_proj.")) for key in state)
    assert validate_stage4_delta_state_dict(state)["status"] == "ok"


def test_safe_memory_delta_rejects_router_keys() -> None:
    bad_state = {
        "transition.weight": torch.ones(1, 1),
        "initial_belief_head.weight": torch.ones(1, 1),
    }

    try:
        validate_stage4_delta_state_dict(bad_state)
    except ValueError as exc:
        assert "forbidden Stage4 delta key" in str(exc)
    else:
        raise AssertionError("router key must be rejected")
```

In `tests/test_stage4_act_train.py`, change the unified-memory freeze expectation to exactly:

```python
assert audit["trainable_modules"] == ["transition", "gate", "action_proj"]
assert all(not parameter.requires_grad for parameter in model.initial_belief_head.parameters())
assert all(not parameter.requires_grad for parameter in model.unified_retriever.parameters())
```

- [ ] **Step 2: Submit RED and verify the missing module/API failures**

```bash
sbatch --parsable \
  --job-name=q06_s4_t1_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:20:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t1_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_safe_memory.py tests/test_stage4_act_train.py::test_stage4_unified_memory_freeze_trains_causal_modules_by_default'
```

Expected: nonzero exit because `clstr.stage4_safe_memory` does not exist and the old freeze policy still trains the router heads.

- [ ] **Step 3: Implement the immutable-router primitives**

Create `clstr/stage4_safe_memory.py` with these public constants and functions:

```python
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


IMMUTABLE_ROUTER_PREFIXES = (
    "encoder.",
    "skill_table.",
    "initial_belief_head.",
    "unified_retriever.",
)
FAST_ROUTER_PREFIXES = (
    "encoder.proj.",
    "initial_belief_head.",
    "unified_retriever.",
)
FAST_ROUTER_EXACT_KEYS = {
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
}
STAGE4_DELTA_PREFIXES = ("transition.", "gate.", "action_proj.")
STAGE4_TRAINABLE_MODULES = ("transition", "gate", "action_proj")


def _router_tensor_selected(name: str, *, scope: str) -> bool:
    if scope not in {"full", "fast"}:
        raise ValueError("router digest scope must be full or fast")
    if scope == "full":
        return any(name.startswith(prefix) for prefix in IMMUTABLE_ROUTER_PREFIXES)
    return (
        any(name.startswith(prefix) for prefix in FAST_ROUTER_PREFIXES)
        or name in FAST_ROUTER_EXACT_KEYS
    )


def router_state_digest(model: Any, *, scope: str = "full") -> str:
    digest = hashlib.sha256()
    selected_count = 0
    for name, tensor in sorted(model.state_dict().items()):
        if not _router_tensor_selected(name, scope=scope):
            continue
        selected_count += 1
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        byte_view = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    if selected_count == 0:
        raise ValueError("immutable router digest selected no tensors")
    return digest.hexdigest()


def set_stage4_safe_training_mode(model: Any) -> None:
    model.eval()
    for name in STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"safe-memory Stage4 requires model.{name}")
        module.train()


def freeze_stage4_safe_memory(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules: list[str] = []
    for name in STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"safe-memory Stage4 requires model.{name}")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        trainable_modules.append(name)
    set_stage4_safe_training_mode(model)
    optimizer_parameter_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    return {
        "frozen_routing_foundation": True,
        "immutable_stage2_router": True,
        "immutable_router_eval_mode": True,
        "frozen_belief_gate": False,
        "train_transition": True,
        "trainable_modules": trainable_modules,
        "optimizer_parameter_names": optimizer_parameter_names,
        "route_scorer": "unified_memory",
        "full_router_digest": router_state_digest(model, scope="full"),
        "fast_router_digest": router_state_digest(model, scope="fast"),
    }


def stage4_delta_state_dict(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(STAGE4_DELTA_PREFIXES)
    }
    validate_stage4_delta_state_dict(state)
    return state


def validate_stage4_delta_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError("Stage4 delta state must be a nonempty dict")
    forbidden = sorted(name for name in state if not str(name).startswith(STAGE4_DELTA_PREFIXES))
    if forbidden:
        raise ValueError(f"forbidden Stage4 delta key: {forbidden[0]}")
    missing = [
        prefix
        for prefix in STAGE4_DELTA_PREFIXES
        if not any(str(name).startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(f"Stage4 delta state missing prefix: {missing[0]}")
    return {"status": "ok", "state_key_count": len(state), "state_keys": sorted(state)}
```

Change `_freeze_for_stage4_act()` so unified-memory mode returns `freeze_stage4_safe_memory(model)` and keeps the legacy scorer branch unchanged.

The full digest implementation must remain streaming as shown: never materialize a second CPU copy of the entire Qwen state. Only one selected tensor may be transferred/hashed at a time.

- [ ] **Step 4: Submit GREEN and inspect the exact optimizer names**

Run the same Slurm command from Step 2 with job name `q06_s4_t1_green` and output log `q06_s4_t1_green-%j.out`.

Expected: both tests pass; the report contains only transition/gate/action-projection optimizer parameters.

- [ ] **Step 5: Run local static checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_safe_memory.py clstr/stage4_act_train.py tests/test_stage4_safe_memory.py
git diff --check -- clstr/stage4_safe_memory.py clstr/stage4_act_train.py \
  tests/test_stage4_safe_memory.py tests/test_stage4_act_train.py
git add clstr/stage4_safe_memory.py clstr/stage4_act_train.py \
  tests/test_stage4_safe_memory.py tests/test_stage4_act_train.py
git commit -m "fix: anchor stage4 routing to frozen stage2"
```

---

### Task 2: Trajectory-Disjoint Split and Four-Domain Balanced Batches

**Files:**

- Create: `clstr/stage4_data_protocol.py`
- Create: `tests/test_stage4_data_protocol.py`

- [ ] **Step 1: Write failing deterministic split and sampler tests**

Create tests that build 100 trajectories for each declared benchmark and assert input-order invariance, no trajectory overlap, deterministic row probes, and four rows per benchmark:

```python
from __future__ import annotations

import random

import pytest

from clstr.stage4_data_protocol import (
    BenchmarkBalancedStage4Batcher,
    build_stage4_data_protocol,
)


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")
SOURCE_DATA_IDENTITY = {
    benchmark: {"sha256": f"{benchmark}-source-sha256"} for benchmark in BENCHMARKS
}
PROMPT_CONTRACT = {
    "prompt_mode": "clstr_state_query_v1",
    "prompt_sha256": "prompt-contract-sha256",
}


def _rows() -> list[dict]:
    rows: list[dict] = []
    for benchmark in BENCHMARKS:
        for trajectory_idx in range(100):
            trajectory_id = f"trajectory-{trajectory_idx:03d}"
            for step_index in range(2):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "trajectory_id": trajectory_id,
                        "step_index": step_index,
                        "row_id": f"{benchmark}/{trajectory_id}/step-{step_index}",
                    }
                )
    return rows


def test_stage4_split_is_trajectory_disjoint_and_input_order_invariant() -> None:
    rows = _rows()
    shuffled = list(rows)
    random.Random(991).shuffle(shuffled)

    first = build_stage4_data_protocol(
        rows,
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )
    second = build_stage4_data_protocol(
        shuffled,
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )

    assert first.manifest == second.manifest
    assert [row["row_id"] for row in first.validation_rows] == [
        row["row_id"] for row in second.validation_rows
    ]
    train_ids = {f'{row["benchmark"]}::{row["trajectory_id"]}' for row in first.train_rows}
    validation_ids = {
        f'{row["benchmark"]}::{row["trajectory_id"]}' for row in first.validation_rows
    }
    assert train_ids.isdisjoint(validation_ids)
    assert "toolbench_g3::trajectory-000" in set(
        first.manifest["train_trajectory_ids"]
        + first.manifest["validation_trajectory_ids"]
    )
    assert "webshop::trajectory-000" in set(
        first.manifest["train_trajectory_ids"]
        + first.manifest["validation_trajectory_ids"]
    )
    assert first.manifest["validation_rows_by_benchmark"] == {
        benchmark: 8 for benchmark in BENCHMARKS
    }
    assert first.manifest["source_data_identity"] == SOURCE_DATA_IDENTITY
    assert first.manifest["prompt_contract"] == PROMPT_CONTRACT


def test_stage4_balanced_batch_contains_four_rows_per_benchmark() -> None:
    protocol = build_stage4_data_protocol(
        _rows(),
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )
    batcher = BenchmarkBalancedStage4Batcher(
        protocol.train_rows,
        benchmarks=BENCHMARKS,
        batch_size=16,
        seed=17,
    )

    batch = batcher.batch_for_step(1)
    assert {benchmark: sum(row["benchmark"] == benchmark for row in batch) for benchmark in BENCHMARKS} == {
        benchmark: 4 for benchmark in BENCHMARKS
    }
    assert [row["row_id"] for row in batch] == [
        row["row_id"] for row in BenchmarkBalancedStage4Batcher(
            protocol.train_rows,
            benchmarks=BENCHMARKS,
            batch_size=16,
            seed=17,
        ).batch_for_step(1)
    ]
```

Add these rejection tests below the two positive cases:

```python
@pytest.mark.parametrize("missing_field", ["trajectory_id", "step_index"])
def test_stage4_protocol_rejects_missing_trajectory_fields(missing_field: str) -> None:
    rows = _rows()
    rows[0] = dict(rows[0])
    rows[0].pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        build_stage4_data_protocol(
            rows,
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=8,
            minimum_validation_rows_per_benchmark=4,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )


def test_stage4_protocol_rejects_unknown_benchmark() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "benchmark": "unknown"}
    with pytest.raises(ValueError, match="unexpected Stage4 benchmark"):
        build_stage4_data_protocol(
            rows,
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=8,
            minimum_validation_rows_per_benchmark=4,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )


def test_stage4_batcher_rejects_nondivisible_batch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BenchmarkBalancedStage4Batcher(
            _rows(),
            benchmarks=BENCHMARKS,
            batch_size=15,
            seed=17,
        )


def test_stage4_protocol_rejects_underfilled_validation() -> None:
    with pytest.raises(ValueError, match="underfilled Stage4 validation benchmark"):
        build_stage4_data_protocol(
            _rows(),
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=256,
            minimum_validation_rows_per_benchmark=128,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )
```

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t2_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:20:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t2_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_data_protocol.py'
```

Expected: import failure because `clstr.stage4_data_protocol` does not exist.

- [ ] **Step 3: Implement exact hash split and deterministic probes**

Create `clstr/stage4_data_protocol.py` with:

```python
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from clstr.memory_utility_records import canonical_digest


STAGE4_SPLIT_VERSION = "stage4_trajectory_split_v1"
STAGE4_VALIDATION_ROW_VERSION = "stage4_validation_rows_v1"
STAGE4_GATE_ROW_VERSION = "stage4_gate_rows_v1"


@dataclass(frozen=True)
class Stage4DataProtocol:
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    gate_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def _stable_hash(*parts: object) -> int:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _row_id(row: dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or "").strip()
    if not benchmark:
        raise ValueError("safe-memory Stage4 requires benchmark")
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not trajectory_id:
        raise ValueError("safe-memory Stage4 requires trajectory_id")
    if row.get("step_index") is None:
        raise ValueError("safe-memory Stage4 requires step_index")
    return str(
        row.get("row_id")
        or f"{benchmark}/{trajectory_id}/step-{int(row['step_index'])}"
    )


def _trajectory_key(benchmark: str, trajectory_id: str) -> str:
    return f"{benchmark}::{trajectory_id}"


def _probe_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    seed: int,
    version: str,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_stable_hash(version, seed, benchmark, _row_id(row)), _row_id(row)))
    return ordered[: max(0, int(count))]


def build_stage4_data_protocol(
    rows: list[dict[str, Any]],
    *,
    expected_benchmarks: Sequence[str],
    seed: int,
    validation_fraction: float,
    validation_rows_per_benchmark: int,
    minimum_validation_rows_per_benchmark: int,
    gate_rows_per_benchmark: int,
    source_data_identity: dict[str, Any],
    prompt_contract: dict[str, Any],
) -> Stage4DataProtocol:
    benchmarks = tuple(str(item) for item in expected_benchmarks)
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("expected_benchmarks must be unique and nonempty")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if set(source_data_identity) != set(benchmarks):
        raise ValueError("source_data_identity must cover every Stage4 benchmark")
    if not prompt_contract.get("prompt_mode") or not prompt_contract.get("prompt_sha256"):
        raise ValueError("safe-memory Stage4 requires a pinned prompt contract")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {benchmark: {} for benchmark in benchmarks}
    for source in rows:
        row = dict(source)
        benchmark = str(row.get("benchmark") or "").strip()
        if benchmark not in grouped:
            raise ValueError(f"unexpected Stage4 benchmark: {benchmark}")
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        _row_id(row)
        row["row_id"] = _row_id(row)
        grouped[benchmark].setdefault(trajectory_id, []).append(row)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    train_trajectory_ids: list[str] = []
    validation_trajectory_ids: list[str] = []
    validation_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    train_trajectory_counts: Counter[str] = Counter()
    validation_trajectory_counts: Counter[str] = Counter()
    threshold = int(round(validation_fraction * 100.0))
    for benchmark in benchmarks:
        benchmark_train: list[dict[str, Any]] = []
        benchmark_validation: list[dict[str, Any]] = []
        for trajectory_id, trajectory_rows in sorted(grouped[benchmark].items()):
            target = benchmark_validation if _stable_hash(STAGE4_SPLIT_VERSION, seed, benchmark, trajectory_id) % 100 < threshold else benchmark_train
            target.extend(sorted(trajectory_rows, key=lambda row: (int(row["step_index"]), row["row_id"])))
            if target is benchmark_validation:
                validation_trajectory_ids.append(_trajectory_key(benchmark, trajectory_id))
                validation_trajectory_counts[benchmark] += 1
            else:
                train_trajectory_ids.append(_trajectory_key(benchmark, trajectory_id))
                train_trajectory_counts[benchmark] += 1
        selected_validation = _probe_rows(
            benchmark_validation,
            benchmark=benchmark,
            seed=seed,
            version=STAGE4_VALIDATION_ROW_VERSION,
            count=validation_rows_per_benchmark,
        )
        if len(selected_validation) < int(minimum_validation_rows_per_benchmark):
            raise ValueError(f"underfilled Stage4 validation benchmark: {benchmark}")
        selected_gate = _probe_rows(
            benchmark_train,
            benchmark=benchmark,
            seed=seed,
            version=STAGE4_GATE_ROW_VERSION,
            count=gate_rows_per_benchmark,
        )
        train_rows.extend(benchmark_train)
        validation_rows.extend(selected_validation)
        gate_rows.extend(selected_gate)
        validation_counts[benchmark] = len(selected_validation)
        gate_counts[benchmark] = len(selected_gate)
        train_counts[benchmark] = len(benchmark_train)

    manifest = {
        "schema_version": "stage4_data_protocol_v1",
        "split_version": STAGE4_SPLIT_VERSION,
        "seed": int(seed),
        "expected_benchmarks": list(benchmarks),
        "source_data_identity": dict(source_data_identity),
        "prompt_contract": dict(prompt_contract),
        "train_trajectory_ids": sorted(train_trajectory_ids),
        "validation_trajectory_ids": sorted(validation_trajectory_ids),
        "train_row_ids": sorted(_row_id(row) for row in train_rows),
        "validation_row_ids": [_row_id(row) for row in validation_rows],
        "gate_row_ids": [_row_id(row) for row in gate_rows],
        "validation_rows_by_benchmark": dict(sorted(validation_counts.items())),
        "gate_rows_by_benchmark": dict(sorted(gate_counts.items())),
        "train_rows_by_benchmark": dict(sorted(train_counts.items())),
        "train_trajectories_by_benchmark": dict(sorted(train_trajectory_counts.items())),
        "validation_trajectories_by_benchmark": dict(
            sorted(validation_trajectory_counts.items())
        ),
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return Stage4DataProtocol(train_rows, validation_rows, gate_rows, manifest)


class BenchmarkBalancedStage4Batcher:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        benchmarks: Sequence[str],
        batch_size: int,
        seed: int,
    ) -> None:
        self.benchmarks = tuple(str(item) for item in benchmarks)
        if int(batch_size) % len(self.benchmarks) != 0:
            raise ValueError("Stage4 batch_size must be divisible by benchmark count")
        self.quota = int(batch_size) // len(self.benchmarks)
        self.seed = int(seed)
        self.rows_by_benchmark = {
            benchmark: [dict(row) for row in rows if str(row.get("benchmark")) == benchmark]
            for benchmark in self.benchmarks
        }
        missing = [benchmark for benchmark, items in self.rows_by_benchmark.items() if not items]
        if missing:
            raise ValueError(f"Stage4 training benchmark has no rows: {missing[0]}")
        self._epoch_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def _ordered_epoch(self, benchmark: str, epoch: int) -> list[dict[str, Any]]:
        key = (benchmark, int(epoch))
        cached = self._epoch_cache.get(key)
        if cached is not None:
            return cached
        rows = list(self.rows_by_benchmark[benchmark])
        random.Random(_stable_hash("stage4_batch", self.seed, benchmark, epoch)).shuffle(rows)
        self._epoch_cache[key] = rows
        return rows

    def batch_for_step(self, step: int) -> list[dict[str, Any]]:
        if int(step) <= 0:
            raise ValueError("Stage4 step must be positive")
        batch: list[dict[str, Any]] = []
        for benchmark in self.benchmarks:
            source = self.rows_by_benchmark[benchmark]
            absolute_start = (int(step) - 1) * self.quota
            for offset in range(self.quota):
                absolute_index = absolute_start + offset
                epoch, index = divmod(absolute_index, len(source))
                batch.append(self._ordered_epoch(benchmark, epoch)[index])
        return batch
```

- [ ] **Step 4: Submit GREEN**

Run the Step 2 command with job name `q06_s4_t2_green` and log `q06_s4_t2_green-%j.out`.

Expected: all protocol tests pass, including shuffled-input parity and exact per-benchmark quotas.

- [ ] **Step 5: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_data_protocol.py tests/test_stage4_data_protocol.py
git diff --check -- clstr/stage4_data_protocol.py tests/test_stage4_data_protocol.py
git add clstr/stage4_data_protocol.py tests/test_stage4_data_protocol.py
git commit -m "feat: add trajectory-disjoint stage4 data protocol"
```

---

### Task 3: Safe Causal Route Batch and Frozen-Router Gradient Semantics

**Files:**

- Modify: `clstr/stage4_act_train.py:780-1110`
- Modify: `tests/test_stage4_act_train.py:957-1152`
- Modify: `tests/test_counterfactual_ranking.py`

- [ ] **Step 1: Turn the old moving-router gradient test RED**

Replace the module loop in `test_stage4_unified_memory_main_ce_backpropagates_through_post_action_modules` with:

```python
_freeze_for_stage4_act(model, route_scorer="unified_memory")

for module in (model.transition, model.gate, model.action_proj):
    grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0

for module in (model.initial_belief_head, model.unified_retriever):
    assert all(parameter.grad is None for parameter in module.parameters())
```

Change the existing helper signature to
`def _stage4_unified_counterfactual_outputs(model: _CounterfactualStage4UnifiedMemoryModel | None = None, **row_overrides):`
and replace its current first assignment with
`model = model or _CounterfactualStage4UnifiedMemoryModel()`. Leave the existing row construction, loss call, and cloned return tensors unchanged.

Add `test_stage4_safe_optimizer_step_preserves_static_logits`:

```python
def test_stage4_safe_optimizer_step_preserves_static_logits() -> None:
    model = _CounterfactualStage4UnifiedMemoryModel()
    _freeze_for_stage4_act(model, route_scorer="unified_memory")
    rows = [
        {
            "state_text": "current state",
            "action_text": "action alpha",
            "next_observation_text": "observation alpha",
            "next_state_text": "future alpha",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }
    ]
    before = _stage4_unified_counterfactual_outputs(model=model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss, _metrics = _compute_stage4_act_loss(
        model,
        rows,
        torch.device("cpu"),
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        counterfactual_utility_weight=0.05,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = _stage4_unified_counterfactual_outputs(model=model)

    assert torch.equal(before["static_logits"], after["static_logits"])
```

Add a RED contract test that imports `_build_stage4_safe_route_batch` and calls it on the same row:

```python
def test_stage4_safe_route_batch_exposes_teacher_and_dynamic_endpoints() -> None:
    model = _CounterfactualStage4UnifiedMemoryModel()
    model.route_logits_from_candidates = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("safe Stage4 must use full memory-conditioned logits")
    )
    _freeze_for_stage4_act(model, route_scorer="unified_memory")
    batch = _build_stage4_safe_route_batch(
        model,
        [{
            "state_text": "current state",
            "action_text": "action alpha",
            "next_observation_text": "observation alpha",
            "next_state_text": "future alpha",
            "skill_id": "skill/current",
            "next_skill_id": "skill/next",
            "skill_idx": 0,
            "positive_next_skill_idx": 1,
            "candidate_next_skill_indices": [0, 1, 2],
        }],
        torch.device("cpu"),
        skill_id_to_idx={"skill/current": 0, "skill/next": 1, "skill/other": 2},
        equivalent_skill_ids_by_skill_id=None,
        trainable_replay_prefix=True,
        counterfactual_utility_weight=0.05,
        counterfactual_gain_margin=0.10,
        counterfactual_safety_tolerance=0.01,
        counterfactual_gain_weight=1.0,
        counterfactual_safety_weight=1.0,
        counterfactual_scale=1.0,
    )
    assert batch.dynamic_full_logits.shape == batch.static_full_logits.shape == (1, 3)
    assert batch.dynamic_full_logits.requires_grad
    assert not batch.static_full_logits.requires_grad
    assert batch.causal_update_count.tolist() == [1.0]
```

Keep the existing action-only and observation-only counterfactual tests unchanged; they must continue proving that only dynamic logits respond.

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t3_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:30:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t3_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_act_train.py::test_stage4_safe_route_batch_exposes_teacher_and_dynamic_endpoints tests/test_stage4_act_train.py::test_stage4_unified_memory_main_ce_backpropagates_through_post_action_modules tests/test_stage4_act_train.py::test_stage4_safe_optimizer_step_preserves_static_logits tests/test_stage4_act_train.py::test_stage4_unified_memory_action_counterfactual_changes_dynamic_not_static_logits tests/test_stage4_act_train.py::test_stage4_unified_memory_observation_counterfactual_changes_dynamic_not_static_logits tests/test_counterfactual_ranking.py'
```

Expected: collection fails because `_build_stage4_safe_route_batch` is missing; if collected separately, the frozen-router assertions document the required gradient boundary.

- [ ] **Step 3: Extract one structured safe-route batch builder**

Add this dataclass immediately before `_compute_stage4_act_loss`:

```python
@dataclass(frozen=True)
class Stage4SafeRouteBatch:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    dynamic_full_logits: torch.Tensor
    static_full_logits: torch.Tensor
    positive_mask: torch.Tensor
    valid_mask: torch.Tensor
    dynamic_memory: torch.Tensor
    static_memory: torch.Tensor
    causal_update_count: torch.Tensor
    metrics: dict[str, Any]
```

Import `dataclass` and `freeze_stage4_safe_memory` at module scope. Extract the current unified/full-pool block into:

```python
def _build_stage4_safe_route_batch(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    trainable_replay_prefix: bool,
    counterfactual_utility_weight: float,
    counterfactual_gain_margin: float,
    counterfactual_safety_tolerance: float,
    counterfactual_gain_weight: float,
    counterfactual_safety_weight: float,
    counterfactual_scale: float,
) -> Stage4SafeRouteBatch:
```

Move the existing current-state encoding, replay, post-action memory, legal-mask, positive-mask, `full_pool_causal_route_objective`, and ranking-metric code into this helper. The scorer portion must be exactly:

```python
dynamic_full = model.unified_route_full_logits(h_next, next_memory)
with torch.no_grad():
    static_memory = model.initial_belief(h_next)
    static_full = model.unified_route_full_logits(h_next, static_memory)
```

The helper must not call `route_logits_from_candidates()` or any h-only candidate scorer. Both teacher and dynamic branches come from `unified_route_full_logits()` before legal-mask gathering.

Use `causal_update_count = torch.tensor([1 + len(row.get("replay_prefix") or []) for row in rows], dtype=dynamic_full.dtype, device=device)` and return all tensors without detaching the dynamic branch. Construct the loss with the unchanged counterfactual defaults and detached static utility.

In `_compute_stage4_act_loss()`, delegate unified/full-pool rows to this helper and return `batch.total_loss, batch.metrics`. Keep legacy candidate-limited code unchanged.

- [ ] **Step 4: Submit GREEN and the complete Stage4 objective regression**

First rerun the exact Step 2 targets under job name `q06_s4_t3_green`. Then submit:

```bash
sbatch --parsable \
  --job-name=q06_s4_t3_reg \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=01:00:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t3_reg-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_act_train.py tests/test_counterfactual_ranking.py tests/test_full_base_train.py -k "stage4 or counterfactual or post_action or replay_prefix_initializes"'
```

Expected: all selected tests pass; static logits remain exact and dynamic action/observation sensitivity remains active.

- [ ] **Step 5: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_act_train.py tests/test_stage4_act_train.py
git diff --check -- clstr/stage4_act_train.py tests/test_stage4_act_train.py \
  tests/test_counterfactual_ranking.py
git add clstr/stage4_act_train.py tests/test_stage4_act_train.py \
  tests/test_counterfactual_ranking.py
git commit -m "fix: train stage4 only through causal memory updates"
```

---

### Task 4: Fixed Validation, Static Baseline Cache, and Dynamic Checkpoint Selection

**Files:**

- Create: `clstr/stage4_validation.py`
- Create: `tests/test_stage4_validation.py`
- Modify: `clstr/stage4_act_train.py`

- [ ] **Step 1: Write failing pure selection and exact-static tests**

Create `tests/test_stage4_validation.py` with deterministic report fixtures:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from clstr.stage4_validation import (
    choose_fixed_alpha,
    select_stage4_dynamic_checkpoint,
    validate_static_baseline_batches,
    write_stage4_dynamic_selection,
)


def _alpha_metrics(mrr: float) -> dict:
    return {
        "balanced_macro_mrr": mrr,
        "zero_history_exact": True,
        "by_benchmark": {
            benchmark: {"mrr": mrr}
            for benchmark in ("toolbench_g3", "traject_bench", "alfworld", "webshop")
        },
    }


def _report(step: int, dynamic_mrr: float, delta_mrr: float, recall5: float) -> dict:
    return {
        "schema_version": "stage4_validation_report_v1",
        "status": "ok",
        "release_status": "ok",
        "step": step,
        "mechanically_eligible": True,
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "report_sha256": "stage2-baseline-sha256",
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.59,
            "dynamic_macro_mrr": 0.60,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.59, "dynamic_mrr": 0.60}
                for benchmark in ("toolbench_g3", "traject_bench", "alfworld", "webshop")
            },
        },
        "candidate_union": {
            "static_k": 500,
            "dynamic_extra_k": 64,
            "final_k": 64,
        },
        "nonfinite_counts": {"loss": 0, "logits": 0, "metrics": 0, "gradients": 0},
        "exclusion_counts": {"missing_positive": 0, "no_legal_candidate": 0},
        "delta_state_validation": {"status": "ok", "state_key_count": 6},
        "balanced_macro": {
            "dynamic_mrr": dynamic_mrr,
            "dynamic_minus_static_mrr": delta_mrr,
            "dynamic_recall@5": recall5,
        },
        "by_benchmark": {
            benchmark: {
                "dynamic_mrr": dynamic_mrr,
                "static_mrr": dynamic_mrr - delta_mrr,
                "dynamic_minus_static_mrr": delta_mrr,
            }
            for benchmark in ("toolbench_g3", "traject_bench", "alfworld", "webshop")
        },
        "fixed_alpha": {
            "0.0": _alpha_metrics(dynamic_mrr - delta_mrr),
            "0.25": _alpha_metrics(dynamic_mrr - 0.01),
            "0.5": _alpha_metrics(dynamic_mrr + 0.01),
            "0.75": _alpha_metrics(dynamic_mrr + 0.005),
            "1.0": _alpha_metrics(dynamic_mrr),
        },
    }


def _persist_report(tmp_path: Path, report: dict) -> tuple[Path, Path]:
    step = int(report["step"])
    checkpoint = tmp_path / f"persisted-stage4-step{step}.pt"
    checkpoint.write_bytes(str(step).encode("utf-8"))
    validation_records = tmp_path / f"persisted-validation-step{step}.jsonl"
    validation_records.write_text('{"row_digest":"row-a"}\n', encoding="utf-8")
    gate_records = tmp_path / f"persisted-gate-step{step}.jsonl"
    gate_records.write_text('{"row_digest":"gate-row-a"}\n', encoding="utf-8")
    gate_manifest = tmp_path / f"persisted-gate-step{step}.manifest.json"
    gate_manifest.write_text("{}", encoding="utf-8")
    report["validation_route_records"] = {
        **sha256_path(validation_records),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["row-a"]),
    }
    report["gate_route_records"] = {
        **sha256_path(gate_records),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["gate-row-a"]),
    }
    report["gate_route_manifest"] = sha256_path(gate_manifest)
    report["manifest_sha256"] = canonical_digest(report)
    report_path = tmp_path / f"persisted-validation-step{step}.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return checkpoint, report_path


def test_dynamic_selection_prefers_best_validation_not_final_step(tmp_path: Path) -> None:
    reports = [
        _report(400, 0.61, 0.04, 0.72),
        _report(800, 0.68, 0.07, 0.79),
        _report(1200, 0.66, 0.08, 0.81),
    ]
    checkpoint_paths = {}
    validation_report_paths = {}
    for report in reports:
        checkpoint = tmp_path / f"stage4-step{report['step']}.pt"
        checkpoint.write_bytes(str(report["step"]).encode("utf-8"))
        checkpoint_paths[report["step"]] = checkpoint
        route_records_path = tmp_path / f"validation-step{report['step']}.route_records.jsonl"
        route_records_path.write_text('{"row_digest":"row-a"}\n', encoding="utf-8")
        report["validation_route_records"] = {
            **sha256_path(route_records_path),
            "record_count": 1,
            "row_digest_sha256": canonical_digest(["row-a"]),
        }
        gate_records_path = tmp_path / f"validation-step{report['step']}.gate_records.jsonl"
        gate_records_path.write_text('{"row_digest":"gate-row-a"}\n', encoding="utf-8")
        report["gate_route_records"] = {
            **sha256_path(gate_records_path),
            "record_count": 1,
            "row_digest_sha256": canonical_digest(["gate-row-a"]),
        }
        gate_manifest_path = tmp_path / f"validation-step{report['step']}.gate_manifest.json"
        gate_manifest_path.write_text("{}", encoding="utf-8")
        report["gate_route_manifest"] = sha256_path(gate_manifest_path)
        report["manifest_sha256"] = canonical_digest(report)
        report_path = tmp_path / f"validation-step{report['step']}.json"
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        validation_report_paths[report["step"]] = report_path

    selected = select_stage4_dynamic_checkpoint(
        reports,
        checkpoint_paths=checkpoint_paths,
        validation_report_paths=validation_report_paths,
    )

    assert selected["selected_step"] == 800
    assert selected["selected_checkpoint_path"] == str(checkpoint_paths[800].resolve())
    assert selected["selected_validation_report_path"] == str(
        validation_report_paths[800].resolve()
    )
    assert selected["release_status"] == "ok"
    assert choose_fixed_alpha(reports[1])["fixed_alpha"] == pytest.approx(0.5)


def test_static_baseline_batches_require_bitwise_equality() -> None:
    baseline = [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]
    assert validate_static_baseline_batches(baseline, [baseline[0].clone()])["status"] == "ok"
    with pytest.raises(ValueError, match="static validation logits drifted"):
        validate_static_baseline_batches(baseline, [torch.tensor([[1.0, 2.001]])])


def test_dynamic_selection_manifest_is_self_hashed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage4-step800.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = _report(800, 0.68, 0.07, 0.79)
    route_records_path = tmp_path / "validation-step800.route_records.jsonl"
    route_records_path.write_text('{"row_digest":"row-a"}\n', encoding="utf-8")
    report["validation_route_records"] = {
        **sha256_path(route_records_path),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["row-a"]),
    }
    gate_records_path = tmp_path / "validation-step800.gate_records.jsonl"
    gate_records_path.write_text('{"row_digest":"gate-row-a"}\n', encoding="utf-8")
    report["gate_route_records"] = {
        **sha256_path(gate_records_path),
        "record_count": 1,
        "row_digest_sha256": canonical_digest(["gate-row-a"]),
    }
    gate_manifest_path = tmp_path / "validation-step800.gate_manifest.json"
    gate_manifest_path.write_text("{}", encoding="utf-8")
    report["gate_route_manifest"] = sha256_path(gate_manifest_path)
    report["manifest_sha256"] = canonical_digest(report)
    report_path = tmp_path / "validation-step800.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    selection = select_stage4_dynamic_checkpoint(
        [report],
        checkpoint_paths={800: checkpoint},
        validation_report_paths={800: report_path},
    )
    output = tmp_path / "stage4_dynamic_selection.json"
    written = write_stage4_dynamic_selection(
        output,
        selection=selection,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["manifest_sha256"]
    assert loaded["selected_checkpoint_sha256"]
    assert loaded["selected_validation_report_sha256"]
    assert loaded["router_integrity"]["static_logits_exact"] is True
    assert loaded["fixed_alpha_selection"]["fixed_alpha"] == pytest.approx(0.5)
    assert loaded["validation_route_records"]["sha256"] == sha256_path(route_records_path)["sha256"]
    assert loaded["gate_route_records"]["sha256"] == sha256_path(gate_records_path)["sha256"]
    assert loaded["gate_route_manifest"]["sha256"] == sha256_path(gate_manifest_path)["sha256"]
```

Add the exact tie/failure cases:

```python
def test_dynamic_selection_uses_earliest_step_as_final_tie_break(tmp_path: Path) -> None:
    reports = [_report(400, 0.68, 0.07, 0.79), _report(800, 0.68, 0.07, 0.79)]
    persisted = [_persist_report(tmp_path, report) for report in reports]
    selected = select_stage4_dynamic_checkpoint(
        reports,
        checkpoint_paths={step: pair[0] for step, pair in zip((400, 800), persisted)},
        validation_report_paths={step: pair[1] for step, pair in zip((400, 800), persisted)},
    )
    assert selected["selected_step"] == 400


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("static_drift", "exact static parity"),
        ("missing_benchmark", "no mechanically eligible"),
        ("nonfinite", "must be finite"),
        ("oversized_final_k", "final_k must not exceed static_k"),
    ],
)
def test_dynamic_selection_rejects_invalid_validation_contract(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    if case == "static_drift":
        report["router_integrity"]["max_abs_static_logit_difference"] = 0.001
    elif case == "missing_benchmark":
        report["by_benchmark"].pop("webshop")
    elif case == "nonfinite":
        report["balanced_macro"]["dynamic_mrr"] = float("nan")
    elif case == "oversized_final_k":
        report["candidate_union"]["final_k"] = 501
    checkpoint, report_path = _persist_report(tmp_path, report)
    with pytest.raises(ValueError, match=message):
        select_stage4_dynamic_checkpoint(
            [report],
            checkpoint_paths={800: checkpoint},
            validation_report_paths={800: report_path},
        )


def test_dynamic_selection_writer_rejects_checkpoint_digest_drift(tmp_path: Path) -> None:
    report = _report(800, 0.68, 0.07, 0.79)
    checkpoint, report_path = _persist_report(tmp_path, report)
    selection = select_stage4_dynamic_checkpoint(
        [report],
        checkpoint_paths={800: checkpoint},
        validation_report_paths={800: report_path},
    )
    checkpoint.write_bytes(b"mutated-after-selection")
    with pytest.raises(ValueError, match="selected Stage4 checkpoint SHA-256 mismatch"):
        write_stage4_dynamic_selection(
            tmp_path / "stage4_dynamic_selection.json",
            selection=selection,
        )
```

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t4_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:25:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t4_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_validation.py'
```

Expected: import failure because `clstr.stage4_validation` does not exist.

- [ ] **Step 3: Implement validation primitives and atomic manifests**

Create `clstr/stage4_validation.py` with these public contracts:

```python
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path


STAGE4_VALIDATION_REPORT_SCHEMA = "stage4_validation_report_v1"
STAGE4_DYNAMIC_SELECTION_SCHEMA = "stage4_dynamic_selection_v1"
FIXED_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def validate_static_baseline_batches(
    baseline_batches: list[torch.Tensor],
    current_batches: list[torch.Tensor],
) -> dict[str, Any]:
    if len(baseline_batches) != len(current_batches):
        raise ValueError("static validation batch count drifted")
    maximum = 0.0
    for baseline, current in zip(baseline_batches, current_batches):
        if baseline.shape != current.shape or baseline.dtype != current.dtype:
            raise ValueError("static validation tensor contract drifted")
        difference = float((baseline - current).abs().max().item()) if baseline.numel() else 0.0
        maximum = max(maximum, difference)
        if not torch.equal(baseline, current):
            raise ValueError("static validation logits drifted")
    return {
        "status": "ok",
        "static_logits_exact": True,
        "max_abs_static_logit_difference": maximum,
    }


def choose_fixed_alpha(report: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for raw_alpha, metrics in dict(report.get("fixed_alpha") or {}).items():
        alpha = float(raw_alpha)
        macro_mrr = float(metrics["balanced_macro_mrr"])
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0 or not math.isfinite(macro_mrr):
            raise ValueError("fixed-alpha validation metrics must be finite")
        candidates.append((macro_mrr, -alpha, alpha, dict(metrics)))
    if not candidates:
        raise ValueError("validation report has no fixed-alpha results")
    macro_mrr, _negative_alpha, alpha, metrics = max(candidates, key=lambda item: item[:2])
    return {"fixed_alpha": alpha, **metrics, "balanced_macro_mrr": macro_mrr}


def _validate_route_record_identity(recorded: dict[str, Any], *, label: str) -> dict[str, Any]:
    identity = dict(recorded or {})
    actual = sha256_path(identity["path"])
    if actual["sha256"] != str(identity.get("sha256") or ""):
        raise ValueError(f"{label} SHA-256 mismatch")
    rows = [
        json.loads(line)
        for line in Path(identity["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(identity.get("record_count", -1)):
        raise ValueError(f"{label} count mismatch")
    if canonical_digest([row["row_digest"] for row in rows]) != str(
        identity.get("row_digest_sha256") or ""
    ):
        raise ValueError(f"{label} row digest mismatch")
    return identity


def select_stage4_dynamic_checkpoint(
    reports: list[dict[str, Any]],
    *,
    checkpoint_paths: dict[int, Path],
    validation_report_paths: dict[int, Path],
) -> dict[str, Any]:
    eligible = []
    for report in reports:
        if report.get("status") != "ok" or report.get("mechanically_eligible") is not True:
            continue
        by_benchmark = dict(report.get("by_benchmark") or {})
        if set(by_benchmark) != {"toolbench_g3", "traject_bench", "alfworld", "webshop"}:
            continue
        router_integrity = dict(report.get("router_integrity") or {})
        stage2_baseline = dict(report.get("stage2_baseline") or {})
        if (
            router_integrity.get("static_logits_exact") is not True
            or float(router_integrity.get("max_abs_static_logit_difference", float("inf"))) != 0.0
            or str(router_integrity.get("full_router_digest") or "")
            != str(stage2_baseline.get("full_router_digest") or "")
        ):
            raise ValueError("Stage4 validation report violates exact static parity")
        if dict(report.get("delta_state_validation") or {}).get("status") != "ok":
            raise ValueError("Stage4 validation report has an invalid delta checkpoint")
        candidate_union = dict(report.get("candidate_union") or {})
        if int(candidate_union["final_k"]) > int(candidate_union["static_k"]):
            raise ValueError("Stage4 final_k must not exceed static_k")
        if report.get("release_status") not in {"ok", "action_required"}:
            raise ValueError("Stage4 validation report has invalid release status")
        if sum(int(value) for value in dict(report.get("nonfinite_counts") or {}).values()) != 0:
            raise ValueError("Stage4 validation report contains nonfinite values")
        macro = dict(report.get("balanced_macro") or {})
        metric_values = [
            float(macro["dynamic_mrr"]),
            float(macro["dynamic_minus_static_mrr"]),
            float(macro["dynamic_recall@5"]),
            *[
                float(metrics[name])
                for metrics in by_benchmark.values()
                for name in ("dynamic_mrr", "static_mrr", "dynamic_minus_static_mrr")
            ],
        ]
        if not all(math.isfinite(value) for value in metric_values):
            raise ValueError("Stage4 validation selection metrics must be finite")
        step = int(report["step"])
        checkpoint = Path(checkpoint_paths[step]).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"missing Stage4 validation checkpoint: {checkpoint}")
        report_path = Path(validation_report_paths[step]).resolve()
        if not report_path.is_file():
            raise ValueError(f"missing Stage4 validation report: {report_path}")
        persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
        if persisted_report != report:
            raise ValueError("Stage4 validation report path does not match selected report")
        report_without_hash = dict(report)
        recorded_report_sha = str(report_without_hash.pop("manifest_sha256", ""))
        if recorded_report_sha != canonical_digest(report_without_hash):
            raise ValueError("Stage4 validation report self-hash mismatch")
        validation_route_records = _validate_route_record_identity(
            report.get("validation_route_records") or {},
            label="Stage4 validation route records",
        )
        gate_route_records = _validate_route_record_identity(
            report.get("gate_route_records") or {},
            label="Stage4 gate route records",
        )
        gate_route_manifest = dict(report.get("gate_route_manifest") or {})
        actual_gate_manifest = sha256_path(gate_route_manifest["path"])
        if actual_gate_manifest["sha256"] != str(gate_route_manifest.get("sha256") or ""):
            raise ValueError("Stage4 gate route-manifest SHA-256 mismatch")
        key = (
            float(macro["dynamic_mrr"]),
            float(macro["dynamic_minus_static_mrr"]),
            float(macro["dynamic_recall@5"]),
            -step,
        )
        eligible.append(
            (
                key,
                report,
                checkpoint,
                report_path,
                validation_route_records,
                gate_route_records,
                gate_route_manifest,
            )
        )
    if not eligible:
        raise ValueError("no mechanically eligible Stage4 validation checkpoint")
    (
        _key,
        report,
        checkpoint,
        report_path,
        validation_route_records,
        gate_route_records,
        gate_route_manifest,
    ) = max(
        eligible,
        key=lambda item: item[0],
    )
    return {
        "status": "ok",
        "release_status": str(report["release_status"]),
        "selected_step": int(report["step"]),
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(report_path),
        "selected_validation_report_sha256": sha256_path(report_path)["sha256"],
        "validation_route_records": validation_route_records,
        "gate_route_records": gate_route_records,
        "gate_route_manifest": gate_route_manifest,
        "router_integrity": dict(report["router_integrity"]),
        "stage2_baseline": dict(report["stage2_baseline"]),
        "candidate_union": dict(report["candidate_union"]),
        "fixed_alpha_selection": choose_fixed_alpha(report),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_file_sha(path: str | Path, expected: str, *, label: str) -> None:
    if sha256_path(path)["sha256"] != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")


def _read_self_hashed_manifest(path: str | Path, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported manifest schema: {path}")
    digest_payload = dict(payload)
    recorded = str(digest_payload.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(digest_payload):
        raise ValueError(f"manifest self-hash mismatch: {path}")
    return payload


def write_stage4_dynamic_selection(
    output_path: str | Path,
    *,
    selection: dict[str, Any],
) -> dict[str, Any]:
    payload = {**selection, "schema_version": STAGE4_DYNAMIC_SELECTION_SCHEMA}
    required = {
        "status",
        "release_status",
        "selected_step",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "selected_validation_report_path",
        "selected_validation_report_sha256",
        "validation_route_records",
        "gate_route_records",
        "gate_route_manifest",
        "router_integrity",
        "stage2_baseline",
        "candidate_union",
        "fixed_alpha_selection",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Stage4 dynamic selection missing field: {missing[0]}")
    if payload["status"] != "ok" or payload["release_status"] not in {
        "ok",
        "action_required",
    }:
        raise ValueError("Stage4 dynamic selection status is invalid")
    _require_file_sha(
        payload["selected_checkpoint_path"],
        payload["selected_checkpoint_sha256"],
        label="selected Stage4 checkpoint",
    )
    _require_file_sha(
        payload["selected_validation_report_path"],
        payload["selected_validation_report_sha256"],
        label="selected Stage4 validation report",
    )
    _validate_route_record_identity(
        payload["validation_route_records"],
        label="Stage4 validation route records",
    )
    _validate_route_record_identity(
        payload["gate_route_records"],
        label="Stage4 gate route records",
    )
    _require_file_sha(
        payload["gate_route_manifest"]["path"],
        payload["gate_route_manifest"]["sha256"],
        label="Stage4 gate route manifest",
    )
    if payload["router_integrity"].get("static_logits_exact") is not True:
        raise ValueError("Stage4 dynamic selection lacks exact static parity")
    if int(payload["candidate_union"]["final_k"]) > int(
        payload["candidate_union"]["static_k"]
    ):
        raise ValueError("Stage4 final_k must not exceed static_k")
    payload["manifest_sha256"] = canonical_digest(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload
```

Import `os` plus `sha256_path` from `clstr.qwen_clstr_lineage`. The writer above revalidates every referenced checkpoint/report/route-record/route-manifest identity plus router integrity before it adds the self-hash.

Add `evaluate_stage4_validation()` plus `persist_stage4_validation_artifacts()`. The evaluator returns metrics and detached record rows in memory; the persister receives the just-written compact checkpoint identity so every route manifest pins the exact Stage4 overlay that produced it. Together they:

1. iterates the fixed row order in batches;
2. calls `_build_stage4_safe_route_batch()` under `torch.no_grad()`;
3. at step0, before any optimizer update, stores static logits in `validation/stage2_static_baseline.pt` and writes a self-hashed `validation/stage2_baseline_report.json` containing both unmodified-Stage2 static and replayed-dynamic metrics; every later report embeds that baseline report SHA and metrics;
4. computes static/dynamic/fused MRR and Recall@1/5 globally and by benchmark;
5. evaluates every value in `FIXED_ALPHA_GRID` through `fuse_route_scores()`;
6. reports memory-active delta and static/dynamic candidate-union recall, records `static_k`, `dynamic_extra_k`, and `final_k`, and rejects `final_k > static_k`;
7. records full/fast router digests and delta-state validation;
8. writes detached `validation/step{step}.route_records.jsonl` plus SHA-256 so an optional learned gate can be evaluated on the fixed validation rows without rerunning or using test data; records retain raw `causal_update_count` and valid masks so count-derived feature slots can later be recomputed from audit train-only caps;
9. writes separate `validation/step{step}.gate_route_records.jsonl` from the deterministic training-source probe plus an existing-schema `memory_utility_route_manifest_v1` manifest; both identities are pinned in the report, and these rows drive only the oracle/gate fit, never memory-updater gradients at validation time;
10. records loss/logit/metric/gradient nonfinite counts and row exclusion counts, making any nonfinite count mechanically ineligible;
11. computes `release_status` as the dynamic-checkpoint release decision only: `ok` exactly when the report satisfies the specification's dynamic absolute/delta/per-benchmark thresholds, otherwise `action_required`; it persists every dynamic release check while deferring the final fused-release decision to `finalize_stage4_selection()`.

Do not use Tau2 or final benchmark data in this module.

- [ ] **Step 4: Submit GREEN and validation integration tests**

Rerun Step 2 as `q06_s4_t4_green`, then submit:

```bash
sbatch --parsable \
  --job-name=q06_s4_t4_reg \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:45:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t4_reg-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_validation.py tests/test_memory_candidate_recall.py tests/test_memory_utility_gate.py tests/test_stage4_act_train.py -k "validation or candidate or fusion or full_pool"'
```

Expected: all selected tests pass and fixed-alpha endpoint parity remains exact.

- [ ] **Step 5: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_validation.py clstr/stage4_act_train.py tests/test_stage4_validation.py
git diff --check -- clstr/stage4_validation.py clstr/stage4_act_train.py \
  tests/test_stage4_validation.py
git add clstr/stage4_validation.py clstr/stage4_act_train.py \
  tests/test_stage4_validation.py
git commit -m "feat: select stage4 on fixed causal validation"
```

---

### Task 5: Trainer Integration, Warmup-Cosine Schedule, Resume, and Dynamic Selection

**Files:**

- Modify: `clstr/stage4_act_train.py:1280-1692`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- Modify: `tests/test_stage4_act_train.py:1383-1760`
- Modify: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Write failing trainer protocol and scheduler tests**

Add signature/default assertions:

```python
parameters = inspect.signature(train_stage4_act_with_model).parameters
assert parameters["learning_rate"].default == 3.0e-5
assert parameters["minimum_learning_rate"].default == 3.0e-6
assert parameters["learning_rate_warmup_fraction"].default == 0.05
assert parameters["validation_interval_steps"].default == 400
assert parameters["validation_rows_per_benchmark"].default == 256
assert parameters["minimum_validation_rows_per_benchmark"].default == 128
assert parameters["gate_rows_per_benchmark"].default == 512
assert parameters["resume_checkpoint_path"].default is None
```

Add a pure scheduler test:

```python
def test_stage4_warmup_cosine_multiplier_has_declared_endpoints() -> None:
    assert _stage4_lr_multiplier(0, total_steps=100, warmup_fraction=0.05, minimum_ratio=0.1) == 0.0
    assert _stage4_lr_multiplier(5, total_steps=100, warmup_fraction=0.05, minimum_ratio=0.1) == pytest.approx(1.0)
    assert _stage4_lr_multiplier(100, total_steps=100, warmup_fraction=0.05, minimum_ratio=0.1) == pytest.approx(0.1)
```

Extend the small end-to-end trainer fixture to four benchmarks and enough trajectories, then assert:

```python
assert report["data_protocol"]["validation_rows_by_benchmark"]
assert report["optimizer"]["peak_learning_rate"] == pytest.approx(3.0e-5)
assert report["scheduler"]["type"] == "linear_warmup_cosine_v1"
assert report["dynamic_selection"]["selected_checkpoint_path"]
assert report["valid_or_test_used_for_training"] is False
assert report["validation_used_for_checkpoint_selection"] is True
assert report["safe_memory_protocol_version"] == "stage4_safe_memory_v1"
payload = torch.load(report["dynamic_selection"]["selected_checkpoint_path"], map_location="cpu")
assert all(key.startswith(("transition.", "gate.", "action_proj.")) for key in payload["model_state_dict"])
assert "optimizer_state_dict" not in payload
assert "scheduler_state_dict" not in payload
resume_payload = torch.load(report["latest_checkpoint"], map_location="cpu")
assert resume_payload["optimizer_state_dict"]
assert resume_payload["scheduler_state_dict"]
```

Add an event-order assertion that `stage4_data_protocol_created` appears before `stage0_candidate_handoff_prepared` in `setup_status.jsonl`.

Add a resume test that saves a step1 checkpoint and verifies step2 restores model, optimizer, scheduler, split manifest, sampler seed, and router digests. Change one split hash and assert resume fails closed.

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t5_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:45:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t5_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_act_train.py tests/test_sbatch_scripts.py -k "stage4 or warmup_cosine or dynamic_selection or data_protocol or resume_checkpoint"'
```

Expected: failures on missing arguments, scheduler helper, split integration, delta-only checkpoint, and resume identity.

- [ ] **Step 3: Integrate the data protocol before handoff**

Add these trainer arguments:

```python
learning_rate: float = 3.0e-5,
minimum_learning_rate: float = 3.0e-6,
learning_rate_warmup_fraction: float = 0.05,
validation_fraction: float = 0.10,
validation_rows_per_benchmark: int = 256,
minimum_validation_rows_per_benchmark: int = 128,
gate_rows_per_benchmark: int = 512,
validation_interval_steps: int = 400,
resume_checkpoint_path: str | Path | None = None,
```

Immediately after `_eligible_stage4_source_rows()`, resolve the exact per-benchmark source-manifest identities and `resolve_state_query_prompt_contract()` output, then pass both into `build_stage4_data_protocol()` on the complete eligible four-domain row set. Do this before benchmark caps, shuffling, handoff materialization, or optimizer construction. Apply benchmark row caps only to `protocol.train_rows`, selecting the lowest stable hashes per benchmark rather than relying on input order; never cap or resample by borrowing rows from `protocol.validation_rows`. Persist both the uncapped trajectory assignment and the exact capped training-row IDs in `stage4_data_protocol.json` and include its hash in every checkpoint.

Prepare handoff rows separately for capped training rows, fixed validation rows, and deterministic gate-probe rows. Validation handoff may encode/cache frozen inputs, but validation rows must never appear in `BenchmarkBalancedStage4Batcher`, a gradient-bearing loss, optimizer statistics, or gate-training records. Every trainer/checkpoint/selection report must keep `valid_or_test_used_for_training=False` and add `validation_used_for_checkpoint_selection=True`.

Materialize the fixed validation state/next-state/action/observation embeddings once under one recorded Qwen batch schedule before step0, hash that cache identity, and reuse it at every validation event. Do not re-encode validation with a different BF16 batch composition at later steps.

Set `safe_memory_protocol_version="stage4_safe_memory_v1"` in the train report and every rolling/validation checkpoint. Do not set this marker on the preserved legacy control path.

Replace `_batch_for_step()` in the safe unified-memory path with `BenchmarkBalancedStage4Batcher.batch_for_step(step_idx)`. Keep `_batch_for_step()` only for explicit legacy scorer compatibility.

- [ ] **Step 4: Add scheduler, validation cadence, delta checkpoints, and resume**

Implement:

```python
def _stage4_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step <= warmup_steps:
        return max(0.0, min(1.0, float(step) / float(warmup_steps)))
    progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine
```

Create AdamW with explicit `weight_decay=0.01` and a `LambdaLR` using `minimum_learning_rate / learning_rate` as the minimum ratio. Save and restore scheduler state.

Never call global `model.train()` on the safe path. Call `set_stage4_safe_training_mode(model)` before optimization starts and again after every `model.eval()` validation event so frozen Qwen/router modules remain in eval mode while only transition/recurrent-gate/action-projection modules return to train mode.

Run validation at step0, every `validation_interval_steps`, and the final step. For each validation event:

1. verify exact optimizer-name equality, the fast router digest, and cached static-logit equality;
2. at step0, reuse the complete router digest already computed once by `freeze_stage4_safe_memory()` and persist the immutable Stage2 static/dynamic baseline report before any optimizer update; do not hash the full Qwen state twice at startup;
3. evaluate fixed validation and the training-source gate probe in memory without writing a route manifest yet;
4. save `resume/clstr_stage4_safe-step{step}.pt` with delta model state plus optimizer/scheduler/sampler state for exact resume;
5. atomically save `checkpoints/clstr_stage4_safe-step{step}.pt` as the compact evaluation overlay containing the stage/config/train report, `model_state_dict=stage4_delta_state_dict(model)`, and no optimizer or scheduler state;
6. persist `validation/step{step}.json`, validation records, gate records, and the gate route manifest with the exact compact checkpoint-chain digest; include the resume-checkpoint identity in the report;
7. pass compact checkpoint paths and persisted validation-report paths to `select_stage4_dynamic_checkpoint()`;
8. update `stage4_dynamic_selection.json` with release status, selected compact artifact/report SHA-256 values, router integrity, baseline identity, candidate-union contract, and the fixed-alpha selection.

After step3000 (or the configured final step), recompute the complete router digest. Before finalization, load the selected delta into a fresh Stage0→Stage2 reconstruction and require the same complete digest plus exact fixed-validation static logits. This is the final expensive content audit; intermediate events use the fast digest and cached logits.

The rolling `latest.pt` and stepwise resume checkpoints must also keep a delta-only `model_state_dict`; their optimizer/scheduler state is separate payload metadata. Resume rejects any mismatch in model stage, split manifest SHA, sampler seed/benchmarks, full router digest, optimizer parameter names, or scheduler configuration. Final-chain resolution is allowed to use only the compact checkpoint directory, never `latest.pt` or `resume/`.

- [ ] **Step 5: Thread every argument through CLI and lower sbatch launcher**

Add CLI flags:

```text
--minimum_learning_rate
--learning_rate_warmup_fraction
--validation_fraction
--validation_rows_per_benchmark
--minimum_validation_rows_per_benchmark
--gate_rows_per_benchmark
--validation_interval_steps
--resume_checkpoint_path
```

Add corresponding environment variables to `run_clstr_unified_stage4_act_train.sh` and forward them exactly once. The launcher must not provide any encoder-unfreeze option.

- [ ] **Step 6: Submit GREEN and Stage4 trainer regressions**

Rerun Step 2 as `q06_s4_t5_green`, then submit:

```bash
sbatch --parsable \
  --job-name=q06_s4_t5_reg \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=01:30:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t5_reg-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_act_train.py tests/test_stage4_validation.py tests/test_stage4_data_protocol.py tests/test_sbatch_scripts.py -k "stage4"'
```

Expected: all selected tests pass with no router keys in any Stage4 checkpoint.

- [ ] **Step 7: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_act_train.py scripts/run_clstr_stage4_act_train.py
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh
git diff --check -- clstr/stage4_act_train.py scripts/run_clstr_stage4_act_train.py \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh tests/test_stage4_act_train.py \
  tests/test_sbatch_scripts.py
git add clstr/stage4_act_train.py scripts/run_clstr_stage4_act_train.py \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh tests/test_stage4_act_train.py \
  tests/test_sbatch_scripts.py
git commit -m "feat: train and select safe-memory stage4"
```

---

### Task 6: Final Selection, Quality Gate, Lineage, and Final-Chain v2

**Files:**

- Modify: `clstr/stage4_validation.py`
- Create: `scripts/finalize_clstr_stage4_selection.py`
- Modify: `clstr/stage4_quality_gate.py`
- Modify: `scripts/audit_clstr_stage4_quality.py`
- Modify: `clstr/qwen_clstr_lineage.py`
- Modify: `clstr/qwen_clstr_final_chain.py`
- Modify: `clstr/frozen_clstr_route_eval.py`
- Modify: `tests/test_stage4_quality_gate.py`
- Modify: `tests/test_qwen_clstr_lineage.py`
- Modify: `tests/test_qwen_clstr_final_chain.py`
- Modify: `tests/test_qwen_clstr_frozen_route_eval.py`

- [ ] **Step 1: Write failing final-selection and v2-chain tests**

Update the final-chain fixture so Stage4 lives under `stage4_safe_full`, writes `stage4_dynamic_selection.json`, and finalizes:

```json
{
  "schema_version": "stage4_selection_v1",
  "status": "ok",
  "release_status": "ok",
  "dynamic_release_status": "ok",
  "release_checks": {
    "dynamic_release_ok": true,
    "fused_macro_improvement": true,
    "fused_per_benchmark_nonregression": true,
    "zero_history_exact": true
  },
  "selected_checkpoint_path": "/absolute/stage4-safe-step800.pt",
  "selected_checkpoint_sha256": "selected-checkpoint-sha256",
  "selected_validation_report_path": "/absolute/validation/step800.json",
  "selected_validation_report_sha256": "selected-validation-report-sha256",
  "router_integrity": {
    "full_router_digest": "stage2-router-digest",
    "fast_router_digest": "stage2-fast-router-digest",
    "static_logits_exact": true,
    "max_abs_static_logit_difference": 0.0
  },
  "stage2_baseline": {
    "report_sha256": "stage2-baseline-sha256",
    "full_router_digest": "stage2-router-digest",
    "static_macro_mrr": 0.60,
    "dynamic_macro_mrr": 0.62,
    "by_benchmark": {
      "toolbench_g3": {"static_mrr": 0.60, "dynamic_mrr": 0.62},
      "traject_bench": {"static_mrr": 0.60, "dynamic_mrr": 0.62},
      "alfworld": {"static_mrr": 0.60, "dynamic_mrr": 0.62},
      "webshop": {"static_mrr": 0.60, "dynamic_mrr": 0.62}
    }
  },
  "candidate_union": {
    "static_k": 500,
    "dynamic_extra_k": 64,
    "final_k": 64
  },
  "reliability": {
    "mode": "fixed_alpha",
    "fixed_alpha": 0.5,
    "gate_checkpoint": null,
    "reliability_sha256": "self-hash-of-reliability"
  },
  "reliability_validation": {
    "balanced_macro_mrr": 0.64,
    "zero_history_exact": true,
    "by_benchmark": {
      "toolbench_g3": {"mrr": 0.64},
      "traject_bench": {"mrr": 0.64},
      "alfworld": {"mrr": 0.64},
      "webshop": {"mrr": 0.64}
    }
  },
  "manifest_sha256": "self-hash"
}
```

Assert `resolve_qwen_clstr_final_chain()` selects that checkpoint rather than step3000 and emits:

```python
assert manifest["schema_version"] == "qwen06_clstr_final_chain_v2"
assert manifest["checkpoints"]["stage4"]["path"] == str(selected_checkpoint.resolve())
assert manifest["stage4_selection"] == sha256_path(stage4_selection_path)
assert manifest["reliability"] == {
    "mode": "fixed_alpha",
    "fixed_alpha": 0.5,
    "gate_checkpoint": None,
    "reliability_sha256": selection["reliability"]["reliability_sha256"],
}
```

Add rejection tests for:

- selection self-hash mismatch;
- selected checkpoint/quality-gate/lineage disagreement;
- `release_status != ok`;
- a Stage4 checkpoint containing `initial_belief_head.*` or `unified_retriever.*`;
- router digest mismatch;
- a quality gate that lacks fixed-validation evidence;
- `reliability.mode=learned` without a gate identity;
- `reliability.mode=learned` without `audit_manifest_sha256` or with a gate/audit identity mismatch;
- fixed alpha outside `[0, 1]`.

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t6_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:45:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t6_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_quality_gate.py tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_final_chain.py tests/test_qwen_clstr_frozen_route_eval.py'
```

Expected: failures because the quality gate still accepts training-tail evidence, lineage allows router heads, and final-chain v1 hard-codes step3000.

- [ ] **Step 3: Implement fixed-alpha finalization and revised quality checks**

Add `finalize_stage4_selection()` to `clstr/stage4_validation.py`:

```python
def _require_identity(path: str | Path, expected_sha256: str, *, label: str) -> None:
    actual = sha256_path(Path(path).resolve())["sha256"]
    if actual != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 mismatch")


def _final_reliability_release_checks(
    dynamic: dict[str, Any],
    fused_summary: dict[str, Any],
) -> dict[str, bool]:
    baseline = dict(dynamic["stage2_baseline"])
    reference = max(
        float(baseline["static_macro_mrr"]),
        float(baseline["dynamic_macro_mrr"]),
    )
    baseline_by_benchmark = dict(baseline["by_benchmark"])
    fused_by_benchmark = dict(fused_summary["by_benchmark"])
    return {
        "dynamic_release_ok": dynamic.get("release_status") == "ok",
        "fused_macro_improvement": (
            float(fused_summary["balanced_macro_mrr"]) >= reference + 0.005
        ),
        "fused_per_benchmark_nonregression": all(
            float(fused_by_benchmark[benchmark]["mrr"])
            >= float(baseline_by_benchmark[benchmark]["static_mrr"]) - 0.01
            for benchmark in baseline_by_benchmark
        ),
        "zero_history_exact": fused_summary.get("zero_history_exact") is True,
    }


def finalize_stage4_selection(
    *,
    dynamic_selection_path: str | Path,
    output_path: str | Path,
    gate_report_path: str | Path | None = None,
) -> dict[str, Any]:
    dynamic = _read_self_hashed_manifest(dynamic_selection_path, "stage4_dynamic_selection_v1")
    if dynamic.get("status") != "ok":
        raise ValueError("dynamic Stage4 selection is not complete")
    if dynamic.get("release_status") not in {"ok", "action_required"}:
        raise ValueError("dynamic Stage4 selection has invalid release status")
    _require_identity(
        dynamic["selected_checkpoint_path"],
        dynamic["selected_checkpoint_sha256"],
        label="selected Stage4 checkpoint",
    )
    _require_identity(
        dynamic["selected_validation_report_path"],
        dynamic["selected_validation_report_sha256"],
        label="selected Stage4 validation report",
    )
    reliability = {
        "mode": "fixed_alpha",
        "fixed_alpha": float(dynamic["fixed_alpha_selection"]["fixed_alpha"]),
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    release_checks = _final_reliability_release_checks(
        dynamic,
        dict(dynamic["fixed_alpha_selection"]),
    )
    payload = {
        "schema_version": "stage4_selection_v1",
        "status": "ok",
        "release_status": "ok" if all(release_checks.values()) else "action_required",
        "dynamic_release_status": str(dynamic["release_status"]),
        "release_checks": release_checks,
        "selected_checkpoint_path": str(Path(dynamic["selected_checkpoint_path"]).resolve()),
        "selected_checkpoint_sha256": str(dynamic["selected_checkpoint_sha256"]),
        "selected_validation_report_path": str(Path(dynamic["selected_validation_report_path"]).resolve()),
        "selected_validation_report_sha256": str(dynamic["selected_validation_report_sha256"]),
        "router_integrity": dict(dynamic["router_integrity"]),
        "stage2_baseline": dict(dynamic["stage2_baseline"]),
        "candidate_union": dict(dynamic["candidate_union"]),
        "reliability": reliability,
        "reliability_validation": dict(dynamic["fixed_alpha_selection"]),
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload
```

The gate-report branch remains inactive until Task 7, but passing a path before that task must raise `ValueError("learned gate finalization is not implemented")`.

Replace the Stage4 quality gate's release authority with the finalized selection and selected validation report. Require:

- immutable router digests and exact static logits;
- selected-report Stage2 baseline identity matching `--stage2_baseline_report_path`;
- delta-only checkpoint validation;
- `candidate_union.final_k <= candidate_union.static_k` and exact alpha-zero Top-K parity;
- dynamic macro MRR improvement at least `0.01` over Stage2 dynamic;
- memory-active delta at least `0.01`;
- at least three nonnegative benchmark deltas;
- no dynamic per-benchmark regression worse than `0.03`;
- fused macro MRR at least `0.005` above the better Stage2 endpoint;
- no fused per-benchmark regression worse than `0.01` versus Stage2 static;
- zero-history fallback exact;
- finalized `release_status=ok` with every persisted release check true;
- `valid_or_test_used_for_training` remains false;
- `validation_used_for_checkpoint_selection` is true;
- no final benchmark or Tau2 rows used for selection.

Keep loss-window/tail Recall@5 only in `health_diagnostics`; they cannot change `status` to ok.

- [ ] **Step 4: Tighten Stage4 lineage and implement final-chain v2**

In `validate_qwen_clstr_derived_checkpoint()`, when `expected_stage == "clstr_stage4_transition_conditioned_next_skill"`, require `train_report["safe_memory_protocol_version"] == "stage4_safe_memory_v1"`, `validation_used_for_checkpoint_selection is True`, and call `validate_stage4_delta_state_dict()`. Reject any router key even if it is in the older belief-calibration allowlist.

Allow `create_derived_lineage_manifest()` to accept `stage_metadata: dict[str, Any] | None = None`; for Stage4 pass:

```python
{
    "stage4_selection": sha256_path(stage4_selection_path),
    "router_integrity": selection["router_integrity"],
    "reliability": selection["reliability"],
}
```

Set `FINAL_CHAIN_SCHEMA_VERSION = "qwen06_clstr_final_chain_v2"`. Resolve Stage4 checkpoint, validation report, quality gate, and lineage through `stage4_safe_full/stage4_selection.json`. Keep the four-role model `checkpoint_chain_digest`; add `stage4_selection=sha256_path(stage4_selection_path)` and the self-hashed `reliability` object to the final manifest.

Update `load_final_chain_manifest()` to require v2, validate `reliability_sha256` after removing only that field, and validate fixed-alpha or learned-gate identity fail-closed. In `load_final_chain_model()`, validate the Stage4 payload with `validate_stage4_delta_state_dict()` before overlay, capture the full router digest after Stage2 and after Stage4, require equality, and include both digests in `checkpoint_load`. Frozen evaluation records the declared reliability mode but keeps zero-history `alpha=0` and exact static scores.

- [ ] **Step 5: Add CLI finalizer and quality-gate arguments**

`scripts/finalize_clstr_stage4_selection.py` accepts:

```text
--dynamic_selection_path
--gate_report_path
--output_path
```

`scripts/audit_clstr_stage4_quality.py` requires:

```text
--selection_path
--stage2_baseline_report_path
```

It may retain legacy window flags only as diagnostic controls. The Qwen wrapper will use the new required paths.

- [ ] **Step 6: Submit GREEN**

Rerun the exact Step 2 suite as `q06_s4_t6_green` with a `01:00:00` limit.

Expected: all four modules pass; final-chain resolution follows the selected checkpoint and rejects stale step3000/control artifacts.

- [ ] **Step 7: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_validation.py clstr/stage4_quality_gate.py \
  clstr/qwen_clstr_lineage.py clstr/qwen_clstr_final_chain.py \
  clstr/frozen_clstr_route_eval.py scripts/finalize_clstr_stage4_selection.py \
  scripts/audit_clstr_stage4_quality.py
git diff --check -- clstr/stage4_validation.py clstr/stage4_quality_gate.py \
  clstr/qwen_clstr_lineage.py clstr/qwen_clstr_final_chain.py \
  clstr/frozen_clstr_route_eval.py scripts/finalize_clstr_stage4_selection.py \
  scripts/audit_clstr_stage4_quality.py tests/test_stage4_quality_gate.py \
  tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_final_chain.py \
  tests/test_qwen_clstr_frozen_route_eval.py
git add clstr/stage4_validation.py clstr/stage4_quality_gate.py \
  clstr/qwen_clstr_lineage.py clstr/qwen_clstr_final_chain.py \
  clstr/frozen_clstr_route_eval.py scripts/finalize_clstr_stage4_selection.py \
  scripts/audit_clstr_stage4_quality.py tests/test_stage4_quality_gate.py \
  tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_final_chain.py \
  tests/test_qwen_clstr_frozen_route_eval.py
git commit -m "feat: gate stage4 release on selected safe memory"
```

---

### Task 7: Conditional Learned Memory-Utility Gate

**Files:**

- Modify: `clstr/memory_utility_gate.py`
- Create: `clstr/memory_utility_gate_train.py`
- Create: `scripts/train_clstr_memory_utility_gate.py`
- Create: `tests/test_memory_utility_gate_train.py`
- Modify: `tests/test_memory_utility_gate.py`
- Modify: `tests/test_unified_training_readiness.py`
- Modify: `scripts/audit_clstr_memory_utility_oracle.py`
- Modify: `tests/test_memory_utility_oracle_audit.py`
- Modify: `clstr/stage4_validation.py`
- Modify: `tests/test_stage4_validation.py`

- [ ] **Step 1: Write failing model, audit gate, and promotion tests**

Add the exact scalar-gate contract to `tests/test_memory_utility_gate.py`:

```python
from clstr.memory_utility_gate import MemoryUtilityGate


def test_memory_utility_gate_uses_train_normalization_and_bounded_alpha() -> None:
    gate = MemoryUtilityGate(
        feature_mean=torch.arange(11, dtype=torch.float32),
        feature_scale=torch.full((11,), 2.0),
    )
    with torch.no_grad():
        gate.net[0].weight.zero_()
        gate.net[0].bias.fill_(0.0)

    alpha = gate(torch.arange(11, dtype=torch.float32).view(1, -1))

    assert alpha.shape == (1,)
    assert alpha.item() == pytest.approx(0.5)
    assert torch.equal(gate.feature_scale, torch.full((11,), 2.0))
```

Create `tests/test_memory_utility_gate_train.py` with complete synthetic route records whose first feature identifies whether dynamic or static should win:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from clstr.memory_utility_gate_train import (
    load_memory_utility_gate_checkpoint,
    train_memory_utility_gate,
)
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")


def _record(
    benchmark: str,
    trajectory_id: str,
    row_index: int,
    *,
    dynamic_better: bool,
    identical_endpoints: bool = False,
) -> dict:
    if identical_endpoints:
        static_logits = dynamic_logits = [4.0, 0.0]
        feature = 0.0
    elif dynamic_better:
        static_logits, dynamic_logits, feature = [0.0, 4.0], [5.0, 0.0], 1.0
    else:
        static_logits, dynamic_logits, feature = [5.0, 0.0], [0.0, 20.0], -1.0
    return {
        "schema_version": "memory_utility_route_record_v1",
        "row_digest": hashlib.sha256(
            f"{benchmark}|{trajectory_id}|{row_index}".encode("utf-8")
        ).hexdigest(),
        "benchmark": benchmark,
        "trajectory_id": trajectory_id,
        "causal_update_count": 1,
        "features": [feature] + [0.0] * 10,
        "static_logits": static_logits,
        "dynamic_logits": dynamic_logits,
        "valid_mask": [True, True],
        "positive_mask": [True, False],
    }


def _records(prefix: str, *, identical_endpoints: bool = False) -> list[dict]:
    return [
        _record(
            benchmark,
            f"{prefix}-{trajectory_index}",
            trajectory_index,
            dynamic_better=(trajectory_index % 2 == 0),
            identical_endpoints=identical_endpoints,
        )
        for benchmark in BENCHMARKS
        for trajectory_index in range(4)
    ]


def _audit(*, recommended: bool, gate_route_manifest: dict) -> dict:
    route_records = _records("train")
    payload = {
        "status": "ok",
        "learned_gate_recommended": recommended,
        "recommendation_blockers": [] if recommended else ["utility_not_predictable"],
        "split": {
            "train_trajectory_ids": [
                f"{benchmark}::train-{trajectory_index}"
                for benchmark in BENCHMARKS
                for trajectory_index in range(3)
            ],
            "dev_trajectory_ids": [f"{benchmark}::train-3" for benchmark in BENCHMARKS],
        },
        "count_feature_caps": {
            "causal_update_count": 4.0,
            "valid_candidate_count": 64.0,
            "derived_from": "train_trajectories_only",
        },
        "route_record_source": {
            "record_count": len(route_records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in route_records]
            ),
        },
        "route_manifests": [
            {
                "manifest_path": gate_route_manifest["path"],
                "manifest_file_identity": dict(gate_route_manifest),
            }
        ],
        "route_manifest_identity": {
            "model_checkpoint_chain_digest": "selected-stage4-chain"
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload


def _dynamic_selection(
    tmp_path: Path,
    validation_records: list[dict],
    *,
    fixed_macro_mrr: float,
) -> dict:
    checkpoint = tmp_path / "selected-stage4.pt"
    checkpoint.write_bytes(b"stage4-delta")
    validation_report = tmp_path / "selected-validation.json"
    validation_report.write_text("{}", encoding="utf-8")
    validation_records_path = tmp_path / "selected-validation.route_records.jsonl"
    validation_records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation_records),
        encoding="utf-8",
    )
    gate_records = _records("train")
    gate_records_path = tmp_path / "selected-gate.route_records.jsonl"
    gate_records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in gate_records),
        encoding="utf-8",
    )
    gate_manifest_path = tmp_path / "selected-gate.route_manifest.json"
    gate_manifest_path.write_text("{}", encoding="utf-8")
    payload = {
        "schema_version": "stage4_dynamic_selection_v1",
        "status": "ok",
        "release_status": "ok",
        "selected_step": 800,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(validation_report.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_report)["sha256"],
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "report_sha256": "stage2-baseline-sha256",
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.60,
            "dynamic_macro_mrr": 0.60,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60, "dynamic_mrr": 0.60}
                for benchmark in BENCHMARKS
            },
        },
        "candidate_union": {
            "static_k": 500,
            "dynamic_extra_k": 64,
            "final_k": 64,
        },
        "fixed_alpha_selection": {
            "fixed_alpha": 0.0,
            "balanced_macro_mrr": fixed_macro_mrr,
            "zero_history_exact": True,
            "by_benchmark": {
                benchmark: {"mrr": fixed_macro_mrr} for benchmark in BENCHMARKS
            },
        },
        "validation_route_records": {
            **sha256_path(validation_records_path),
            "record_count": len(validation_records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in validation_records]
            ),
        },
        "gate_route_records": {
            **sha256_path(gate_records_path),
            "record_count": len(gate_records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in gate_records]
            ),
        },
        "gate_route_manifest": sha256_path(gate_manifest_path),
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload


def test_gate_training_refuses_a_negative_audit(tmp_path: Path) -> None:
    validation_records = _records("validation")
    selection = _dynamic_selection(
        tmp_path,
        validation_records,
        fixed_macro_mrr=0.75,
    )
    report = train_memory_utility_gate(
        route_records=_records("train"),
        audit_report=_audit(
            recommended=False,
            gate_route_manifest=selection["gate_route_manifest"],
        ),
        validation_route_records=validation_records,
        dynamic_selection=selection,
        output_dir=tmp_path,
        max_steps=20,
        learning_rate=0.05,
    )
    assert report["status"] == "not_recommended"
    assert report["checkpoint_path"] is None
    assert (tmp_path / "gate_report.json").is_file()
    assert not (tmp_path / "memory_utility_gate.pt").exists()


def test_gate_training_writes_audit_bound_checkpoint_when_validation_improves(
    tmp_path: Path,
) -> None:
    validation_records = _records("validation")
    selection = _dynamic_selection(
        tmp_path,
        validation_records,
        fixed_macro_mrr=0.75,
    )
    audit = _audit(
        recommended=True,
        gate_route_manifest=selection["gate_route_manifest"],
    )
    report = train_memory_utility_gate(
        route_records=_records("train"),
        audit_report=audit,
        validation_route_records=validation_records,
        dynamic_selection=selection,
        output_dir=tmp_path,
        max_steps=100,
        learning_rate=0.05,
    )
    payload = torch.load(report["checkpoint_path"], map_location="cpu")
    gate, load_report = load_memory_utility_gate_checkpoint(
        report["checkpoint_path"],
        expected_sha256=report["checkpoint_sha256"],
        expected_audit_sha256=audit["manifest_sha256"],
    )
    assert report["status"] == "ok"
    assert report["promoted"] is True
    assert payload["stage"] == "clstr_memory_utility_gate"
    assert payload["schema_version"] == "memory_utility_gate_checkpoint_v1"
    assert payload["feature_schema"] == "memory_utility_features_v1"
    assert payload["zero_history_fallback"] == "exact_static"
    assert payload["audit_manifest_sha256"] == audit["manifest_sha256"]
    assert payload["memory_utility_gate_state_dict"]
    assert gate.training is False
    assert load_report["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert load_report["feature_update_count_cap"] == 4.0
    assert load_report["feature_candidate_count_cap"] == 64.0


def test_gate_training_keeps_fixed_alpha_when_validation_has_no_advantage(
    tmp_path: Path,
) -> None:
    validation_records = _records("validation", identical_endpoints=True)
    selection = _dynamic_selection(
        tmp_path,
        validation_records,
        fixed_macro_mrr=1.0,
    )
    report = train_memory_utility_gate(
        route_records=_records("train"),
        audit_report=_audit(
            recommended=True,
            gate_route_manifest=selection["gate_route_manifest"],
        ),
        validation_route_records=validation_records,
        dynamic_selection=selection,
        output_dir=tmp_path,
        max_steps=100,
        learning_rate=0.05,
    )
    assert report["status"] == "ok"
    assert report["promoted"] is False
    assert report["checkpoint_path"] is None
    assert (tmp_path / "gate_report.json").is_file()
    assert not (tmp_path / "memory_utility_gate.pt").exists()
```

In `tests/test_memory_utility_oracle_audit.py`, add `finalize_memory_utility_oracle_audit_report` to the existing audit-script import and import `sha256_path` from `clstr.qwen_clstr_lineage`, then add:

```python
def test_finalized_oracle_audit_is_self_hashed_and_pins_source_rows(tmp_path: Path) -> None:
    records = _qualifying_records()
    route_manifest = tmp_path / "route-manifest.json"
    route_manifest.write_text("{}", encoding="utf-8")
    base = run_memory_utility_oracle_audit(
        records,
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=10,
        seed=17,
    )
    finalized = finalize_memory_utility_oracle_audit_report(
        base,
        records=records,
        route_manifests=[{"manifest_path": str(route_manifest)}],
        route_manifest_identity={"model_checkpoint_chain_digest": "chain-a"},
    )
    recorded = finalized.pop("manifest_sha256")
    assert recorded == canonical_digest(finalized)
    assert finalized["route_record_source"] == {
        "record_count": len(records),
        "row_digest_sha256": canonical_digest([row["row_digest"] for row in records]),
    }
    assert finalized["route_manifests"][0]["manifest_file_identity"] == sha256_path(
        route_manifest
    )
```

Extend `tests/test_stage4_validation.py` with a non-null `gate_report_path` case. It selects learned mode only when `promoted=True`, the selected dynamic checkpoint digest matches, the gate checkpoint SHA matches, the audit SHA matches the report, and every benchmark regression is within `0.01`. The promoted reliability object is exactly:

```python
expected_reliability = {
    "mode": "learned",
    "fixed_alpha": None,
    "gate_checkpoint": {
        "path": str(gate_checkpoint.resolve()),
        "sha256": sha256_path(gate_checkpoint)["sha256"],
    },
    "audit_manifest_sha256": audit["manifest_sha256"],
}
expected_reliability["reliability_sha256"] = canonical_digest(expected_reliability)
assert selection["reliability"] == expected_reliability
```

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t7_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:30:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t7_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py tests/test_stage4_validation.py tests/test_memory_utility_oracle_audit.py tests/test_unified_training_readiness.py -k "gate or learned or finalize or memory_utility_gate"'
```

Expected: failures on the missing class, trainer module, checkpoint loader, and learned finalization branch.

- [ ] **Step 3: Implement the normalized scalar gate and strict loader**

Add to `clstr/memory_utility_gate.py`:

```python
class MemoryUtilityGate(torch.nn.Module):
    def __init__(
        self,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        width = len(MEMORY_UTILITY_FEATURE_NAMES)
        mean = torch.zeros(width, dtype=torch.float32) if feature_mean is None else feature_mean.float()
        scale = torch.ones(width, dtype=torch.float32) if feature_scale is None else feature_scale.float()
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("memory utility normalization must match feature schema")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("memory utility normalization must be finite with positive scale")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.net = torch.nn.Sequential(torch.nn.Linear(width, 1), torch.nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.size(1)) != len(MEMORY_UTILITY_FEATURE_NAMES):
            raise ValueError("memory utility gate features do not match schema")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.net(normalized).squeeze(-1)
```

Add `load_memory_utility_gate_checkpoint(path, *, expected_sha256=None, expected_audit_sha256=None)` that verifies the file SHA-256 before `torch.load`, then checks stage, schema, feature schema, exact-static zero-history policy, `reliability_changes_memory_state=False`, optional audit SHA, finite positive train-derived feature caps, normalization tensors, and nonempty state dict before returning an eval-mode gate plus a load report containing `checkpoint_path`, `checkpoint_sha256`, `audit_manifest_sha256`, `feature_update_count_cap`, and `feature_candidate_count_cap`.

- [ ] **Step 4: Implement audit-bound gate training and promotion**

Add this finalizer to `scripts/audit_clstr_memory_utility_oracle.py` and call it in `main()` after loading route manifests and before writing JSON:

```python
def finalize_memory_utility_oracle_audit_report(
    report: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    route_manifests: list[dict[str, Any]],
    route_manifest_identity: dict[str, Any],
) -> dict[str, Any]:
    normalized_manifests = [
        {
            **manifest,
            "manifest_file_identity": sha256_path(manifest["manifest_path"]),
        }
        for manifest in route_manifests
    ]
    payload = {
        **report,
        "route_manifests": normalized_manifests,
        "route_manifest_identity": route_manifest_identity,
        "route_record_source": {
            "record_count": len(records),
            "row_digest_sha256": canonical_digest(
                [str(row["row_digest"]) for row in records]
            ),
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload
```

Import `sha256_path` from `clstr.qwen_clstr_lineage` in the audit script.

Create `clstr/memory_utility_gate_train.py` with:

```python
def train_memory_utility_gate(
    *,
    route_records: list[dict[str, Any]],
    audit_report: dict[str, Any],
    validation_route_records: list[dict[str, Any]],
    dynamic_selection: dict[str, Any],
    output_dir: str | Path,
    max_steps: int = 300,
    learning_rate: float = 1.0e-2,
    seed: int = 17,
) -> dict[str, Any]:
```

Required behavior:

1. require a valid self-hashed oracle audit with `audit_report.status == "ok"`, pinned route-manifest identity, and pinned route-record source;
2. return `not_recommended` without creating a checkpoint when `learned_gate_recommended` is false;
3. require a valid self-hashed `stage4_dynamic_selection_v1` payload with `status=ok` and `release_status=ok`;
4. require one audit route-manifest `manifest_file_identity` to equal `dynamic_selection["gate_route_manifest"]` exactly;
5. require `route_records` count and ordered row-digest SHA to match `dynamic_selection["gate_route_records"]` and the oracle audit's recorded source identity;
6. select only route records whose `benchmark::trajectory_id` keys belong to the audit train partition;
7. overwrite the two count-derived feature slots for audit-train, audit-dev, and fixed-validation records using the audit's train-only caps, raw `causal_update_count`, and each row's valid-candidate count;
8. initialize the linear gate deterministically from `seed`, record the seed, derive feature mean/std from those transformed train records only, and clamp std to at least `1e-6`;
9. optimize multi-positive NLL over `fuse_route_scores(static, dynamic, gate(features), valid)` with branch tensors detached;
10. use the audit dev partition for calibration diagnostics;
11. require the validation record count and canonical ordered row-digest SHA to match `dynamic_selection["validation_route_records"]`;
12. evaluate the trained gate on transformed `validation_route_records` only, with zero-history alpha clamped to zero, and store balanced/per-benchmark MRR plus `zero_history_exact=True` as `learned_validation_summary`;
13. compare learned balanced/per-benchmark MRR to `dynamic_selection["fixed_alpha_selection"]`;
14. promote only for at least `0.005` macro-MRR improvement and no benchmark regression worse than `0.01`;
15. atomically write a self-hashed `gate_report.json` in all cases;
16. write `memory_utility_gate.pt` only when promoted, then record its path and SHA-256 in the report.

The checkpoint payload is:

```python
{
    "stage": "clstr_memory_utility_gate",
    "schema_version": "memory_utility_gate_checkpoint_v1",
    "feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
    "feature_names": list(MEMORY_UTILITY_FEATURE_NAMES),
    "audit_manifest_sha256": audit_report["manifest_sha256"],
    "zero_history_fallback": "exact_static",
    "reliability_changes_memory_state": False,
    "selected_stage4_checkpoint_sha256": dynamic_selection["selected_checkpoint_sha256"],
    "training_seed": int(seed),
    "feature_update_count_cap": float(
        audit_report["count_feature_caps"]["causal_update_count"]
    ),
    "feature_candidate_count_cap": float(
        audit_report["count_feature_caps"]["valid_candidate_count"]
    ),
    "memory_utility_gate_state_dict": gate.state_dict(),
    "validation": {
        "learned": learned_validation_summary,
        "fixed_alpha": dynamic_selection["fixed_alpha_selection"],
        "balanced_macro_mrr_improvement": (
            learned_validation_summary["balanced_macro_mrr"]
            - dynamic_selection["fixed_alpha_selection"]["balanced_macro_mrr"]
        ),
    },
}
```

Implement the CLI with exact arguments `--route_records_path`, `--audit_report_path`, `--validation_route_records_path`, `--dynamic_selection_path`, and `--output_dir`. It verifies every persisted JSON/JSONL identity before calling the trainer. Exit zero for `ok` and `not_recommended`; exit two for malformed or inconsistent artifacts.

Enable the learned branch in `finalize_stage4_selection()` only when the self-hashed gate report and checkpoint pass all identity/promotion checks. For a promoted gate, build the learned reliability object, add `reliability_sha256=canonical_digest(reliability_without_hash)`, set `reliability_validation` to `gate_report["validation"]["learned"]`, recompute `_final_reliability_release_checks(dynamic, reliability_validation)`, and derive the final `release_status` from those checks; do not inherit the fixed-alpha release result. Otherwise retain the fixed-alpha object and fixed-alpha validation summary without silently referencing a gate. A report that claims `promoted=True` but has a missing/mismatched checkpoint is a hard error; an honest `promoted=False` report is a valid fixed-alpha fallback.

- [ ] **Step 5: Submit GREEN**

Rerun Step 2 as `q06_s4_t7_green` with a `00:45:00` limit.

Expected: all gate, finalization, and readiness tests pass; negative audits create no checkpoint.

- [ ] **Step 6: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py \
  scripts/train_clstr_memory_utility_gate.py scripts/audit_clstr_memory_utility_oracle.py \
  clstr/stage4_validation.py
git diff --check -- clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py \
  scripts/train_clstr_memory_utility_gate.py scripts/audit_clstr_memory_utility_oracle.py \
  clstr/stage4_validation.py \
  tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py \
  tests/test_memory_utility_oracle_audit.py tests/test_stage4_validation.py \
  tests/test_unified_training_readiness.py
git add clstr/memory_utility_gate.py clstr/memory_utility_gate_train.py \
  scripts/train_clstr_memory_utility_gate.py scripts/audit_clstr_memory_utility_oracle.py \
  clstr/stage4_validation.py \
  tests/test_memory_utility_gate.py tests/test_memory_utility_gate_train.py \
  tests/test_memory_utility_oracle_audit.py tests/test_stage4_validation.py \
  tests/test_unified_training_readiness.py
git commit -m "feat: calibrate memory reliability after stage4 selection"
```

---

### Task 8: Reliability Artifact Through Native, Frozen, and ALFWorld Evaluation

**Files:**

- Modify: `clstr/toolbench_full_clstr_route_eval.py`
- Modify: `clstr/global_pool_route_eval.py`
- Modify: `clstr/tau2_route_eval.py`
- Modify: `clstr/toolsandbox_route_eval.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `clstr/native_benchmark_checkpoint_adapter.py`
- Modify: `clstr/qwen_clstr_multibench_submit.py`
- Modify: `clstr/qwen_clstr_multibench_report.py`
- Modify: `scripts/run_toolbench_g3_full_clstr_route_eval.py`
- Modify: `scripts/run_global_pool_clstr_route_eval.py`
- Modify: `scripts/run_tau2_full_clstr_route_eval.py`
- Modify: `scripts/run_toolsandbox_full_clstr_route_eval.py`
- Modify: `scripts/run_alfworld_clstr_eval.py`
- Modify: `scripts/sbatch/run_qwen06_clstr_native_route_eval.sh`
- Modify: `scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh`
- Modify: `tests/test_toolbench_full_clstr_route_eval.py`
- Modify: `tests/test_global_pool_route_eval.py`
- Modify: `tests/test_tau2_route_eval.py`
- Modify: `tests/test_toolsandbox_route_eval.py`
- Modify: `tests/test_alfworld_eval.py`
- Modify: `tests/test_native_benchmark_checkpoint_adapter.py`
- Modify: `tests/test_qwen_clstr_multibench_submit.py`
- Modify: `tests/test_qwen_clstr_multibench_report.py`
- Modify: `tests/test_qwen_clstr_frozen_route_eval.py`
- Modify: `tests/test_qwen_clstr_training_launchers.py`

- [ ] **Step 1: Write failing learned-mode pass-through tests**

For ToolBench/global/Tau2/ToolSandbox, monkeypatch `load_memory_utility_gate_checkpoint()` to return a sentinel module and assert the Stage4 evaluator call receives:

```python
assert stage4_call["reliability_mode"] == "learned"
assert stage4_call["memory_utility_gate"] is sentinel_gate
assert stage4_call["fixed_alpha"] == pytest.approx(1.0)
assert stage4_call["feature_update_count_cap"] == pytest.approx(4.0)
assert stage4_call["feature_candidate_count_cap"] == pytest.approx(64.0)
```

The monkeypatched loader report for this test returns the same two caps; evaluator entrypoints must not fall back to their legacy default of `1.0` in learned mode.

Assert learned mode without a checkpoint raises `ValueError("learned reliability requires a gate checkpoint")`, while static/dynamic/fixed-alpha/heuristic do not load a gate.

Extend `tests/test_alfworld_eval.py` with a two-candidate fake model where static ranks candidate A first and one real post-action update ranks candidate B first. The fake records the action, next observation, and next state supplied to the updater. Assert:

```python
static_scorer.reset_episode_batch([state_t])
dynamic_scorer.reset_episode_batch([state_t])
fixed_scorer.reset_episode_batch([state_t])
zero_history_learned_scorer.reset_episode_batch([state_t])

static_scores = static_scorer([state_t], [["a", "b"]])
zero_history_scores = zero_history_learned_scorer([state_t], [["a", "b"]])
dynamic_scorer.observe_transitions(
    chosen_actions=["b"],
    next_observation_texts=["real next observation"],
    next_state_texts=[state_t_plus_1],
    active_mask=[True],
)
fixed_scorer.observe_transitions(
    chosen_actions=["b"],
    next_observation_texts=["real next observation"],
    next_state_texts=[state_t_plus_1],
    active_mask=[True],
)
dynamic_scores = dynamic_scorer([state_t_plus_1], [["a", "b"]])
fixed_scores = fixed_scorer([state_t_plus_1], [["a", "b"]])

assert static_scores.argmax(dim=-1).item() == 0
assert dynamic_scores.argmax(dim=-1).item() == 1
assert torch.equal(zero_history_scores, static_scores)
assert not torch.equal(fixed_scores, static_scores)
assert dynamic_scorer.last_transition_inputs == {
    "action_text": "b",
    "next_observation_text": "real next observation",
    "next_state_text": state_t_plus_1,
}
assert dynamic_scorer.last_metadata[0]["causal_update_count"] == 1
assert dynamic_scorer.last_metadata[0]["memory_protocol"] == "stateful_post_action_v1"
```

Update the ALFWorld structural-report fixture to use `memory_protocol="stateful_post_action_v1"` and per-step `causal_update_count`, then assert `validate_alfworld_stage()` accepts real stateful updates and rejects the legacy pattern where `uses_recurrent_m_t=True` is supported only by `clstr_replay_prefix_used_count`.

Update multibench tests to require every stage export:

```text
RELIABILITY_MODE
FIXED_ALPHA
MEMORY_UTILITY_GATE_CHECKPOINT_PATH
MEMORY_UTILITY_GATE_CHECKPOINT_SHA256
MEMORY_UTILITY_GATE_AUDIT_SHA256
```

For fixed-alpha chains, all three gate/audit exports are empty strings. For learned chains, they match the final-chain reliability identity.

For both ALFWorld splits, require `MEMORY_PROTOCOL=stateful_post_action_v1` and forward `--memory_protocol stateful_post_action_v1`. Final-chain v2 launcher tests reject `REPLAY_PREFIX_MAX_STEPS` as the active memory protocol; that variable may remain only in explicitly legacy scorer tests.

Extend `tests/test_native_benchmark_checkpoint_adapter.py` with a safe-memory mode that captures `router_state_digest(model, scope="full")` before and after the Stage4 overlay, asserts equality, and rejects a Stage4 payload containing `initial_belief_head.weight`. Add `require_safe_memory_delta: bool = False` to `restore_native_benchmark_checkpoint_chain()`; final-chain v2 callers pass `True`, which validates `model_state_dict` through `validate_stage4_delta_state_dict()` before loading and records `router_digest_unchanged=True` afterward. Legacy direct callers retain the default only for backward-compatible diagnostics.

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t8_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=01:00:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t8_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_toolbench_full_clstr_route_eval.py tests/test_global_pool_route_eval.py tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py tests/test_alfworld_eval.py tests/test_native_benchmark_checkpoint_adapter.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_training_launchers.py -k "reliability or learned or final_chain or unified_memory_admissible or stateful_post_action or safe_memory_delta"'
```

Expected: failures on missing learned API arguments, gate loading, ALFWorld fusion, and exports.

- [ ] **Step 3: Add one validated reliability resolver**

Add a private helper to `clstr/memory_utility_gate_train.py`:

```python
def resolve_reliability_gate(
    *,
    reliability_mode: str,
    gate_checkpoint_path: str | Path | None,
    expected_gate_sha256: str | None = None,
    expected_audit_sha256: str | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    mode = str(reliability_mode or "dynamic")
    if mode not in RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability mode: {mode}")
    normalized_path = (
        None
        if gate_checkpoint_path is None or not str(gate_checkpoint_path).strip()
        else Path(gate_checkpoint_path)
    )
    if mode != "learned":
        if normalized_path is not None:
            raise ValueError("non-learned reliability must not load a gate checkpoint")
        return None, {"mode": mode, "gate_loaded": False}
    if normalized_path is None:
        raise ValueError("learned reliability requires a gate checkpoint")
    gate, report = load_memory_utility_gate_checkpoint(
        normalized_path,
        expected_sha256=(
            None
            if expected_gate_sha256 is None or not str(expected_gate_sha256).strip()
            else str(expected_gate_sha256).strip()
        ),
        expected_audit_sha256=(
            None
            if expected_audit_sha256 is None or not str(expected_audit_sha256).strip()
            else str(expected_audit_sha256).strip()
        ),
    )
    if device is not None:
        gate = gate.to(device)
    gate.eval()
    return gate, {"mode": mode, "gate_loaded": True, **report}
```

Use this resolver once per evaluator entrypoint with the resolved model device, then pass the module and the loader report's two feature caps to `evaluate_logged_online_stage4_rows()`. For non-learned modes, retain the explicitly configured caps already used by that evaluator. Every final-chain v2 native/ALFWorld restoration passes `require_safe_memory_delta=True`; do not infer it merely from a filename.

- [ ] **Step 4: Implement ALFWorld static/dynamic fusion**

Replace history-text replay in `ClstrUnifiedMemoryAdmissibleActionScorer` with an episode-stateful memory contract. Extend `__init__()` with:

```python
reliability_mode: str = "dynamic",
fixed_alpha: float = 1.0,
memory_utility_gate: torch.nn.Module | None = None,
feature_update_count_cap: float = 1.0,
feature_candidate_count_cap: float = 1.0,
```

Add `reset_episode_batch(state_texts)`, which encodes the initial states, sets `dynamic_memory=model.initial_belief(h_0)`, and zeros one causal-update counter per episode. Add `observe_transitions(chosen_actions, next_observation_texts, next_state_texts, active_mask)`, which maps each chosen action to its skill, encodes the real action/next-observation/next-state values, calls the shared post-action memory helper, and updates only active episode slots. It must reject calls before reset, batch-size drift, unmapped active actions, or missing next observations/states.

Inside `__call__()`, require an initialized episode batch, encode the current state, compute `static_memory=model.initial_belief(h_t)`, and use the stored recurrent memory as `dynamic_memory`. Gather both logits over the same mapped candidate rows, compute versioned features with the stored causal-update counts and validated caps, resolve raw/effective alpha, and call `fuse_route_scores()`.

Modify `run_alfworld_closed_loop_eval()` without changing other scorer APIs: after each environment reset, call `reset_episode_batch()` when the scorer exposes it; immediately after `eval_env.step(execute_actions)`, build the true next state texts and call `observe_transitions()` with the executed actions, returned observations, and prior-active mask. Do not reconstruct safe-memory state from the rendered `history:` string. Keep `_alfworld_replay_prefix_from_state_text()` only for explicitly named legacy diagnostics, never for final-chain v2.

Update `validate_alfworld_stage()` and its Markdown/report fields to require `memory_protocol=stateful_post_action_v1`, sum positive `causal_update_count` values as recurrent evidence, and reject replay-prefix-only evidence for final-chain v2. Remove the old full-run requirement that `replay_prefix_max_steps == 6`; retain that field only when validating an explicitly legacy artifact.

Record per-row metadata:

```python
{
    "memory_utility_reliability_mode": self.reliability_mode,
    "memory_utility_alpha": float(effective_alpha[row_idx].item()),
    "zero_history_fallback": "exact_static",
    "uses_recurrent_m_t": bool(self.causal_update_count[row_idx].item() > 0),
    "causal_update_count": int(self.causal_update_count[row_idx].item()),
    "memory_protocol": "stateful_post_action_v1",
}
```

The ALFWorld metrics/report payload records `memory_active_protocol=True`, `memory_protocol="stateful_post_action_v1"`, and aggregate recurrent-update counts derived from run metadata. It must not label the safe run as `replay_prefix` or use replay depth as causal evidence.

Do not change the concrete-action scorer in this task because the Qwen final protocol uses `unified_memory_admissible_action`.

- [ ] **Step 5: Thread final-chain reliability through multibench jobs and wrappers**

In `_common_exports()`, read `final_chain["reliability"]` and export the declared mode plus optional gate identity. Export the selected numeric alpha for fixed-alpha mode and `FIXED_ALPHA=1.0` for learned mode (the learned evaluator ignores that sentinel); never serialize Python `None` into a CLI float. The native and ALFWorld wrappers forward the expected identity to Python, where validation uses the repository's `sha256_path()` convention; do not compare it with raw shell `sha256sum`. Forward:

```text
--reliability_mode
--fixed_alpha
--memory_utility_gate_checkpoint_path
--expected_memory_utility_gate_checkpoint_sha256
--expected_memory_utility_gate_audit_sha256
--memory_protocol
```

The reliability arguments apply to native and ALFWorld entrypoints; `--memory_protocol` is ALFWorld-only.

Frozen comparable evaluation does not load the gate because every row has zero causal history; it records `zero_history_fallback=exact_static` and the final-chain reliability identity in its report.

- [ ] **Step 6: Submit GREEN and benchmark-adapter regression**

Rerun Step 2 as `q06_s4_t8_green`, then submit the complete affected benchmark suite with a `02:00:00` limit:

```bash
sbatch --parsable \
  --job-name=q06_s4_t8_reg \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=02:00:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t8_reg-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_toolbench_full_clstr_route_eval.py tests/test_global_pool_route_eval.py tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py tests/test_alfworld_eval.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_frozen_route_eval.py tests/test_native_benchmark_checkpoint_adapter.py'
```

Expected: all tests pass; learned mode is fail-closed and zero-history frozen rows are exact static.

- [ ] **Step 7: Run local checks and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/toolbench_full_clstr_route_eval.py clstr/global_pool_route_eval.py \
  clstr/tau2_route_eval.py clstr/toolsandbox_route_eval.py clstr/alfworld_eval.py \
  clstr/native_benchmark_checkpoint_adapter.py clstr/qwen_clstr_multibench_submit.py \
  clstr/qwen_clstr_multibench_report.py \
  scripts/run_toolbench_g3_full_clstr_route_eval.py \
  scripts/run_global_pool_clstr_route_eval.py scripts/run_tau2_full_clstr_route_eval.py \
  scripts/run_toolsandbox_full_clstr_route_eval.py scripts/run_alfworld_clstr_eval.py
bash -n scripts/sbatch/run_qwen06_clstr_native_route_eval.sh \
  scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh
git diff --check
git add clstr/toolbench_full_clstr_route_eval.py clstr/global_pool_route_eval.py \
  clstr/tau2_route_eval.py clstr/toolsandbox_route_eval.py clstr/alfworld_eval.py \
  clstr/native_benchmark_checkpoint_adapter.py \
  clstr/qwen_clstr_multibench_submit.py clstr/qwen_clstr_multibench_report.py \
  scripts/run_toolbench_g3_full_clstr_route_eval.py \
  scripts/run_global_pool_clstr_route_eval.py scripts/run_tau2_full_clstr_route_eval.py \
  scripts/run_toolsandbox_full_clstr_route_eval.py scripts/run_alfworld_clstr_eval.py \
  scripts/sbatch/run_qwen06_clstr_native_route_eval.sh \
  scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh \
  tests/test_toolbench_full_clstr_route_eval.py tests/test_global_pool_route_eval.py \
  tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py \
  tests/test_alfworld_eval.py tests/test_qwen_clstr_multibench_submit.py \
  tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_frozen_route_eval.py \
  tests/test_native_benchmark_checkpoint_adapter.py \
  tests/test_qwen_clstr_training_launchers.py
git commit -m "feat: apply selected memory reliability in evaluation"
```

---

### Task 9: Qwen Safe-Memory Stage4 Launcher and Artifact Chain

**Files:**

- Modify: `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`
- Modify: `tests/test_qwen_clstr_training_launchers.py`
- Modify: `tests/test_stage4_quality_gate.py`
- Modify: `tests/test_qwen_clstr_final_chain.py`

- [ ] **Step 1: Write failing launcher-contract assertions**

Replace the moving-static Stage4 expectations with:

```python
text = _launcher("run_qwen06_clstr_stage4_train.sh")
assert "FULL_MAX_STEPS=3000" in text
assert "OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_safe_full}" in text
assert "LEARNING_RATE=3.0e-5" in text
assert "MINIMUM_LEARNING_RATE=3.0e-6" in text
assert "LEARNING_RATE_WARMUP_FRACTION=0.05" in text
assert "VALIDATION_INTERVAL_STEPS=400" in text
assert "FULL_VALIDATION_ROWS_PER_BENCHMARK=256" in text
assert "FULL_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=128" in text
assert "FULL_GATE_ROWS_PER_BENCHMARK=512" in text
assert "stage4_dynamic_selection.json" in text
assert "audit_clstr_memory_utility_oracle.py" in text
assert "train_clstr_memory_utility_gate.py" in text
assert "finalize_clstr_stage4_selection.py" in text
assert "stage4_selection.json" in text
assert "audit_clstr_stage4_quality.py" in text
assert "create-derived" in text
assert "clstr_stage4_act-step3000.pt" not in text
assert "train_encoder_backbone" not in text
assert "unfreeze" not in text.lower()
```

Add a smoke assertion for `SMOKE_VALIDATION_ROWS_PER_BENCHMARK=4`, `SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=1`, and `SMOKE_GATE_ROWS_PER_BENCHMARK=16`.

- [ ] **Step 2: Submit RED**

```bash
sbatch --parsable \
  --job-name=q06_s4_t9_red \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=00:30:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_t9_red-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_clstr_training_launchers.py tests/test_stage4_quality_gate.py tests/test_qwen_clstr_final_chain.py'
```

Expected: failures because the launcher still writes `stage4_full`, uses LR `1e-4`, hard-codes step3000, and lacks selection/gate finalization.

- [ ] **Step 3: Implement separate smoke/full safe output profiles**

Keep the existing control directory untouched. Use:

```bash
if [[ "${FULL_RUN}" == "1" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_safe_full}
  VALIDATION_ROWS_PER_BENCHMARK=${FULL_VALIDATION_ROWS_PER_BENCHMARK}
  MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${FULL_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}
  GATE_ROWS_PER_BENCHMARK=${FULL_GATE_ROWS_PER_BENCHMARK}
else
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_safe_smoke}
  VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_VALIDATION_ROWS_PER_BENCHMARK}
  MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK}
  GATE_ROWS_PER_BENCHMARK=${SMOKE_GATE_ROWS_PER_BENCHMARK}
fi
```

Export the safe LR/scheduler/data-protocol parameters and run the lower Stage4 trainer. Require `stage4_dynamic_selection.json` afterward.

Resolve `SELECTED_GATE_ROUTE_RECORDS_PATH`, `SELECTED_GATE_ROUTE_MANIFEST_PATH`, and `SELECTED_VALIDATION_ROUTE_RECORDS_PATH` with a bounded Python JSON verifier that checks the selection self-hash and file identities before printing values. Do not depend on unavailable `jq` or source untrusted JSON as shell code.

For full mode:

1. read and verify `stage4_dynamic_selection.json`; if `release_status=action_required`, skip learned-gate calibration and preserve that release outcome;
2. when `release_status=ok`, run `audit_clstr_memory_utility_oracle.py --record_manifest "${SELECTED_GATE_ROUTE_MANIFEST_PATH}" --output_path "${OUTPUT_DIR}/memory_utility_oracle_audit.json"` on the selected-checkpoint training-source records; accept exit `0` as audit-complete and exit `2` only when the persisted self-hashed report says `status=action_required`, treating the latter as an explicit fixed-alpha fallback;
3. only for an audit with `status=ok`, run `train_clstr_memory_utility_gate.py --route_records_path "${SELECTED_GATE_ROUTE_RECORDS_PATH}" --audit_report_path "${OUTPUT_DIR}/memory_utility_oracle_audit.json" --validation_route_records_path "${SELECTED_VALIDATION_ROUTE_RECORDS_PATH}" --dynamic_selection_path "${OUTPUT_DIR}/stage4_dynamic_selection.json" --output_dir "${OUTPUT_DIR}/memory_utility_gate"`; `learned_gate_recommended=false` returns success without a checkpoint;
4. call `finalize_clstr_stage4_selection.py` with the self-hashed gate report when present;
5. audit the finalized selection and selected checkpoint;
6. create Stage4 lineage from the selected checkpoint, not the final optimizer step;
7. write `stage4_quality_gate.json`, `lineage.json`, and `stage4_selection.json` under `stage4_safe_full`.

For smoke mode, finalize the best fixed alpha without running learned-gate qualification, then run only the mechanical/static-safety gate. A smoke `release_status=action_required` is allowed because four-row validation is not release evidence; smoke must never generate or replace the full final-chain manifest.

- [ ] **Step 4: Submit GREEN and shell checks**

Rerun Step 2 as `q06_s4_t9_green`.

Then locally run:

```bash
bash -n scripts/sbatch/run_qwen06_clstr_stage4_train.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage4_act_train.py --help >/dev/null
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/finalize_clstr_stage4_selection.py --help >/dev/null
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/train_clstr_memory_utility_gate.py --help >/dev/null
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit launcher integration**

```bash
git add scripts/sbatch/run_qwen06_clstr_stage4_train.sh \
  tests/test_qwen_clstr_training_launchers.py tests/test_stage4_quality_gate.py \
  tests/test_qwen_clstr_final_chain.py
git commit -m "feat: orchestrate qwen safe-memory stage4"
```

---

### Task 10: Full Verification, Safe Training, and Benchmark Handoff

**Files:**

- Modify: `.planning/2026-07-11-qwen-handoff-acceleration/progress.md`
- Modify: `.planning/2026-07-11-qwen-handoff-acceleration/findings.md`
- Modify only if verification exposes a defect: files from Tasks 1-9.

- [ ] **Step 1: Submit the focused implementation regression**

```bash
sbatch --parsable \
  --job-name=q06_s4_safe_focus \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=02:30:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_safe_focus-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage4_safe_memory.py tests/test_stage4_data_protocol.py tests/test_stage4_validation.py tests/test_memory_utility_gate_train.py tests/test_stage4_act_train.py tests/test_stage4_quality_gate.py tests/test_memory_utility_gate.py tests/test_memory_utility_oracle_audit.py tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_final_chain.py tests/test_qwen_clstr_frozen_route_eval.py tests/test_native_benchmark_checkpoint_adapter.py tests/test_toolbench_full_clstr_route_eval.py tests/test_global_pool_route_eval.py tests/test_tau2_route_eval.py tests/test_toolsandbox_route_eval.py tests/test_alfworld_eval.py tests/test_qwen_clstr_multibench_submit.py tests/test_qwen_clstr_multibench_report.py tests/test_qwen_clstr_training_launchers.py'
```

Expected: zero failures. Inspect the complete Slurm log before continuing.

- [ ] **Step 2: Submit the near-full repository regression**

```bash
sbatch --parsable \
  --job-name=q06_s4_safe_fulltest \
  --partition=gpu_a800,gpu_h100,gpu_h200 \
  --gpus=1 \
  --time=05:00:00 \
  --exclude=d1n41a15g01 \
  --output=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel/slurm_logs/q06_s4_safe_fulltest-%j.out \
  --wrap='cd /data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel && CUDA_VISIBLE_DEVICES="" /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q --deselect=tests/test_appworld_mt_fusion_train.py::test_multistep_oracle_policy_loss_uses_step_positive_skill_ids'
```

This repository is not currently near-full green. The pre-change control job `110661` produced `27 failed, 1669 passed, 1 deselected`; its complete log is `.tmp/slurm/clstr-fulltest-110661.out` with SHA-256 `e1827388072f1031a708832a2e5bbe2001526fefacc6c792a4b6397b830ef3df`. Compare exact `FAILED <nodeid>` lines against that baseline. Acceptance is: no new failing node ID, no failure in the focused safe-memory suite from Step 1, and every changed baseline failure either disappears or is explained with evidence. Do not describe the repository-wide suite as green while baseline failures remain.

- [ ] **Step 3: Run local compilation, shell, and diff verification**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_qwen06_clstr_stage4_train.sh \
  scripts/sbatch/run_qwen06_clstr_native_route_eval.sh \
  scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Run the real-Qwen safe-memory smoke**

Submit the canonical wrapper with:

```bash
sbatch --parsable \
  --export=ALL,FULL_RUN=0,PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel,RUN_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix/outputs/qwen06_clstr_postfix \
  scripts/sbatch/run_qwen06_clstr_stage4_train.sh
```

Expected smoke evidence:

- frozen Qwen and zero trainable backbone parameters;
- optimizer names only under transition/gate/action projection;
- exact Stage2 static-logit parity;
- delta-only checkpoint;
- four-domain smoke validation;
- fixed-alpha finalization;
- `stage4_quality_gate.json`, `stage4_selection.json`, and `lineage.json` all status ok;
- no NaN, nonfinite logit, CUDA error, or OOM marker.

- [ ] **Step 5: Fast-forward master only after source verification and smoke**

Run a final focused source regression against the exact commit used by smoke. If it passes and the worktree contains only the pre-existing unrelated multibench-plan changes, fast-forward `master` to the verified implementation commit. Do not open a PR.

- [ ] **Step 6: Submit and supervise safe-memory Stage4 full**

Submit:

```bash
sbatch --parsable \
  --export=ALL,FULL_RUN=1,PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel,RUN_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-clstr-postfix/outputs/qwen06_clstr_postfix \
  scripts/sbatch/run_qwen06_clstr_stage4_train.sh
```

Monitor event-wise rather than polling frequently. At each 400-step validation event, inspect only small reports for router parity, dynamic MRR, memory delta, per-benchmark regressions, LR, and nonfinite counts.

Do not promote the final optimizer step automatically. The release artifact is the checkpoint named by `stage4_safe_full/stage4_selection.json`.

- [ ] **Step 7: Handle release outcomes without changing method rules**

If the dynamic checkpoint fails its release thresholds, keep Stage2 as the safe endpoint, preserve both Stage4 runs, and stop automatic Stage4 benchmark claims. Do not unfreeze Qwen or weaken the gate.

If the dynamic checkpoint passes but the learned-gate audit fails, finalize the validation-selected fixed alpha and record the negative learned-gate result.

If the learned gate passes the audit but misses its validation promotion threshold, keep fixed alpha and preserve the gate report without a checkpoint.

- [ ] **Step 8: Regenerate the final chain and run benchmark smoke/full**

Delete no historical manifest. Resolve a new final-chain v2 manifest from `stage4_safe_full/stage4_selection.json`, update the immutable multibench config to that manifest, and submit the nine-job smoke evaluation chain.

Only when all smoke jobs, parity checks, structural validators, and the consolidated smoke gate pass may the full ToolBench, Tau2, ToolSandbox, and ALFWorld jobs be submitted. All full jobs must share the same final-chain manifest SHA, model checkpoint-chain digest, reliability identity, and accepted smoke-gate SHA.

- [ ] **Step 9: Record final evidence and close the plan**

Update the active planning files with:

- implementation commits;
- focused and near-full pytest Slurm job IDs/results;
- safe smoke/full Stage4 job IDs and selected step;
- Stage2/static/dynamic/fused validation metrics;
- learned-gate recommendation and promotion result;
- final-chain v2 SHA and reliability identity;
- benchmark job IDs and consolidated report paths;
- any blocker without converting it into a zero score.

Commit only the planning/report updates that belong to this branch. Mark the active phase complete only after the selected Stage4 artifact and required benchmark outcomes are genuinely available.
