# Stage0 v4 Handoff-Aligned Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing trajectory-aware Stage0 v3 into a handoff-aligned Stage0 v4/v3-fix that improves real Stage2 candidate coverage without task-specific rules or oracle candidate injection.

**Architecture:** Keep the SkillRouter-compatible Stage0 bi-encoder protocol. Change only the training sampler/gates so Stage0 sees enough sequential routing queries that match Stage2 handoff text distribution, then require per-benchmark handoff coverage before entering Stage2.

**Tech Stack:** Python, PyTorch, pytest, existing CLSTR `retrieval_warmup.py`, existing Slurm scripts under `scripts/sbatch`.

---

## Current Evidence

Stage0 v3 is already trajectory-aware:

- data: `data/clstr_unified_pretrain_v3_trajectory_retrieval_caps`
- checkpoint: `outputs/clstr_unified_stage0_biencoder_v3_traj_retrieval_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- retrieval rows: `552219`
- trajectory rows: `275271`
- trajectory-derived retrieval pairs: `111122`

The new valid handoff audit shows v3 is not yet handoff-aligned enough:

```text
job id: 81803
report: outputs/clstr_stage0_handoff_coverage_audit/v3_toolbench_traject_smoke_v2/report.json
row_count: 320
top_k_values: [20, 50, 100, 200, 500]

toolbench_g3 next_recall@100 = 0.7582
toolbench_g3 next_recall@200 = 0.8571
toolbench_g3 next_recall@500 = 0.9231

traject_bench next_recall@100 = 0.4857
traject_bench next_recall@200 = 0.6667
traject_bench next_recall@500 = 0.8381
```

Conclusion: do not add benchmark-specific hacks. Improve the generic Stage0 handoff alignment and make TrajectBench a gate because it is the source where the general sequential routing objective currently fails.

## Non-Goals

- Do not use AppWorld-specific logic.
- Do not inject gold skills into eval candidates.
- Do not add a cross-encoder to the CLSTR mainline.
- Do not submit full training jobs without user confirmation.
- Do not move files outside `/data/home/scyb713/run/xzf/AAAI/autodl-tmp`.

## Task 1: Add Handoff-Aware Stage0 Sampling

**Files:**

- Modify: `clstr/retrieval_warmup.py`
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Test: `tests/test_stage0_biencoder_protocol.py`

### Rationale

The current `source_balanced` sampler buckets rows only by `source`. That treats `trajectory_derived_traject_bench` as one source, but does not guarantee enough training budget for:

- trajectory-derived current queries;
- trajectory-derived next queries;
- target benchmarks where Stage2 handoff coverage is weak;
- static retrieval sources needed to preserve SkillRouter-style coarse retrieval.

Add a new generic sampler `handoff_balanced`. It should bucket by a stable training bucket:

```python
def _query_training_bucket(row: dict[str, Any]) -> str:
    source = _query_source(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    target = str(provenance.get("target") or "")
    if source.startswith("trajectory_derived_") and target:
        return f"{source}:{target}"
    return source
```

This is not task-specific: all trajectory-derived sources receive current/next separation.

### Steps

- [ ] Add a failing test `test_stage0_handoff_balanced_sampler_splits_trajectory_current_and_next`.

Expected test body:

```python
def test_stage0_handoff_balanced_sampler_splits_trajectory_current_and_next():
    rows = [
        {
            "query_id": "traj-current",
            "source_dataset": "x",
            "metadata": {
                "source": "trajectory_derived_traject_bench",
                "provenance": {"target": "current"},
            },
        },
        {
            "query_id": "traj-next",
            "source_dataset": "x",
            "metadata": {
                "source": "trajectory_derived_traject_bench",
                "provenance": {"target": "next"},
            },
        },
        {"query_id": "skillret", "source_dataset": "skillret", "metadata": {"source": "skillret"}},
    ]

    buckets = retrieval_warmup._build_training_buckets(rows, sampling_strategy="handoff_balanced")

    assert sorted(buckets) == [
        "skillret",
        "trajectory_derived_traject_bench:current",
        "trajectory_derived_traject_bench:next",
    ]
```

- [ ] Run:

```bash
PYTHONPATH=.vendor_pytest:$PYTHONPATH /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py::test_stage0_handoff_balanced_sampler_splits_trajectory_current_and_next -q
```

Expected: fail because `_build_training_buckets(..., sampling_strategy="handoff_balanced")` does not exist yet.

- [ ] Implement `_query_training_bucket(...)` and `_build_training_buckets(...)`.

Minimal implementation:

```python
def _query_training_bucket(row: dict[str, Any]) -> str:
    source = _query_source(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
    target = str(provenance.get("target") or "")
    if source.startswith("trajectory_derived_") and target:
        return f"{source}:{target}"
    return source


def _build_training_buckets(
    queries: list[dict[str, Any]],
    *,
    sampling_strategy: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in queries:
        key = _query_training_bucket(row) if sampling_strategy == "handoff_balanced" else _query_source(row)
        buckets.setdefault(key, []).append(row)
    return buckets
```

- [ ] Update `_batch_queries_for_step(...)` to accept `handoff_balanced` using the same deterministic round-robin logic as `source_balanced`.

- [ ] Update allowed strategies:

```python
if sampling_strategy not in {"batch_stride", "source_balanced", "handoff_balanced"}:
    raise ValueError(...)
```

- [ ] Update CLI and sbatch:

```python
parser.add_argument("--sampling_strategy", default="handoff_balanced", choices=["batch_stride", "source_balanced", "handoff_balanced"])
```

```bash
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}
```

- [ ] Run focused tests:

```bash
PYTHONPATH=.vendor_pytest:$PYTHONPATH /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_is_stage0_compatibility_wrapper \
  tests/test_sbatch_scripts.py::test_stage0_handoff_coverage_audit_sbatch_runs_on_compute_node_with_small_default -q
```

Expected: pass.

## Task 2: Record Handoff-Balanced Buckets in Training Metadata

**Files:**

- Modify: `clstr/retrieval_warmup.py`
- Test: `tests/test_stage0_biencoder_protocol.py`

### Rationale

For paper credibility and debugging, Stage0 reports must show what buckets were actually sampled. Without this, a Stage0 v4 result cannot be audited later.

### Steps

- [ ] Add a failing test `test_stage0_records_handoff_balanced_bucket_counts`.

Expected assertion:

```python
assert report["training_bucket_counts"] == {
    "skillret": 1,
    "trajectory_derived_traject_bench:current": 1,
    "trajectory_derived_traject_bench:next": 1,
}
```

- [ ] Add `training_bucket_counts` and `training_bucket_count` to:

```python
setup_status.jsonl phase="selected_skills_written"
latest checkpoint metadata
train_report.json
```

- [ ] Run:

```bash
PYTHONPATH=.vendor_pytest:$PYTHONPATH /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py -q
```

Expected: pass.

## Task 3: Add Handoff Coverage Gate Defaults

**Files:**

- Modify: `scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh`
- Modify: `tests/test_sbatch_scripts.py`
- Optional Modify: `clstr/stage0_handoff_audit.py`

### Rationale

The current handoff audit reports metrics but does not define pass/fail. Add documented defaults so Stage0 v4 cannot be treated as ready unless per-benchmark handoff coverage is acceptable.

### Initial Gate

Use this as a reporting gate, not a hard claim of final performance:

```text
global next_recall@500 >= 0.90
toolbench_g3 next_recall@500 >= 0.90
traject_bench next_recall@500 >= 0.88
traject_bench next_recall@200 >= 0.75
```

These thresholds are intentionally about candidate coverage, not final routing success.

### Steps

- [ ] Add sbatch defaults:

```bash
MIN_GLOBAL_NEXT_RECALL_AT_500=${MIN_GLOBAL_NEXT_RECALL_AT_500:-0.90}
MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500=${MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500:-0.90}
MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500:-0.88}
MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200:-0.75}
```

- [ ] Add a lightweight post-report Python gate inside the sbatch script after the audit writes `report.json`.

The gate should fail only after the report exists and should print clear blockers.

- [ ] Run:

```bash
PYTHONPATH=.vendor_pytest:$PYTHONPATH /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py::test_stage0_handoff_coverage_audit_sbatch_runs_on_compute_node_with_small_default -q
```

Expected: pass.

## Task 4: Update Documentation and Experiment Protocol

**Files:**

- Modify: `description.md`
- Modify: `docs/clstr_v4_plan.md`

### Rationale

This is a significant methodological change. The paper-facing framing must be explicit:

- Stage0 v3 was already trajectory-aware.
- Stage0 v4/v3-fix is handoff-aligned.
- No eval-time oracle injection is allowed in the main result.
- The change is benchmark-generic because it separates current/next trajectory-derived queries for all sources.

### Steps

- [ ] Add a short entry to `description.md` after the 2026-06-02 handoff audit entry.

- [ ] Add a short section to `docs/clstr_v4_plan.md` under Stage0 foundation-first pipeline:

```text
Stage0 v4 handoff-aligned fix:
source_balanced is replaced by handoff_balanced, which splits trajectory-derived current/next buckets while preserving SkillRET, ToolRet, ToolBench-G3, and TrajectBench static retrieval buckets. This is a generic sequential-routing alignment change, not a benchmark-specific rule.
```

## Task 5: Smoke, Then Stop Before Full Job

**Files:**

- No code changes expected.

### Steps

- [ ] Run only local non-model checks:

```bash
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh

git diff --check -- \
  clstr/retrieval_warmup.py \
  scripts/run_clstr_stage0_biencoder_train.py \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_sbatch_scripts.py \
  description.md \
  docs/clstr_v4_plan.md
```

- [ ] If user permits a compute-node smoke, submit a tiny Stage0 v4 smoke only:

```bash
sbatch --gpus=1 -p gpu_a800 --job-name=clstr_s0v4_smoke \
  --export=ALL,OUTPUT_DIR=outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_smoke,MAX_STEPS=1,BATCH_SIZE=4,GRADIENT_ACCUMULATION_STEPS=1,MAX_SKILLS=128,MAX_QUERIES=128,SAMPLING_STRATEGY=handoff_balanced \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

- [ ] Do not submit full Stage0 v4 training unless the user explicitly confirms.

## Paper-Safety Checklist

- [ ] No benchmark-specific positive injection in eval.
- [ ] No dev/test leakage; existing split filters remain active.
- [ ] Handoff gate is reported as candidate coverage, not final task success.
- [ ] Main comparison uses same skill pool and same candidate budget as SkillRouter-compatible baseline.
- [ ] Ablation distinguishes:
  - Stage0 v3 source-balanced;
  - Stage0 v4 handoff-balanced;
  - Stage2 with real top-M candidates;
  - any train-time injection only as auxiliary/ablation.
