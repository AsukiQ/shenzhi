# Qwen Stage0 Handoff Acceleration Implementation Plan

> **Historical plan notice (2026-07-12):** Tasks that introduce exact-query deduplication or combined current/next scoring are superseded by `2026-07-12-qwen-handoff-exact-schedule-revision.md`. They remain here only to preserve the implementation history and must not be re-enabled.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, exact-query-deduplicated `row_sharded_v1` Stage0 handoff path that reuses compact raw candidates across Stage1/2/4 without changing retained rows, candidate order, scores, masks, or frozen-backbone semantics.

**Architecture:** Keep `legacy_jsonl` as the default. Put deterministic query planning, raw-record packing, manifest validation, and immutable Torch shards in a focused `clstr/stage0_handoff_acceleration.py` module. Refactor `clstr/full_base_train.py` so both legacy and accelerated computation feed one unchanged label-aware post-processing loop; Stage4 calls the same shared preparation API.

**Tech Stack:** Python 3.11, PyTorch, pytest, argparse, Bash/Slurm, JSON manifests, atomic filesystem replacement.

---

## File Structure

- Create `clstr/stage0_handoff_acceleration.py`: ordered query plans, raw candidate records, row/global cache identity, strict shard load/append/reset.
- Modify `clstr/full_base_train.py`: raw candidate computation, label-aware post-processing boundary, compact-cache orchestration, public training parameters.
- Modify `clstr/stage4_act_train.py`: use shared compact handoff preparation and expose cache settings.
- Modify `scripts/run_clstr_stage1_heads_init.py`: Stage1 cache-format CLI and forwarding.
- Modify `scripts/run_clstr_stage2_full_base_train.py`: Stage2 cache-format CLI and forwarding.
- Modify `scripts/run_clstr_stage4_act_train.py`: Stage4 cache-format CLI and forwarding.
- Modify `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`: environment-to-CLI forwarding with legacy defaults.
- Modify `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`: environment-to-CLI forwarding with legacy defaults.
- Modify `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`: environment-to-CLI forwarding with legacy defaults.
- Modify `scripts/sbatch/run_qwen06_clstr_stage{1,2,4}_train.sh`: define a shared compact-cache root only after the real-Qwen gate is accepted.
- Create `scripts/audit_qwen06_stage0_handoff_acceleration.py`: fixed-row parity, storage, speed, and freeze audit.
- Create `scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh`: GPU-only Slurm launcher for the audit.
- Create `tests/test_stage0_handoff_acceleration.py`: focused unit tests for planning and cache artifacts.
- Modify `tests/test_clstr_topm_candidate_handoff.py`: raw-computation and label-aware parity tests.
- Modify `tests/test_full_base_train.py`: trainer integration and fail-closed tests.
- Modify `tests/test_sbatch_scripts.py`: CLI/sbatch propagation assertions.

### Task 1: Deterministic query planning and raw record representation

**Files:**
- Create: `tests/test_stage0_handoff_acceleration.py`
- Create: `clstr/stage0_handoff_acceleration.py`

- [x] **Step 1: Write failing tests for first-occurrence query deduplication**

```python
from clstr.stage0_handoff_acceleration import build_ordered_query_plan


def test_query_plan_deduplicates_current_and_next_in_first_occurrence_order():
    plan = build_ordered_query_plan(
        ["state-a", "state-b", "state-a"],
        ["state-b", "state-c", "state-c"],
    )
    assert plan.unique_queries == ("state-a", "state-b", "state-c")
    assert plan.current_query_indices == (0, 1, 0)
    assert plan.next_query_indices == (1, 2, 2)
```

- [x] **Step 2: Run the test and verify RED**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_handoff_acceleration.py::test_query_plan_deduplicates_current_and_next_in_first_occurrence_order -q`

Expected: FAIL because `clstr.stage0_handoff_acceleration` does not exist.

- [x] **Step 3: Implement the immutable query plan**

```python
@dataclass(frozen=True)
class OrderedQueryPlan:
    unique_queries: tuple[str, ...]
    current_query_indices: tuple[int, ...]
    next_query_indices: tuple[int, ...]


def build_ordered_query_plan(current_queries, next_queries):
    if len(current_queries) != len(next_queries):
        raise ValueError("current and next query counts must match")
    unique_queries = []
    query_to_idx = {}
    current_indices = []
    next_indices = []
    for current, next_query in zip(current_queries, next_queries):
        for query, destination in ((str(current), current_indices), (str(next_query), next_indices)):
            if query not in query_to_idx:
                query_to_idx[query] = len(unique_queries)
                unique_queries.append(query)
            destination.append(query_to_idx[query])
    return OrderedQueryPlan(tuple(unique_queries), tuple(current_indices), tuple(next_indices))
```

- [x] **Step 4: Add RED tests for raw records and row keys**

```python
from clstr.stage0_handoff_acceleration import RawStage0Candidates, stage0_handoff_row_key


def test_row_key_tracks_exact_queries_and_ordered_inventory():
    base = stage0_handoff_row_key("current", "next", ["skill/a", "skill/b"])
    assert base != stage0_handoff_row_key("current changed", "next", ["skill/a", "skill/b"])
    assert base != stage0_handoff_row_key("current", "next changed", ["skill/a", "skill/b"])
    assert base != stage0_handoff_row_key("current", "next", ["skill/b", "skill/a"])


def test_raw_candidate_record_rejects_mismatched_indices_and_scores():
    with pytest.raises(ValueError, match="current candidate"):
        RawStage0Candidates((0, 1), (1.0,), (2,), (0.5,))
```

- [x] **Step 5: Implement validated raw records and exact row keys**

```python
@dataclass(frozen=True)
class RawStage0Candidates:
    current_indices: tuple[int, ...]
    current_scores: tuple[float, ...]
    next_indices: tuple[int, ...]
    next_scores: tuple[float, ...]

    def __post_init__(self):
        if len(self.current_indices) != len(self.current_scores):
            raise ValueError("current candidate indices and scores must match")
        if len(self.next_indices) != len(self.next_scores):
            raise ValueError("next candidate indices and scores must match")


def stage0_handoff_row_key(current_query, next_query, ordered_inventory_skill_ids):
    return stable_json_digest({
        "current_query": str(current_query),
        "next_query": str(next_query),
        "ordered_inventory_skill_ids": [str(item) for item in ordered_inventory_skill_ids],
    })
```

- [x] **Step 6: Run the focused tests and commit**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_handoff_acceleration.py -q`

Expected: PASS.

Commit: `git add clstr/stage0_handoff_acceleration.py tests/test_stage0_handoff_acceleration.py && git commit -m "feat: plan deduplicated stage0 handoff queries"`

### Task 2: Separate raw candidate computation from label-aware post-processing

**Files:**
- Modify: `tests/test_clstr_topm_candidate_handoff.py`
- Modify: `clstr/full_base_train.py:1136-1453`

- [x] **Step 1: Write a failing test proving each exact query is encoded once**

```python
class _RecordingStage0TopMModel(_Stage0TopMModel):
    def __init__(self):
        super().__init__()
        self.encoded_texts = []

    def encode_observations(self, texts):
        self.encoded_texts.extend(str(text) for text in texts)
        return super().encode_observations(texts)


def test_accelerated_handoff_encodes_each_exact_query_once():
    model = _RecordingStage0TopMModel()
    rows = [
        {"state_text": "state-a", "next_state_text": "state-b"},
        {"state_text": "state-b", "next_state_text": "state-c"},
    ]
    skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]
    skill_map = {"skill/a": 0, "skill/b": 1, "skill/c": 2}
    raw, report = _compute_stage0_raw_candidates_deduplicated(
        model, rows, skills, skill_map, top_m=2,
        query_mode="raw_state", encode_batch_size=16, device=torch.device("cpu"),
    )
    assert model.encoded_texts == ["state-a", "state-b", "state-c"]
    assert len(raw) == 2
    assert report["unique_query_count"] == 3
```

- [x] **Step 2: Run the test and verify RED**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_clstr_topm_candidate_handoff.py::test_accelerated_handoff_encodes_each_exact_query_once -q`

Expected: FAIL because `_compute_stage0_raw_candidates_deduplicated` is absent.

- [x] **Step 3: Implement deduplicated encoding and combined row scoring**

Add `_compute_stage0_raw_candidates_deduplicated` with the model, rows, skill metadata, top-M, query mode, batch size, device, progress, and inventory arguments already used by the legacy computation.

```python
current_queries = [_stage0_handoff_query_text(row, target="current", mode=query_mode) for row in rows]
next_queries = [_stage0_handoff_query_text(row, target="next", mode=query_mode) for row in rows]
plan = build_ordered_query_plan(current_queries, next_queries)
unique_h = _encode_stage0_handoff_query_batches(
    model, list(plan.unique_queries), query_mode=query_mode, batch_size=encode_batch_size,
).detach().cpu()
for start in range(0, len(rows), encode_batch_size):
    current_h = unique_h[list(plan.current_query_indices[start:end])]
    next_h = unique_h[list(plan.next_query_indices[start:end])]
    logits = _stage0_unified_static_logits(model, torch.cat([current_h, next_h]).to(device), skill_count)
    current_logits, next_logits = logits.split(end - start, dim=0)
    # Call `_stage0_topk_with_explicit_inventory()` unchanged for both halves.
```

Return aligned `RawStage0Candidates` plus inventory/query statistics. Preserve model train/eval state and use `torch.no_grad()`.

- [x] **Step 4: Write a failing parity test for label-aware output**

```python
def test_deduplicated_raw_candidates_preserve_legacy_handoff_rows_and_report():
    rows = [{
        "state_text": "state-a", "next_state_text": "next-b",
        "skill_id": "skill/a", "next_skill_id": "skill/b",
        "loss_mask": {"routing": True, "L_trans_skill_ce": True},
    }]
    skills = [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}]
    skill_map = {"skill/a": 0, "skill/b": 1, "skill/c": 2}
    legacy_model = _Stage0TopMModel()
    accelerated_model = _Stage0TopMModel()
    legacy_rows, legacy_report = _attach_stage0_topm_candidates(
        legacy_model, rows, skills, skill_map, top_m=2,
        positive_missing_policy="skip", query_mode="raw_state",
        encode_batch_size=16, device=torch.device("cpu"),
    )
    raw, raw_report = _compute_stage0_raw_candidates_deduplicated(
        accelerated_model, rows, skills, skill_map, top_m=2,
        query_mode="raw_state", encode_batch_size=16,
        device=torch.device("cpu"), inventory_min_candidates=0,
    )
    accelerated_rows, accelerated_report = _attach_stage0_topm_candidates(
        accelerated_model, rows, skills, skill_map, top_m=2,
        positive_missing_policy="skip", query_mode="raw_state",
        encode_batch_size=16, device=torch.device("cpu"),
        raw_candidates=raw, raw_candidate_report=raw_report,
    )
    assert accelerated_rows == legacy_rows
    label_aware_report_keys = {
        "retained_rows", "skipped_rows", "skipped_reasons",
        "current_positive_covered_rows", "next_positive_covered_rows",
        "masked_next_skill_ce_rows", "current_skill_candidate_added_rows",
        "next_positive_rank_bucket_counts",
    }
    for key in label_aware_report_keys:
        assert accelerated_report[key] == legacy_report[key]
```

- [x] **Step 5: Extract one post-processing path**

Change `_attach_stage0_topm_candidates()` to accept:

```python
raw_candidates: list[RawStage0Candidates] | None = None,
raw_candidate_report: dict[str, Any] | None = None,
```

When raw candidates are absent, compute them with a new `_compute_stage0_raw_candidates_legacy()` containing the old two-encode loop. In both cases, feed the same existing row loop using:

```python
for row, raw in zip(rows, raw_candidates, strict=True):
    current_candidates = list(raw.current_indices)
    current_candidate_scores = list(raw.current_scores)
    next_candidates_raw = list(raw.next_indices)
    next_candidate_scores = list(raw.next_scores)
    # Existing positive/mask/provenance logic continues unchanged.
```

Reject count mismatches before mutating rows.

- [x] **Step 6: Run focused parity regressions and commit**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_clstr_topm_candidate_handoff.py -q`

Expected: all tests PASS, including existing positive-missing, inventory, and provenance tests.

Commit: `git add clstr/full_base_train.py tests/test_clstr_topm_candidate_handoff.py && git commit -m "refactor: isolate raw stage0 handoff candidates"`

### Task 3: Compact row-keyed immutable shard cache

**Files:**
- Modify: `tests/test_stage0_handoff_acceleration.py`
- Modify: `clstr/stage0_handoff_acceleration.py`

- [x] **Step 1: Write RED tests for cold write, warm hit, and subset reuse**

```python
def _identity():
    return stage0_handoff_global_identity(
        checkpoint_digest={"sha256": "checkpoint-a"},
        skills_digest="skills-a",
        top_m=2,
        candidate_count=2,
        query_mode="checkpoint_state_query",
        inventory_min_candidates=0,
        initial_belief_top_k=64,
        state_text_format="default",
        skill_text_format="default",
        skill_embedding_digest="embedding-a",
        declared_pool_order_digest="order-a",
        candidate_selection_version="stable_declared_pool_v1",
        tie_break_policy="declared_pool_index_ascending",
        unified_static_scorer_digest="scorer-a",
    )


def _raw(offset):
    return RawStage0Candidates(
        current_indices=(offset, offset + 1),
        current_scores=(2.0, 1.0),
        next_indices=(offset + 1, offset),
        next_scores=(3.0, 0.5),
    )


def test_row_sharded_cache_reuses_rows_across_different_subsets(tmp_path):
    identity = _identity()
    first = {"row-a": _raw(0), "row-b": _raw(1)}
    append_row_sharded_cache(tmp_path, identity, first, shard_size=1)
    hits, report = load_row_sharded_cache(tmp_path, identity, ["row-b", "row-c"])
    assert hits == {"row-b": first["row-b"]}
    assert report["hit_rows"] == 1
    assert report["miss_rows"] == 1
```

- [x] **Step 2: Run and verify RED**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_handoff_acceleration.py::test_row_sharded_cache_reuses_rows_across_different_subsets -q`

Expected: FAIL because the cache API is absent.

- [x] **Step 3: Implement global identity, entry layout, padding, and atomic writes**

Use this manifest contract:

```json
{
  "format": "row_sharded_v1",
  "global_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "global_identity": {"candidate_count": 500},
  "shards": [
    {"path": "shards/shard-123e4567-e89b-12d3-a456-426614174000.pt", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "row_count": 2048}
  ],
  "row_count": 2048
}
```

Each Torch shard contains exactly:

```python
{
    "row_keys": tuple[str, ...],
    "current_indices": torch.int32,   # padded with -1
    "current_scores": torch.float32,  # padding is NaN
    "next_indices": torch.int32,
    "next_scores": torch.float32,
}
```

Write shard and manifest temporary files in their target directories, `flush()`/`os.fsync()`, then `os.replace()`. Existing shards remain immutable.

- [x] **Step 4: Write RED corruption and conflict tests**

```python
def test_row_sharded_cache_fails_closed_when_manifest_references_missing_shard(tmp_path):
    identity = _identity()
    append_row_sharded_cache(tmp_path, identity, {"row-a": _raw(0)})
    manifest_path = row_sharded_cache_entry_dir(tmp_path, identity) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (manifest_path.parent / manifest["shards"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="row_sharded_v1"):
        load_row_sharded_cache(tmp_path, identity, ["row-a"])


def test_row_sharded_cache_fails_closed_on_wrong_tensor_dtype(tmp_path):
    identity = _identity()
    append_row_sharded_cache(tmp_path, identity, {"row-a": _raw(0)})
    manifest_path = row_sharded_cache_entry_dir(tmp_path, identity) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / manifest["shards"][0]["path"]
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    payload["current_indices"] = payload["current_indices"].to(torch.int64)
    torch.save(payload, shard_path)
    manifest["shards"][0]["sha256"] = file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="current_indices.*int32"):
        load_row_sharded_cache(tmp_path, identity, ["row-a"])


def test_row_sharded_cache_rejects_conflicting_duplicate_key(tmp_path):
    append_row_sharded_cache(tmp_path, _identity(), {"row-a": _raw(0)})
    with pytest.raises(ValueError, match="conflicting duplicate row key"):
        append_row_sharded_cache(tmp_path, _identity(), {"row-a": _raw(1)})
```

- [x] **Step 5: Implement strict load validation and refresh reset**

Validation must check format/version, exact global identity, shard existence and SHA-256, allowed payload keys, tensor dtype/rank/shape, candidate width, `-1` tail padding, finite valid scores, NaN padding, manifest row counts, and cross-shard duplicate consistency. `reset_row_sharded_cache()` atomically replaces the manifest with an empty one and leaves old immutable shards unreferenced.

- [x] **Step 6: Run cache tests and commit**

Run: `/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_handoff_acceleration.py -q`

Expected: PASS.

Commit: `git add clstr/stage0_handoff_acceleration.py tests/test_stage0_handoff_acceleration.py && git commit -m "feat: add compact row-sharded handoff cache"`

### Task 4: Integrate the opt-in format through Stage1/2/4

**Files:**
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_clstr_topm_candidate_handoff.py`
- Modify: `tests/test_sbatch_scripts.py`
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage4_act_train.py`
- Modify: `scripts/run_clstr_stage1_heads_init.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Modify: `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`

- [x] **Step 1: Write RED trainer tests for default compatibility and incremental reuse**

```python
def test_full_base_defaults_to_legacy_jsonl_handoff_cache():
    parameter = inspect.signature(train_clstr_full_base_with_model).parameters["stage0_handoff_cache_format"]
    assert parameter.default == "legacy_jsonl"


def test_full_base_row_sharded_cache_computes_only_missing_rows(tmp_path, monkeypatch):
    # First call writes rows a/b. Second call asks for b/c.
    # Capture `_compute_stage0_raw_candidates_deduplicated` input on the second call.
    assert captured_second_call_row_ids == ["c"]
    assert second_report["cache"]["hit_rows"] == 1
    assert second_report["cache"]["miss_rows"] == 1
```

- [x] **Step 2: Add and validate new parameters**

Add to Stage1/2 shared training entry points and Stage4:

```python
stage0_handoff_cache_format: str = "legacy_jsonl",
stage0_handoff_cache_shard_size: int = 2048,
```

Accepted formats are `legacy_jsonl` and `row_sharded_v1`. Reject unsupported values before handoff work and require a positive shard size.

- [x] **Step 3: Implement compact orchestration**

For `row_sharded_v1`:

```python
queries = _stage0_handoff_row_queries(rows, query_mode)
row_keys = [
    stage0_handoff_row_key(current_query, next_query, ordered_inventory)
    for current_query, next_query, ordered_inventory in queries
]
if cache_mode == "refresh":
    reset_row_sharded_cache(cache_dir, global_identity)
hits, cache_report = load_row_sharded_cache(cache_dir, global_identity, row_keys)
missing_positions = [idx for idx, key in enumerate(row_keys) if key not in hits]
missing_rows = [rows[idx] for idx in missing_positions]
missing_keys = [row_keys[idx] for idx in missing_positions]
missing_raw, compute_report = _compute_stage0_raw_candidates_deduplicated(
    model, missing_rows, skills, skill_id_to_idx,
    top_m=top_m, query_mode=query_mode,
    encode_batch_size=encode_batch_size, device=device,
    inventory_min_candidates=inventory_min_candidates,
)
computed = dict(zip(missing_keys, missing_raw, strict=True))
append_row_sharded_cache(
    cache_dir, global_identity, computed,
    shard_size=cache_shard_size,
)
raw_in_original_order = [hits[key] if key in hits else computed[key] for key in row_keys]
rows, report = _attach_stage0_topm_candidates(
    model, rows, skills, skill_id_to_idx,
    top_m=top_m, positive_missing_policy=positive_missing_policy,
    query_mode=query_mode, inventory_min_candidates=inventory_min_candidates,
    next_skill_pool_mode=next_skill_pool_mode,
    raw_candidates=raw_in_original_order,
    raw_candidate_report=compute_report,
)
```

The global identity excludes row-set digest and label-only policy fields. The legacy block remains byte-for-byte behavior-compatible when `stage0_handoff_cache_format="legacy_jsonl"`.

- [x] **Step 4: Add CLI and shell forwarding tests**

Assert all three Python CLIs expose:

```text
--stage0_handoff_cache_format {legacy_jsonl,row_sharded_v1}
--stage0_handoff_cache_shard_size 2048
```

Assert the generic shell launchers default to:

```bash
STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}
STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}
```

and forward both arguments.

- [x] **Step 5: Implement CLI/shell forwarding and Stage4 shared use**

Stage4 must call the same compact preparation helper as Stage1/2 rather than implement a second cache format. Keep Qwen production wrapper defaults unchanged in this task.

- [x] **Step 6: Run integration regressions and commit**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_handoff_acceleration.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_full_base_train.py \
  tests/test_sbatch_scripts.py -q
bash -n scripts/sbatch/run_clstr_unified_stage1_heads_init.sh
bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh
```

Expected: all tests and syntax checks PASS.

Commit: `git add clstr scripts tests && git commit -m "feat: integrate row-sharded handoff cache"`

### Task 5: Real-Qwen parity, speed, storage, and freeze gate

**Files:**
- Create: `scripts/audit_qwen06_stage0_handoff_acceleration.py`
- Create: `scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh`
- Modify: `tests/test_stage0_handoff_acceleration.py`
- Modify: `tests/test_sbatch_scripts.py`

- [x] **Step 1: Write RED tests for parity comparison and gate decisions**

```python
def test_handoff_gate_rejects_candidate_order_mismatch():
    result = compare_handoff_outputs(_rows([0, 1]), _rows([1, 0]), score_atol=1e-5)
    assert result["candidate_ids_equal"] is False
    assert result["status"] == "action_required"


def test_handoff_gate_requires_all_speed_storage_and_freeze_thresholds():
    result = evaluate_gate(
        parity_ok=True, backbone_trainable_parameters=0,
        cold_speedup=1.25, warm_speedup=5.0, compact_to_legacy_ratio=0.25,
    )
    assert result["status"] == "ok"
```

- [x] **Step 2: Implement the fixed-row audit**

The audit must:

1. load the selected step-5000 Qwen Stage0 checkpoint and exact skill pool;
2. deterministically select 2,048 post-filter rows;
3. run legacy batch-16 once;
4. run accelerated cold cache at batch 16, 64, and 128 in separate cache entries;
5. run one warm-cache pass for every cold candidate that did not OOM;
6. compare retained row identity/order, current/next candidate IDs, score max-absolute difference, loss masks, and positive-hit/injection decisions;
7. count trainable Qwen backbone parameters;
8. compare total compact entry bytes with a legacy JSONL artifact for the same rows;
9. choose the fastest configuration that passes every requirement and write a complete JSON report.

- [x] **Step 3: Encode exact gate thresholds**

```python
REQUIRED_COLD_SPEEDUP = 1.25
REQUIRED_WARM_SPEEDUP = 5.0
MAX_COMPACT_TO_LEGACY_RATIO = 0.25
SCORE_ATOL = 1.0e-5
```

Any CUDA error, OOM, non-finite score, fallback scorer/query mode, candidate mismatch, retained-row mismatch, nonzero backbone trainable count, or threshold miss yields `status="action_required"` and exit code 2.

- [x] **Step 4: Add a GPU-only Slurm wrapper**

The wrapper requests one A800/H100/H200 GPU, uses the selected Stage0 checkpoint from `stage0_selection.json`, sets `TRANSFORMERS_OFFLINE=1`, writes stdout and `handoff_acceleration_gate.json`, and never launches training.

- [x] **Step 5: Run CPU tests and shell syntax, then commit**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_stage0_handoff_acceleration.py tests/test_sbatch_scripts.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile scripts/audit_qwen06_stage0_handoff_acceleration.py
bash -n scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh
```

Expected: PASS.

Commit: `git add scripts tests && git commit -m "test: add qwen handoff acceleration gate"`

### Task 6: Full verification and guarded production promotion

**Files:**
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage1_train.sh`
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage2_train.sh`
- Modify only after gate PASS: `scripts/sbatch/run_qwen06_clstr_stage4_train.sh`
- Modify: `.planning/2026-07-11-qwen-handoff-acceleration/{task_plan.md,findings.md,progress.md}`

- [ ] **Step 1: Run the full CPU verification suite**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_handoff_acceleration.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_full_base_train.py \
  tests/test_stage12_consolidated_train.py \
  tests/test_sbatch_scripts.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
git diff --check
```

- [ ] **Step 2: Submit and supervise the real-Qwen Slurm gate**

Run:

```bash
sbatch --parsable \
  --export=ALL,PROJECT_ROOT=/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstr-qwen06-handoff-accel \
  scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh
```

Poll with `squeue`/`sacct` without blocking the storage node. Inspect the JSON artifact and stdout after terminal state.

- [ ] **Step 3: Promote only the selected passing batch size**

If and only if the report is `status="ok"`, set the Qwen wrappers to a shared cache root and opt-in format:

```bash
SELECTED_BATCH_SIZE=$(
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python - \
    outputs/qwen06_clstr_postfix/handoff_acceleration_gate/handoff_acceleration_gate.json <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "ok":
    raise SystemExit("handoff acceleration gate is not ok")
print(int(report["selected_batch_size"]))
PY
)
export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1
export STAGE0_HANDOFF_CACHE_SHARD_SIZE=2048
export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"
export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE="${SELECTED_BATCH_SIZE}"
```

If the gate fails, leave all Qwen production wrappers on `legacy_jsonl` and record every blocker.

- [ ] **Step 4: Re-run wrapper tests, syntax, and smoke-level cache reuse checks**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_sbatch_scripts.py tests/test_clstr_topm_candidate_handoff.py -q
bash -n scripts/sbatch/run_qwen06_clstr_stage1_train.sh
bash -n scripts/sbatch/run_qwen06_clstr_stage2_train.sh
bash -n scripts/sbatch/run_qwen06_clstr_stage4_train.sh
git diff --check
```

- [ ] **Step 5: Review and commit production promotion**

Inspect `git status --short`, `git diff --stat`, and the complete diff. Confirm no backbone-unfreezing, optimizer-cache, loss, memory, replay, or candidate-selection changes entered the branch.

Commit only after fresh verification: `git add scripts/sbatch .planning docs && git commit -m "perf: enable verified qwen handoff cache"`

## Self-Review

- Spec coverage: query deduplication, current/next combined planning, compact raw-only cache, cross-subset reuse, fail-closed validation, Stage1/2/4 CLI propagation, real-Qwen parity/speed/storage/freeze gates, and guarded wrapper promotion each have an explicit task.
- Non-goals: no optimizer schedule cache, no backbone unfreezing, no loss/memory/replay change, no learned gate, and no legacy deletion.
- Type consistency: `RawStage0Candidates`, `OrderedQueryPlan`, `stage0_handoff_cache_format`, and `stage0_handoff_cache_shard_size` are used consistently across tasks.
- Placeholder scan: no TBD, deferred implementation marker, or unspecified validation step remains.
