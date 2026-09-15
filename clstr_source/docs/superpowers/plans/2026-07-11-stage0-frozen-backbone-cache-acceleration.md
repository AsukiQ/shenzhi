# Stage0 Frozen-Backbone Cache Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate frozen-Qwen CLSTR Stage0 continuation without changing its query schedule, trainable graph, full-pool objective, or checkpoint lineage.

**Architecture:** Split `StateEncoder` at the frozen-backbone pooled-output boundary, precompute only deterministic query rows scheduled for the requested step interval, and apply the trainable projection after lookup. Separately validate ordered skill-pool identity so a resume checkpoint can supply `skill_table.E` without rebuilding 67,557 frozen skill embeddings.

**Tech Stack:** Python 3.11, PyTorch, Transformers, pytest, Bash/Slurm, JSON and JSONL audit artifacts.

---

## File Map

- Modify `clstr/encoders.py`: pooled-backbone and projection APIs.
- Create `clstr/stage0_frozen_backbone_cache.py`: schedule planning, cache construction, lookup, reporting.
- Create `clstr/stage0_skill_pool_identity.py`: ordered hashing and resume validation.
- Modify `clstr/retrieval_warmup.py`: training integration.
- Modify the Stage0 Python and shell launchers: option forwarding and Qwen defaults.
- Create a real-Qwen audit script and bounded Slurm launcher.
- Create new focused test files so the unrelated dirty launcher test remains untouched.

### Task 1: Split StateEncoder at the Pooled Boundary

**Files:**
- Modify: `clstr/encoders.py:117-140`
- Create: `tests/test_state_encoder_pooled_cache.py`

- [ ] **Step 1: Write the failing composition test**

Create a fake backbone returning a known last hidden state, construct `StateEncoder` without Transformers, and assert:

```python
pooled = encoder.encode_backbone_pooled(['a', 'b'])
expected = encoder.project_pooled(pooled)
actual = encoder(['a', 'b'])
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Also assert normalized output has unit norm and projection gradients remain populated after backward.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_state_encoder_pooled_cache.py
```

Expected: failure because both split methods are absent.

- [ ] **Step 3: Implement the shared boundary**

```python
def encode_backbone_pooled(self, text_batch: list[str]) -> torch.Tensor:
    tok = self.tokenize(text_batch)
    out = self.backbone(**tok)
    return pool_hidden(out.last_hidden_state, tok['attention_mask'], self.pooling)

def project_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
    pooled = pooled.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
    projected = self.proj(pooled)
    if self.normalize_embeddings:
        projected = F.normalize(projected, p=2, dim=1)
    return projected

def forward(self, text_batch: list[str]) -> torch.Tensor:
    return self.project_pooled(self.encode_backbone_pooled(text_batch))
```

Make `encode_tokenized` use `project_pooled` after its existing pooling operation.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_state_encoder_pooled_cache.py tests/test_encoders.py
git add clstr/encoders.py tests/test_state_encoder_pooled_cache.py
git commit -m 'refactor: split frozen encoder pooling from projection'
```

Expected: all selected tests pass before commit.

### Task 2: Implement the Deterministic Schedule Cache

**Files:**
- Create: `clstr/stage0_frozen_backbone_cache.py`
- Create: `tests/test_stage0_frozen_backbone_cache.py`

- [ ] **Step 1: Write failing schedule tests**

Use five indexed rows and a fake sampler returning rows `[2, 1]` then `[1, 4]`. Assert the planner calls micro-indices 1 and 2 and stores `[2, 1, 4]` in first-use order. Construct a two-row cache, look up `[7, 3, 7]`, apply `nn.Linear`, backpropagate, and assert duplicate order, non-null projection gradient, and lookup count 3. Add explicit rejection tests for a missing scheduled index and a trainable backbone parameter.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_frozen_backbone_cache.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the cache object**

```python
STAGE0_QUERY_INDEX_KEY = '_stage0_query_index'

@dataclass
class Stage0FrozenBackboneCache:
    pooled_cpu: torch.Tensor
    query_index_to_cache_row: dict[int, int]
    identity: dict[str, Any]
    build_seconds: float
    lookup_count: int = 0

    def project_batch(self, rows, *, projection_fn, device):
        query_indices = [int(row[STAGE0_QUERY_INDEX_KEY]) for row in rows]
        missing = [idx for idx in query_indices if idx not in self.query_index_to_cache_row]
        if missing:
            raise RuntimeError(f'Stage0 frozen-backbone cache schedule miss: {missing[:8]}')
        positions = torch.tensor(
            [self.query_index_to_cache_row[idx] for idx in query_indices],
            dtype=torch.long,
        )
        pooled = self.pooled_cpu.index_select(0, positions).to(device=device, non_blocking=True)
        self.lookup_count += len(query_indices)
        return projection_fn(pooled)
```

`report()` records mode, shape, dtype, bytes, build seconds, lookup count, and identity.

- [ ] **Step 4: Implement schedule planning and construction**

`assign_stage0_query_indices` assigns consecutive private indices. `plan_stage0_scheduled_rows` calls the supplied existing sampler for every exact micro-index in `start_step ... max_steps`, deduplicating storage only. `build_stage0_frozen_backbone_cache` verifies every backbone parameter is frozen, formats texts with `model._serialize_state_for_encoder`, calls `model.encoder.encode_backbone_pooled` in configurable batches under `torch.no_grad()`, retains native pooled dtype on contiguous CPU storage, and pins only when CUDA is available.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_frozen_backbone_cache.py
git add clstr/stage0_frozen_backbone_cache.py tests/test_stage0_frozen_backbone_cache.py
git commit -m 'feat: cache scheduled frozen stage0 query features'
```

Expected: all tests pass before commit.

### Task 3: Implement Ordered Skill-Pool Identity

**Files:**
- Create: `clstr/stage0_skill_pool_identity.py`
- Create: `tests/test_stage0_resume_skill_table.py`

- [ ] **Step 1: Write failing identity tests**

Test identity changes under pool reorder or serialized text changes. Test matching checkpoint tensor shape, config, and sidecar succeeds. Add separate failures for missing sidecar or embedded identity, reordered pool, changed text, wrong shape, and incompatible `d` or `skill_text_format`.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_resume_skill_table.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement ordered length-framed SHA-256**

```python
def _update_field(digest, value: str) -> None:
    encoded = value.encode('utf-8')
    digest.update(len(encoded).to_bytes(8, 'big'))
    digest.update(encoded)

def ordered_skill_pool_identity(skills, *, skill_text_fn):
    digest = hashlib.sha256()
    for index, skill in enumerate(skills):
        payload = SkillTable._skill_payload(skill)
        skill_id = str(payload.get('skill_id') or payload.get('canonical_skill_id') or payload.get('id') or index)
        _update_field(digest, skill_id)
        _update_field(digest, str(skill_text_fn(payload)))
    return {
        'algorithm': 'sha256_ordered_skill_id_and_serialized_text_v1',
        'count': len(skills),
        'digest': digest.hexdigest(),
    }
```

- [ ] **Step 4: Implement strict compatibility validation**

`validate_verified_resume_skill_table` checks `skill_table.E` rank and exact shape, each expected config key, and current identity against either embedded identity or captured resume sidecar identity. Every unverifiable or mismatched case raises `ValueError` containing `verified resume skill table`; success reports identity source and tensor shape.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_resume_skill_table.py
git add clstr/stage0_skill_pool_identity.py tests/test_stage0_resume_skill_table.py
git commit -m 'feat: verify stage0 skill table identity on resume'
```

Expected: all tests pass before commit.

### Task 4: Integrate Cache and Resume into Stage0 Training

**Files:**
- Modify: `clstr/retrieval_warmup.py:304-329,930-1889`
- Extend: `tests/test_stage0_frozen_backbone_cache.py`
- Extend: `tests/test_stage0_resume_skill_table.py`

- [ ] **Step 1: Write failing orchestration tests**

Using existing tiny-model monkeypatch patterns, run a matching resume with `resume_skill_table_mode='verified_checkpoint'`, assert `rebuild_skill_table` is not called, and inspect `report['skill_table_resume']['rebuild_skipped']`. Add a two-step schedule-cache run asserting nonzero unique rows, exact lookup count, and a changed trainable projection parameter.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_frozen_backbone_cache.py tests/test_stage0_resume_skill_table.py
```

Expected: failures because trainer options and reports are absent.

- [ ] **Step 3: Capture resume sidecar before output rewrite**

Extend `_load_stage0_resume_state` to resolve the concrete checkpoint path, read sibling `selected_skills.jsonl` before the current run rewrites it, and return `resume_selected_skills_path` plus `resume_selected_skill_rows`.

- [ ] **Step 4: Add validated trainer options**

```python
frozen_backbone_cache_mode: str = 'off'
frozen_backbone_cache_batch_size: int = 128
resume_skill_table_mode: str = 'rebuild'
```

Accept only `off|schedule` and `rebuild|verified_checkpoint`. Reject schedule mode when `train_encoder_backbone` is true or the loaded backbone contains a trainable parameter.

- [ ] **Step 5: Gate skill rebuilding**

Compute current ordered identity with `model.skill_table.skill_text_fn`. In verified mode, validate checkpoint tensor, config, and identity and log `skill_table_rebuild_skipped_verified_resume`; otherwise preserve the existing rebuild and progress callbacks. Load checkpoint state afterward as before. Store identity in all new checkpoint configs and payloads.

- [ ] **Step 6: Build cache after start-step resolution**

Assign row indices after shuffling. Once resume determines `start_step`, build the exact remaining schedule cache. Log requested mode, unique rows, bytes, build duration, and identity before training.

- [ ] **Step 7: Use cached pooled features without objective changes**

```python
if frozen_backbone_cache is None:
    h = model.encode_states(query_texts)
else:
    h = frozen_backbone_cache.project_batch(
        batch,
        projection_fn=model.encoder.project_pooled,
        device=device,
    )
if route_scorer == 'unified_memory':
    m_0 = model.initial_belief(h, top_k=belief_top_k)
    logits = model.unified_route_logits(h, m_0)
else:
    logits = model.skill_table.retrieval_logits(h)
```

Do not change labels, positives, online top-32 mining, loss weights, optimizer state, clipping, or metrics.

- [ ] **Step 8: Record durable reports**

Put cache configuration and pool identity in `checkpoint_config`; put live cache and resume reports in setup status, named checkpoints, `latest.pt`, and `train_report.json`. Call `report()` at finalization so lookup count is current.

- [ ] **Step 9: Verify regression suite and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_state_encoder_pooled_cache.py tests/test_stage0_frozen_backbone_cache.py tests/test_stage0_resume_skill_table.py tests/test_stage0_biencoder_protocol.py tests/test_retrieval_warmup_checkpoint_init.py tests/test_model_pipeline.py
git add clstr/retrieval_warmup.py tests/test_stage0_frozen_backbone_cache.py tests/test_stage0_resume_skill_table.py
git commit -m 'feat: accelerate frozen qwen stage0 continuation'
```

Expected: all selected tests pass before commit.

### Task 5: Wire CLI and Qwen Launchers

**Files:**
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Modify: `scripts/sbatch/run_qwen06_clstr_stage0_train.sh`
- Create: `tests/test_stage0_frozen_backbone_cache_launchers.py`

- [ ] **Step 1: Write failing static launcher tests**

Assert the Python CLI exposes and forwards all three options. Assert the shared shell forwards them. Assert the Qwen shell keeps `TRAIN_ENCODER_BACKBONE=0`, requests schedule caching, selects verified resume only with a checkpoint, and contains no `--unfreeze_backbone`.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_frozen_backbone_cache_launchers.py
```

Expected: missing-option assertions fail.

- [ ] **Step 3: Add CLI and shell forwarding**

Use argparse choices and defaults `off`, `128`, and `rebuild`. Add matching shared-shell environment defaults and append all three flags to `ARGS` unconditionally.

- [ ] **Step 4: Set Qwen policy**

Keep the backbone frozen. Export schedule cache and batch size 128. Choose `verified_checkpoint` only when `RESUME_CHECKPOINT_PATH` is nonempty; otherwise choose `rebuild`.

- [ ] **Step 5: Verify and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_frozen_backbone_cache_launchers.py tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_stage0_gate.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
bash -n scripts/sbatch/run_qwen06_clstr_stage0_train.sh
git add scripts/run_clstr_stage0_biencoder_train.py scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh scripts/sbatch/run_qwen06_clstr_stage0_train.sh tests/test_stage0_frozen_backbone_cache_launchers.py
git commit -m 'feat: enable verified qwen stage0 cache launch'
```

Expected: all commands pass before commit.

### Task 6: Add Real-Qwen Parity and Speed Gate

**Files:**
- Create: `scripts/audit_stage0_frozen_backbone_cache.py`
- Create: `scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh`
- Create: `tests/test_stage0_cache_benchmark_contract.py`

- [ ] **Step 1: Write failing audit-contract tests**

Parse the Python audit with `ast.parse` and require fields for pooled, projected, logits, loss, gradient, parameter-update differences, steady-state speedup, projected end-to-end speedup, peak GPU memory, frozen-backbone parameter count, and status. Require the shell to request one GPU, keep the backbone frozen, and invoke the audit.

- [ ] **Step 2: Verify RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_cache_benchmark_contract.py
```

Expected: failures because audit files are absent.

- [ ] **Step 3: Implement fixed parity thresholds**

Use one checkpoint-aware model and deterministic batch. Save trainable state, run uncached backward and one AdamW update, restore state, then run cached. Declare before measurement:

```python
POOLED_ATOL = 0.0
PROJECTED_ATOL = 1e-6
LOGITS_ATOL = 1e-5
LOSS_ATOL = 1e-6
GRADIENT_ATOL = 2e-5
PARAMETER_UPDATE_ATOL = 2e-6
MIN_SPEEDUP = 1.5
```

Record maximum and mean differences and block on any threshold failure.

- [ ] **Step 4: Implement CUDA timing and wall-time projection**

Warm up CUDA, time repeated training-shaped uncached and cached iterations with CUDA events, measure cache build duration and peak allocation, and project a 1200-step continuation including cache construction. Status is `ok` only when parity passes, both speedups are at least 1.5, and trainable backbone count is zero. Exit 0 for `ok` and 2 otherwise.

- [ ] **Step 5: Add bounded Slurm wrapper**

Request one A800/H100/H200 GPU for 90 minutes. Default to the canonical step-1200 checkpoint, current data root, batch 128, two microbatches, five warmups, twenty timed iterations, and 1200 projected continuation steps. Write under `outputs/qwen06_clstr_postfix/stage0_cache_benchmark`.

- [ ] **Step 6: Verify and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_cache_benchmark_contract.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile scripts/audit_stage0_frozen_backbone_cache.py
bash -n scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh
git add scripts/audit_stage0_frozen_backbone_cache.py scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh tests/test_stage0_cache_benchmark_contract.py
git commit -m 'test: gate qwen stage0 cache parity and speed'
```

Expected: all checks pass before commit.

### Task 7: Verify and Apply the Measured Decision

**Files:**
- Modify only scoped files above when verification exposes a defect.
- Produce untracked artifacts under `outputs/qwen06_clstr_postfix/stage0_cache_benchmark`.

- [ ] **Step 1: Run complete focused CPU verification**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_state_encoder_pooled_cache.py tests/test_stage0_frozen_backbone_cache.py tests/test_stage0_resume_skill_table.py tests/test_stage0_frozen_backbone_cache_launchers.py tests/test_stage0_cache_benchmark_contract.py tests/test_stage0_biencoder_protocol.py tests/test_retrieval_warmup_checkpoint_init.py tests/test_model_pipeline.py tests/test_qwen_clstr_lineage.py tests/test_qwen_clstr_stage0_gate.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/encoders.py clstr/stage0_frozen_backbone_cache.py clstr/stage0_skill_pool_identity.py clstr/retrieval_warmup.py scripts/run_clstr_stage0_biencoder_train.py scripts/audit_stage0_frozen_backbone_cache.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
bash -n scripts/sbatch/run_qwen06_clstr_stage0_train.sh
bash -n scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh
git diff --check
```

Expected: all commands pass. The unrelated already-RED Stage1/2/4 wrapper tests are excluded from this acceleration gate.

- [ ] **Step 2: Finish the unchanged step-1200 gate**

After Job 110272 terminates, inspect small artifacts only. Run the fixed 2048-row handoff audit and Stage0 promotion/release gate. Do not continue if the Stage0 gate is not `ok`.

- [ ] **Step 3: Submit and monitor GPU benchmark**

```bash
sbatch scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh
```

Expected: Slurm returns a job ID. The JSON must show parity `ok`, both speedups at least 1.5, and zero trainable backbone parameters.

- [ ] **Step 4: Apply measured production decision**

If the benchmark passes, retain schedule cache and verified skill loading for step 1200-to-2400. If parity or speed is blocked, change the Qwen cache default back to `off`; retain verified skill loading only if its identity and parity checks pass. Re-run launcher tests and commit only a changed default.
