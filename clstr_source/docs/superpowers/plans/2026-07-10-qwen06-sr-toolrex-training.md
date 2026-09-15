# Qwen 0.6B SR and ToolREx Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build training-ready, test-covered Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B SR/ToolREx pipelines on this repository's own leakage-safe data, then validate them with small GPU smokes without submitting full jobs.

**Architecture:** Keep the new baselines separate from CLSTR and the BGE trainers. A shared data protocol owns protected-eval exclusion and deterministic splits; shared Qwen embedding/reranker primitives own model-specific pooling, prompts, losses, and checkpoint compatibility; SR and ToolREx orchestration modules supply their distinct mining, document, candidate, and objective protocols.

**Tech Stack:** Python 3.11, PyTorch, Transformers >=4.51, JSONL, pytest, existing `TrainingMonitor`, Slurm, local Qwen3-14B/Qwen3-Embedding-0.6B/Qwen3-Reranker-0.6B checkpoints.

---

## File Map

New core files:

- `clstr/qwen_method_data.py`: protected-eval manifests, normalized query hashes, grouped splits, and versioned SR/ToolREx data roots.
- `clstr/qwen_method_documents.py`: canonical original documents, validated ToolREx profiles, and expanded-document serialization.
- `clstr/toolrex_profile_generation.py`: pluggable Qwen3-14B generation/judging, JSON validation, resume, and progress output.
- `clstr/qwen_method_models.py`: last-token embedding, instruction formatting, causal reranker token scores, SR listwise loss, and ToolREx binary-token loss.
- `clstr/qwen_method_checkpoints.py`: fingerprints, RNG capture/restore, complete optimizer/scheduler state, and incompatible-resume rejection.
- `clstr/sr_paper_mining.py`: semantic/BM25/taxonomy/random mining and three-layer false-negative filtering.
- `clstr/qwen_embedding_method_train.py`: shared full-parameter Qwen embedding loop used by SR-Emb and Tool-Embed.
- `clstr/qwen_sr_train.py`: SR artifact loading, SR-Emb wrapper, top-20 group construction, and SR-Rank listwise training.
- `clstr/qwen_toolrex_train.py`: 50K Tool-Embed sampling, top-100/200K Tool-Rank construction, Tool-Embed wrapper, and binary-token Tool-Rank training.

New CLIs and launchers:

- `scripts/build_qwen06_method_data.py`
- `scripts/run_qwen06_toolrex_profile_generation.py`
- `scripts/run_qwen06_sr_negative_mining.py`
- `scripts/run_qwen06_sr_embedding_train.py`
- `scripts/run_qwen06_sr_reranker_train.py`
- `scripts/run_qwen06_toolrex_embedding_train.py`
- `scripts/run_qwen06_toolrex_reranker_train.py`
- matching files under `scripts/sbatch/`

Tests:

- `tests/test_qwen_method_data.py`
- `tests/test_qwen_method_documents.py`
- `tests/test_toolrex_profile_generation.py`
- `tests/test_qwen_method_models.py`
- `tests/test_qwen_method_checkpoints.py`
- `tests/test_sr_paper_mining.py`
- `tests/test_qwen_embedding_method_train.py`
- `tests/test_qwen_sr_train.py`
- `tests/test_qwen_toolrex_train.py`
- `tests/test_qwen06_training_launchers.py`

One existing compatibility surface may change:

- `clstr/qwen_route_eval.py`: identify fine-tuned Qwen method checkpoints from local metadata while preserving existing BGE/Qwen behavior.
- `tests/test_qwen_route_eval.py`: cover that metadata path.

## Task 0: Isolate the Work and Record the Baseline

**Files:**
- Read: `docs/superpowers/specs/2026-07-10-qwen06-sr-toolrex-training-design.md`
- Read: `clstr/qwen_route_eval.py`
- Read: `clstr/toolbench_qwen_reranker.py`
- Read: `clstr/training_monitor.py`

- [ ] **Step 1: Create an isolated feature worktree from the approved design commit**

Run through the `using-git-worktrees` skill. Use a branch such as
`feat/qwen06-sr-toolrex` rooted at the commit containing this implementation
plan. Place the worktree under
`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/` and verify it is ignored by the
source repository.

- [ ] **Step 2: Link only required ignored assets into the worktree**

Create untracked links for `models`, `data`, `.cache`, and `.tmp` to the current
project assets. Keep smoke `outputs/` local to the worktree so new runs cannot
overwrite existing results.

- [ ] **Step 3: Run the focused baseline**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_route_eval.py \
  tests/test_qwen_backend.py \
  tests/test_toolbench_qwen_reranker.py \
  tests/test_bge_sr_embedding_full_train.py \
  tests/test_bge_sr_reranker_train.py \
  tests/test_bge_toolrex_reranker_train.py
```

Expected: PASS. Record exact counts and runtime before creating production files.

## Task 1: Protected-Eval Manifest and Train-Safe Data Roots

**Files:**
- Create: `clstr/qwen_method_data.py`
- Create: `scripts/build_qwen06_method_data.py`
- Create: `scripts/sbatch/run_qwen06_method_data_build.sh`
- Create: `tests/test_qwen_method_data.py`

- [ ] **Step 1: Write failing normalization and exclusion tests**

Add tests with this public API:

```python
from clstr.qwen_method_data import (
    ProtectedEvalManifest,
    build_protected_eval_manifest,
    filter_protected_training_rows,
    normalized_query_hash,
    split_rows_by_group,
)


def test_normalized_query_hash_ignores_case_and_whitespace():
    assert normalized_query_hash("  Find   WEATHER\n") == normalized_query_hash("find weather")


def test_filter_protected_training_rows_rejects_id_and_text_overlap():
    manifest = ProtectedEvalManifest(
        query_ids=frozenset({"toolbench-g3-17"}),
        task_ids=frozenset(),
        trajectory_ids=frozenset(),
        query_hashes=frozenset({normalized_query_hash("protected request")}),
        positive_skill_ids=frozenset({"tool/weather"}),
        source_paths=(),
    )
    rows = [
        {"query_id": "toolbench-g3-17", "query_text": "different", "positive_skill_id": "tool/a"},
        {"query_id": "safe-1", "query_text": "Protected request", "positive_skill_id": "tool/b"},
        {"query_id": "safe-2", "query_text": "safe request", "positive_skill_id": "tool/c"},
    ]
    kept, report = filter_protected_training_rows(rows, manifest)
    assert [row["query_id"] for row in kept] == ["safe-2"]
    assert report["removed_by_reason"] == {"query_id": 1, "query_hash": 1}


def test_split_rows_by_group_never_splits_a_query_group():
    rows = [
        {"query_id": "q1", "positive_skill_id": "a"},
        {"query_id": "q1", "positive_skill_id": "b"},
        {"query_id": "q2", "positive_skill_id": "c"},
        {"query_id": "q3", "positive_skill_id": "d"},
    ]
    train, validation, report = split_rows_by_group(rows, validation_fraction=0.34, seed=13)
    train_ids = {row["query_id"] for row in train}
    validation_ids = {row["query_id"] for row in validation}
    assert train_ids.isdisjoint(validation_ids)
    assert report["group_overlap_count"] == 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_method_data.py
```

Expected: collection FAIL because `clstr.qwen_method_data` does not exist.

- [ ] **Step 3: Implement the immutable manifest and grouped split primitives**

Implement these exact value rules:

```python
@dataclass(frozen=True)
class ProtectedEvalManifest:
    query_ids: frozenset[str]
    task_ids: frozenset[str]
    trajectory_ids: frozenset[str]
    query_hashes: frozenset[str]
    positive_skill_ids: frozenset[str]
    source_paths: tuple[str, ...]


def normalize_query_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalized_query_hash(value: object) -> str:
    return hashlib.sha256(normalize_query_text(value).encode("utf-8")).hexdigest()


def _stable_group_bucket(group_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
```

`build_protected_eval_manifest()` must accept multiple JSONL paths and extract
IDs, normalized text hashes, and positive IDs from the field variants already
used by `qwen_route_eval.py`. `filter_protected_training_rows()` must apply the
ordered reasons `query_id`, `task_id`, `trajectory_id`, then `query_hash` and
return a count for every removal. `split_rows_by_group()` must hash the query ID
and keep all positives for that query together.

- [ ] **Step 4: Add the versioned-root builder and CLI**

The CLI must expose:

```text
--sr_source_root
--toolret_source_root
--toolbench_source_root
--protected_eval_paths
--sr_output_dir
--toolrex_output_dir
--validation_fraction
--seed
--max_rows_per_source
```

It must write `skill_pool.jsonl`/`tools.jsonl`, `train_queries.jsonl`,
`validation_queries.jsonl`, `protected_eval_manifest.json`, `leakage_report.json`,
and `manifest.json`. It exits `2` unless both leakage reports are `status=ok`.
The Slurm launcher defaults to `MAX_ROWS_PER_SOURCE=100`, writes stdout under a
unique smoke directory, and requires `FULL_RUN=1` before accepting `ALL`.

- [ ] **Step 5: Verify GREEN and regression coverage**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_method_data.py tests/test_toolbench_clean_training_export.py
bash -n scripts/sbatch/run_qwen06_method_data_build.sh
```

Expected: PASS.

- [ ] **Step 6: Commit the data boundary**

```bash
git add clstr/qwen_method_data.py scripts/build_qwen06_method_data.py \
  scripts/sbatch/run_qwen06_method_data_build.sh tests/test_qwen_method_data.py
git commit -m "feat: build leakage-safe qwen method data"
```

## Task 2: Canonical and Expanded Tool Documents

**Files:**
- Create: `clstr/qwen_method_documents.py`
- Create: `tests/test_qwen_method_documents.py`

- [ ] **Step 1: Write failing serialization/profile tests**

```python
import pytest

from clstr.qwen_method_documents import (
    normalize_tool_profile,
    serialize_expanded_tool,
    serialize_original_tool,
    tool_document_hash,
)


TOOL = {
    "skill_id": "tool/weather/current",
    "name": "current_weather",
    "description": "Return current weather for coordinates.",
    "environment": "weather",
    "input_schema": {"required": ["lat", "lon"]},
    "output_schema": {"type": "object"},
    "body": "",
}


def test_original_tool_keeps_schema_when_body_is_empty():
    text = serialize_original_tool(TOOL)
    assert "current_weather" in text
    assert '"lat"' in text
    assert '"lon"' in text


def test_profile_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="three to five tags"):
        normalize_tool_profile({"function": "Get weather", "tags": ["weather"]})


def test_expanded_tool_is_stable_and_query_independent():
    profile = normalize_tool_profile(
        {"function": "Return current weather", "tags": ["weather", "location", "forecast"]}
    )
    first = serialize_expanded_tool(TOOL, profile)
    second = serialize_expanded_tool(dict(reversed(list(TOOL.items()))), profile)
    assert first == second
    assert tool_document_hash(TOOL) == tool_document_hash(dict(reversed(list(TOOL.items()))))
```

- [ ] **Step 2: Verify RED**

Run the new module tests and confirm import failure.

- [ ] **Step 3: Implement deterministic field-aware serialization**

Use `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`
for schemas. Omit empty sections. Normalize tags by lowercase whitespace collapse
and stable de-duplication. Reject unknown top-level profile keys, empty function,
function/optional fields over 20 whitespace-delimited words, and tag counts
outside 3-5.

The exported profile representation is:

```python
{
    "function": str,
    "tags": list[str],
    "when_to_use": str | None,
    "limitations": str | None,
}
```

- [ ] **Step 4: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_method_documents.py
git add clstr/qwen_method_documents.py tests/test_qwen_method_documents.py
git commit -m "feat: serialize qwen method tool documents"
```

## Task 3: Resumable Qwen3-14B Tool Profile Generation

**Files:**
- Create: `clstr/toolrex_profile_generation.py`
- Create: `scripts/run_qwen06_toolrex_profile_generation.py`
- Create: `scripts/sbatch/run_qwen06_toolrex_profile_generation.sh`
- Create: `tests/test_toolrex_profile_generation.py`

- [ ] **Step 1: Write failing parser, retry, and resume tests**

Define a fake backend with the same interface as the real backend:

```python
from clstr.toolrex_profile_generation import generate_tool_profiles


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return next(self.outputs)


def test_generate_profiles_retries_invalid_json_and_resumes(tmp_path):
    tools = [{"skill_id": "tool/a", "name": "A", "description": "Search documents"}]
    generator = FakeBackend([
        "not json",
        '{"tool_profile":{"function":"Search documents","tags":["search","documents","text"]}}',
    ])
    judge = FakeBackend(["true"])
    first = generate_tool_profiles(
        tools=tools,
        output_dir=tmp_path,
        generator=generator,
        judge=judge,
        generator_name="fake-generator",
        judge_name="fake-judge",
        max_attempts=2,
    )
    assert first["accepted_count"] == 1
    assert generator.calls == 2
    second = generate_tool_profiles(
        tools=tools,
        output_dir=tmp_path,
        generator=FakeBackend([]),
        judge=FakeBackend([]),
        generator_name="fake-generator",
        judge_name="fake-judge",
        max_attempts=2,
    )
    assert second["resumed_count"] == 1
```

Also assert that changing the original document hash triggers regeneration and
that `self_judged` is true when generator/judge names match.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_toolrex_profile_generation.py`; expect missing-module failure.

- [ ] **Step 3: Implement the backend-neutral pipeline**

Implement:

```python
class TextGenerationBackend(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("generation does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("generation JSON must be an object")
    return payload
```

`generate_tool_profiles()` writes append-only `profile_attempts.jsonl`, atomic
`profiles.jsonl`, `expanded_tools.jsonl`, `progress.json`, and `manifest.json`.
Resume keys are `(skill_id, original_document_hash, prompt_version)`. A judge
accepts only normalized `true`; every other output is rejection.
Generation parsing accepts either the normalized profile object itself or the
paper-style wrapper `{"tool_profile": {...}}`, then passes only the inner object
to `normalize_tool_profile()`.

- [ ] **Step 4: Wire the local Qwen3-14B CLI**

Use existing `QwenGenerationBackend(QwenBackendConfig(...))` with thinking
disabled, temperature 0, local files only, and a bounded output token count.
The CLI defaults to `models/Qwen3-14B`, exposes independent generator/judge
paths, and supports `--max_tools` for smoke.

The Slurm launcher defaults to a one-tool smoke and never defaults to `ALL`.

- [ ] **Step 5: Verify GREEN and shell syntax**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_toolrex_profile_generation.py
bash -n scripts/sbatch/run_qwen06_toolrex_profile_generation.sh
```

- [ ] **Step 6: Commit**

```bash
git add clstr/toolrex_profile_generation.py scripts/run_qwen06_toolrex_profile_generation.py \
  scripts/sbatch/run_qwen06_toolrex_profile_generation.sh tests/test_toolrex_profile_generation.py
git commit -m "feat: generate resumable toolrex profiles"
```

## Task 4: Shared Qwen Embedding and Reranker Primitives

**Files:**
- Create: `clstr/qwen_method_models.py`
- Create: `tests/test_qwen_method_models.py`

- [ ] **Step 1: Write failing embedding and relevance-loss tests**

```python
import torch

from clstr.qwen_method_models import (
    binary_relevance_token_loss,
    format_embedding_query,
    last_token_pool,
    multi_positive_listwise_loss,
    relevance_scores_from_logits,
)


def test_last_token_pool_handles_left_and_right_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    left_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert torch.equal(last_token_pool(hidden, left_mask), hidden[:, -1])
    assert torch.equal(last_token_pool(hidden, right_mask), torch.stack([hidden[0, 1], hidden[1, 2]]))


def test_relevance_scores_and_losses_keep_gradients():
    logits = torch.zeros(2, 7, requires_grad=True)
    logits.data[:, 5] = torch.tensor([2.0, -1.0])
    logits.data[:, 3] = torch.tensor([0.0, 1.0])
    scores = relevance_scores_from_logits(logits, relevant_token_id=5, irrelevant_token_id=3)
    labels = torch.tensor([1, 0])
    binary = binary_relevance_token_loss(logits, labels, relevant_token_id=5, irrelevant_token_id=3)
    listwise = multi_positive_listwise_loss(scores.view(1, 2), torch.tensor([[True, False]]))
    (binary + listwise).backward()
    assert logits.grad is not None
    assert scores.tolist() == [2.0, -2.0]


def test_embedding_query_uses_the_sr_instruction():
    text = format_embedding_query("fix a deployment", method="sr")
    assert text.startswith("Instruct: Given a task description")
    assert text.endswith("Query:fix a deployment")
```

- [ ] **Step 2: Verify RED**

Run the new test module and confirm missing-module failure.

- [ ] **Step 3: Implement model-specific pure functions**

Implement token scoring as:

```python
def relevance_scores_from_logits(logits, *, relevant_token_id, irrelevant_token_id):
    if logits.ndim != 2:
        raise ValueError("expected final-token logits [batch, vocab]")
    return logits[:, relevant_token_id] - logits[:, irrelevant_token_id]


def binary_relevance_token_loss(logits, labels, *, relevant_token_id, irrelevant_token_id):
    pair_logits = torch.stack(
        [logits[:, irrelevant_token_id], logits[:, relevant_token_id]], dim=-1
    ).float()
    return torch.nn.functional.cross_entropy(pair_logits, labels.long())


def multi_positive_listwise_loss(scores, positive_mask, temperature=1.0):
    scaled = scores.float() / float(temperature)
    if not positive_mask.any(dim=-1).all():
        raise ValueError("every listwise group needs at least one positive")
    positive_logits = scaled.masked_fill(~positive_mask, float("-inf"))
    return -(torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(scaled, dim=-1)).mean()
```

Add Qwen loaders that set left padding, ensure a pad token, set
`model.config.use_cache=False` during gradient checkpointing, and validate the
relevance token IDs from `1_LogitScore/config.json` or explicit CLI values.

- [ ] **Step 4: Verify GREEN and existing scorer compatibility**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_method_models.py tests/test_toolbench_qwen_reranker.py
```

- [ ] **Step 5: Commit**

```bash
git add clstr/qwen_method_models.py tests/test_qwen_method_models.py
git commit -m "feat: add qwen method model primitives"
```

## Task 5: Complete and Validated Checkpoint State

**Files:**
- Create: `clstr/qwen_method_checkpoints.py`
- Create: `tests/test_qwen_method_checkpoints.py`

- [ ] **Step 1: Write failing fingerprint/RNG/resume tests**

```python
import random
import torch
import pytest

from clstr.qwen_method_checkpoints import (
    capture_rng_state,
    config_fingerprint,
    restore_rng_state,
    validate_resume_metadata,
)


def test_rng_state_round_trip_restores_python_and_torch():
    random.seed(13)
    torch.manual_seed(13)
    state = capture_rng_state()
    expected = (random.random(), torch.rand(1).item())
    restore_rng_state(state)
    actual = (random.random(), torch.rand(1).item())
    assert actual == expected


def test_resume_rejects_changed_dataset_fingerprint():
    expected = {"dataset_fingerprint": "a", "pooling": "last_token", "objective": "sr_infonce"}
    actual = {**expected, "dataset_fingerprint": "b"}
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        validate_resume_metadata(expected=expected, actual=actual)
```

- [ ] **Step 2: Verify RED**

Run the new tests and confirm import failure.

- [ ] **Step 3: Implement deterministic metadata and full state helpers**

`config_fingerprint()` serializes with sorted keys and SHA-256. RNG state
contains Python, torch CPU, and all CUDA generator states when CUDA exists.
`validate_resume_metadata()` compares dataset fingerprint, model family,
pooling, query prompt version, document view, objective, top-k, and label-token
IDs; it lists every mismatch in one exception.

Provide `save_training_checkpoint()` and `load_training_checkpoint()` helpers
that persist model/tokenizer HF files plus `training_state.pt` containing step,
epoch, data cursor, optimizer, scheduler, scaler, RNG, metadata, and last
metrics. Write `latest.json` atomically and write the JSON-safe metadata subset
to `qwen_method_metadata.json` inside every Hugging Face checkpoint directory.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_method_checkpoints.py
git add clstr/qwen_method_checkpoints.py tests/test_qwen_method_checkpoints.py
git commit -m "feat: validate qwen method checkpoints"
```

## Task 6: Paper-Aligned SR Negative Mining

**Files:**
- Create: `clstr/sr_paper_mining.py`
- Create: `scripts/run_qwen06_sr_negative_mining.py`
- Create: `scripts/sbatch/run_qwen06_sr_negative_mining.sh`
- Create: `tests/test_sr_paper_mining.py`

- [ ] **Step 1: Write failing filter and quota tests**

```python
from clstr.sr_paper_mining import (
    FalseNegativeContext,
    filter_false_negative,
    mine_query_negatives,
    trigram_jaccard,
)


def test_false_negative_filters_name_body_and_embedding_similarity():
    context = FalseNegativeContext(
        positive_ids=frozenset({"p"}),
        positive_names=frozenset({"weather search"}),
        positive_texts=("find current weather by city",),
        max_embedding_similarity=0.92,
    )
    assert filter_false_negative("x", "Weather Search", "different", 0.2, context) == "name_match"
    assert filter_false_negative("x", "other", "find current weather by city", 0.2, context) == "body_overlap"
    assert filter_false_negative("x", "other", "different", 0.95, context) == "embedding_similarity"


def test_miner_uses_4_3_2_1_source_quota():
    result = mine_query_negatives(
        query_id="q",
        positive_ids={"p"},
        semantic_ranked=[f"s{i}" for i in range(10)],
        lexical_ranked=[f"l{i}" for i in range(10)],
        taxonomy_ids=[f"t{i}" for i in range(10)],
        random_ids=[f"r{i}" for i in range(10)],
        skill_records={skill_id: {"name": skill_id, "body": skill_id} for skill_id in ["p"] + [f"{x}{i}" for x in "sltr" for i in range(10)]},
        similarity_by_id={},
        seed=13,
    )
    assert [row["source"] for row in result.rows].count("semantic") == 4
    assert [row["source"] for row in result.rows].count("bm25") == 3
    assert [row["source"] for row in result.rows].count("taxonomy") == 2
    assert [row["source"] for row in result.rows].count("random") == 1
```

- [ ] **Step 2: Verify RED**

Run the test and confirm missing-module failure.

- [ ] **Step 3: Implement pure mining and filtering logic**

Compute normalized character trigrams after lowercase whitespace collapse.
Use `intersection / union`, returning zero for two empty sets. Apply canonical
ID/alias, name, Jaccard `>0.6`, then cosine `>0.92` filters in that order.

Implement BM25 statistics over the full skill corpus, but score only the base
semantic top-50 candidate IDs for each query. Use `k1=1.5` and `b=0.75` and
persist those values in the manifest. Deduplicate across source buckets; fill
quota shortfalls deterministically and report the original and fallback source.

- [ ] **Step 4: Implement shardable Qwen semantic mining**

The CLI loads `models/Qwen3-Embedding-0.6B`, uses the shared last-token encoder,
encodes skill shards once, writes fingerprinted embedding shards, and appends
one query result per line. It exposes `--max_queries`, `--query_start`, and
`--encode_batch_size`. Resume skips completed query IDs only when fingerprints
match.

- [ ] **Step 5: Verify GREEN and shell syntax**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_sr_paper_mining.py
bash -n scripts/sbatch/run_qwen06_sr_negative_mining.sh
```

- [ ] **Step 6: Commit**

```bash
git add clstr/sr_paper_mining.py scripts/run_qwen06_sr_negative_mining.py \
  scripts/sbatch/run_qwen06_sr_negative_mining.sh tests/test_sr_paper_mining.py
git commit -m "feat: mine paper-aligned sr negatives"
```

## Task 7: Shared Full-Parameter Qwen Embedding Trainer

**Files:**
- Create: `clstr/qwen_embedding_method_train.py`
- Create: `tests/test_qwen_embedding_method_train.py`

- [ ] **Step 1: Write failing batch/loss and resume-metadata tests**

```python
import torch

from clstr.qwen_embedding_method_train import build_embedding_batch, embedding_infonce_loss


def test_embedding_batch_places_each_positive_at_its_label_index():
    rows = [
        {"query_text": "q1", "positive_text": "p1", "negative_texts": ["n1", "n2"]},
        {"query_text": "q2", "positive_text": "p2", "negative_texts": ["n3", "n4"]},
    ]
    batch = build_embedding_batch(rows, method="sr")
    assert batch.labels.tolist() == [0, 3]
    assert batch.document_texts == ["p1", "n1", "n2", "p2", "n3", "n4"]


def test_embedding_infonce_loss_has_query_and_document_gradients():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    documents = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], requires_grad=True)
    loss, metrics = embedding_infonce_loss(queries, documents, torch.tensor([0, 1]), temperature=0.05)
    loss.backward()
    assert queries.grad is not None
    assert documents.grad is not None
    assert metrics["recall@1"] == 1.0
```

- [ ] **Step 2: Verify RED**

Run the new tests and confirm missing-module failure.

- [ ] **Step 3: Implement the shared train loop**

Define an `EmbeddingMethodConfig` dataclass containing method, model path,
document view, prompt version, max lengths, batch/GA, epoch/max-step controls,
LR, temperature, warmup ratio, checkpoint interval, and resume path.

The loop must:

1. load Qwen with left padding and last-token pooling;
2. enable gradient checkpointing and disable cache;
3. use BF16 autocast only on CUDA;
4. divide loss by gradient accumulation steps;
5. use AdamW weight decay 0.01 and `get_cosine_schedule_with_warmup`;
6. check loss and gradients for finite values before stepping;
7. write the existing `TrainingMonitor` outputs;
8. save complete state through `qwen_method_checkpoints`; and
9. support one epoch or a lower `max_steps` cap.

The run report must include parameter counts and verify that at least one named
backbone parameter changed from its initial checksum during smoke.

- [ ] **Step 4: Verify GREEN**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_embedding_method_train.py tests/test_qwen_method_models.py \
  tests/test_qwen_method_checkpoints.py
```

- [ ] **Step 5: Commit**

```bash
git add clstr/qwen_embedding_method_train.py tests/test_qwen_embedding_method_train.py
git commit -m "feat: train qwen embedding methods"
```

## Task 8: SR-Emb and SR-Rank Orchestration

**Files:**
- Create: `clstr/qwen_sr_train.py`
- Create: `scripts/run_qwen06_sr_embedding_train.py`
- Create: `scripts/run_qwen06_sr_reranker_train.py`
- Create: `scripts/sbatch/run_qwen06_sr_embedding_train.sh`
- Create: `scripts/sbatch/run_qwen06_sr_reranker_train.sh`
- Create: `tests/test_qwen_sr_train.py`

- [ ] **Step 1: Write failing SR row/group tests**

```python
import torch

from clstr.qwen_sr_train import build_sr_embedding_rows, build_sr_rank_groups


def test_sr_rows_use_mined_negatives_and_full_text():
    queries = [{"query_id": "q1", "query_text": "need weather", "positive_skill_ids": ["p"]}]
    skills = {
        "p": {"skill_id": "p", "name": "weather", "description": "current weather"},
        "n": {"skill_id": "n", "name": "climate", "description": "historical climate"},
    }
    mined = {"q1": [{"skill_id": "n", "source": "semantic"}]}
    rows, report = build_sr_embedding_rows(queries=queries, skills_by_id=skills, negatives_by_query=mined)
    assert rows[0]["negative_skill_ids"] == ["n"]
    assert "current weather" in rows[0]["positive_text"]
    assert report["missing_negative_query_count"] == 0


def test_sr_rank_groups_keep_multiple_positives():
    groups, report = build_sr_rank_groups(
        queries=[{"query_id": "q1", "query_text": "q", "positive_skill_ids": ["p1", "p2"]}],
        ranked_by_query={"q1": ["n", "p2", "p1"]},
        documents={"n": "N", "p1": "P1", "p2": "P2"},
        top_k=3,
    )
    assert groups[0]["positive_mask"] == [False, True, True]
    assert report["positive_not_in_topk"] == 0
```

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_qwen_sr_train.py`; expect missing-module failure.

- [ ] **Step 3: Implement SR-Emb wrapper and artifacts**

Read only `leakage_report.status=ok` data. Join `sr_negatives.jsonl` to train
queries, serialize full skill documents with SR field caps, and call the shared
embedding trainer with method `sr`, temperature 0.05, micro-batch 8, GA 4, max
length 2048, and the SR instruction.

Write `train_embedding_rows.jsonl`, `validation_embedding_rows.jsonl`, and a
join report before model loading.

- [ ] **Step 4: Implement top-20 construction and Qwen listwise training**

Encode the complete skill inventory and train/validation queries with the saved
SR-Emb checkpoint. Save candidate groups before loading the reranker. Count
missing positives as retriever misses. Apply SR false-negative decisions and
retain all valid positives in `positive_mask`.

Flatten each group into Qwen query/document prompts, collect final-token logits,
reshape relevance scores to `[groups, candidates]`, and use
`multi_positive_listwise_loss`. The defaults are batch 1, GA 16, LR 1e-5,
temperature 1, max length 4096, and one epoch.

- [ ] **Step 5: Add CLIs, Slurm wrappers, and smoke-safe defaults**

Both CLIs expose `--max_steps`, `--max_train_queries`, `--max_validation_queries`,
and `--resume_checkpoint_path`. Slurm scripts default to small smoke caps and
require `FULL_RUN=1` before accepting `ALL`.

- [ ] **Step 6: Verify GREEN and launchers**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_sr_train.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_qwen06_sr_embedding_train.py --help
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_qwen06_sr_reranker_train.py --help
bash -n scripts/sbatch/run_qwen06_sr_embedding_train.sh
bash -n scripts/sbatch/run_qwen06_sr_reranker_train.sh
```

- [ ] **Step 7: Commit**

```bash
git add clstr/qwen_sr_train.py scripts/run_qwen06_sr_embedding_train.py \
  scripts/run_qwen06_sr_reranker_train.py scripts/sbatch/run_qwen06_sr_embedding_train.sh \
  scripts/sbatch/run_qwen06_sr_reranker_train.sh tests/test_qwen_sr_train.py
git commit -m "feat: prepare qwen sr training"
```

## Task 9: Tool-Embed Data and Training

**Files:**
- Create: `clstr/qwen_toolrex_train.py`
- Create: `scripts/run_qwen06_toolrex_embedding_train.py`
- Create: `scripts/sbatch/run_qwen06_toolrex_embedding_train.sh`
- Create: `tests/test_qwen_toolrex_train.py`

- [ ] **Step 1: Write failing 50K sampling and random-negative tests**

```python
from clstr.qwen_toolrex_train import build_tool_embed_rows, stratified_query_sample


def test_tool_embed_sampling_is_deterministic_and_cross_query():
    queries = [
        {"query_id": "q1", "query_text": "alpha", "positive_skill_ids": ["a"], "source": "s1", "category": "c1"},
        {"query_id": "q2", "query_text": "beta", "positive_skill_ids": ["b"], "source": "s1", "category": "c2"},
        {"query_id": "q3", "query_text": "gamma", "positive_skill_ids": ["c"], "source": "s2", "category": "c1"},
    ]
    documents = {"a": "A", "b": "B", "c": "C"}
    first = stratified_query_sample(queries, max_rows=3, seed=13)
    second = stratified_query_sample(queries, max_rows=3, seed=13)
    assert first == second
    rows, report = build_tool_embed_rows(first, documents, negatives_per_query=2, seed=13)
    for row in rows:
        assert row["positive_skill_id"] not in row["negative_skill_ids"]
        assert len(row["negative_skill_ids"]) == 2
    assert report["row_count"] == 3
```

- [ ] **Step 2: Verify RED**

Run the ToolREx test and confirm missing-module failure.

- [ ] **Step 3: Implement expanded/original document views and 50K builder**

Require accepted profiles for the expanded primary view. Join by tool ID and
document hash; reject stale profiles. Stratify by source then taxonomy, cap at
50,000, and select five negatives from other query groups while excluding all
known positives and aliases.

Write the sampled IDs once and reuse them for the original-document ablation.

- [ ] **Step 4: Wire Tool-Embed to the shared trainer**

Use method `toolrex`, Qwen3-Embedding-0.6B, full parameters, last-token pooling,
five random other-query negatives, InfoNCE, one epoch, BF16, cosine scheduling,
LR 2e-5, and complete resume state. The document view and query instruction are
explicit checkpoint metadata.

- [ ] **Step 5: Verify GREEN, CLI, and shell syntax**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_toolrex_train.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_qwen06_toolrex_embedding_train.py --help
bash -n scripts/sbatch/run_qwen06_toolrex_embedding_train.sh
```

- [ ] **Step 6: Commit**

```bash
git add clstr/qwen_toolrex_train.py scripts/run_qwen06_toolrex_embedding_train.py \
  scripts/sbatch/run_qwen06_toolrex_embedding_train.sh tests/test_qwen_toolrex_train.py
git commit -m "feat: prepare qwen tool embed training"
```

## Task 10: Tool-Rank Top-100 Data and Binary-Token Training

**Files:**
- Modify: `clstr/qwen_toolrex_train.py`
- Create: `scripts/run_qwen06_toolrex_reranker_train.py`
- Create: `scripts/sbatch/run_qwen06_toolrex_reranker_train.sh`
- Modify: `tests/test_qwen_toolrex_train.py`

- [ ] **Step 1: Add failing balanced-pair and health-gate tests**

```python
from clstr.qwen_toolrex_train import build_tool_rank_pairs, tool_rank_health


def test_tool_rank_pairs_are_group_aware_and_balanced():
    pairs, report = build_tool_rank_pairs(
        queries=[{"query_id": "q1", "query_text": "q", "positive_skill_ids": ["p"]}],
        ranked_by_query={"q1": ["n1", "p", "n2"]},
        documents={"p": "P", "n1": "N1", "n2": "N2"},
        max_pairs=2,
        seed=13,
    )
    assert [row["label"] for row in pairs] == [1, 0]
    assert len({row["query_id"] for row in pairs}) == 1
    assert report["positive_pair_count"] == report["negative_pair_count"] == 1


def test_tool_rank_health_rejects_negative_class_collapse():
    report = tool_rank_health(
        positive_accuracy=0.0,
        negative_accuracy=1.0,
        positive_score_mean=-1.0,
        negative_score_mean=-2.0,
        finite=True,
    )
    assert report["status"] == "action_required"
    assert "zero_positive_accuracy" in report["blockers"]
```

- [ ] **Step 2: Verify RED**

Run the focused test names and confirm missing symbols.

- [ ] **Step 3: Implement top-100 candidate construction and 200K pair sampling**

Use the saved Tool-Embed checkpoint and the same expanded document view to
retrieve top-100. For each retained query, emit one positive and one
deterministically sampled hard negative. Cycle through query groups until the
200K cap; never truncate in the middle of a positive/negative pair. Persist
embedding top-100 positive coverage and label balance.

- [ ] **Step 4: Implement Qwen binary-token training and ranking validation**

Load `AutoModelForCausalLM`, construct the ToolREx relevance prompt, gather the
final-token relevant/irrelevant logits, and optimize two-class cross entropy.
Use full parameters, one epoch, BF16, gradient checkpointing, cosine scheduling,
LR 1e-5, and complete resume state.

Validation must evaluate whole query groups rather than a fixed prefix batch and
report class accuracy, score means/gap, R@1/5/10, MRR, and NDCG@10. Apply
`tool_rank_health()` before writing `status=ok`.

- [ ] **Step 5: Verify GREEN, CLI, and shell syntax**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_toolrex_train.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_qwen06_toolrex_reranker_train.py --help
bash -n scripts/sbatch/run_qwen06_toolrex_reranker_train.sh
```

- [ ] **Step 6: Commit**

```bash
git add clstr/qwen_toolrex_train.py scripts/run_qwen06_toolrex_reranker_train.py \
  scripts/sbatch/run_qwen06_toolrex_reranker_train.sh tests/test_qwen_toolrex_train.py
git commit -m "feat: prepare qwen tool rank training"
```

## Task 11: Launcher Guardrails and Route-Eval Compatibility

**Files:**
- Create: `tests/test_qwen06_training_launchers.py`
- Modify: `clstr/qwen_route_eval.py`
- Modify: `tests/test_qwen_route_eval.py`
- Modify: all new `scripts/sbatch/run_qwen06_*.sh`

- [ ] **Step 1: Write failing launcher guard tests**

```python
from pathlib import Path


def test_qwen06_launchers_default_to_smoke_and_require_full_run_opt_in():
    launchers = sorted(Path("scripts/sbatch").glob("run_qwen06_*.sh"))
    assert launchers
    for path in launchers:
        text = path.read_text(encoding="utf-8")
        assert "FULL_RUN=${FULL_RUN:-0}" in text
        assert any(token in text for token in (
            "MAX_STEPS=${MAX_STEPS:-2}",
            "MAX_TOOLS=${MAX_TOOLS:-1}",
            "MAX_QUERIES=${MAX_QUERIES:-8}",
            "MAX_ROWS_PER_SOURCE=${MAX_ROWS_PER_SOURCE:-100}",
        ))
        assert "FULL_RUN" in text and "ALL" in text
        assert "stdout.log" in text
```

Add a route-eval test that creates a checkpoint directory with
`qwen_method_metadata.json` and asserts method family/name, last-token pooling,
prompt mode, and reranker family are detected from metadata rather than the
directory name.

- [ ] **Step 2: Verify RED**

Run the launcher and route-eval tests; confirm guard/metadata assertions fail.

- [ ] **Step 3: Implement metadata-based checkpoint detection**

Add a small metadata reader to `qwen_route_eval.py`. Existing BGE path/name
detection remains the fallback. Fine-tuned embedding checkpoints must resolve
to `qwen3_embedding`; fine-tuned rerank checkpoints resolve to
`qwen3_causal_yes_no`. Route reports include method recipe, data fingerprint,
document view, and intentional paper deviations from the checkpoint metadata.

- [ ] **Step 4: Harden all Slurm defaults**

Every new launcher must:

- source `_clstr_gpu_env.sh`;
- default to a tiny cap;
- reject an uncapped value unless `FULL_RUN=1`;
- create its output directory before tee;
- accept a resume checkpoint; and
- print model, data, objective, cap, and output directory before Python starts.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen06_training_launchers.py tests/test_qwen_route_eval.py
for f in scripts/sbatch/run_qwen06_*.sh; do bash -n "$f"; done
git add clstr/qwen_route_eval.py tests/test_qwen_route_eval.py \
  tests/test_qwen06_training_launchers.py scripts/sbatch/run_qwen06_*.sh
git commit -m "feat: guard qwen method launch and eval"
```

## Task 12: CPU Verification, GPU Smokes, and Handoff

**Files:**
- Create: `scripts/sbatch/run_qwen06_method_smoke_chain.sh`
- Modify: `tests/test_qwen06_training_launchers.py`
- Create after runs: ignored smoke outputs under `outputs/qwen06_method_smoke/`

- [ ] **Step 1: Add a failing smoke-chain structure test**

The test must assert that the chain script contains only capped smoke commands,
never calls a full launcher with `ALL`, and passes each produced checkpoint to
the dependent stage.

- [ ] **Step 2: Verify RED, implement the smoke chain, and verify GREEN**

The chain is a Slurm job script, not a submission wrapper. It runs in order:

1. one-tool Qwen3-14B profile generation and judging;
2. fixture-sized method data build and SR mining;
3. two SR-Emb optimizer steps, checkpoint reload, then one resumed step;
4. two SR-Rank steps, checkpoint reload, then one resumed step;
5. two Tool-Embed steps, checkpoint reload, then one resumed step;
6. two Tool-Rank steps, checkpoint reload, then one resumed step; and
7. a tiny route eval from the reloaded SR and ToolREx checkpoints.

It declares `FULL_RUN=${FULL_RUN:-0}` and a fixed smoke cap variable so the
launcher guard test treats it like every other Qwen06 script.

Run the launcher test and `bash -n` after writing it.

- [ ] **Step 3: Run all focused CPU tests**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_qwen_method_data.py \
  tests/test_qwen_method_documents.py \
  tests/test_toolrex_profile_generation.py \
  tests/test_qwen_method_models.py \
  tests/test_qwen_method_checkpoints.py \
  tests/test_sr_paper_mining.py \
  tests/test_qwen_embedding_method_train.py \
  tests/test_qwen_sr_train.py \
  tests/test_qwen_toolrex_train.py \
  tests/test_qwen06_training_launchers.py \
  tests/test_qwen_route_eval.py \
  tests/test_toolbench_qwen_reranker.py
```

Expected: PASS with no warnings attributable to the new code.

- [ ] **Step 4: Run static verification**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/qwen_method_data.py clstr/qwen_method_documents.py \
  clstr/toolrex_profile_generation.py clstr/qwen_method_models.py \
  clstr/qwen_method_checkpoints.py clstr/sr_paper_mining.py \
  clstr/qwen_embedding_method_train.py clstr/qwen_sr_train.py \
  clstr/qwen_toolrex_train.py scripts/run_qwen06_*.py
for f in scripts/sbatch/run_qwen06_*.sh; do bash -n "$f"; done
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Submit only the capped GPU smoke after CPU verification**

Submit `scripts/sbatch/run_qwen06_method_smoke_chain.sh`. Record the Slurm job
ID and inspect output after model load and after every stage boundary. Do not
submit any full job or dependent full chain.

- [ ] **Step 6: Validate smoke artifacts**

Require:

- `progress.json`, `metrics.json`, curves, and checkpoint for each trainer;
- finite losses and gradients;
- changed backbone-parameter checksum;
- exact resume continuation rather than restart;
- both Tool-Rank label classes present;
- non-zero Tool-Rank positive accuracy and positive held-out score gap;
- successful HF checkpoint reload; and
- tiny route reports that identify the correct SR or ToolREx recipe.

Any failed requirement writes a blocker and returns to its owning task with a
new failing regression test before code changes.

- [ ] **Step 7: Run final verification from the isolated worktree**

Re-run Task 12 Steps 3 and 4 after all smoke fixes. Inspect `git status`, confirm
no generated data/output/model artifacts are tracked, and confirm the original
causal-mainline worktree is untouched.

- [ ] **Step 8: Commit the smoke chain and final documentation**

```bash
git add scripts/sbatch/run_qwen06_method_smoke_chain.sh tests/test_qwen06_training_launchers.py
git commit -m "test: verify qwen method smoke chain"
```

## Completion Gate

Before claiming training readiness:

1. Invoke `verification-before-completion` and rerun the exact final commands.
2. Invoke `requesting-code-review` for the complete branch diff.
3. Resolve every critical or important finding with a fresh RED/GREEN cycle.
4. Confirm no full training or full benchmark job was submitted.
5. Report the worktree, branch, commits, smoke job ID, artifacts, remaining
   paper deviations, and exact commands for later full submissions.
