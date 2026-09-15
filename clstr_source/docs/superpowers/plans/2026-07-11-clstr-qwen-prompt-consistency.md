# CLSTR Qwen State-Query Prompt Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task before resuming `2026-07-11-qwen06-clstr-postfix-training.md`. Steps use checkbox (`- [ ]`) syntax for tracking. Do not delegate unless the user explicitly requests subagents.

**Goal:** Make one versioned causal Qwen state-query prompt apply consistently to CLSTR Stage0, Stage1/2/4, handoff audits, route evaluation, and inference while keeping action/observation and skill-document encoding unchanged.

**Architecture:** Add a small prompt-contract module, make `CLSTRModel.encode_states()` the only prompted state-query boundary, and keep `encode_observations()` as the raw transition-text boundary. Persist the resolved contract in Stage0 checkpoints, restore it when rebuilding the model, and add a checkpoint-driven Stage0 handoff mode that never manually prepends another instruction.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing CLSTR model/Stage0/Stage1/2/4 code, argparse, Bash.

---

## Fixed Contract

The Qwen main run uses:

```text
state_query_prompt_version=clstr_causal_state_v1
state_query_max_chars=2000
state_query_truncation=head_tail_v1
```

Rendered query:

```text
Instruct: Given an agent task or current execution state and interaction history, retrieve the skill or tool document most useful for the next action.
Query:{state_text}
```

Legacy compatibility remains explicit:

```text
query_text_format=raw         -> raw_state_v1
query_text_format=skillrouter -> sr_task_description_v1
```

No SR-prompt training control is added.

---

### Task 1: Add the Versioned State-Query Prompt Contract

**Files:**
- Create: `clstr/state_query_prompt.py`
- Modify: `clstr/model.py`
- Create: `tests/test_state_query_prompt.py`
- Modify: `tests/test_model_pipeline.py`

- [x] **Step 1: Write RED tests for rendering, truncation, and role separation**

Create `tests/test_state_query_prompt.py`:

```python
import pytest

from clstr.state_query_prompt import (
    CLSTR_CAUSAL_STATE_V1,
    HEAD_TAIL_V1,
    format_state_query,
    resolve_state_query_prompt_contract,
)


def test_clstr_causal_state_prompt_renders_exact_qwen_query_contract():
    text = format_state_query(
        "goal: inspect invoice",
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=2000,
        truncation=HEAD_TAIL_V1,
    )
    assert text == (
        "Instruct: Given an agent task or current execution state and interaction history, "
        "retrieve the skill or tool document most useful for the next action.\n"
        "Query:goal: inspect invoice"
    )


def test_head_tail_truncation_preserves_goal_and_latest_context():
    raw = "goal:" + ("a" * 90) + "latest:" + ("z" * 90)
    text = format_state_query(
        raw,
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=80,
        truncation=HEAD_TAIL_V1,
    )
    query = text.split("\nQuery:", 1)[1]
    assert len(query) == 80
    assert query.startswith("goal:")
    assert query.endswith("z" * 20)
    assert "...[state truncated]..." in query


def test_prompted_state_rejects_preformatted_instruct_text():
    with pytest.raises(ValueError, match="raw state text"):
        format_state_query(
            "Instruct: stale prompt\nQuery:goal",
            prompt_version=CLSTR_CAUSAL_STATE_V1,
            max_chars=2000,
            truncation=HEAD_TAIL_V1,
        )


def test_causal_contract_records_canonical_instruction_and_length_policy():
    contract = resolve_state_query_prompt_contract(
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=2000,
        truncation=HEAD_TAIL_V1,
    )
    assert contract["state_query_prompt_version"] == CLSTR_CAUSAL_STATE_V1
    assert contract["state_query_max_chars"] == 2000
    assert contract["state_query_truncation"] == HEAD_TAIL_V1
    assert "next action" in contract["state_query_instruction"]
```

Extend `tests/test_model_pipeline.py` with a capture encoder and this behavior:

```python
def test_model_prompts_states_but_not_transition_observations():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(
        d=3,
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
    )

    class CaptureEncoder:
        def __init__(self):
            self.calls = []

        def __call__(self, texts):
            self.calls.append(list(texts))
            return torch.zeros(len(texts), 3)

    model.encoder = CaptureEncoder()
    model.encode_states(["goal: find weather"])
    model.encode_observations(["weather result"])

    assert model.encoder.calls[0][0].startswith("Instruct:")
    assert model.encoder.calls[0][0].endswith("Query:goal: find weather")
    assert model.encoder.calls[1] == ["weather result"]
```

- [x] **Step 2: Run the focused tests and confirm RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_state_query_prompt.py \
  tests/test_model_pipeline.py::test_model_prompts_states_but_not_transition_observations
```

Expected: import failure because `clstr.state_query_prompt` and the new config fields do not exist.

- [x] **Step 3: Implement the prompt contract module**

Create `clstr/state_query_prompt.py` with these public surfaces:

```python
from __future__ import annotations

from typing import Any


RAW_STATE_V1 = "raw_state_v1"
SR_TASK_DESCRIPTION_V1 = "sr_task_description_v1"
CLSTR_CAUSAL_STATE_V1 = "clstr_causal_state_v1"
NO_TRUNCATION = "none"
HEAD_V1 = "head_v1"
HEAD_TAIL_V1 = "head_tail_v1"
STATE_TRUNCATION_MARKER = "\n...[state truncated]...\n"

_INSTRUCTIONS = {
    RAW_STATE_V1: None,
    SR_TASK_DESCRIPTION_V1: (
        "Given a task description, retrieve the most relevant skill document "
        "that would help an agent complete the task"
    ),
    CLSTR_CAUSAL_STATE_V1: (
        "Given an agent task or current execution state and interaction history, "
        "retrieve the skill or tool document most useful for the next action."
    ),
}

_DEFAULTS = {
    RAW_STATE_V1: {"max_chars": None, "truncation": NO_TRUNCATION},
    SR_TASK_DESCRIPTION_V1: {"max_chars": 1500, "truncation": HEAD_V1},
    CLSTR_CAUSAL_STATE_V1: {"max_chars": 2000, "truncation": HEAD_TAIL_V1},
}


def _truncate_state(text: str, *, max_chars: int | None, truncation: str) -> str:
    if max_chars is None or len(text) <= int(max_chars):
        return text
    budget = int(max_chars)
    if budget <= 0:
        raise ValueError("state_query_max_chars must be positive")
    if truncation == HEAD_V1:
        return text[:budget]
    if truncation == HEAD_TAIL_V1:
        marker = STATE_TRUNCATION_MARKER
        if budget <= len(marker) + 1:
            raise ValueError("state_query_max_chars is too small for head_tail_v1")
        remaining = budget - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return text[:head] + marker + text[-tail:]
    if truncation == NO_TRUNCATION:
        raise ValueError("over-length state cannot use truncation=none")
    raise ValueError(f"unsupported state_query_truncation: {truncation}")


def resolve_state_query_prompt_contract(
    *,
    prompt_version: str,
    max_chars: int | None = None,
    truncation: str | None = None,
    recorded_instruction: str | None = None,
) -> dict[str, Any]:
    version = str(prompt_version or "").strip()
    if version not in _INSTRUCTIONS:
        raise ValueError(f"unsupported state_query_prompt_version: {prompt_version}")
    instruction = _INSTRUCTIONS[version]
    if recorded_instruction is not None and str(recorded_instruction) != str(instruction or ""):
        raise ValueError("state query instruction does not match prompt version")
    resolved_max = _DEFAULTS[version]["max_chars"] if max_chars is None else int(max_chars)
    resolved_truncation = str(truncation or _DEFAULTS[version]["truncation"])
    if resolved_truncation not in {NO_TRUNCATION, HEAD_V1, HEAD_TAIL_V1}:
        raise ValueError(f"unsupported state_query_truncation: {resolved_truncation}")
    return {
        "state_query_prompt_version": version,
        "state_query_instruction": str(instruction or ""),
        "state_query_max_chars": resolved_max,
        "state_query_truncation": resolved_truncation,
    }


def format_state_query(
    text: str,
    *,
    prompt_version: str,
    max_chars: int | None = None,
    truncation: str | None = None,
) -> str:
    contract = resolve_state_query_prompt_contract(
        prompt_version=prompt_version,
        max_chars=max_chars,
        truncation=truncation,
    )
    raw = str(text or "")
    if contract["state_query_instruction"] and raw.lstrip().startswith("Instruct:"):
        raise ValueError("state query formatter requires raw state text, not preformatted Instruct text")
    raw = _truncate_state(
        raw,
        max_chars=contract["state_query_max_chars"],
        truncation=contract["state_query_truncation"],
    )
    instruction = contract["state_query_instruction"]
    return raw if not instruction else f"Instruct: {instruction}\nQuery:{raw}"
```

- [x] **Step 4: Route CLSTR state encoding through the contract**

Add to `CLSTRConfig`:

```python
state_query_prompt_version: str | None = None
state_query_max_chars: int | None = None
state_query_truncation: str | None = None
```

In `CLSTRConfig.__post_init__`, resolve legacy `state_text_format` only when no explicit prompt version is present:

```python
if self.state_query_prompt_version is None:
    self.state_query_prompt_version = (
        SR_TASK_DESCRIPTION_V1
        if self.state_text_format == "appworld_skillrouter_query"
        else RAW_STATE_V1
    )
contract = resolve_state_query_prompt_contract(
    prompt_version=self.state_query_prompt_version,
    max_chars=self.state_query_max_chars,
    truncation=self.state_query_truncation,
)
self.state_query_prompt_version = contract["state_query_prompt_version"]
self.state_query_max_chars = contract["state_query_max_chars"]
self.state_query_truncation = contract["state_query_truncation"]
```

Replace `_serialize_state_for_encoder` with one call to `format_state_query`; leave `encode_observations` unchanged.

- [x] **Step 5: Run tests and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_state_query_prompt.py \
  tests/test_model_pipeline.py::test_model_prompts_states_but_not_transition_observations \
  tests/test_model_pipeline.py::test_appworld_skillrouter_text_formats_match_skillrouter_baseline_templates
git add clstr/state_query_prompt.py clstr/model.py \
  tests/test_state_query_prompt.py tests/test_model_pipeline.py
git commit -m "feat: version clstr state query prompts"
```

---

### Task 2: Make Stage0 Use Raw Queries and Persist the Resolved Contract

**Files:**
- Modify: `clstr/retrieval_warmup.py`
- Modify: `scripts/run_clstr_stage0_biencoder_train.py`
- Modify: `scripts/run_skillret_retrieval_warmup.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Modify: `tests/test_stage0_biencoder_protocol.py`

- [x] **Step 1: Write RED tests for Stage0 prompt metadata and raw input**

Extend the defaults test without changing the legacy default:

```python
assert STAGE0_BIENCODER_DEFAULTS["query_text_format"] == "skillrouter"
assert STAGE0_BIENCODER_DEFAULTS["state_query_prompt_version"] is None
```

Add:

```python
def test_stage0_biencoder_records_explicit_causal_state_query_contract(monkeypatch, tmp_path):
    captured = {}

    def fake_run_skillret_retrieval_warmup(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        retrieval_warmup,
        "run_skillret_retrieval_warmup",
        fake_run_skillret_retrieval_warmup,
    )
    run_stage0_biencoder_train(
        data_root=tmp_path / "data",
        base_model_name="model",
        output_dir=tmp_path / "out",
        max_steps=1,
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
    )
    assert captured["state_query_prompt_version"] == "clstr_causal_state_v1"
    assert captured["state_query_max_chars"] == 2000
    assert captured["state_query_truncation"] == "head_tail_v1"
    assert captured["protocol_metadata"]["state_query_prompt_version"] == "clstr_causal_state_v1"


def test_stage0_raw_query_texts_do_not_prepend_instruction():
    rows = [{"query": "goal: inspect invoice"}]
    assert retrieval_warmup._stage0_raw_query_texts(rows) == ["goal: inspect invoice"]
```

- [x] **Step 2: Run tests and confirm RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py::test_stage0_biencoder_records_explicit_causal_state_query_contract \
  tests/test_stage0_biencoder_protocol.py::test_stage0_raw_query_texts_do_not_prepend_instruction
```

- [x] **Step 3: Add compatibility resolution and Stage0 arguments**

Add a resolver in `retrieval_warmup.py`:

```python
def _resolve_stage0_state_query_contract(
    *,
    query_text_format: str,
    state_query_prompt_version: str | None,
    state_query_max_chars: int | None,
    state_query_truncation: str | None,
) -> dict[str, Any]:
    version = state_query_prompt_version
    if version is None:
        if query_text_format == "skillrouter":
            version = SR_TASK_DESCRIPTION_V1
        elif query_text_format == "raw":
            version = RAW_STATE_V1
        else:
            raise ValueError(f"unsupported query_text_format: {query_text_format}")
    return resolve_state_query_prompt_contract(
        prompt_version=version,
        max_chars=state_query_max_chars,
        truncation=state_query_truncation,
    )


def _stage0_raw_query_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["query"]) for row in rows]
```

Add `state_query_prompt_version`, `state_query_max_chars`, and `state_query_truncation` to both Stage0 entry points. Configure `CLSTRConfig` with the resolved values. Replace the training-loop `query_text_format` branch with:

```python
query_texts = _stage0_raw_query_texts(batch)
```

The model's `encode_states()` applies the instruction exactly once.

- [x] **Step 4: Persist the contract in every Stage0 checkpoint/report**

Add these resolved fields to initial, per-step, final checkpoint configs, `stage0_protocol`, setup status, and final train report:

```python
"state_query_prompt_version": state_query_contract["state_query_prompt_version"],
"state_query_instruction": state_query_contract["state_query_instruction"],
"state_query_max_chars": state_query_contract["state_query_max_chars"],
"state_query_truncation": state_query_contract["state_query_truncation"],
```

Keep `query_text_format` as compatibility metadata; it must not drive encoding once an explicit prompt version exists.

- [x] **Step 5: Add CLI and shell controls**

Add to both Python CLIs:

```python
parser.add_argument("--state_query_prompt_version")
parser.add_argument("--state_query_max_chars", type=int)
parser.add_argument("--state_query_truncation", choices=["none", "head_v1", "head_tail_v1"])
```

Add to the generic Stage0 sbatch wrapper:

```bash
STATE_QUERY_PROMPT_VERSION=${STATE_QUERY_PROMPT_VERSION:-}
STATE_QUERY_MAX_CHARS=${STATE_QUERY_MAX_CHARS:-}
STATE_QUERY_TRUNCATION=${STATE_QUERY_TRUNCATION:-}
```

Append the corresponding CLI arguments only when non-empty, preserving historical defaults.

- [x] **Step 6: Run tests and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
git add clstr/retrieval_warmup.py scripts/run_clstr_stage0_biencoder_train.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  tests/test_stage0_biencoder_protocol.py
git commit -m "feat: persist stage0 state query contract"
```

---

### Task 3: Restore the Contract and Separate State From Transition Encoding

**Files:**
- Modify: `clstr/stage_checkpoint_init.py`
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage4_act_train.py`
- Modify: `clstr/alfworld_eval.py`
- Modify: `clstr/prior_gate_sweep.py`
- Modify: `clstr/stage2_transition_row_diagnostics.py`
- Modify: `clstr/stage2_memory_rerank_eval.py`
- Modify: `clstr/current_state_route_eval.py`
- Modify: `clstr/mt_ablation_eval.py`
- Modify: `tests/test_stage_checkpoint_init.py`
- Modify: `tests/test_full_base_train.py`
- Modify: `tests/test_stage4_act_train.py`
- Modify: `tests/test_alfworld_eval.py`

- [x] **Step 1: Write RED checkpoint-reconstruction test**

Extend the Stage0 checkpoint fixture with:

```python
"state_query_prompt_version": "clstr_causal_state_v1",
"state_query_instruction": (
    "Given an agent task or current execution state and interaction history, "
    "retrieve the skill or tool document most useful for the next action."
),
"state_query_max_chars": 2000,
"state_query_truncation": "head_tail_v1",
```

Assert:

```python
assert config["state_query_prompt_version"] == "clstr_causal_state_v1"
assert model.config.kwargs["state_query_prompt_version"] == "clstr_causal_state_v1"
assert model.config.kwargs["state_query_max_chars"] == 2000
assert model.config.kwargs["state_query_truncation"] == "head_tail_v1"
```

- [x] **Step 2: Write RED role-aware cache test**

Add a fake model that records calls separately:

```python
class _RoleAwareCountingModel(_CountingEncodeModel):
    def __init__(self, dim=3):
        super().__init__(dim=dim)
        self.encoded_states = []
        self.encoded_observations = []

    def encode_states(self, texts):
        self.encoded_states.extend(str(text) for text in texts)
        return super().encode_observations(texts)

    def encode_observations(self, texts):
        self.encoded_observations.extend(str(text) for text in texts)
        return super().encode_observations(texts)


def test_full_base_cache_uses_prompted_role_only_for_state_and_next_state():
    model = _RoleAwareCountingModel()
    rows = [{
        "state_text": "current state",
        "action_text": "current action",
        "next_observation_text": "new observation",
        "next_state_text": "successor state",
    }]
    _attach_full_base_embedding_cache(model, rows, encode_batch_size=8)
    assert set(model.encoded_states) == {"current state", "successor state"}
    assert set(model.encoded_observations) == {"current action", "new observation"}
```

Add a Stage4 test with a fake model whose `encode_states` and `encode_observations` return distinguishable tensors; assert `h`/`h_next` use the state path while action/next-observation use the observation path.

- [x] **Step 3: Run focused tests and confirm RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage_checkpoint_init.py::test_build_model_from_stage0_checkpoint_uses_stage0_config_and_weights \
  tests/test_full_base_train.py::test_full_base_cache_uses_prompted_role_only_for_state_and_next_state
```

- [x] **Step 4: Restore and validate the checkpoint contract**

In `_stage0_config_from_checkpoint`, resolve explicit fields first and map legacy checkpoints only when they are absent:

```python
legacy_query_format = str(raw.get("query_text_format") or "skillrouter")
prompt_version = raw.get("state_query_prompt_version")
if prompt_version is None:
    prompt_version = (
        SR_TASK_DESCRIPTION_V1
        if legacy_query_format == "skillrouter"
        else RAW_STATE_V1
    )
prompt_contract = resolve_state_query_prompt_contract(
    prompt_version=str(prompt_version),
    max_chars=raw.get("state_query_max_chars"),
    truncation=raw.get("state_query_truncation"),
    recorded_instruction=raw.get("state_query_instruction"),
)
config.update(prompt_contract)
```

Pass all four resolved fields into `CLSTRConfig` in `build_clstr_model_from_stage0_checkpoint`.

- [x] **Step 5: Add role-aware encoding helpers**

In `full_base_train.py`:

```python
STATE_QUERY_ROLE = "state_query"
TRANSITION_TEXT_ROLE = "transition_text"


def _encode_state_queries(model: Any, texts: list[str]) -> torch.Tensor:
    encode = getattr(model, "encode_states", None)
    if callable(encode):
        return encode(texts)
    config = getattr(model, "config", None)
    prompt_version = getattr(config, "state_query_prompt_version", RAW_STATE_V1)
    if prompt_version != RAW_STATE_V1:
        raise TypeError("prompted model must expose encode_states(texts)")
    return _encode(model, texts)


def _batch_cached_or_encode(
    model: Any,
    rows: list[dict[str, Any]],
    embedding_key: str,
    text_key: str,
    device: torch.device,
    *,
    text_role: str,
) -> torch.Tensor:
    cached = [_cached_embedding(row, embedding_key, device) for row in rows]
    if cached and all(item is not None for item in cached):
        return torch.cat([item for item in cached if item is not None], dim=0)
    texts = [str(row.get(text_key) or "") for row in rows]
    if text_role == STATE_QUERY_ROLE:
        return _encode_state_queries(model, texts).to(device)
    if text_role == TRANSITION_TEXT_ROLE:
        return _encode(model, texts).to(device)
    raise ValueError(f"unsupported text_role: {text_role}")
```

Update every call site explicitly:

- `state_text`, `next_state_text`, policy state: `STATE_QUERY_ROLE`;
- `action_text`, admissible actions, observations, next observations, replay-prefix fields: `TRANSITION_TEXT_ROLE`.

- [x] **Step 6: Split embedding caches by role**

In `_attach_policy_embedding_cache`, encode state texts through a state-query batching helper and action texts through `_encode_text_batches`.

In `_attach_full_base_embedding_cache`, maintain separate state and transition reference collections. Deduplicate within each role, not across roles, so identical literal text can produce different role embeddings. Return separate counts:

```python
"state_query_unique_text_count": len(state_texts),
"transition_text_unique_text_count": len(transition_texts),
```

- [x] **Step 7: Run focused tests and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage_checkpoint_init.py \
  tests/test_full_base_train.py \
  tests/test_stage4_act_train.py
git add clstr/stage_checkpoint_init.py clstr/full_base_train.py \
  clstr/stage4_act_train.py tests/test_stage_checkpoint_init.py \
  tests/test_full_base_train.py tests/test_stage4_act_train.py
git commit -m "fix: restore prompted state encoding downstream"
```

If this test set is too large for the storage node, run the named new tests locally and submit the complete set as a short CPU Slurm job. Do not retry a killed local suite.

---

### Task 4: Make Stage0 Handoff and Audit Use the Checkpoint Prompt

**Files:**
- Modify: `clstr/full_base_train.py`
- Modify: `clstr/stage0_handoff_audit.py`
- Modify: `clstr/stage0_handoff_row_diagnostics.py`
- Modify: `scripts/audit_clstr_stage0_handoff_coverage.py`
- Modify: `scripts/run_clstr_stage1_heads_init.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Modify: `scripts/run_clstr_stage4_act_train.py`
- Modify: `tests/test_clstr_topm_candidate_handoff.py`
- Modify: `tests/test_stage0_handoff_audit.py`

- [x] **Step 1: Write RED handoff test for checkpoint-driven prompting**

Add `checkpoint_state_query` to the requested mode and use a fake model that records `encode_states` inputs:

```python
def test_stage0_handoff_checkpoint_mode_passes_raw_state_to_model_prompt_boundary():
    model = _Stage0TopMModel()
    model.state_inputs = []

    def encode_states(texts):
        model.state_inputs.extend(texts)
        return model.encode_observations(texts)

    model.encode_states = encode_states
    rows = [{
        "state_text": "goal: inspect invoice",
        "action_text": "open inbox",
        "next_observation_text": "invoice visible",
        "loss_mask": {},
    }]
    current = _stage0_handoff_query_text(
        rows[0],
        target="current",
        mode="checkpoint_state_query",
    )
    next_text = _stage0_handoff_query_text(
        rows[0],
        target="next",
        mode="checkpoint_state_query",
    )
    assert current == "goal: inspect invoice"
    assert not current.startswith("Instruct:")
    assert "previous_action: open inbox" in next_text
```

Extend `tests/test_stage0_handoff_audit.py` so `checkpoint_state_query` is evaluated with `_AuditModel.encode_states`, and assert the report key exists and reaches the expected positive.

- [x] **Step 2: Run tests and confirm RED**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py::test_stage0_handoff_checkpoint_mode_passes_raw_state_to_model_prompt_boundary \
  tests/test_stage0_handoff_audit.py
```

- [x] **Step 3: Add checkpoint-driven handoff mode**

Change:

```python
STAGE0_HANDOFF_QUERY_MODES = {
    "raw_state",
    "skillrouter_state",
    "checkpoint_state_query",
}
```

For `checkpoint_state_query`, `_stage0_handoff_query_text` returns the raw current/next state composition. In `_attach_stage0_topm_candidates` and `_rank_stage0_candidates`:

```python
if query_mode == "checkpoint_state_query":
    state_h = _encode_state_queries(model, state_texts).to(device)
    next_h = _encode_state_queries(model, next_texts).to(device)
else:
    state_h = _encode(model, state_texts).to(device)
    next_h = _encode(model, next_texts).to(device)
```

This preserves historical `raw_state` and manually wrapped `skillrouter_state` behavior while making the new Qwen run checkpoint-driven and single-prefix.

- [x] **Step 4: Expose the mode in active Stage1/2/4 CLIs**

Add `--stage0_handoff_query_mode` to Stage1 and extend Stage2/4 choices:

```python
choices=sorted(STAGE0_HANDOFF_QUERY_MODES)
```

Pass Stage1's value through `run_clstr_stage1_heads_init`. Keep generic defaults unchanged for historical launchers; the Qwen wrappers in the parent plan will set `checkpoint_state_query` explicitly.

Change the audit CLI default to `checkpoint_state_query` only when invoked by the new Qwen audit wrapper; retain its generic multi-mode default for existing diagnostics.

- [x] **Step 5: Run tests and commit**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage0_handoff_audit.py \
  tests/test_stage_checkpoint_init.py
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage1_heads_init.py --help >/dev/null
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage2_full_base_train.py --help >/dev/null
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage4_act_train.py --help >/dev/null
git add clstr/full_base_train.py clstr/stage0_handoff_audit.py \
  scripts/audit_clstr_stage0_handoff_coverage.py \
  scripts/run_clstr_stage1_heads_init.py scripts/run_clstr_stage2_full_base_train.py \
  scripts/run_clstr_stage4_act_train.py \
  tests/test_clstr_topm_candidate_handoff.py tests/test_stage0_handoff_audit.py
git commit -m "fix: use checkpoint prompt for stage0 handoff"
```

---

### Task 5: Verify Prompt Consistency Before Resuming Qwen Training Readiness

**Files:**
- All files changed in Tasks 1-4.

- [x] **Step 1: Run the focused prompt suite**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_state_query_prompt.py \
  tests/test_model_pipeline.py \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_stage_checkpoint_init.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage0_handoff_audit.py \
  tests/test_alfworld_eval.py
```

Run `tests/test_full_base_train.py` and `tests/test_stage4_act_train.py` as a short CPU Slurm job if the storage node cannot safely complete them.

- [x] **Step 2: Run syntax/static checks**

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/state_query_prompt.py clstr/model.py clstr/retrieval_warmup.py \
  clstr/stage_checkpoint_init.py clstr/full_base_train.py \
  clstr/stage4_act_train.py clstr/stage0_handoff_audit.py \
  clstr/stage0_handoff_row_diagnostics.py clstr/alfworld_eval.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
git diff --check
```

- [x] **Step 3: Record completion and resume the parent plan**

Update `.planning/2026-07-11-qwen06-clstr-postfix-training/` so Phase 2B is complete and Phase 3 is in progress. Resume Task 1 of `docs/superpowers/plans/2026-07-11-qwen06-clstr-postfix-training.md`.

---

## Completion Criteria

- `encode_states` is the only prompted state-query boundary.
- `encode_observations` remains raw for action and observation transition inputs.
- Stage0 receives raw query strings and applies the checkpoint prompt once.
- new Qwen checkpoints record `clstr_causal_state_v1`, 2000 chars, and `head_tail_v1`.
- checkpoint reconstruction restores and validates the exact prompt contract.
- Stage1/2/4 state and next-state paths use the restored prompt.
- Stage0 handoff/audit uses `checkpoint_state_query` for the Qwen run.
- existing raw and SkillRouter legacy paths remain explicit and test-covered.
- no prompt-control training branch or backbone unfreezing is introduced.
