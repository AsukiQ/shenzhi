# Modern Qwen3 Executor Signal Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconnect Qwen3 through a modern shared backend and run only low-cost executor-signal checks before deciding whether AppWorld RL is worth funding.

**Architecture:** Add one shared Qwen generation backend that owns model/tokenizer loading, chat-template formatting, local cache behavior, generation config, and model metadata. Refactor direct Qwen baselines and AppWorld executor generation to consume this backend, while keeping CLSTR routing/ACT checkpoints separate. Treat AppWorld as the first downstream reward gate, not the only benchmark.

**Tech Stack:** Python, PyTorch, Transformers, local HuggingFace cache under `/data/home/scyb713/run/xzf/AAAI/autodl-tmp`, Slurm sbatch for GPU smoke jobs.

---

### Task 1: Add Shared Qwen Backend

**Files:**
- Create: `clstr/qwen_backend.py`
- Test: `tests/test_qwen_backend.py`

- [ ] **Step 1: Write failing backend tests**

Add tests for:
- chat-template formatting passes `enable_thinking=False` when supported;
- fallback works when tokenizer does not accept `enable_thinking`;
- generation strips prompt tokens and records backend metadata;
- model path is configurable and defaults to local `models/Qwen3-8B`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_backend.py
```

Expected: fail because `clstr.qwen_backend` does not exist.

- [ ] **Step 3: Implement backend**

Create a small API:

```python
QwenBackendConfig(...)
QwenGenerationBackend(config, model=None, tokenizer=None)
backend.format_chat_prompt(system_prompt, user_prompt)
backend.generate_text(system_prompt, user_prompt)
backend.metadata()
```

The backend must not download on compute nodes by default: `local_files_only=True`.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_backend.py
```

Expected: pass.

### Task 2: Refactor AppWorld Qwen Executor To Use Backend

**Files:**
- Modify: `clstr/appworld_executor.py`
- Test: `tests/test_appworld_executor.py`

- [ ] **Step 1: Write failing test**

Add a test showing `QwenCodeGenerator` delegates prompt formatting and generation to the shared backend, and exposes metadata containing `qwen_backend_version`, `model_name_or_path`, `local_files_only`, and `enable_thinking`.

- [ ] **Step 2: Run focused test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_executor.py::test_qwen_code_generator_uses_shared_backend_metadata
```

Expected: fail because the current generator loads Transformers directly.

- [ ] **Step 3: Refactor generator**

Keep `QwenCodeGeneratorConfig` stable for callers, but internally construct `QwenGenerationBackend`. `generate(prompt)` should call:

```python
backend.generate_text(
    system_prompt="You are an AppWorld Python-code executor. Return code only.",
    user_prompt=prompt,
)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_appworld_executor.py tests/test_appworld_act_hrpo.py
```

Expected: pass.

### Task 3: Refactor Direct Qwen Action Scorer To Use Backend Prompt Rules

**Files:**
- Modify: `clstr/qwen_direct_policy.py`
- Test: `tests/test_qwen_direct_policy.py`

- [ ] **Step 1: Write failing test**

Add a test proving the direct action scorer can be built with an injected backend and does not separately instantiate a second model/tokenizer.

- [ ] **Step 2: Run focused test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_direct_policy.py::test_qwen_direct_scorer_can_use_shared_backend
```

Expected: fail because the scorer currently owns model/tokenizer directly.

- [ ] **Step 3: Implement minimal backend injection**

Preserve current constructor compatibility. Add optional `backend` argument; if provided, `QwenDirectAdmissibleActionScorer` uses `backend.generate_text(...)` and `backend.metadata()` instead of loading another model.

- [ ] **Step 4: Run focused tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_qwen_direct_policy.py tests/test_qwen_planner_intent.py
```

Expected: pass.

### Task 4: Add Low-Cost AppWorld Signal Smoke Wrapper

**Files:**
- Create: `scripts/sbatch/run_appworld_qwen3_signal_smoke.sh`
- Test: `tests/test_sbatch_scripts.py`

- [ ] **Step 1: Write failing sbatch test**

Assert the wrapper:
- uses shared CLSTR environment setup;
- defaults to `models/Qwen3-8B`;
- defaults to `MAX_TASKS=2`, `MAX_STEPS=3`, `MAX_INTERACTIONS=10`;
- writes to a fresh output directory;
- uses `--local_files_only`;
- does not submit full RL.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_sbatch_scripts.py::test_appworld_qwen3_signal_smoke_is_low_cost_and_offline
```

Expected: fail because wrapper does not exist.

- [ ] **Step 3: Add wrapper**

The wrapper should call `scripts/run_appworld_multistep_executor_eval.py` with `METHOD=qwen_only` by default. It should be a downstream executor signal check, not CLSTR training.

- [ ] **Step 4: Run syntax and focused test**

Run:

```bash
bash -n scripts/sbatch/run_appworld_qwen3_signal_smoke.sh
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_sbatch_scripts.py::test_appworld_qwen3_signal_smoke_is_low_cost_and_offline
```

Expected: pass.

### Task 5: Execute Smoke Only If Local Tests Pass

**Files:**
- No production code changes.
- Output: `outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_smoke`

- [ ] **Step 1: Preflight local model**

Run:

```bash
test -s models/Qwen3-8B/config.json
test -s models/Qwen3-8B/model.safetensors.index.json
```

Expected: both commands pass.

- [ ] **Step 2: Submit one small sbatch job**

Run only after Tasks 1-4 pass:

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_appworld_qwen3_signal_smoke.sh
```

- [ ] **Step 3: Supervise without frequent polling**

Check short smoke after about 15 minutes unless it ends earlier. Inspect `setup`, stdout, `report.json`, and `runs.jsonl`.

- [ ] **Step 4: Decision rule**

If success/reward is near zero and failures are due to Qwen code quality rather than CLSTR routing, do not start AppWorld RL. Next choices are:
- upgrade to a same-size newer Qwen local model and rerun the same smoke;
- or shift RL/executor testing to a dataset where the executor has denser/cleaner reward.

---

## Self-Review

- This plan does not make AppWorld the only benchmark; it makes AppWorld the first low-cost executor-signal gate.
- It keeps CLSTR routing/ACT artifacts separate from Qwen direct executor results.
- It avoids compute-node downloads by default.
- It does not submit full RL or full benchmark jobs.
