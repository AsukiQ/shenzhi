# AppWorld Official Executor Stage4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an official-style AppWorld executor path with Qwen3-14B and connect CLSTR as a schema-grounded skill evidence provider before any Stage4 training.

**Architecture:** Add a new AppWorld official-style executor module instead of further expanding the already large `clstr/appworld_executor.py`. Reuse `QwenGenerationBackend`, the existing CLSTR multi-step controllers, and AppWorld `world.execute()` / official evaluation, while changing the prompt protocol to ReAct/API-doc discovery and raising interaction budget through gated sbatch runs.

**Tech Stack:** Python 3.11, PyTorch/Transformers, AppWorld SDK, Slurm sbatch, existing CLSTR modules, pytest.

---

## File Structure

- Create `clstr/appworld_official_executor.py`
  - Owns official-style ReAct prompt/session loop, history trimming, evidence insertion, execution, and report row formatting.
- Create `scripts/run_appworld_official_executor_eval.py`
  - CLI entrypoint for qwen-only, static predictions, and CLSTR controller modes.
- Create `scripts/sbatch/run_appworld_official_executor_eval.sh`
  - Compute-node runner with Qwen model path, split path, max tasks, max interactions, and method env vars.
- Create `scripts/download_qwen3_14b.py`
  - Reproducible model download helper that unsets proxy variables and uses `HF_ENDPOINT=https://hf-mirror.com`.
- Create `tests/test_appworld_official_executor.py`
  - Unit tests for prompt, history, fake world execution, evidence handoff, and report metrics.
- Create `tests/test_appworld_official_executor_cli.py`
  - CLI argument and controller-construction tests without loading real models.
- Modify `clstr/qwen_backend.py`
  - Add explicit load metadata only if Task 2 shows the existing backend cannot report Qwen3-14B device/dtype cleanly. If Task 2 passes without this, leave the file unchanged.
- Modify `docs/superpowers/specs/2026-06-13-appworld-official-executor-stage4-design.md`
  - Only if implementation reveals a design correction.
- Modify `description.md`
  - Append concise results and gate decisions after each completed phase.

## Task 1: Add Qwen3-14B Download Helper

**Files:**
- Create: `scripts/download_qwen3_14b.py`
- Test: no pytest; verify by `--dry_run` command.

- [x] **Step 1: Create a dry-run-capable downloader**

Create `scripts/download_qwen3_14b.py` with:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


def clear_proxy_env() -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in PROXY_VARS}
    for name in PROXY_VARS:
        os.environ.pop(name, None)
    return previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="Qwen/Qwen3-14B")
    parser.add_argument("--local_dir", default="models/Qwen3-14B")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    previous_proxy = clear_proxy_env()
    os.environ["HF_ENDPOINT"] = str(args.endpoint)
    local_dir = Path(args.local_dir)
    print({"repo_id": args.repo_id, "local_dir": str(local_dir), "HF_ENDPOINT": os.environ["HF_ENDPOINT"], "cleared_proxy_vars": previous_proxy})
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(args.repo_id),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run dry-run**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/download_qwen3_14b.py --dry_run
```

Expected: prints `repo_id=Qwen/Qwen3-14B`, `local_dir=models/Qwen3-14B`, `HF_ENDPOINT=https://hf-mirror.com`, and cleared proxy variables.

- [x] **Step 3: Commit**

```bash
git add scripts/download_qwen3_14b.py
git commit -m "chore: add qwen3 14b mirror downloader"
```

## Task 2: Download and Load-Gate Qwen3-14B

**Files:**
- Use: `scripts/download_qwen3_14b.py`
- Create: `scripts/sbatch/run_qwen3_14b_load_smoke.sh`
- Test: compute-node smoke only.

- [x] **Step 1: Download from mirror on login node**

Run with proxy disabled:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/download_qwen3_14b.py
```

Expected: `models/Qwen3-14B` contains `config.json`, tokenizer files, and safetensors shards.

- [x] **Step 2: Create compute-node load smoke script**

Create `scripts/sbatch/run_qwen3_14b_load_smoke.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3_14b_load
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=00:30:00

set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python - <<'PY'
from clstr.qwen_backend import QwenBackendConfig, QwenGenerationBackend

backend = QwenGenerationBackend(
    QwenBackendConfig(
        model_name_or_path="models/Qwen3-14B",
        local_files_only=True,
        torch_dtype="bfloat16",
        max_new_tokens=16,
    )
)
print(backend.metadata())
print(backend.generate_text("You are a test runner.", "Reply with OK only."))
PY
```

- [x] **Step 3: Submit only this smoke**

Run:

```bash
sbatch scripts/sbatch/run_qwen3_14b_load_smoke.sh
```

Expected: job exits within 30 minutes and output includes backend metadata plus a short completion.

- [x] **Step 4: Commit**

```bash
git add scripts/sbatch/run_qwen3_14b_load_smoke.sh
git commit -m "test: add qwen3 14b load smoke"
```

## Task 3: Add Official-Style ReAct Prompt and Session Unit Tests

**Files:**
- Create: `tests/test_appworld_official_executor.py`
- Create: `clstr/appworld_official_executor.py`

- [x] **Step 1: Write failing prompt test**

Add to `tests/test_appworld_official_executor.py`:

```python
from clstr.appworld_official_executor import build_official_react_system_prompt


def test_official_react_prompt_uses_api_doc_discovery_not_static_api_dump():
    prompt = build_official_react_system_prompt(max_interactions=40)
    assert "apis.api_docs.show_app_descriptions()" in prompt
    assert "apis.api_docs.show_api_descriptions(app_name=" in prompt
    assert "apis.api_docs.show_api_doc(app_name=" in prompt
    assert "multi-step conversation" in prompt
    assert "Do not treat retrieved skill evidence as executable APIs" in prompt
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py::test_official_react_prompt_uses_api_doc_discovery_not_static_api_dump -q
```

Expected: FAIL because module/function does not exist.

- [x] **Step 3: Implement prompt function**

Create `clstr/appworld_official_executor.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


def build_official_react_system_prompt(*, max_interactions: int = 40) -> str:
    return f"""I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.

You will interact with AppWorld apps through a multi-step conversation using a Python REPL. Write one Python code block per turn. The environment will execute it and return printed output. Use the output to decide the next step until the task is complete.

Use these AppWorld API documentation helpers before assuming an API signature:
```python
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

You have at most {int(max_interactions)} interactions. Prefer minimal, reversible reads before writes. Call `apis.supervisor.complete_task()` only after the requested state change or answer is complete.

Retrieved skill evidence may be provided. It is advisory evidence only. Do not treat retrieved skill evidence as executable APIs, credentials, or a complete plan. Valid executable APIs must come from `apis.api_docs` or the AppWorld runtime.
"""
```

- [x] **Step 4: Run test to verify pass**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py::test_official_react_prompt_uses_api_doc_discovery_not_static_api_dump -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add clstr/appworld_official_executor.py tests/test_appworld_official_executor.py
git commit -m "feat: add official appworld react prompt"
```

## Task 4: Implement Official ReAct Session Loop with Fake World Tests

**Files:**
- Modify: `clstr/appworld_official_executor.py`
- Modify: `tests/test_appworld_official_executor.py`

- [x] **Step 1: Write fake-world execution test**

Add:

```python
from clstr.appworld_official_executor import OfficialReActExecutorConfig, run_official_react_task


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate_text(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls == 1:
            return "```python\nprint(apis.api_docs.show_app_descriptions())\n```"
        return "```python\napis.supervisor.complete_task()\n```"


class FakeWorld:
    def __init__(self):
        self.outputs = []
        self.completed = False

    def execute(self, code):
        self.outputs.append(code)
        if "complete_task" in code:
            self.completed = True
            return "completed"
        return "[{'name': 'spotify'}]"

    def task_completed(self):
        return self.completed

    def evaluate(self):
        class Report:
            def report(self):
                return {"success": True}
        return Report()


def test_official_react_loop_persists_history_and_stops_on_completion():
    row = run_official_react_task(
        task={"task_id": "fake_1", "instruction": "finish it", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=4),
    )
    assert row["task_completed"] is True
    assert row["evaluation_success"] is True
    assert row["steps"][0]["execute_output"] == "[{'name': 'spotify'}]"
    assert "Output:" in row["steps"][1]["user_prompt"]
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py::test_official_react_loop_persists_history_and_stops_on_completion -q
```

Expected: FAIL because loop classes/functions do not exist.

- [x] **Step 3: Implement minimal loop**

Add:

```python
@dataclass
class OfficialReActExecutorConfig:
    max_interactions: int = 40
    timeout_seconds: int = 120
    preflight_max_repairs: int = 0
    include_skill_evidence: bool = True
    max_history_chars: int = 16000


def _extract_code(text: str) -> str:
    from clstr.appworld_executor import extract_python_code
    return extract_python_code(text)


def _evaluation_success(world: Any) -> bool:
    try:
        report = world.evaluate().report()
    except Exception:
        return False
    if isinstance(report, dict):
        return bool(report.get("success", report.get("passed", False)))
    return "success" in str(report).lower() and "false" not in str(report).lower()


def _build_user_prompt(task: dict[str, Any], history: list[dict[str, Any]], evidence: str = "") -> str:
    parts = [
        f"Task: {task.get('instruction', '')}",
        f"Task id: {task.get('task_id', task.get('query_id', ''))}",
    ]
    if task.get("task_datetime"):
        parts.append(f"Task datetime: {task['task_datetime']}")
    if evidence:
        parts.append(evidence)
    for step in history:
        parts.append("Assistant code:\n```python\n" + str(step.get("code", "")) + "\n```")
        parts.append("Output:\n```\n" + str(step.get("execute_output", "")) + "\n```")
    parts.append("Write the next Python code block.")
    return "\n\n".join(parts)


def run_official_react_task(
    *,
    task: dict[str, Any],
    world: Any,
    generator: Any,
    config: OfficialReActExecutorConfig,
    evidence_builder: Any | None = None,
) -> dict[str, Any]:
    system_prompt = build_official_react_system_prompt(max_interactions=config.max_interactions)
    history: list[dict[str, Any]] = []
    row = {"task_id": task.get("task_id") or task.get("query_id"), "steps": []}
    for step_idx in range(max(1, int(config.max_interactions))):
        evidence = evidence_builder(task=task, history=history) if evidence_builder else ""
        user_prompt = _build_user_prompt(task, history, evidence)
        response = generator.generate_text(system_prompt, user_prompt)
        code = _extract_code(response)
        execute_output = str(world.execute(code))
        step = {
            "step_idx": step_idx,
            "user_prompt": user_prompt,
            "raw_response": response,
            "code": code,
            "execute_output": execute_output,
            "task_completed": bool(world.task_completed()),
        }
        row["steps"].append(step)
        history.append(step)
        if step["task_completed"]:
            break
    row["task_completed"] = bool(row["steps"][-1]["task_completed"]) if row["steps"] else False
    row["evaluation_success"] = _evaluation_success(world)
    row["success"] = bool(row["task_completed"] and row["evaluation_success"])
    return row
```

- [x] **Step 4: Run tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py -q
```

Expected: all tests in this file PASS.

- [x] **Step 5: Commit**

```bash
git add clstr/appworld_official_executor.py tests/test_appworld_official_executor.py
git commit -m "feat: add official appworld react loop"
```

## Task 5: Add CLSTR Evidence Builder Interface

**Files:**
- Modify: `clstr/appworld_official_executor.py`
- Modify: `tests/test_appworld_official_executor.py`

- [x] **Step 1: Write evidence test**

Add:

```python
def test_official_executor_hides_raw_skill_text_and_records_evidence():
    captured = {}

    class EvidenceBuilder:
        def __call__(self, task, history):
            captured["called"] = True
            return "Retrieved evidence:\n- Relevant valid APIs: apis.spotify.show_song()\n- Useful workflow hint: inspect detail rows"

    row = run_official_react_task(
        task={"task_id": "fake_2", "instruction": "count songs", "required_apps": ["spotify"]},
        world=FakeWorld(),
        generator=FakeGenerator(),
        config=OfficialReActExecutorConfig(max_interactions=1),
        evidence_builder=EvidenceBuilder(),
    )
    assert captured["called"] is True
    assert "Retrieved evidence:" in row["steps"][0]["user_prompt"]
    assert "skillx/appworld" not in row["steps"][0]["user_prompt"].lower()
```

- [x] **Step 2: Run test**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py::test_official_executor_hides_raw_skill_text_and_records_evidence -q
```

Expected: PASS after Task 4 minimal evidence hook.

- [x] **Step 3: Add typed evidence result**

Add:

```python
@dataclass
class EvidenceResult:
    prompt_text: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

Update `run_official_react_task()` so an evidence builder may return `str` or `EvidenceResult`, and each step records `skill_evidence`.

- [x] **Step 4: Add test for diagnostics**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py -q
```

Expected: PASS and step rows contain evidence diagnostics.

- [x] **Step 5: Commit**

```bash
git add clstr/appworld_official_executor.py tests/test_appworld_official_executor.py
git commit -m "feat: add official executor evidence hook"
```

## Task 6: Add Official Executor CLI

**Files:**
- Create: `scripts/run_appworld_official_executor_eval.py`
- Create: `tests/test_appworld_official_executor_cli.py`

- [x] **Step 1: Write CLI parse test**

Create `tests/test_appworld_official_executor_cli.py`:

```python
from scripts import run_appworld_official_executor_eval as cli


def test_cli_accepts_qwen3_14b_and_official_react_defaults():
    parser = cli.build_parser()
    args = parser.parse_args([
        "--method", "qwen_only",
        "--tasks_path", "data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl",
        "--output_dir", "outputs/tmp",
        "--model_name_or_path", "models/Qwen3-14B",
    ])
    assert args.method == "qwen_only"
    assert args.model_name_or_path == "models/Qwen3-14B"
    assert args.max_interactions == 40
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor_cli.py -q
```

Expected: FAIL because script does not exist.

- [x] **Step 3: Implement parser skeleton**

Create script with `build_parser()` supporting:

```python
parser.add_argument("--method", choices=["qwen_only", "static_predictions", "clstr_multistep"], default="qwen_only")
parser.add_argument("--tasks_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
parser.add_argument("--model_name_or_path", default="models/Qwen3-14B")
parser.add_argument("--max_tasks", type=int, default=1)
parser.add_argument("--max_interactions", type=int, default=40)
parser.add_argument("--torch_dtype", default="bfloat16")
parser.add_argument("--max_new_tokens", type=int, default=768)
```

- [x] **Step 4: Run CLI test**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor_cli.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add scripts/run_appworld_official_executor_eval.py tests/test_appworld_official_executor_cli.py
git commit -m "feat: add official appworld executor cli"
```

### Task 6b: Add Qwen-Only Official Runtime Before Sbatch

**Reason:** Task 7's sbatch wrapper must execute a real single-task gate, not a parser-only script.

- [x] **Step 1: Add runtime test**

Add a fake-world/fake-generator CLI test that verifies `run_official_executor_eval()` writes `runs.jsonl` and `report.json`, uses `instruction_text`, forwards AppWorld runtime parameters, and closes the world.

- [x] **Step 2: Implement qwen-only runtime**

Implement task loading, `QwenGenerationBackend` construction, AppWorld world creation, `run_official_react_task()` execution, run/report writing, and clear `NotImplementedError` for CLSTR evidence modes until Task 8.

- [x] **Step 3: Verify**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py -q
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_official_executor.py scripts/run_appworld_official_executor_eval.py tests/test_appworld_official_executor_cli.py
```

Expected: PASS.

## Task 7: Add Sbatch Wrapper and Single-Task Smoke

**Files:**
- Create: `scripts/sbatch/run_appworld_official_executor_eval.sh`
- Test: `bash -n`.

- [x] **Step 1: Create wrapper**

Create:

```bash
#!/bin/bash
#SBATCH --job-name=appworld_official_exec
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=01:00:00

set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface}
export APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}
export APPWORLD_CACHE=${APPWORLD_CACHE:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld}
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
METHOD=${METHOD:-qwen_only}
TASKS_PATH=${TASKS_PATH:-data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_official_executor_smoke/${METHOD}}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-14B}
MAX_TASKS=${MAX_TASKS:-1}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-40}

"${PYTHON_BIN}" scripts/run_appworld_official_executor_eval.py \
  --method "${METHOD}" \
  --tasks_path "${TASKS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --appworld_root "${APPWORLD_ROOT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_tasks "${MAX_TASKS}" \
  --max_interactions "${MAX_INTERACTIONS}"
```

- [x] **Step 2: Validate shell**

Run:

```bash
bash -n scripts/sbatch/run_appworld_official_executor_eval.sh
```

Expected: no output, exit code 0.

- [ ] **Step 3: Submit single-task qwen-only smoke**

Run only after Qwen3-14B load smoke passes:

```bash
MAX_TASKS=1 METHOD=qwen_only OUTPUT_DIR=outputs/appworld_official_executor_smoke/qwen14b_qwen_only_task1 sbatch scripts/sbatch/run_appworld_official_executor_eval.sh
```

Expected: job writes `report.json` and `runs.jsonl`; no full/dev57 job is submitted.

- [ ] **Step 4: Commit**

```bash
git add scripts/sbatch/run_appworld_official_executor_eval.sh
git commit -m "feat: add official appworld executor sbatch"
```

## Task 8: Add CLSTR Evidence Mode to Official CLI

**Files:**
- Modify: `scripts/run_appworld_official_executor_eval.py`
- Modify: `clstr/appworld_official_executor.py`
- Modify: `tests/test_appworld_official_executor_cli.py`

- [x] **Step 1: Write controller argument test**

Add test parsing:

```python
def test_cli_accepts_clstr_evidence_arguments():
    parser = cli.build_parser()
    args = parser.parse_args([
        "--method", "clstr_multistep",
        "--tasks_path", "tasks.jsonl",
        "--output_dir", "out",
        "--skill_pool_path", "data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl",
        "--base_skill_pool_path", "data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl",
        "--clstr_checkpoint_path", "outputs/checkpoints/model.pt",
        "--ranking_mode", "transition_blend",
        "--candidate_top_k", "350",
    ])
    assert args.method == "clstr_multistep"
    assert args.ranking_mode == "transition_blend"
    assert args.candidate_top_k == 350
```

- [x] **Step 2: Implement args and reuse existing controller builder**

Import `_build_controller` from `scripts.run_appworld_multistep_executor_eval` and build the same controller modes. Evidence builder should call controller `select()` and then use the existing `build_verified_skill_handoff()` or current safe metadata compressor to create `EvidenceResult`.

- [x] **Step 3: Run CLI tests**

Run:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_appworld_official_executor_eval.py clstr/appworld_official_executor.py tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py
git commit -m "feat: connect clstr evidence to official executor"
```

## Task 9: Gate Runs

**Files:**
- Modify: `description.md`
- Use: `scripts/sbatch/run_appworld_official_executor_eval.sh`

- [ ] **Step 1: Gate 0 single-task qwen-only**

Submit:

```bash
MAX_TASKS=1 METHOD=qwen_only OUTPUT_DIR=outputs/appworld_official_executor_smoke/qwen14b_gate0_single sbatch scripts/sbatch/run_appworld_official_executor_eval.sh
```

Expected: output starts within a few minutes after model load; if it hangs before the first model call for too long, cancel and inspect.

- [ ] **Step 2: Gate 1 dev10 qwen-only**

Only if Gate 0 passes:

```bash
MAX_TASKS=10 METHOD=qwen_only OUTPUT_DIR=outputs/appworld_official_executor_smoke/qwen14b_gate1_dev10 sbatch scripts/sbatch/run_appworld_official_executor_eval.sh
```

Expected: report shows `success_count`, `execution_failures`, `average_steps`, and enough trace to diagnose failures.

- [ ] **Step 3: Gate 2 dev10 CLSTR evidence**

Only if Gate 1 is healthy:

```bash
MAX_TASKS=10 METHOD=clstr_multistep OUTPUT_DIR=outputs/appworld_official_executor_smoke/qwen14b_gate2_clstr_dev10 sbatch scripts/sbatch/run_appworld_official_executor_eval.sh
```

Expected: CLSTR does not increase invalid API/preflight/execution failures and improves success or failure profile.

- [ ] **Step 4: Record results**

Append to `description.md`:

```markdown
### 2026-06-13 AppWorld official executor gate results

- Gate 0:
  - job id:
  - output dir:
  - model load status:
  - official evaluation reached:
  - blocker, if any:
- Gate 1:
  - job id:
  - output dir:
  - success_count / task_count:
  - execution_failures:
  - average_steps:
  - comparison to Qwen3-8B qwen-only anchor:
- Gate 2:
  - job id:
  - output dir:
  - success_count / task_count:
  - execution_failures:
  - CLSTR-vs-qwen-only delta:
- Decision:
  - proceed_to_stage4_smoke: true/false
  - reason:
```

- [ ] **Step 5: Commit results**

```bash
git add description.md
git commit -m "docs: record official appworld executor gates"
```

## Task 10: Stage4 Smoke Plan Only After Gate 2

**Files:**
- Modify only after Gate 2 passes:
  - `clstr/qwen_stage4_act_train.py`
  - `scripts/run_clstr_stage4_act_train.py`
  - `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
  - focused tests under `tests/`

- [ ] **Step 1: Stop if Gate 2 does not pass**

If CLSTR evidence does not beat or cleanly improve qwen-only dev10, do not implement Stage4 online training for AppWorld. Record the blocker in `description.md`.

- [ ] **Step 2: If Gate 2 passes, write a new focused Stage4 ACT implementation plan**

Create a separate plan file before changing Stage4 training code:

```text
docs/superpowers/plans/2026-06-13-appworld-official-stage4-act.md
```

Expected: separate plan includes reward definition, train split, checkpoint update cadence, and small smoke only.

## Verification Before Completion

Run these before claiming implementation is ready:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_official_executor.py tests/test_appworld_official_executor_cli.py -q
python -m py_compile clstr/appworld_official_executor.py scripts/run_appworld_official_executor_eval.py scripts/download_qwen3_14b.py
bash -n scripts/sbatch/run_appworld_official_executor_eval.sh scripts/sbatch/run_qwen3_14b_load_smoke.sh
git diff --check
```

Do not run full AppWorld jobs from this plan. Full jobs require explicit user approval after Gate 0-2 reports are reviewed.
