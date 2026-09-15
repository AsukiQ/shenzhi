# CLSTR SkillRouter Bridge 实现计划

Archived historical plan, superseded by the SkillsBench + SkillRouter eval-only + ALFWorld experiment skeleton.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `/root/autodl-tmp/clstr` 中独立实现 CLSTR 的训练、推理、bridge、验证与测试代码，并将 `/root/autodl-tmp/skillrouter` 保持为只读上游镜像。

**Architecture:** `clstr` 是唯一的训练与推理工程；`SkillRouter` 只提供上游代码参考、warm-start checkpoint 来源和静态评测格式对齐。核心实现分为数据层、编码器层、belief/heads、模型编排、loss、rollout、validator、train/infer 入口和 `bridges/skillrouter` 兼容层。

**Tech Stack:** Python 3.11, PyTorch, Transformers, PyYAML, pytest, Hugging Face compatible checkpoints

---

## 文件结构与职责

- `README.md`：项目说明、目录边界、最小运行方式。
- `pyproject.toml`：包安装、pytest 配置。
- `requirements.txt`：运行和测试依赖。
- `.gitignore`：缓存、checkpoint、pytest 输出。
- `scripts/bootstrap_skillrouter.sh`：克隆或更新 `/root/autodl-tmp/skillrouter`。
- `scripts/build_verified_pairs.py`：离线构造 verified pair 模板。
- `scripts/train_stage.sh`：按 stage 启动训练。
- `scripts/infer_static.sh`：导出 SkillRouter 兼容静态检索结果。
- `scripts/infer_agentic.sh`：运行 CLSTR rollout 推理。
- `clstr/__init__.py`：包版本、公开导出。
- `clstr/data.py`：dataclass、JSONL/GZ 读取、执行状态序列化、SkillsBench 数据角色、generic replay trajectory loading 和 SkillRouter eval-only mapping。
- `clstr/encoders.py`：`StateEncoder`、`CrossEncoder`、`SkillTable`。
- `clstr/belief.py`：`subspace_obs`、`TransitionPredictor`、`BeliefGate`。
- `clstr/heads.py`：`SkillHead`、`StopHead`、`TransHead`。
- `clstr/model.py`：`CLSTRConfig`、`CLSTRModel`、`policy_forward`、`step_update`。
- `clstr/losses.py`：`transition_loss`、`retrieval_loss_*`、`policy_loss`、`action_loss`、`total_loss`。
- `clstr/rollout.py`：环境协议、`rollout`、top-K、replay belief 恢复。
- `clstr/validator.py`：validator prompt、严格 JSON 解析、verified template 构造。
- `clstr/metrics.py`：任务成功率、next-skill、STOP、belief 和 raw top-K 指标。
- `clstr/train.py`：stage 0-3 训练入口、checkpoint、配置解析。
- `clstr/infer.py`：静态导出与 agentic 推理入口。
- `clstr/bridges/skillrouter/checkpoints.py`：上游权重 key 映射与 non-strict load 报告。
- `clstr/bridges/skillrouter/datasets.py`：SkillRouter eval 数据读取。
- `clstr/bridges/skillrouter/evaluation.py`：SkillRouter 兼容导出格式。
- `clstr/bridges/skillrouter/serialization.py`：skill metadata 序列化与缓存。
- `tests/*`：单测与 smoke test。
- `tests/fixtures/skillrouter/*`：SkillRouter 任务与技能样例。
- `configs/model/base.yaml`：模型结构默认配置。
- `configs/train/base.yaml`：优化器、精度、stage 默认配置。
- `configs/data/base.yaml`：数据路径与缓存路径。
- `configs/eval/base.yaml`：评测导出参数。

### 任务 1：基础工程、包安装与上游拉取脚本

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `clstr/__init__.py`
- Create: `scripts/bootstrap_skillrouter.sh`
- Test: `tests/test_package_smoke.py`

- [ ] **Step 1: 先写包导入与脚本存在性的失败测试**

```python
# tests/test_package_smoke.py
from pathlib import Path


def test_package_importable():
    import clstr

    assert clstr.__version__ == "0.1.0"
    assert "get_version" in clstr.__all__


def test_bootstrap_script_exists():
    script = Path("scripts/bootstrap_skillrouter.sh")
    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
```

- [ ] **Step 2: 运行测试，确认当前确实失败**

Run: `pytest tests/test_package_smoke.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError: No module named 'clstr'`

- [ ] **Step 3: 创建基础工程文件和上游拉取脚本**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "clstr"
version = "0.1.0"
description = "Closed-loop Latent Skill Transition Router"
requires-python = ">=3.11"
dependencies = [
  "torch",
  "transformers>=4.51.0",
  "numpy",
  "PyYAML",
  "tqdm",
  "sentencepiece",
  "safetensors",
]

[tool.setuptools.packages.find]
include = ["clstr*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```text
# requirements.txt
torch
transformers>=4.51.0
numpy
PyYAML
tqdm
sentencepiece
safetensors
pytest
```

```text
# .gitignore
__pycache__/
.pytest_cache/
.venv/
*.pyc
*.pyo
*.pt
*.bin
*.safetensors
outputs/
artifacts/
cache/
```

```python
# clstr/__init__.py
__version__ = "0.1.0"


def get_version() -> str:
    return __version__


__all__ = ["get_version"]
```

```bash
# scripts/bootstrap_skillrouter.sh
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${SKILLROUTER_REPO_URL:-https://github.com/zhengyanzhao1997/SkillRouter.git}"
TARGET_DIR="${SKILLROUTER_TARGET_DIR:-/root/autodl-tmp/skillrouter}"

if [ -d "${TARGET_DIR}/.git" ]; then
  git -C "${TARGET_DIR}" fetch --all --tags
  git -C "${TARGET_DIR}" pull --ff-only
else
  rm -rf "${TARGET_DIR}"
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" rev-parse --short HEAD
```

```markdown
# README.md

## CLSTR

该仓库独立实现 CLSTR。

- `clstr/`：训练、推理、rollout、validator、metrics 和 bridge
- `scripts/bootstrap_skillrouter.sh`：拉取只读上游 SkillRouter
- `docs/superpowers/specs/`：设计文档
- `docs/superpowers/plans/`：实现计划
```

- [ ] **Step 4: 再次运行失败测试，确认基础工程通过**

Run: `pytest tests/test_package_smoke.py -q`
Expected: PASS，输出 `2 passed`

- [ ] **Step 5: 提交基础工程**

```bash
git add README.md pyproject.toml requirements.txt .gitignore clstr/__init__.py scripts/bootstrap_skillrouter.sh tests/test_package_smoke.py
git commit -m "chore: bootstrap CLSTR package skeleton"
```

### 任务 2：数据模型、状态序列化与基础 IO

**Files:**
- Create: `clstr/data.py`
- Create: `tests/test_data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: 先写 dataclass 和状态序列化失败测试**

```python
# tests/test_data.py
from clstr.data import ExecutionState, Skill, Task, VerifiedPair, serialize_execution_state


def test_serialize_execution_state_contains_all_fields():
    state = ExecutionState(
        query="book a train",
        history=[("search_train", "found trains")],
        observation="found 3 results",
        artifact={"ticket": "draft"},
        error=None,
    )

    text = serialize_execution_state(state)
    assert "query:book a train" in text
    assert "history:search_train -> found trains" in text
    assert "observation:found 3 results" in text
    assert "artifact:{'ticket': 'draft'}" in text
    assert "error:none" in text


def test_verified_pair_defaults():
    pair = VerifiedPair(
        state_before=ExecutionState("q", [], "", {}, None),
        action_at_t=3,
        obs_at_t="ok",
        candidates_next=[3, 4, 5],
        a_next_plus=4,
    )
    assert pair.was_in_raw_topk is True
    assert pair.m_t_exact is None
```

- [ ] **Step 2: 运行测试，确认 `clstr.data` 尚不存在**

Run: `pytest tests/test_data.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 实现 dataclass、JSONL/GZ 读取与执行状态序列化**

```python
# clstr/data.py
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor


@dataclass
class Skill:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    executor_desc: str
    failure_modes: list[str]
    skill_id: str | None = None


@dataclass
class Task:
    task_id: str | int
    query: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    query: str
    history: list
    observation: str
    artifact: dict[str, Any]
    error: str | None


@dataclass
class TrajectoryStep:
    x: ExecutionState
    m: Tensor
    candidates: list[int]
    action_local: int
    skill_idx: int | None
    log_prob: Tensor
    obs: str
    m_hat: Tensor | None
    m_tilde_next: Tensor | None


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    reward: float
    task_id: str | int


@dataclass
class ReplayStep:
    x: ExecutionState
    skill_idx: int
    obs: str


@dataclass
class VerifiedPair:
    state_before: ExecutionState
    action_at_t: int
    obs_at_t: str
    candidates_next: list[int]
    a_next_plus: int
    was_in_raw_topk: bool = True
    m_t_exact: Tensor | None = None
    replay_prefix: list[ReplayStep] | None = None


@dataclass
class RetrievalPositive:
    state: ExecutionState
    positive_skill_idx: int


def serialize_execution_state(state: ExecutionState) -> str:
    history_bits = [f"{name} -> {summary}" for name, summary in state.history]
    history_text = " | ".join(history_bits) if history_bits else "empty"
    error_text = state.error if state.error is not None else "none"
    return "\n".join(
        [
            f"query:{state.query}",
            f"history:{history_text}",
            f"observation:{state.observation}",
            f"artifact:{state.artifact}",
            f"error:{error_text}",
        ]
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
```

- [ ] **Step 4: 运行数据层测试**

Run: `pytest tests/test_data.py -q`
Expected: PASS，输出 `2 passed`

- [ ] **Step 5: 提交数据层**

```bash
git add clstr/data.py tests/test_data.py
git commit -m "feat: add CLSTR data contracts"
```

### 任务 3：SkillRouter bridge 的序列化、数据读取与 checkpoint 映射

**Files:**
- Create: `clstr/bridges/skillrouter/__init__.py`
- Create: `clstr/bridges/skillrouter/serialization.py`
- Create: `clstr/bridges/skillrouter/datasets.py`
- Create: `clstr/bridges/skillrouter/evaluation.py`
- Create: `clstr/bridges/skillrouter/checkpoints.py`
- Create: `tests/fixtures/skillrouter/tasks.jsonl`
- Create: `tests/fixtures/skillrouter/easy.jsonl`
- Create: `tests/test_bridge_skillrouter.py`
- Test: `tests/test_bridge_skillrouter.py`

- [ ] **Step 1: 先写 bridge 失败测试和最小 fixture**

```json
{"task_id":"task-1","instruction_text":"find a weather skill"}
```

```json
{"skill_id":"skill-1","name":"weather.lookup","description":"lookup weather","input_schema":{"city":"str"},"output_schema":{"temp":"float"},"executor_desc":"calls weather API","failure_modes":["timeout"],"body":"def run(city): return city"}
```

```python
# tests/test_bridge_skillrouter.py
from pathlib import Path

import torch

from clstr.bridges.skillrouter.checkpoints import map_backbone_state_dict
from clstr.bridges.skillrouter.datasets import load_eval_pool, load_eval_tasks
from clstr.bridges.skillrouter.evaluation import write_retrieval_predictions
from clstr.bridges.skillrouter.serialization import serialize_skill_text


def test_serialize_skill_text_contains_body_and_schema():
    skill = {
        "name": "weather.lookup",
        "description": "lookup weather",
        "input_schema": {"city": "str"},
        "output_schema": {"temp": "float"},
        "executor_desc": "calls weather API",
        "failure_modes": ["timeout"],
        "body": "def run(city): return city",
    }
    text = serialize_skill_text(skill)
    assert "name:weather.lookup" in text
    assert "body:def run(city): return city" in text


def test_load_eval_files_and_write_predictions(tmp_path: Path):
    tasks = load_eval_tasks(Path("tests/fixtures/skillrouter/tasks.jsonl"))
    pool = load_eval_pool(Path("tests/fixtures/skillrouter/easy.jsonl"))
    output = tmp_path / "retrieval.json"
    write_retrieval_predictions({tasks[0].task_id: [pool[0].skill_id]}, output)
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("{")


def test_map_backbone_state_dict_strips_known_prefixes():
    state_dict = {
        "model.encoder.layer.weight": torch.ones(1),
        "model.encoder.layer.bias": torch.zeros(1),
        "reranker.head.weight": torch.ones(1),
    }
    mapped = map_backbone_state_dict(state_dict, prefixes=("model.",))
    assert "encoder.layer.weight" in mapped
    assert "encoder.layer.bias" in mapped
```

- [ ] **Step 2: 运行 bridge 测试，确认模块不存在而失败**

Run: `pytest tests/test_bridge_skillrouter.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError: No module named 'clstr.bridges'`

- [ ] **Step 3: 实现 SkillRouter bridge**

```python
# clstr/bridges/skillrouter/__init__.py
from clstr.bridges.skillrouter.serialization import serialize_skill_text

__all__ = ["serialize_skill_text"]
```

```python
# clstr/bridges/skillrouter/serialization.py
from __future__ import annotations

from typing import Any


def serialize_skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"name:{skill.get('name', '')}",
            f"desc:{skill.get('description', '')}",
            f"in:{skill.get('input_schema', {})}",
            f"out:{skill.get('output_schema', {})}",
            f"exec:{skill.get('executor_desc', '')}",
            f"fail:{skill.get('failure_modes', [])}",
            f"body:{skill.get('body', '')}",
        ]
    )
```

```python
# clstr/bridges/skillrouter/datasets.py
from __future__ import annotations

from pathlib import Path

from clstr.data import Skill, Task, read_jsonl


def load_eval_tasks(path: Path) -> list[Task]:
    rows = read_jsonl(path)
    return [Task(task_id=row["task_id"], query=row.get("instruction_text", row.get("query", "")), meta=row) for row in rows]


def load_eval_pool(path: Path) -> list[Skill]:
    rows = read_jsonl(path)
    return [
        Skill(
            name=row["name"],
            description=row.get("description", ""),
            input_schema=row.get("input_schema", {}),
            output_schema=row.get("output_schema", {}),
            executor_desc=row.get("executor_desc", ""),
            failure_modes=row.get("failure_modes", []),
            skill_id=row.get("skill_id") or row.get("id"),
        )
        for row in rows
    ]
```

```python
# clstr/bridges/skillrouter/evaluation.py
from __future__ import annotations

import json
from pathlib import Path


def write_retrieval_predictions(predictions: dict[str | int, list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(key): value for key, value in predictions.items()}
    output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
```

```python
# clstr/bridges/skillrouter/checkpoints.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def map_backbone_state_dict(state_dict: dict[str, torch.Tensor], prefixes: Iterable[str]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                mapped[key[len(prefix):]] = value
                break
    return mapped


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj
```

- [ ] **Step 4: 运行 bridge 测试**

Run: `pytest tests/test_bridge_skillrouter.py -q`
Expected: PASS，输出 `3 passed`

- [ ] **Step 5: 提交 bridge 基础层**

```bash
git add clstr/bridges/skillrouter tests/fixtures/skillrouter tests/test_bridge_skillrouter.py
git commit -m "feat: add SkillRouter bridge primitives"
```

### 任务 4：编码器、池化逻辑与技能嵌入表

**Files:**
- Create: `clstr/encoders.py`
- Create: `tests/test_encoders.py`
- Test: `tests/test_encoders.py`

- [ ] **Step 1: 先写编码器失败测试**

```python
# tests/test_encoders.py
import torch
from types import SimpleNamespace

from clstr.encoders import SkillTable, pool_hidden


def test_masked_mean_pooling_ignores_padding():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_hidden(hidden, mask, "masked_mean")
    assert torch.allclose(pooled, torch.tensor([[2.0, 0.0]]))


def test_skill_table_rebuild_embeddings():
    encoder = lambda texts: torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    skills = [
        SimpleNamespace(name="a", description="b", input_schema={}, output_schema={}, executor_desc="", failure_modes=[]),
        SimpleNamespace(name="c", description="d", input_schema={}, output_schema={}, executor_desc="", failure_modes=[]),
    ]
    table = SkillTable(skills=skills, encoder_fn=encoder, d=2, trainable=False)
    assert table.E.shape == (2, 2)
```

- [ ] **Step 2: 运行测试，确认编码器模块缺失**

Run: `pytest tests/test_encoders.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 实现池化、状态编码器和技能表**

```python
# clstr/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from clstr.bridges.skillrouter.serialization import serialize_skill_text


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "masked_mean":
        mask_f = mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    if pooling == "last_token":
        seq_lens = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]
    if pooling == "cls":
        return hidden[:, 0]
    raise ValueError(pooling)


class StateEncoder(nn.Module):
    def __init__(self, base_model_name: str, d: int, pooling: str = "masked_mean"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.proj = nn.Linear(self.backbone.config.hidden_size, d)
        self.pooling = pooling

    def forward(self, text_batch: list[str]) -> torch.Tensor:
        tok = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(self.backbone.device) for k, v in tok.items()}
        out = self.backbone(**tok)
        pooled = pool_hidden(out.last_hidden_state, tok["attention_mask"], self.pooling)
        return self.proj(pooled)


class CrossEncoder(StateEncoder):
    pass


class SkillTable(nn.Module):
    def __init__(self, skills, encoder_fn: Callable[[list[str]], torch.Tensor], d: int, trainable: bool = False):
        super().__init__()
        self.skills = list(skills)
        self.encoder_fn = encoder_fn
        self.W = nn.Linear(d, d, bias=False)
        embeds = self.rebuild_embeddings()
        self.E = nn.Parameter(embeds, requires_grad=trainable)

    def rebuild_embeddings(self) -> torch.Tensor:
        texts = [serialize_skill_text(skill.__dict__) for skill in self.skills]
        return self.encoder_fn(texts)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.W(h) @ self.E.t()
```

- [ ] **Step 4: 运行编码器测试**

Run: `pytest tests/test_encoders.py -q`
Expected: PASS，输出 `2 passed`

- [ ] **Step 5: 提交编码器层**

```bash
git add clstr/encoders.py tests/test_encoders.py
git commit -m "feat: add encoders and skill table"
```

### 任务 5：belief 更新与动作头

**Files:**
- Create: `clstr/belief.py`
- Create: `clstr/heads.py`
- Create: `tests/test_belief.py`
- Create: `tests/test_heads.py`
- Test: `tests/test_belief.py`
- Test: `tests/test_heads.py`

- [ ] **Step 1: 先写 belief 和 heads 的失败测试**

```python
# tests/test_belief.py
import torch

from clstr.belief import BeliefGate, TransitionPredictor, subspace_obs


def test_subspace_obs_shape():
    class DummyTable:
        def __init__(self):
            self.E = torch.eye(2)
        def logits(self, h):
            return h
    obs = subspace_obs(DummyTable(), torch.tensor([[2.0, 0.0]]), tau_s=1.0)
    assert obs.shape == (1, 2)


def test_bayes_scalar_gate_is_in_unit_interval():
    gate = BeliefGate(d=4, mode="bayes_scalar")
    gamma = gate(torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 4))
    assert gamma.shape == (1, 1)
    assert torch.all((gamma >= 0.0) & (gamma <= 1.0))
```

```python
# tests/test_heads.py
import torch

from clstr.heads import StopHead, TransHead


def test_stop_head_output_shape():
    head = StopHead(d=4)
    out = head(torch.ones(2, 4), torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_trans_head_output_shape():
    head = TransHead(d=4, use_sn=True)
    out = head(torch.ones(2, 4), torch.zeros(2, 4))
    assert out.shape == (2, 1)
```

- [ ] **Step 2: 运行 belief 和 heads 测试，确认失败**

Run: `pytest tests/test_belief.py tests/test_heads.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError`

- [ ] **Step 3: 实现 belief 与动作头**

```python
# clstr/belief.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def subspace_obs(skill_table, h: torch.Tensor, tau_s: float) -> torch.Tensor:
    logits = skill_table.logits(h) / tau_s
    probs = F.softmax(logits, dim=-1)
    return probs @ skill_table.E


class TransitionPredictor(nn.Module):
    def __init__(self, d: int, d_a: int, n_actions: int):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, d_a)
        self.obs_proj = nn.Linear(d, d)
        self.cell = nn.GRUCell(d_a + d, d)

    def forward(self, m_t: torch.Tensor, a_t: torch.Tensor, o_t_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([self.action_emb(a_t), self.obs_proj(o_t_emb)], dim=-1)
        return self.cell(inp, m_t)


class BeliefGate(nn.Module):
    def __init__(self, d: int, mode: str = "learned_elementwise"):
        super().__init__()
        self.mode = mode
        self.linear = nn.Linear(d * 3, d)
        self.log_sigma_t = nn.Parameter(torch.zeros(1))
        self.log_sigma_o = nn.Parameter(torch.zeros(1))

    def forward(self, m_hat: torch.Tensor, m_obs: torch.Tensor, obs_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "learned_elementwise":
            return torch.sigmoid(self.linear(torch.cat([m_hat, m_obs, obs_emb], dim=-1)))
        sigma_t2 = self.log_sigma_t.exp().pow(2)
        sigma_o2 = self.log_sigma_o.exp().pow(2)
        gamma = sigma_t2 / (sigma_t2 + sigma_o2)
        return gamma.view(1, 1).expand(m_hat.size(0), 1)
```

```python
# clstr/heads.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SkillHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, m, u * m], dim=-1))


class StopHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, m], dim=-1))


class TransHead(nn.Module):
    def __init__(self, d: int, use_sn: bool = True):
        super().__init__()
        linear_1 = nn.Linear(d * 2, d)
        linear_2 = nn.Linear(d, 1)
        if use_sn:
            linear_1 = spectral_norm(linear_1)
            linear_2 = spectral_norm(linear_2)
        self.net = nn.Sequential(linear_1, nn.GELU(), linear_2)

    def forward(self, m_hat: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([m_hat, candidate_emb], dim=-1))
```

- [ ] **Step 4: 运行 belief 与 heads 测试**

Run: `pytest tests/test_belief.py tests/test_heads.py -q`
Expected: PASS，输出 `4 passed`

- [ ] **Step 5: 提交 belief 与 heads**

```bash
git add clstr/belief.py clstr/heads.py tests/test_belief.py tests/test_heads.py
git commit -m "feat: add belief update and policy heads"
```

### 任务 6：模型编排与损失函数

**Files:**
- Create: `clstr/model.py`
- Create: `clstr/losses.py`
- Create: `tests/test_losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: 先写模型与损失的失败测试**

```python
# tests/test_losses.py
import torch
from dataclasses import dataclass

from clstr.losses import transition_loss


@dataclass
class DummyStep:
    m_hat: torch.Tensor | None
    m_tilde_next: torch.Tensor | None


@dataclass
class DummyTraj:
    steps: list
    reward: float
    task_id: str


def test_transition_loss_skips_stop_steps():
    traj = DummyTraj(
        steps=[
            DummyStep(m_hat=torch.tensor([1.0, 0.0]), m_tilde_next=torch.tensor([0.0, 1.0])),
            DummyStep(m_hat=None, m_tilde_next=None),
        ],
        reward=1.0,
        task_id="t1",
    )
    loss = transition_loss([traj])
    assert loss.ndim == 0
    assert loss.item() > 0.0
```

- [ ] **Step 2: 运行测试，确认损失模块尚不存在**

Run: `pytest tests/test_losses.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError`

- [ ] **Step 3: 实现 `CLSTRConfig`、`CLSTRModel` 和基础损失**

```python
# clstr/model.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from clstr.belief import BeliefGate, TransitionPredictor, subspace_obs
from clstr.encoders import CrossEncoder, SkillTable, StateEncoder
from clstr.heads import SkillHead, StopHead, TransHead


@dataclass
class CLSTRConfig:
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    d: int = 256
    d_a: int = 64
    top_k: int = 16
    tau_s: float = 1.0
    gate_mode: str = "learned_elementwise"
    use_sn: bool = True


class CLSTRModel(nn.Module):
    def __init__(self, config: CLSTRConfig, skills):
        super().__init__()
        self.config = config
        self.tau_s = config.tau_s
        self.K = config.top_k
        self.skills = list(skills)
        self.encoder = StateEncoder(config.base_model_name, config.d)
        self.cross_encoder = CrossEncoder(config.base_model_name, config.d)
        self.skill_table = SkillTable(self.skills, self.encoder, config.d, trainable=False)
        self.transition = TransitionPredictor(config.d, config.d_a, len(self.skills) + 1)
        self.gate = BeliefGate(config.d, mode=config.gate_mode)
        self.skill_head = SkillHead(config.d)
        self.stop_head = StopHead(config.d)
        self.trans_head = TransHead(config.d, use_sn=config.use_sn)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def rebuild_skill_table(self) -> None:
        with torch.no_grad():
            self.skill_table.E.copy_(self.skill_table.rebuild_embeddings())

    def step_update(self, m_t: torch.Tensor, a_t: torch.Tensor, o_t_emb: torch.Tensor, x_next_text: list[str]):
        m_hat = self.transition(m_t, a_t, o_t_emb)
        h_next = self.encoder(x_next_text)
        m_obs = subspace_obs(self.skill_table, h_next, self.tau_s)
        gamma = self.gate(m_hat, m_obs, o_t_emb)
        if gamma.shape[-1] == 1:
            m_next = gamma * m_obs + (1.0 - gamma) * m_hat
        else:
            m_next = gamma * m_obs + (1.0 - gamma) * m_hat
        return m_next, m_hat, m_obs
```

```python
# clstr/losses.py
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def transition_loss(trajectories) -> torch.Tensor:
    losses = []
    device = None
    for traj in trajectories:
        for step in traj.steps:
            if step.m_hat is None or step.m_tilde_next is None:
                continue
            device = step.m_hat.device
            losses.append(F.mse_loss(step.m_hat, step.m_tilde_next.detach()))
    if not losses:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.stack(losses).mean()


def policy_loss(trajectories) -> torch.Tensor:
    grouped = defaultdict(list)
    for traj in trajectories:
        grouped[traj.task_id].append(traj.reward)
    centered = {}
    for task_id, rewards in grouped.items():
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        centered[task_id] = (rewards_t - rewards_t.mean()) / rewards_t.std(unbiased=False).clamp(min=1e-8)
    losses = []
    counters = defaultdict(int)
    for traj in trajectories:
        adv = centered[traj.task_id][counters[traj.task_id]]
        counters[traj.task_id] += 1
        log_prob_sum = torch.stack([step.log_prob for step in traj.steps]).sum()
        losses.append(-(adv.to(log_prob_sum.device) * log_prob_sum))
    return torch.stack(losses).mean() if losses else torch.zeros(())
```

- [ ] **Step 4: 运行损失测试**

Run: `pytest tests/test_losses.py -q`
Expected: PASS，输出 `1 passed`

- [ ] **Step 5: 提交模型编排与损失**

```bash
git add clstr/model.py clstr/losses.py tests/test_losses.py
git commit -m "feat: add model orchestration and core losses"
```

### 任务 7：rollout、环境协议与指标

**Files:**
- Create: `clstr/rollout.py`
- Create: `clstr/metrics.py`
- Create: `tests/test_rollout.py`
- Test: `tests/test_rollout.py`

- [ ] **Step 1: 先写 rollout 的失败测试**

```python
# tests/test_rollout.py
import torch

from clstr.data import ExecutionState, Skill, Task
from clstr.rollout import MockEnv, topk_recall_fn


def test_topk_recall_returns_descending_indices():
    logits = torch.tensor([[0.1, 0.7, 0.3]])
    indices = topk_recall_fn(logits, k=2)
    assert indices == [1, 2]


def test_mock_env_reset_and_execute():
    env = MockEnv(
        responses={"weather.lookup": "sunny"},
        success_after=1,
    )
    skill = Skill("weather.lookup", "", {}, {}, "", [], "skill-1")
    task = Task(task_id="task-1", query="weather")
    state = env.reset(task.query)
    obs = env.execute(skill, state)
    assert obs == "sunny"
```

- [ ] **Step 2: 运行 rollout 测试，确认模块缺失**

Run: `pytest tests/test_rollout.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError`

- [ ] **Step 3: 实现环境协议、top-K 和指标**

```python
# clstr/rollout.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from clstr.data import ExecutionState


def topk_recall_fn(logits: torch.Tensor, k: int) -> list[int]:
    values, indices = torch.topk(logits.squeeze(0), k=min(k, logits.size(-1)))
    ordered = sorted(zip(values.tolist(), indices.tolist()), reverse=True)
    return [idx for _, idx in ordered]


@dataclass
class MockEnv:
    responses: dict[str, str]
    success_after: int = 1

    def reset(self, query: str) -> ExecutionState:
        return ExecutionState(query=query, history=[], observation="", artifact={}, error=None)

    def execute(self, skill, x: ExecutionState) -> str:
        return self.responses.get(skill.name, "tool unavailable")

    def update_state(self, x: ExecutionState, skill, obs: str) -> ExecutionState:
        history = list(x.history) + [(skill.name, obs)]
        return ExecutionState(query=x.query, history=history, observation=obs, artifact=x.artifact, error=None)

    def is_success(self, x: ExecutionState) -> bool:
        return len(x.history) >= self.success_after
```

```python
# clstr/metrics.py
from __future__ import annotations

import numpy as np


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(any(item in relevant_ids for item in ranked_ids[:k]))


def stop_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def belief_mse(values_a, values_b) -> float:
    return float(np.mean((np.asarray(values_a) - np.asarray(values_b)) ** 2))
```

- [ ] **Step 4: 运行 rollout 测试**

Run: `pytest tests/test_rollout.py -q`
Expected: PASS，输出 `2 passed`

- [ ] **Step 5: 提交 rollout 与指标**

```bash
git add clstr/rollout.py clstr/metrics.py tests/test_rollout.py
git commit -m "feat: add rollout helpers and metrics"
```

### 任务 8：validator 与 verified pair 模板生成

**Files:**
- Create: `clstr/validator.py`
- Create: `scripts/build_verified_pairs.py`
- Create: `tests/test_validator.py`
- Test: `tests/test_validator.py`

- [ ] **Step 1: 先写 validator 的失败测试**

```python
# tests/test_validator.py
from clstr.validator import parse_validator_output, render_validator_prompt


def test_render_validator_prompt_mentions_llm_based_validator():
    prompt = render_validator_prompt({"query": "find weather", "steps": [], "answer": "done"})
    assert "LLM-based validator" in prompt


def test_parse_validator_output_returns_indices():
    parsed = parse_validator_output('{"essential_steps":[1,3],"reason":"kept only required calls"}')
    assert parsed["essential_steps"] == [1, 3]
```

- [ ] **Step 2: 运行 validator 测试，确认失败**

Run: `pytest tests/test_validator.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError`

- [ ] **Step 3: 实现 validator 与脚本入口**

```python
# clstr/validator.py
from __future__ import annotations

import json


def render_validator_prompt(traj: dict) -> str:
    lines = ["You are an LLM-based validator for tool-use trajectories.", "", f"Task: {traj['query']}"]
    for idx, step in enumerate(traj.get("steps", []), start=1):
        lines.append(f"Step {idx}: Call {step['tool']}({step.get('args', {})}) -> {step['result']}")
    lines.append(f"Final answer: {traj.get('answer', '')}")
    lines.append('Output strict JSON: {"essential_steps": [1], "reason": "short"}')
    return "\n".join(lines)


def parse_validator_output(payload: str) -> dict:
    obj = json.loads(payload)
    if "essential_steps" not in obj or not isinstance(obj["essential_steps"], list):
        raise ValueError("essential_steps missing")
    return obj
```

```python
# scripts/build_verified_pairs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clstr.validator import parse_validator_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()

    rows = []
    for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    output_lines = []
    for row in rows:
        parsed = parse_validator_output(row["validator_output"])
        output_lines.append(json.dumps({"task_id": row["task_id"], "essential_steps": parsed["essential_steps"]}, ensure_ascii=False))

    Path(args.output_jsonl).write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 validator 测试**

Run: `pytest tests/test_validator.py -q`
Expected: PASS，输出 `2 passed`

- [ ] **Step 5: 提交 validator 层**

```bash
git add clstr/validator.py scripts/build_verified_pairs.py tests/test_validator.py
git commit -m "feat: add validator prompt and verified pair script"
```

### 任务 9：训练与推理入口、配置、shell 脚本和 smoke test

**Files:**
- Create: `clstr/train.py`
- Create: `clstr/infer.py`
- Create: `configs/model/base.yaml`
- Create: `configs/train/base.yaml`
- Create: `configs/data/base.yaml`
- Create: `configs/eval/base.yaml`
- Create: `scripts/train_stage.sh`
- Create: `scripts/infer_static.sh`
- Create: `scripts/infer_agentic.sh`
- Create: `tests/test_train_smoke.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: 先写训练入口与配置 smoke test**

```python
# tests/test_train_smoke.py
from pathlib import Path

import yaml

from clstr.train import load_yaml_config


def test_load_yaml_config():
    config = load_yaml_config(Path("configs/train/base.yaml"))
    assert config["precision"] == "bf16"
    assert config["stage"] == "stage0_warmstart"
```

- [ ] **Step 2: 运行 smoke test，确认 train/infer 与配置缺失**

Run: `pytest tests/test_train_smoke.py -q`
Expected: FAIL，错误包含 `ModuleNotFoundError` 或 `FileNotFoundError`

- [ ] **Step 3: 实现 train/infer 入口、配置与 shell 脚本**

```python
# clstr/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--data_config", required=True)
    args = parser.parse_args()

    train_cfg = load_yaml_config(Path(args.train_config))
    model_cfg = load_yaml_config(Path(args.model_config))
    data_cfg = load_yaml_config(Path(args.data_config))

    print({"train": train_cfg["stage"], "model_dim": model_cfg["d"], "skillrouter_eval_root": data_cfg["skillrouter_eval_root"]})


if __name__ == "__main__":
    main()
```

```python
# clstr/infer.py
from __future__ import annotations

import argparse

from clstr.data import Task


def infer_query(query: str, task_id: str | int = "infer") -> Task:
    return Task(task_id=task_id, query=query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    task = infer_query(args.query)
    print({"task_id": task.task_id, "query": task.query})


if __name__ == "__main__":
    main()
```

```yaml
# configs/model/base.yaml
base_model_name: sentence-transformers/all-MiniLM-L6-v2
d: 256
d_a: 64
top_k: 16
tau_s: 1.0
gate_mode: learned_elementwise
use_sn: true
```

```yaml
# configs/train/base.yaml
stage: stage0_warmstart
precision: bf16
learning_rate_backbone: 1.0e-5
learning_rate_heads: 1.0e-4
gradient_accumulation_steps: 4
gradient_checkpointing: true
max_steps: 1000
```

```yaml
# configs/data/base.yaml
skillrouter_eval_root: /root/autodl-tmp/clstr/data/skillrouter_eval_core
skillsbench_root: /root/autodl-tmp/skillsbench
alfworld_root: /root/autodl-tmp/alfworld
leakage_audit_dir: /root/autodl-tmp/clstr/outputs/leakage_audit
clean_router_data_root: /root/autodl-tmp/clstr/data/clean_router
training_data_source: mock
cache_root: /root/autodl-tmp/clstr/cache
```

```yaml
# configs/eval/base.yaml
top_k: 50
rerank_top_k: 20
output_dir: /root/autodl-tmp/clstr/outputs
```

```bash
# scripts/train_stage.sh
#!/usr/bin/env bash
set -euo pipefail
python -m clstr.train \
  --train_config configs/train/base.yaml \
  --model_config configs/model/base.yaml \
  --data_config configs/data/base.yaml
```

```bash
# scripts/infer_static.sh
#!/usr/bin/env bash
set -euo pipefail
python -m clstr.infer --query "${1:-weather lookup}"
```

```bash
# scripts/infer_agentic.sh
#!/usr/bin/env bash
set -euo pipefail
python -m clstr.infer --query "${1:-agentic weather lookup}"
```

- [ ] **Step 4: 运行 smoke test**

Run: `pytest tests/test_train_smoke.py -q`
Expected: PASS，输出 `1 passed`

- [ ] **Step 5: 提交入口与配置**

```bash
git add clstr/train.py clstr/infer.py configs scripts tests/test_train_smoke.py
git commit -m "feat: add train and infer entrypoints"
```

### 任务 10：集成检查、上游仓库拉取与最终验证

**Files:**
- Modify: `scripts/bootstrap_skillrouter.sh`
- Test: `tests/test_package_smoke.py`
- Test: `tests/test_data.py`
- Test: `tests/test_bridge_skillrouter.py`
- Test: `tests/test_encoders.py`
- Test: `tests/test_belief.py`
- Test: `tests/test_heads.py`
- Test: `tests/test_losses.py`
- Test: `tests/test_rollout.py`
- Test: `tests/test_validator.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: 执行上游拉取脚本，确认 `SkillRouter` 镜像到位**

Run: `bash scripts/bootstrap_skillrouter.sh`
Expected: 输出一个上游 commit short SHA，并在 `/root/autodl-tmp/skillrouter/.git` 下存在 git 仓库

- [ ] **Step 2: 运行全量单测**

Run: `pytest tests -q`
Expected: PASS，输出所有测试通过，没有 `ModuleNotFoundError`

- [ ] **Step 3: 运行训练入口 smoke 命令**

Run: `python -m clstr.train --train_config configs/train/base.yaml --model_config configs/model/base.yaml --data_config configs/data/base.yaml`
Expected: PASS，打印包含 `stage0_warmstart`、`256` 和 `/root/autodl-tmp/skillrouter`

- [ ] **Step 4: 运行推理入口 smoke 命令**

Run: `python -m clstr.infer --query "book a train"`
Expected: PASS，打印 `{"task_id": "infer", "query": "book a train"}` 风格结果

- [ ] **Step 5: 提交集成状态**

```bash
git add .
git commit -m "feat: scaffold first-pass CLSTR project"
```

## 自检清单

- 设计边界是否保持：`SkillRouter` 只读，`CLSTR` 独立训练。
- 是否避免使用 `oracle validator` 术语。
- 是否显式区分 `STOP_IDX = N` 和 rollout local stop index `K`。
- 是否把 raw top-K 指标和 injected candidate 指标拆开。
- 是否所有测试都可在无外网、无 GPU 环境下完成最小 smoke 验证。
- 是否所有 GPU-first 训练配置都已经进入 `configs/train/base.yaml` 和 `clstr/train.py` 的主路径。
