# CLSTR v2 实施指导：转向 Closed-Loop Skill Routing + AppWorld 主表

> **本文性质**：方向转向 + 实施路线图，不是 method spec 重写。
> **Method spec 仍以 markdown_fold 三份原文档为准**：
> - [Hybrid_Latent_Next_Skill_Router_Method_方法与实现指导版_…实验版.md](../../markdown_fold/Hybrid_Latent_Next_Skill_Router_Method_方法与实现指导版_含SkillRouter初始化_SkillsBench_SkillRouter_ALFWorld实验版.md)（method spec source of truth）
> - [Hybrid_Latent_Next_Skill_Router_Method_AAAI优化版_…实验版.md](../../markdown_fold/Hybrid_Latent_Next_Skill_Router_Method_AAAI优化版_含SkillRouter初始化_SkillsBench_SkillRouter_ALFWorld实验版.md)（AAAI 优化版叙事）
> - [CLSTR_SkillsBench_SkillRouter_ALFWorld_实验计划_无泄漏版.md](../../markdown_fold/CLSTR_SkillsBench_SkillRouter_ALFWorld_实验计划_无泄漏版.md)（无泄漏实验计划）
> **现状参考**：[description.md](../description.md)（2026-05-20 / 2026-05-21 实验记录）
> **配套问题清单**：[clstr_alfworld_issues_and_fixes.md](./clstr_alfworld_issues_and_fixes.md)（P0–P6 实现问题）

---

## 0. 转向缘起（One-Page Summary）

**为什么转向**：

1. ALFWorld 闭环 success_rate=0 不是单点 bug，是结构性不匹配 —— CLSTR 的"大 skill pool routing + closed-loop belief filter"假设在 ALFWorld 的"admissible_commands 实例化字符串"动作空间上失效（详见 [clstr_alfworld_issues_and_fixes.md](./clstr_alfworld_issues_and_fixes.md) P0–P6）。
2. paper 2605.05413v1（"From History to State: Constant-Context Skill Learning"）证明在 ALFWorld 上 LoRA SFT+RL 能到 89.6%，而其方法和 CLSTR 在抽象层级完全不同。CLSTR 不可能在 ALFWorld 上以"frozen Qwen + pooled embedding + small head"路径竞争该数字。
3. SkillsBench 是 CLSTR 设计的天然主表（`SKILL.md` 体系正是 CLSTR `serialize_skill_text` 的对照物），但存在两个硬阻塞：(a) docker-in-docker 权限问题导致本地跑不通；(b) **SkillRouter warm-start 与 SkillsBench eval 存在数据泄漏**，违反 markdown_fold §1.1 无泄漏约束。

**转向后做什么**：

- **论文 framing**：从"CLSTR 是更好的 agent"调整为"CLSTR 是更好的 closed-loop skill routing 层，与任意 LM agent backbone 解耦"。
- **主表 benchmark**：[AppWorld](https://github.com/StonyBrookNLP/appworld)（ACL 2024 Best Resource Paper, 457 APIs, 9 apps, DB-state deterministic verifier）。
- **Skill 池来源**：[SkillX](https://github.com/zjunlp/SkillX) 已经把 AppWorld trajectory 抽象成 `SKILL.md` 格式的 skill 库，直接复用，避免给 AppWorld API docstring 手写扩展 body 的合成数据风险。
- **Warm-start 路径**：放弃 SkillRouter checkpoint（泄漏），改用 SKILLRET (ThakiCloud, 16k skills) 或 clean synthetic router data；任一选择都要配 leakage audit。
- **保留**：CLSTR 方法核心（SkillTable / belief / TransHead / SkillHead / gate / retrieval_loss / action_loss）零改动。
- **辅助表**：SKILLRET 大池 retrieval scaling；ALFWorld 降级为 sequential decision transfer 辅助 case study。

---

## 1. 论文叙事 reposition

### 1.1 原叙事（markdown_fold 版）

> CLSTR 是 hybrid latent next-skill router，提供 closed-loop belief filtering + skill subspace transition 用于 skill-using agent。主表是 SkillsBench closed-loop。

### 1.2 新叙事

> CLSTR 是与 LM agent 解耦的 closed-loop skill routing 层。给定大 skill pool 和正在进行的 task trajectory，CLSTR 在每一步基于 belief 状态从 pool 中检索 top-K candidate skill 并 rank。CLSTR 不替代 LM agent 的 action 生成层，但显著提升其在 large-pool routing scenario 下的成功率和效率。

**关键区别**：

| 维度 | 原叙事 | 新叙事 |
|---|---|---|
| CLSTR 与 LM 关系 | CLSTR 替代或包含 LM agent | CLSTR 作为 routing 前置层，LM agent 仍负责 action 生成 |
| 主表评估 | 整 agent 闭环成功率 | 在固定 LM agent 下，加 CLSTR routing 层 vs 不加（消融对比） + retrieval metrics |
| Ablation 对手 | 不同 belief filter / 不同 head 设计 | 不同 routing 策略（dense retrieval / BM25 / SkillX original / CLSTR） |
| 论文卖点 | "更好的 agent" | "更好的 routing，独立于 backbone" |

### 1.3 不变的方法主张

CLSTR 论文以下方法主张**完全保留**，不需要重写 method spec：

- §3.1–§3.9 所有模块规约（StateEncoder / SkillTable / TransitionPredictor / BeliefGate / CrossEncoder / SkillHead / StopHead / TransHead）
- §4 闭环 belief 更新（step_update）
- §5 一步策略前向（policy_forward）
- §6 全部 loss（policy / transition / action / retrieval / KL）
- §7 trajectory rollout 流程
- §10.2–§10.3 LLM-based validator（**改名继续用**：避免"oracle"措辞）
- §11 关键超参数初值
- §12 全部实现陷阱（尤其 #1 STOP local/global、#3 masked pooling、#10 m_t 来源优先级）

---

## 2. 主表 benchmark 选型：AppWorld + SkillX skill 库

### 2.1 为什么选 AppWorld

按四项硬约束打分（详见前期对话）：

| 论点 | AppWorld | SkillsBench | ToolBench | SKILLRET |
|---|---|---|---|---|
| skill body 丰富（CLSTR `serialize_skill_text` 7 字段假设） | ✓ SKILL.md（via SkillX） | ✓ 原生 SKILL.md | △ 短描述 | △ 中等 |
| 大 skill pool routing | 457 + SkillX 抽象层 | 249 全局化 | 16k | 16k |
| closed-loop belief filter（单 episode 多步 + observation） | ✓ 典型 10+ 步跨 app | △ per-task 2-3 skill | ✓ | ✗ static |
| skill transition prior（跨 app 调用序列） | ✓ Venmo→Gmail→FileSystem 这种真实链 | ✗ task 内孤立 | △ | ✗ |
| deterministic verifier | ✓ DB state diff | ✓ docker DB diff | ✗ LLM judge | retrieval only |
| 是否泄漏 SkillRouter warm-start | ✗ 新 benchmark, 无关 | ✓ 有泄漏 | ✗ | 需 audit |

AppWorld 四项满分，且**不与现有 SkillRouter checkpoint 存在数据泄漏关系**。

### 2.2 Skill 池构造：SkillX as adapter

直接对接 AppWorld 457 个 API 作为 skill 池存在两个问题：

1. **Skill body 缺失**：AppWorld API 是 Python function + docstring，只有几行；CLSTR `serialize_skill_text` 期望几 KB markdown body。
2. **粒度过细**：457 个 atomic API 对 routing 而言粒度太低；很多 API 互为参数组合，路由层应该看到的是更高层的 "send_email_with_attachment" 这种复合 skill。

**SkillX 解决了这两件事**：

- SkillX 已经对 AppWorld 跑了 "trajectory → skill abstraction" pipeline，输出 SKILL.md 格式的高层 skill 库。
- 每个 skill 包含 description + workflow + failure modes + 可执行 scripts —— 正是 CLSTR `serialize_skill_text` 的 7 字段所需。

**实施步骤**：

1. Clone [SkillX repo](https://github.com/zjunlp/SkillX)，确认其 AppWorld skill 库规模和 schema。
2. 写 `clstr/bridges/skillx/appworld_adapter.py`，把 SkillX skill JSON → CLSTR `Skill` dataclass（[clstr/data.py:18](../clstr/data.py#L18)）。
3. 写 `scripts/build_appworld_skill_pool.py`，从 SkillX 输出生成 `data/appworld_skill_pool/skill_pool.jsonl`，schema 与 [skillsbench_skill_pool/skill_pool.jsonl](../data/skillsbench_skill_pool/skill_pool.jsonl) 对齐。
4. **验证**：随机抽 20 个 skill，确认 `name / description / body / failure_modes` 都非空；用 [bridges/skillrouter/serialization.py](../clstr/bridges/skillrouter/serialization.py) 序列化后 token 数 ≥ 200（这是 SKILL.md-rich 的下限）。

**Fallback**：若 SkillX skill 库太小（< 100）或 schema 不匹配，退化方案是用 GPT-5 给 AppWorld atomic API 生成扩展 body —— 但要在 paper 中明确标注 synthetic body，并做 ablation（with synthetic body vs without）证明 CLSTR 优势不来自合成数据。

### 2.3 AppWorld closed-loop 接入

- AppWorld 有官方 Python harness `appworld.environment`，每步返回 observation 文本（API 调用结果）+ done flag + reward。
- 写 `clstr/envs/appworld_env.py` 实现 `ExecutionEnv` 接口（markdown_fold §10.4 已规定的最小接口）。
- 写 `clstr/appworld_eval.py` 复刻 `alfworld_eval.py` 结构，但**避开 ALFWorld 出现的两个问题**：
  - 不要硬编码 verb 前缀做 skill_id 映射（[alfworld_eval.py:351](../clstr/alfworld_eval.py#L351) 那种 hack）；用 SkillX skill 库作为统一 vocabulary。
  - controller `transition_scores / belief_scores` 用 `trans_head(m_hat, action_emb)` 和 `skill_head(candidate_embs, m_next)` 真实 head 输出，**不再用 cos-to-goal 自创公式**（[clstr_alfworld_issues_and_fixes.md P2](./clstr_alfworld_issues_and_fixes.md#L41) 已记录该坑）。

---

## 3. Warm-start 重做（解决泄漏）

### 3.1 泄漏诊断

markdown_fold 实验计划 §1.1 / §7.1 / §7.2 明确：

> No evaluation query, relevance label, or ground-truth skill mapping from SkillRouter benchmark is used for CLSTR training or router warm-start.

当前 CLSTR 用现成 SkillRouter-style checkpoint warm-start，而该 checkpoint 训练数据**与 SkillsBench 评测集存在重叠**（用户回忆，待 audit 量化）。这是 **train-on-test contamination**，在新主表（AppWorld）上虽然没有直接评测污染（AppWorld 不在 SkillRouter 训练集中），但仍然要在论文里诚实声明 warm-start 数据来源，否则审稿人会问。

### 3.2 三条 warm-start 重做路径

| 路径 | 数据 | 干净度 | 工作量 | 推荐度 |
|---|---|---|---|---|
| **A. 不做 warm-start，从零初始化** | – | ✓ 完全干净 | 风险高（markdown_fold §6 警告从零训不稳定） | △ |
| **B. clean synthetic router data warm-start** | 待生成；[data/clean_router/manifest.json](../data/clean_router/manifest.json) 当前为空 | ✓ 干净（合成 query / 排除所有 SkillsBench + AppWorld + SKILLRET 出现的 skill_id） | 中：需要构造 + audit | ✓✓ **推荐** |
| **C. SKILLRET (ThakiCloud) warm-start** | [data/skillret/](../data/skillret/) 已 imported（16,783 skills / 68,256 queries） | △ 需 audit | 低：数据已就绪 | ✓ 如果 audit 通过 |

### 3.3 Leakage audit checklist（无论走哪条路径都必须做）

写 `scripts/run_warm_start_leakage_audit.py`，对 warm-start corpus 和 eval set 做以下检查并写入 `outputs/warm_start_leakage_audit/report.json`：

1. **Skill ID 交集**：warm-start corpus 中出现的 skill_id 与 AppWorld / SkillX skill 池的精确交集；要求交集 = 0。
2. **Query 文本近似交集**：用 MinHash / SimHash 对 warm-start query 和 AppWorld task instruction 做 near-dup 检测；jaccard ≥ 0.8 的 pair 视为重叠，要求 ≤ 5 条且 paper 中列出。
3. **Skill body 近似交集**：warm-start skill 描述和 SkillX skill body 的近似交集；同上阈值。
4. **Task ID 交集**：如果 warm-start 数据带 task_id，与 AppWorld task_id 取交集；要求 = 0。
5. **生成器同源性**：若 warm-start corpus 是 GPT 生成（如 SKILLRET 用 Qwen3.5 生成），声明 generator model；要求 generator ≠ AppWorld task author 模型。

Audit 通过的标准：上述 5 项全部 PASS，并在 paper appendix 列表格。

### 3.4 推荐方案：B（clean synthetic）+ C（SKILLRET）并行

- **B 作为主 warm-start**：用 GPT-5 / Claude 生成 50k 条合成 (query, skill_id) pair，skill_id 取自一个**专门的合成 skill 池**（不与 AppWorld / SkillsBench / SKILLRET 重叠的 skill 集合）。
- **C 作为 ablation**：报告"warm-start = SKILLRET" vs "warm-start = clean synthetic" vs "no warm-start"三栏，证明 warm-start 选择对最终 AppWorld 主表性能不敏感（即 CLSTR 增益不来自 warm-start corpus）。

---

## 4. 训练 protocol 重构

### 4.1 三阶段流水线（沿用 markdown_fold §6.6 但 benchmark 换 AppWorld）

```
Stage 0  Leakage audit                 (scripts/run_warm_start_leakage_audit.py)
Stage 1  Warm-start retrieval backbone  (clean synthetic 或 SKILLRET)
Stage 2  CLSTR-base                     (frozen backbone, 训 heads)
         数据：AppWorld train split 上 rollout 出的 success trajectories
         loss：policy + transition + retrieval (weak positives, §6.4)
Stage 3  CLSTR-act                      (加 verified pairs)
         数据：LLM-based validator 标注的 essential subsequences
         loss：policy + transition + retrieval (strong) + action (§6.3)
Stage 4  Optional joint fine-tune       (小学习率联合微调 backbone + heads)
```

### 4.2 关键修复（继承 [clstr_alfworld_issues_and_fixes.md](./clstr_alfworld_issues_and_fixes.md)）

| Issue | 修复在 AppWorld pipeline 中怎么落 |
|---|---|
| P0 — L_policy 退化为 supervised CE | AppWorld pipeline 一开始就用 `losses.policy_loss`（GRPO outcome）+ `losses.reference_kl_loss`，不沿用 `_compute_full_base_loss` 那条 supervised CE 路径 |
| P1 — L_act 未接入 | AppWorld pipeline 必须用 `losses.action_loss(allow_approximate_m_t=False)`，verified pair 来自 LLM-validated essential subsequences |
| P2 — 推理 transition/belief 用 cos-to-goal | AppWorld controller 用 `trans_head` 和 `skill_head` 的真实 logits |
| P4 — 训练/推理 vocabulary 不一致 | SkillX skill 库就是统一 vocabulary，训练和推理都用同一份 |
| P5 — STOP 实质失效 | STOP 作为虚拟动作加入候选集合，argmax 在 K+1 上做 |
| P6 — Penalty hardcoded | penalty 权重归零或显著降低，让 head signal 主导 |

### 4.3 不再使用的代码路径

下列代码路径**在 AppWorld pipeline 中不接入**（保留供 ALFWorld 辅助实验用，主表不依赖）：

- [clstr/full_base_train.py](../clstr/full_base_train.py) 中 `_compute_full_base_loss` 的 supervised CE `L_policy`（被 §4.2 P0 替代）
- [clstr/alfworld_eval.py:486-494](../clstr/alfworld_eval.py#L486-L494) 的 cos-to-goal transition/belief 公式
- [clstr/closed_loop_controller.py:18-39](../clstr/closed_loop_controller.py#L18-L39) 的 hardcoded penalty 矩阵（保留接口但权重 = 0）
- [clstr/qwen_external_encoder.py](../clstr/qwen_external_encoder.py) frozen 8B encoder 作为主路径（AppWorld 主表用较小 fine-tunable encoder；Qwen3-8B external 留作 ablation）

---

## 5. 评估 protocol

### 5.1 主表：AppWorld closed-loop task success

按 AppWorld 官方 evaluation：

- splits: `dev / test_normal / test_challenge`
- 指标：Task Success Rate（DB state verifier）、Average API Calls、Average Steps
- decoding：3 random seeds，mean ± std

### 5.2 主表对比组（最关键的 ablation）

CLSTR 的卖点是 routing layer，所以对比组必须把 LM agent backbone **固定**，只换 routing 策略：

| Method | Routing | LM backbone | Action 生成 |
|---|---|---|---|
| Baseline 1: LM-only | LM 自己看完整 skill 列表 | Qwen3-8B / GPT-5 | LM |
| Baseline 2: Dense retrieval + LM | BM25 / sentence-transformer | Qwen3-8B / GPT-5 | LM |
| Baseline 3: SkillX original routing | SkillX 自己的 routing | Qwen3-8B / GPT-5 | LM |
| **CLSTR-base + LM** | CLSTR base | Qwen3-8B / GPT-5 | LM |
| **CLSTR-act + LM** | CLSTR act（verified pairs） | Qwen3-8B / GPT-5 | LM |

CLSTR 的论文主张是 "CLSTR routing 优于 dense retrieval 和 SkillX 原始 routing"，**不是** "CLSTR agent 优于 LM agent"。后者打不过 paper 2605.05413 的 89.6%，前者是干净的 routing-layer 对比，赢面大。

### 5.3 辅助表

| 辅助表 | benchmark | 角色 |
|---|---|---|
| 大池 retrieval scaling | SKILLRET (16k) + SkillRouter eval core (78k) | Hit@1 / MRR@10 / NDCG，证明 retriever 能上到 16k 级别 |
| Sequential decision transfer | ALFWorld (在 P0–P6 修复后) | belief filter 对 long-horizon transition 的迁移价值 case study；不与主表比性能 |
| Closed-loop transfer to new domain | τ²-Bench / BFCL-v3 | zero-shot adapter 转移，证明 CLSTR routing 不绑定 AppWorld 训练分布 |

### 5.4 消融（沿用 markdown_fold §6 + 实验计划 §6）

- without warm-start
- without belief state
- projection-only latent（去 transition）
- no action loss
- no reference KL
- without spectral norm
- pointwise vs listwise rerank
- with synthetic skill body vs SkillX original SKILL.md（如果走 fallback）

---

## 6. 保留 vs 弃用代码路径速查表

| 路径 | 主表用？ | 辅助/ablation 用？ | 备注 |
|---|---|---|---|
| [clstr/model.py](../clstr/model.py) CLSTRModel | ✓ | ✓ | 零改动 |
| [clstr/encoders.py](../clstr/encoders.py) StateEncoder / SkillTable / CrossEncoder | ✓ | ✓ | 零改动 |
| [clstr/belief.py](../clstr/belief.py) subspace_obs / TransitionPredictor / BeliefGate | ✓ | ✓ | 零改动 |
| [clstr/heads.py](../clstr/heads.py) SkillHead / StopHead / TransHead | ✓ | ✓ | 零改动 |
| [clstr/losses.py](../clstr/losses.py) policy / transition / action / retrieval | ✓ | ✓ | 零改动；现在 AppWorld 主表真正用上 `policy_loss` 和 `action_loss` |
| [clstr/online_hrpo.py](../clstr/online_hrpo.py) | ✓ | ✓ | 沿用；shaped reward 沿用 paper 2605.05413 §3.3 设计（基于 deterministic tracker fields） |
| [clstr/full_base_train.py](../clstr/full_base_train.py) `_compute_full_base_loss` | ✗ supervised CE 路径 deprecate | ✓ ALFWorld 辅助保留 | 新增 `_compute_appworld_outcome_loss` 走 GRPO outcome |
| [clstr/alfworld_eval.py](../clstr/alfworld_eval.py) `make_controller_component_scorer` cos-to-goal 公式 | ✗ | △ 仅作旧实现对照 | 新 `appworld_eval.py` 重写 |
| [clstr/qwen_external_encoder.py](../clstr/qwen_external_encoder.py) | ✗ 主表用较小 encoder | ✓ ablation | 不作为主路径，避免 ALFWorld 同样问题 |
| [clstr/closed_loop_controller.py](../clstr/closed_loop_controller.py) hardcoded penalty | △ 权重 = 0 | ✓ 作 controller calibration ablation | 接口保留 |
| `clstr/bridges/skillrouter/` | △ 仅作 warm-start 备选 | ✓ ablation | 主路径换为 clean synthetic warm-start |
| **新增：** `clstr/bridges/skillx/` | ✓ 主表 skill 库适配 | ✓ | 本 pivot 引入 |
| **新增：** `clstr/envs/appworld_env.py` | ✓ | – | 主表 env |
| **新增：** `clstr/appworld_eval.py` | ✓ | – | 主表 eval 入口 |
| **新增：** `scripts/run_warm_start_leakage_audit.py` | ✓ Stage 0 必跑 | – | leakage audit |
| **新增：** `scripts/build_appworld_skill_pool.py` | ✓ | – | 从 SkillX 输出构建 skill 池 |

---

## 7. 风险与 fallback

| 风险 | 触发条件 | Fallback |
|---|---|---|
| SkillX AppWorld skill 库太小（< 100）或 schema 不匹配 | clone + audit 后发现 | (a) 用 GPT-5 扩展 AppWorld atomic API 的 body 字段，明标 synthetic；(b) 退回 SkillsBench 主表 + 重做 warm-start (§3.4 路径 B) |
| AppWorld harness 本地跑不通 | `pip install appworld + appworld init` 失败 | (a) AppWorld 提供 docker image，换允许 docker 的机器；(b) 退回 SkillsBench |
| Clean synthetic warm-start corpus 构造成本过高 | 50k 条 GPT 生成开销不可接受 | 退回 SKILLRET warm-start，加严 audit 并在 paper 公开 leakage report |
| AppWorld 上 CLSTR 增益不显著 | CLSTR + LM ≈ dense retrieval + LM | (a) 切换到 τ²-Bench 作为主表（同样有 deterministic verifier，skill 池由 SkillX 构造）；(b) 转 routing-only paper，主表只报 Hit@K / MRR，闭环作 case study |
| 论文叙事被 reviewer 质疑"为什么不直接 LoRA + tracker" | reviewer 引用 paper 2605.05413 | 在 related work 节明确：CLSTR routing 与 LoRA-tuned agent backbone **正交**，可同时使用；做 "CLSTR + LoRA agent" 的组合实验证明 CLSTR 的增量贡献 |

---

## 8. 实施路线图（4 周）

### Week 1 — Pivot foundation

- [ ] 在 `clstr/docs/` 加本文档作为方向锁定（done by this commit）
- [ ] Clone [SkillX repo](https://github.com/zjunlp/SkillX)，导出其 AppWorld skill 库 schema，写入 `outputs/skillx_audit/schema.json`
- [ ] 实测 SkillX skill 库规模、token 数分布、字段覆盖率
- [ ] 决策点 1：SkillX 是否可直接用？不行则启动 fallback A（GPT 扩 body）
- [ ] 安装 AppWorld harness，跑通官方 example
- [ ] 写 `clstr/envs/appworld_env.py` 最小可工作版本（reset / execute / is_success）

### Week 2 — Data + warm-start

- [ ] 写 `scripts/build_appworld_skill_pool.py`，产出 `data/appworld_skill_pool/skill_pool.jsonl`
- [ ] 写 `scripts/build_clean_router_data.py`（如果选路径 B）或 audit SKILLRET（如果选路径 C）
- [ ] 写 `scripts/run_warm_start_leakage_audit.py`，跑 audit
- [ ] 决策点 2：audit 通过？不过则 (a) 调整 warm-start corpus；(b) 接受残留泄漏并在 paper 列表
- [ ] 用选定 warm-start corpus 训共享编码器 backbone

### Week 3 — CLSTR-base 接 AppWorld

- [ ] 写 `clstr/appworld_eval.py`（重写 controller scorer，避开 ALFWorld 5 个坑）
- [ ] 跑 CLSTR-base：rollout AppWorld train split，训 heads，eval dev split
- [ ] 决策点 3：dev set success_rate ≥ dense retrieval baseline？不达标则诊断（参照 P0–P6 的 root cause 分析方法）

### Week 4 — CLSTR-act + 主表

- [ ] LLM-based validator 标注 essential subsequences
- [ ] 跑 CLSTR-act，加 action_loss + strong retrieval positives
- [ ] AppWorld test_normal / test_challenge 上跑主表
- [ ] 跑全部 §5.2 对比组（LM-only / dense / SkillX / CLSTR-base / CLSTR-act）
- [ ] 跑核心 ablation：without warm-start / without belief / projection-only

每个 Week 结束都更新 [description.md](../description.md) 写入当周 blocker、决策点结果、数据快照路径。

---

## 9. 引用与外部依赖

- AppWorld: Trivedi et al., *AppWorld: A Controllable World of Apps and People for Benchmarking Function Calling and Interactive Coding Agent*, ACL 2024 Best Resource Paper. [GitHub](https://github.com/StonyBrookNLP/appworld) / [website](https://appworld.dev/)
- SkillX (zjunlp): *SkillX: Automatically Constructing Skill Knowledge Bases for Agents*. [GitHub](https://github.com/zjunlp/SkillX)
- SkillNet (zjunlp): *Create, Evaluate, and Connect AI Skills*. [GitHub](https://github.com/zjunlp/SkillNet) / [arXiv](https://arxiv.org/pdf/2603.04448)
- Paper 2605.05413v1 (constant-context skill learning): Xie et al., *From History to State: Constant-Context Skill Learning for LLM Agents*, 2026. 用作"为什么 ALFWorld 上不和它正面竞争"的 framing 依据。
- Anthropic Agent Skills: [SKILL.md spec](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)。CLSTR `serialize_skill_text` 与该 spec 完全对齐。
- SKILLRET: [ThakiCloud/SKILLRET on HuggingFace](https://huggingface.co/datasets/ThakiCloud/SKILLRET)（已 imported 到 [data/skillret/](../data/skillret/)）
