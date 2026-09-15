# CLSTR 研究介绍

本文档介绍 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 中的 CLSTR 研究代码与当前实验状态。CLSTR 指 **Closed-loop Latent Skill Transition Routing**，当前定位不是端到端替代 LM agent，而是作为下游 LM/Qwen executor 前面的 **闭环技能路由层**：在任务执行过程中，根据当前任务、历史执行反馈和 latent belief state，动态选择更合适的 SkillX/AppWorld skill context。

截至 2026-05-24，项目已经打通 AppWorld + SkillX + Qwen3-8B 的单步和多步评测链路，也实现了多种 CLSTR recurrent belief / ACT / HRPO 训练与消融。但当前证据仍是阶段性诊断结果：CLSTR 已能在若干设置中追平 `skillrouter_base_live` 的 dev10 `7/10`，尚未稳定超过 SkillRouter，也尚未证明 recurrent `m_t` 在 AppWorld official task success 上有显著正贡献。

## 1. 研究问题

大规模 agent skill 库带来一个核心问题：当 skill pool 变大且技能之间高度重叠时，不能把所有 skill 都塞给下游 LM agent，必须先做 skill routing。传统静态路由通常只基于初始 query 检索 top-k skill，忽略执行过程中的环境反馈、错误信息、已完成步骤和任务阶段变化。

CLSTR 关注的问题是：

1. 在 AppWorld 这类多 API、多应用、长程任务中，skill routing 是否应建模为一个闭环过程，而不是一次性检索。
2. 一个维护在 skill embedding 子空间中的 recurrent belief state `m_t`，能否利用执行历史和 environment output 改善下一步 skill/context 选择。
3. 如果 task success 没提升，瓶颈到底来自 routing、skill context 污染、Qwen executor schema grounding、ACT reward 稀疏，还是 `m_t` 更新方式本身。

当前项目把主问题收敛为一个可验证的 gate：`CLSTR multi-step + m_t` 必须在 AppWorld dev10/dev57 official task success 上超过 `skillrouter_base_live`，并且 recurrent/no-recurrent 或 candidate-source 消融必须支持 `m_t` 的作用，才可写成论文主张。

## 2. 参照论文与材料

| 参照材料 | 本项目中的作用 |
|---|---|
| SkillRouter: Skill Routing for LLM Agents at Scale (`papers/skillrouter.pdf`) | 强 baseline。该工作研究大规模 skill routing，强调 full skill text 对路由很关键，并提供 SkillRouter-Embedding-0.6B / trainable scorer 思路。本项目用它作为 AppWorld per-step live routing baseline，并尝试 SkillRouter-init CLSTR。 |
| SkillX: Automatically Constructing Skill Knowledge Bases for Agents (`../SkillX/README.md`) | skill pool 来源。SkillX 从成功 trajectory 中抽取 planning / functional / atomic skills。本项目规范化得到 188 条 AppWorld skills，作为 CLSTR 与 SkillRouter 的共同候选池。 |
| AppWorld benchmark | 主闭环评测环境。AppWorld 提供真实 API、数据库状态与 deterministic verifier，适合评估 task success，而不仅是 retrieval recall。 |
| `Hybrid_Latent_Next_Skill_Router_Method_AAAI_v2_Routing_AppWorld_SkillX.md` | CLSTR 当前方法 spec：把 closed-loop skill routing 形式化为 skill-subspace belief filtering，并将主 benchmark 从 SkillsBench/ALFWorld 转向 AppWorld + SkillX。 |
| `19876_Hybrid_Latent_Reasoning_.pdf` | HRPO/GRPO-style grouped rollout 思路参考。本项目据此实现同一 task 多 rollout、组内 reward 标准化和 KL/reference prior。 |
| `papers/2605.05413v1.pdf` / ALFWorld 相关记录 | 主要作为边界参照：ALFWorld 上强 LoRA/SFT/RL agent 路线与 CLSTR 的 skill-routing 层定位不同，因此 ALFWorld 在当前项目中降级为辅助 case study。 |

同时，项目保留 SKILLRET、ALFWorld、ScienceWorld、WebShop 等路径作为辅助诊断或历史实验，但当前论文主线是 AppWorld + SkillX。

## 3. 创新性

CLSTR 的创新点不在于提出另一个静态 retriever，而在于把 skill routing 放到闭环决策过程中：

1. **Skill-subspace belief state**：CLSTR 维护 recurrent `m_t`，它位于 skill embedding 子空间中，用于表示当前任务阶段、已执行动作和环境反馈共同诱导出的 skill belief。
2. **Prediction-correction 更新**：每一步先根据上一步 selected skill 和 observation 做 transition，再用新 state text 的 skill-subspace observation 校正 belief，类似 POMDP belief filtering。
3. **Belief-conditioned policy**：skill scoring 不只依赖当前 query embedding `h_t`，还引入 `m_t`、candidate skill embedding 以及二者交互项，用于动态重排候选技能。
4. **Multi-step executor coupling**：CLSTR 不生成最终 Python code，而是选择 top-k skill context；Qwen3-8B 仍负责代码生成，AppWorld official verifier 给最终 task success。
5. **ACT/HRPO 训练链路**：在 train split 上对同一 AppWorld task 采样多条 skill/context trajectory，用 official success、execution_ok、wrong-completion penalty 等训练信号更新 CLSTR policy，同时通过 routing prior/KL 避免破坏强初始化。
6. **安全上下文与泄漏边界**：项目区分 `raw` SkillX body 与 `safe_metadata` context，避免未校验 SkillX body 中的硬编码凭证或过时 API 直接污染 executor；同时对 SkillX AppWorld skill pool 做 dev/test 泄漏审计。

因此，CLSTR 的目标贡献应表述为：在强 SkillRouter 或 SkillX skill library 基础上，利用执行历史和 latent belief 做更好的连续 skill/context selection。当前结果还没有支持“CLSTR 已超过 SkillRouter”的强结论。

## 4. 研究方法框架

当前代码框架可以分为六层。

### 4.1 数据与 skill pool

- AppWorld split/task/API 数据：`data/appworld_routing/`
- SkillX AppWorld skill pool：`data/appworld_skill_pool/skill_pool.jsonl`
- 规模：188 条 SkillX AppWorld skills；AppWorld split 统计为 train 90、dev 57、test_normal 168、test_challenge 417。
- 审计：`outputs/skillx_appworld_leakage_audit/report.json` 未发现 dev/test task id、instruction 或 high-overlap near-duplicate；SkillX plan user_task 与 train split 有交集，与 dev/test 无交集。

### 4.2 静态 routing 与初始化

- `clstr/appworld_routing.py` 构造 AppWorld task-to-SkillX qrels。
- `clstr/appworld_clstr_eval.py` 支持用 CLSTR checkpoint 导出 top-k skill ranking。
- SkillRouter-init 配置使用 `.cache/hf_models/SkillRouter-Embedding-0.6B`、`d=1024`、SkillRouter-style query/skill serialization。

代表性 routing 结果：

| 方法 | AppWorld dev recall@1 | recall@5 | MRR@20 | 说明 |
|---|---:|---:|---:|---|
| SkillRouter-base trainable scorer | 0.982456 | 1.0 | 0.991228 | 强静态 baseline |
| CLSTR SkillRouter-init routing-only | 0.964912 | 1.0 | 0.976608 | 更强初始化后接近 SkillRouter |
| CLSTR MiniLM routing-only | 0.912281 | 1.0 | 0.956140 | 轻量初始化版本 |

这些是 routing-layer 指标，不等价于 AppWorld official task success。

### 4.3 Single-shot executor benchmark

早期链路把 retrieved skill context、task instruction 和 AppWorld API docs 一次性交给 Qwen3-8B 生成 Python code，再用 AppWorld `task_completed()` 和 `evaluate()` 判断 success。结果显示 raw SkillX body 可能污染 Qwen executor，因此加入 `safe_metadata` context。

代表性 dev57 single-shot 结果：

| 方法 | success | success_rate | execution failures |
|---|---:|---:|---:|
| Qwen-only | 14/57 | 0.245614 | 24 |
| SkillRouter embedding | 15/57 | 0.263158 | 32 |
| SkillRouter-base | 14/57 | 0.245614 | 32 |
| CLSTR-base routing-only | 9/57 | 0.157895 | 32 |
| CLSTR-act v2 | 13/57 | 0.228070 | 29 |

结论：single-shot 设置下，CLSTR 没有超过 Qwen-only 或 SkillRouter；这推动项目转向 multi-step controller。

### 4.4 Multi-step CLSTR controller

核心模块是 `clstr/appworld_multistep.py`：

- 每个 episode 维护 `m_t`；
- 每一步构造 `state_text_t = user goal + compact history + latest environment output/error`；
- CLSTR 或 SkillRouter live controller 选择 top-k skill context；
- Qwen3-8B 生成下一段 code；
- AppWorld 执行 code，并将 `execute_output` 和 `next_state_text` 反馈给 CLSTR 更新 `m_t`；
- `MultiStepStopPolicy` 在 `done/task_completed/success/STOP head` 条件下停止，避免重复副作用。

当前对照方法包括：

- `qwen_only`
- `skillrouter_base_live`
- `clstr_multistep`
- `clstr_skillrouter_hybrid_live`
- recurrent/no-recurrent 消融
- `routing_belief_union` / `routing_belief_union_after_update` candidate source 消融
- gated residual / prior residual / strict reward ACT 版本

### 4.5 ACT/HRPO 训练

核心模块是 `clstr/appworld_act_hrpo.py`：

- single-step 和 multi-step HRPO 路径均已实现；
- 训练只用 train split rollout；
- 组内 advantage 来自同一 task 的多 rollout reward；
- 加入 routing prior、KL reference、wrong-completion penalty、step-level advantage fallback；
- Phase 1 修复后支持 `enable_closed_loop_gradient`，使 reward gradient 可以穿过 `m_t` update 影响 transition/gate 路径。

Phase 1 修复后全量测试曾达到 `447 passed`，说明代码层面的关键 bug 已被测试覆盖。但 Phase 1 后的 dev10 gate 仍只有 `6/10`，未达到继续 dev57/test 的成功门槛。

### 4.6 边界与防泄漏

- SkillRouter 论文/实现侧没有发现 AppWorld test 泄漏证据。
- SkillX AppWorld 发布文件层面没有发现 dev/test contamination。
- 当前所有 ACT/HRPO 训练应只用 AppWorld train split；dev/test 只能用于评测。
- 不应将 prompt/API schema 修复写成 CLSTR 方法收益；这类修复属于 executor grounding。

## 5. 初步实验与结果

### 5.1 AppWorld + SkillX 基础接入

已完成：

- SkillX AppWorld skill pool 规范化为 188 条 skills；
- AppWorld smoke 与 runtime adapter smoke 通过；
- skill embedding input 构建完成；
- multi-step AppWorld executor、SkillRouter live baseline、CLSTR recurrent controller、ACT/HRPO 训练入口均已实现。

### 5.2 Multi-step dev10 主要结果

代表性 completed results：

| 实验设置 | Qwen-only | SkillRouter-base live | 最佳/相关 CLSTR 结果 | 结论 |
|---|---:|---:|---:|---|
| 初始 multi-step dev10 | 6/10 | 7/10 | CLSTR step500 3/10 | CLSTR context 排序明显不稳定 |
| next-state + sampling-prior v3 | 6/10 | 7/10 | recurrent 7/10；no-recurrent 7/10；belief-union 7/10 | 追平但未超过；无 `m_t` 增益 |
| stop-policy dev10 | 7/10 | 7/10 | CLSTR v3 after-update 6/10 | 通用 stop policy 改善副作用控制，但 CLSTR 未超过 baseline |
| full90 ACT v4 | 7/10 | 7/10 | gated recurrent 7/10，gated no-recurrent 6/10 | 有弱信号，但成功集合未超过 SkillRouter/Qwen |
| prior-residual strict-reward v1 | 7/10 | 7/10 | gated no-m_t 7/10；gated recurrent 6/10 | strict reward 未解决稀疏 credit；`m_t` 未显示正贡献 |
| Phase 1 dev10 gate | - | - | 6/10 | 代码修复后仍未过 dev10 success gate |

重要结论：

- 当前 CLSTR 最好只是追平 `skillrouter_base_live` 的 `7/10`，没有稳定超过。
- 多个 recurrent 版本没有优于 no-recurrent；有些设置中 recurrent 反而更差。
- `530b157_*` phone+Venmo 任务是共同失败簇，Qwen-only、SkillRouter 与 CLSTR 都没有解决，说明部分瓶颈在 executor/task-family 上限，而不是 CLSTR routing 单点失败。
- step-positive hit 与 task success 不完全一致：有些方法命中正例但仍因 Qwen API schema/答案构造失败而 evaluation 不通过。

### 5.3 ALFWorld 辅助结果

ALFWorld 当前不是主表，但保留为 sequential decision case study。较新的 DAgger + Q_success 辅助结果：

| 方法 | episodes | success_rate | 备注 |
|---|---:|---:|---|
| Qwen3-8B direct reference | 274 | 0.109489 | reference，不计为 CLSTR |
| DAgger + Q_success full | 274 | 0.167883 | CLSTR structured controller，Qwen frozen |
| 0.6B CLSTR controller best | 274 | 0.003650 | 历史弱 baseline |

该结果说明 CLSTR-style structured controller 在 ALFWorld 辅助路径上有一定收益，但不能替代 AppWorld + SkillX 主线，也不能与 ALFWorld 强 LoRA/SFT/RL 路线直接比较。

## 6. 当前判断

当前项目最重要的研究判断是保守的：

1. **Pipeline 已经打通**：AppWorld + SkillX + SkillRouter + Qwen3-8B + CLSTR multi-step/ACT 训练评测链路已经可运行。
2. **Routing 能力不弱**：SkillRouter-init CLSTR 在 AppWorld dev routing 上接近 SkillRouter-base，且 recall@5 常为 1.0。
3. **Task success 尚未证明**：在 official AppWorld task success 上，CLSTR 还没有稳定超过 `skillrouter_base_live`。
4. **`m_t` 价值尚未证明**：recurrent belief、candidate union、gated residual、prior residual 等变体还没有给出清晰的 recurrent > no-recurrent 证据。
5. **主要瓶颈混合存在**：包括 Qwen executor API grounding、SkillX context 污染、routing/selection mismatch、ACT reward 稀疏、trajectory-level credit assignment 弱、transition/gate 训练不足。

因此，论文表述应避免写成“CLSTR 已在 AppWorld 上优于 SkillRouter”。更准确的当前表述是：CLSTR 提供了一个闭环 latent skill routing 研究框架，已完成 AppWorld/SkillX/Qwen3-8B 的完整实验管线和多轮诊断，当前正在验证 recurrent belief 与 ACT 是否能在强 SkillRouter prior 上带来稳定 task-success 增益。

## 7. 下一步建议

短期应优先做 benchmark-agnostic 的方法修复，而不是继续堆 AppWorld dev-template prompt：

1. 将 Phase 1 的 `enable_closed_loop_gradient`、step-level advantage fallback、STOP head 接入新的 full train ACT 作业。
2. 在 train split 上增加更稳定的 step-level / pairwise / leave-one-step credit assignment，缓解 240 rollout 中大量 zero-success update 的问题。
3. 对 `m_t` 做离线 transition/gate 监督，先证明 belief candidate 或 rerank 能稳定改善 selected hit，再进入 expensive Qwen rollout。
4. 保留 `skillrouter_base_live`、hybrid alpha=0、no-recurrent 作为强对照，只有 recurrent/prior-residual 版本超过它们时才扩展 dev57/test。
5. 如果 CLSTR 仍只追平 SkillRouter，应将结果作为负结果/诊断记录，并调整论文贡献为“闭环 routing 框架与失败分析”，而不是 task-success SOTA。

## 8. 相关文件索引

- 当前状态总览：`README.md`
- 长实验记录：`description.md`
- AppWorld pivot 设计：`docs/clstr_pivot_routing_appworld.md`
- 核心 multi-step executor：`clstr/appworld_multistep.py`
- ACT/HRPO 训练：`clstr/appworld_act_hrpo.py`
- AppWorld executor prompt/runtime：`clstr/appworld_executor.py`
- CLSTR model：`clstr/model.py`
- SkillX adapter：`clstr/bridges/skillx/appworld_adapter.py`
- 最新关键 comparison：`outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_dev10_comparison/comparison.md`
- Phase 1 dev10 gate：`outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json`
- SkillX leakage audit：`outputs/skillx_appworld_leakage_audit/report.md`
