---
marp: true
theme: default
paginate: true
size: 16:9
---

# CLSTR 当前成果与下一阶段展望

面向大规模 Skill Pool 的状态化 Skill Routing

2026-06-10

---

# 今天想讲清楚什么

1. CLSTR 现在到底是什么，不是什么。
2. 当前 pipeline 已经完成到哪一层。
3. 现有结果能支持哪些结论，不能支持哪些结论。
4. 下一阶段怎样把 CLSTR 的价值和 SkillRouter 公平地区分出来。

---

# 一句话口径

> CLSTR 已经从静态 routing 设想推进到可训练、可评估、可接 online feedback 的 skill-routing pipeline。

但当前还不能说：

> CLSTR 已经全面超过 SkillRouter，或者已经解决 ALFWorld/AppWorld。

更准确的结论是：

> 工程链路已经打通，关键诊断问题已经暴露并部分修复；下一步要在多步、状态化、动态 skill 和 online adaptation 中证明增益。

---

# 背景问题：大规模 Skill Routing

给定任务 query、当前 observation/history，以及一个很大的 skill pool：

```text
goal / query / observation / history
  -> choose useful skill or tool from S = {s1, s2, ..., sN}
```

难点：

- skill pool 可以到几万甚至更大。
- skill 之间高度相似，name/description 往往不够。
- 单步 query 不一定包含完整状态。
- 多步任务要求模型根据环境反馈持续改变选择。

---

# 关键概念 1：Skill 和 Skill Pool

**Skill** 可以理解为可被 agent 调用的一段能力、工具、API 或动作模式。

在当前项目里，skill 可能来自：

- TRAJECT-Bench / ToolBench-G3 / ToolRet 等工具或轨迹数据。
- AppWorld + SkillX 的 188 条 AppWorld skills。
- ALFWorld 等历史诊断环境中的 action skill。

**Skill Pool** 是统一后的候选集合。当前 v4.1b 主线规模：

```text
skill_pool:     37,615
trajectories:  111,488
retrieval rows: 545,564
```

---

# 关键概念 2：Routing 不是 Execution

CLSTR 当前定位不是完整端到端 agent。

```text
CLSTR:
  负责选择/排序 skill 或工具候选

Executor / LM / environment:
  负责真正执行动作、调用 API、产生环境反馈
```

因此要区分两类指标：

| 类型 | 说明 |
|---|---|
| routing 指标 | recall@K、MRR、next-skill accuracy |
| task success 指标 | AppWorld/ALFWorld 最终任务是否完成 |

当前 CLSTR 的主要证据仍集中在 routing 和 pipeline viability，task success 还没有形成强正结果。

---

# 关键概念 3：Closed-loop Skill Routing

静态 skill retrieval 更像：

```text
query -> skill
```

CLSTR 想解决的是多步闭环：

```text
goal / obs_t / history_t
  -> belief m_t
  -> choose skill_t
  -> environment feedback / obs_{t+1}
  -> update belief m_{t+1}
  -> choose next skill
```

核心变量：

- `m_t`：latent belief / skill memory，表示当前任务进度和状态。
- transition：根据历史动作和新 observation 预测下一步 skill 状态。
- belief gate：融合当前观测和历史记忆。

---

# SkillRouter 是什么强基线

SkillRouter 的典型方式是 full-text retrieve-and-rerank：

```text
query
  -> bi-encoder retrieve top-K from large skill pool
  -> cross-encoder / reranker refine candidates
```

它的强点：

- 使用完整 skill text。
- bi-encoder 负责大池召回。
- reranker 负责 top candidates 精排。
- listwise reranking 对相似 skill 区分有效。
- 新 skill 只要编码并加入索引，就天然支持 append。

所以不能把 SkillRouter 当弱 baseline。

---

# CLSTR 不应该怎么讲

不建议说：

> CLSTR 是一个更复杂的 SkillRouter，所以应该在单步检索上直接超过 SkillRouter。

原因：

- SkillRouter 专门优化单步 full-text retrieval/reranking。
- CLSTR 多了 belief、transition、ACT/HRPO，如果只做单步检索，复杂度不一定转化为收益。
- 当前结果也还没有证明 CLSTR 单步全面超过 SkillRouter。

合理口径是：

> CLSTR 应该吸收强 retrieval 初始化，然后在多步状态化 routing 和 online adaptation 上证明额外价值。

---

# 当前主线 Pipeline

```text
Pre-stage data/audit
  -> Stage0 coarse retrieval
  -> Stage1 heads init / candidate reranking
  -> Stage2 supervised CLSTR base
  -> Stage3 HRPO-style policy update
  -> Stage4 transition-conditioned ACT
  -> Stage4-online environment feedback
```

报告时可以把 Stage3/Stage4-online 讲成探索性路径：

- Stage0/1/2 是当前最关键的稳定 routing 主线。
- Stage4/online 是为了验证 closed-loop 和环境反馈价值。

---

# 阶段解释：Pre-stage

目标：把不同来源的数据变成干净、可审计、可训练的统一 skill routing 任务。

输入：

- 工具/skill 文本。
- trajectory / retrieval / qrels。
- benchmark split 信息。

关键工作：

- canonical skill id。
- partial dedup。
- positive / negative label 整理。
- leakage audit，确保 dev/test/eval query 或 task id 不进入训练。

输出：

```text
skill_pool.jsonl
trajectories.jsonl
retrieval.jsonl
leakage_audit.json
```

---

# 阶段解释：Stage0 Coarse Retrieval

目标：先从大池子里把可能相关的 skill 召回。

```text
query/state text
  -> encoder
  -> skill embedding table
  -> top-M candidates
```

Stage0 不负责最终决策，它负责保证候选集质量。

如果 Stage0 把正确 skill 排得太低，后面的 Stage1/2 很难补救。

当前相关基线结果：

```text
outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json
qrel queries: 328,320
Recall@20: 0.706692
Recall@50: 0.811626
Recall@100: 0.884804
NDCG@20: 0.482335
```

---

# 阶段解释：Stage1 Heads Init

目标：在 Stage0 top-M candidates 上初始化 CLSTR 的 heads。

它学习的不只是“大池召回”，而是在候选集内做更接近 CLSTR 的评分：

- skill head。
- STOP head。
- belief / transition 相关参数。
- candidate handoff 逻辑。

关键点：

> Stage1 消费 Stage0 top-M，不是另起一个模型，也不是 ensemble。

如果 positive 不在 top-M，需要明确记录 skip 或 deterministic inject，避免指标口径不清。

---

# 阶段解释：Stage2 CLSTR Base

Stage2 是 supervised CLSTR-base 主训练。

它从 Stage1 checkpoint 继续训练同一个 CLSTRModel，目标是让模型在多步轨迹中学习：

- 当前状态下应该选择哪个 skill。
- `m_t` 和 history 对选择有什么影响。
- transition/belief 是否能帮助下一步 routing。

当前主线 checkpoint 被后续 online smoke 使用：

```text
outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/
  checkpoints/clstr_full_base-step10000.pt
```

---

# Stage2 当前诊断

`outputs/clstr_stage2_v4_pre_audit_top350_v1/stage2_v4_pre_audit_report.md`

状态是 `ok`，但结论不是“没有问题”，而是定位了下一步瓶颈：

| Benchmark | 现象 | 解释 |
|---|---|---|
| ToolBench-G3 | Stage2 r@5 低于 Stage0 | Stage2 reranker 存在退化，需要检查 label/input/loss |
| TRAJECT-Bench | Stage2 对部分 switch rows 有帮助 | transition 信号可能有用，但 Stage0 candidate quality 和 top5 calibration 不足 |

推荐修复方向：

- 加 available-skill inventory mask。
- ToolBench-G3 加 multi-positive labels 和 listwise loss。
- TrajectBench 继续审计低 Stage0 rank 是 query 问题还是 label equivalence 问题。

---

# 阶段解释：Stage3 / Stage4

Stage3 和 Stage4 用来验证更接近 closed-loop 的学习。

**Stage3 HRPO-style policy update**

- 根据 preference / rollout signal 做 policy update。
- 主要是策略层面的改进尝试。

**Stage4 transition-conditioned ACT**

- 学习 transition-conditioned next-skill objective。
- 目标是让 `m_t`、transition 和下一步 skill 选择真正相关。

当前口径：

> Stage4 offline 证明了训练 artifact 和链路可用，但还没有形成明显超过 Stage2 的强证据。

---

# 阶段解释：Stage4-online

Stage4-online 是把 CLSTR 放到环境反馈中小步验证。

```text
rollout in environment
  -> collect obs_t, action_t, obs_{t+1}, reward
  -> compute grouped advantage / auxiliary losses
  -> update action policy and belief/transition
```

当前主要使用 ALFWorld smoke 诊断。

它的作用不是马上报主表成绩，而是回答：

- online feedback 是否能进入 transition/belief。
- reward/advantage 是否非退化。
- 模型能否产生目标行为前缀。
- 失败来自 routing、executor、探索，还是环境状态建模。

---

# 当前工程成果概览

| 模块 | 当前状态 |
|---|---|
| v4.1b unified skill pool | 已构建，37,615 skills |
| trajectory/retrieval 数据 | 111,488 trajectories，545,564 retrieval rows |
| Stage0 retrieval | 有 SkillRouter-compatible baseline 和 gate |
| Stage1/Stage2 | conservative-listwise 主线 checkpoint 可用 |
| Stage4 offline ACT | 可训练、可 gate，但增益不明显 |
| Stage4-online HRPO | 已接入 ALFWorld smoke |
| AppWorld + SkillX | runtime、adapter、skill pool smoke 均 ok |
| 动态 skill registry | 尚未工程化完成 |

---

# AppWorld / SkillX 当前状态

已经完成的 AppWorld smoke：

```text
outputs/appworld_smoke/report.json: status=ok
outputs/appworld_adapter_smoke/report.json: status=ok
outputs/appworld_skill_pool_smoke/report.json: status=ok
outputs/appworld_oracle_solution_smoke/report.json: status=ok
```

关键事实：

- AppWorld package：`appworld==0.1.3.post1`
- adapter 能 reset 单个 train task：`82e2fac_1`
- instruction、required apps、API docs、DB、ground truth/verifier 均可访问
- SkillX AppWorld pool：188 skills
- oracle compiled solution 可以执行成功

这说明 runtime 和数据链路可用，不代表 CLSTR 已完成 AppWorld task success。

---

# AppWorld Routing 对比

`outputs/appworld_routing_comparison_full_v1_with_skillrouter_base/comparison.md`

| method | recall@1 | recall@5 | recall@20 | mrr@20 |
|---|---:|---:|---:|---:|
| SkillRouter embedding 0.6B | 0.245614 | 0.526316 | 0.666667 | 0.371327 |
| SkillRouter trainable scorer | 0.982456 | 1.0 | 1.0 | 0.991228 |
| CLSTR skill-table logits | 0.824561 | 1.0 | 1.0 | 0.897661 |

可以说：

> CLSTR 在这个小 AppWorld routing setting 下能做到 recall@5/20 满召回，但 top1/MRR 还不如强 SkillRouter trainable scorer。

不能说：

> CLSTR 已经超过 SkillRouter。

---

# AppWorld Executor 对比边界

`outputs/appworld_executor_comparison_dev3_v3/comparison.md`

| method | task_count | success_count | success_rate |
|---|---:|---:|---:|
| qwen_only | 3 | 0 | 0.0 |
| skillrouter_embedding | 3 | 0 | 0.0 |
| skillrouter_base | 3 | 0 | 0.0 |
| clstr_base | 3 | 0 | 0.0 |

解释：

- 这是 executor/runtime 端到端结果。
- 所有方法都没有成功，说明当前瓶颈不只是 CLSTR routing。
- 这组结果应作为诊断，不应作为 CLSTR 负面或正面结论的主要依据。

---

# Stage4-online 修复了什么

最初 online HRPO 的问题：

1. `env.step(action_t)` 后的 `obs_{t+1}` 没有接回 transition/belief。
2. 不同失败轨迹 reward 一样，advantage 退化为 0。
3. action trace 大量无关动作。
4. rollout advantage 被 step 数二次缩小。
5. goal prior 只偏向目标物体，不偏向目标位置。

当前已修复：

- post-action observation auxiliary loss。
- progress reward / goal interaction reward / irrelevant action penalty。
- advantage scaling。
- 目标物体 + receptacle/location prior。
- step credit、self-imitation、progress memory 等诊断组件。

---

# Stage4-online 关键结果

receptacle-prior smoke：

```text
outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s12_smoke

viability.status: ok
rollout_success_rate: 0.0
mean_rollout_reward: -0.3565
updates_with_nonzero_advantage: 5
online_observation_aux_count: 240
mean_goal_interaction_points: 0.15
mean_irrelevant_action_count: 2.30
```

相对上一版：

```text
mean_rollout_reward:       -0.4525 -> -0.3565
mean_irrelevant_action:       3.95 -> 2.30
sidetable action count:          17 -> 64
```

---

# Stage4-online 出现了什么正向行为

代表性目标行为前缀：

```text
take tissuebox 3 from diningtable 1
go to sidetable 2
move tissuebox 3 to sidetable 2
```

这支持的结论：

> online Stage4 已经不是空跑，模型能产生目标导向行为前缀。

这不支持的结论：

> 模型已经稳定完成 ALFWorld task。

因为：

```text
rollout_success_rate = 0.0
```

---

# 后续 online 诊断结果

更强 credit / progress-memory 版本仍没有解决 task success：

| run | success | mean reward | mean goal interaction | mean irrelevant |
|---|---:|---:|---:|---:|
| credit imitation w0.5 lr2e-4 | 0.0 | -0.6085 | 0.95 | 3.65 |
| strict credit + progress memory | 0.0 | -0.6055 | 0.75 | 3.45 |

解释：

- learning signal 存在。
- 目标物体操作变多。
- 但还没有形成稳定的“完成子目标、保持正确放置、停止”的策略。

下一步不应该盲目扩大 online RL，而应先做离线 trajectory/preference 预热或更稳定环境诊断。

---

# 当前已经证明什么

可以比较稳地说：

1. CLSTR 的大池数据、训练和评估链路已经成型。
2. AppWorld runtime、adapter、SkillX pool 和 oracle sanity 已经通过 smoke。
3. Stage0/1/2/4 的工程路径已经可跑。
4. Stage2 诊断明确指出了下一步该修什么。
5. Stage4-online 已经接入 post-action observation，并且 reward/advantage 不再退化。
6. Online smoke 出现了目标行为前缀。

---

# 当前还没有证明什么

不能说：

1. CLSTR 已经全面超过 SkillRouter。
2. CLSTR 已经在官方 task-success benchmark 上取得正结果。
3. Stage4 offline ACT 已经带来显著收益。
4. Stage4-online 已经解决 ALFWorld。
5. CLSTR 已经成熟支持 append-only dynamic skill pool。

这些不是坏消息，而是当前最清晰的风险边界。

---

# 最大风险 1：单步 routing 上 SkillRouter 很强

SkillRouter 在 full-text retrieve-and-rerank 上非常专门。

如果 CLSTR 只在单步检索上和它硬比，风险很大：

- CLSTR 的 belief/transition 在单步 setting 里发挥不出来。
- 复杂 heads 可能不如强 text reranker。
- AppWorld routing 已显示 CLSTR top1/MRR 低于强 SkillRouter scorer。

应对：

> 把 SkillRouter 作为强初始化和强基线，在多步状态化任务中证明 CLSTR 的额外收益。

---

# 最大风险 2：Stage2 reranking 仍需修

Stage2 预审说明：

- ToolBench-G3 有系统性 reranker degradation。
- TrajectBench 存在 Stage0 candidate low-rank 问题。

下一步优先修：

1. available-skill inventory mask。
2. multi-positive next-skill labels。
3. listwise loss over top-M candidates。
4. self/switch transition 分开审计。
5. 持久化 Stage0 top candidate ids，方便 row-level 复盘。

---

# 最大风险 3：Online RL 还没转成成功率

当前 online evidence 是：

```text
non-degenerate training signal
goal-directed prefix
some action-quality improvement
```

不是：

```text
stable task success
official benchmark gain
```

下一步建议：

- 不直接烧 full online RL。
- 先用离线 trajectory/preference 做更稳定的 action-quality warmup。
- 在小环境中验证 progress memory、STOP、inventory/state tracking 是否真的有效。

---

# 最大风险 4：动态 Skill 还没工程化

SkillRouter 的天然优势：

```text
new skill text
  -> encode
  -> add to index
  -> retrieve / rerank
```

CLSTR 当前更多是：

```text
new skill_pool.jsonl
  -> rebuild skill table
  -> rerun Stage0 / handoff / eval
```

还缺：

- append-only registry。
- embedding/index 增量更新或可靠重建。
- checkpoint shape mismatch handling。
- old/new skill split metrics。
- unseen-skill fallback。

---

# 下一阶段：公平比较矩阵

不要只报一个总分。建议拆成四层：

| 层级 | SkillRouter | CLSTR | 要回答的问题 |
|---|---|---|---|
| 单步静态 routing | bi-encoder/reranker | Stage0/1/2 no-history | 基础能力是否接近 |
| 同候选集 reranking | fixed top-M reranker | CLSTR heads/transition | 排序能力是否有增益 |
| 多步 trajectory routing | per-step independent | stateful `m_t` | history/belief 是否有用 |
| 动态 skill append | dynamic index | dynamic registry + fallback | 新 skill 能否 no-retrain 使用 |

---

# 下一阶段：DynamicSkillRegistry

建议实现：

```text
clstr/dynamic_skill_registry.py
scripts/build_dynamic_skill_registry.py
scripts/audit_dynamic_skill_registry.py
tests/test_dynamic_skill_registry.py
```

核心流程：

```text
base skill_pool
  + append new skill rows
  + validate schema
  + encode new skill text
  + rebuild / update embeddings
  + update skill_id -> index
  + invalidate old top-M cache
  + export registry manifest
```

---

# 新 Skill 的 Scoring 策略

对 old skill：

```text
score = retrieval + policy/head + transition/belief
```

对 new/unseen skill：

```text
score = retrieval-heavy + text-rerank-heavy + weaker transition prior
```

原因：

- 新 skill 没有被 transition/belief 训练过。
- 直接用完整 CLSTR score 可能压低新 skill。
- retrieval-first fallback 更接近 SkillRouter 的动态能力。

---

# 下一阶段：30 天路线建议

第一周：

- 完成 Stage2 row-level 诊断复盘。
- 加 available-skill inventory mask。
- 修 ToolBench-G3 multi-positive / listwise loss smoke。

第二周：

- 跑单步静态 routing 和同候选 reranking 对比。
- 用 SkillRouter 强基线定义 CLSTR 必须达到的基础线。

第三周：

- 实现 DynamicSkillRegistry 最小闭环。
- 报 old/new skill split metrics。

第四周：

- 做多步 trajectory routing 对比。
- 决定 online RL 是继续 ALFWorld，还是切到更稳定的 executor/benchmark。

---

# 论文叙事建议收敛

不要把所有 stage 都包装成贡献。

建议贡献收敛为三点：

1. **Stateful Skill Memory**
   - 用 `m_t` / belief 表示多步任务进度。

2. **Transition-aware Multi-step Routing**
   - skill 选择依赖 history、observation 和 transition，而不是每步独立检索。

3. **Dynamic / Online Adaptation**
   - 在大 skill pool 上支持新增 skill 和环境反馈驱动的改进。

Stage0/1/2/4 是实现路径，不是全部都要写成独立创新。

---

# 今晚可直接使用的开场

> 我目前不是要说 CLSTR 已经全面超过 SkillRouter，而是想说明：系统已经推进到了可以公平回答这个问题的阶段。
>
> SkillRouter 在单步大池检索上非常强；CLSTR 要证明的是状态化、多步和 online adaptation 的额外价值。

---

# 今晚可直接使用的成果表述

> 工程上，CLSTR 已经完成了从 unified skill pool、Stage0/1/2 到 Stage4 offline/online 的完整链路。
>
> 当前 v4.1b skill pool 有 37,615 条 skill，配套 111,488 条 trajectory 和 545,564 条 retrieval rows。
>
> AppWorld runtime、adapter、SkillX pool 和 oracle sanity 都已经通过 smoke。

---

# 今晚可直接使用的结果表述

> 在 AppWorld routing 上，CLSTR skill-table logits 的 recall@5/20 达到 1.0，但 recall@1 和 MRR 还低于强 SkillRouter trainable scorer。
>
> 在 ALFWorld online smoke 上，Stage4-online 已经有非退化训练信号，并出现了目标行为前缀，但 rollout success 仍为 0。

---

# 今晚可直接使用的风险表述

> 当前最大风险不是系统跑不起来，而是如何证明 CLSTR 的额外价值。
>
> 单步 routing 上 SkillRouter 很强；online RL 还没有 task success；动态 skill registry 还没工程化。
>
> 所以下一步要把比较拆成单步、同候选 reranking、多步 trajectory 和动态 skill 四层，而不是只报一个混合总分。

---

# 如果被问：现在比 SkillRouter 好吗？

建议回答：

> 目前不能这么说。SkillRouter 在单步 full-text retrieval/reranking 上是强基线。
>
> CLSTR 的价值要看多步、状态化和 online adaptation。我们已经在 AppWorld routing 上接近部分指标，但 top1/MRR 还不如强 SkillRouter scorer。
>
> 下一步会拆成四层对比，明确 CLSTR 到底在哪些场景有额外收益。

---

# 如果被问：为什么不直接照搬 SkillRouter？

建议回答：

> Stage0/Stage1 可以吸收 SkillRouter 的设计，比如 full-text embedding、top-M handoff 和 listwise reranking。
>
> 但 SkillRouter 每一步更接近独立检索，没有显式维护 `m_t`、transition 和 history-dependent belief。
>
> CLSTR 的目标是在强检索基础上做 closed-loop routing。

---

# 如果被问：Stage4-online 为什么还没成功？

建议回答：

> 一开始是 observation 没接回 transition/belief，后来是 reward/advantage 退化，再后来是探索和状态记忆不足。
>
> 这些问题现在已经逐步定位并修复了一部分，所以有了目标前缀。
>
> 但成功率还没有起来，说明下一步不该盲目扩大 RL，而要先加强离线 trajectory/preference 预热和状态建模。

---

# 如果被问：动态 skill 能做到吗？

建议回答：

> 单步检索层可以做到接近 SkillRouter：append skill、编码、更新索引即可。
>
> CLSTR 还需要处理 skill table shape、bias、cache、handoff 和 checkpoint 兼容问题。
>
> 新 skill 会先走 retrieval-first fallback，transition/belief 的收益需要 few-shot 或 online adaptation。

---

# 结论页

当前 CLSTR 已经具备完整 pipeline，并且 online Stage4 不再是空跑：

- 有 post-action observation。
- 有非退化 reward/advantage。
- 有目标行为前缀。

但还没有证明：

- 官方 task success 提升。
- 单步超过强 SkillRouter。
- 成熟动态 skill registry。

下一步核心：

> 用 SkillRouter 作为强单步基线，在多步 trajectory、动态 skill pool 和 online adaptation 上证明 CLSTR 的额外价值。

