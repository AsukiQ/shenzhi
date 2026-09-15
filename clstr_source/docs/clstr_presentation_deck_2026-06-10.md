---
marp: true
theme: default
paginate: true
size: 16:9
---

# CLSTR 当前进展与下一阶段计划

面向大规模 Skill Pool 的状态化 Skill Routing

2026-06-10

---

# 我今天想讲清楚的三件事

1. **我目前已经推进到哪里**
   - unified skill pool
   - Stage0 / Stage1 / Stage2 / Stage4
   - ALFWorld online HRPO 初步闭环

2. **这些结果目前能说明什么**
   - 训练链路已经打通
   - online Stage4 出现正向行为前缀
   - 但还没有 task success 结果

3. **我下一步准备怎么证明方法价值**
   - 和 SkillRouter 公平比较
   - 补动态 skill pool
   - 在多步/状态化场景中证明 CLSTR 的增益

---

# 一句话总结

> 我现在已经把 CLSTR 从静态 routing 设想推进到了完整训练与评估 pipeline。

但我现在还不能说：

> CLSTR 已经全面超过 SkillRouter。

更准确的定位是：

> SkillRouter 是强单步 retrieve-and-rerank 基线；我接下来要证明的是 CLSTR 在多步、状态化、online adaptation 和动态 skill pool 下的额外价值。

---

# 背景：Skill Routing 为什么难

给定一个任务 query 和一个很大的 skill pool：

```text
query q
large skill pool S = {s1, s2, ..., sN}
```

目标是找到真正能解决任务的 skill。

困难点：

- skill pool 大，候选可能有几万到八万级；
- skill 之间高度相似；
- name/description 不够，skill body 很关键；
- 单步 query 不一定包含完整状态；
- 多步任务需要根据 observation/history 改变选择。

---

# SkillRouter 是什么

SkillRouter 的核心是 full-text 两阶段检索：

```text
query
  -> bi-encoder retrieve top-20 from ~80K skill pool
  -> cross-encoder rerank top-20
```

它的强点：

- 使用完整 skill text；
- bi-encoder 负责大池 recall；
- cross-encoder 负责精细区分；
- listwise reranking loss 很有效；
- 工程上天然支持新增 skill。

---

# 我不会把 CLSTR 讲成什么

我不准备这样讲：

> CLSTR 是一个更复杂的 SkillRouter，所以应该在单步检索上直接打败 SkillRouter。

问题：

- SkillRouter 专门优化单步 full-text routing；
- CLSTR 多了 belief / transition / online ACT；
- 如果只做单步检索，CLSTR 的复杂性不一定带来收益；
- 目前结果也还没有证明单步超过 SkillRouter。

---

# CLSTR 的合理定位

CLSTR 应该被定位为 closed-loop skill router：

```text
goal / observation_t / history_t
  -> belief / m_t
  -> skill_t or action_t
  -> environment feedback / obs_{t+1}
  -> belief update
  -> next skill selection
```

核心价值不只是：

```text
query -> skill
```

而是：

```text
stateful routing across multiple steps
```

---

# 我目前的 Pipeline

```text
data / skill pool construction
  -> Stage0 coarse retrieval
  -> Stage1 heads init / reranking
  -> Stage2 CLSTR base
  -> Stage4 offline ACT
  -> Stage4-online / ALFWorld HRPO
```

我目前使用的主线 checkpoint：

```text
outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/
  checkpoints/clstr_full_base-step10000.pt
```

我目前使用的 large skill pool：

```text
data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl
skill_count = 37615
```

---

# 我已经完成的工程成果

| 模块 | 当前状态 |
|---|---|
| Unified skill pool | 已构建 v4.1b 主线 |
| Stage0 retrieval | 已训练/评估过多版 |
| Stage1 heads init | conservative-listwise 路线可用 |
| Stage2 CLSTR base | v4.1b checkpoint 可用 |
| Stage4 offline ACT | 可训练、可 gate，但增益不明显 |
| Stage4-online HRPO | 已接入 ALFWorld train split |
| Post-action observation | 已接入 |
| Reward/advantage | 已修复退化问题 |
| Dynamic skill pool | 尚未工程化完成 |

---

# 为什么转向 Stage4-online

Stage4 offline ACT 的问题：

- 它能证明训练 artifact 可用；
- 但主要还是 offline next-skill ranking；
- 没有真实环境反馈；
- 没有明显超过 Stage2。

因此我转向了 online HRPO：

```text
rollout
  -> environment reward / progress
  -> grouped advantage
  -> update CLSTR action policy
```

目标：验证 CLSTR 是否能从真实交互中产生 closed-loop 改善。

---

# Stage4-online 最初的问题

最初 online HRPO 不是“跑起来就有意义”。

发现的问题：

1. 没有把 `obs_{t+1}` 接回 transition/belief；
2. 失败轨迹 reward 完全一样；
3. advantage 为 0，policy loss 为 0；
4. action trace 大量无关动作；
5. rollout advantage 被 step 数二次缩小；
6. goal prior 只偏向目标物体，不偏向目标位置。

---

# 已修复：Observation 链路

现在 rollout step 记录：

```text
observation_text
next_observation_text
next_admissible_actions
done
env_reward
won
goal_condition_success_rate
```

训练时：

```text
state_t + action_t + obs_{t+1}
  -> transition / belief consistency loss
```

同时保证：

```text
pre_action_next_observation_leakage = false
```

---

# 已修复：Reward / Advantage

原问题：

```text
不同失败轨迹 reward 相同
-> grouped advantage = 0
-> HRPO policy loss = 0
```

现在 reward 包含：

- success reward；
- goal condition progress；
- goal object discovery；
- goal object interaction；
- irrelevant action penalty；
- repeated/info action penalty。

结果：

```text
updates_with_nonzero_advantage > 0
hrpo_policy_loss != 0
```

---

# 已修复：Advantage Scaling

原实现：

```text
step_advantage = rollout_advantage / len(steps)
```

但 policy loss 已经对 step 求均值。

这会导致：

```text
long rollout -> policy gradient 被二次削弱
```

修复后：

```text
每个 action 继承 rollout-level grouped advantage
```

验证：

```text
tests/test_online_hrpo.py: 18 passed
```

---

# 已修复：目标位置探索

原 goal-action prior 只偏向目标物体：

```text
tissuebox: yes
sidetable: no
```

这会导致：

```text
可能拿到 tissuebox
但不稳定去 sidetable
```

修复后：

```text
目标物体：强 prior
目标 receptacle / location：弱 prior
```

注意：这是 online exploration 诊断组件，不是主论文默认收益。

---

# Online HRPO Smoke 结果概览

| Run | 关键结果 | 结论 |
|---|---|---|
| observation smoke | observation aux 非零，advantage 为 0 | observation 接通，reward 退化 |
| progress smoke | reward_unique=2，advantage 非零 | reward 信号可用 |
| multi-update smoke | 3/3 updates 有 advantage | 信号稳定存在 |
| goal prior smoke | 采到 tissuebox | 有目标物体探索 |
| stronger smoke | reward 变差，无关动作多 | 不是简单训练量问题 |
| receptacle prior smoke | reward 改善，无关动作减少 | 出现正向行为前缀 |

---

# 关键结果：Receptacle Prior Smoke

输出目录：

```text
outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s12_smoke
```

配置：

```text
MAX_GAMES = 1
GROUP_SIZE = 4
MAX_STEPS = 12
UPDATES = 5
LR = 5e-5
GOAL_ACTION_BONUS = 1.0
```

结果：

```text
mean_rollout_reward = -0.3565
mean_irrelevant_action_count = 2.30
max_rollout_reward = 0.01
rollout_success_rate = 0.0
```

---

# 相比上一版的改善

相对 advfix smoke：

```text
mean_rollout_reward:
  -0.4525 -> -0.3565

mean_irrelevant_action_count:
  3.95 -> 2.30

sidetable action count:
  17 -> 64

tissuebox action count:
  3 -> 5
```

解释：

> 策略开始更频繁地朝目标位置移动，无关动作减少。

---

# 出现的目标行为前缀

最新 smoke 中出现了更接近成功的动作链：

```text
take tissuebox 3 from diningtable 1
go to sidetable 2
move tissuebox 3 to sidetable 2
```

这说明：

```text
Stage4-online has goal-directed behavior prefix.
```

但也要明确：

```text
rollout_success_rate = 0.0
```

所以目前还不能声称 ALFWorld task success 提升。

---

# 我现在会把结论讲得克制

我现在可以说：

> Stage4-online 已经不再是空跑；它有 post-action observation、有非退化 reward/advantage，并产生了目标行为前缀。

但我现在不能说：

> Stage4-online 已经解决 ALFWorld。

目前证据支持的是：

```text
non-degenerate online learning signal
goal-directed prefix
reduced irrelevant actions
```

还不支持：

```text
stable task success
official benchmark improvement
```

---

# 动态 Skill：为什么现在必须考虑

SkillRouter 的一个工程优势：

```text
new skill text
  -> encode
  -> add to index
  -> retrieve / rerank
```

它天然接近 append-only skill registry。

如果我不能支持动态 skill，审稿人可能会问：

> 复杂的 CLSTR 是否反而绑定了固定 skill pool？

因此动态 skill 既是工程问题，也是论文防御点。

---

# CLSTR 目前能否动态加 Skill

我目前可以做到：

```text
new skill_pool.jsonl
  -> rebuild_skill_table()
  -> rerun Stage0 top-M / handoff / eval
```

但还不是真正的随时 append：

- `skill_table.E` 和 skill 数量绑定；
- `skill_bias_retr` / `skill_bias_belief` 和 skill 数量绑定；
- checkpoint shape 可能不匹配；
- top-M cache / handoff / alias groups 要更新；
- 新 skill 没有 transition/belief 轨迹监督。

---

# 我准备怎么实现动态 Skill

我建议新增：

```text
clstr/dynamic_skill_registry.py
scripts/build_dynamic_skill_registry.py
scripts/audit_dynamic_skill_registry.py
tests/test_dynamic_skill_registry.py
```

核心流程：

```text
base skill_pool
  + append new skills
  + validate schema
  + encode skill text
  + rebuild or append embeddings
  + update skill_id -> index
  + update alias/equivalent groups
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

- 新 skill 没有被 transition/belief 训练过；
- 直接用完整 CLSTR score 可能会压低新 skill；
- retrieval-first fallback 更接近 SkillRouter 的动态能力。

---

# 动态 Skill 能否达到 SkillRouter 程度

| 能力 | SkillRouter | CLSTR 当前 | CLSTR 可实现目标 |
|---|---:|---:|---:|
| append 新 skill text | 强 | 弱 | 强 |
| 不重训加入检索 | 强 | 中 | 强 |
| 单步新 skill 检索 | 强 | 中 | 接近 |
| 新 skill rerank | 强 | 中 | 需要 text fallback |
| 新 skill 多步 transition | 弱/无 | 弱 | 需要 adaptation |
| 状态化多步选择 | 弱 | 中 | 强 |

结论：

> 单步动态 skill 可以接近 SkillRouter；多步 transition 能力需要 few-shot 或 online adaptation。

---

# 我准备如何公平比较 SkillRouter 和 CLSTR

不能只报一个总分。

我会把比较拆成四层：

```text
1. 单步静态 routing
2. 同候选集 reranking
3. 多步 trajectory routing
4. 动态 skill append evaluation
```

这样才能回答：

```text
CLSTR 到底在哪些场景比 SkillRouter 有额外价值？
```

---

# 比较一：单步静态 Routing

比较对象：

```text
SkillRouter bi-encoder
SkillRouter bi-encoder + reranker
CLSTR Stage0
CLSTR Stage1
CLSTR Stage2 without history
```

指标：

```text
Hit@1
Recall@5 / @10 / @20 / @50
MRR
FC@10
```

预期：

> SkillRouter 可能很强；CLSTR 至少要接近，不能差太多。

---

# 比较二：同候选集 Reranking

目的：

> 排除 Stage0 candidate quality 差异，只看 reranking/head 本身。

设计：

```text
fixed top-M candidates
  -> SkillRouter reranker
  -> CLSTR Stage1/2 heads
```

指标：

```text
Hit@1
MRR
positive rank improvement
conditional accuracy given positive in candidates
```

---

# 比较三：多步 Trajectory Routing

这是 CLSTR 最应该赢的地方。

输入：

```text
goal + observation_t + history_t
```

SkillRouter：

```text
每步独立 retrieve/rerank
```

CLSTR：

```text
维护 m_t / belief / transition
```

指标：

```text
step-level next-skill recall
trajectory prefix accuracy
trajectory-level all-correct
state-dependent improvement
```

---

# 比较四：动态 Skill Pool

设计：

```text
训练时隐藏一批 skill
评估时 append 到 skill pool
```

比较：

```text
SkillRouter dynamic index
CLSTR dynamic registry + unseen fallback
```

指标：

```text
new_skill Hit@1 / Recall@K
old_skill performance drop
append time
index rebuild time
dynamic manifest correctness
```

---

# 我目前看到的主要风险

1. **单步 routing 不一定超过 SkillRouter**
   - 应把 SkillRouter 定位为强基线，而不是弱 baseline。

2. **online RL 还没有 task success**
   - 目前只有正向前缀，不是成功率结果。

3. **动态 skill 还没工程化**
   - 需要 append-only registry 和 old/new split metrics。

4. **论文叙事可能过度复杂**
   - Stage0/1/2/4 是实现路径，不应全部包装成贡献。

---

# 我建议把论文贡献收敛为三点

1. **Stateful Skill Memory**

```text
m_t / belief tracks task progress across steps
```

2. **Transition-aware Multi-step Routing**

```text
skill selection conditioned on history and next observation
```

3. **Dynamic / Online Adaptation**

```text
large skill pool + appendable skills + online feedback
```

这样比“我做了很多 stage”更容易说服审稿人。

---

# 我下一步的短期实验

不要直接 full RL。

我建议先做小成本诊断：

```text
MAX_GAMES = 1
GROUP_SIZE = 4
UPDATES = 5
MAX_STEPS = 20
```

看是否能从：

```text
move one tissuebox to sidetable
```

推进到：

```text
two tissueboxes / complete success
```

如果仍卡住，诊断 carrying state、placement reward 和 STOP。

---

# 我下一步的工程建设

实现动态 skill registry：

```text
append new skill
encode text
update embedding/index
invalidate cache
evaluate old/new split
```

最低验收：

- 新 skill 能进入检索；
- append 后 old skill 性能不显著下降；
- 输出 dynamic registry manifest；
- 支持 no-retrain eval；
- 能和 SkillRouter dynamic index 对比。

---

# 我下一步的对比实验矩阵

| 维度 | SkillRouter | CLSTR | 目的 |
|---|---|---|---|
| 单步 routing | bi-encoder/reranker | Stage0/1/2 | 看基础能力 |
| 同候选 rerank | reranker | heads/transition | 看排序能力 |
| 多步 routing | per-step retrieve | stateful belief | 看 CLSTR 核心 |
| 动态 skill | append index | dynamic registry | 看工程泛化 |
| online RL | optional | Stage4-online | 看交互收益 |

---

# 我今晚可以这样开场

> 我目前还不是要说 CLSTR 已经全面超过 SkillRouter，而是想说明：我已经把系统推进到了可以公平回答这个问题的阶段。
>
> SkillRouter 在单步大池检索上非常强；CLSTR 要证明的是状态化、多步和 online adaptation 的额外价值。

---

# 我今晚可以这样讲结果

> 最新 ALFWorld online smoke 还没有 task success，但已经出现了正向行为前缀：模型能拿到 tissuebox，并移动到 sidetable。
>
> 相对上一版，平均 reward 从 -0.4525 提升到 -0.3565，无关动作从 3.95 降到 2.30。

---

# 我今晚可以这样讲风险

> 当前风险是，单步 routing 上 SkillRouter 可能仍然更强。
>
> 所以我下一步不会只报一个混合总分，而是把比较拆成单步、同候选 rerank、多步 trajectory 和动态 skill 四层。

---

# 我今晚可以这样讲动态 Skill

> SkillRouter 天然支持动态 skill，因为新 skill 只要编码后加入索引即可。
>
> CLSTR 目前支持重建 skill pool，但还不是成熟的 append-only registry。
>
> 下一步我会补 dynamic skill registry，新 skill 先走 retrieval-first fallback，transition/belief 的收益通过 few-shot 或 online adaptation 获得。

---

# 如果被问：现在比 SkillRouter 好吗？

我会回答：

> 目前不能这么说。
>
> SkillRouter 在单步 full-text retrieval/reranking 上是强基线。
>
> CLSTR 的价值要看多步、状态化和 online adaptation。
>
> 我正在把实验拆成四层来验证，而不是用一个总分混过去。

---

# 如果被问：为什么不直接照搬 SkillRouter？

我会回答：

> Stage0/Stage1 可以吸收 SkillRouter 的设计，例如 full-text embedding、top-M handoff 和 listwise reranking。
>
> 但 SkillRouter 本身没有维护 m_t / belief / transition，它每一步更接近独立检索。
>
> CLSTR 的目标是在强检索基础上做 closed-loop routing。

---

# 如果被问：Stage4-online 为什么还没成功？

我会回答：

> 一开始是 reward/advantage 退化，后面是探索太弱。
>
> 现在已经出现目标前缀，但 ALFWorld 的 pick-two-object task 需要更长链条。
>
> 下一步我不会直接 full，而是先诊断能否完成第二个 object 和 placement。

---

# 结论

我现在可以说：

```text
CLSTR pipeline 已经完整打通。
Stage4-online 有非退化训练信号。
模型已经出现目标行为前缀。
```

我现在不能说：

```text
CLSTR 已经超过 SkillRouter。
CLSTR 已经解决 ALFWorld。
```

我的下一步核心：

```text
用 SkillRouter 作为强单步基线，
在多步 trajectory、动态 skill pool 和 online adaptation 上证明 CLSTR 的额外价值。
```
