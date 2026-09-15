# CLSTR 当前成果、风险与下一阶段展望

日期：2026-06-10

用途：今晚组会/讨论汇报材料。本文档的口径是“阶段性成果 + 关键风险 + 下一步验证路线”，不是最终论文结果总结。

## 1. 一句话总结

我们已经把 CLSTR 从一个静态 routing 设想推进成了一个可训练、可评估、可接 online ACT/RL 的完整 skill-routing pipeline。当前最重要的结论不是“CLSTR 已经超过 SkillRouter”，而是：

> SkillRouter 是非常强的单步 full-text retrieve-and-rerank 基线；CLSTR 的价值需要在多步、状态化、online adaptation 和动态 skill pool 场景中证明。

当前工程上已经完成：

- unified skill pool 和多数据源训练集构建；
- Stage0 coarse retrieval；
- Stage1 heads / reranking 初始化；
- Stage2 CLSTR base 训练；
- Stage4 offline ACT；
- ALFWorld online HRPO / Stage4-online 初步闭环；
- post-action observation、reward/advantage、goal/receptacle exploration prior 等关键问题修复；
- 与 SkillRouter 比较所需的初步对照思路。

当前尚未完成：

- 官方下游 task-success benchmark 的完整正结果；
- 动态 append-only skill registry；
- CLSTR 对 SkillRouter 的系统性、多层对比；
- online RL 从“正向行为前缀”推进到稳定 task success。

## 2. 推荐汇报主线

建议今晚汇报时采用以下叙事：

1. 大规模 skill routing 里，SkillRouter 已经证明 full-text bi-encoder + reranker 非常强。
2. 我们不应把 CLSTR 讲成“更复杂的单步检索器”，否则正面打 SkillRouter 很吃亏。
3. CLSTR 的合理定位是 closed-loop skill router：

```text
goal / observation / history
  -> belief / m_t
  -> skill or action selection
  -> environment feedback / next observation
  -> belief update
  -> next skill selection
```

4. 目前工程 pipeline 已经打通，但实验发现了几个关键问题：

- 单步 routing 上 SkillRouter 很强；
- CLSTR 的 Stage4 offline 没有显著提升；
- online HRPO 一开始没有有效 reward/advantage；
- ALFWorld online 探索太弱；
- 动态 skill pool 能力还不完整。

5. 我们已经修复了一批会让方法“跑了也没意义”的问题，现在 online CLSTR 至少出现了正向行为前缀。
6. 下一阶段的目标不是盲目 full training，而是做出可以说服审稿人的比较：

- 单步 routing 是否接近强 SkillRouter；
- 同候选集下 CLSTR 是否有 reranking/transition 增益；
- 多步 trajectory routing 是否优于每步独立检索；
- 动态新增 skill 时 CLSTR 是否还能工作；
- online ACT/RL 是否能把正向前缀推进到 task success。

## 3. 当前 pipeline 状态

当前主线可以概括为：

```text
data / skill pool construction
  -> Stage0 coarse retrieval
  -> Stage1 heads init / candidate reranking
  -> Stage2 CLSTR base
  -> Stage4 offline ACT
  -> Stage4-online / ALFWorld HRPO viability
```

### 3.1 Stage0 / Stage1 / Stage2

当前主线已经从早期 v4.2 progressive 回到更稳定的 v4.1b conservative-listwise 路线：

```text
Stage2 checkpoint:
outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
```

这个 checkpoint 后续被 Stage4-online ALFWorld smoke 使用，且保持 unified large skill pool：

```text
skill pool:
data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl

skill_count:
37615
```

### 3.2 Stage4 offline ACT

Stage4 offline ACT 已经完成过训练和 gate，但目前观察到：

- 它证明了 Stage4 artifact 和训练链路可用；
- 但没有明显超过 Stage2；
- 因此它还不是论文中“ACT 带来显著收益”的强证据。

这也是我们后来转向 online HRPO / Stage4-online 的原因：如果 offline ACT 只是拟合 next-skill ranking，而没有真实环境反馈，可能难以证明 closed-loop 价值。

## 4. Stage4-online 当前进展

### 4.1 已修复的问题

最初的 ALFWorld online HRPO 有几个严重问题：

1. 只优化 policy loss，没有把 `env.step(action_t)` 后得到的 `obs_{t+1}` 接回 transition/belief。
2. reward 太稀疏，不同失败轨迹 reward 一样，导致 advantage 为 0。
3. 即使有 reward variation，策略也经常执行无关动作，例如拿 pillow、laptop、pen。
4. rollout-level advantage 被按 step 数二次缩小，长 rollout 下 policy gradient 被严重削弱。
5. goal-action prior 只偏向目标物体，不偏向目标 receptacle/navigation。

目前已修复：

- rollout step 记录 `observation_text`、`next_observation_text`、`next_admissible_actions`；
- 动作选择前不使用未来 observation，报告中固定 `pre_action_next_observation_leakage=false`；
- 增加 post-action observation auxiliary loss；
- 增加 progress reward、goal interaction reward、irrelevant action penalty；
- 修复 advantage scaling；
- goal-action prior 现在同时覆盖目标物体和目标 receptacle/location。

### 4.2 关键 smoke 结果

#### Observation smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_observation_smoke
post_action_observation_used = true
pre_action_next_observation_leakage = false
online_observation_aux_count = 10
reward_unique_count = 1
updates_with_nonzero_advantage = 0
```

结论：observation 链路接通，但 reward/advantage 退化。

#### Progress reward smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_progress_smoke
reward_unique_count = 2
updates_with_nonzero_advantage = 1
hrpo_policy_loss = -0.0372
viability.status = ok
```

结论：reward/advantage 不再退化，但 action quality 仍弱。

#### Multi-update progress smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_progress_update_smoke
rollout_count = 12
rollout_step_count = 96
reward_unique_count = 9
updates_with_nonzero_advantage = 3
online_observation_aux_count = 96
viability.status = ok
```

结论：训练信号稳定存在，但仍大量无关动作。

#### Goal prior smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_smoke
GOAL_ACTION_BONUS = 1.0
mean_goal_interaction_points = 0.1667
rollout_success_rate = 0.0
```

出现过目标动作：

```text
take tissuebox 2 from diningtable 1
```

结论：goal prior 能让策略采到目标物体，但不稳定。

#### Stronger smoke with longer rollout

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_smoke
MAX_STEPS = 12
UPDATES = 5
LR = 5e-5
rollout_success_rate = 0.0
mean_rollout_reward = -0.4525
mean_irrelevant_action_count = 3.95
```

结论：更长 rollout 没有自然变好，说明问题不是简单“训练量不够”。

#### Advantage scaling fix smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_advfix_smoke
mean_rollout_reward = -0.4525
mean_irrelevant_action_count = 3.95
rollout_success_rate = 0.0
```

修复后 policy loss 放大，但 20 条 action trace 与修复前完全一致，最大动作概率变化约 `3.3e-4`。

结论：advantage scaling 是真实 bug，但 5 个 update 内还不足以改变采样分布。

#### Receptacle prior smoke

```text
output = outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s12_smoke
mean_rollout_reward = -0.3565
mean_irrelevant_action_count = 2.30
max_rollout_reward = 0.01
rollout_success_rate = 0.0
```

相对 advfix smoke：

```text
mean_rollout_reward:          -0.4525 -> -0.3565
mean_irrelevant_action_count:    3.95 -> 2.30
sidetable action count:             17 -> 64
tissuebox action count:              3 -> 5
```

出现了更接近成功的目标前缀：

```text
take tissuebox 3 from diningtable 1
go to sidetable 2
move tissuebox 3 to sidetable 2
```

结论：Stage4-online 已经从“没有训练信号”推进到“能产生目标前缀并减少无关动作”。但成功率仍为 0，不能声称 ALFWorld task success 已经提升。

## 5. 当前和 SkillRouter 的关系

### 5.1 SkillRouter 的强点

SkillRouter 论文的核心是 full-text retrieve-and-rerank：

```text
query
  -> bi-encoder retrieve top-20 from ~80K skill pool
  -> cross-encoder rerank top-20
```

关键特点：

- 使用完整 skill text，而不是只用 name/description；
- bi-encoder 负责大池 recall；
- cross-encoder 负责 top candidates 内的精细区分；
- 使用 false-negative filtering 处理近重复 skill；
- 使用 listwise reranking loss；
- 工程上天然接近动态 skill pool：新 skill 只要编码并加入索引，就可以参与检索。

因此，在单步大池 routing 上，SkillRouter 是非常强的基线。

### 5.2 我们不能怎么讲

不建议这样讲：

> CLSTR 是一个更复杂的 SkillRouter，所以应该直接超过 SkillRouter。

原因：

- SkillRouter 专门优化单步 full-text retrieval/reranking；
- CLSTR 多了 belief/transition/online ACT，如果只做单步检索，复杂性反而可能变成负担；
- 当前结果还没有证明 CLSTR 在单步检索上超过 SkillRouter。

### 5.3 我们应该怎么讲

建议口径：

> SkillRouter 是强单步 skill retriever。CLSTR 的目标不是简单替代它，而是在其单步检索能力之上，引入状态化 belief、transition 和 online adaptation，使 skill selection 能在多步环境交互中持续更新。

更具体地说：

```text
SkillRouter:
  good at q -> skill

CLSTR:
  target q, obs_t, history_t, m_t -> skill_t -> obs_{t+1} -> m_{t+1}
```

所以 CLSTR 应该在以下场景证明价值：

- 多步 trajectory routing；
- state/history-dependent skill selection；
- online adaptation；
- dynamic skill pool 下的新 skill 使用；
- task success，而不只是单步 recall。

## 6. 动态 skill pool 问题

### 6.1 SkillRouter 为什么天然支持动态 skill

SkillRouter 的架构是：

```text
skill text -> embedding/index -> retrieve -> rerank
```

新 skill 加入时：

1. 准备 skill text；
2. 用 bi-encoder 编码；
3. 加入 ANN / vector index；
4. reranker 在 top-K 中读取该 skill text；
5. 不一定需要重训模型。

因此它工程上非常适合 append-only skill registry。

### 6.2 CLSTR 当前能做到什么

CLSTR 当前可以做到“离线重建 skill pool”：

```text
new skill_pool.jsonl
  -> rebuild_skill_table()
  -> rerun Stage0 top-M / handoff / evaluation
```

但还不是成熟的“随时 append skill”：

- `skill_table.E` 与 skill 数量绑定；
- `skill_bias_retr` / `skill_bias_belief` 与 skill 数量绑定；
- checkpoint 里的 skill table shape 可能和新 pool 不匹配；
- Stage0 top-M cache、handoff candidates、alias groups 都需要重建；
- 新 skill 没有 Stage1/Stage2/Stage4 轨迹监督，transition/belief 对它只能 zero-shot 泛化。

### 6.3 我们可以怎么实现动态 skill

建议实现一个 `DynamicSkillRegistry`：

```text
base skill_pool.jsonl
  + append new skill rows
  + validate schema
  + encode new skill text
  + append / rebuild skill embeddings
  + update skill_id -> index
  + update alias / equivalent groups
  + invalidate old top-M cache
  + export registry manifest
```

推理路径：

```text
query/state
  -> Stage0 retrieve over updated skill pool
  -> top-M candidates include old + new skills
  -> CLSTR Stage1/2/4 rerank or transition score
```

为了防止新 skill 被未见过的 transition head 压下去，应加一个 unseen-skill fallback：

```text
old skill:
  score = retrieval + policy/head + transition/belief

new/unseen skill:
  score = retrieval-heavy + text-rerank-heavy + weaker transition prior
```

可以在 report 中显式写：

```text
new_skill_count
old_skill_count
new_skill_recall@K
old_skill_recall@K
new_skill_score_blend_mode
dynamic_registry_manifest
```

### 6.4 能不能达到 SkillRouter 程度

分层回答：

| 能力 | SkillRouter | CLSTR 当前 | CLSTR 可实现目标 |
|---|---:|---:|---:|
| append 新 skill text | 强 | 弱 | 强 |
| 不重训加入检索 | 强 | 中 | 强 |
| 单步新 skill 检索 | 强 | 中 | 接近 SkillRouter |
| 新 skill rerank | 强 | 中 | 需要 text-rerank fallback |
| 新 skill 多步 transition | 弱/无 | 弱 | 需要 few-shot / online adaptation |
| 状态化多步选择 | 弱 | 中 | 强 |

结论：

> CLSTR 可以做到接近 SkillRouter 的动态 skill engineering，但不能声称新 skill 一加入就拥有完整 transition/belief 能力。合理做法是 retrieval-first fallback + optional online/few-shot adaptation。

## 7. 应该如何公平比较 CLSTR 和 SkillRouter

不能只做一个总分表。建议拆成四层比较。

### 7.1 单步静态 routing

目的：确认 CLSTR 在最基础的大池检索上是否接近强基线。

比较对象：

```text
SkillRouter bi-encoder
SkillRouter bi-encoder + reranker
CLSTR Stage0
CLSTR Stage1
CLSTR Stage2 score without history
```

指标：

```text
Hit@1
Recall@5 / @10 / @20 / @50
MRR
FC@10 for multi-skill queries
```

预期：

- SkillRouter 可能很强；
- CLSTR 不一定要在这里超过它；
- 但至少不能差太多，否则后续多步价值难以建立。

### 7.2 同候选集 reranking

目的：排除 Stage0 candidate quality 差异，看 reranker/head 本身。

设计：

```text
固定同一个 top-M candidate list
  -> SkillRouter reranker rerank
  -> CLSTR Stage1/2 heads rerank
```

指标：

```text
Hit@1
MRR
positive rank improvement
candidate coverage conditional accuracy
```

关键问题：

> 在相同候选集下，CLSTR 的 learned heads / transition prior 是否比纯 text reranker 更好？

### 7.3 多步 trajectory routing

目的：证明 CLSTR 的核心卖点。

设计：

```text
每一步输入:
goal + observation_t + history_t

SkillRouter:
每步独立 retrieve/rerank

CLSTR:
维护 m_t / belief / transition
```

指标：

```text
step-level next_skill recall
trajectory prefix accuracy
trajectory-level all-correct / FC
switch/self stability
state-dependent improvement
```

关键问题：

> CLSTR 是否能利用 history / observation / transition，在多步选择中超过每步独立 SkillRouter？

### 7.4 动态 skill pool

目的：回应“SkillRouter 可以随时补 skill，我们能不能”的工程问题。

设计：

```text
训练时隐藏一批 skill
评估时 append 到 skill pool
比较 append 前后：
  old skill stability
  new skill recall
  dynamic pool latency
  no-retrain usability
```

指标：

```text
new_skill Hit@1 / Recall@K
old_skill performance drop
append time
index rebuild time
dynamic manifest correctness
```

关键问题：

> CLSTR 是否可以在不重训的情况下使用新增 skill？新增 skill 是否至少能在单步 retrieval-first fallback 下被召回？

## 8. 当前风险

### 8.1 单步 routing 不一定超过 SkillRouter

这是最大现实风险。SkillRouter 的方法非常直接，而且专门为这个目标优化。CLSTR 如果只在单步 retrieval 上比较，可能没有优势。

应对方式：

- 把 SkillRouter 定位为强单步基线；
- 把 CLSTR 的贡献放在多步 / closed-loop / online adaptation；
- 如果可能，直接使用 SkillRouter-style Stage0/Stage1 作为 CLSTR 的强初始化。

### 8.2 online RL 仍未成功

最新 ALFWorld online smoke 出现了目标前缀，但成功率仍为 0。

当前证据只能支持：

```text
Stage4-online has non-degenerate learning signal and can produce goal-directed prefixes.
```

不能支持：

```text
Stage4-online solves ALFWorld.
```

下一步应继续小成本诊断，而不是直接 full RL。

### 8.3 动态 skill 还没有工程化

目前只能说“架构上可以支持”，不能说“已经实现”。必须补：

- append-only skill registry；
- cache invalidation；
- checkpoint shape mismatch handling；
- dynamic skill audit；
- old/new skill split metrics。

### 8.4 论文叙事不能过度复杂

CLSTR 已经包含：

- Stage0；
- Stage1；
- Stage2；
- Stage4 offline；
- online HRPO；
- dynamic skill；
- AppWorld / ALFWorld / ToolBench / TrajectBench。

如果叙事不收敛，容易显得工程堆砌。建议核心贡献只保留三点：

1. stateful skill memory / belief；
2. transition-aware multi-step routing；
3. dynamic / online adaptation on top of large skill pools。

## 9. 下一阶段建议目标

### 9.1 短期目标

不建议马上 full RL。建议先做：

1. `MAX_STEPS=20` 的 ALFWorld 小 smoke；
2. 检查是否能从一个 tissuebox 扩展到两个 tissuebox；
3. 如果仍卡住，诊断 carrying/inventory state、placement reward 和 STOP；
4. 不要在没有成功前缀扩展的情况下扩大训练规模。

### 9.2 中期目标

实现动态 skill registry：

```text
scripts/build_dynamic_skill_registry.py
clstr/dynamic_skill_registry.py
scripts/audit_dynamic_skill_registry.py
tests/test_dynamic_skill_registry.py
```

最低验收：

- 能 append 新 skill；
- 能重建或增量更新 skill embeddings；
- 能输出 manifest；
- 能在 Stage0 eval 中召回新 skill；
- 能报告 old/new skill 分开指标。

### 9.3 对比实验目标

至少做四组：

```text
1. SkillRouter static routing baseline
2. CLSTR static routing
3. SkillRouter per-step routing on trajectories
4. CLSTR stateful trajectory routing
```

如果时间允许，再加：

```text
5. Dynamic skill append eval
6. Online ACT/RL small-task success eval
```

## 10. 今晚可以直接使用的表达

### 10.1 开场

> 我们目前不是已经证明 CLSTR 全面超过 SkillRouter，而是把系统推进到了可以公平回答这个问题的阶段。SkillRouter 在单步大池检索上非常强；CLSTR 要证明的是状态化、多步和 online adaptation 的额外价值。

### 10.2 讲成果

> 工程上，我们已经完成了从 unified skill pool、Stage0/1/2 到 Stage4 offline/online 的完整训练链路。最近重点修复了 online HRPO 中几个会导致训练无意义的问题，包括 post-action observation 接入、reward/advantage 退化、advantage scaling 和目标位置探索。

### 10.3 讲结果

> 最新 ALFWorld online smoke 还没有 task success，但已经出现了正向行为前缀：模型能拿到 tissuebox，并移动到 sidetable。相对上一版，平均 reward 从 -0.4525 提升到 -0.3565，无关动作从 3.95 降到 2.30。

### 10.4 讲风险

> 当前风险是，单步 routing 上 SkillRouter 可能仍然更强；CLSTR 的优势必须在多步 trajectory、动态 skill 和 online adaptation 中体现。我们下一步会按这几个维度拆开比较，而不是只报一个混合总分。

### 10.5 讲动态 skill

> SkillRouter 天然支持动态 skill，因为新 skill 只要编码后加入索引即可。CLSTR 目前还只是支持重建 skill pool，下一步要实现 append-only dynamic registry。新 skill 会先走 retrieval-first fallback，transition/belief 的收益需要少量轨迹或 online adaptation。

### 10.6 讲下一步

> 下一阶段我们会先补动态 skill registry 和 SkillRouter 分层对比，再继续小成本推进 online ACT/RL。我们的目标不是烧大作业，而是先证明 CLSTR 在 SkillRouter 不擅长的多步和状态化场景里确实有增益。

## 11. 如果被问到的问题

### Q1：CLSTR 现在比 SkillRouter 好吗？

建议回答：

> 目前不能这么说。SkillRouter 在单步 full-text retrieval/reranking 上是强基线。CLSTR 的价值要看多步、状态化和 online adaptation。我们正在把实验拆成单步、同候选 rerank、多步 trajectory、动态 skill 四层来验证。

### Q2：为什么不直接照搬 SkillRouter？

建议回答：

> Stage0/Stage1 可以吸收 SkillRouter 的设计，比如 full-text embedding、top-M candidate handoff、listwise reranking。但 SkillRouter 本身没有维护 m_t / belief / transition，它每一步更接近独立检索。CLSTR 的目标是在强检索基础上做 closed-loop routing。

### Q3：动态 skill 我们能做吗？

建议回答：

> 可以做，但要诚实区分。单步检索层可以接近 SkillRouter：append skill、编码、更新索引即可。CLSTR 的 transition/belief 对新 skill 需要 zero-shot fallback 或后续少量 adaptation。因此我们会实现 dynamic registry，并报告 old/new skill 分开指标。

### Q4：Stage4-online 为什么还没成功？

建议回答：

> 一开始是 reward/advantage 退化，后面是探索太弱。现在已经出现目标前缀，但 ALFWorld 的 pick-two-object task 需要更长链条。下一步不会直接 full，而是诊断能否完成第二个 object 和 placement。

### Q5：论文会不会显得太工程化？

建议回答：

> 我们需要把叙事收敛到三个核心：stateful skill memory、transition-aware multi-step routing、dynamic/online adaptation。Stage0/Stage1/Stage2/Stage4 是实现路径，不应该都作为独立贡献。

## 12. 建议今晚结论页

可以用下面几句话收尾：

> 当前 CLSTR 已经具备完整 pipeline，且 online Stage4 不再是空跑：它有 post-action observation、非退化 reward/advantage，并产生了目标行为前缀。
>
> 但我们还没有证明 task success，也没有证明单步超过 SkillRouter。
>
> 下一步的核心是把比较做对：用 SkillRouter 作为强单步基线，在多步 trajectory、动态 skill pool 和 online adaptation 上证明 CLSTR 的额外价值。

