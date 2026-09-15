# CLSTR v4 Plan — Trajectory-Aware Skill Routing Layer

定稿日期: 2026-05-25
基于: 2026-05-25 ~20 轮 design-choice 讨论的最终结论
取代: goal v3 (SKILLRET 主战场) 与所有更早的 framing

---

## 1. 战略锁定

**核心 reposition**: CLSTR 重定位为 **"Trajectory-aware Skill Routing Layer for Sequential Tool-Calling Agents"**。

### 主战场（§15 主表三个并列）

| 角色 | benchmark | 主指标（各报原生 metric） |
|---|---|---|
| **§15 主表 #1** | TRAJECT-Bench (parallel + sequential) | EM / Inclusion / Usage / Traj-Satisfy / Acc |
| **§15 主表 #2** | ToolBench-G3 (intra-collection multi-tool) | pass rate (LLM-judge via StableToolBench simulation) |
| **§15 主表 #3** | ToolRet | NDCG@5/10, Recall@5/10, MAP@10 |
| **§15.5.5** | AppWorld dev57 combination plot | token cost ↓ + task success 不退步 |
| **case study only** | ALFWorld / ScienceWorld / SKILLRET | 不进主表 |

### 成功条件（任一达成即可主张论文）

1. **TRAJECT-Bench sequential routing recall@5** 超过 ToolBench_IR_bert + e5-mistral-7b 两个 baseline ≥ 3 pp
2. **TRAJECT-Bench sequential Acc** 比同 ckpt 在 parallel 上 Acc 高 ≥ 5 pp（**self-contained ablation 直接证明 belief filter 仅在 sequential 上贡献——最核心论点**）
3. **ToolBench-G3 routing recall@5** 超过 ToolRetriever ≥ 2 pp
4. **AppWorld combination plot**: CLSTR + Qwen 比 dense retrieval + Qwen 在 token cost 上降 ≥ 30% 且 task success 不退步

### 强制约束

- 76632 已停（08:11:04，cancelled at update 86），结果归档不再 follow-up
- 全程 leakage audit: 三个 benchmark 的 test query / task ids 禁入训练
- L_act 不使用任何 dev/test signal；Stage 4 改为跨 benchmark 的
  transition-conditioned next-skill training，AppWorld verified_pairs 仅作小规模
  domain adaptation / §15.5.5 secondary eval
- 重要进展写入 `docs/clstr_v4_plan.md`（本文件）+ `description.md` Step 进度更新
- 论文正文等所有 stage 验收后再写

### 2026-06-09 Stage4-online observation 修复记录

Stage4-online 不再只被表述为 policy-only admissible-action HRPO。当前修复把 ALFWorld rollout 中 `env.step(action_t)` 返回的 `obs_{t+1}` 作为后验训练信号接入：

- 决策前输入保持 `goal + obs_t + history + admissible_actions_t`，禁止使用未来 observation，报告字段固定 `pre_action_next_observation_leakage=false`。
- rollout step schema 记录 `observation_text`、`next_observation_text`、`next_admissible_actions` 和环境反馈字段。
- online 训练 loop 在 HRPO policy loss 之外增加 observation-connected auxiliary loss：`m_t + action_t + obs_{t+1}` 对齐 detached next-observation skill-subspace memory，用于训练 transition/belief consistency。
- 该修复不制造在线 ALFWorld `next_skill_id` 伪标签；有 expert/replay next-skill 的离线 Stage4 仍走原 transition-conditioned next-skill CE，纯 online 阶段先用 observation-belief consistency。

### 2026-06-09 Stage4-online reward viability 修复记录

Observation smoke 证明后验 loss 已接通，但两个不同 action trace 得到完全相同 reward，导致 `updates_with_nonzero_advantage=0`。根因是 ALFWorld online reward 太稀疏，原实现无法奖励“发现目标物体/目标位置”等中间进展。

当前修复只作用于 ALFWorld online HRPO viability，不改变 Stage0/1/2/离线 Stage4 主线：

- 从 goal 和 post-action observation/action 中估计通用 progress points；
- 目标物体权重大于 receptacle/location，避免只到达容器位置就压过发现目标物体；
- reward report 记录 `progress_points`、`progress_reward`、`mean_progress_points`、`progress_reward_unique_count`，便于判断 advantage 是否来自可解释的进展信号；
- 旧 smoke 离线重算已能把 `-0.15/-0.15` 的 reward tie 打开为 `-0.10/-0.05`。
- 低成本 progress smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_progress_smoke` 已通过：A800 job `86218`，elapsed `00:02:25`，`reward_unique_count=2`、`updates_with_nonzero_advantage=1`、`hrpo_policy_loss=-0.0372`、`online_observation_aux_count=10`、`viability.status=ok`。这证明 Stage4-online 现在能产生正向训练信号；仍需后续更大但受控的 smoke/评估证明 task success 或 action quality 改善。

---

## 2. 最终决策汇总表（按 ID 索引）

| ID | 决策点 | 选择 |
|---|---|---|
| **战略** | | |
| D3 | 主战场 | TRAJECT + ToolBench + ToolRet **三个并列** |
| D10 | 论文 framing 名词 | **"Trajectory-aware skill routing layer"** |
| D17 | 主表 metric | **各报原生 benchmark metric** |
| **架构改造** | | |
| D1 | m_t 重定义方向 | 保留 m_t，**扩 vocab 信息容量** |
| D11-v2 | vocab 去重策略 | **Partial dedup**（entity resolved，多描述拼接保留） |
| D15-v2 | partial dedup 方法 | **Two-pass: rule (name normalize + param sig hash) → LLM 补漏 borderline** |
| D4 | F1 logits 修复 | **全套 SkillRouter-base 同构**（L2 norm + learnable scale + skill_bias） |
| F2-v2 | τ_s / logit scale 处理 | **retrieval scale 与 belief scale 分离**：SkillRouter-like retrieval logits 用 `logit_scale_retr`，`subspace_obs` 单独用 `logit_scale_belief` / `tau_belief` |
| D5 | F4 SkillHead 输入 | **`[u, m_t - h_t, u * (m_t - h_t)]`** 用 belief residual |
| D6 | F5 action_emb 共享 | **软共享 `action_proj(E)`** |
| D6 fol | action_proj d_a | **d_a = d = 1024**（action 与 state 同维） |
| **训练** | | |
| D8-v2 | Stage 数 | **4 stage**（论文 §10.1 原划分） |
| D2 | HRPO 退化修复 | **Routing freeze + 升 β** |
| D14 | routing freeze 实现 | **硬冻 SkillTable retrieval path**（`W / E / logit_scale_retr / skill_bias_retr`），belief scale/bias 是否 low-lr 更新必须作为 ablation |
| D13 | β_routing_prior 值 | **0.15**（中拉回） |
| D18 | Stage 4 L_act 范围 | **不限 AppWorld**，定义为 transition-conditioned next-skill training；TRAJECT sequential / ToolBench simulated / AppWorld verified train pairs 都可进入 |
| G1 | candidate sampling | **top-M widen (M=K×4) + multinomial 不放回** |
| G2 | cross_encoder 性能 | **per-step 缓存**（key = `(state_text_hash, skill_idx)`） |
| F7 | SkillHead Transformer 化 | **defer**，K≥50 大池场景再考虑 |
| **节奏** | | |
| D9 | Phase A 实施范围 | **F1+F2-v2+G1 一并做** |

---

## 3. 架构图（v4 修复后）

```
input: q + x_t (状态文本)
   │
   ▼
StateEncoder([q; x_t])              # SkillRouter-Emb-0.6B frozen, proj=identity
   │
   ▼ h_t ∈ R^d (d=1024)
   │
   ├──→ SkillTable.retrieval_logits(h_t):                                  ← F1 修复
   │       q_norm = L2normalize(W(h_t))                                       
   │       e_norm = L2normalize(E)
   │       ℓ_retr = exp(logit_scale_retr) * (q_norm @ e_norm.T)
   │                + skill_bias_retr
   │       (Stage 3: 冻 W / retrieval scale / retrieval bias / E)           ← D14 修复
   │
   ▼ ℓ_retr ∈ R^N
   │
   ├──→ subspace_obs:
   │      ℓ_belief = exp(logit_scale_belief) * (q_norm @ e_norm.T)
   │                 + skill_bias_belief
   │      p_t = softmax(ℓ_belief)                                         ← F2-v2: belief scale 单独控制
   │      m̃_t = p_t @ E
   │
   ├──→ TransitionPredictor:
   │      m̂_t = GRUCell(m_{t-1}, cat[action_proj(E)[a_{t-1}],              ← F5/D6 软共享
   │                                  obs_proj(o_{t-1})])
   │
   ├──→ BeliefGate:
   │      γ_t = σ(W_γ[m̂_t, m̃_t, o_emb])
   │      m_t = γ_t · m̃_t + (1-γ_t) · m̂_t
   │
   ├──→ candidate sampling:                                                 ← G1 修复
   │      train: top-M(M=K×4) + multinomial(K, no-replacement)
   │      eval:  deterministic top-K
   │
   ├──→ batch_cross_encode([state], [candidates]):                         ← G2 修复
   │      per-step cache by (state_text_hash, skill_idx)
   │
   ▼ candidate_embs (u) ∈ R^{K×d}
   │
   ├──→ SkillHead:                                                          ← F4 修复
   │      feat = cat([u, m_t - h_t, u * (m_t - h_t)])
   │      o_skill = MLP(d×3 → d → 1)
   │
   ├──→ StopHead:
   │      o_stop = MLP(cat[h_t, m_t])
   │
   ▼
policy_logits = cat([o_skill, o_stop]) ∈ R^{K+1}
   │
   ▼ multinomial sample
selected_action_local ∈ [0, K]
   │
   ├──→ skill execute → outcome reward
   └──→ Stage 4: TransHead(m̂_{t+1}, action_proj(E)[c_i]) → L_act
        # generic transition-conditioned next-skill objective, not AppWorld-only
```

---

## 4. 数据准备 (Phase B-C)

### 数据源 inventory

| Source | 角色 | Stage | 备注 |
|---|---|---|---|
| TRAJECT-Bench parallel/sequential | 主表 #1 训练 + eval | 2 + 3 + eval | paper pending, 用 `public_data/`; sequential data 2025.9 才完整 |
| ToolBench-G3 + StableToolBench simulation | 主表 #2 训练 + eval | 2 + 3 + eval | 用 simulated response 避开 RapidAPI 死亡 |
| ToolRet 16k pool + ToolRet-Training-20w | 主表 #3 retrieval | 1 + eval | 多源汇集（已含 ToolBench-G3 部分 + apigen） |
| SKILLRET 127k retrieval | retrieval warmup | 1 | leakage audit 已扣 4997 test query |
| ALFWorld 109k / ScienceWorld 93k / AgentGym 32k | unified traj base | 2 | 已构建 unified v1 |
| AppWorld 618 verified pair | L_act domain adaptation + secondary eval | 4 | data/appworld_act/verified_pairs_train.jsonl |
| SkillNet 400k+ | **不进训练**（D11-v2） | 仅 OOD eval 候选 | 质量未审 |

### Partial dedup (D11-v2 + D15-v2)

```
unified vocab v2 构造流程:

For each source ∈ {TRAJECT, ToolBench-G3, ToolRet, SKILLRET}:
   For each tool entry:
      key = (normalize(tool_name), hash(param_signature))

# First-pass: rule-based merge by key
merge same-key tools across sources: 
   {
     skill_id (canonical),
     descriptions: [desc_from_source_a, desc_from_source_b, ...],
     param_schema (canonical from highest-priority source),
     provenance: [source_a, source_b, ...]
   }

# Second-pass: LLM judge for borderline cases
For each pair (entry_i, entry_j) where:
   - tool_name fuzzy similarity ∈ (0.6, 0.9)  # 边界
   - 或 param_sig partial match
   - 或 functional-description embedding cosine > 0.85
Call LLM-judge (GPT-4 / Claude): "are these the same functional tool?"
If yes: merge them; if no: keep separate.

Output: vocab v2 with ~50-150k canonical skill_ids + multi-description per id
```

### Leakage audit v2

- 跨 source 检查同 tool 是否被同时标为某 query 的 positive + 另一 query 的 negative → 标记 ambiguous, 排除
- TRAJECT/ToolBench/ToolRet 各自 dev/test split 严格隔离
- AppWorld dev/test 642 task_ids 继续排除（unified v1 已做）

### unified pretrain corpus v2

| | v1 (current) | v2 (target) |
|---|---|---|
| trajectories.jsonl | 235k | 235k + TRAJECT ~60k + ToolBench-G3 ~40k = **~335k** |
| retrieval.jsonl | 127k (SKILLRET) | 127k + TRAJECT retrieval + ToolBench retrieval + ToolRet 20w = **~350-500k** |
| skill_pool.jsonl | 188 (AppWorld 用) | **~50-150k canonical tools** (partial-dedup 后) |

---

## 5. Stage 训练流程（按 D8-v2 4 stage）

### 5.0 当前生效修订：Stage0 foundation-first pipeline

2026-06-07 后，当前主线训练流程以
`Pre-Stage0 -> Stage0 coarse retriever -> Stage1 heads init -> Stage2 CLSTR base -> Joint Stage4 ACT`
为准。旧的 `Stage 1 — L_retr warmup` 不能再直接作为进入 Stage2/Joint Stage4 的充分前置。
standalone Stage3 offline HRPO-style 仍保留为历史兼容/辅助诊断入口，但不再是
`v4.2_progressive_final` 主线进入 Stage4 的必要阶段。

- **Pre-Stage0**：完成 skill pool partial dedup、canonical skill id、multi-positive qrels
  与 leakage audit。borderline review 未完成、placeholder skill、positive 不在
  canonical pool 时，readiness gate 必须阻断。
- **Stage0 coarse retriever**：采用 SkillRouter-compatible bi-encoder 协议，只做大池
  top-M 粗召回：SkillRouter query/skill serialization、left padding、last-token
  pooling、L2 normalize、cosine scoring、identity projection / skill-table adapter init。
- **Stage0 不包含 cross-encoder/listwise reranker**。SkillRouter 的 cross-encoder 或
  listwise reranker 只作为 optional baseline / upper bound，用来解释上界，不作为
  CLSTR 主线依赖。
- **Stage1 heads init**：消费 Stage0 top-M candidate set，冻结 Stage0 retrieval
  foundation，在同一个 CLSTRModel 上初始化 policy / belief / transition / STOP heads。
  positive 不在 top-M 时必须 skip 或 deterministic inject，并记录 provenance。
- **Stage2 CLSTR base**：从 Stage1 checkpoint 继续训练同一个 CLSTRModel，进入更完整的
  supervised multi-step CLSTR-base。内部恢复 checkpoint 时可以读取 Stage0 frozen routing
  foundation，但论文和实验口径上不是两个模型相加，也不是 ensemble。
- **Joint Stage4 ACT**：直接从通过 quality gate 的 Stage2 checkpoint 初始化 ACT
  head/base 参数，训练 transition-conditioned next-skill ACT；可选加入
  `lambda_pref * L_offline_preference`。当 `lambda_pref=0` 时就是纯 Stage4 ACT；
  当 `lambda_pref>0` 时才启用 Stage3-style offline preference auxiliary。
- **Qwen 不属于主线训练**：Qwen 仅作为 AppWorld downstream executor 或 optional
  encoder ablation。CLSTR-native Stage2 默认入口是
  `scripts/run_clstr_stage2_full_base_train.py`，不是
  `scripts/run_clstr_qwen3_full_base_train.py`。

### 5.0.1 Stage0 v4 handoff-aligned fix

2026-06-02 的 Stage0 handoff audit 说明：Stage0 v3 已经包含
trajectory-derived retrieval，但在真实 Stage2 handoff query 上，TRAJECT-Bench 的
top-100/top-200 candidate coverage 仍不足。因此 v4/fix 不重做数据集，也不引入
benchmark-specific oracle candidate，而是修正 Stage0 训练采样和进入下游训练的 gate：

- Stage0 sampler 从 `source_balanced` 升级为 `handoff_balanced`；
- `handoff_balanced` 对所有 `trajectory_derived_*` source 按 `current / next`
  target 拆分训练 bucket，同时保留 SkillRET、ToolRet、ToolBench-G3、TRAJECT-Bench
  静态 retrieval bucket；
- 训练报告记录 `training_bucket_counts`，便于审计每个 source/target bucket 是否获得
  训练预算；
- Stage0 handoff audit 增加 coverage gate，至少检查 global、ToolBench-G3 和
  TRAJECT-Bench 的 `next_recall@500`，以及 TRAJECT-Bench `next_recall@200`；
- 主线评估仍使用真实 top-M candidate，不允许 eval-time gold injection。

这项修改的论文口径是：CLSTR 的连续 skill routing 需要 handoff-aligned coarse
retriever。它不是 TRAJECT-Bench 特化规则，而是对所有 sequential trajectory source
统一应用的 current/next query alignment。

进入 Stage1 的硬条件：same-pool SkillRouter frozen baseline 已生成，Stage0
coarse recall 不低于该 baseline，且 Stage0 checkpoint、training metrics、loss/recall
curve 与 quality gate 均存在并通过。进入 Stage2 的硬条件是 Stage1 heads-init
checkpoint 与 Stage1 quality gate 均通过。

### Stage 0 — Backbone + adapter init

```python
# 不训练，只构造
StateEncoder: SkillRouter-Emb-0.6B, freeze_backbone=True, proj=identity (d=1024)
SkillTable:
    W = nn.Linear(d, d, bias=False), init=identity                  ← 已有
    E = build_embeddings(skill_texts)  # ~50-150k × 1024            ← 新构造
    logit_scale_retr = nn.Parameter(torch.tensor(0.0))              ← F1/F2-v2
    skill_bias_retr = nn.Parameter(torch.zeros(N))                  ← F1/F2-v2
    logit_scale_belief = nn.Parameter(torch.tensor(init_belief))    ← F2-v2, softer than retrieval
    skill_bias_belief = nn.Parameter(torch.zeros(N))                ← F2-v2
TransitionPredictor:
    action_proj = nn.Linear(d, d, bias=False)                       ← 新增 D6
    obs_proj = nn.Linear(d, d)
    cell = nn.GRUCell(d + d, d)                                     ← d_a=d
BeliefGate / SkillHead / StopHead / TransHead: 同 v3 但 SkillHead 输入改 D5
```

### Stage 1 — L_retr warmup

```
trainable: W, logit_scale_retr, skill_bias_retr, E (D14 在 Stage 3 才冻)
optional/low-lr: logit_scale_belief, skill_bias_belief
data: SKILLRET 127k + TRAJECT retrieval + ToolBench-G3 retrieval + ToolRet train
objective: L_retr (InfoNCE on partial-dedup vocab)
batch: ~256 pair
updates: ~5000
GPU: H200 × 2 days
验收: SKILLRET test NDCG@10 ≥ 0.7014 (SkillRouter frozen baseline)
       TRAJECT-Bench dev recall@5 ≥ 0.6
       ToolRet test NDCG@10 ≥ 0.65
工程 gate: 当前实现已将本段 retrieval warmup 重新定位为 Stage0 coarse retriever。
       Stage1 前必须先运行 `scripts/audit_clstr_stage0_quality.py` 或
       `bash scripts/submit_clstr_stage1_after_stage0_gate.sh`，并得到
       Stage0 quality gate `status=ok`。该 gate 检查最终 checkpoint、
       `training_metrics.jsonl`、`loss_curve.svg`、Recall@20/50/100、query
       source 覆盖，以及 same-pool SkillRouter frozen baseline。它只证明
       Stage0 coarse retriever 可以 hand off 给 CLSTR Stage1，不能替代正式
       SKILLRET/TRAJECT/ToolRet benchmark。
```

### Stage 2 — Supervised L_trans + L_policy(CE on traj)

```
trainable: 全 head + W + retrieval scale/bias + belief scale/bias (E 可选冻)
data: unified v2 traj ~335k step-pairs
objective: L_trans (cos sim 相邻 belief) + L_policy_CE (cross-entropy on traj ground-truth action) + L_retr 弱辅
batch: ~64 traj
updates: ~10000
GPU: H200 × 3 days
验收: TRAJECT dev sequential recall@5 ≥ 0.70
       ToolBench-G3 dev recall@5 ≥ 0.70
```

### Stage 3 — HRPO RL on TRAJECT + ToolBench

```
trainable: BeliefGate + SkillHead + StopHead + TransitionPredictor + action_proj
冻结: SkillTable retrieval path (W + logit_scale_retr + skill_bias_retr + E) ← D14 硬冻
可选: belief scale/bias freeze 或 low-lr 更新，必须作为 ablation 记录
data: TRAJECT-Bench train (parallel + sequential) + ToolBench-G3 train rollout
objective: L_policy_HRPO + β_routing_prior * KL(policy || supervised_routing)
β_routing_prior = 0.15                                              ← D13
candidate sampling: top-M(M=K×4) + multinomial 不放回               ← G1
cross_encoder: per-step cache                                      ← G2
group_size: 8, task_batch_size: 3, max_steps: 8 (TRAJECT 3-10 step)
updates: ~500
GPU: H200 × 3-5 days
验收: TRAJECT sequential Acc - parallel Acc ≥ +3 pp                ← 自证 belief filter
       TRAJECT recall@5 不退步 vs Stage 2
       ToolBench-G3 pass rate ≥ ToolLLaMA-2-7b
```

### Stage 4 — Transition-conditioned next-skill L_act

```
trainable: TransHead 主 (with spectral norm) + 少量 head fine-tune
冻结: SkillTable retrieval path + BeliefGate 主体 (沿用 Stage 3 ckpt)
data:
  primary: TRAJECT sequential train + ToolBench-G3 simulated train trajectories
  secondary: data/appworld_act/verified_pairs_train.jsonl (618 pair)
objective:
  L_act = -log P_T(a+_{t+1} | m̂_{t+1}, C_{t+1})
  可配 pairwise ranking loss: score(a+_{t+1}) > score(a-_{t+1})
输入要求:
  state/history at t, action_t, observation_t, candidate_skills_{t+1},
  positive_next_skill_{t+1}
约束:
  - 只用 train split；TRAJECT / ToolBench / AppWorld dev/test 禁入
  - TransHead.use_sn = True
  - AppWorld m_t exact replay, allow_approximate_m_t = False
  - C_{t+1} 必含 a+_{t+1}, 否则 skip 或 deterministic inject 并打 provenance
updates: ~1000-3000 (按 primary 数据规模调整)
GPU: A800/H200 × 1-2 days
验收:
  - TRAJECT sequential next-skill acc / recall@K 高于 Stage 3 ckpt
  - ToolBench-G3 next-tool selection 不退步
  - AppWorld dev57 verified-pair action acc 高于 Stage 3 ckpt，用于 §15.5.5 combination plot
```

---

## 6. Phase 化执行计划

| Phase | 内容 | 时长 | 阻塞 | 验收 gate |
|---|---|---|---|---|
| **A. 实现层修复** | F1+F2-v2+G1 + 软共享 action_proj + belief residual + 单元测试 | 3-4 天 | — | pytest 全过；从 Stage 0 用新 logits supervised 重训，AppWorld dev57 routing recall@1 ≥ 0.95 |
| **B. 数据准备** | TRAJECT + ToolBench-G3 + ToolRet 拉数据 + schema 转换 + leakage audit | 5-7 天 | 镜像 ghfast.top 可用 | 3 个数据 manifest + leakage audit 报告通过 |
| **C. partial dedup vocab + unified v2** | rule + LLM two-pass dedup; merge 4 source → unified v2 | 2-3 天 | B 完成 | vocab 50-150k canonical skill, smoke 1k subset 训练 loss 下降 |
| **D. Pipeline 重构** | Stage 0-4 接通新 vocab + sbatch | 3-4 天 | A+C 完成 | smoke training 跑通 |
| **E. Stage 1+2 supervised** | unified v2 supervised pretrain | 3-5 天 GPU | D 完成 | 三 benchmark dev 全部 ≥ baseline |
| **F. Stage 3 HRPO** | TRAJECT + ToolBench RL with D14 freeze | 5-7 天 GPU | E 完成 | TRAJECT sequential Acc - parallel Acc ≥ +3 pp |
| **G. Stage 4 L_act** | 跨 benchmark transition-conditioned next-skill 训练 + AppWorld 小规模适配 | 1-2 天 GPU | F 完成 | TRAJECT/ToolBench next-skill 不退步且 AppWorld action acc 提升 |
| **H. Evaluation matrix** | 3 主表 + ablation + combination plot 全跑 | 2-3 天 GPU | F+G 完成 | §15-§17 数据完整 |
| **I. 论文回写** | §1 / §5-§8 / §10 / §13-§19 | 5-7 天 | H 数据齐 | draft 完成 |

**总时长估计**: ~4-5 周

**并行机会**:
- Phase A 与 Phase B 可同时启动
- Phase E 子任务（Stage 1 vs Stage 2 部分子集）可并行小规模 sanity
- Phase H 三主表评估可并行 GPU job

---

## 7. Fallback 矩阵

| 风险 | fallback |
|---|---|
| TRAJECT-Bench sequential data 9 月发布**仍不稳定** | 退到 ToolBench-G3 主 + TRAJECT parallel 辅；§15 主表降为 2 个 |
| ToolBench live execution 完全死 | 全程 StableToolBench simulation；明确论文 reward 是 simulated |
| F1-G2 修完后三 benchmark 仍打不过 baseline | 退到 ToolRet retrieval scaling 主表，CLSTR 在 16k 大池上的优势会被放大 |
| Stage 3 HRPO 长期 success_rate=0 | 退到 Stage 2 supervised 主表，HRPO 降为 §17 ablation |
| LLM-judge dedup 成本 / quota 不够 | 退到 rule-only dedup (D15-v2 选项 B)，vocab 多 5-10% 重复但训练上影响小 |
| 时间不够 4-5 周 | 砍 G2 cache + Phase G L_act + Phase H combination plot；只保 ABCDEF |
| Phase A F1 修完后 AppWorld dev57 routing recall@1 仍 <0.95 | 说明还有未发现 root cause，回到 dia方案 D11-v2/D14 之前再诊断；不进 Phase B |

---

## 8. Phase A 立即可启动清单

下面是 Phase A（3-4 天）即将动手的所有代码改动：

### A.1 文件清单

**改动文件**:
- `clstr/encoders.py`: `SkillTable.__init__` + `retrieval_logits` / `belief_logits` / `subspace_obs_logits` (F1 + F2-v2)
- `clstr/model.py`: `CLSTRModel.__init__` 加 action_proj (D6) + `sample_candidates` (G1) + `_policy_skill_logits` 改 belief residual (D5)
- `clstr/belief.py`: `TransitionPredictor.__init__` 删 action_emb 用 action_proj(E) (D6) + `subspace_obs` 使用 belief logits (F2-v2)
- `clstr/heads.py`: `SkillHead.forward` 输入改 (D5)
- `clstr/appworld_act_hrpo.py`: rollout candidate selection 改 G1 + per-step cross_encoder cache (G2 简版)
- `configs/model/*.yaml`: 删 fixed τ_s，加 `logit_scale_retr_init` / `logit_scale_belief_init`
- `configs/train/*.yaml`: 加 candidate_sample_temperature / widen_factor / β_routing_prior=0.15

**新增测试**:
- `tests/test_v4_skill_table_logits.py`: SkillRouter-base 同构 verify + identity init 等价 + retrieval/belief scale 和 bias 梯度
- `tests/test_v4_belief_residual.py`: m_t - h_t 计算正确 + ablation 分支测试
- `tests/test_v4_action_proj_sharing.py`: action_proj(E) 维度 + 梯度 flow
- `tests/test_v4_candidate_sampling.py`: train mode 不放回 + eval mode deterministic + temperature 趋 0 退化为 argmax-K
- `tests/test_v4_cross_encoder_cache.py`: cache hit/miss + key 唯一性

### A.2 验收脚本

```bash
# A.2.1 pytest
PYTHONPATH=.vendor_pytest:. pytest tests/test_v4_*.py -v
# 要求: 全过, 不破坏现有 447 个测试

# A.2.2 旧 ckpt 用新 logits supervised retrain verify
python -m clstr.train \
  --train_config configs/train/appworld_routing_train_v4.yaml \
  --model_config configs/model/appworld_skillrouter_init_v4.yaml \
  --data_config configs/data/appworld_routing.yaml \
  --checkpoint_dir outputs/v4_phase_a_retrain_verify/

# A.2.3 AppWorld dev57 routing recall@1 verify
python scripts/run_appworld_clstr_eval.py \
  --checkpoint outputs/v4_phase_a_retrain_verify/checkpoints/stage2_base-stepN.pt
# 期望: recall@1 ≥ 0.95 (从当前 0.825 跳上来; 如果跳不到，回头诊断)
```

### A.3 时间细分

- A.1 代码改动 + 单元测试: 2 天 (Claude 主导，pure 工程)
- A.2 重训 verify: 0.5 天 GPU
- A.3 报告 + Phase B 启动准备: 0.5-1 天

---

## 9. 文档与归档

- 本文件 `docs/clstr_v4_plan.md`: v4 完整方案（不再改），作为执行参考
- `description.md`: 持续更新各 Phase 进展，按时间追加
- 旧方案 (v3 SKILLRET 主战场) 已 cleared，仅作历史记录
- 76632 (Phase 2 v3 training) 已 cancelled，结果归档 `outputs/appworld_clstr_train_phase2_full_v3/` 不再 follow-up

---

## 10. 主要风险与开放问题

1. **TRAJECT-Bench paper 还没发布**，benchmark 公允性需在论文 related work 处明确论证
2. **ToolBench LLM-judge reward** 与 TRAJECT execution-based Acc 不同质，combination 分析需谨慎
3. **partial dedup LLM-judge cost** 取决于 borderline case 比例，可能需要 ~5-10k LLM call (Claude Sonnet 一次几分钱量级)
4. **Stage 3 HRPO 在新 vocab 50-150k 上的 candidate 多样性** 需要重新调 widen_factor M（当前 M=K×4 是 188 skill 量级，大 vocab 可能要 M=K×10）
5. **action_proj(E) 软共享时** 若 E 在 Stage 1 后被 frozen，action_proj 仍可学，但要 verify 梯度 flow 正确

这些都列在 description.md 后续 Phase 进展处更新。

---

## 11. Phase A 实施 sub-decisions (Q1-Q5)

为避免实施时再次"自作主张"，所有 Phase A 实施层面的 sub-design choice 在动手前显式决策。

| ID | sub-design | 决策 |
|---|---|---|
| Q1 | D14 SkillTable freeze 实现方式 | **拆成 `set_retrieval_trainable(flag)` 与 `set_belief_scale_trainable(flag)`**。Stage 3 默认调用 `set_retrieval_trainable(False)` 冻结 `W / E / logit_scale_retr / skill_bias_retr`；belief scale/bias 走 `frozen` vs `low-lr` ablation。 |
| Q2 | D6 后 GRUCell input dim 1152→2048 旧权重 | **完全重 init TransitionPredictor**。放弃旧 GRUCell / action_emb 权重 |
| Q3 | 旧 stage2_base ckpt 处理 | **完全放弃，从 Stage 1 重训**。`outputs/appworld_clstr_train_supervised_smoke_v1/checkpoints/stage2_base-step1.pt` 仅作历史参考 |
| Q4 | cross_encoder cache scope | **per-task**（group_size=8 同 task 共享）。每个 task 启动 rollout 前清 cache，task 内 8 rollout 共享。key = (state_text_hash, skill_idx) |
| Q5-v2 | F2 τ_s / logit scale 接管方式 | **retrieval scale 与 belief scale 分离**。retrieval 用 `logit_scale_retr` / `skill_bias_retr` 保持 SkillRouter-like sharp ranking；`subspace_obs` 用 `logit_scale_belief` / `skill_bias_belief` 保持更软的 belief distribution。删除旧 fixed `tau_s`，但不再用单一 `logit_scale` 同时控制两路。 |

### Q5-v2 含义补充

```python
# Before (v3):
def subspace_obs(skill_table, h, tau_s):
    logits = skill_table.logits(h) / tau_s    # ← v3 在外面除 tau_s
    return softmax(logits) @ skill_table.E

# After (v4 / Q5-v2):
def retrieval_logits(skill_table, h):
    q = F.normalize(skill_table.W(h), p=2, dim=-1)
    e = F.normalize(skill_table.E, p=2, dim=-1)
    return (
        skill_table.logit_scale_retr.exp().clamp(max=100.0) * (q @ e.t())
        + skill_table.skill_bias_retr.unsqueeze(0)
    )

def belief_logits(skill_table, h):
    q = F.normalize(skill_table.W(h), p=2, dim=-1)
    e = F.normalize(skill_table.E, p=2, dim=-1)
    return (
        skill_table.logit_scale_belief.exp().clamp(max=100.0) * (q @ e.t())
        + skill_table.skill_bias_belief.unsqueeze(0)
    )

def subspace_obs(skill_table, h):
    logits = belief_logits(skill_table, h)
    return softmax(logits) @ skill_table.E
```

动机：retrieval logits 需要 sharp ranking 来保证 recall；belief update 需要更软的分布，
否则 `m_t` 容易退化成 top-1 skill embedding，连续 belief 的价值会被抹掉。
因此 Q5-v2 明确分离两套 scale/bias，并把 `shared_scale` 作为 ablation，而不是默认方案。

推荐初始化：
- `logit_scale_retr_init = 0.0`（与 SkillRouter-like cosine init 对齐，后续由 L_retr 学 sharpness）
- `logit_scale_belief_init = log(0.2) ~ log(0.5)`（等价于更软的 belief sharpness）
- `skill_bias_retr = 0`
- `skill_bias_belief = 0`

必须报告的 ablation：
- `shared_scale` vs `separate_scale`
- `belief_scale frozen` vs `belief_scale low-lr`
- `m_t` entropy / routing recall / sequential next-skill acc 的联动变化

### Q3 含义补充

放弃旧 ckpt 意味着 **Phase A verify 步骤需要重新跑一次 supervised retrain**：

```bash
# v3 中 Phase A verify 是用旧 stage2_base ckpt 在新 logits 下 supervised retrain
# v4 中改为：从 Stage 0 init 直接 supervised retrain (Stage 1+2 数据)
# 注意: Phase A verify 不用 unified corpus v2 (Phase C 才有)，而是用当前 AppWorld 90 task supervised
#       这只是 sanity 验证新 logits 能否在熟悉的小池上拉到 ≥0.95，不是完整 Stage 1+2

python -m clstr.train \
  --train_config configs/train/appworld_routing_train_v4.yaml \
  --model_config configs/model/appworld_skillrouter_init_v4.yaml \
  --data_config configs/data/appworld_routing.yaml \
  --checkpoint_dir outputs/v4_phase_a_verify/

python scripts/run_appworld_clstr_eval.py \
  --checkpoint outputs/v4_phase_a_verify/checkpoints/stage2_base-stepN.pt
# 期望 AppWorld dev57 routing recall@1 ≥ 0.95
```

### Q4 实现细节

```python
# clstr/model.py: batch_cross_encode 加 cache 参数
class CLSTRModel(nn.Module):
    def __init__(self, ...):
        ...
        self._cross_encoder_cache = None  # init 时 None

    def reset_task_cache(self):
        """每个 task 启动 rollout 前调用"""
        self._cross_encoder_cache = {}

    def batch_cross_encode(self, states, candidate_rows, use_cache=True):
        if not use_cache or self._cross_encoder_cache is None:
            return self.cross_encoder.batch_forward(states, candidate_rows)
        
        results = []
        for state, cands in zip(states, candidate_rows):
            state_hash = hash(state if isinstance(state, str) else self._serialize_state_for_encoder(state))
            row_emb = []
            for c in cands:
                key = (state_hash, c)
                if key in self._cross_encoder_cache:
                    row_emb.append(self._cross_encoder_cache[key])
                else:
                    emb = self.cross_encoder.forward([state], [self.skills[c]])[0]  # single forward
                    self._cross_encoder_cache[key] = emb
                    row_emb.append(emb)
            results.append(torch.stack(row_emb))
        return torch.stack(results)
```

在 `clstr/appworld_act_hrpo.py:run_appworld_multistep_hrpo_rollout` 启动每个 task 前调用 `model.reset_task_cache()`。

### Q5 其它事项
以下为三个数据集的github仓库：
https://github.com/mangopy/tool-retrieval-benchmark
https://github.com/OpenBMB/ToolBench
https://github.com/PengfeiHePower/TRAJECT-Bench
如遇网络问题可使用国内镜像github:https://github.ur1.fun/，https://ghfast.top/
huggingface:https://hf-mirror.com/
实在下载不了可以暂停并告知我，由我来下载
你可以安装你需要的文件和依赖

---

## 12. 执行记录（2026-05-26）

### 12.1 历史 Stage1 retrieval warmup 结果

以下为历史记录。该 checkpoint 当时通过旧 Stage1 quality gate，但当前主线已改为
Stage0 coarse retriever foundation；它不再作为进入 Stage2 的 canonical 前置。

修复后的 Stage1 retrieval warmup checkpoint：

```text
outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

Stage1 quality gate 结果为 `status=ok`：

- `metric_count=5000`, `max_step=5000`
- `loss_drop=0.9851275682449341`
- last-window `recall_at_50=0.4146875`
- observed sources 覆盖 `skillret/toolret_training/toolbench_g3/traject_bench`
- checkpoint metadata 包含 `sampling_strategy=batch_stride` 与 `shuffle_queries=true`

该旧 gate 只证明历史 warmup 实验本身可复现；当前 Stage1 handoff 以
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
和 Stage0 quality gate 为准。当前 Stage2 handoff 以 Stage1 heads-init checkpoint
和 Stage1 quality gate 为准。

### 12.2 Stage2 提交状态

历史 Stage2 曾通过旧安全入口提交。当前 canonical 入口已经改为先提交 Stage1，再由
Stage1 gate hand off 到 Stage2：

```bash
bash scripts/submit_clstr_stage1_after_stage0_gate.sh
bash scripts/submit_clstr_stage2_after_stage1_gate.sh
```

旧 `scripts/submit_clstr_stage2_after_stage0_gate.sh` 现在只是兼容 alias，会转发到
`scripts/submit_clstr_stage1_after_stage0_gate.sh`。Stage1 入口先运行 Stage0
quality gate 与 preflight，均通过后提交 Stage1；Stage2 入口再运行 Stage1 quality
gate，通过后提交 Stage2：

```text
Submitted batch job 77555
```

Stage2 preflight 结果：

- `skill_count=37623`
- `embedding_shape=[37623, 1024]`
- `has_retrieval_adapter=true`
- `has_retrieval_scale_and_bias=true`

当前约束：Stage2 完成并产出通过 quality gate 的
`checkpoints/clstr_full_base-step10000.pt` 之前，不提交 Joint Stage4。旧
Stage3->Stage4 submit 链路只用于复现实验或辅助 ablation，不作为当前主线 gate。

### 12.3 v4 配置一致性修正

本轮将 v4 配置与 D6 保持一致：

- `CLSTRConfig.d_a` 未显式指定时自动等于 `d`；
- `configs/model/*.yaml` 中主 v4 配置的 `d_a` 与 `d` 同维；
- Qwen external CLSTR config 显式设置 `skill_head_context=belief_residual`、
  `candidate_widen_factor=4`、`candidate_sample_temperature=1.0`。

本轮验证通过：

- `python -m py_compile` 覆盖 `clstr/model.py`、`clstr/qwen_external_encoder.py`、
  Stage2/3/4 训练入口相关 Python 文件；
- `bash -n` 覆盖 Stage2/3/4 sbatch 与安全提交脚本；
- `git diff --check` 覆盖本轮修改文件。

注意：登录节点上 `torch` import 在 20-60 秒 timeout 内无输出，本轮未完成
torch-heavy pytest。后续如需重新跑 targeted tests，建议通过短 GPU sbatch 执行。

### 12.4 Stage3/Stage4 提交保护

为避免 Stage2 尚未完成时误启动后续阶段，本轮给后续 sbatch 入口加入硬检查：

- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
  - `ROUTING_CHECKPOINT_PATH` 必须存在且非空；
  - `HEAD_CHECKPOINT_PATH` 必须存在且非空，默认指向 Stage2 final checkpoint；
  - 缺失时直接退出，不启动 Qwen / 训练。
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
  - `ROUTING_CHECKPOINT_PATH` 必须存在且非空；
  - `HEAD_CHECKPOINT_PATH` 必须存在且非空，默认指向 Stage3 final checkpoint；
  - 缺失时直接退出，不启动 Qwen / 训练。

配套测试：

```bash
PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  -q --basetemp=.tmp_pytest
# 2 passed
```

该保护只防止误提交；Stage3/Stage4 是否可运行仍取决于上游训练产物和各自的
train/eval gate。

### 12.5 训练曲线可观测性与 Stage3 证据边界

本轮针对 Stage1 loss 图“单步跳动”容易误判的问题，增强
`clstr/training_monitor.py`：

- `loss_curve.svg` 同时显示 raw loss、rolling loss、EMA loss；
- 若训练指标中存在 `recall_at_50/10/5/1`，右轴叠加 recall 曲线；
- `TrainingMonitor` 轻量导入不再强制依赖 `torch`，只有保存 checkpoint 时才
  lazy import torch；
- 已用现有 Stage1 `training_metrics.jsonl` 刷新
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/loss_curve.svg`。

当前 Stage1 的解释口径：raw loss 单点跳动主要来自 batch size=16 与多数据源混合；
判断训练是否有效应看滑动趋势和 recall。当前 Stage1 loss 均值下降且
`recall_at_50` 上升，因此可进入 Stage2，但仍不能替代 TRAJECT / ToolBench-G3 /
ToolRet 正式 benchmark。

同时，readiness audit 将 Stage3 拆成两个机器可读 gate：

- `stage3_offline_hrpo`: 当前工程实现，训练制度为
  `offline_train_split_grouped_preference_update`，显式标注
  `on_policy_rollout_used=false`、`not_full_simulator_rollout_hrpo=true`、
  `paper_evidence_ready=false`；
- `stage3_full_rollout_hrpo`: 论文主 RL 主张所需的 simulator/on-policy rollout
  gate，当前固定 not ready，blocker 为 `full_rollout_hrpo_not_implemented`。

因此，Stage3 offline surrogate 可以在 Stage2 后作为工程阶段继续跑，但不能被写成
完整 HRPO RL 证据。若论文需要强 RL claim，后续必须实现并验证 TRAJECT/ToolBench
simulator rollout loop。

### 12.6 Stage2-to-Stage3 质量门与安全提交入口

本轮新增 Stage2 full-base handoff gate，避免 Stage2 只产出 checkpoint 就进入
Stage3：

- 新增 `clstr/stage2_quality_gate.py`；
- 新增 CLI `scripts/audit_clstr_stage2_quality.py`；
- 新增安全提交入口 `scripts/submit_clstr_stage3_after_stage2_gate.sh`。

Stage2 quality gate 默认检查：

- `train_report.json` 存在且 `status=ok`；
- `training_objective=component_complete_masked_multi_loss`；
- `training_regime=offline_replay_supervised_pretraining`；
- `valid_or_test_used_for_training=false`；
- `on_policy_rollout_used=false`；
- `frozen_routing_foundation=true`；
- final checkpoint、`checkpoints/latest.pt`、`training_metrics.jsonl`、
  `loss_curve.svg` 均存在；
- `max_step >= 10000`；
- first/last window loss 至少下降 `0.02`；
- `L_policy/L_trans/L_trans_skill_ce/STOP/routing` 均在数据中激活且被采样。

安全提交入口使用方式：

```bash
bash scripts/submit_clstr_stage3_after_stage2_gate.sh
```

该入口先检查 Stage0 routing checkpoint 与 Stage2 final checkpoint，再运行：

```bash
scripts/audit_clstr_stage2_quality.py --fail_on_action_required
```

只有 gate 通过才提交 `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`。该入口不会
提交 Stage4。

边界：该 gate 只证明 Stage2 可以工程上 hand off 给 Stage3 offline surrogate；
不替代 TRAJECT / ToolBench-G3 / ToolRet 正式 benchmark，也不改变
`stage3_full_rollout_hrpo` 当前未实现的事实。

当前状态：Stage2 job `77555` 仍在队列中，Stage2 output 目录尚无
`clstr_full_base-step10000.pt`、`training_metrics.jsonl` 或 `train_report.json`，
因此本轮没有提交 Stage3。

### 12.7 Legacy Stage3-to-Stage4 质量门与安全提交入口

本节记录的是旧 standalone Stage3->Stage4 handoff 保护，主要用于历史复现和
ablation。`v4.2_progressive_final` 当前主线不再要求先产出 standalone Stage3
checkpoint，而是由 Joint Stage4 直接从通过 Stage2 quality gate 的 Stage2 base
checkpoint 初始化。

旧链路曾新增 Stage3 offline HRPO-style handoff gate，避免 Stage3 只产出
checkpoint 就进入 Stage4：

- 新增 `clstr/stage3_quality_gate.py`；
- 新增 CLI `scripts/audit_clstr_stage3_quality.py`；
- 新增安全提交入口 `scripts/submit_clstr_stage4_after_stage3_gate.sh`；
- `scripts/audit_clstr_unified_training_readiness.py` 历史版本曾让 `stage4_act`
  被 Stage3 quality gate 阻断；当前版本对主线 Stage4 改为检查 Stage2 checkpoint
  与 Stage2 quality gate，Stage3 gate 不再阻断 `stage4_act`。

Stage3 quality gate 默认检查：

- `train_report.json` 存在且 `status=ok`；
- `stage=clstr_unified_stage3_hrpo_style`；
- `training_objective=unified_offline_hrpo_style_action_policy_update`；
- `training_regime=offline_train_split_grouped_preference_update`；
- `valid_or_test_used_for_training=false`；
- `on_policy_rollout_used=false`；
- `not_full_simulator_rollout_hrpo=true`；
- `hrpo_style_loss_used=true`；
- final checkpoint、`checkpoints/latest.pt`、`training_metrics.jsonl`、
  `loss_curve.svg` 均存在；
- `max_update >= 500`；
- `data_report.hrpo_rows > 0`；
- `traject_bench/toolbench_g3` 均有 Stage3 训练覆盖。

安全提交入口使用方式：

```bash
bash scripts/submit_clstr_stage4_after_stage3_gate.sh
```

该 legacy 入口先检查 Stage0 routing checkpoint 与 Stage3 final checkpoint，再运行：

```bash
scripts/audit_clstr_stage3_quality.py --fail_on_action_required
```

只有 gate 通过才提交 `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`。该入口
不会提交 Stage3，也不应用作 `v4.2_progressive_final` 主线提交入口。

边界：该 gate 只证明 Stage3 offline surrogate 可以工程上 hand off 给旧 Stage4；
不证明 full simulator/on-policy HRPO，也不替代后续 benchmark evaluation。

### 12.8 Phase H evaluation matrix readiness gate

本轮新增 Phase H 前置 readiness gate，避免把 routing/proxy 指标误写成论文主表的
official benchmark 结果：

- 新增 `clstr/eval_matrix_readiness.py`；
- 新增 CLI `scripts/audit_clstr_eval_matrix_readiness.py`；
- 新增 sbatch 包装 `scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh`。

该 gate 输出三个分离字段：

- `official_main_table_ready`: 只有 ToolRet official retrieval run、TRAJECT
  official `EM/Inclusion/Usage/Traj-Satisfy/Acc`、ToolBench-G3 StableToolBench
  pass-rate 条件都满足时才为 true；
- `proxy_or_routing_eval_ready`: TRAJECT sequence proxy、ToolBench-G3 static
  routing eval、ToolRet data/run 等 proxy 或 routing 证据可用时为 true，但它不
  能满足主表 official claim；
- `appworld_secondary_ready`: AppWorld §15.5.5 combination plot 是否可用，明确
  不进入三主表。

默认 sbatch 入口：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh
```

该脚本只运行 audit，不启动训练、raw generation、pass-rate eval 或其他 benchmark
执行。若 H200 队列忙，可用：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh
```

当前意义：Stage4 完成后，先跑该 gate；如果它仍提示
`traject_official_outputs_missing`、`toolbench_g3_pass_rate_not_ready` 或
`appworld_secondary_combination_report_missing`，说明只能报告 proxy/routing 或
secondary 结果，不能声称 §15 三主表已经完整。

2026-05-27 更新：Phase H readiness 现在还会先调用 Stage4 quality gate，而不是
只检查 Stage4 checkpoint 是否存在。该 gate 需要 `training_metrics.jsonl` 中持续记录
`stage4_act_loss` 与 `stage4_act_count`，并要求 Stage4 训练 artifact、expected
benchmarks coverage、latest checkpoint、loss curve 全部满足；否则加入
`stage4_quality_gate_not_ok`，`official_main_table_ready=false`。这避免“只有一个
checkpoint 文件但没有 L_act 训练质量证据”被误当作可进入官方主表评测。

已在当前项目状态下生成一次报告：

```text
outputs/clstr_eval_matrix_readiness/eval_matrix_readiness_current.json
```

当前 `status=action_required`，主要 blocker 为：

- `missing_stage4_checkpoint`
- `toolret_run_missing`
- `traject_official_outputs_missing`
- `toolbench_g3_pass_rate_not_ready`
- `appworld_secondary_combination_report_missing`

其中 ToolRet eval data、TRAJECT eval data、ToolBench-G3 normalized data 已通过各自
data audit；缺口主要在 Stage4 产物与三主表/secondary 的实际评测产物。

### 12.9 Smoke-first 提交约束

2026-05-27 更新：后续 full job 只由用户最终确认提交。Stage1/Stage2 已增加
`--max_rows` / `MAX_ROWS` 小样本 smoke 参数；Stage3/Stage4 已有 `--max_rows`。
训练侧 JSONL reader 已修复 `read_text().splitlines()` 问题，避免 U+0085 等 Unicode
line separator 把合法 JSONL 拆坏。新增：

- `scripts/audit_clstr_training_health.py`：早期检查 `training_metrics.jsonl`、
  `checkpoints/latest.pt` 和 last loss；
- `scripts/guard_clstr_job_budget.sh`：Stage1/2/3/4 guarded submit wrapper 在
  `sbatch` 前调用，默认 `MAX_ACTIVE_JOBS=4`。

执行纪律：完整训练前必须先跑本地小样本 smoke 和短作业 smoke；提交后必须在启动早期
检查日志与关键产物，异常时立即中止，不能无人看管地长时间占卡。

### 12.10 v4.2_progressive_final 单版本修复计划

2026-06-06 更新：由于预算不足以跑多组 full 对照，下一阶段不再并行尝试多个
Stage1/Stage2 组合，而是收敛到一个可解释、可复用、论文风险较低的最终候选：
`v4.2_progressive_final`。

已完成的 Stage1 full 对比：

```text
v4.1b top350 CE single:
  last300 transition_skill_recall@1 = 0.437
  last300 transition_skill_recall@5 = 0.839

v4.2 nowweak top350 CE single:
  last300 transition_skill_recall@1 = 0.333
  last300 transition_skill_recall@5 = 0.844

v4.2 nowweak inventory + listwise:
  last300 transition_skill_recall@1 = 0.389
  last300 transition_skill_recall@5 = 0.848
```

结论：

- inventory/listwise 不是退步主因；它把 v4.2 从 0.333 修回到 0.389；
- v4.2 严格移除 AgentGym weak_policy 后，ALFWorld policy/belief 覆盖显著下降；
- AgentGym weak_policy 原始行大多没有 `next_skill_id`，不能作为
  `L_trans_skill_ce` 正例来源，只能作为 policy / belief / STOP 的 filtered
  trajectory augmentation；
- hard inventory mask 把 transition CE 候选均值收缩到约 18，虽然降低 CE 难度，
  但限制了 listwise ranking 的学习空间。

最终方案：

1. **数据：v4.2 高质量基底 + filtered weak_policy augmentation**
   - 基底继续使用 v4.2 的 official replay、DAgger、HF ALFWorld
     admissible-success、ToolBench-G3、TRAJECT-Bench、WebShop；
   - 加回最多 12k 条 filtered AgentGym weak_policy；
   - weak rows 必须满足 train split、state/action 非空；
   - 优先保留 `L_policy=True` 且有 admissible actions 的首步/高置信行；
   - 剩余额度用于有 state/action/next_observation 的 belief/STOP 行；
   - 保持 weak rows 的 `L_trans_skill_ce=False`，不把缺失
     `next_skill_id` 的 weak data 注入 transition CE；
   - provenance 必须标注为 filtered weak-policy augmentation，论文中不混同为
     official data。

2. **候选：soft inventory mask**
   - 保留 `transition_inventory_mask_mode=auto`；
   - 新增 `transition_inventory_min_candidates=64`；
   - inventory/root 过滤后若候选少于 64，从原始 Stage0 top350 候选中回填；
   - gold positive 仍必须保留；
   - 这不是 eval-time gold injection，而是通用 inventory prior + Stage0 recall
     backfill。

3. **loss：增强 policy anchor，保持 transition ranking**
   - `L_policy=0.6`
   - `L_trans_skill_ce=0.6`
   - `L_trans=0.2`
   - `STOP=0.1`
   - `belief=0.1`
   - `routing=0.0`

4. **Stage0**
   - 不重训 Stage0；
   - 继续复用
     `outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`；
   - 只有当 progressive 数据的 top350 handoff coverage 明显低于 0.95 时，才重新讨论
     Stage0。

5. **推进 gate**
   - 先完成代码、数据构建和 CPU/smoke 健康检查；
   - full job 由用户最终确认；
   - Stage1 full 目标为：
     - last300 `transition_skill_recall@1 >= 0.43`；
     - last300 `transition_skill_recall@5 >= 0.85`；
     - last300 `policy_expert_recall@1 >= 0.82`；
     - no positive missing / no injected positives。

若该单版本 Stage1 仍达不到目标，不进入 Stage2，而应回到数据标签/候选结构审计。

### 12.11 v4.2_progressive_final 实现状态

2026-06-06 已完成代码层实现和 CPU 级验证，尚未提交 full GPU job。

已落地：

- `scripts/build_clstr_unified_pretrain.py` 新增
  `dataset_recipe=v4_2_progressive_final`；
- progressive recipe 复用 v4.2 高质量基底，并额外加入最多 12k 条
  filtered AgentGym weak_policy；
- filtered weak rows 只作为 `L_policy` / `L_trans` / `belief` / `STOP`
  augmentation，其中 `L_trans_skill_ce=False`、`routing=False`；
- `source_quality=weak_policy_filtered` 会被跳过 trajectory-derived retrieval
  pair 构建，避免污染 Stage0/routing 监督；
- provenance 记录
  `augmentation=filtered_weak_policy_v4_2_progressive_final`；
- `clstr/full_base_train.py` 新增
  `transition_inventory_min_candidates`，inventory/root 过滤后若候选不足会从原始
  Stage0 top-M 顺序回填；
- Stage1/Stage2 Python CLI 和 sbatch 主脚本都透传
  `--transition_inventory_min_candidates`；
- 新增 Stage1 final wrapper：
  `scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh`，
  默认使用 top350、inventory64、listwise NLL、gold-plus-equivalent、最终 loss
  配置。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_2_progressive_adds_filtered_weak_policy_cap \
  tests/test_full_base_train.py::test_transition_inventory_mask_backfills_to_min_candidates_from_stage0_topm \
  tests/test_clstr_topm_candidate_handoff.py::test_generic_stage2_cli_exposes_stage0_topm_handoff_without_qwen \
  tests/test_clstr_topm_candidate_handoff.py::test_stage1_heads_cli_exposes_sampling_and_benchmark_caps_for_unified_data \
  tests/test_sbatch_scripts.py::test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage1_sbatch_uses_rebalanced_loss_and_top350_handoff \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage2_sbatch_uses_inventory_listwise_and_stage1_checkpoint \
  tests/test_sbatch_scripts.py::test_v4_2_progressive_final_stage1_sbatch_uses_single_final_recipe

result: 9 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_sbatch_scripts.py
result: 49 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  clstr/full_base_train.py \
  scripts/run_clstr_stage1_heads_init.py \
  scripts/run_clstr_stage2_full_base_train.py
result: passed

bash -n \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh
result: passed
```

下一步在提交 full job 前应先构建
`data/clstr_unified_pretrain_v4_2_progressive_final`，再检查 manifest、
skill pool 与现有 v4.2 Stage0 checkpoint 的兼容性，以及 top350 handoff coverage。

### 12.12 Joint ACT Stage4 主线合并

2026-06-07 更新：为减少阶段膨胀并保持论文叙事清晰，当前主线将原 standalone
Stage3 的 offline preference surrogate 合并进 Stage4，作为可选辅助项：

```text
L_joint = L_act_next_skill_ce + lambda_pref * L_offline_preference
```

实现口径：

- `lambda_pref=0`：纯 Stage4 ACT，只优化 transition-conditioned next-skill CE；
- `lambda_pref>0`：在同一个 Stage4 训练循环中额外启用 Stage3-style offline
  preference auxiliary，并记录 preference loss / count / KL 等指标；
- Stage4 readiness gate 现在要求 Stage2 checkpoint 与 Stage2 quality gate 通过；
- standalone Stage3 仍保留为 legacy/offline surrogate ablation，不再阻断主线
  Stage4；
- `v4.2_progressive_final` 默认入口为
  `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`，
  其 `HEAD_CHECKPOINT_PATH` 默认指向
  `outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise/checkpoints/clstr_full_base-step10000.pt`。

下一步实验纪律：

- 先跑 Stage4 小样本 smoke，检查 `setup_status.jsonl`、`training_metrics.jsonl`、
  `loss_curve.svg` 与 `stage4_quality_gate.json`；
- smoke 正常后再由用户确认是否提交 full job；
- full 结束后先过 Stage4 quality gate，再进入 evaluation matrix readiness gate，
  不能把 routing/proxy 结果当作 official benchmark 主表。

### 12.13 Stage4 Candidate Handoff 对齐

2026-06-07 进一步修正：`v4.2_progressive_final` 的 Stage4 ACT 不能再依赖
`candidate_next_skill_ids` 字段，因为当前 trajectories 中可训练 next-skill 行没有该字段。
旧 builder 会退化为 gold `next_skill_id` 注入加随机负例，这与 Stage0/Stage1/Stage2 的
真实 top-M 候选空间不一致。

已落地：

- `clstr/stage4_act_train.py` 复用
  `clstr.full_base_train._attach_stage0_topm_candidates`；
- Stage4 handoff 主线使用 `stage0_next_candidate_skill_indices` /
  `stage0_next_candidate_skill_ids` 作为 ACT 候选；
- 若 Stage0 top-M 未召回 gold `next_skill_id`，Stage4 跳过该 ACT row，不注入 gold；
- Stage4 ACT 只需要 next-skill handoff，handoff 前会临时设置
  `routing=False, L_trans_skill_ce=True`，避免 current routing positive miss 误删
  ACT 行；
- `scripts/run_clstr_stage4_act_train.py` 新增并透传
  `--stage0_top_m`、`--stage0_positive_missing_policy`、
  `--stage0_handoff_query_mode`、`--stage0_handoff_sample_multiplier`、
  `--stage0_candidate_encode_batch_size`；
- `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`
  当前默认：
  `STAGE0_TOP_M=350`，
  `STAGE0_POSITIVE_MISSING_POLICY=skip`，
  `STAGE0_HANDOFF_QUERY_MODE=skillrouter_state`。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_v4_2_progressive_final_stage4_sbatch_starts_from_stage2_checkpoint_and_joint_act \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_does_not_override_cluster_gpu_cpu_bundle \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_uses_existing_default_conda_env_path \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_sources_shared_clstr_environment

result: 44 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_act_train.py \
  clstr/stage4_quality_gate.py \
  scripts/run_clstr_stage4_act_train.py \
  scripts/audit_clstr_unified_training_readiness.py
result: passed

bash -n \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
result: passed
```

下一步仍然是 Stage4 小样本 smoke，而不是直接 full。smoke 必须确认
`candidate_source=stage0_topm_online`、`positive_injected_rows=0`，并检查
`stage0_candidate_handoff.json` 与训练指标后，才讨论 full job。

### 12.14 Stage4 设计再审计与质量门加固

2026-06-07 复查 Stage4 主线时又发现一个质量门层面的设计风险：只要求
checkpoint、训练步数和 `stage4_act_loss/stage4_act_count` 不够。即使 Stage4 使用了
真实 Stage0 top-M candidate handoff，模型也可能在候选内没有学会 next-skill ranking，
但仍因 artifact 完整而通过 gate。

已修正：

- `clstr/stage4_quality_gate.py` 新增 Stage4 ranking/ACT 质量指标：
  - `first_stage4_act_loss`
  - `last_stage4_act_loss`
  - `stage4_act_loss_increase`
  - `last_stage4_next_skill_recall@5`
- 新增 blocker：
  - `missing_stage4_next_skill_quality_metrics`
  - `stage4_next_skill_recall_at_5_too_low`
  - `stage4_act_loss_regressed`
- Stage4 gate 默认 expected training benchmarks 对齐当前
  `v4.2_progressive_final` wrapper：
  `toolbench_g3`、`traject_bench`、`alfworld`、`webshop`。
- `scripts/audit_clstr_stage4_quality.py` 默认审计目录改为：
  `outputs/clstr_unified_stage4_v4_2_progressive_final_joint_act`。
- `scripts/submit_clstr_stage4_after_stage3_gate.sh` 作为 legacy standalone
  Stage3->Stage4 入口保留，但默认退出；只有显式设置
  `ALLOW_LEGACY_STAGE3_STAGE4=1` 才能继续，用于历史复现或 ablation。

边界：

- 这仍然只是 Stage4 训练质量门，不是 official benchmark 结果；
- 通用 Stage4 train wrapper 保持 v3-compatible 默认，避免破坏历史复现；
- 当前主线 smoke/full 必须使用
  `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`；
- 本轮没有提交 Slurm/GPU 作业。

### 12.15 Stage4 主线可训练模块与提交安全复查

2026-06-07 再次复查 Stage4 主线后，确认并修正两个非算法但会影响实验可信度的问题。

首先，`v4.2_progressive_final` Stage4 ACT 主线不能冻结 `TransitionPredictor`。
Stage4 训练的是 observation-conditioned next-skill ranking，如果只训练
`trans_head/action_proj`，核心 transition function 不能适配 Stage4 的
`state/m_obs + current_skill + next_observation` 输入分布，论文中的 closed-loop
transition 说服力会变弱。

已修正：

- `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`
  默认 `TRAIN_TRANSITION=1`；
- `clstr/stage4_quality_gate.py` 主线默认要求 `transition` 可训练，否则 blocker 为
  `stage4_transition_not_trainable`；
- `scripts/audit_clstr_stage4_quality.py` 仅为 legacy/ablation 保留
  `--allow_frozen_transition`。

其次，专用 Stage4 wrapper 已加入：

```bash
#SBATCH --time=06:00:00
```

这样即使用户直接按集群示例提交
`sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`，
也不会再次生成 unlimited time 作业。

候选集合边界：

- Stage4 主线复用 shared Stage0 top-M handoff；
- shared handoff 会保留“当前 skill 作为 next-candidate 的 self-skill safeguard”；
- 这不是 gold `next_skill_id` 注入；
- smoke/full 的关键证据仍必须是
  `candidate_source=stage0_topm_online`、`positive_injected_rows=0`、
  `stage0_candidate_handoff.injected_positive_rows=0`。

本轮没有提交 Slurm/GPU 作业。下一步仍是 Stage4 小样本 smoke，然后运行 Stage4
quality gate；full job 需要在 smoke 指标正常后再决定。

### 12.16 Stage2/Stage4 Transition 输入语义复查

2026-06-07 继续复查闭环 transition / belief 数据流时，发现一个会影响 Stage4
说服力的设计问题：`clstr/full_base_train.py` 的 Stage1/Stage2 full-base 训练中，
`L_trans`、`L_trans_skill_ce`、`belief` 以及 replay-prefix belief reconstruction
把 `action_text` embedding 作为 `_transition_prediction()` 的第三个输入传入。

这与当前模型定义不一致：

- `TransitionPredictor.forward(m_t, a_t, o_t_emb)` 的第三项是 observation embedding；
- `BeliefGate.forward(m_hat, m_obs, obs_emb)` 的第三项也是 observation embedding；
- Stage4 ACT 已经使用 `next_observation_text` embedding；
- 因此旧 Stage2 预训练和 Stage4 ACT 在 transition 输入语义上不一致。

已修正：

- full-base `L_trans` 使用 `_next_observation_embedding` / `next_observation_text`
  作为 transition observation input；
- full-base `L_trans_skill_ce` 使用同样的 next-observation embedding；
- full-base `belief` 的 transition 和 gate observation input 同步改为
  next-observation embedding；
- replay-prefix belief reconstruction 在有 `next_observation_text` 时也用
  next-observation embedding；
- `clstr/stage2_transition_row_diagnostics.py` 同步修复，避免 row-level audit
  继续审计旧语义；
- action/current-skill 输入仍由 `transition_action_input(model, labels, like=obs_emb)`
  产生，所以 soft-shared skill/action embedding 设计没有被削弱。

新增回归测试：

- `test_full_base_transition_and_belief_condition_on_next_observation_embedding_not_action_text`
  会在 transition 或 gate 收到 action-text embedding 时失败；
- 同时保留 action projection 共享测试，确认 current skill/action 输入仍来自
  skill embedding。

### 12.17 Legacy pre-action controller proxy observation 修复

Phase 19 后继续复查历史 ALFWorld controller 和旧 HRPO `controller_complete`，
确认这些 pre-action 候选评分路径没有真实 post-action observation，不能继续把
candidate/action embedding 当作 `TransitionPredictor` / `BeliefGate` 的
observation input。

已修正：

- `clstr/alfworld_eval.py::make_controller_component_scorer` 不再用 transition、
  gate、`trans_head` 或 `skill_head` 做 pre-action proxy transition/belief scoring；
- ALFWorld component scorer 当前只保留 policy / Q-success / current-state STOP
  相关信号，`transition_scores` 和 `belief_scores` 显式置零并 mask；
- trace metadata 显式记录
  `component_score_source=policy_q_success_stop_only_no_proxy_observation`、
  `transition_score_source=disabled_no_post_action_observation`、
  `belief_score_source=disabled_no_post_action_observation`；
- 旧 HRPO `controller_complete` 不再把 candidate/action embedding 作为 proxy
  observation 传给 transition/gate，也不再通过这条路径给 transition/gate/STOP
  产生梯度；
- 删除 ALFWorld 旧 proxy helper，避免测试或后续实现继续依赖这条错误语义。

论文边界：

- 历史 ALFWorld/HRPO pre-action controller 只能作为 policy/Q/STOP 诊断路径；
- 不能把它写成 closed-loop transition/belief 证据；
- `v4.2_progressive_final` 主线的 transition/belief 证据应来自 Stage2/Stage4
  的真实 `next_observation_text` / observation-conditioned 训练与评估。

本轮没有提交任何 Slurm/GPU 作业。

### 12.18 Stage4 checkpoint init 证据复查

2026-06-07 继续检查 Stage4 主线后，确认了一个证据链风险：Stage4 从
Stage0 routing checkpoint 构建模型后，还会合并 Stage2/head checkpoint。若未来
head checkpoint 含有 `encoder.*`、`cross_encoder.*` 或 `skill_table.*`，就可能
覆盖 Stage0 routing foundation，使 Stage4 top-M handoff 不再可被清楚地解释为
Stage0 候选分布。

当前 `v4.2_progressive_final` Stage2 checkpoint 已经排除了 frozen routing
foundation，因此这不是现有结果无效，而是防止未来手工/legacy artifact 污染主线证据。

已修正：

- CLSTR-native Stage4 主线使用
  `load_routing_and_head_checkpoints(..., protect_routing_foundation=True)`；
- legacy Qwen Stage4 入口也同步启用该保护；
- Stage4 quality gate 默认要求
  `checkpoint_init_report.protect_routing_foundation=True`；
- 若缺少该证据，blocker 为 `stage4_checkpoint_init_not_protected`；
- 若 artifact 报告 head checkpoint 已覆盖 `encoder/cross_encoder/skill_table`，
  blocker 为 `stage4_head_checkpoint_overrode_routing_foundation`；
- `scripts/audit_clstr_stage4_quality.py` 仅为历史/ablation 产物保留
  `--allow_unprotected_routing_init`。

数据流边界：

- Stage4 top-M handoff 虽然在合并 Stage2/head checkpoint 后计算；
- 但 handoff 只使用 `encoder + skill_table.retrieval_logits`；
- 在 routing foundation 保护开启后，该 handoff 仍是 Stage0 top-M candidate
  distribution，不是 Stage2 policy/transition reranking。

论文边界：

- Stage4 ACT 当前训练 transition-conditioned next-skill ranking；
- 主线训练 `TransitionPredictor`，但不直接优化 `BeliefGate`；
- belief/gate 的证据应来自 Stage1/Stage2 belief loss 和 next-observation-conditioned
  diagnostics，除非后续明确新增 Stage4 belief objective。

本轮没有提交 Slurm/GPU 作业。

### 12.19 Stage1 heads-init 设计复查

2026-06-07 复查 Stage1 后，当前 `v4.2_progressive_final` Stage1 结果没有被
推翻，但发现了需要加固的证据链边界。

当前已完成 Stage1 artifact 的主线条件是健康的：

- `candidate_source=stage0_topm_online`
- `positive_missing_policy=skip`
- `injected_positive_rows=0`
- `checkpoint_excludes_frozen_routing_foundation=True`
- `transition_inventory_mask_mode=auto`
- `transition_inventory_min_candidates=64`
- `transition_loss_type=listwise_nll`
- `transition_positive_mode=gold_plus_equivalent`

重新运行 Stage1 quality gate 后仍为 `status=ok`、`blockers=[]`。tail-100 指标保持：
loss `1.3690`、transition@1 `0.3842`、transition@5 `0.8417`、transition CE
`1.6114`。

需要修复的问题不是当前结果坏了，而是旧代码允许未来 artifact 污染证据链：

- Stage2 加载 Stage1 heads checkpoint 时，如果 checkpoint 含有 `encoder.*`、
  `cross_encoder.*` 或 `skill_table.*`，旧 loader 会加载这些 key，可能覆盖
  Stage0 routing foundation；
- 旧 Stage1 gate 只要求 `frozen_routing_foundation=True`，但没有要求 checkpoint
  真的排除了 frozen foundation；
- 旧 Stage1 gate 没有阻断 `positive_missing_policy=inject` 或
  `injected_positive_rows>0` 的 Stage1 artifact；
- 专用 Stage1 progressive-final wrapper 没有脚本级 walltime，直接 `sbatch` 时可能
  复现 unlimited-time pending 问题。

已修正：

- `load_head_checkpoint_into_model()` 默认启用
  `protect_routing_foundation=True`，过滤 `encoder/cross_encoder/skill_table`
  foundation keys，并在 load report 里记录 skipped protected keys；
- `clstr/stage1_heads_quality_gate.py` 新增 checkpoint/handoff safety blockers：
  `stage1_checkpoint_includes_frozen_routing_foundation`、
  `stage1_positive_missing_policy_not_skip`、
  `stage1_injected_stage0_positives`；
- Stage1 quality report 新增 `checkpoint_safety` 和 `handoff_safety`；
- `scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh`
  加入 `#SBATCH --time=06:00:00`。

边界：

- Stage1 gate 当前硬检查进入 Stage2 最关键的 policy / transition-ranking /
  candidate-handoff / checkpoint-init 安全；
- belief/gate 仍有 Stage1/Stage2 loss 与 diagnostics 证据，但当前 Stage4 不直接优化
  `BeliefGate`，所以 belief-specific pass/fail gate 应在后续明确新增 belief
  objective 或 belief diagnostics 后再单独加入；
- 这次修复不要求重训 Stage1/Stage2，但未来任何缺少上述 safety metadata 或依赖
  injected positives 的 Stage1 artifact 都不能作为主线前置。

### 12.20 Stage1 no-hardneg/top100 soft-inventory 修订

2026-06-08 重新按 Stage1 对 Stage1 比较后，确认 hardneg01 不是健康的 heads-init
默认配置：它在小候选集上让 transition top-1 ranking 退化。Stage1 的当前主线修订为：

- Stage1 默认关闭 `transition_hard_negative_margin`；hard negative 只作为后续可选
  ablation，不作为 gate 必需项；
- `transition_inventory_min_candidates` 从 64 提到 100，并且同时作用于 Stage0 handoff
  阶段与 loss 内 inventory filter；
- explicit inventory 仍优先作为 soft prior，但候选数不足时从真实 Stage0 top-M 顺序
  回填，不使用 gold injection；
- Stage1 loss 默认改为 `L_policy=0.7`、`L_trans=0.2`、
  `L_trans_skill_ce=0.5`、`STOP=0.1`、`belief=0.1`；
- 新 Stage1 输出目录：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init`。

这项修订不削弱论文设定：Stage1 仍消费真实 Stage0 top-M candidate set，仍使用通用
inventory prior，只是避免 inventory/hard-negative 在 heads-init 阶段过度压缩候选和
压制 policy anchor。Stage2/Stage4 仍需在真实 top-M/no-injection 条件下通过 gate 后
才能作为主线证据。

### 12.21 Stage1 候选方式回退到 inv64

`inventory100 + prior_residual_l025` full 结果显示：policy head 正常学习，但
transition@1 / @5 仍未过 gate；同时 top100 把 transition@5 从旧 inv64 的约 `0.84`
压到约 `0.65`。因此当前主线只回退候选方式，不回退 transition 语义：

- `transition_inventory_min_candidates=64`；
- active Stage1 输出：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_prior_residual_l025_heads_init`；
- Stage2/Stage3/Stage4/readiness 默认消费该 Stage1 checkpoint；
- 保留 `skill_prior_plus_action_observation_residual` 与 `transition_residual_lambda=0.25`。

当前判断：两种 transition 输入语义相加没有提升，是 residual branch 没有形成有效重排，
不是 ACT 主导。Stage1 尚未进入 ACT；下一步若 inv64 仍不过 gate，应先诊断 transition
prior/residual 的可分性和 residual 训练信号，而不是直接提交 Stage2/Stage4。

### 12.22 Phase 1: v4.1b conservative Stage2 baseline

当前执行优先级调整为先验证一个忠实 v4.1b Stage2 baseline：

1. 使用 v4.1b 数据、Stage0 checkpoint、Stage1 heads checkpoint；
2. Stage2 candidate/loss 保持 v4.1b 原样：
   top350、`positive_missing_policy=skip`、inventory mask off、
   single-positive cross-entropy；
3. Stage2 loss 权重保持 v4.1b 原样：
   `L_policy=1.0`、`L_trans=0.05`、`L_trans_skill_ce=0.2`、
   `STOP=0.2`、`belief=0.1`；
4. 显式使用 `transition_scoring_mode=v4_1b_action_observation`，避免当前代码默认的
   prior+residual scoring 污染 v4.1b baseline。

专用提交脚本：

```text
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative.sh
```

该脚本会先把旧 Stage1 的 handoff manifest 转成 coverage gate，要求：
top_m=350、skip、无 injected positives、current/next coverage >= 0.98。通过后再调用
generic Stage2 脚本。

判断标准：

- 如果 v4.1b conservative Stage2 指标健康，则先进行下游评估，作为论文 solid baseline；
- 如果 Stage2 不能延续 v4.1b Stage1 表现，则不继续提交 Stage4，而回到 transition
  标签、候选构建和输入语义的根因诊断；
- listwise/top-K/soft inventory 等改进移入 Phase 2，可在 baseline 成立后再做。

### 12.23 Phase 2 方向B实现准备：top50 listwise + trajectory prior

方向B已实现为候选模式和专用 sbatch wrapper，等待 Phase 1 baseline 稳定后再提交：

- 新候选模式：`stage0_topk_trajectory_prior`；
- 语义：Stage0 top-M 顺序仍是主信号，`transition_inventory_min_candidates=50`
  作为目标候选数；trajectory inventory 只占少量 prior slots，不做 hard filter；
- 训练时若 gold 在 Stage0 top-M 内但被 top50 截断，则保留 gold，避免 listwise loss
  无正例；
- loss：`listwise_nll` + `gold_plus_equivalent`；
- 数据/Stage0/transition scoring：仍锁 v4.1b，避免和 v4.2 prior-residual 混在一起。

专用脚本：

```text
scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_aggressive_top50_listwise.sh
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_aggressive_top50_listwise.sh
```

执行 gate：

- 先等 v4.1b conservative Stage2 完成并判断是否健康；
- 再跑方向B Stage1；
- 只有方向B Stage1 相比 v4.1b 原始 Stage1 不退化且 transition@1/@5 更好时，才跑方向B Stage2。

### 12.24 v4.1b conservative Stage2 baseline 结果

Phase 1 conservative Stage2 已完成并通过 gate：

```text
job_id = 85617
output_dir = outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask
checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask/checkpoints/clstr_full_base-step10000.pt
stage2_quality_gate.status = ok
stage2_quality_gate.blockers = []
```

核心指标：

```text
policy_expert_recall@1 = 0.91795
transition_skill_recall@1 = 0.56471
transition_skill_recall@5 = 0.82680
transition_skill_mrr = 0.67345
transition_skill_ce_candidate_count = 118.52
last-window transition@5 = 0.83792
```

这轮结果支持将 v4.1b conservative Stage2 作为 solid baseline。下一步已经提交方向B
Stage1 对比：

```text
job_id = 85627
script = scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_aggressive_top50_listwise.sh
```

若方向B Stage1 不优于该 baseline 对应的 Stage1/Stage2 表现，则不继续提交方向B Stage2。

### 12.25 Phase 2 方向B Stage1 对比结果

方向B Stage1 已完成，作业正常结束：

```text
job_id = 85627
state = COMPLETED
exit_code = 0:0
output_dir = outputs/clstr_unified_stage1_v4_1b_aggressive_top50_listwise_trajectory_prior_heads_init
checkpoint = outputs/clstr_unified_stage1_v4_1b_aggressive_top50_listwise_trajectory_prior_heads_init/checkpoints/clstr_stage1_heads-step3000.pt
```

配置确认：

```text
data = data/clstr_unified_pretrain_v4_1b
transition_inventory_mask_mode = stage0_topk_trajectory_prior
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
transition_scoring_mode = v4_1b_action_observation
transition_residual_lambda = 0.0
transition_skill_ce_candidate_count = 50
```

最终聚合指标：

```text
policy_expert_recall@1 = 0.90786
transition_skill_recall@1 = 0.32539
transition_skill_recall@5 = 0.66550
transition_skill_mrr = 0.47167
transition_skill_ce_loss = 2.53417
transition_current_skill_switch_false_positive@1 = 0.15406
```

结论：方向B的 top50/listwise/trajectory-prior 候选构建没有优于 v4.1b
conservative baseline。它虽然减少了候选数，但 transition top-5 明显低于保守
Stage2 baseline 的 `0.82680`，也低于旧 v4.1b Stage1 参考水平。因此本轮不提交方向B
Stage2，避免继续消耗 GPU 经费用在已明显退化的分支上。

当前论文主线应回到 Phase 1 的 v4.1b conservative Stage2 checkpoint，并进入下游评估：

```text
outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask/checkpoints/clstr_full_base-step10000.pt
```

### 12.26 Phase 2 方向A保守改进：top350 no-mask listwise

根据方向B结果，保守改进只改 loss，不改成功的候选分布：

- 数据、Stage0、transition scoring 均锁定 v4.1b；
- candidate 保持 `stage0_top350 + positive_missing_policy=skip`；
- `transition_inventory_mask_mode=off`，不做 top50 截断或 trajectory hard filter；
- `transition_loss_type=listwise_nll`，`transition_positive_mode=gold_plus_equivalent`；
- loss weights 仍使用 v4.1b 原始配方。

新增脚本：

```text
scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_conservative_listwise.sh
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative_listwise.sh
```

Stage1 对比结果：

```text
old_v4_1b_stage1:
  output = outputs/clstr_unified_stage1_v4_1b_top350_heads_init
  transition@1 = 0.37436
  transition@5 = 0.83317
  mrr = 0.55366
  switch@1 = 0.23306
  switch_false_positive@1 = 0.17986

conservative_listwise_stage1:
  job_id = 85637
  output = outputs/clstr_unified_stage1_v4_1b_conservative_top350_listwise_nomask_heads_init
  transition@1 = 0.41936
  transition@5 = 0.83444
  mrr = 0.58944
  switch@1 = 0.28600
  switch_false_positive@1 = 0.12086
```

Stage1 指标相对旧 v4.1b Stage1 全面不退化，尤其 transition@1、MRR 和 switch@1
提升明显，因此进入 Stage2。

Stage2 结果：

```text
job_id = 85644
output = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask
checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
stage2_quality_gate.status = ok
```

与上一版 v4.1b conservative Stage2 对比：

```text
old_ce_nomask_stage2:
  transition@1 = 0.56471
  transition@5 = 0.82680
  mrr = 0.67345
  switch@1 = 0.47058
  switch_false_positive@1 = 0.10264

conservative_listwise_stage2:
  transition@1 = 0.56938
  transition@5 = 0.83126
  mrr = 0.67739
  switch@1 = 0.47822
  switch_false_positive@1 = 0.07777
```

结论：方向A保守改进是当前最强且风险最低的 v4.1b baseline。虽然当前数据中
`transition_multi_positive_rows=0`，listwise 在数学上接近单正例 CE，但训练结果显示它在
不改变候选分布的前提下带来稳定小幅收益，尤其降低 current-skill false positive。
下一步应优先用该 checkpoint 做下游评估。

### 12.27 v4.1b conservative-listwise Stage4 ACT 适配

Stage4 已对齐当前最强路线，而不是继续使用 v4.2 progressive-final 默认路径：

```text
data = data/clstr_unified_pretrain_v4_1b
Stage0 routing checkpoint = outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt
Stage2 head/base checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
Stage2 quality gate = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/stage2_quality_gate.json
Stage4 output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act
```

Stage4 训练语义保持保守：

- `lambda_pref=0.0`，当前先跑纯 Stage4 ACT，不把离线 preference surrogate 混入；
- `transition_scoring_mode=v4_1b_action_observation`、`transition_residual_lambda=0.0`，
  沿用 v4.1b action-observation conservative scoring；
- `train_transition=1`，继续训练 `transition/trans_head/action_proj`；
- `stage0_top_m=350`，`positive_missing_policy=skip`，Stage4 候选仍来自 Stage0 top-M，
  不注入 gold candidate。

新增的 Stage4 wrapper：

```text
scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh
```

为避免 full 训练前 Stage0 top-M 预计算过大，同时避免小样本只覆盖文件开头的
`alfworld`，Stage4 入口新增 `benchmark_caps`：

```text
default benchmark_caps = toolbench_g3=4000,traject_bench=4000,alfworld=-1,webshop=4000
```

注意：通过 `sbatch --export` 覆盖该参数时应使用冒号分隔，例如
`toolbench_g3=4000:traject_bench=4000:alfworld=-1:webshop=4000`，脚本内部会转换为逗号。

已完成低成本验证：

```text
smoke20:
  job_id = 85664
  output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act_smoke
  status = completed
  conclusion = 代码可运行；quality gate 因 tiny smoke 只覆盖 alfworld 而提示 expected benchmarks/loss 窗口问题

smoke200:
  job_id = 85670
  output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act_smoke200
  status = completed
  tail stage4_next_skill_recall@1 ≈ 0.7425
  tail stage4_next_skill_recall@5 ≈ 0.96
  tail stage4_next_skill_mrr ≈ 0.8231
  tail stage4_act_loss ≈ 1.08
  smoke-specific quality gate = ok
```

当前代码验证：

```text
python -m pytest tests/test_stage4_act_train.py -k benchmark_caps_before_max_rows -q
python -m pytest tests/test_sbatch_scripts.py -k 'unified_stage4_act or v4_1b_conservative_listwise_stage4' -q
python -m pytest tests/test_stage4_quality_gate.py -k 'zero_residual_scoring or expected_transition_scoring_overrides or rejects_legacy_transition_scoring_metadata or accepts_complete_transition_act_training' -q
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh scripts/sbatch/run_clstr_unified_stage4_act_train.sh
python -m py_compile clstr/stage4_act_train.py clstr/stage4_quality_gate.py scripts/run_clstr_stage4_act_train.py scripts/audit_clstr_stage4_quality.py
git diff --check -- stage4 related files
```

下一步执行顺序：

1. 使用带 `benchmark_caps` 的 v4.1b Stage4 wrapper 再跑一个覆盖多 benchmark 的 smoke；
2. smoke 通过并且 Stage4 quality gate 无不可解释 blocker 后，提交 Stage4 full；
3. full 完成后运行 Stage4 quality gate，再进入 downstream evaluation/readiness。

### 12.28 v4.1b conservative-listwise Stage4 full 结果

Stage4 适配过程中发现并修正一个设计问题：训练行原本按 `trajectories.jsonl` 文件顺序进入
batch，导致 small smoke 的 first/last window 会被 benchmark 分段顺序影响。修复后
Stage4 在训练前对 ACT rows 做固定 seed shuffle，并在 `train_report.row_order` /
`setup_status.jsonl` 中记录：

```text
row_order.strategy = seeded_shuffle
row_order.seed = 17
```

修复后重新提交多 benchmark smoke：

```text
job_id = 85700
output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act_smoke_multibench_shuffle
state = COMPLETED
exit_code = 0:0
stage4_quality_gate.status = ok
stage4_quality_gate.blockers = []

stage4_rows = 248
benchmark_counts = {
  alfworld: 64,
  toolbench_g3: 56,
  traject_bench: 64,
  webshop: 64
}
stage0 next positive coverage@350 = 0.96875
positive_injected_rows = 0
first20 loss = 2.0220
last20 loss = 2.0229
last20 recall@5 = 0.775
transition_scoring_mode = v4_1b_action_observation
transition_residual_lambda = 0.0
```

full job 已提交并完成：

```text
job_id = 85703
output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act
state = COMPLETED
exit_code = 0:0
elapsed = 00:12:00
checkpoint = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt
stage4_quality_gate.status = ok
stage4_quality_gate.blockers = []
```

full 数据与 handoff：

```text
benchmark_caps = {
  alfworld: -1,
  toolbench_g3: 4000,
  traject_bench: 4000,
  webshop: 4000
}
retained source rows = 15307
stage4_rows = 14701
benchmark_counts = {
  alfworld: 3307,
  toolbench_g3: 3394,
  traject_bench: 4000,
  webshop: 4000
}
stage0 next positive coverage@350 = 14701 / 15307 = 0.96041
stage0_topm_next_positive_coverage@350 = 0.95747
positive_injected_rows = 0
candidate_source = stage0_topm_online
```

full 训练质量：

```text
steps = 2000
first200 stage4_act_loss = 2.0910
last200 stage4_act_loss = 2.0339
stage4_act_loss_increase = -0.0571
last200 stage4_next_skill_recall@5 = 0.72375

first500 recall@5 = 0.72075
last500 recall@5 = 0.74150
first500 mrr = 0.57985
last500 mrr = 0.59301
```

结论：Stage4 已经适配当前最强 v4.1b conservative-listwise 路线，并通过训练质量门。
这仍然是 Stage4 next-skill ACT 训练质量证据，不是 official downstream benchmark 结果；
下一步应进入 downstream evaluation/readiness，与 SkillRouter 做同口径比较。

### 12.29 Modern Qwen3 executor signal gate

2026-06-09 重新接入 Qwen3 时，不再沿用分散的旧 Qwen 接口。新增 shared generation
backend 作为 AppWorld executor / direct action proposal 的唯一 Qwen 生成入口：

```text
backend = clstr/qwen_backend.py::QwenGenerationBackend
default_model = models/Qwen3-8B
local_files_only = true
chat_template = tokenizer.apply_chat_template(..., enable_thinking=false)
backend_version = shared_generation_v1
```

已改动：

- `QwenCodeGenerator` 只包装 shared backend；
- `QwenDirectAdmissibleActionScorer` 支持 shared backend 注入；
- 新增低成本 AppWorld smoke wrapper：
  `scripts/sbatch/run_appworld_qwen3_signal_smoke.sh`；
- wrapper 默认只做 `qwen_only` executor signal check：
  `MAX_TASKS=2`、`MAX_STEPS=3`、`MAX_INTERACTIONS=10`，不训练、不跑 HRPO。

小样本 smoke 结果：

```text
job_id = 85712
partition = gpu_a800
state = COMPLETED
elapsed = 00:01:19
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_smoke
task_count = 2
success_count = 2
success_rate = 1.0
generation_failures = 0
execution_failures = 0
average_steps = 2.0
```

策略边界：

- AppWorld 是第一道 executor/RL reward signal gate，不是唯一 benchmark；
- 当前 smoke 只能证明 Qwen3-8B 在修正后的协议下不是立即零信号；
- 不能把 `qwen_only` 成功当成 CLSTR 结果；
- 在进入 AppWorld RL 前，应先扩大到小型 dev10/dev57 gate，确认 Qwen executor 本身有稳定
  task-success/reward；
- 如果 dev10/dev57 仍弱，下一步应优先换同规模更新 Qwen 模型，或切换到 reward 更密、更稳定的
  executor benchmark，而不是直接提交 full AppWorld RL。

dev10 gate 已完成：

```text
job_id = 85724
partition = gpu_a800
state = COMPLETED
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_dev10
task_count = 10
success_count = 3
success_rate = 0.3
generation_failures = 0
execution_failures = 2
task_completed_count = 10
evaluate_success_count = 3
average_steps = 1.9
```

解释：

- Qwen3-8B 修正接口后有非零 AppWorld executor reward，不应再用旧 0 分接口结论否定；
- 但 `0.3` 仍不足以直接启动 full AppWorld RL；
- 下一步应该在同一 dev10 gate 上加入 CLSTR skill-context 做对照；
- 若 CLSTR skill-context 不能提升，再考虑同规模更新 Qwen 或切换 benchmark。

CLSTR skill-context dev10 对照已完成：

```text
method = clstr_act
prediction_file = outputs/appworld_clstr_eval/act_v2_dev/predictions.jsonl
job_id = 85732
partition = gpu_a800
state = COMPLETED
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_dev10_clstr_act
task_count = 10
success_count = 7
success_rate = 0.7
generation_failures = 0
execution_failures = 0
task_completed_count = 10
evaluate_success_count = 7
average_steps = 2.0
step_positive_skill_hit_rate = 1.0
```

与 qwen-only 对比：

```text
qwen_only dev10 = 3 / 10 = 0.3
clstr_act skill-context dev10 = 7 / 10 = 0.7
```

主要改善来自 `fac291d_*` unique-song count 和 `4ec8de5_1` release-year count；仍失败的是
`530b157_*` phone+venmo grocery 三个任务。当前证据说明 CLSTR skill context 对
AppWorld executor 有明显帮助，但这条对照使用旧 AppWorld CLSTR ACT predictions，不是当前
unified v4.1b 主线。下一步应先用 dev57 复核 uplift，再决定是否接 AppWorld ACT/HRPO
rollout；若 phone+venmo 仍是主失败簇，优先修 prompt/schema hints 或换更强同尺寸 Qwen。

### 12.30 AppWorld current-route 接入

AppWorld 不再作为旧 AppWorld-specific prediction 文件的旁路实验处理。新的主线是：

```text
AppWorld SkillX / routing / replay / verified_pairs
  -> data/clstr_appworld_current_route_v1
  -> current Stage0 bi-encoder
  -> current Stage1 heads-init
  -> current Stage2 full-base
  -> current Stage4 ACT
  -> AppWorld multistep executor + Qwen backend
```

已完成的数据接入：

```text
skill_pool.jsonl      188 AppWorld SkillX skills
trajectories.jsonl    1068 rows
retrieval.jsonl       2496 pairs
manifest.json         status=ok, train-only leakage boundary
```

新增执行入口：

- `scripts/build_clstr_appworld_current_route.py`
- `scripts/sbatch/run_clstr_appworld_current_stage0_biencoder_train.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage1_heads_init.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage4_act_train.sh`
- `scripts/sbatch/run_appworld_current_route_multistep_executor_eval.sh`

large skill pool 叙事不放弃，但需要分层：

- routing 层：可用 `extra_skill_pool_paths` 生成 large-pool distractor 版本，让 AppWorld
  positives 在统一大池中被检索，报告 mask 前 recall/MRR/rank；
- executor/RL 层：AppWorld 环境只能执行 AppWorld-compatible SkillX skills，因此 executor
  前使用 `appworld_executor_compatible_only` 过滤不可执行 distractors，并在 diagnostics 中保留
  mask 前候选用于论文审计。

后续低成本顺序：

1. 先跑 AppWorld current-route Stage0 smoke / routing audit；
2. Stage0 正常后跑 Stage1 smoke，检查 handoff manifest 和 transition metrics；
3. 只有 Stage1 gate 正常才进入 Stage2；
4. Stage2 后再跑 Stage4 ACT；
5. 最后用 `run_appworld_current_route_multistep_executor_eval.sh` 做 dev10/dev57 task-success gate；
6. 若 dev10/dev57 明显优于 qwen-only，再考虑 AppWorld online RL；否则优先换更强同尺寸 Qwen 或转向
   ToolBench executor RL。

TRAJECT-Bench 可以不接 live executor 做离线 sequential routing / ACT / preference，但这不是
online RL；论文中应称为 offline trajectory optimization 或 sequence-level routing evaluation。

### 12.31 v4.1b ALFWorld online HRPO viability smoke

Stage4 offline ACT full 已完成并通过 gate，但指标只显示稳定、没有明显超过 Stage2。因此下一步
不继续调 Stage4 offline，而是做一个极小的 ALFWorld train-split online HRPO viability smoke，
用来判断 online reward/advantage 是否非退化。

新增入口：

```text
scripts/run_clstr_v4_1b_alfworld_online_hrpo.py
scripts/sbatch/run_clstr_v4_1b_alfworld_online_hrpo_smoke.sh
```

默认配置：

```text
init_checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
data_root = data/clstr_unified_pretrain_v4_1b
output = outputs/clstr_v4_1b_alfworld_online_hrpo_smoke
max_games = 1
group_size = 2
max_steps = 5
updates = 1
score_mode = controller_complete
```

安全边界：

- 只使用 ALFWorld train split rollout，不使用 valid/test 训练；
- 保留 v4.1b Stage2 checkpoint 中的 `skills_path`，即使用 unified large skill pool；
- checkpoint 内已有 `skill_table.E` 时不重建 skill table，避免 smoke 作业在启动阶段重编码
  37k skills；
- 输出 `viability` gate，要求 rollout/step 非零、reward 有变化、advantage 非零、admissible
  action 空间非退化；
- 这个 smoke 不是 full RL，不能作为论文最终 online RL 结果，只用于决定是否值得继续小规模
  ALFWorld online gate。

2026-06-09 执行结果：

```text
job_id = 85898
partition = gpu_a800
output = outputs/clstr_v4_1b_alfworld_online_hrpo_smoke
checkpoint = outputs/clstr_v4_1b_alfworld_online_hrpo_smoke/checkpoints/clstr_v4_1b_alfworld_online_hrpo-step1.pt
rollout_count = 2
rollout_step_count = 10
rollout_success_rate = 0.0
mean_rollout_reward = -0.15
reward_unique_count = 1
updates_with_nonzero_advantage = 0
mean_admissible_action_count = 13.1
viability.status = action_required
viability.blockers = [no_reward_variation, no_nonzero_advantage_updates]
hrpo_policy_loss = 0.0
loss = 0.0
```

本次先修复了两个工程阻断点：

- checkpoint skill pool JSONL 读取改为文件句柄逐行读取，避免 `U+0085` 等合法 JSON
  字符被 `splitlines()` 误切成多行；
- blocker report 写入 traceback，并在 run 开始时清理 stale `blocker_report.json` /
  `train_report.json`。

技术链路已经打通：v4.1b Stage2 large-pool checkpoint 可以加载，ALFWorld train split
可以 rollout，并能写出 checkpoint、rollouts、metrics 和 train report。但 RL 信号尚不可用：
两条 rollout reward 完全相同，advantage 为 0，HRPO loss 为 0。因此当前不能进入 full
ALFWorld online RL。

2026-06-09/10 后续修复与低成本 smoke：

```text
observation_smoke:
  job_id = 86203
  output = outputs/clstr_v4_1b_alfworld_online_hrpo_observation_smoke
  post_action_observation_used = true
  pre_action_next_observation_leakage = false
  online_observation_aux_count = 10
  online_transition_loss = 0.08017
  online_belief_loss = 0.00129
  reward_unique_count = 1
  updates_with_nonzero_advantage = 0

progress_reward_smoke:
  job_id = 86218
  output = outputs/clstr_v4_1b_alfworld_online_hrpo_progress_smoke
  reward_unique_count = 2
  updates_with_nonzero_advantage = 1
  nonzero_advantage_count = 2
  hrpo_policy_loss = -0.0372
  viability.status = ok

multi_update_progress_smoke:
  job_id = 86284
  output = outputs/clstr_v4_1b_alfworld_online_hrpo_progress_update_smoke
  rollout_count = 12
  rollout_step_count = 96
  reward_unique_count = 9
  updates_with_nonzero_advantage = 3
  online_observation_aux_count = 96
  viability.status = ok

interaction_reward_smoke:
  job_id = 86290
  output = outputs/clstr_v4_1b_alfworld_online_hrpo_interaction_smoke
  viability.status = ok
  mean_goal_interaction_points = 0.0
  mean_irrelevant_action_count = 2.58

goal_action_prior_smoke:
  job_id = 86300
  output = outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_smoke
  goal_action_bonus = 1.0
  reward_unique_count = 10
  updates_with_nonzero_advantage = 3
  mean_goal_interaction_points = 0.1667
  mean_irrelevant_action_count = 2.4167
  rollout_success_rate = 0.0
  viability.status = ok
```

结论边界：

- Stage4-online 现在有真实的 post-action observation auxiliary loss，且动作前不泄露
  `obs_{t+1}`；
- shaped reward 已能产生非零 reward variation、advantage 和 HRPO policy loss；
- `GOAL_ACTION_BONUS=1.0` 只作为探索诊断 prior，证明当前策略可以采到目标物体动作，
  但 update 2/3 未稳定延续，`rollout_success_rate` 仍为 0；
- 因此当前还不能作为 ALFWorld online RL 成功结果，只能说明瓶颈已经从
  “无训练信号” 移到 “探索/导航与训练强度不足”。

下一步仍应保持小成本：先用同一 v4.1b Stage2 checkpoint 跑稍强 smoke
（例如 `MAX_GAMES=1, GROUP_SIZE=4, MAX_STEPS=12, UPDATES=5,
GOAL_ACTION_BONUS=1.0, LR=5e-5`），观察目标物体交互是否能跨 update 保持；
只有出现稳定 goal interaction / reward 上升趋势后，才讨论更大 online RL。

2026-06-10 stronger smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_smoke`
（job `86312`）已完成：

```text
elapsed = 00:04:22
rollout_count = 20
rollout_step_count = 240
updates_with_nonzero_advantage = 5
reward_unique_count = 16
mean_rollout_reward = -0.4525
max_rollout_reward = -0.3300
mean_goal_interaction_points = 0.15
mean_irrelevant_action_count = 3.95
rollout_success_rate = 0.0
viability.status = ok
```

这轮证明更长 rollout 下训练仍非退化，但没有带来稳定提升。复查后发现
`_flatten_rollout_steps()` 把 rollout-level advantage 再除以 step 数，而后续 policy loss
已经对 step 求均值，导致长 rollout 的 policy gradient 被二次削弱。该实现不符合
`admissible_action` 级 HRPO：每个 action 应继承同一 rollout 的 group-normalized advantage。
已修复为不按 `len(steps)` 缩放，并新增
`test_flatten_rollout_steps_keeps_rollout_advantage_per_action` 回归测试。

下一步用相同 stronger smoke 配置重跑一次小作业，隔离验证 advantage scaling 修复是否改善
goal interaction/reward 趋势；若仍无改善，再优先诊断 action candidate scoring / exploration
prior，而不是继续加大 rollout 规模。

advantage scaling 修复后的同配置 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_advfix_smoke`
（job `86314`）完成：

```text
elapsed = 00:04:17
rollout_count = 20
rollout_step_count = 240
updates_with_nonzero_advantage = 5
mean_rollout_reward = -0.4525
mean_goal_interaction_points = 0.15
mean_irrelevant_action_count = 3.95
rollout_success_rate = 0.0
viability.status = ok
```

修复后 policy loss 从约 `0.012` 放大到约 `0.149`，说明原来的 step-division
确实压低了梯度；但两次 run 的 20 条 action trace 完全一致，最大动作概率变化只有
约 `3.3e-4`。因此，单靠放大 advantage 还不能在 5 个 update 内改变探索分布。

继续从 action trace 定位到一个 exploration prior 缺口：`GOAL_ACTION_BONUS=1.0`
此前只使用目标物体词（如 `tissuebox`），没有给目标 receptacle/navigation
（如 `go to sidetable`）任何 prior。这导致策略偶尔拿到目标物体后仍没有稳定向目标位置移动。
已把 `_goal_action_bonus_logits()` 改为使用所有 goal terms，但保留权重差异：
目标物体最高，receptacle/location 较弱。新增
`test_goal_action_bonus_prior_also_supports_goal_receptacle_navigation`。

注意：这个 goal-action prior 仍是 ALFWorld online exploration 诊断组件，不应直接包装成
主论文方法收益。若 receptacle prior smoke 有明显改善，下一步应把它定位为
“低成本确认 online RL 能拿到可学习成功前缀”，再讨论是否改为更通用的 text-goal
exploration prior 或换更强 executor。

receptacle prior smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s12_smoke`
（job `86321`）完成：

```text
elapsed = 00:03:46
rollout_count = 20
rollout_step_count = 240
updates_with_nonzero_advantage = 5
mean_rollout_reward = -0.3565
mean_goal_interaction_points = 0.15
mean_irrelevant_action_count = 2.30
max_rollout_reward = 0.01
rollout_success_rate = 0.0
viability.status = ok
```

相对 advfix smoke：

```text
mean_rollout_reward:       -0.4525 -> -0.3565
mean_irrelevant_action_count: 3.95 -> 2.30
sidetable action count:        17 -> 64
tissuebox action count:         3 -> 5
```

最关键的行为证据是 update 5 产生了：

```text
take tissuebox 3 from diningtable 1
go to sidetable 2
move tissuebox 3 to sidetable 2
```

这说明 Stage4-online 已经从“无 reward/advantage 信号”推进到“能产生目标前缀并减少无关动作”。
但当前仍未完成 ALFWorld `pick_two_obj_and_place`，`rollout_success_rate=0`。因此下一步不应
直接 full RL；更合理的是继续做一个小规模但更接近任务完成的诊断：

- `MAX_STEPS` 增到 20 左右，让第二个 tissuebox 有执行空间；
- 保留 `MAX_GAMES=1, GROUP_SIZE=4, UPDATES=5` 控制成本；
- 观察是否出现第二个 target-object interaction 或完整 success；
- 若成功前缀仍停在一个物体，应诊断 inventory/carrying state 与 STOP/placement reward，而不是
  继续堆 update。

20-step horizon 诊断
`outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s20_smoke`
（job `86472`）完成：

```text
elapsed = 00:05:07
rollout_count = 20
rollout_step_count = 400
updates_with_nonzero_advantage = 5
mean_rollout_reward = -0.6215
mean_goal_interaction_points = 0.70
mean_irrelevant_action_count = 3.85
take_tissuebox_count = 11
move_tissuebox_count = 8
rollout_success_rate = 0.0
viability.status = ok
```

这轮排除了“12 步 horizon 太短是主因”的解释。增加到 20 步后，目标物体交互明显变多，
但 success 仍为 0，且 reward 下降。典型失败轨迹包括：

```text
take tissuebox 3 from diningtable 1
move tissuebox 3 to sidetable 1
take tissuebox 3 from sidetable 1
...
move tissuebox 3 to sidetable 1
```

以及：

```text
take tissuebox 2 from diningtable 1
move tissuebox 2 to diningtable 1
take tissuebox 1 from diningtable 1
move tissuebox 1 to diningtable 1
```

因此当前低成功率主要来自 ALFWorld online bridge 的 credit assignment / state tracking
设计，而不是 Qwen。当前 run 不使用 Qwen direct generator，也不使用 Qwen external encoder；
动作选择是 CLSTR policy head + goal-text prior 在 admissible actions 上排序。失败点是：

- 没有显式追踪 `carrying object`、`object already placed in target receptacle`、`remaining count`
  等子目标状态；
- reward 对“正确放置后又拿回”缺少惩罚，对“放到非目标 receptacle”惩罚不足；
- action prior 能引导到目标物体/目标 receptacle，但不能表达“已完成的放置不要撤销”；
- post-action observation auxiliary loss 已接通，但 pre-action controller 仍主要由 policy/action
  ranking 决策，transition/belief 尚未形成可用于 action selection 的任务状态约束。

下一步若继续 Stage4-online，应先加通用 ALFWorld subgoal tracker/reward audit，而不是直接加
`MAX_STEPS` 或 full job：

- 从 observation/inventory/action trace 解析 object location 与 carrying state；
- 奖励 `target_object placed in target receptacle`，惩罚 `take target_object from target receptacle`
  和 `move target_object to non-target receptacle`；
- report 中显式输出 `placed_target_count`、`reverted_target_count`、
  `wrong_receptacle_move_count`；
- 先用已有 rollouts 做离线 reward 重算，再决定是否提交新的小 smoke。

subgoal reward smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_subgoal_reward_lr5e5_u5_s20_smoke`
（job `86489`）完成：

```text
elapsed = 00:04:32
rollout_count = 20
rollout_step_count = 400
updates_with_nonzero_advantage = 5
mean_rollout_reward = -0.6220
mean_goal_interaction_points = 0.70
mean_irrelevant_action_count = 3.85
mean_placed_target_count = 0.15
mean_placed_target_event_count = 0.25
mean_reverted_target_count = 0.10
mean_wrong_receptacle_move_count = 0.15
rollout_success_rate = 0.0
```

该 run 和上一轮 s20 的 20 条 action trace 完全一致，最大动作概率变化约 `0.0032`。
这说明仅把 subgoal tracker 放进 rollout-level total reward 还不够：一条包含正确放置、
拿回、放错、无关动作的 rollout 仍只得到一个 advantage，坏动作会被同一优势一起强化或惩罚。

因此继续修复为 step-level credit assignment：

- 每一步根据通用 ALFWorld action schema 计算即时 credit；
- 正确 `move/put/place target_object to target_receptacle` 获得正 credit；
- `take target_object from target_receptacle` 和
  `move/put/place target_object to non_target_receptacle` 获得负 credit；
- info/repeat/irrelevant manipulation 保持负 credit；
- step credit 在同一 task 内标准化后叠加到 rollout-level advantage。

这不是 `tissuebox/sidetable` 特化：对象词、目标位置词和目标数量都从 goal text 中解析；
动作事件来自 ALFWorld 通用 admissible action schema。新增测试：

```text
test_alfworld_subgoal_tracker_rewards_final_placement_and_penalizes_reverts
test_flatten_rollout_steps_adds_step_credit_advantages
```

下一步仍只跑同规模小 smoke，验证 step-level credit 是否让 action trace 真正不同；
如果仍不改变轨迹，再考虑提高 learning rate / update 数或引入 self-imitation，而不是直接 full。

第一版 step-level credit smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_step_credit_lr5e5_u5_s20_smoke`
（job `86503`）完成，但没有改善：

```text
mean_rollout_reward = -0.6400
mean_placed_target_count = 0.10
mean_reverted_target_count = 0.15
mean_wrong_receptacle_move_count = 0.15
online_credit_reward_nonzero_count = 400
rollout_success_rate = 0.0
```

诊断发现它把每一步的 step/info/repeat penalty 也放入 credit，导致 400/400 步都有非零
step credit，稀释了真正的子目标事件。已改为 sparse event credit：

- 普通导航、`look`、`inventory`、`help` 不再直接产生 step credit；
- rollout-level reward 仍保留 step/info/repeat penalty；
- step-level credit 只来自目标发现、目标物体交互、正确放置、拿回、放错、无关
  manipulation；
- 新增 `test_alfworld_step_credit_is_sparse_for_neutral_navigation`。

下一轮继续同规模 smoke，验证稀疏 credit 是否能减少 wrong/reverted 并提升 placed final。

sparse step-credit smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_sparse_step_credit_lr5e5_u5_s20_smoke`
（job `86519`）完成：

```text
mean_rollout_reward = -0.6425
mean_placed_target_count = 0.10
mean_reverted_target_count = 0.15
mean_wrong_receptacle_move_count = 0.15
online_credit_reward_nonzero_count = 130
rollout_success_rate = 0.0
```

稀疏化把非零 step credit 从 400 降到 130，但指标仍未改善。继续检查发现
`move target_object to wrong_receptacle` 和 `take target_object from target_receptacle`
仍会先得到 target-object interaction 正分，再扣 wrong/reverted penalty，导致负信号被抵消。
已改为互斥 target-transfer credit：

- `take target_object from non_target_receptacle`：正 credit；
- `move/put/place target_object to target_receptacle`：更强正 credit；
- `take target_object from target_receptacle`：负 credit；
- `move/put/place target_object to non_target_receptacle`：负 credit；
- transfer 事件不再额外叠加泛化 target interaction 正分。

新增 `test_alfworld_step_credit_scores_target_transfers_by_outcome`；下一轮继续同规模 smoke，
只验证互斥 transfer credit 是否减少 reverted/wrong。

下一步如果继续 online RL，不应直接训练，而应先做更便宜的 reward-variance gate：

```text
candidate gate:
  max_games = 3-5
  group_size = 4
  max_steps = 10-20
  updates = 1
  save_checkpoint = false 或只保存小诊断产物

pass condition:
  reward_unique_count >= 2
  updates_with_nonzero_advantage > 0
  mean_admissible_action_count > 1
```

若该 gate 仍失败，应优先修 ALFWorld action scorer / reward shaping / task sampling，而不是继续
提交 online RL 训练；若 gate 通过，再进入很小规模 multi-update online training。

互斥 transfer step-credit smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_transfer_step_credit_lr5e5_u5_s20_smoke`
（job `86535`）已完成：

```text
rollout_success_rate = 0.0
mean_rollout_reward = -0.6425
mean_goal_interaction_points = 0.70
mean_placed_target_count = 0.10
mean_placed_target_event_count = 0.25
mean_reverted_target_count = 0.15
mean_wrong_receptacle_move_count = 0.15
online_credit_reward_nonzero_count = 130
mean_online_credit_reward_sum = 0.077
```

结论：互斥 transfer credit 让正负方向更干净，但 5 个 update 后 action trace 仍基本不变，
因此当前主要瓶颈是 step credit 对 policy advantage 的影响太弱，而不是 Qwen、步数、
reward variation 或 target/receptacle prior 缺失。

已新增 `step_credit_advantage_weight`：

- 默认 `1.0`，保持旧行为；
- `0.0` 可关闭 step credit 做 ablation；
- 大于 `1.0` 可放大 step credit 对 action-level advantage 的贡献；
- CLI/sbatch 通过 `--step_credit_advantage_weight` /
  `STEP_CREDIT_ADVANTAGE_WEIGHT` 暴露。

下一步只跑一个低成本小 smoke：`STEP_CREDIT_ADVANTAGE_WEIGHT=3.0`、
`MAX_GAMES=1`、`GROUP_SIZE=4`、`MAX_STEPS=20`、`UPDATES=5`。如果
target placement/revert/wrong receptacle 或 action probability 有方向性改善，再加大
updates；如果仍无变化，停止继续 reward shaping，转向 high-credit action
self-imitation / auxiliary objective。

`STEP_CREDIT_ADVANTAGE_WEIGHT=3.0` 小 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_step_credit_w3_lr5e5_u5_s20_smoke`
（job `86577`）已完成：

```text
20/20 rollout action trace 与上一轮完全一致
mean_rollout_reward = -0.6425
mean_goal_interaction_points = 0.70
mean_placed_target_count = 0.10
mean_reverted_target_count = 0.15
mean_wrong_receptacle_move_count = 0.15
prob_max_abs_delta = 0.0047
prob_mean_abs_delta = 0.00027
```

这说明单纯放大 advantage 通道不够。下一步实现并验证通用 high-credit action auxiliary
objective：

- `step_credit_imitation_weight=0.0` 默认关闭，不改变已有 Stage4-online 行为；
- 正 step credit 的已选动作做 self-imitation；
- 负 step credit 的已选动作做 unlikelihood penalty；
- objective 只依赖 `online_credit_reward`，可以迁移到任何能提供 step-level
  credit/preference 的在线或 replay 环境；
- CLI/sbatch 通过 `--step_credit_imitation_weight` /
  `STEP_CREDIT_IMITATION_WEIGHT` 暴露。

小 smoke 建议配置：`STEP_CREDIT_ADVANTAGE_WEIGHT=1.0`、
`STEP_CREDIT_IMITATION_WEIGHT=0.5`、`MAX_GAMES=1`、`GROUP_SIZE=4`、
`MAX_STEPS=20`、`UPDATES=5`、`LEARNING_RATE=5e-5`。如果仍无法改变 action trace，
应停止 ALFWorld reward shaping 路径，改为先在 replay/trajectory 环境做离线 preference
或 imitation 预热，再回到 online RL。

后续小 smoke 结果：

```text
86599: imitation_w05_lr5e5
  step_credit_imitation_loss ~= 0.563
  20/20 action trace 仍与上一轮一致
  结论：auxiliary loss 接入，但 LR/更新强度不足以改变策略。

86667: imitation_w05_lr2e4
  20 条中 8 条 action trace 改变
  mean_rollout_reward: -0.6425 -> -0.6085
  update5 mean_goal_interaction_points = 2.25
  update5 mean_irrelevant_action_count = 2.25
  rollout_success_rate = 0.0
  问题：wrong/reverted 没降，模型学会更多目标操作但不会稳定保持正确放置。

86680: strict_credit_imitation_w05_lr2e4
  mean_rollout_reward = -0.6050
  update5 mean_placed_target_count = 0.25
  update5 mean_irrelevant_action_count = 1.75
  update5 mean_wrong_receptacle_move_count = 0.75
  rollout_success_rate = 0.0
  结论：严格 credit 略好，但仍不是 task success。
```

同时修复了一个 checkpoint 续训/评估问题：online HRPO checkpoint 原本没有顶层
`skills_path`，只在 `routing_init.skill_source_path` 中记录 skill pool。`86620` 的
post-update eval 因此触发旧 `pseudo_skills.jsonl` fallback 并被 blocker 阻断。现在：

- `_load_clstr_alfworld_model()` 会解析 checkpoint 顶层、`routing_init` 和嵌套
  `policy_checkpoint` 的 skill source path；
- online HRPO checkpoint 写入顶层 `skills_path/resolved_skills_path`；
- 新增 `test_load_clstr_alfworld_model_uses_checkpoint_routing_skill_source_path`。

当前诊断：Stage4-online 已有真实 RL/credit 信号，但仅靠 reward/credit 还不够。失败轨迹
的核心模式是“拿到目标 -> 放到目标位置 -> 又拿回或错放”，说明策略缺少可读的
progress/belief state。下一步不再继续加 reward shaping，而是把 action history 归纳成
`progress_memory` 输入：

- `build_alfworld_policy_state_text()` 新增可选 `progress_memory` 字段，默认不改变旧行为；
- online HRPO 用完整 action history 解析 `placed_target_count`、
  `inventory_target_count`、`reverted_target_count`、`wrong_receptacle_move_count`；
- 这是通用 closed-loop belief/progress 摘要，不绑定某个具体物体名；
- 新增 `test_build_alfworld_policy_state_text_accepts_optional_progress_memory` 和
  `test_alfworld_progress_memory_tracks_full_history_target_state`。

下一步只跑一个小 smoke：
`strict_credit_imitation_w05_lr2e4_progress_memory_u5_s20`。通过条件不是 task success，
而是 compared with `86680`：

- `mean_wrong_receptacle_move_count` 和 `mean_reverted_target_count` 不再上升；
- `mean_placed_target_count` 或最高 rollout reward 有方向性改善；
- action trace 中 “move target to target 后又 take from target” 减少。

若仍无改善，应停止 ALFWorld online 小修路线，转向离线 trajectory/preference 预热或换更强
executor/环境，而不是继续提交更多 ALFWorld smoke。

`progress_memory` 小 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_strict_credit_imitation_w05_lr2e4_progress_memory_u5_s20_smoke`
（job `86695`）已完成：

```text
strict vs memory: 18/20 action trace 完全一致
rollout_success_rate = 0.0
mean_rollout_reward = -0.6055
update5 mean_placed_target_count = 0.25
update5 mean_reverted_target_count = 0.25
update5 mean_wrong_receptacle_move_count = 0.75
```

结论：`progress_memory` 没有带来实质改善。当前 ALFWorld online path 已证明：

- online HRPO/step-credit/self-imitation 能产生非零学习信号；
- 提高 LR 后策略确实会移动；
- 但在弱 ALFWorld admissible-action bridge 上，它只学到更多目标物体操作，仍不能稳定完成
  多物体放置任务。

因此下一阶段不要继续提交 ALFWorld online smoke。更合理的路线是：

1. 在离线 trajectory/replay 数据上做 high-credit action imitation / preference warmup；
2. 或切到 executor 更稳定、reward 更直接的 benchmark 做 online RL；
3. ALFWorld online 只保留为附录/诊断，不作为证明 CLSTR Stage4 价值的主实验。

### 12.32 Qwen executor gate for Stage4/RL

2026-06-10 更新：Stage4/RL 的下一步不再继续 CLSTR-direct ALFWorld admissible-action
policy。该路径虽然能产生非零 online credit，但角色不对：CLSTR 应作为 selector/router，
而不是直接承担 text-action executor。

当前改为低成本 executor gate：

```text
Qwen-only:
  state/history/admissible_actions -> Qwen action scorer -> env.step

Qwen+CLSTR:
  state/history/admissible_actions -> Qwen action scorer
  state/history/admissible_actions -> CLSTR large-pool selector prior
  fused score = qwen_weight * qwen_score + clstr_weight * clstr_prior
  -> env.step
```

实现文件：

```text
clstr/alfworld_qwen_clstr_gate.py
scripts/run_alfworld_qwen_clstr_executor_gate.py
scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh
```

默认配置保持省钱：

```text
MAX_EPISODES=2
MAX_STEPS=20
SCORING_METHOD=generate
QWEN_MODEL_NAME_OR_PATH=models/Qwen3-8B
LOCAL_FILES_ONLY=1
CHECKPOINT_PATH=outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/latest.pt
```

gate 判定：

- 若 `qwen_plus_clstr` 在 success 或 reward 上优于 `qwen_only`，再考虑只更新 CLSTR 的
  online RL；
- 若两者都差或 CLSTR prior 不提升，不继续烧 ALFWorld online RL，改用 executor 更稳定的
  benchmark 或 trajectory-level preference/imitation；
- 该 gate 不是 official benchmark，也不是 full RL，只是决定 ALFWorld 是否值得继续接入
  online executor loop。

2026-06-10 smoke 结果：

```text
raw row-zscore, clstr_weight=0.5:
  output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_v1
  qwen_only reward/success = 0 / 0
  qwen_plus_clstr reward/success = 0 / 0
  qwen_clstr_disagreement_rate = 0.70
  qwen_hybrid_disagreement_rate = 0.00

raw row-zscore, clstr_weight=2.0:
  output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_clstrw2
  qwen_plus_clstr reward/success = 0 / 0
  qwen_clstr_disagreement_rate = 0.70
  qwen_hybrid_disagreement_rate = 0.00

proposal-bonus + shared loop guard, clstr_weight=0.5:
  output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_loopguard
  qwen_only reward/success = 0 / 0
  qwen_plus_clstr reward/success = 0 / 0
  qwen_clstr_disagreement_rate = 0.65
  qwen_hybrid_disagreement_rate = 0.125
```

结论：

- generate 模式不能直接把 Qwen one-hot 分数做 zscore 融合；它会压制 CLSTR prior；
- 已新增 `qwen_score_mode=proposal_bonus`，用于把 Qwen generation 视为 proposal，而不是
  不可撼动的 executor decision；
- loop guard 后 hybrid 确实产生更任务相关的动作，例如 `take/cool/move apple`，但官方 reward
  仍为 0；
- 因此不能直接 full online RL；需要以 trace-progress gate 复核这些动作是否真的命中目标物体
  和任务进展。复核结果如下。

2026-06-10 trace-progress gate 复核：

```text
output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_loopguard/trace_progress_gate.json
status = action_required
recommendation = do_not_start_full_rl_from_this_gate

qwen_plus_clstr - qwen_only:
  mean_progress_reward_delta = -0.075
  mean_progress_points_delta = 0.0
  mean_goal_interaction_points_delta = 0.0
  mean_placed_target_count_delta = 0.0
  mean_placed_target_event_count_delta = 0.0
  mean_irrelevant_action_count_delta = +2.5
  mean_reverted_target_count_delta = 0.0
  mean_wrong_receptacle_move_count_delta = 0.0
```

这次复核修正了前面对 hybrid trace 的乐观判断：按 `gamefile/traj_data.json`
解析真实 ALFWorld goal 后，两个样本分别要求操作 `soapbottle` 和 `potato`，但 hybrid trace
操作了 `spraybottle/soapbar/apple` 等非目标物体。也就是说，CLSTR prior 能影响 Qwen 的
action choice，但当前 executor gate 没有带来目标 grounding 或 progress uplift，反而增加了
irrelevant actions。

当前决策：不要从这个 gate 直接启动 full ALFWorld online RL。Stage4/RL 的下一步应先解决
executor/target grounding，或把主证明转向 ToolBench-G3 / TRAJECT 这类 executor 歧义更低的
selector-routing / trajectory preference 设置。ALFWorld online 暂时只作为诊断路径保留。

### 12.33 Dynamic Skill Registry v1

2026-06-11 更新：AppWorld/SkillRouter 对比暴露出一个更基础的问题：CLSTR 不能继续假设
一个完全静态的 skill pool。真实 agent runtime 的可用 skill/tool inventory 会随 benchmark、
环境、用户安装的工具和新 SkillX/SkillNet 资源变化；SkillRouter 的强项之一正是文本 skill
可以动态检索。因此 CLSTR 主线新增 Dynamic Skill Registry v1。

当前实现：

```text
SkillTable.append_skills(new_skill_rows)
  -> encode new skill text
  -> append to skill_table.E
  -> append retrieval/belief bias
  -> mark is_appended_after_checkpoint + zero coverage counts

CLSTRModel.append_skills(new_skill_rows)
  -> keep model.skills and skill_table.skills in sync
  -> invalidate cross-encoder task cache

DynamicSkillRegistry
  -> ordered rows + skill_id index
  -> duplicate skipping
  -> executor-domain available inventory

coverage-aware blending
  score = Stage0/text score
        + gamma_policy_i * policy_score
        + gamma_transition_i * transition_score
        + gamma_belief_i * belief_score
```

默认策略：

- old skill 且缺少 coverage metadata：保持 full CLSTR head weight，兼容历史 checkpoint；
- appended/unseen skill 且无 coverage：`gamma=0`，主要依赖 Stage0/text retrieval；
- appended skill 有 `transition_seen_count/act_seen_count` 后，head weight 按 coverage 逐步恢复。

这不是 AppWorld 特化。AppWorld 的合理接法是：

```text
global registry = v4.1b unified skills + AppWorld SkillX skills

routing diagnostics:
  在 global registry 中报告 AppWorld positives 的 rank/recall/MRR

executor eval:
  先全池打分，再只把 appworld_executor_compatible=true 的 skills 交给 Qwen
```

当前追加实现：

```text
data/clstr_appworld_dynamic_v4_1b_append
  skill_pool = v4.1b unified base 37,615 rows
             + AppWorld SkillX appended 188 rows
  retrieval = 2,496 rows
  trajectories = 1,068 rows

scripts/audit_appworld_dynamic_routing.py
  load Stage0 checkpoint with original base pool
  model.append_skills(AppWorld appended rows)
  report global positive ranks before executor mask
  report AppWorld-compatible positive ranks after executor mask
```

注意：真实 checkpoint audit 会加载 encoder 并对 37k+ skill pool 打分，不能在登录/存储节点
直接运行。下一步只允许先做小 sbatch smoke，例如限制 `--max_pairs 32`，确认动态 append
路径健康后再做全量 routing audit。

Smoke 入口：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_dynamic_routing_audit_smoke.sh
```

默认输出：

```text
outputs/appworld_dynamic_routing_audit/smoke_max32/report.json
```

判读：

- global rank 差、AppWorld-compatible rank 好：大池粗召回/score calibration 是主要瓶颈；
- global rank 和 AppWorld-compatible rank 都差：AppWorld SkillX skill text/query 格式或 encoder
  初始化仍有问题；
- 两者都好：再扩大到全量 AppWorld routing audit，然后接 executor eval。

第一次 smoke job `86993` 只可作为路径健康检查：它完成了 `appended_count=188` 和
`processed_pairs=32`，但 audit 使用 raw query，而 checkpoint config 是
`query_text_format=skillrouter`。当前已修复为 `query_text_format=auto` 默认跟随 checkpoint config。
因此必须重跑一次同样的 smoke，再使用 rank 指标做方向判断。

采样注意：文件头部 `MAX_PAIRS=32` 会偏向早期 Spotify routing qrels。smoke 脚本已改为默认
`SAMPLING_STRATEGY=stride`。下一次更有价值的低成本诊断是：

```bash
MAX_PAIRS=128 SAMPLING_STRATEGY=stride \
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_appworld_dynamic_routing_audit_smoke.sh
```

仍然不要直接 full audit，除非 stride smoke 显示动态 AppWorld routing 有继续放大的价值。

当前 stride smoke (`87028`) 已显示值得继续放大：

```text
MAX_PAIRS=128, SAMPLING_STRATEGY=stride, query_text_format=skillrouter
global Recall@50/100/350 = 0.515625 / 0.554688 / 0.710938
AppWorld-compatible Recall@20/50/100 = 0.664062 / 0.898438 / 1.0
```

这说明 AppWorld SkillX append 和 executor-compatible rerank 不是坏的；大池 global recall 仍是瓶颈。
下一步建议只扩大到 full routing audit，先不要直接烧 executor eval。

Full routing audit (`87084`) 已完成：

```text
positive pairs = 2,496 / 2,496
query_text_format = skillrouter
global Recall@50/100/350 = 0.48117 / 0.55008 / 0.704327
global MRR = 0.091053
AppWorld-compatible Recall@20/50/100 = 0.65625 / 0.913462 / 0.999599
AppWorld-compatible MRR = 0.278028
```

阶段判断：

- 这不是 Stage4，也不是训练；它只是证明动态 skill append 后的 routing 可用性；
- AppWorld-compatible mask 后几乎能在 top-100 覆盖所有 positives，因此可以进入小规模
  AppWorld executor eval；
- global 大池 Recall@350 只有约 70%，所以如果要主张“无 executor mask 的大池 top-K
  召回稳定”，还需要额外的 calibration/rerank 改进。

边界：

- 可以主张 CLSTR 支持动态追加 skill 的单步 plug-in retrieval；
- 不能主张 appended skill 在没有轨迹监督时立刻获得完整多步 transition/belief 能力；
- 多步动态 skill 的提升需要 few-shot trajectory、ACT 更新或 online feedback。

当前验证：

```text
tests/test_dynamic_skill_registry.py: passed
tests/test_v4_action_proj_sharing.py: passed
tests/test_stage_checkpoint_init.py: passed
```

只读 skill-pool 审计：

```text
data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl      = 37,615 rows
data/clstr_appworld_current_route_v1/skill_pool.jsonl   = 188 rows
unified_v4_1b appworld_like rows                         = 1
appworld_current appworld_like rows                      = 188
overlap skill_id count                                   = 0
```

结论：不能直接拿 v4.1b unified pool 跑 AppWorld dev57/current-route gate。当前 pool 几乎没有
AppWorld SkillX candidates，失败不能归因到 CLSTR 方法本身。AppWorld 下一步应先走
dynamic registry：

```text
base = data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl
append = data/clstr_appworld_current_route_v1/skill_pool.jsonl
eval routing = full dynamic registry, mask 前报告 AppWorld positive rank/recall/MRR
eval executor = full scoring 后仅传 appworld_executor_compatible=true skills
```

## 2026-06-11 AppWorld dynamic executor smoke 接入

Full dynamic routing audit 之后，AppWorld executor 不能直接用
`data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl` 去构造 checkpoint model。原因是
v4.1b Stage0 checkpoint 的 skill table 只覆盖 base pool：

```text
base skill pool      = 37,615 rows
dynamic skill pool   = 37,803 rows
appended AppWorld    = 188 rows
```

如果直接用 dynamic pool 建模再加载 checkpoint，skill table 形状和 checkpoint prefix 语义会错位。
当前 executor 路径已改为 checkpoint-prefix 动态加载：

```text
1. 用 base_skill_pool_path 还原 Stage0 checkpoint model
2. 从 dynamic skill_pool_path 中抽取 checkpoint prefix 之后的 AppWorld-compatible rows
3. 调用 model.append_skills(...) 追加 188 个 AppWorld SkillX skills
4. executor select 时仍在完整 dynamic pool 上打分
5. prompt handoff 时默认只传 appworld_executor_compatible=true skills
```

新增低成本 smoke 入口：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_appworld_dynamic_multistep_executor_smoke.sh
```

默认配置：

```text
METHOD=clstr_multistep
RANKING_MODE=skill_table
CANDIDATE_TOP_K=350
TOP_K=5
MAX_TASKS=3
MAX_STEPS=3
APPWORLD_EXECUTOR_COMPATIBLE_ONLY=1
MODEL_NAME_OR_PATH=models/Qwen3-8B
```

这一步只验证 dynamic CLSTR routing 能否接入真实 AppWorld/Qwen executor loop，不是 Stage4/RL，
也不是 Stage0/1/2 训练。若 smoke 卡在模型/环境初始化，应及时停止，不应扩大到 full eval。

Smoke 结果与修复：

- Job `87194` 暴露出 JSONL reader 问题：executor loader 使用
  `read_text().splitlines()`，会把 skill body 中的 Unicode line separator 切开，导致
  `JSONDecodeError`。已改为复用 dynamic audit 的逐文件行 reader；
- Job `87200` 暴露出 executor-compatible filter 问题：旧逻辑只过滤
  `appworld_executor_compatible=False`，会放行缺字段的 ToolRet/ToolBench base skills。已改为
  必须有正向证据：`appworld_executor_compatible=True`、`executor_domain=appworld` 或
  `skill_id` 以 `skillx/appworld/` 开头；
- 修复后的 job `87217` 完成：

```text
task_count = 3
success_count = 2
success_rate = 0.666667
generation_failures = 0
execution_failures = 3
average_steps = 1.666667
step_positive_skill_hit_rate = 0.0
runs_path = outputs/appworld_dynamic_multistep_executor_smoke/dev3_strict_compat_20260611_164719/runs.jsonl
```

重要判读：

- `87217` 的 selected skills 已是 `skillx/appworld/...`，dynamic SkillX handoff 路径成立；
- `step_positive_skill_hit_rate=0.0` 不能直接当作 routing 失败，因为
  `data/appworld_routing/dev_tasks.jsonl` 的前三个 Spotify “top most played song titles”任务，
  positive labels 却是 `spotify-find-playlist-by-duration-criteria` /
  `spotify-analyze-playlist-durations` / auth；而模型实际选中的是
  `spotify-find-songs-based-on-play-count`，语义更贴近目标；
- 因此下一步不应直接扩大 full executor eval。应先修 AppWorld routing/eval label 构造或改用
  verified step labels，再做更有代表性的 stratified executor smoke。

## 2026-06-11 AppWorld dynamic SkillX executor eval gate 更新

为避免继续用有噪声的 task-level `positive_skill_ids` 误判 AppWorld executor 结果，当前新增 eval-only step-label 构建：

```bash
python scripts/build_appworld_eval_step_label_tasks.py \
  --tasks_path data/appworld_routing/dev_tasks.jsonl \
  --skill_pool_path data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl \
  --output_path data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl
```

关键约束：

- 该文件只用于评估，manifest 标记 `eval_only=true`、`leakage_guard=dev_eval_only_no_training`；
- 标签空间限定为 AppWorld-compatible SkillX rows，不允许 checkpoint base skills 污染 executor step-hit；
- 每个 oracle API ref 使用 multi-positive：所有包含该 API ref 的 SkillX skills 都是该 API step 的 positive。这样比单个 heuristic best skill 更稳，但仍只是辅助诊断。

真实 dev57 构建结果：

```text
labeled_task_count = 57 / 57
label_skill_count = 188
avg_step_label_set_size = 9.188867
max_step_label_set_size = 30
```

3-task strict-compatible smoke `87267`：

```text
output = outputs/appworld_dynamic_multistep_executor_smoke/dev3_step_labels_20260611_184823
success_rate = 1.0
generation_failures = 0
execution_failures = 0
average_steps = 1.0
step_positive_skill_hit_rate = 0.0
```

判读与后续 gate：

- 这 3 个任务的 official success 是有效正信号；
- `step_positive_skill_hit_rate=0.0` 不是 routing 失败证据，因为 executor step 与 oracle API step 粒度不同：Qwen 可以一步写出完整多 API 程序，而 oracle step 0 只是 login；
- Stage4 full 不能只看 step-hit。下一步应先跑小规模 stratified executor eval，并同时跑 qwen-only / skillrouter-live 对照，确认 CLSTR skill context 是否带来真实增益；
- 在这些对照前，不应提交 AppWorld Stage4 full。

## 2026-06-11 AppWorld dev10 stratified 对照结果

已新增 required-app round-robin 子集：

```text
data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl
spotify = 2
phone+venmo = 2
venmo = 2
file_system = 2
file_system+simple_note = 2
```

对照结果：

| method | output | success | exec_fail | avg_steps | step_hit |
|---|---|---:|---:|---:|---:|
| qwen_only | `dev10_stratified_qwen_only_20260611_200032` | 2/10 | 5 | 2.3 | 0.0 |
| clstr_dynamic safe_metadata | `dev10_stratified_clstr_dynamic_20260611_201624` | 2/10 | 5 | 1.8 | 0.388889 |
| clstr_dynamic raw | `dev10_stratified_clstr_dynamic_raw_20260611_203529` | 2/10 | 9 | 2.0 | 0.4 |

决策：

- 不能进入 AppWorld Stage4 full：CLSTR 还没有在 official success 上超过 qwen-only；
- raw SkillX body 不作为主线，至少当前 prompt 下会增加 execution failures；
- CLSTR safe_metadata 对 Spotify 任务有局部帮助：同样成功但步骤更少；
- 下一步应先解决 executor handoff / API usage reliability，或换更强 executor，再考虑 Stage4/online
  是否值得跑。

## 2026-06-11 Stage4 前置 gate 调整：先修 executor reliability

当前 Stage4 不能直接 full 的原因不是 AppWorld dynamic skill pool 不可用，而是 dev10 official
success 没超过 qwen-only。复查失败样本后，主要缺陷集中在 executor prompt 与 AppWorld API
schema 对齐：

- relative-date 任务缺少 `specs.json` 里的 AppWorld task datetime；
- Venmo likes 任务需要 `show_transactions(min_like_count=1, direction=..., min_created_at=...)`
  并分页到空页；
- file_system rename/move 必须基于 `show_directory` path + `show_file` detail +
  `move_file(source_file_path, destination_file_path)`，不能发明 API；
- simple_note export 必须把 `search_notes` 当 list，逐条 `show_note`，导出 exact content。

已完成修复：

- `enrich_task_with_appworld_specs()` 从 task specs 注入 `task_datetime`；
- single-step 与 multistep executor eval 都接入该 enrichment；
- executor prompt 增加 AppWorld datetime 环境块与上述 schema-level task hints；
- 本地验证通过 `tests/test_appworld_executor.py`、`tests/test_appworld_multistep.py`、
  `tests/test_appworld_multistep_cli.py`、`py_compile`、`git diff --check`。

新的执行门槛：

- 先复跑 `dev10_stratified_clstr_dynamic safe_metadata`；
- 若 official success 或 execution failure profile 没有改善，不进入 Stage4，继续修 executor
  或评估更强 executor；
- 若 dev10 出现稳定正增益，再跑 Stage4 smoke；Stage4 full 仍需 smoke 通过后再决定。

## 2026-06-12 Stage4 ACT 推理对齐修复

新的 Stage4 gate 不能继续使用旧 `policy_blend` 作为 ACT 评估，因为：

- v4.1b conservative Stage4 checkpoint 是 delta checkpoint，排除了 frozen routing foundation；
- Stage4 训练目标是 transition-conditioned next-skill ranking，而旧 AppWorld controller 的
  `policy_blend` 主要使用 `skill_head`，没有直接使用 Stage4 `trans_head` scorer。

主线修复：

- dynamic AppWorld executor loader 对 Stage4 delta 执行 composite restore：
  `Stage0 routing checkpoint -> Stage2 base/head checkpoint -> Stage4 delta checkpoint -> append AppWorld SkillX`；
- `CLSTRMultiStepController` 新增 `transition_head` / `transition_blend`：
  - step 0 没有 previous skill，回退 Stage0 routing；
  - step >= 1 使用上一轮 selected skill 和当前 state 调用 `_transition_candidate_logits_for_mode`；
  - `transition_blend` 用 Stage0 routing 与 Stage4 transition logits 做标准化混合；
- CLI `ranking_mode=auto` 对 Stage4 checkpoint 默认使用 `transition_blend`，HRPO/旧 ACT checkpoint
  仍使用 `policy_blend`。

当前 gate 策略：

1. 先跑 dev5 小门控：`RANKING_MODE=transition_blend`、`POLICY_BLEND_ALPHA=0.25`、
   `TRANSITION_SCORING_MODE=v4_1b_action_observation`；
2. 若 dev5 至少不低于 Stage0-style CLSTR 且 diagnostics 显示
   `transition_scores_available=true`，再跑 dev10；
3. 只有 dev10 相比 qwen-only/schema4 或 Stage0-style CLSTR 出现正收益，才讨论 Stage4 full。

Gate `87395` 的第一轮结果不能作为有效 Stage4 结论：第二步任务触发
`1x256 vs 1024x1024` 维度错误。已确认是推理端错误地把 `action_proj` 后的 256 维动作向量
传给 Stage4 helper；v4.1b Stage4 训练需要的是 1024 维 raw skill/action embedding。当前已改为
使用 `skill_table.E[previous_skill]`，需重跑同样 dev5 gate。

## 2026-06-12 Stage4 AppWorld 动态 SkillX gate 更新

当前 Stage4 主线不是直接 full，而是低成本 gate：

1. dynamic AppWorld SkillX append：v4.1b unified skill pool 作为 checkpoint prefix，追加 188 条
   AppWorld SkillX；
2. Stage4 checkpoint 使用 composite restore：
   `Stage0 routing -> Stage2 base/head -> Stage4 delta -> dynamic append`；
3. 推理 ranking 使用 `transition_blend`，step 0 回退 Stage0 routing，step >= 1 用 Stage4
   transition scorer rerank；
4. `transition_blend` gate 只在 dev10 上先证明 official success 或 execution-failure profile
   优于 qwen-only / Stage0-style CLSTR 后，才允许考虑 Stage4 full。

最新结果：

```text
dev5  stage4 transition_blend fixed-embedding: 4/5
dev10 stage4 transition_blend fixed-embedding: 6/10, exec_fail=5
qwen-only schema4 dev10:                         6/10, exec_fail=3
Stage0-style CLSTR schema4 dev10:                5/10
```

判读：

- Stage4 transition scorer 已经真实参与推理，并能从 Stage0-style CLSTR 的 5/10 提升到 6/10；
- 但它还没有超过 qwen-only，且 execution failures 更高，因此不能进入 full；
- 当前优先修复 executor handoff，而不是继续盲跑更大 Stage4。

已新增 executor-schema 防污染规则：

- prompt builder 从 `[AppWorld API Docs]` 解析当前真实可调用 API；
- 每个 retrieved skill 标注真实 `executable_appworld_apis_from_skill`；
- 对 SkillX title 派生出的不存在 API 标注 `blocked_nonexistent_skill_title_api`；
- 对 safe metadata 中没有真实写操作 API 支撑且标题像伪 API 的技能隐藏 SkillX id/name。

下一步 gate：用同一 dev10 stratified subset 重跑 Stage4 `transition_blend`。只有当成功率超过
qwen-only `6/10` 或 execution failures 明显下降且成功率不降时，才继续扩大评估。

### Gate 修正：prompt sanitizer 证伪，改为 preflight repair

两轮 prompt sanitizer gate 结果：

```text
87399 full sanitizer:   5/10, exec_fail=6
87401 narrow sanitizer: 4/10, exec_fail=5
```

结论：

- prompt-level schema annotation 会干扰 Qwen，不适合作为 Stage4 主线；
- `schema_guard_skill_metadata` 保留为显式诊断选项，默认关闭；
- 主线改为 executor 侧 preflight：生成代码后、执行前用 AST 和当前 API docs 检查语法与
  `apis.<app>.<api>` 是否存在；
- preflight 失败时允许一次 repair generation，修复后再执行。

下一轮 dev10 gate 应使用恢复后的 safe_metadata prompt + preflight repair。若成功率不能恢复到
`6/10` 或降低 execution failures，则继续检查 executor/模型本身，不进入 full。

### 当前有用 Stage4 信号

后续结果：

```text
87405 Stage4 + preflight repair1: 6/10, exec_fail=4
87406 Stage4 + preflight repair2: 7/10, exec_fail=0
87407 qwen-only + preflight repair2: 6/10, exec_fail=2
```

判读：

- `preflight repair2` 是当前 executor 层默认：生成代码后检查语法与 API schema，最多修复两次；
- 在相同 preflight 条件下，Stage4+CLSTR 比 qwen-only 多成功 1 个 dev10 任务，且平均步骤更少；
- 这说明当前 Stage4 transition route 已经能产生可观测任务成功增益，但证据规模仍是 dev10 gate；
- 不应直接声称 AppWorld benchmark 已完成。下一步需要显式批准后跑 dev57 或更大对照矩阵。

### dev57 扩展验证后的修正

dev57 已跑完，当前 Stage4 handoff 未通过更大 gate：

```text
87408 qwen-only + preflight2 dev57:         16/57, exec_fail=39, preflight_fail=0
87409 Stage4 transition_blend + preflight2: 11/57, exec_fail=69, preflight_fail=24
```

行级结果是：共同成功 10 个任务，Stage4 只新增 1 个成功，却让 6 个 qwen-only 成功任务退化。
这些退化主要来自 SkillX metadata 的 executor 诱导，而不是 Stage4 loader 或 transition scorer 崩溃。

本轮同时修复了一个必要实现问题：preflight 不再用截断后的 prompt API docs 判断 API 是否存在，而是从
完整 AppWorld schema 加载有效 API refs。相关 65 个测试、`py_compile`、`git diff --check` 已通过。
但修复后的 6 条退化任务小回归仍为 `0/6`，只是 `preflight_failures` 降到 0。

因此下一阶段不应继续提交 AppWorld dev57/full 作业。若继续 AppWorld，应先重设计
SkillX-to-executor handoff：只给 schema-grounded operation plan、allowed API constraints、必要字段约束，
避免把 SkillX title/metadata 作为自由文本参考直接暴露给 Qwen。

### gated_schema handoff 小门槛

已实现 `skill_context_mode=gated_schema`，用于验证“保留有用 SkillX 语义，同时抑制污染标题”的
折中路线：

- 强匹配 skill 保留 `safe_metadata`；
- 弱匹配、action mismatch 或已知污染风险 skill 降级为 `schema_plan`；
- 无 schema-grounded API 证据的 skill 直接 drop；
- 每步输出记录 `skill_handoff` 诊断，后续 regression/dev gate 必须检查这些字段。

继续执行顺序：

1. 跑 `regression6 gated_schema`，要求不低于 `schema_plan_not_allowlist` 的 `4/6`；
2. 若通过，再跑 `dev10 gated_schema`，目标不低于当前 Stage4+preflight2 的 `7/10`；
3. 若 dev10 不能超过 qwen-only+preflight2 的 `6/10`，停止 AppWorld 扩展验证；
4. 只有 regression6 和 dev10 同时通过，才讨论 dev57。

当前结果：

```text
gated_schema v1 regression6: 0/6
gated_schema v2 regression6: 4/6, preflight_fail=0
gated_schema v2 dev10:       5/10, preflight_fail=1, exec_fail=8
```

因此 `gated_schema_v2` 不能作为下一阶段主线，也不能进入 dev57/full。它的价值是定位了污染边界：
schema-only 对 unique-count / queue regression 有效，但复杂 per-skill mixed handoff 会扰动原本
`safe_metadata + preflight2` 能完成的 file-system/simple-note 任务。下一轮若继续 AppWorld，应优先做
“默认 safe_metadata，只对高污染任务族切 schema-only”的更简单策略。

### AppWorld postmortem 后的决策边界

已新增无 GPU 审计报告：`docs/appworld_stage4_postmortem_2026-06-12.md`。

结论：

- Stage4 transition 在 dev57 中确实运行：137 个 step 中 75 个后续 step 有
  `transition_scores_available=true`；
- dev57 退化主要发生在 step 0 的 selected SkillX context / handoff 污染，而不是 Stage4 transition
  scorer 没加载；
- 6 个 qwen-only regression 集中在 Spotify unique-count、Spotify add-to-queue、Simple Note export；
- Stage4 有 1 个唯一成功和若干共同成功任务的步数收益，但不足以抵消 dev57 的整体退化。

因此，AppWorld 当前只能作为动态 SkillX + executor case study。除非最后一轮非常简单的 handoff
策略能先过 regression6 和 dev10，否则不再把 AppWorld task success 作为 Stage4 主证据。

### final api_evidence handoff gate

为避免继续把 SkillX metadata 当 executor prompt 的自然语言计划，新增
`skill_context_mode=api_evidence`：

- CLSTR 仍选择 top-k SkillX，选择结果写入 `steps[*].skill_handoff`；
- prompt 不暴露 selected SkillX 的 `skill_id`、`name`、`description`、`body`、`skill_md`；
- prompt 只显示从 selected SkillX 文本中解析、并被完整 AppWorld API schema 校验过的
  `valid_appworld_apis_from_selected_skills`、`read_support_apis`、
  `state_changing_action_apis`；
- invalid 或 action-filtered refs 只作为 `invalid_or_filtered_refs` / diagnostics，不能被当作
  usable API evidence；
- 这是 benchmark adapter 层改动，不改变 Stage4 checkpoint、ACT 训练或 transition scorer。

本地验证已通过：

```text
tests/test_appworld_executor.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py
71 passed
```

执行顺序仍是小门槛：

1. 跑 `regression6 stage4 transition_blend + preflight2 + api_evidence`；
2. 只有 `>=4/6` 且 `preflight_failures=0` 时，才跑 dev10；
3. 只有 dev10 达到 `>=7/10` 且 failure profile 不差于当前最好 Stage4+preflight2，才重新讨论
   dev57/full。否则 AppWorld task success 不再作为 Stage4 主证据。

结果：

```text
87752 regression6 api_evidence: 1/6, preflight_fail=0, exec_fail=6
```

判读：

- adapter 路径没有崩溃，`skill_handoff` 诊断正常写入；
- Simple Note export 成功，说明隐藏 SkillX 文本确实能缓解某些污染；
- Spotify regression 失败说明纯 API evidence 过弱，缺少完整 library traversal / queue write 操作语义；
- 因此不跑 dev10/dev57/full。AppWorld task-success 路线继续保持阻断，除非以后有新的 handoff
  设计能同时保留通用 procedural evidence 并避免 title/description 污染。
